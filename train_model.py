from __future__ import division
import time
import os
import argparse
from util import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-model_file', type=str)
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='1')
parser.add_argument('-resume', type=str2bool, default='False', help='restore from previous')
parser.add_argument('-dataset', type=str, default='thumos')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import numpy as np
import json

import STPN

if args.dataset == 'thumos':
    from thumos_i3d_per_video import Thumos as Dataset
    from thumos_i3d_per_video import mt_collate_fn as collate_fn

    train_split = 'data/thumos14_val.json'
    test_split = 'data/thumos14_test.json'
    rgb_flow_train_root = '/media/zjg/workspace/action/npy/'
    classes = 20
    batch_size = 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data(train_split, root, mode, train):
    # Load Data

    if len(train_split) > 0:
        dataset = Dataset(train_split, root, batch_size, mode, train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:
        dataset = None
        dataloader = None

    dataloaders = {'train': dataloader}
    datasets = {'train': dataset}
    return dataloaders, datasets


# train the model
def run(models, criterion, num_epochs=50):

    best_loss = 10000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for model, gpu, dataloader, optimizer, sched, model_file in models:
            adjust_learning_rate(optimizer, epoch)
            print('lr:{}'.format(optimizer.param_groups[0]['lr']))
            loss = train_step(model, gpu, optimizer, dataloader['train'],criterion)

            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'models/' + model_file + "/flow_model.pkl")


def l1_penalty(var):
    return torch.abs(var).sum()


def run_network(model, data, gpu, criterion):
    # get the inputs
    inputs, labels, other = data

    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    labels = Variable(labels.cuda(gpu)).float()

    # forward
    outputs, attention = model(inputs)
    loss = criterion(outputs, labels)
    loss += l1_penalty(attention) / 5000

    return outputs, loss


def train_step(model, gpu, optimizer, dataloader, criterion):

    model.train(True)
    tot_loss = 0.0
    num_iter = 0.

    # Iterate over data.
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1
        outputs, loss = run_network(model, data, gpu,criterion)
        tot_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    epoch_loss = tot_loss / num_iter
    print('train-{} Loss: {:.4f}'.format(dataloader.root, epoch_loss))
    return epoch_loss

def adjust_learning_rate(optimizer, epoch):
    lr = 0.0001 * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def returnCAM(feature, fc, class_idx):
    # generate the class activation maps
    b, f = feature.shape
    for idx in class_idx:
        cam = feature.reshape((-1, 1024)).dot(fc[idx].reshape(1024, -1))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

    return cam


if __name__ == '__main__':

    if args.train:

        if args.mode == 'flow':
            dataloaders, datasets = load_data(train_split, rgb_flow_train_root, args.mode, args.train)
        elif args.mode == 'rgb':
            dataloaders, datasets = load_data(train_split, rgb_flow_train_root, args.mode, args.train)

        model = STPN.get_model(0, classes)
        criterion = torch.nn.BCELoss()

        if args.resume:
            if os.path.isfile('models/thumos/flow_model.pkl'):
                print("loading checkpoint model")
                checkpoint = torch.load('models/thumos/flow_model.pkl')
                model.load_state_dict(checkpoint)
                print("loaded checkpoint")
            else:
                print("no checkpoint found")

        lr = 0.1 * batch_size / len(datasets['train'])

        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

        run([(model, 0, dataloaders, optimizer, lr_sched, args.model_file)], criterion, num_epochs=120)

    else:
        print("do CAM")
        model = STPN.get_model(0, classes)
        if os.path.isfile('models/thumos/flow_model.pkl'):
            print("loading checkpoint model")
            checkpoint = torch.load('models/thumos/flow_model.pkl')
            model.load_state_dict(checkpoint)
            print("loaded checkpoint")

        if args.mode == 'flow':
            dataloaders, datasets = load_data(test_split, rgb_flow_train_root, args.mode, args.train)
        elif args.mode == 'rgb':
            dataloaders, datasets = load_data(test_split, rgb_flow_train_root, args.mode, args.train)

        # right = 0
        # all = 0
        # fp = 0
        f_result = open("/home/zjg/Desktop/result.txt", "w")
        f_test_cam = open("/media/zjg/workspace/action/test/cam_test.txt", "r") # 测试集rgb的cam向量
        f_test = open("/media/zjg/workspace/action/test/rgb_test.txt", "r")  # 测试集rgb的结果(起始帧，终止帧，得分)

        cam_400 = f_test_cam.readlines()
        val_rgb_cam = []
        for item in cam_400:
            item = item.strip().split(' ')
            val_rgb_cam.append(item)


        rgb_proposal = f_test.readlines()
        per_video = []
        tot_rgb = []
        rgb_num = 0
        while rgb_num < len(rgb_proposal):
            tmp = rgb_proposal[rgb_num].strip().split(' ')
            if len(tmp) == 1: # 个数
                for _ in range(int(tmp[0])):
                    rgb_num += 1
                    tmp = rgb_proposal[rgb_num].strip().split(' ')
                    tmp = list(map(float, tmp))
                    per_video.append(tmp)
                tot_rgb.append(per_video)
                per_video = []
                rgb_num += 1

        num_step = 0
        for data in dataloaders['train']:
            inputs, labels, other = data
            inputs = Variable(inputs.cuda(0))
            outputs, attention = model(inputs)

            params = list(model.parameters())
            fc = np.squeeze(params[-2].data.cpu().numpy())

            inputs = inputs.squeeze()
            inputs = inputs.data.cpu().numpy()
            labels = labels.numpy()
            predict = outputs.data.cpu().numpy()
            attention = attention.data.cpu().numpy().squeeze()
            sort_output = np.argsort(-predict) # score由大到小排序
            sort_output = sort_output.squeeze()
            sort_i=-1

            while True:
                sort_i+=1

                if predict[0][sort_output[sort_i]]<0.1:
                    break

                cam = returnCAM(inputs, fc, [sort_output[sort_i]])
                cam = cam.squeeze()

                ## store the value of rgb cam
                # for item in cam:
                #     f.write("{:.3f} ".format(item))
                # f.write("\n")

                ## other[2]是action的gt
                # for item in other[2]:
                #     print("start:{},end:{}".format(round(item[1].numpy()[0]/30,1),round(item[2].numpy()[0]/30,1)))

                f_start_frame = open("/media/zjg/workspace/action/data/start_frame/" + other[0][0] + ".txt")
                lines = f_start_frame.readlines()
                line = lines[0]
                segment_start = line.strip().split(' ')


                cam = sigmoid(cam)
                cam = cam * attention
                pos = np.where(cam > 0.05)

                # for i in range(len(segment_start)):
                #     flag=0
                #     for j in other[2]:
                #         if j[1].numpy()[0]<int(segment_start[i])+24 and j[2].numpy()[0]>int(segment_start[i])+24:
                #             print("centor:{},cam:{}".format(round((int(segment_start[i])+24)/30,1),round(cam[i],6)))
                #             flag=1
                #             break
                #     if flag==0:
                #         print("   centor:{},cam:{}".format(round((int(segment_start[i])+24)/30,1), round(cam[i], 6)))
                # print("video:{}".format(other[0]))


                # -------------------------------------------------------------------------
                # evaluation result.txt
                #
                sort_start = []

                for i in pos[0]:
                    sort_start.append([int(segment_start[i]), cam[i]])
                sort_start.sort(key=lambda x: x[0])

                res = []
                ## proposal generator
                for i in range(len(sort_start)):
                    if i == 0:
                        start = sort_start[i][0]
                        end = sort_start[i][0] + 48
                        segment = 1
                        score = sort_start[i][1]
                        continue
                    if sort_start[i][0] < end:
                        end = sort_start[i][0] + 48
                        score = max(score, sort_start[i][1])
                        segment += 1
                    else:
                        res.append([start, end, round(score, 6)])
                        segment = 1
                        start = sort_start[i][0]
                        end = sort_start[i][0] + 48
                        score = sort_start[i][1]


                ## store the value of rgb-test proposal

                # f_test.write("{}\n".format(len(res)))
                # for item in res:
                #     if item:
                #         f_test.write("{} {} {:.3f}".format(item[0],item[1],item[2]))
                #     f_test.write("\n")

                ## combine two stream result
                # for item in tot_rgb[num_step]:
                #     res.append(item)

                # result = []
                # for item in res:
                #     score = 0
                #     number=0
                #     for i in range(len(segment_start)):
                #         if int(segment_start[i])+24 > item[1]:
                #             result.append([item[0], item[1], score])
                #             break
                #         if int(segment_start[i])+24 >= item[0]:
                #             number+=1
                #             score = max(max(float(val_rgb_cam[num_step][i]),cam[i]),score)


                num_step += 1
                # res = np.array(result)    # 双流融合结果
                res = np.array(res)
                res = temporal_nms(res, 0.5)

                for part in res:
                    if part[1] - part[0] < 30: # 忽略小于30帧的action
                        continue
                    ## (视频名称， 起始帧， 终止帧， 类别编号， 预测得分)
                    f_result.write("{} {} {} {} {}".format(other[0][0], round(part[0] / 30, 1), round(part[1] / 30, 1),
                                                     sort_output[sort_i] + 1, part[2]))
                    f_result.write('\n')
