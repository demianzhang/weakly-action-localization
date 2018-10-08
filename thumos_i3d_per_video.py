import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    #return torch.from_numpy(pic.transpose([3, 0, 1, 2]))
    return torch.from_numpy(pic.reshape((-1,1024)))



def make_dataset(split_file):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        gt = []
        label = np.zeros(2, np.int64)
        id = data[vid]['actions'][0][0]
        label[0] = id                              # 从1开始
        for ann in data[vid]['actions']:
            ann = [ann[0],int(ann[1]*30),int(ann[2]*30)] # 转换为帧 30fps
            gt.append(ann)
            if label[0]!=int(ann[0]):
                label[1]=int(ann[0])
        dataset.append((vid, label, data[vid]['frame'], gt))
        i += 1

    return dataset


class Thumos(data_utl.Dataset):

    def __init__(self, split_file, root, batch_size, mode, train):

        self.data = make_dataset(split_file)
        self.split_file = split_file
        self.batch_size = batch_size
        self.root = root
        self.in_mem = {}
        self.mode = mode
        self.train = train


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        entry = self.data[index]  # name, label, frame, gt
       
        feature=[]
        if entry[0] in self.in_mem:
            feat = self.in_mem[entry[0]]
        else:
            ## concat rgb and flow featuremap
            # for i in range(1,401):
            #     feat = np.load(os.path.join(self.root, entry[0], 'flow/'+ str(i) + '.npy'))
            #     feat_rgb = np.load(os.path.join(self.root, entry[0], 'rgb/' + str(i) + '.npy'))
            #     feat = feat.reshape((-1, 1024))
            #     feat_rgb = feat_rgb.reshape((-1, 1024))
            #     feat = feat.astype(np.float32)
            #     feat_rgb = feat_rgb.astype(np.float32)
            #     feat = np.concatenate((feat,feat_rgb),axis=1)
            #     feature.append(feat)

            ## 固定400段切分

            # during = entry[2]/400
            #
            # for i in range(1,entry[2]):
            #     if i*during + 49 > entry[2]:
            #         continue

            ## 等间隔切分

            during = 24
            if entry[2] > 15000:  # over 15000 frames
                during = 48
            if entry[2] > 30000:
                during = 96
            i=0
            for item in range(0, entry[2] + 1, during):
                if item + 49 > entry[2]:
                    continue
                i+=1
                feat = np.load(os.path.join(self.root, entry[0], self.mode +'/'+ str(i) + '.npy'))
                feat = feat.astype(np.float32)
                feature.append(feat)
        feature = np.array(feature)

        if self.train:
            label = np.zeros(20)              # for train multi-label
            for i in range(len(entry[1])):
                if entry[1][i]==0:
                    continue
                label[entry[1][i]-1]=1
        else:
            label = entry[1]-1                # for test

        return feature, label, [entry[0], entry[2], entry[3]]


    def __len__(self):
        return len(self.data)


def mt_collate_fn(batch):
    # "Pads data and puts it into a tensor of same dimensions"
    # max_len = 0
    # for b in batch:
    #     if b[0].shape[0] > max_len:
    #         max_len = b[0].shape[0]

    new_batch = []
    for b in batch:

        new_batch.append([video_to_tensor(b[0]),torch.from_numpy(b[1]) , b[2]])
        

    return default_collate(new_batch)