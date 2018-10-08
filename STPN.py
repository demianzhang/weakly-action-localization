import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class TemporalProposal(nn.Module):
    def __init__(self, classes=20):
        super(TemporalProposal, self).__init__()
        self.classes = classes


        self.fc1 = nn.Linear(1024, 256)
        nn.init.normal_(self.fc1.weight, std=0.001)
        nn.init.constant_(self.fc1.bias, 0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.constant_(self.fc2.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(1024, 20)
        nn.init.normal_(self.fc3.weight, std=0.001)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, inp):
        x = inp
        inp = self.fc1(inp)
        inp = self.relu(inp)
        inp = self.fc2(inp)
        inp = self.sigmoid(inp)
        x = inp*x
        x = torch.sum(x, dim=1)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x, inp

# class TemporalProposal(nn.Module):
#     def __init__(self, classes=20):
#         super(TemporalProposal, self).__init__()
#
#         self.classes = classes
#         self.dropout = nn.Dropout(0.7)
#
#         self.fc11 = nn.Linear(400, 400 // 4)
#         nn.init.normal(self.fc11.weight, std=0.001)
#         nn.init.constant(self.fc11.bias, 0)
#         self.fc22 = nn.Linear(400 // 4, 400)
#         nn.init.normal(self.fc22.weight, std=0.001)
#         nn.init.constant(self.fc22.bias, 0)
#         self.conv1 = nn.Conv1d(2, 1, 3, stride=1, padding=1, bias=False)
#         nn.init.kaiming_normal(self.conv1.weight.data)
#
#         self.fc1 = nn.Linear(1024, 256)
#         nn.init.normal(self.fc1.weight, std=0.001)
#         nn.init.constant(self.fc1.bias,0)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(256, 1)
#         nn.init.normal(self.fc2.weight, std=0.001)
#         nn.init.constant(self.fc2.bias,0)
#         self.sigmoid = nn.Sigmoid()
#         self.fc3 = nn.Linear(1024, 20)
#         nn.init.normal(self.fc3.weight, std=0.001)
#         nn.init.constant(self.fc3.bias,0)
#
#     def forward(self, inp):
#
#         # channel attention
#         module_input = inp
#         avg = torch.mean(inp, 2)
#         mx,_ = torch.max(inp, 2)
#
#         avg = self.fc11(avg)
#         avg = self.relu(avg)
#         mx = self.fc11(mx)
#         mx = self.relu(mx)
#         avg = self.fc22(avg)
#         mx = self.fc22(mx)
#
#         avg = avg.unsqueeze(-1)
#         mx = mx.unsqueeze(-1)
#         inp = torch.cat([avg, mx], dim=-1)
#         inp = torch.transpose(inp, 1,2)
#         inp = self.conv1(inp)
#
#
#         inp = self.sigmoid(inp)
#         inp = torch.transpose(inp, 1, 2)
#         inp = module_input * inp
#
#         # temporal attention
#         x = inp
#         inp = self.fc1(inp)
#         inp = self.relu(inp)
#         inp = self.fc2(inp)
#         inp = self.sigmoid(inp)
#         x = inp*x
#         x = torch.sum(x, dim=1)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#
#         return x, inp

def get_model(gpu, classes=20):
    model = TemporalProposal(classes)
    return model.cuda()