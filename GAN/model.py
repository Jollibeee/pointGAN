# https://github.com/leondelee/PointGCN/blob/master/model/edge_conv.py
# https://github.com/fxia22/pointGAN/blob/74b6c432c5eaa1e0a833e755f450df2ee2c5488e/pointnet.py#L159
# https://github.com/ToughStoneX/DGCNN/blob/master/Model/dynami_graph_cnn.py

# https://github.com/AnTao97/UnsupervisedPointCloudReconstruction/tree/b5896942c8a322d1934bcff480f5ac4a90222461

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv_bn_block(input, output, kernel_size):
    return nn.Sequential(
        nn.Conv1d(input, output, kernel_size),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)    )

def fc_bn_block(input, output):
    return nn.Sequential(
        nn.Linear(input, output),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)    )

class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.z_size = args.feat_dim
        self.np = args.num_points

        self.model = nn.Sequential(
            fc_bn_block(self.z_size, 64),
            fc_bn_block(64, 128),
            fc_bn_block(128, 512),
            fc_bn_block(512, 1024),
            fc_bn_block(1024, self.np * 3),
        )

    def forward(self, input):
        output = self.model(input.squeeze())
        output = output.view(-1, 3, self.np)
        return output


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.z_size = args.feat_dim

        self.conv = nn.Sequential(
            conv_bn_block(64, 128, 1),
            conv_bn_block(128, 256, 1),
            conv_bn_block(256, 256, 1),
            conv_bn_block(256, 512, 1))
        self.pool = nn.AvgPool1d(512)
        self.fc = nn.Sequential(
            fc_bn_block(512, 128),
            fc_bn_block(128, 64),
            fc_bn_block(64, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        logit = self.fc(x)
        prob = nn.Sigmoid(logit)
        return logit, prob

#%%
# https://github.com/fxia22/pointGAN/blob/74b6c432c5eaa1e0a833e755f450df2ee2c5488e/pointnet.py#L159

class STN3d(nn.Module):
    def __init__(self, opt):
        super(STN3d, self).__init__()
        self.num_points = opt.num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, opt, global_feat = True, trans = True):
        super(PointNetfeat, self).__init__()

        self.global_feat = global_feat
        self.num_points = opt.num_points

        self.stn = STN3d(opt)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans

        #self.mp1 = torch.nn.MaxPool1d(num_points)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

class PointNetReg(nn.Module):
    def __init__(self, opt, k = 1):
        super(PointNetReg, self).__init__()
        self.num_points = opt.num_points
        self.feat = PointNetfeat(opt, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, trans


class PointGen(nn.Module):
    def __init__(self, args):
        super(PointGen, self).__init__()
        self.num_points = args.num_points
        self.feat_dim = args.feat_dim
        self.fc1 = nn.Linear(self.feat_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3)

        self.th = nn.Tanh()
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points)
        return x
#%%
# class Generator(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#
#         self.z_size = args.feat_dim
#         self.use_bias = True
#
#         self.model = nn.Sequential(
#             nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
#             nn.ReLU(inplace=True),
#
#             nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
#             nn.ReLU(inplace=True),
#
#             nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
#             nn.ReLU(inplace=True),
#
#             nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
#             nn.ReLU(inplace=True),
#
#             nn.Linear(in_features=1024, out_features=2048 * 3, bias=self.use_bias),
#         )
#
#     def forward(self, input):
#         output = self.model(input.squeeze())
#         output = output.view(-1, 3, 2048)
#         return output
