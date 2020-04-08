# https://github.com/ChrisWu1997/PQ-NET

# https://github.com/yifita/3PU_pytorch

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def get_graph_feature(x, k=40, idx=None):
    # print(x.size())
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    #
    # idx = idx + idx_base
    #
    # idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)
    # print("graph feat", feature.size())
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)  # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)
    # print("graph feat", feature.size())
    return feature      # (batch_size, 2*num_dims, num_points, k)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input



class DGCNN_Encoder(nn.Module):
    def __init__(self, k = 40, feat_dim = 128, num_points = 2048):
        super(DGCNN_Encoder, self).__init__()
        self.k = k
        self.feat_dim = feat_dim
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.feat_dim)

        self.conv1 = nn.Sequential(nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.feat_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        # x = x.transpose(2, 1)   # batch, points, dim
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)

        x0 = self.conv5(x)  # (batch_size, 512, num_points) -> (batch_size, feat_dims, num_points)
        x = x0.max(dim=-1, keepdim=False)[0]  # (batch_size, feat_dims, num_points) -> (batch_size, feat_dims)
        feat = x.unsqueeze(1)  # (batch_size, feat_dims) -> (batch_size, 1, feat_dims)
        return feat  # (batch_size, 1, feat_dims)


class Decoder(nn.Module):
    def __init__(self, feat_dim = 128, num_points = 2048, k = 40):
        super(Decoder, self).__init__()
        self.feat_dim = feat_dim
        self.num_points = num_points

        self.fc1 = nn.Linear(self.feat_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batch = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.th(x)
        x = x.view(batch, 3, self.num_points)
        return x

class DGCNN_AE(nn.Module):
    def __init__(self, num_points = 2048, feat_dim = 128, k = 40):
        super(DGCNN_AE, self).__init__()
        self.num_points = num_points
        self.feat_dim = feat_dim
        self.k = k
        self.encoder = DGCNN_Encoder(k=self.k, feat_dim=self.feat_dim, num_points=self.num_points)
        self.decoder = Decoder(feat_dim=self.feat_dim, num_points=self.num_points)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return z, y

    def z(self, x):
        x = self.encoder(x)
        return x

class Generator(nn.Module):
    def __init__(self, num_points = 2048, feat_dim = 128):
        super(Generator, self).__init__()
        self.num_points = num_points
        self.feat = DGCNN_Encoder(feat_dim)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, feat_dim)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_points = 2048, feat_dim = 128):
        super(Discriminator, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3)

        def forward(self, x):
            batchsize = x.size()[0]
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.th(self.fc4(x))
            x = x.view(batchsize, self.num_points, 3)
            return x


class PointDis(nn.Module):
    def __init__(self, feat_dim=128, num_points = 2048):
        super(PointDis, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]   # batch, 3-dim, num_points
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points)
        return x

class PointGen(nn.Module):
    def __init__(self, feat_dim=128, num_points = 2048, k=40):
        super(PointGen, self).__init__()
        self.num_points = num_points
        self.feat_dim =feat_dim
        self.k = k
        self.feat = DGCNN_Encoder(k = self.k, feat_dim = self.feat_dim)
        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_points)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(512)
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, feat_dim)
        # self.bn1 = torch.nn.BatchNorm1d(512)
        # self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat(x)    # [batch, feat_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class ResBlock(nn.Module):
#     def __init__(self):
#         super(ResBlock, self).__init__()
#         self.res_block = nn.Sequential(
#             nn.ReLu(True),
#
#         )
#     def forward(self, input):
#         res = self.res_block(input)
#         return input + res