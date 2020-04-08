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


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)  # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNN_Encoder(nn.Module):
    def __init__(self, args):
        super(DGCNN_Encoder, self).__init__()
        self.k = args.k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.feat_dim)

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
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.feat_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        # print("Dis ", x.shape)
        x = x.transpose(2, 1)
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


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2



class Dis_Decoder(nn.Module):
    def __init__(self, args):
        super(Dis_Decoder, self).__init__()
        self.k = args.k

        self.fc_1 = fc_bn_block(args.feat_dim, 128)
        self.drop_1 = nn.Dropout(0.5)
        self.fc_2 = fc_bn_block(128, 64)
        self.drop_2 = nn.Dropout(0.5)
        self.fc_3 = fc_bn_block(64, 1)
        self.drop_3 = nn.Dropout(0.5)
        # self.fc_6 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        # print("dis", x.shape)
        # x = x.transpose(2, 1)
        B, N, C = x.shape
        x = x.reshape([B, -1])
        x = self.fc_1(x)
        x = self.drop_1(x)
        x = self.fc_2(x)
        x = self.drop_2(x)
        x = self.fc_3(x)
        # softmax
        # x = F.log_softmax(x, dim=-1)
        return x

class Gen(nn.Module):
    def __init__(self, args):
        super(Gen, self).__init__()
        self.k = args.k

        self.fc_1 = fc_bn_block(args.feat_dim, 512)
        self.drop_1 = nn.Dropout(0.5)
        self.fc_2 = fc_bn_block(512, 1024)
        self.drop_2 = nn.Dropout(0.5)
        # self.fc_3 = fc_bn_block(1024, 1024)
        # self.drop_3 = nn.Dropout(0.5)
        self.fc_3 = nn.Linear(1024, args.num_points*3)
        self.num_points = args.num_points

    def forward(self, x):
        B, N = x.shape
        x = self.fc_1(x)
        x = self.drop_1(x)
        x = self.fc_2(x)
        x = self.drop_2(x)
        x = self.fc_3(x)
        # x = self.drop_3(x)
        # x = self.fc_4(x)
        # softmax
        # x = F.log_softmax(x, dim=-1)
        # print("Gen ", x.shape)  #[batch, 1024]
        x = x.reshape([-1, self.num_points, 3])     # [batch, 2048, 3]
        return x


    # class ReconstructionNet(nn.Module):


class Gen2(nn.Module):
    def __init__(self, args):
        super(Gen2, self).__init__()
        self.k = args.k

        self.fc_1 = fc_bn_block(args.feat_dim, 64)
        self.drop_1 = nn.Dropout(0.5)
        self.fc_2 = fc_bn_block(64, 128)
        self.drop_2 = nn.Dropout(0.5)
        self.fc_3 = fc_bn_block(128, 512)
        self.drop_3 = nn.Dropout(0.5)
        self.fc_4 = fc_bn_block(512, 1024)
        self.drop_4 = nn.Dropout(0.5)
        self.fc_5 = nn.Linear(1024, args.num_points*3)
        self.num_points = args.num_points

    def forward(self, x):
        B, N = x.shape
        x = self.fc_1(x)
        x = self.drop_1(x)
        x = self.fc_2(x)
        x = self.drop_2(x)
        x = self.fc_3(x)
        x = self.drop_3(x)
        x = self.fc_4(x)
        x = self.drop_4(x)
        x = self.fc_5(x)
        # softmax
        # x = F.log_softmax(x, dim=-1)
        # print("Gen ", x.shape)  #[batch, 1024]
        x = x.reshape([-1, self.num_points, 3])     # [batch, 2048, 3]
        return x


    # class ReconstructionNet(nn.Module):
class Dis(nn.Module):
    def __init__(self, args):
        # super(ReconstructionNet, self).__init__()
        super(Dis, self).__init__()
        self.encoder = DGCNN_Encoder(args)
        self.decoder = Dis_Decoder(args)
        self.loss = ChamferLoss()

    def forward(self, input):
        feature = self.encoder(input)
        logit = self.decoder(feature)
        # prob = F.sigmoid(logit)
        prob = torch.sigmoid(logit)
        return logit, prob

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, input, output):
        # input shape  (batch_size, 2048, 3)
        # output shape (batch_size, 2025, 3)
        return self.loss(input, output)



#%%



def conv_bn_block(input, output, kernel_size):
    '''
    标准卷积块（conv + bn + relu）
    :param input: 输入通道数
    :param output: 输出通道数
    :param kernel_size: 卷积核大小
    :return:
    '''
    return nn.Sequential(
        nn.Conv1d(input, output, kernel_size),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )


def fc_bn_block(input, output):
    '''
    标准全连接块（fc + bn + relu）
    :param input:  输入通道数
    :param output:  输出通道数
    :return:  卷积核大小
    '''
    return nn.Sequential(
        nn.Linear(input, output),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )

# https://github.com/ToughStoneX/DGCNN/blob/master/Model/dynami_graph_cnn.py

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_points', type=int, default=2048, help='number of points')
    parser.add_argument('--feat_dim', type=int, default=128, help='number of feature dim')
    parser.add_argument('--k', type=int, default=20, help='number of neighbor')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # real = torch.randn([5, 2048, 3])
    real = Variable(torch.randn(5, 2048, 3)).cuda()
    # noise = Variable(torch.randn(5, args.feat_dim)).cuda()
    # real = real.to(device)
    noise = torch.randn([5, 128])
    # noise = noise.to(device)

    Gen = Gen(args)
    Dis = Dis(args)

    g_out = Gen(noise)
    real_logit, real_prob = Dis(real)
    fake_logit, fake_prob = Dis(g_out)

