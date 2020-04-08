import torch
import torch.nn as nn
from collections import OrderedDict


# https://github.com/gyshgx868/graph-ter

class EdgeConvolution(nn.Module):
    def __init__(self, k, in_features, out_features):
        super(EdgeConvolution, self).__init__()
        self.k = k
        self.conv = nn.Conv2d(
            in_features * 2, out_features, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = get_edge_feature(x, k=self.k)
        x = self.relu(self.bn(self.conv(x)))
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class Pooler(nn.Module):
    def __init__(self):
        super(Pooler, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.max_pool(x).view(batch_size, -1)
        x2 = self.avg_pool(x).view(batch_size, -1)
        x = torch.cat((x1, x2), dim=1)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def get_total_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total, 'Trainable': trainable}


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    x2 = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -x2 - inner - x2.transpose(2, 1)
    indices = pairwise_distance.topk(k=k, dim=-1)[1]
    return indices  # (batch_size, num_points, k)


def get_edge_feature(x, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    num_feats = x.size(1)

    indices = knn(x, k=k)  # (batch_size, num_points, k)
    indices_base = torch.arange(
        0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    indices = indices + indices_base
    indices = indices.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[indices, :]
    feature = feature.view(batch_size, num_points, k, num_feats)
    x = x.view(batch_size, num_points, 1, num_feats).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


class Encoder(nn.Module):
    def __init__(self, k=20):
        super(Encoder, self).__init__()
        self.conv0 = EdgeConvolution(k, in_features=3, out_features=64)
        self.conv1 = EdgeConvolution(k, in_features=64, out_features=64)
        self.conv2 = EdgeConvolution(k, in_features=64, out_features=128)
        self.conv3 = EdgeConvolution(k, in_features=128, out_features=256)
        self.conv4 = EdgeConvolution(k, in_features=256, out_features=512)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        features = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return features

class Tail(nn.Module):
    def __init__(self, k=20, in_features=1024):
        super(Tail, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(in_features, 512, kernel_size=1, bias=False)),
            ('bn0', nn.BatchNorm1d(512)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2)),
            ('conv1', EdgeConvolution(k, in_features=512, out_features=256)),
            ('conv2', EdgeConvolution(k, in_features=256, out_features=128))
        ]))

    def forward(self, x):
        x = self.conv(x)
        return x


class Backbone(nn.Module):
    def __init__(self, k=20, out_features=3):
        super(Backbone, self).__init__()
        self.encoder = Encoder(k=k)
        self.tail = Tail(k=k)
        self.decoder = nn.Sequential(OrderedDict([
            ('conv0', EdgeConvolution(k, in_features=256, out_features=128)),
            ('conv1', EdgeConvolution(k, in_features=128, out_features=64)),
            ('conv2', nn.Conv1d(64, out_features, kernel_size=1))
        ]))

    def forward(self, *args):
        if len(args) == 2:
            x, y = args[0], args[1]
            x1 = self.tail(self.encoder(x))
            x2 = self.tail(self.encoder(y))
            x = torch.cat((x1, x2), dim=1)  # B * 2F * N
            matrix = self.decoder(x)
            return matrix
        elif len(args) == 1:
            x = args[0]
            features = self.encoder(x)
            return features
        else:
            raise ValueError('Invalid number of arguments.')

def main():
    encoder = Encoder()
    encoder_para = get_total_parameters(encoder)
    print('Encoder:', encoder_para)
    x1 = torch.rand(4, 3, 1024)
    x2 = torch.rand(4, 3, 1024)
    y1 = encoder(x1)
    y2 = encoder(x2)
    print('Encoded:', y1.size(), y2.size())

    tail = Tail()
    tail_para = get_total_parameters(tail)
    print('Tail:', tail_para)
    z = tail(y1)
    print('Tail:', z.size())

    backbone = Backbone(k=20)
    print(backbone)
    backbone_para = get_total_parameters(backbone)
    print('Backbone:', backbone_para)
    matrix = backbone(x1, x2)
    print('Reconstruction:', matrix.size())
    features = backbone(x1)
    print('Features:', features.size())


if __name__ == '__main__':
    main()