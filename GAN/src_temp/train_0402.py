from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from in_out import PartDataset
from model import DGCNN_Encoder, Decoder, DGCNN_AE, Discriminator, Generator, PointDis, PointGen

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--class_name', type=str, default='Chair')
parser.add_argument('--batchSize', type=int, default=5, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default=2048, help='number of points')
parser.add_argument('--feat_dim', type=int, default=128, help='number of feature dim')
parser.add_argument('--k', type=int, default=40, help='number of neighbor')
parser.add_argument('--outf', type=str, default='wgan',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--clamp_lower', type=float, default=-0.02)
parser.add_argument('--clamp_upper', type=float, default=0.02)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

opt = parser.parse_args()
print (opt)

# opt.manualSeed = random.randint(1, 10000) # fix seed
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)

#%%

dataset = PartDataset(root = 'dataset/shapenetcore_partanno_segmentation_benchmark_v0', class_choice = opt.class_name, classification = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'dataset/shapenetcore_partanno_segmentation_benchmark_v0', class_choice = opt.class_name,classification = True, train = False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

print("[ train set:", len(dataset), "\ttest set:", len(test_dataset), "]")
num_classes = len(dataset.classes)
# print('classes', num_classes)

#%%
cudnn.benchmark = True

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# enc = DGCNN_Encoder(k = opt.k, feat_dim=opt.feat_dim)
dis = DGCNN_AE(num_points = opt.num_points, feat_dim=opt.feat_dim, k=opt.k)
gen = Decoder(feat_dim=opt.feat_dim, num_points = opt.num_points, k=opt.k)

if opt.model != '':
    dis.load_state_dict(torch.load(opt.model))

# print(gen)
# print(dis)

#%%

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

gen.apply(weights_init)
dis.apply(weights_init)

gen.cuda()
dis.cuda()

#%%

optimizerG = optim.Adagrad(gen.parameters(), lr = 0.0005)
optimizerD = optim.Adagrad(dis.parameters(), lr = 0.0005)

num_batch = len(dataset)/opt.batchSize
one = torch.FloatTensor([1]).cuda()
mone = one * -1


from tqdm import tqdm
for epoch in tqdm(range(opt.nepoch)):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        for diter in range(opt.Diters):
            for p in gen.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            optimizerD.zero_grad()
            data = data_iter.next()
            i += 1

            if i >= len(dataloader):
                break
            points, _ = data
            points = Variable(points)
            bs = points.size()[0]
            points = points.transpose(2, 1) # [batch, dim, points]
            points = points.cuda()

            real_z, real_pred = dis(points)
            # print("real noise = ", real_noise.size())
            # real_z = dis.z(points).detach()
            loss_real = torch.mean(real_pred)

            fake_noise = Variable(torch.randn(bs, 1, opt.feat_dim)).cuda()  # batch, 1, z_dim(128)
            # print("fake noise = ", fake_noise.size())
            fake = gen(fake_noise)       # batch, 3-dim, num_points
            fake_np = fake.transpose(2,1).data.cpu().numpy()    # batch, num_points, 3-dim

            fake_z, fake_pred = dis(fake)
            loss_fake = torch.mean(fake_pred)

            lossD = loss_real - loss_fake
            lossD.backward(one)
            optimizerD.step()
            # print('[%d: %d/%d] train lossD: %f' % (epoch, i, num_batch, lossD.data[0]))
            print('[%d: %d/%d] train \tlossD: %f' % (epoch, i, num_batch, lossD.item()))

        optimizerG.zero_grad()
        fake_noise = Variable(torch.randn(bs, 1, opt.feat_dim)).cuda()
        fake_points = gen(fake_noise)
        fake_z, fake_pred = dis(fake_points)
        lossG = torch.mean(fake_pred)
        lossG.backward(one)

        optimizerG.step()

        print('[%d: %d/%d] train \tlossD: %f \tlossG: %f' % (epoch, i, num_batch, lossD.item(), lossG.item()))

    torch.save(gen.state_dict(), '%s/modelD_%d.pth' % (opt.outf, epoch))
    torch.save(gen.state_dict(), '%s/modelG_%d.pth' % (opt.outf, epoch))