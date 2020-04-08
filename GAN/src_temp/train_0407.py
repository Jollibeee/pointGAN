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
from tensorboardX import SummaryWriter  #https://seongkyun.github.io/others/2019/05/11/pytorch_tensorboard/
# https://keep-steady.tistory.com/14

from in_out import PartDataset
from model import *
from visualize import plot_point_cloud


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("cuda")
    device = torch.device("cuda")


#%%

parser = argparse.ArgumentParser()
parser.add_argument('--class_name', type=str, default='Chair')
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default=2048, help='number of points')
parser.add_argument('--feat_dim', type=int, default=128, help='number of feature dim')
parser.add_argument('--k', type=int, default=20, help='number of neighbor')
parser.add_argument('--outf', type=str, default='wgan',  help='output folder')
# parser.add_argument('--model', type=str, default = 'wgan/model',  help='model path')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--e', type=int, default = '3',  help='model path')
parser.add_argument('--clamp_lower', type=float, default=-0.02)
parser.add_argument('--clamp_upper', type=float, default=0.02)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

opt = parser.parse_args()
print (opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#%%

dataset = PartDataset(root = 'dataset/shapenetcore_partanno_segmentation_benchmark_v0', class_choice = opt.class_name, classification = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'dataset/shapenetcore_partanno_segmentation_benchmark_v0', class_choice = opt.class_name, classification = True, train = False)
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

summary = SummaryWriter(opt.outf)
# summary = SummaryWriter()

# enc = DGCNN_Encoder(k = opt.k, feat_dim=opt.feat_dim)
dis = Dis(opt)
# gen = Gen(opt)
gen = Gen2(opt)

# if opt.model != '':
#     path = '%s/model_%d.pth' %(opt.model, opt.e)
#     modelD = '%s/modelD_%d.pth' %(opt.model, opt.e)
#     modelG = '%s/modelG_%d.pth' % (opt.model, opt.e)
#     dis.load_state_dict(torch.load(modelD))
#     gen.load_state_dict(torch.load(modelG))
#     # optimizerG.load_state_dict

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


gen.cuda()
dis.cuda()

gen.apply(weights_init)
dis.apply(weights_init)

optimizerG = optim.Adagrad(gen.parameters(), lr = 0.0005)
optimizerD = optim.Adagrad(dis.parameters(), lr = 0.0005)

#%%
## BeGAN https://github.com/eriklindernoren/PyTorch-GAN/blob/a163b82beff3d01688d8315a3fd39080400e7c01/implementations/began/began.py#L38
# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)
# gen.cuda()
# dis.cuda()
# gen.apply(weights_init_normal)
# dis.apply(weights_init_normal)
# optimizerG = optim.Adam(gen.parameters(), lr = 0.0002, betas=(0.5, 0.999))
# optimizerD = optim.Adam(dis.parameters(), lr = 0.0002, betas=(0.5, 0.999))

# # BEGAN hyper parameters
# gamma = 0.75
# lambda_k = 0.001
# k = 0.0

#%%

if opt.model != '':
    print("load model")
    path = '%s/model_%d.pth' %(opt.model, opt.e)
    ckpt = torch.load(path)
    dis.load_state_dict(ckpt['modelD'])
    gen.load_state_dict(ckpt['modelG'])
    optimizerG.load_state_dict(ckpt['optimizerG'])
    optimizerD.load_state_dict(ckpt['optimizerD'])


num_batch = len(dataset)/opt.batchSize
one = torch.FloatTensor([1]).cuda()
mone = one * -1


from tqdm import tqdm

# for epoch in tqdm(range(opt.e, opt.nepoch)):
for epoch in tqdm(range(opt.nepoch)):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
    # while i < 5:
        for diter in range(opt.Diters):
            for p in dis.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            optimizerD.zero_grad()
            data = data_iter.next()
            i += 1

            if i >= len(dataloader):
                break
            points, _ = data
            points = Variable(points)
            bs = points.size()[0]
            # points = points.transpose(2, 1) # [batch, dim, points]
            points = points.cuda()


            ## Train with real
            real_logit, real_prob = dis(points)
            loss_real = torch.mean(real_prob)

            fake_noise = Variable(torch.randn(bs, opt.feat_dim)).cuda()  # batch, 1, z_dim(128)
            # print("fake noise = ", fake_noise.size())
            fake = gen(fake_noise)       # batch, 3-dim, num_points
            # fake_np = fake.transpose(2,1).data.cpu().numpy()    # batch, num_points, 3-dim

            fake_logit, fake_prob = dis(fake)
            loss_fake = torch.mean(fake_prob)

            lossD = loss_real - loss_fake
            # lossD.backward()
            lossD.backward(mone)
            optimizerD.step()
            # print('[%d: %d/%d] train lossD: %f' % (epoch, i, num_batch, lossD.data[0]))
            print('[%d: %d/%d] train \tlossD: %f' % (epoch, i, num_batch, lossD.item()))

        optimizerG.zero_grad()
        fake_noise = Variable(torch.randn(bs, opt.feat_dim)).cuda()
        fake_points = gen(fake_noise)

        fake_logit, fake_pred = dis(fake_points)
        lossG = torch.mean(fake_pred)
        lossG.backward(mone)
        # lossG.backward()

        optimizerG.step()
        print('[%d: %d/%d] train \tlossD: %f \tlossG: %f' % (epoch, i, num_batch, lossD.item(), lossG.item()))

        niter = epoch *  len(dataloader) + i
        # print(niter)
        summary.add_scalar('loss/lossD', lossD.item(), niter)
        summary.add_scalar('loss/lossG', lossG.item(), niter)
        # summary.add_scalar('loss/loss', lossD.item() + lossG.item(), niter)
        # summary.add_scalars('loss', {"lossD": lossD.item(), "lossG": lossG.item()}, niter)
        summary.add_histogram('synthetic_logit', fake_logit)
        summary.add_histogram('real_logit', real_logit)
        # summary.close()

    renf = os.path.join(opt.outf, 'render')
    try:
        os.makedirs(renf)
    except OSError:
        pass
    import scipy.io as sio

    # points = points.transpose(2, 1)
    # print(points.size())
    fake_points_np = fake_points.cpu()
    # print(fake_points_np.size())
    # fake_points_np = fake_points_np.transpose(2, 1)
    fake_points_np = fake_points_np.detach().numpy()

    path = os.path.join(renf, str(epoch))
    try:
        os.makedirs(path)
    except OSError:
        pass

    # sio.savemat('%s/Gen_points_%d.mat' %(renf, epoch), {'Y': fake_points_np})
    np.savez('%s/points_%d.npz' %(path, epoch), points.cpu().numpy())
    np.savez('%s/Gen_points_%d.npz' % (path, epoch), fake_points_np)

    for n in range(fake_points_np.shape[0]):

        file = 'points_' + str(epoch) + '_' + str(n) + '.png'
        file = os.path.join(path, file)
        # points_np = points.cpu().numpy()
        plot_point_cloud(fake_points_np[n], file)

    modelf = os.path.join(opt.outf, 'model')
    try:
        os.makedirs(modelf)
    except OSError:
        pass
    # https://tutorials.pytorch.kr/beginner/saving_loading_models.html
    torch.save(dis.state_dict(), '%s/modelD_%d.pth' % (modelf, epoch))
    torch.save(gen.state_dict(), '%s/modelG_%d.pth' % (modelf, epoch))
    torch.save({'modelD':dis.state_dict(),'modelG':gen.state_dict(),
                'optimizerD': optimizerD.state_dict(), 'optimizerG':optimizerG.state_dict(),
                'lossD':lossD.item(), 'lossG': lossG.item(), 'epoch': epoch},
               '%s/model_%d.pth' %(modelf, epoch))




# https://github.com/fxia22/pointGAN/blob/74b6c432c5eaa1e0a833e755f450df2ee2c5488e/gen_gan.py