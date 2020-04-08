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
from datetime import datetime
from torch.autograd import grad
import logging
from pcutil import plot_3d_point_cloud
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(device)


#%%

parser = argparse.ArgumentParser()
parser.add_argument('--class_name', type=str, default='Chair')
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default=2048, help='number of points')
parser.add_argument('--feat_dim', type=int, default=128, help='number of feature dim')
parser.add_argument('--k', type=int, default=20, help='number of neighbor')
parser.add_argument('--outf', type=str, default='AAE',  help='output folder')
# parser.add_argument('--model', type=str, default = 'wgan/model',  help='model path')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--starting_epoch', type=int, default = '1',  help='model path')
parser.add_argument('--clamp_lower', type=float, default=-0.02)
parser.add_argument('--clamp_upper', type=float, default=0.02)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

opt = parser.parse_args()
print (opt)



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

one = torch.FloatTensor([1]).cuda()
mone = one * -1

try:
    os.makedirs(opt.outf)
except OSError:
    pass

summary = SummaryWriter(opt.outf)
# summary = SummaryWriter()

import re
from os import listdir, makedirs
from os.path import join, exists
from shutil import rmtree
from time import sleep

def setup_logging(log_dir):
    makedirs(log_dir, exist_ok=True)

    logpath = join(log_dir, 'log.txt')
    filemode = 'a' if exists(logpath) else 'w'

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=logpath,
                        filemode=filemode)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def main(opt):
    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    setup_logging(opt.outf)
    log = logging.getLogger(__name__)
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    G = Generator(opt).to(device)
    E = Encoder(opt).to(device)
    D = Discriminator(opt).to(device)

    G = Generator(opt).cuda()
    E = Encoder(opt).cuda()
    D = Discriminator(opt).cuda()

    from loss import ChamferLoss
    recon_loss = ChamferLoss().to(device)
    # recon_loss = ChamferLoss().cuda()

    # fixed_noise = torch.FloatTensor(opt.batchSize, opt.feat_dim, 1)
    fixed_noise = Variable(torch.FloatTensor(opt.batchSize, opt.feat_dim, 1))
    fixed_noise.normal_(mean = 0.0, std = 0.2)
    # noise = torch.FloatTensor(opt.batchSize, opt.feat_dim)
    noise = Variable(torch.FloatTensor(opt.batchSize, opt.feat_dim))
    fixed_noise = fixed_noise.to(device)
    noise = noise.to(device)
    # fixed_noise = fixed_noise.cuda()
    # noise = noise.cuda()
    from itertools import chain
    EG_optim = optim.Adagrad(chain(E.parameters(), G.parameters()), lr = 0.0005)
    D_optim = optim.Adagrad(D.parameters(), lr = 0.0005)

    if opt.starting_epoch > 1:
        path = '%s/model_%d.pth' % (opt.model, opt.starting_epoch)
        ckpt = torch.load(path)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        E.load_state_dict(ckpt['E'])
        D_optim.load_state_dict(ckpt['D_optim'])
        EG_optim.load_state_dict(ckpt['EG_optim'])


    for epoch in range(opt.starting_epoch, opt.nepoch):
        start_epoch_time = datetime.now()

        G.train()
        E.train()
        D.train()

        total_loss_d = 0.0
        total_loss_eg = 0.0

        data_iter = iter(dataloader)
        for i, point_data in enumerate(data_iter, 1):
            log.debug('-' * 20)
            X, _ = point_data
            X = Variable(X)
            X = X.to(device)
            # X = Variable(X)

            # print(X.type())
            # X = X.cuda()

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)

            codes, _, _ = E(X)
            # print(codes.type())
            noise.normal_(mean=0.0, std=0.2)
            synth_logit = D(codes)
            real_logit = D(noise)
            loss_d = torch.mean(synth_logit) - torch.mean(real_logit)

            alpha = torch.rand(opt.batchSize, 1).to(device)
            # alpha = torch.rand(opt.batchSize, 1).cuda()
            differences = codes - noise
            interpolates = noise + alpha * differences
            disc_interpolates = D(interpolates)

            gradients = grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(disc_interpolates).to(device),
                # grad_outputs=torch.ones_like(disc_interpolates).cuda(),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
            gradient_penalty = ((slopes - 1) ** 2).mean()
            gp_lambda = 10
            loss_gp = gp_lambda * gradient_penalty
            ###
            loss_d += loss_gp

            D_optim.zero_grad()
            D.zero_grad()

            loss_d.backward(retain_graph=True)
            # loss_d.backward(mone)
            total_loss_d += loss_d.item()
            D_optim.step()

            # EG part of training
            X_rec = G(codes)

            loss_e = torch.mean(0.05 * recon_loss(X.permute(0, 2, 1) + 0.5, X_rec.permute(0, 2, 1) + 0.5))

            synth_logit = D(codes)

            loss_g = -torch.mean(synth_logit)

            loss_eg = loss_e + loss_g
            EG_optim.zero_grad()
            E.zero_grad()
            G.zero_grad()

            loss_eg.backward()
            # loss_eg.backward(mone)
            total_loss_eg += loss_eg.item()
            EG_optim.step()

            log.debug(f'[{epoch}: ({i})] '
                      f'Loss_D: {loss_d.item():.4f} '
                      f'(GP: {loss_gp.item(): .4f}) '
                      f'Loss_EG: {loss_eg.item():.4f} '
                      f'(REC: {loss_e.item(): .4f}) '
                      f'Time: {datetime.now() - start_epoch_time}')

            X = X.cpu()

        log.debug(
            f'[{epoch}/{opt.nepoch}] '
            f'Loss_D: {total_loss_d / i:.4f} '
            f'Loss_EG: {total_loss_eg / i:.4f} '
            f'Time: {datetime.now() - start_epoch_time}')

        #
        # Save intermediate results
        #
        G.eval()
        E.eval()
        D.eval()
        with torch.no_grad():
            fake = G(fixed_noise).data.cpu().numpy()
            codes, _, _ = E(X)
            X_rec = G(codes).data.cpu().numpy()

        renf = os.path.join(opt.outf, 'render')
        try:
            os.makedirs(renf)
        except OSError:
            pass



        png_path = join(renf, 'samples')
        try:
            os.makedirs(png_path)
        except OSError:
            pass

        for k in range(X.shape[0]):
            fig = plot_3d_point_cloud(X[k][0], X[k][1], X[k][2],
                                      in_u_sphere=True, show=False,
                                      title=str(epoch))
            fig.savefig(join(renf, 'samples', f'{epoch:05}_{k}_real.png'))
            plt.close(fig)

        for k in range(fake.shape[0]):

            fig = plot_3d_point_cloud(fake[k][0], fake[k][1], fake[k][2],
                                      in_u_sphere=True, show=False,
                                      title=str(epoch))
            fig.savefig(
                join(renf, 'samples', f'{epoch:05}_{k}_fixed.png'))
            plt.close(fig)

        for k in range(X_rec.shape[0]):

            fig = plot_3d_point_cloud(X_rec[k][0],
                                      X_rec[k][1],
                                      X_rec[k][2],
                                      in_u_sphere=True, show=False,
                                      title=str(epoch))
            fig.savefig(join(renf, 'samples',
                             f'{epoch:05}_{k}_reconstructed.png'))
            plt.close(fig)

        modelf = os.path.join(opt.outf, 'model')
        try:
            os.makedirs(modelf)
        except OSError:
            pass

        if epoch % opt.Diters == 0:

            torch.save({'G':G.state_dict(),
                        'D':D.state_dict(),
                        'E':E.state_dict(),
                        'D_optim':D_optim.state_dict(),
                        'EG_optim': EG_optim.state_dict()},
                       '%s/model_%d.pth' %(modelf, epoch))



#%%
if __name__ == '__main__':
    logger = logging.getLogger()
    main(opt)
#%%
    #
    # if opt.starting_epoch > 1:
    #     path = '%s/model_%d.pth' % (opt.model, opt.starting_epoch)
    #     ckpt = torch.load(path)
    #     G.load_state_dict(ckpt['G'])
    #     D.load_state_dict(ckpt['D'])
    #     E.load_state_dict(ckpt['E'])
    #     D_optim.load_state_dict(ckpt['D_optim'])
    #     EG_optim.load_state_dict(ckpt['EG_optim'])
