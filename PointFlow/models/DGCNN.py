import os
import sys
import copy
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn
from models.flow import get_point_cnf
from models.flow import get_latent_cnf
from utils import truncated_normal, reduce_tensor, standard_normal_logprob


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature



class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(Encoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        if self.use_deterministic_encoder:
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc_bn1 = nn.BatchNorm1d(256)
            self.fc_bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, zdim)
        else:
            # Mapping to [c], cmean
            self.fc1_m = nn.Linear(512, 256)
            self.fc2_m = nn.Linear(256, 128)
            self.fc3_m = nn.Linear(128, zdim)
            self.fc_bn1_m = nn.BatchNorm1d(256)
            self.fc_bn2_m = nn.BatchNorm1d(128)

            # Mapping to [c], cmean
            self.fc1_v = nn.Linear(512, 256)
            self.fc2_v = nn.Linear(256, 128)
            self.fc3_v = nn.Linear(128, zdim)
            self.fc_bn1_v = nn.BatchNorm1d(256)
            self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        if self.use_deterministic_encoder:
            ms = F.relu(self.fc_bn1(self.fc1(x)))
            ms = F.relu(self.fc_bn2(self.fc2(ms)))
            ms = self.fc3(ms)
            m, v = ms, 0
        else:
            m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
            m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
            m = self.fc3_m(m)
            v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
            v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
            v = self.fc3_v(v)

        return m, v


# Model

class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()

        self.args = args
        self.k = args.k

        self.input_dim = args.input_dim
        self.zdim = args.zdim
        self.use_deterministic_encoder = args.use_deterministic_encoder

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
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
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        self.encoder = Encoder(
                zdim=args.zdim, input_dim=args.input_dim,
                use_deterministic_encoder=args.use_deterministic_encoder)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class PointFlow(nn.Module):
    def __init__(self, args):
        super(PointFlow, self).__init__()
        self.input_dim = args.input_dim
        self.zdim = args.zdim
        self.use_latent_flow = args.use_latent_flow
        self.use_deterministic_encoder = args.use_deterministic_encoder
        self.prior_weight = args.prior_weight
        self.recon_weight = args.recon_weight
        self.entropy_weight = args.entropy_weight
        self.distributed = args.distributed
        self.truncate_std = None
        self.encoder = Encoder(
                zdim=args.zdim, input_dim=args.input_dim,
                use_deterministic_encoder=args.use_deterministic_encoder)
        self.point_cnf = get_point_cnf(args)
        self.latent_cnf = get_latent_cnf(args) if args.use_latent_flow else nn.Sequential()

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent

    def multi_gpu_wrapper(self, f):
        self.encoder = f(self.encoder)
        self.point_cnf = f(self.point_cnf)
        self.latent_cnf = f(self.latent_cnf)

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        opt = _get_opt_(list(self.encoder.parameters()) + list(self.point_cnf.parameters())
                        + list(list(self.latent_cnf.parameters())))
        return opt

    def forward(self, x, opt, step, writer=None):
        opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            z = z_mu + 0 * z_sigma
        else:
            z = self.reparameterize_gaussian(z_mu, z_sigma)

        # Compute H[Q(z|X)]
        if self.use_deterministic_encoder:
            entropy = torch.zeros(batch_size).to(z)
        else:
            entropy = self.gaussian_entropy(z_sigma)

        # Compute the prior probability P(z)
        if self.use_latent_flow:
            w, delta_log_pw = self.latent_cnf(z, None, torch.zeros(batch_size, 1).to(z))
            log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
            delta_log_pw = delta_log_pw.view(batch_size, 1)
            log_pz = log_pw - delta_log_pw
        else:
            log_pz = torch.zeros(batch_size, 1).to(z)

        # Compute the reconstruction likelihood P(X|z)
        z_new = z.view(*z.size())
        z_new = z_new + (log_pz * 0.).mean()
        y, delta_log_py = self.point_cnf(x, z_new, torch.zeros(batch_size, num_points, 1).to(x))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py

        # Loss
        entropy_loss = -entropy.mean() * self.entropy_weight
        recon_loss = -log_px.mean() * self.recon_weight
        prior_loss = -log_pz.mean() * self.prior_weight
        loss = entropy_loss + prior_loss + recon_loss
        loss.backward()
        opt.step()

        # LOGGING (after the training)
        if self.distributed:
            entropy_log = reduce_tensor(entropy.mean())
            recon = reduce_tensor(-log_px.mean())
            prior = reduce_tensor(-log_pz.mean())
        else:
            entropy_log = entropy.mean()
            recon = -log_px.mean()
            prior = -log_pz.mean()

        recon_nats = recon / float(x.size(1) * x.size(2))
        prior_nats = prior / float(self.zdim)

        if writer is not None:
            writer.add_scalar('train/entropy', entropy_log, step)
            writer.add_scalar('train/prior', prior, step)
            writer.add_scalar('train/prior(nats)', prior_nats, step)
            writer.add_scalar('train/recon', recon, step)
            writer.add_scalar('train/recon(nats)', recon_nats, step)

        return {
            'entropy': entropy_log.cpu().detach().item()
            if not isinstance(entropy_log, float) else entropy_log,
            'prior_nats': prior_nats,
            'recon_nats': recon_nats,
        }

    def encode(self, x):
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            return self.reparameterize_gaussian(z_mu, z_sigma)

    def decode(self, z, num_points, truncate_std=None):
        # transform points from the prior to a point cloud, conditioned on a shape code
        y = self.sample_gaussian((z.size(0), num_points, self.input_dim), truncate_std)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return y, x

    def sample(self, batch_size, num_points, truncate_std=None, truncate_std_latent=None, gpu=None):
        assert self.use_latent_flow, "Sampling requires `self.use_latent_flow` to be True."
        # Generate the shape code from the prior
        w = self.sample_gaussian((batch_size, self.zdim), truncate_std_latent, gpu=gpu)
        z = self.latent_cnf(w, None, reverse=True).view(*w.size())
        # Sample points conditioned on the shape code
        y = self.sample_gaussian((batch_size, num_points, self.input_dim), truncate_std, gpu=gpu)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return z, x

    def reconstruct(self, x, num_points=None, truncate_std=None):
        num_points = x.size(1) if num_points is None else num_points
        z = self.encode(x)
        _, x = self.decode(z, num_points, truncate_std)
        return x