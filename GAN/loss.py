# # import torch
# # from chamferdist import ChamferDistance
# #
# #
# #
# # # Create two random pointclouds
# # # (Batchsize x Number of points x Number of dims)
# # pc1 = torch.randn(1, 100, 3).cuda().contiguous()
# # pc2 = torch.randn(1, 50, 3).cuda().contiguous()
# # pc1.requires_grad = True
# #
# # # Initialize Chamfer distance module
# # chamferDist = ChamferDistance()
# # # Compute Chamfer distance, and indices of closest points
# # # - dist1 is direction pc1 -> pc2 (for each point in pc1,
# # #   gets closest point in pc2)
# # # - dist 2 is direction pc2 -> pc1 (for each point in pc2,
# # #   gets closest point in pc1)
# # dist1, dist2, idx1, idx2 = chamferDist(pc1, pc2)
# # print(dist1.shape, dist2.shape, idx1.shape, idx2.shape)
# #
# # # Usually, dist1 is not equal to dist2 (as the associations
# # # vary, in both directions). To get a consistent measure,
# # # usually we average both the distances
# # cdist = 0.5 * (dist1.mean() + dist2.mean())
# # print('Chamfer distance:', cdist)
# # cdist.backward()
#
# from __future__ import print_function
# import time
# from torch.autograd import Variable
# from ChamferDistancePytorch.fscore import fscore
# from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
# from ChamferDistancePytorch import chamfer_python
#
# # from __future__ import print_function
# import argparse
# import os
# import random
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
# from torch.autograd import Variable
# from torch.autograd import Function
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import pdb
# import torch.nn.functional as F
#
#
#
#
# cham3D = dist_chamfer_3D.chamfer_3DDist()
#
# def ChamferLoss1(p1, p2):
#     p1 = p1.cuda()
#     p2 = p2.cuda()
#     points1 = Variable(p1, requires_grad=True)
#     points2 = Variable(p2)
#     dist1, dist2, idx1, idx2 = cham3D(points1, points2)
#     cdist = 0.5 * (dist1.mean() + dist2.mean())
#     # print('Chamfer distance:', cdist)
#     print('Chamfer distance:', cdist.tolist())
#     print('Chamfer distance:', cdist)
#     # print(cdist.tolist())
#     cdist.backward()
#
#
# def ChamferDistance(x, y):  # for example, x = batch,2025,3 y = batch,2048,3
#     #   compute chamfer distance between tow point clouds x and y
#
#     x_size = x.size()
#     y_size = y.size()
#     assert (x_size[0] == y_size[0])
#     assert (x_size[2] == y_size[2])
#     x = torch.unsqueeze(x, 1)  # x = batch,1,2025,3
#     y = torch.unsqueeze(y, 2)  # y = batch,2048,1,3
#
#     x = x.repeat(1, y_size[1], 1, 1)  # x = batch,2048,2025,3
#     y = y.repeat(1, 1, x_size[1], 1)  # y = batch,2048,2025,3
#
#     x_y = x - y
#     x_y = torch.pow(x_y, 2)  # x_y = batch,2048,2025,3
#     x_y = torch.sum(x_y, 3, keepdim=True)  # x_y = batch,2048,2025,1
#     x_y = torch.squeeze(x_y, 3)  # x_y = batch,2048,2025
#     x_y_row, _ = torch.min(x_y, 1, keepdim=True)  # x_y_row = batch,1,2025
#     x_y_col, _ = torch.min(x_y, 2, keepdim=True)  # x_y_col = batch,2048,1
#
#     x_y_row = torch.mean(x_y_row, 2, keepdim=True)  # x_y_row = batch,1,1
#     x_y_col = torch.mean(x_y_col, 1, keepdim=True)  # batch,1,1
#     x_y_row_col = torch.cat((x_y_row, x_y_col), 2)  # batch,1,2
#     chamfer_distance, _ = torch.max(x_y_row_col, 2, keepdim=True)  # batch,1,1
#     # chamfer_distance = torch.reshape(chamfer_distance,(x_size[0],-1))  #batch,1
#     # chamfer_distance = torch.squeeze(chamfer_distance,1)    # batch
#     chamfer_distance = torch.mean(chamfer_distance)
#     return chamfer_distance
#
#
# class ChamferLoss(nn.Module):
#     # chamfer distance loss
#     def __init__(self):
#         super(ChamferLoss, self).__init__()
#
#     def forward(self, x, y):
#         return ChamferDistance(x, y)
#
#
#
# if __name__ == '__main__':
#     p1 = torch.rand(32, 2000, 3).cuda()
#     p2 = torch.rand(32, 1000, 3).cuda()
#
#     ChamferLoss1(p1, p2)
#
#     print("")
#
#     chamferloss = ChamferLoss()
#     chamferloss.cuda()
#     loss = chamferloss(p1, p2)
#     print(loss)
#     print(loss.item())
#
#


#%%

import torch
import torch.nn as nn


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
            zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P

import torch
from pyinn.utils import Stream, load_kernel
from torch.autograd import Function
from torch.nn.modules.module import Module


class EMD(Module):
    def __init__(self):
        super().__init__()
        self.emd_function = EMDFunction.apply

    def forward(self, input1, input2):
        return self.emd_function(input1, input2)


class EMDFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        assert xyz1.dim() == 3 and xyz1.is_cuda and xyz2.is_cuda
        assert xyz1.shape[-1] == 3  # as done by Panos
        batch_size, num_pts, pt_dim = xyz1.size()
        _, m, _ = xyz2.size()

        match = torch.zeros(batch_size, m, num_pts).cuda()
        cost = torch.zeros(batch_size, ).cuda()
        temp = torch.zeros(batch_size, 2 * (m + num_pts)).cuda()

        with torch.cuda.device_of(xyz1):
            # 1) get matching
            f = load_kernel('approxmatch', approxmatch_kernel)
            f(block=(512, 1, 1),  # (CUDA_NUM_THREADS,1,1),
              grid=(32, 1, 1),  # GET_BLOCKS(n),1,1),
              args=[batch_size, num_pts, m, xyz1.data_ptr(), xyz2.data_ptr(),
                    match.data_ptr(), temp.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            # 2) calculate matching cost
            g = load_kernel('matchcost', matchcost_kernel)
            g(block=(512, 1, 1),  # (CUDA_NUM_THREADS, 1, 1),
              grid=(32, 1, 1),  # (GET_BLOCKS(n), 1, 1),
              args=[batch_size, num_pts, m, xyz1.data_ptr(), xyz2.data_ptr(),
                    match.data_ptr(), cost.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(xyz1, xyz2, match)
        del temp

        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors

        batch_size, num_pts, _ = xyz1.size()
        _, m, _ = xyz2.size()

        grad1 = torch.zeros_like(xyz1).cuda()
        grad2 = torch.zeros_like(xyz2).cuda()

        with torch.cuda.device_of(grad_cost):
            if xyz1.requires_grad:
                f = load_kernel('matchcostgrad1', matchcostgrad1_kernel)
                f(block=(512, 1, 1),  # (CUDA_NUM_THREADS, 1, 1),
                  grid=(32, 1, 1),  # (GET_BLOCKS(xyz1.numel()), 1, 1),
                  args=[batch_size, num_pts, m, xyz1.data_ptr(),
                        xyz2.data_ptr(), match.data_ptr(), grad1.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            if xyz2.requires_grad:
                g = load_kernel('matchcostgrad2', matchcostgrad2_kernel)
                g(block=(256, 1, 1),  # (CUDA_NUM_THREADS, 1, 1),
                  grid=(32, 32, 1),  # (GET_BLOCKS(xyz2.numel()), 1, 1),
                  args=[batch_size, num_pts, m, xyz1.data_ptr(),
                        xyz2.data_ptr(), match.data_ptr(), grad2.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad1 * grad_cost.view(-1, 1, 1), grad2 * grad_cost.view(-1, 1,
                                                                        1)


approxmatch_kernel = '''
extern "C"
__global__ void approxmatch(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,float * __restrict__ match,float * temp){
        float * remainL=temp+blockIdx.x*(n+m)*2, * remainR=temp+blockIdx.x*(n+m)*2+n,*ratioL=temp+blockIdx.x*(n+m)*2+n+m,*ratioR=temp+blockIdx.x*(n+m)*2+n+m+n;
        float multiL,multiR;
        if (n>=m){
                multiL=1;
                multiR=n/m;
        }else{
                multiL=m/n;
                multiR=1;
        }
        const int Block=1024;
        __shared__ float buf[Block*4];
        for (int i=blockIdx.x;i<b;i+=gridDim.x){
                for (int j=threadIdx.x;j<n*m;j+=blockDim.x)
                        match[i*n*m+j]=0;
                for (int j=threadIdx.x;j<n;j+=blockDim.x)
                        remainL[j]=multiL;
                for (int j=threadIdx.x;j<m;j+=blockDim.x)
                        remainR[j]=multiR;
                __syncthreads();
                for (int j=7;j>=-2;j--){
                        float level=-powf(4.0f,j);
                        if (j==-2){
                                level=0;
                        }
                        for (int k0=0;k0<n;k0+=blockDim.x){
                                int k=k0+threadIdx.x;
                                float x1=0,y1=0,z1=0;
                                if (k<n){
                                        x1=xyz1[i*n*3+k*3+0];
                                        y1=xyz1[i*n*3+k*3+1];
                                        z1=xyz1[i*n*3+k*3+2];
                                }
                                float suml=1e-9f;
                                for (int l0=0;l0<m;l0+=Block){
                                        int lend=min(m,l0+Block)-l0;
                                        for (int l=threadIdx.x;l<lend;l+=blockDim.x){
                                                float x2=xyz2[i*m*3+l0*3+l*3+0];
                                                float y2=xyz2[i*m*3+l0*3+l*3+1];
                                                float z2=xyz2[i*m*3+l0*3+l*3+2];
                                                buf[l*4+0]=x2;
                                                buf[l*4+1]=y2;
                                                buf[l*4+2]=z2;
                                                buf[l*4+3]=remainR[l0+l];
                                        }
                                        __syncthreads();
                                        for (int l=0;l<lend;l++){
                                                float x2=buf[l*4+0];
                                                float y2=buf[l*4+1];
                                                float z2=buf[l*4+2];
                                                float d=level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
                                                float w=__expf(d)*buf[l*4+3];
                                                suml+=w;
                                        }
                                        __syncthreads();
                                }
                                if (k<n)
                                        ratioL[k]=remainL[k]/suml;
                        }
                        /*for (int k=threadIdx.x;k<n;k+=gridDim.x){
                                float x1=xyz1[i*n*3+k*3+0];
                                float y1=xyz1[i*n*3+k*3+1];
                                float z1=xyz1[i*n*3+k*3+2];
                                float suml=1e-9f;
                                for (int l=0;l<m;l++){
                                        float x2=xyz2[i*m*3+l*3+0];
                                        float y2=xyz2[i*m*3+l*3+1];
                                        float z2=xyz2[i*m*3+l*3+2];
                                        float w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*remainR[l];
                                        suml+=w;
                                }
                                ratioL[k]=remainL[k]/suml;
                        }*/
                        __syncthreads();
                        for (int l0=0;l0<m;l0+=blockDim.x){
                                int l=l0+threadIdx.x;
                                float x2=0,y2=0,z2=0;
                                if (l<m){
                                        x2=xyz2[i*m*3+l*3+0];
                                        y2=xyz2[i*m*3+l*3+1];
                                        z2=xyz2[i*m*3+l*3+2];
                                }
                                float sumr=0;
                                for (int k0=0;k0<n;k0+=Block){
                                        int kend=min(n,k0+Block)-k0;
                                        for (int k=threadIdx.x;k<kend;k+=blockDim.x){
                                                buf[k*4+0]=xyz1[i*n*3+k0*3+k*3+0];
                                                buf[k*4+1]=xyz1[i*n*3+k0*3+k*3+1];
                                                buf[k*4+2]=xyz1[i*n*3+k0*3+k*3+2];
                                                buf[k*4+3]=ratioL[k0+k];
                                        }
                                        __syncthreads();
                                        for (int k=0;k<kend;k++){
                                                float x1=buf[k*4+0];
                                                float y1=buf[k*4+1];
                                                float z1=buf[k*4+2];
                                                float w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*buf[k*4+3];
                                                sumr+=w;
                                        }
                                        __syncthreads();
                                }
                                if (l<m){
                                        sumr*=remainR[l];
                                        float consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
                                        ratioR[l]=consumption*remainR[l];
                                        remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
                                }
                        }
                        /*for (int l=threadIdx.x;l<m;l+=blockDim.x){
                                float x2=xyz2[i*m*3+l*3+0];
                                float y2=xyz2[i*m*3+l*3+1];
                                float z2=xyz2[i*m*3+l*3+2];
                                float sumr=0;
                                for (int k=0;k<n;k++){
                                        float x1=xyz1[i*n*3+k*3+0];
                                        float y1=xyz1[i*n*3+k*3+1];
                                        float z1=xyz1[i*n*3+k*3+2];
                                        float w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*ratioL[k];
                                        sumr+=w;
                                }
                                sumr*=remainR[l];
                                float consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
                                ratioR[l]=consumption*remainR[l];
                                remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
                        }*/
                        __syncthreads();
                        for (int k0=0;k0<n;k0+=blockDim.x){
                                int k=k0+threadIdx.x;
                                float x1=0,y1=0,z1=0;
                                if (k<n){
                                        x1=xyz1[i*n*3+k*3+0];
                                        y1=xyz1[i*n*3+k*3+1];
                                        z1=xyz1[i*n*3+k*3+2];
                                }
                                float suml=0;
                                for (int l0=0;l0<m;l0+=Block){
                                        int lend=min(m,l0+Block)-l0;
                                        for (int l=threadIdx.x;l<lend;l+=blockDim.x){
                                                buf[l*4+0]=xyz2[i*m*3+l0*3+l*3+0];
                                                buf[l*4+1]=xyz2[i*m*3+l0*3+l*3+1];
                                                buf[l*4+2]=xyz2[i*m*3+l0*3+l*3+2];
                                                buf[l*4+3]=ratioR[l0+l];
                                        }
                                        __syncthreads();
                                        float rl=ratioL[k];
                                        if (k<n){
                                                for (int l=0;l<lend;l++){
                                                        float x2=buf[l*4+0];
                                                        float y2=buf[l*4+1];
                                                        float z2=buf[l*4+2];
                                                        float w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*rl*buf[l*4+3];
                                                        match[i*n*m+(l0+l)*n+k]+=w;
                                                        suml+=w;
                                                }
                                        }
                                        __syncthreads();
                                }
                                if (k<n)
                                        remainL[k]=fmaxf(0.0f,remainL[k]-suml);
                        }
                        /*for (int k=threadIdx.x;k<n;k+=blockDim.x){
                                float x1=xyz1[i*n*3+k*3+0];
                                float y1=xyz1[i*n*3+k*3+1];
                                float z1=xyz1[i*n*3+k*3+2];
                                float suml=0;
                                for (int l=0;l<m;l++){
                                        float x2=xyz2[i*m*3+l*3+0];
                                        float y2=xyz2[i*m*3+l*3+1];
                                        float z2=xyz2[i*m*3+l*3+2];
                                        float w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*ratioL[k]*ratioR[l];
                                        match[i*n*m+l*n+k]+=w;
                                        suml+=w;
                                }
                                remainL[k]=fmaxf(0.0f,remainL[k]-suml);
                        }*/
                        __syncthreads();
                }
        }
}'''

matchcost_kernel = '''
extern "C"
__global__ void matchcost(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ out){
        __shared__ float allsum[512];
        const int Block=1024;
        __shared__ float buf[Block*3];
        for (int i=blockIdx.x;i<b;i+=gridDim.x){
                float subsum=0;
                for (int k0=0;k0<n;k0+=blockDim.x){
                        int k=k0+threadIdx.x;
                        float x1=0,y1=0,z1=0;
                        if (k<n){
                                x1=xyz1[i*n*3+k*3+0];
                                y1=xyz1[i*n*3+k*3+1];
                                z1=xyz1[i*n*3+k*3+2];
                        }
                        for (int l0=0;l0<m;l0+=Block){
                                int lend=min(m,l0+Block)-l0;
                                for (int l=threadIdx.x;l<lend*3;l+=blockDim.x)
                                        buf[l]=xyz2[i*m*3+l0*3+l];
                                __syncthreads();
                                if (k<n){
                                        for (int l=0;l<lend;l++){
                                                float x2=buf[l*3+0];
                                                float y2=buf[l*3+1];
                                                float z2=buf[l*3+2];
                                                float d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
                                                subsum+=d*match[i*n*m+(l0+l)*n+k];
                                        }
                                }
                                __syncthreads();
                        }
                }
                allsum[threadIdx.x]=subsum;
                for (int j=1;j<blockDim.x;j<<=1){
                        __syncthreads();
                        if ((threadIdx.x&j)==0 && threadIdx.x+j<blockDim.x){
                                allsum[threadIdx.x]+=allsum[threadIdx.x+j];
                        }
                }
                if (threadIdx.x==0)
                        out[i]=allsum[0];
                __syncthreads();
        }
}'''

matchcostgrad2_kernel = '''
extern "C"
__global__ void matchcostgrad2(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ grad2){
        __shared__ float sum_grad[256*3];
        for (int i=blockIdx.x;i<b;i+=gridDim.x){
                int kbeg=m*blockIdx.y/gridDim.y;
                int kend=m*(blockIdx.y+1)/gridDim.y;
                for (int k=kbeg;k<kend;k++){
                        float x2=xyz2[(i*m+k)*3+0];
                        float y2=xyz2[(i*m+k)*3+1];
                        float z2=xyz2[(i*m+k)*3+2];
                        float subsumx=0,subsumy=0,subsumz=0;
                        for (int j=threadIdx.x;j<n;j+=blockDim.x){
                                float x1=x2-xyz1[(i*n+j)*3+0];
                                float y1=y2-xyz1[(i*n+j)*3+1];
                                float z1=z2-xyz1[(i*n+j)*3+2];
                                float d=match[i*n*m+k*n+j]*rsqrtf(fmaxf(x1*x1+y1*y1+z1*z1,1e-20f));
                                subsumx+=x1*d;
                                subsumy+=y1*d;
                                subsumz+=z1*d;
                        }
                        sum_grad[threadIdx.x*3+0]=subsumx;
                        sum_grad[threadIdx.x*3+1]=subsumy;
                        sum_grad[threadIdx.x*3+2]=subsumz;
                        for (int j=1;j<blockDim.x;j<<=1){
                                __syncthreads();
                                int j1=threadIdx.x;
                                int j2=threadIdx.x+j;
                                if ((j1&j)==0 && j2<blockDim.x){
                                        sum_grad[j1*3+0]+=sum_grad[j2*3+0];
                                        sum_grad[j1*3+1]+=sum_grad[j2*3+1];
                                        sum_grad[j1*3+2]+=sum_grad[j2*3+2];
                                }
                        }
                        if (threadIdx.x==0){
                                grad2[(i*m+k)*3+0]=sum_grad[0];
                                grad2[(i*m+k)*3+1]=sum_grad[1];
                                grad2[(i*m+k)*3+2]=sum_grad[2];
                        }
                        __syncthreads();
                }
        }
}'''

matchcostgrad1_kernel = '''
extern "C"
__global__ void matchcostgrad1(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ grad1){
        for (int i=blockIdx.x;i<b;i+=gridDim.x){
                for (int l=threadIdx.x;l<n;l+=blockDim.x){
                        float x1=xyz1[i*n*3+l*3+0];
                        float y1=xyz1[i*n*3+l*3+1];
                        float z1=xyz1[i*n*3+l*3+2];
                        float dx=0,dy=0,dz=0;
                        for (int k=0;k<m;k++){
                                float x2=xyz2[i*m*3+k*3+0];
                                float y2=xyz2[i*m*3+k*3+1];
                                float z2=xyz2[i*m*3+k*3+2];
                                float d=match[i*n*m+k*n+l]*rsqrtf(fmaxf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2),1e-20f));
                                dx+=(x1-x2)*d;
                                dy+=(y1-y2)*d;
                                dz+=(z1-z2)*d;
                        }
                        grad1[i*n*3+l*3+0]=dx;
                        grad1[i*n*3+l*3+1]=dy;
                        grad1[i*n*3+l*3+2]=dz;
                }
        }
}
'''


# void approxmatchLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,float * match,float * temp){
#       approxmatch<<<32,512>>>(b,n,m,xyz1,xyz2,match,temp);
# }
