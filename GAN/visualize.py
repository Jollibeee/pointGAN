import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_pc(point):

    x = point[:, 0]
    y = point[:, 1]
    z = point[:, 2]


    fig = plt.figure()
    # fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')


    sc = ax.scatter(x, y, z, s=25)
    ax.view_init(elev=10, azim=240)

    plt.tight_layout()
    plt.show()

    return fig

def plot_point_cloud(points, filepath = '', step = 1):
    """
    Plot a point cloud using the given points.
    :param points: N x 3 point matrix
    :type points: numpy.ndarray
    :param filepath: path to file to save plot to; plot is shown if empty
    :type filepath: str
    :param step: take every step-th point only
    :type step: int
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    # xx = points[::step, 0]
    # zz = points[::step, 1]
    # yy = points[::step, 2]
    xx = points[::step, 0]
    zz = points[::step, 1]
    yy = points[::step, 2]

    ax.view_init(30,-30)

    ax.scatter(xx, yy, zz, c=zz, s=15)

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.view_init(-30, 60)
    # ax.view_init(-55, 90) #-60 90 #120 -90

    minbound = -.7
    maxbound = .7
    ax.auto_scale_xyz([minbound, maxbound], [minbound, maxbound], [minbound, maxbound])


    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
        plt.cla()  ##
    else:
        plt.show()
    plt.close(fig)

def plot_point_clouds(point_clouds, filepath = ''):
    assert len(point_clouds) > 0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    c = 0
    for points in point_clouds:
        xx = points[:, 0]
        yy = points[:, 1]
        zz = points[:, 2]

        ax.scatter(xx, yy, zz, c = 0)
        c = c + 1

    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
        plt.cla()   ##
    else:
        plt.show()
    plt.close(fig)

def plot_point_cloud_error(point_clouds, filepath = ''):
    assert len(point_clouds) == 2

    points_a = point_clouds[0]
    points_b = point_clouds[1]

    distances = np.zeros((points_a.shape[0], points_b.shape[0]))
    for n in range(points_a.shape[0]):
        points = np.repeat(points_a[n, :].reshape((1, 3)), points_b.shape[0], axis = 0)
        distances[n, :] = np.sum(np.square(points - points_b), axis = 1).transpose()

    min_indices = np.argmin(distances, axis = 1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for n in range(points_a.shape[0]):
        ax.plot(np.array([points_a[n, 0], points_b[min_indices[n], 0]]),
                np.array([points_a[n, 1], points_b[min_indices[n], 1]]),
                np.array([points_a[n, 2], points_b[min_indices[n], 2]]))

    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
    else:
        plt.show()


import argparse
from in_out import PartDataset
import torch
from torch.autograd import Variable


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default='Chair')
    parser.add_argument('--batchSize', type=int, default=5, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
    parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--num_points', type=int, default=2048, help='number of points')
    parser.add_argument('--feat_dim', type=int, default=128, help='number of feature dim')
    parser.add_argument('--k', type=int, default=40, help='number of neighbor')
    parser.add_argument('--outf', type=str, default='wgan', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--clamp_lower', type=float, default=-0.02)
    parser.add_argument('--clamp_upper', type=float, default=0.02)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

    opt = parser.parse_args()
    print(opt)

    # %%

    dataset = PartDataset(root='dataset/shapenetcore_partanno_segmentation_benchmark_v0', class_choice=opt.class_name,
                          classification=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,
                                             num_workers=int(opt.workers))

    test_dataset = PartDataset(root='dataset/shapenetcore_partanno_segmentation_benchmark_v0',
                               class_choice=opt.class_name, classification=True, train=False)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True,
                                                 num_workers=int(opt.workers))

    data_iter = iter(dataloader)
    data = data_iter.next()
    points, _ = data
    points = Variable(points)
    bs = points.size()[0]
    # points = points.transpose(2, 1)  # [batch, dim, points]
    # points = points.cuda()

    path = os.path.join(opt.outf, 'render')
    try:
        os.makedirs(path)
    except OSError:
        pass

    for n in range(points.shape[0]):
        file = 'points_' + str(n) + '.png'
        file = os.path.join(path, file)
        plot_point_cloud(points[n], file)
        # plot_point_cloud(points[n])
        # plot_pc(points[n])
        # print(points[n].size())