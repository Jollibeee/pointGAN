#!/home/valsesia/tensorflow-python2.7/bin/python
import os
import os.path as osp
import numpy as np
import shutil
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

BASE = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(BASE)

from config import Config
from src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder
from metric.evaluation_metrics import minimum_mathing_distance, jsd_between_point_cloud_sets, coverage
from gan import GAN
from src.general_utils import *
from PIL import Image

import scipy.io as sio

code = 'gconv_up_aggr'
class_name = 'chair'
render_dir = osp.join(BASE, 'Results', code, class_name, 'renders') + '/'
log_dir = osp.join(BASE, 'log_dir', code, class_name) + '/'
save_dir = osp.join(BASE, 'Results', code, class_name, 'saved_models') + '/'
top_in_dir = 'data/shape_net_core_uniform_samples_2048/'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--class_name', default=class_name, help='Shapenet class')
parser.add_argument('--render_dir', default=render_dir, help='Renders directory')
parser.add_argument('--save_dir', default=save_dir, help='Trained model directory')

param = parser.parse_args()



# import config
config = Config()
config.render_dir = param.render_dir
config.save_dir = param.save_dir

#class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
class_name = param.class_name

# model = GAN(config)
# model.do_variables_init()
# model.restore_model(config.save_dir+'model.ckpt')

gt = osp.join(BASE, param.render_dir, 'ref_1000.npy')
pd = osp.join(BASE, param.render_dir, 'render_1000.npy')

gt_data = np.load(gt)
n_pcs, n_pts, dim = gt_data.shape

pd_data = np.load(pd)

if dim == 2:
    gt_data = np.concatenate((gt_data, np.zeros((n_pcs,n_pts, 1), dtype=gt_data.dtype)),3)

batch_size = 100  # Find appropriate number that fits in GPU.
normalize = True  # Matched distances are divided by the number of

#
# JSD = 0
# MMD_CD = 0
# MMD_EMD = 0
# COV_CD = 0
# COV_EMD = 0

# for i in range(n_pcs):
#     # gt_seq = np.expand_dims(gt_data[i], axis=0)
#     # pd_seq = np.expand_dims(pd_data[i], axis=0)
#
#     jsd = jsd_between_point_cloud_sets(gt_data, pd_data, resolution=28)
#     mmd_cd, matched_dists = minimum_mathing_distance(gt_data, pd_data, batch_size, normalize=normalize,
#                                                      use_EMD=False)
#     mmd_emd, matched_dists = minimum_mathing_distance(gt_data, pd_data, batch_size, normalize=normalize,
#                                                       use_EMD=True)
#     cov_cd, matched_ids = coverage(gt_data, pd_data, batch_size, normalize=normalize, use_EMD=False)
#     cov_emd, matched_ids = coverage(gt_data, pd_data, batch_size, normalize=normalize, use_EMD=True)

jsd = jsd_between_point_cloud_sets(gt_data, pd_data, resolution=28)
mmd_cd, matched_dists = minimum_mathing_distance(gt_data, pd_data, batch_size, normalize=normalize, use_EMD=False)
mmd_emd, matched_dists = minimum_mathing_distance(gt_data, pd_data, batch_size, normalize=normalize, use_EMD=True)
cov_cd, matched_ids = coverage(gt_data, pd_data, batch_size, normalize=normalize, use_EMD=False)
cov_emd, matched_ids = coverage(gt_data, pd_data, batch_size, normalize=normalize, use_EMD=True)
print ('jsd :', jsd, '\nmmd_cd :', mmd_cd, '\nmmd_emd :', mmd_emd, '\ncov_cd :', cov_cd, '\ncov_emd :', cov_emd)

with open(config.save_dir + 'eval_metric', "w") as text_file:
    text_file.write('jsd = %f\n' %(jsd))
    text_file.write('mmd_cd = %f\n'%(mmd_cd))
    text_file.write('mmd_emd = %f\n' %(mmd_emd))
    text_file.write('cov_cd = %f\n' %(cov_cd))
    text_file.write('cov_emd = %f\n' %(cov_emd))

#     JSD += jsd
#     MMD_CD += mmd_cd
#     MMD_EMD += mmd_emd
#     COV_CD += cov_cd
#     COV_EMD += cov_emd
#
# JSD /= float(n_pcs)
# MMD_CD /= float(n_pcs)

# n_ref = 100 # size of ref_pcs.
# n_sam = 100 # size of sample_pcs.
# all_ids = np.arange(gt_data.shape[0])
# ref_ids = np.random.choice(all_ids, n_ref, replace=False)
# sam_ids = np.random.choice(all_ids, n_sam, replace=False)

#
# try:
#     from metric.structural_loss.tf_nndistance import nn_distance
# except:
#     print('nndistance cannot be loaded. Please install them first.')
#
# try:
#     from metric.structural_loss.tf_approxmatch import approx_match, match_cost
# except:
#     print('approxmatch cannot be loaded. Please install them first.')
#
# from metric.evaluation_metrics import minimum_mathing_distance, jsd_between_point_cloud_sets, coverage
#
#
#
#
# # import data
# syn_id = snc_category_to_synth_id()[class_name]
# class_dir = osp.join(config.top_in_dir , syn_id)
# all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)
# #
# # def placeholder_inputs(batch_size, signal_size):
# # 	pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, signal_size))
# # 	labels_pl = tf.placeholder(tf.float32, shape=(batch_size, signal_size))
# # 	return pointclouds_pl, labels_pl
# #
# #
# # pointclouds_pl, labels_pl = placeholder_inputs(config.batch_size, config.signal_size)
#
#
#
# # testing
# for test_no in range(config.N_test):
# 	data = all_pc_data.next_batch(config.batch_size)[0]
# 	noise = np.random.normal(size=[config.batch_size, config.z_size], scale=0.2)
# 	ref_pc, pc_gen = model.eval(data, noise)
#
#
# 	sio.savemat('%ssample.mat' % (config.render_dir,), {'X_hat': pc_gen})
# 	sio.savemat('%slatent.mat' % (config.render_dir,), {'Z': noise})
# 	sio.savemat('%sref.mat' % (config.render_dir,), {'X': ref_pc})
# 	np.save('%ssample' % (config.render_dir), pc_gen)
# 	np.save('%slatent' % (config.render_dir), noise)
# 	np.save('%sref' % (config.render_dir), ref_pc)
#
#
# # https://github.com/laughtervv/3DN/blob/master/shapenet/3D/test.py
# # https://github.com/hehefan/PointRNN/blob/master/cal-cd-emd.py
# # https://github.com/liruihui/PU-GAN/blob/master/evaluate.py
# # https://github.com/seowok/TreeGAN/blob/584b772548e4f43b8d0d991464b50ff9b78b2ebb/evaluation/FPD.py#L179