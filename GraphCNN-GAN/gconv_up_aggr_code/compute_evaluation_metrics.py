#!/home/valsesia/tensorflow-python2.7/bin/python
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  ## v2.0.0


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os.path as osp
import numpy as np
import shutil
import sys

BASE = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(BASE)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from config import Config
from src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder
# from gan import GAN
from src.general_utils import *
from PIL import Image

from metric.evaluation_metrics import minimum_mathing_distance, jsd_between_point_cloud_sets, coverage

import scipy.io as sio

from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--class_name', default='', help='Shapenet class')
parser.add_argument('--start_iter', type=int, default=1, help='Start iteration (ex: 10001)')
parser.add_argument('--render_dir', default='', help='Renders directory')
parser.add_argument('--log_dir', default='', help='Tensorboard log directory')
parser.add_argument('--save_dir', default='', help='Trained model directory')
param = parser.parse_args()


# import config
config = Config()
config.render_dir = param.render_dir
config.log_dir = param.log_dir
config.save_dir = param.save_dir


#class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
class_name = param.class_name

# import data
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(config.top_in_dir , syn_id)
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)


n_ref = 100 # size of ref_pcs.
n_sam = 150 # size of sample_pcs.
all_ids = np.arange(all_pc_data.num_examples)
ref_ids = np.random.choice(all_ids, n_ref, replace=False)
sam_ids = np.random.choice(all_ids, n_sam, replace=False)
ref_pcs = all_pc_data.point_clouds[ref_ids]
sample_pcs = all_pc_data.point_clouds[sam_ids]

print(ref_pcs.shape)
print(sample_pcs.shape)

# ae_loss = 'chamfer'  # Which distance to use for the matchings.
#
# if ae_loss == 'emd':
# 	use_EMD = True
# else:
# 	use_EMD = False  # Will use Chamfer instead.

batch_size = 100  # Find appropriate number that fits in GPU.
normalize = True  # Matched distances are divided by the number of
# points of thepoint-clouds.

# mmd, matched_dists = minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, normalize=normalize, use_EMD=use_EMD)
# cov, matched_ids = coverage(sample_pcs, ref_pcs, batch_size, normalize=normalize, use_EMD=use_EMD)
# jsd = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28)
#
# print (mmd, cov, jsd)

jsd = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28)
mmd_cd, matched_dists = minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, normalize=normalize, use_EMD=False)
mmd_emd, matched_dists = minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, normalize=normalize, use_EMD=True)
cov_cd, matched_ids = coverage(sample_pcs, ref_pcs, batch_size, normalize=normalize, use_EMD=False)
cov_emd, matched_ids = coverage(sample_pcs, ref_pcs, batch_size, normalize=normalize, use_EMD=True)

print ('jsd :', jsd, '\nmmd_cd :', mmd_cd, '\nmmd_emd :', mmd_emd, '\ncov_cd :', cov_cd, '\ncov_emd :', cov_emd)

with open(config.save_dir + 'eval_metric_', "w") as text_file:
    text_file.write('jsd = %f\n' %(jsd))
    text_file.write('mmd_cd = %f\n'%(mmd_cd))
    text_file.write('mmd_emd = %f\n' %(mmd_emd))
    text_file.write('cov_cd = %f\n' %(cov_cd))
    text_file.write('cov_emd = %f\n' %(cov_emd))



# print (coverage.__doc__)
# print (minimum_mathing_distance.__doc__)
# print (jsd_between_point_cloud_sets.__doc__)