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
from gan import GAN
from src.general_utils import *
from PIL import Image

import scipy.io as sio

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--class_name', default='', help='Shapenet class')
parser.add_argument('--render_dir', default='', help='Renders directory')
parser.add_argument('--save_dir', default='', help='Trained model directory')
param = parser.parse_args()


# import config
config = Config()
config.render_dir = param.render_dir
config.save_dir = param.save_dir

#class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
class_name = param.class_name

model = GAN(config)
model.do_variables_init()
model.restore_model(config.save_dir+'model.ckpt')

# import data
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(config.top_in_dir , syn_id)
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)


# testing
for test_no in range(config.N_test):
	data = all_pc_data.next_batch(config.batch_size)[0]
	noise = np.random.normal(size=[config.batch_size, config.z_size], scale=0.2)
	pc_gen = model.generate(noise)

	sio.savemat('%srender.mat' % (config.render_dir,),{'X_hat':pc_gen})
	sio.savemat('%snoise.mat' % (config.render_dir,), {'Z': noise})
	sio.savemat('%sref.mat' % (config.render_dir,), {'X': data})
	np.save('%snoise' % (config.render_dir), noise)
	np.save('%sref' % (config.render_dir), data)
