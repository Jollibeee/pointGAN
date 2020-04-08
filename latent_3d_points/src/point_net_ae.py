'''
Created on January 26, 2017

@author: optas
'''

import time
import tensorflow as tf
import os.path as osp

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

# from . in_out import create_dir
# from . autoencoder import AutoEncoder
# from . general_utils import apply_augmentations

import os
import os.path as osp
import sys

BASE = os.path.dirname(os.path.abspath(os.path.dirname("file")))
sys.path.append(BASE)  # latent_3D

from src.general_utils import rand_rotation_matrix
from src.in_out import create_dir
from src.autoencoder import AutoEncoder, LDGCNN_AutoEncoder
from src.general_utils import apply_augmentations

# try:
#     from external.structural_losses.tf_approxmatch import approx_match, match_cost
#     from external.structural_losses.tf_nndistance import nn_distance
# except:
#     print('External Losses (Chamfer-EMD) cannot be loaded. Please install them first.')

try:
    from external.structural_losses.tf_nndistance import nn_distance
except:
    print('tf_nn_distance cannot be loaded. Please install / check path.')

try:
    from external.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print('tf_approxmatch cannot be loaded. Please check path.')

# try:
#     from .. external.structural_losses.tf_nndistance import nn_distance
# except:
#     print('tf_nn_distance.so cannot be loaded. Please install / check path.')
#
# try:
#     from .. external.structural_losses.tf_approxmatch import approx_match, match_cost
# except:
#     print('tf_approxmatch cannot be loaded. Please check path.')
#
# try:
#     from .. external.structural_losses.tf_multiemd import multi_emd, multi_emd_cost
# except:
#     print('tf_multiemd cannot be loaded. Please check path.')

class PointNetAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c

        AutoEncoder.__init__(self, name, graph, configuration)

        with tf.variable_scope(name):
            self.z = c.encoder(self.x, **c.encoder_args)
            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)
            
            if c.exists_and_is_not_none('close_with_tanh'):
                layer = tf.nn.tanh(layer)

            self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])
            
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            self._create_loss()
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    # def euclidean_dist_sq(self, A):
    #     row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    #     row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
    #     return row_norms_A - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(row_norms_A)
    #
    # def shift_points(self, pnts):
    #     pnts_shape = tf.shape(pnts)
    #     pnts_allb = tf.reshape(pnts, [pnts_shape[0]*pnts_shape[1],pnts_shape[2]])
    #
    #     dist = self.euclidean_dist_sq(pnts_allb)
    #     # get all elements smaller than window=2.0 (indices)
    #     # this is a nxn matrix where each row does indicate the values to be averaged
    #     idx = tf.less(dist, 4.0)
    #
    #     # calculate mean of those elements
    #     fun = lambda x : tf.reduce_mean(tf.boolean_mask(pnts_allb,x), 0)
    #     mean = tf.reshape(tf.map_fn (fun, idx, dtype=tf.float32), shape=pnts_shape)
    #     opnts = tf.subtract(pnts,mean)
    #     return opnts
    #
    # def shift_points_batch(self, pnts):
    #     dist = self.euclidean_dist_sq(pnts)
    #
    #     # get all elements smaller than window=2.0 (indices)
    #     # this is a nxn matrix where each row does indicate the values to be averaged
    #     idx = tf.less(dist, 4.0)
    #
    #     # calculate mean of those elements
    #     fun = lambda x : tf.reduce_mean(tf.boolean_mask(pnts,x), 0)
    #     mean = tf.map_fn (fun, idx, dtype=tf.float32)
    #     opnts = tf.subtract(pnts,mean)
    #     return opnts
    #
    # def shift_points_all(self, pnts):
    #     opnts = tf.map_fn (self.shift_points_batch,pnts)
    #     return opnts

    def _create_loss(self):
        c = self.configuration
        #
        # if c.exists_and_is_not_none('lagrange'):
        #     lagrange = c.lagrange
        # else:
        #     lagrange = 0.001
        # print('Lagrange loss set to', lagrange)

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
            tf.summary.scalar('chamfer', self.loss)

        elif c.loss == 'emd':
            # match = tf.constant(1.0)
            # self.loss = tf.constant(1.0)
            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))
            tf.summary.scalar('emd', self.loss)
        # elif c.loss == 'lagrange_emd':
        #     match1 = tf.constant(1.0)
        #     match2 = tf.constant(1.0)
        #     self.loss = tf.constant(1.0)
        #     x_reconstr_shift = self.shift_points(self.x_reconstr)
        #     gt_shift = self.shift_points(self.gt)
        #     match1 = approx_match(x_reconstr_shift, gt_shift)
        #     match2 = approx_match(self.x_reconstr, self.gt)
        #     self.loss = tf.reduce_mean(
        #             lagrange * match_cost(x_reconstr_shift, gt_shift, match1) +
        #             (1.0 - lagrange) * match_cost(self.x_reconstr, self.gt, match2))
        # elif c.loss == 'shift_emd':
        #     match = tf.constant(1.0)
        #     self.loss = tf.constant(1.0)
        #     x_reconstr_shift = self.shift_points(self.x_reconstr)
        #     gt_shift = self.shift_points(self.gt)
        #     match = approx_match(x_reconstr_shift, gt_shift)
        #     self.loss = tf.reduce_mean(match_cost(x_reconstr_shift, gt_shift, match))
        # elif c.loss == 'match_shift_emd':
        #     match = tf.constant(1.0)
        #     self.loss = tf.constant(1.0)
        #     x_reconstr_shift = self.shift_points(self.x_reconstr)
        #     gt_shift = self.shift_points(self.gt)
        #     # calculate match using translation invariance
        #     match = approx_match(x_reconstr_shift, gt_shift)
        #     # calculate loss given match but using original point clouds
        #     self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))
        # elif c.loss == 'batch_shift_emd':
        #     match = tf.constant(1.0)
        #     self.loss = tf.constant(1.0)
        #     x_reconstr_shift = self.shift_points_all(self.x_reconstr)
        #     gt_shift = self.shift_points_all(self.gt)
        #     match = approx_match(x_reconstr_shift, gt_shift)
        #     self.loss = tf.reduce_mean(match_cost(x_reconstr_shift, gt_shift, match))
        # elif c.loss == 'multi_emd':
        #     match = tf.constant(1.0)
        #     offset1 = tf.constant(1.0)
        #     offset2 = tf.constant(1.0)
        #     self.loss = tf.constant(1.0)
        #     match,offset1,offset2 = multi_emd(self.x_reconstr, self.gt)
        #     self.loss = tf.reduce_mean(multi_emd_cost(self.x_reconstr, self.gt, match, offset1, offset2))

        # Add regularization losses to self.loss
        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.loss += (w_reg_alpha * rl)

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def _single_epoch_train(self, train_data, configuration, only_fw=False):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        if only_fw:
            fit = self.reconstruct
        else:
            fit = self.partial_fit

        # Loop over all batches
        for _ in range(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if batch_i is None:  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                batch_i, _, _ = train_data.next_batch(batch_size)

            batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.

            if self.is_denoising:
                _, loss = fit(batch_i, original_data)
            else:
                _, loss = fit(batch_i)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time

        if configuration.loss == 'emd':
            epoch_loss /= len(train_data.point_clouds[0])
        # elif configuration.loss == 'lagrange_emd':
        #     epoch_loss /= len(train_data.point_clouds[0])
        # elif configuration.loss == 'shift_emd':
        #     epoch_loss /= len(train_data.point_clouds[0])
        # elif configuration.loss == 'batch_shift_emd':
        #     epoch_loss /= len(train_data.point_clouds[0])
        # elif configuration.loss == 'match_shift_emd':
        #     epoch_loss /= len(train_data.point_clouds[0])
        # elif configuration.loss == 'multi_emd':
        #     epoch_loss /= len(train_data.point_clouds[0])
        
        return epoch_loss, duration

    # Function that calculates the gradient given two point clouds. If there are not points generated yet, the
    # point clouds are assumed to be the same (gradient 0).
    #
    # @param in_points   Input point cloud
    # @param gt_points   Generated point cloud
    #
    def gradient_of_input_wrt_loss(self, in_points, gt_points=None):
        if gt_points is None:
            gt_points = in_points
        return self.sess.run(tf.gradients(self.loss, self.x), feed_dict={self.x: in_points, self.gt: gt_points})


class LDGCNNAutoEncoder(LDGCNN_AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c

        LDGCNN_AutoEncoder.__init__(self, name, graph, configuration)

        with tf.variable_scope(name):
            self.z = c.encoder(self.x, **c.encoder_args)
            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)

            if c.exists_and_is_not_none('close_with_tanh'):
                layer = tf.nn.tanh(layer)

            self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            self._create_loss()
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    # def euclidean_dist_sq(self, A):
    #     row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    #     row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
    #     return row_norms_A - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(row_norms_A)
    #
    # def shift_points(self, pnts):
    #     pnts_shape = tf.shape(pnts)
    #     pnts_allb = tf.reshape(pnts, [pnts_shape[0]*pnts_shape[1],pnts_shape[2]])
    #
    #     dist = self.euclidean_dist_sq(pnts_allb)
    #     # get all elements smaller than window=2.0 (indices)
    #     # this is a nxn matrix where each row does indicate the values to be averaged
    #     idx = tf.less(dist, 4.0)
    #
    #     # calculate mean of those elements
    #     fun = lambda x : tf.reduce_mean(tf.boolean_mask(pnts_allb,x), 0)
    #     mean = tf.reshape(tf.map_fn (fun, idx, dtype=tf.float32), shape=pnts_shape)
    #     opnts = tf.subtract(pnts,mean)
    #     return opnts
    #
    # def shift_points_batch(self, pnts):
    #     dist = self.euclidean_dist_sq(pnts)
    #
    #     # get all elements smaller than window=2.0 (indices)
    #     # this is a nxn matrix where each row does indicate the values to be averaged
    #     idx = tf.less(dist, 4.0)
    #
    #     # calculate mean of those elements
    #     fun = lambda x : tf.reduce_mean(tf.boolean_mask(pnts,x), 0)
    #     mean = tf.map_fn (fun, idx, dtype=tf.float32)
    #     opnts = tf.subtract(pnts,mean)
    #     return opnts
    #
    # def shift_points_all(self, pnts):
    #     opnts = tf.map_fn (self.shift_points_batch,pnts)
    #     return opnts

    def _create_loss(self):
        c = self.configuration
        #
        # if c.exists_and_is_not_none('lagrange'):
        #     lagrange = c.lagrange
        # else:
        #     lagrange = 0.001
        # print('Lagrange loss set to', lagrange)

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
            tf.summary.scalar('chamfer', self.loss)

        elif c.loss == 'emd':
            # match = tf.constant(1.0)
            # self.loss = tf.constant(1.0)
            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))
            tf.summary.scalar('emd', self.loss)
        # elif c.loss == 'lagrange_emd':
        #     match1 = tf.constant(1.0)
        #     match2 = tf.constant(1.0)
        #     self.loss = tf.constant(1.0)
        #     x_reconstr_shift = self.shift_points(self.x_reconstr)
        #     gt_shift = self.shift_points(self.gt)
        #     match1 = approx_match(x_reconstr_shift, gt_shift)
        #     match2 = approx_match(self.x_reconstr, self.gt)
        #     self.loss = tf.reduce_mean(
        #             lagrange * match_cost(x_reconstr_shift, gt_shift, match1) +
        #             (1.0 - lagrange) * match_cost(self.x_reconstr, self.gt, match2))
        # elif c.loss == 'shift_emd':
        #     match = tf.constant(1.0)
        #     self.loss = tf.constant(1.0)
        #     x_reconstr_shift = self.shift_points(self.x_reconstr)
        #     gt_shift = self.shift_points(self.gt)
        #     match = approx_match(x_reconstr_shift, gt_shift)
        #     self.loss = tf.reduce_mean(match_cost(x_reconstr_shift, gt_shift, match))
        # elif c.loss == 'match_shift_emd':
        #     match = tf.constant(1.0)
        #     self.loss = tf.constant(1.0)
        #     x_reconstr_shift = self.shift_points(self.x_reconstr)
        #     gt_shift = self.shift_points(self.gt)
        #     # calculate match using translation invariance
        #     match = approx_match(x_reconstr_shift, gt_shift)
        #     # calculate loss given match but using original point clouds
        #     self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))
        # elif c.loss == 'batch_shift_emd':
        #     match = tf.constant(1.0)
        #     self.loss = tf.constant(1.0)
        #     x_reconstr_shift = self.shift_points_all(self.x_reconstr)
        #     gt_shift = self.shift_points_all(self.gt)
        #     match = approx_match(x_reconstr_shift, gt_shift)
        #     self.loss = tf.reduce_mean(match_cost(x_reconstr_shift, gt_shift, match))
        # elif c.loss == 'multi_emd':
        #     match = tf.constant(1.0)
        #     offset1 = tf.constant(1.0)
        #     offset2 = tf.constant(1.0)
        #     self.loss = tf.constant(1.0)
        #     match,offset1,offset2 = multi_emd(self.x_reconstr, self.gt)
        #     self.loss = tf.reduce_mean(multi_emd_cost(self.x_reconstr, self.gt, match, offset1, offset2))

        # Add regularization losses to self.loss
        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.loss += (w_reg_alpha * rl)

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5,
                                                 staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def _single_epoch_train(self, train_data, configuration, only_fw=False):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        if only_fw:
            fit = self.reconstruct
        else:
            fit = self.partial_fit

        # Loop over all batches
        for _ in range(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if batch_i is None:  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                batch_i, _, _ = train_data.next_batch(batch_size)

            batch_i = apply_augmentations(batch_i, configuration)  # This is a new copy of the batch.

            if self.is_denoising:
                _, loss = fit(batch_i, original_data)
            else:
                _, loss = fit(batch_i)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time

        if configuration.loss == 'emd':
            epoch_loss /= len(train_data.point_clouds[0])
        # elif configuration.loss == 'lagrange_emd':
        #     epoch_loss /= len(train_data.point_clouds[0])
        # elif configuration.loss == 'shift_emd':
        #     epoch_loss /= len(train_data.point_clouds[0])
        # elif configuration.loss == 'batch_shift_emd':
        #     epoch_loss /= len(train_data.point_clouds[0])
        # elif configuration.loss == 'match_shift_emd':
        #     epoch_loss /= len(train_data.point_clouds[0])
        # elif configuration.loss == 'multi_emd':
        #     epoch_loss /= len(train_data.point_clouds[0])

        return epoch_loss, duration

    # Function that calculates the gradient given two point clouds. If there are not points generated yet, the
    # point clouds are assumed to be the same (gradient 0).
    #
    # @param in_points   Input point cloud
    # @param gt_points   Generated point cloud
    #
    def gradient_of_input_wrt_loss(self, in_points, gt_points=None):
        if gt_points is None:
            gt_points = in_points
        return self.sess.run(tf.gradients(self.loss, self.x), feed_dict={self.x: in_points, self.gt: gt_points})