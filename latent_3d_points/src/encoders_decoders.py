'''
Created on February 4, 2017

@author: optas

'''

import tensorflow as tf
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d, conv_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout

# from . tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers

import os
import os.path as osp
import sys
BASE = os.path.dirname(os.path.abspath(os.path.dirname("file")))
sys.path.append(BASE)  # latent_3D

from src.tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers
from src import tf_util ##



# def ldgcnn_encoder(in_signal, n_filters=[64, 128, 128, 256, 128], filter_sizes=[1, 1], strides=[1, 1],
#                                         b_norm=True, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
#                                         symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
#                                         reuse=False, padding='same', verbose=True, closing=None, conv_op=conv_2d):
#     if verbose:
#         print ('Building Encoder')
#
#     n_layers = len(n_filters)
#
#     num_point = in_signal.get_shape()[1].value
#     end_points = {}
#     k = 20
#
#     if n_layers < 2:
#         raise ValueError('More than 1 layers are expected.')
#
#     adj_matrix = tf_util.pairwise_distance(in_signal)
#     nn_idx = tf_util.knn(adj_matrix, k=k)
#     in_signal = tf.expand_dims(in_signal, axis = -2)
#
#     edge_feature = tf_util.get_edge_feature(in_signal, nn_idx=nn_idx, k=k)
#
#     name = 'encoder_1'
#     scope_i = expand_scope_by_name(scope, name)
#     layer = conv_op(edge_feature, nb_filter=64, filter_size=filter_sizes, strides=strides,
#                     regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)
#
#     if verbose:
#         print (name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)
#
#     if b_norm:
#             name += '_bnorm'
#             scope_i = expand_scope_by_name(scope, name)
#             layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
#             if verbose:
#                 print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))
#
#     if non_linearity is not None:
#         layer = non_linearity(layer)
#
#     if dropout_prob is not None and dropout_prob > 0:
#         layer = dropout(layer, 1.0 - dropout_prob)
#
#     if verbose:
#         print (layer)
#         print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')
#
#     layer = tf.reduce_max(layer, axis=-2, keepdims=True)
#     net1 = layer
#
#     ###############################################################
#     adj_matrix = tf_util.pairwise_distance(layer)
#     nn_idx = tf_util.knn(adj_matrix, k=k)
#     layer = tf.concat([in_signal, net1], aixs=-1)
#     edge_feature = tf_util.get_edge_feature(layer, nn_idx=nn_idx, k=k)
#
#     name = 'encoder_2'
#     scope_i = expand_scope_by_name(scope, name)
#     layer = conv_op(edge_feature, nb_filter=128, ilter_size=filter_sizes, strides=strides,
#                     regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)
#
#     if verbose:
#         print(name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()), )
#
#     if b_norm:
#         name += '_bnorm'
#         scope_i = expand_scope_by_name(scope, name)
#         layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
#         if verbose:
#             print('bnorm params = ',
#                   np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))
#
#     if non_linearity is not None:
#         layer = non_linearity(layer)
#
#     if dropout_prob is not None and dropout_prob > 0:
#         layer = dropout(layer, 1.0 - dropout_prob)
#
#     if verbose:
#         print(layer)
#         print('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')
#
#     layer = tf.reduce_max(layer, axis=-2, keepdims=True)
#     net2 = layer
#
#
#     ###############################################################
#
#     adj_matrix = tf_util.pairwise_distance(layer)
#     nn_idx = tf_util.knn(adj_matrix, k=k)
#     layer = tf.concat([in_signal, net1, net2], aixs=-1)
#     edge_feature = tf_util.get_edge_feature(layer, nn_idx=nn_idx, k=k)
#
#     name = 'encoder_3'
#     scope_i = expand_scope_by_name(scope, name)
#     layer = conv_op(edge_feature, nb_filter=128, ilter_size=filter_sizes, strides=strides,
#                     regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i,
#                     padding=padding)
#
#     if verbose:
#         print(name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()), )
#
#     if b_norm:
#         name += '_bnorm'
#         scope_i = expand_scope_by_name(scope, name)
#         layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
#         if verbose:
#             print('bnorm params = ',
#                   np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))
#
#     if non_linearity is not None:
#         layer = non_linearity(layer)
#
#     if dropout_prob is not None and dropout_prob > 0:
#         layer = dropout(layer, 1.0 - dropout_prob)
#
#     if verbose:
#         print(layer)
#         print('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')
#
#     layer = tf.reduce_max(layer, axis=-2, keepdims=True)
#     net3 = layer
#
#     ###############################################################
#
#     adj_matrix = tf_util.pairwise_distance(layer)
#     nn_idx = tf_util.knn(adj_matrix, k=k)
#     layer = tf.concat([in_signal, net1, net2, net3], aixs=-1)
#     edge_feature = tf_util.get_edge_feature(layer, nn_idx=nn_idx, k=k)
#
#     name = 'encoder_4'
#     scope_i = expand_scope_by_name(scope, name)
#     layer = conv_op(edge_feature, nb_filter=256, ilter_size=filter_sizes, strides=strides,
#                     regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i,
#                     padding=padding)
#
#     if verbose:
#         print(name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()), )
#
#     if b_norm:
#         name += '_bnorm'
#         scope_i = expand_scope_by_name(scope, name)
#         layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
#         if verbose:
#             print('bnorm params = ',
#                   np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))
#
#     if non_linearity is not None:
#         layer = non_linearity(layer)
#
#     if dropout_prob is not None and dropout_prob > 0:
#         layer = dropout(layer, 1.0 - dropout_prob)
#
#     if verbose:
#         print(layer)
#         print('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')
#
#     layer = tf.reduce_max(layer, axis=-2, keepdims=True)
#     net4 = layer
#
#     ###############################################################
#
#     adj_matrix = tf_util.pairwise_distance(layer)
#     nn_idx = tf_util.knn(adj_matrix, k=k)
#     layer = tf.concat([in_signal, net1, net2, net3, net4], aixs=-1)
#     edge_feature = tf_util.get_edge_feature(layer, nn_idx=nn_idx, k=k)
#
#     name = 'encoder_5'
#     scope_i = expand_scope_by_name(scope, name)
#     layer = conv_op(edge_feature, nb_filter=128, ilter_size=filter_sizes, strides=strides,
#                     regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i,
#                     padding=padding)
#
#     if verbose:
#         print(name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()), )
#
#     if b_norm:
#         name += '_bnorm'
#         scope_i = expand_scope_by_name(scope, name)
#         layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
#         if verbose:
#             print('bnorm params = ',
#                   np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))
#
#     if non_linearity is not None:
#         layer = non_linearity(layer)
#
#     if dropout_prob is not None and dropout_prob > 0:
#         layer = dropout(layer, 1.0 - dropout_prob)
#
#     if verbose:
#         print(layer)
#         print('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')
#
#     layer = tf.reduce_max(layer, axis=-2, keepdims=True)
#     net5 = layer
#
#     ###############################################################
#
#     if closing is not None:
#         layer = closing(layer)
#         print (layer)

def ldgcnn_encoder(in_signal, n_filters=[64,128,128,256,128], filter_size=[1, 1], strides=[1, 1],
                   b_norm = True, verbose = True, conv_op = tf_util.conv2d):
    if verbose:
        print ('Building Encoder')

    n_layers = len(n_filters)

    num_point = in_signal.get_shape()[1].value
    end_points = {}
    k = 20

    adj_matrix = tf_util.pairwise_distance(in_signal)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    in_signal = tf.expand_dims(in_signal, axis=-2)

    edge_feature = tf_util.get_edge_feature(in_signal, nn_idx=nn_idx, k=k)

    net = conv_op(edge_feature, 64, filter_size=filter_size, padding='VALID', stride=strides, bn=True, )

def ldgcnn_decoder(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    if verbose:
        print ('Building Decoder')

    batch_size = latent_signal.get_shape()[0].value
    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in range(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if verbose:
            print (name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print (layer)
            print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    if verbose:
        print (name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
        if verbose:
            print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

    if verbose:
        print (layer)
        print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    return layer





def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=True, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                        reuse=False, padding='same', verbose=False, closing=None, conv_op=conv_1d):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''

    if verbose:
        print ('Building Encoder')
    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    for i in range(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)
        print(name, layer)

        if verbose:
            print (name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            print(name, layer)
            if verbose:
                print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

        if non_linearity is not None:
            layer = non_linearity(layer)
            print(name, layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])
            print(name, layer)

        if verbose:
            print(layer)
            print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        if verbose:
            print (layer)

    if closing is not None:
        layer = closing(layer)
        print (layer)

    return layer


def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    if verbose:
        print ('Building Decoder')

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in range(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
        print(name, layer)
        if verbose:
            print (name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            print(name, layer)
            if verbose:
                print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print (layer)
            print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    print(name, layer)
    if verbose:
        print (name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
        print(name, layer)
        if verbose:
            print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

    if verbose:
        # print (layer)
        print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    return layer


def decoder_with_convs_only(in_signal, n_filters, filter_sizes, strides, padding='same', b_norm=True, non_linearity=tf.nn.relu,
                            conv_op=conv_1d, regularizer=None, weight_decay=0.001, dropout_prob=None, upsample_sizes=None,
                            b_norm_finish=False, scope=None, reuse=False, verbose=False):

    if verbose:
        print ('Building Decoder')

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    for i in range(n_layers):
        if i == 0:
            layer = in_signal

        name = 'decoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i],
                        strides=strides[i], padding=padding, regularizer=regularizer, weight_decay=weight_decay,
                        name=name, reuse=reuse, scope=scope_i)

        if verbose:
            print (name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

        if (b_norm and i < n_layers - 1) or (i == n_layers - 1 and b_norm_finish):
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

        if non_linearity is not None and i < n_layers - 1:  # Last layer doesn't have a non-linearity.
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if upsample_sizes is not None and upsample_sizes[i] is not None:
            layer = tf.tile(layer, multiples=[1, upsample_sizes[i], 1])

        if verbose:
            print (layer)
            print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    return layer
