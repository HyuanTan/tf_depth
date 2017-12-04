#!/usr/bin/python

import tensorflow as tf
import os
import time
import cv2
import numpy as np

import skimage
import skimage.io

from nets.sq_net import SQSegNet
from nets.erf_net import ERFSegNet
from data.data_loader import instance_label_generator
import data.image_process as image_process
from config import cfg
from utils import log_helper



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('ckpt_path', '', """ path to restore model ckpt """)
tf.app.flags.DEFINE_string('vname', '', """ variable name """)
tf.app.flags.DEFINE_string('tname', '', """ tensor name """)
tf.app.flags.DEFINE_string('output_node', '', """ output node names splitted by ',' """)
#tf.app.flags.DEFINE_string('mask_path', '', """ path to validate label image """)



def main(args):
    with tf.Graph().as_default() as graph:
        saver = tf.train.import_meta_graph(FLAGS.ckpt_path + '.meta')

        #print('======================================================')
        #print([x.name for x in tf.global_variables()])
        #all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        all_vars = tf.global_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        default_graph_def = tf.get_default_graph().as_graph_def()
        sub_graph_def = tf.graph_util.extract_sub_graph(default_graph_def, ['output/prob'])
        all_inf_tensors = sub_graph_def.node
        #all_tensors = default_graph_def.node
        print('======================================================')
        print('all variables:')
        print('\n'.join([x.op.name for x in all_vars]))
        print('======================================================')
        print('\n'.join([x.name for x in all_vars]))
        print('======================================================')
        print(all_vars)
        print('======================================================')
        print('all inference tensors:')
        print('\n'.join([x.name for x in all_inf_tensors]))
        print('======================================================')
        #print('all tensors:')
        #print('\n'.join([x.name for x in all_tensors]))
        #print('======================================================')
        print('all update ops:')
        print('\n'.join([x.name for x in update_ops]))
        print('======================================================')

        #print(all_vars)
        #print('======================================================')
        with tf.Session() as sess:
            # restore model
            print("retoring")
            saver.restore(sess, FLAGS.ckpt_path)
            for var in all_vars:
                if FLAGS.vname:
                    if var.name == FLAGS.vname:
                        _var = var.eval(sess)
                        print(var.name)
                        print(_var)
                        print(_var.shape)
                        print(_var.max())
                        print(_var.min())
                else:
                    if 'Adam' in var.name:
                        continue
                    #var_shape = tf.shape(var)
                    _var = var.eval(sess)
                    print(var.name)
                    print(_var)
                    print(_var.shape)
                    print(_var.dtype)
                    print(_var.max())
                    print(_var.min())
                    #print(var_shape.eval(session=sess))
                    print('======================================================')


if __name__ == '__main__':
    tf.app.run()
