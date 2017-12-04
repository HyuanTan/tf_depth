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
from config import cfg
from utils.config_utils import cfg_from_file, print_config
from utils import log_helper



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'sq', """ which model to use """)
tf.app.flags.DEFINE_string('ckpt_path', '', """ path to restore model ckpt """)
tf.app.flags.DEFINE_string('cfg_file', '', """ experimental config file """)
tf.app.flags.DEFINE_string('tname', '', """ tensor to inspect """)
tf.app.flags.DEFINE_string('input_image', '', """ path to sample image """)
tf.app.flags.DEFINE_string('output_dir', '', """ dir to output """)
tf.app.flags.DEFINE_boolean('is_training', False, """ input is_training """)
tf.app.flags.DEFINE_boolean('restore_avg', True, """ whether to restore moving avg version """)


output_layers = ['l1_downsampler/l1_downsampler_bn/Relu',
                 'l2_downsampler/l2_downsampler_bn/Relu',
                 'l3_nonbt1d/Relu',
                 'l5_nonbt1d/Relu',
                 'l6_nonbt1d/Relu',
                 'l7_nonbt1d/Relu',
                 'l8_downsampler/l8_downsampler_bn/Relu',
                 'l9_nonbt1d/Relu',
                 'l10_nonbt1d/Relu',
                 'l11_nonbt1d/Relu',
                 'l12_nonbt1d/Relu',
                 'l13_nonbt1d/Relu',
                 'l14_nonbt1d/Relu',
                 'l15_nonbt1d/Relu',
                 'l16_nonbt1d/Relu',
                 'l17_deconv/Relu',
                 'l18_nonbt1d/Relu',
                 'l19_nonbt1d/Relu',
                 'l20_deconv/Relu',
                 'l21_nonbt1d/Relu',
                 'l22_nonbt1d/Relu',
                 'l23_classifier/l23_classifier_bn/batchnorm/add_1'
                ]




def main(args):
    if FLAGS.cfg_file:
        print('loading config setting')
        cfg_from_file(FLAGS.cfg_file, cfg)
    cfg.BATCH_SIZE = 1
    print_config(cfg)

    if FLAGS.output_dir and not os.path.isdir(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    logger = log_helper.get_logger()

    with tf.Graph().as_default() as graph:

        if FLAGS.model == 'sq':
            model = SQSegNet(cfg, logger)
        elif FLAGS.model == 'erf':
            model = ERFSegNet(cfg, logger)

        if FLAGS.restore_avg:
            # get moving avg
            variable_averages = tf.train.ExponentialMovingAverage(cfg.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
        else:
            saver = tf.train.Saver(model.all_variables)

        graph_def = graph.as_graph_def()
        sub_graph_def = tf.graph_util.extract_sub_graph(graph_def, ['output/prob'])
        all_inference_nodes = sub_graph_def.node
        all_nodes = graph_def.node

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        print('======================================================')
        print('all nodes:')
        print('\n'.join([x.name for x in all_nodes]))
        print('======================================================')
        print('all inference nodes:')
        print('\n'.join([x.name for x in all_inference_nodes]))
        print('======================================================')
        print('all update ops:')
        print('\n'.join([x.name for x in update_ops]))
        print('======================================================')

        with tf.Session() as sess:
            print("retoring")
            saver.restore(sess, FLAGS.ckpt_path)

            img = np.array(skimage.io.imread(FLAGS.input_image), np.float32)
            #imageValue = tf.read_file(FLAGS.input_image)
            #image_bytes = tf.image.decode_png(imageValue)
            #image_reshape = tf.reshape(image_bytes, (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.IMAGE_DEPTH))
            #image = tf.cast(image_reshape, tf.float32)
            #img = sess.run(image)
            print(img)
            input_img_ph = graph.get_tensor_by_name('input/image:0')
            is_training = graph.get_tensor_by_name('input/is_training:0')
            feed_dict = {
                    input_img_ph: [img],
                    is_training: FLAGS.is_training
                }

            for node in all_inference_nodes:
                if FLAGS.tname:
                    if node.name == FLAGS.tname:
                        print(node.name)
                        tensor = graph.get_tensor_by_name(node.name + ':0')
                        try:
                            _tensor = sess.run(tensor, feed_dict=feed_dict)
                        except:
                            print("cannot be fetched!")
                            continue
                        print(_tensor)
                        print(_tensor.shape)
                        print(_tensor.max())
                        print(_tensor.min())
                    if node.name == 'output/prob':
                        tensor = graph.get_tensor_by_name(node.name + ':0')
                        _tensor = sess.run(tensor, feed_dict=feed_dict)
                        prob = _tensor.reshape((cfg.IMAGE_HEIGHT,cfg.IMAGE_WIDTH,2))
                        output_img0 = (prob[:,:,0] * 255).astype(np.uint8)
                        output_img1 = (prob[:,:,1] * 255).astype(np.uint8)
                        cv2.imwrite(FLAGS.output_image0, output_img0)
                        cv2.imwrite(FLAGS.output_image1, output_img1)
                else:
                    print(node.name)
                    name_fields = node.name.split('/')
                    tensor = graph.get_tensor_by_name(node.name + ':0')
                    try:
                        _tensor = sess.run(tensor, feed_dict=feed_dict)
                    except:
                        print("cannot be fetched!")
                        continue
                    print(_tensor)
                    print(_tensor.shape)
                    print(_tensor.max())
                    print(_tensor.min())

                    print('======================================================')

                    if FLAGS.output_dir and node.name == 'output/prob':
                        #tensor = graph.get_tensor_by_name(node.name + ':0')
                        #_tensor = sess.run(tensor, feed_dict=feed_dict)
                        prob = _tensor.reshape((cfg.IMAGE_HEIGHT,cfg.IMAGE_WIDTH,2))
                        output_img0 = (prob[:,:,0] * 255).astype(np.uint8)
                        output_img1 = (prob[:,:,1] * 255).astype(np.uint8)
                        cv2.imwrite('{}/out0.png'.format(FLAGS.output_dir), output_img0)
                        cv2.imwrite('{}/out1.png'.format(FLAGS.output_dir), output_img1)

                    if FLAGS.output_dir and output_layers and node.name in output_layers:
                        layer_shape = _tensor.shape
                        output_subdir = node.name.replace('/', '_')
                        output_path = '{}/{}'.format(FLAGS.output_dir, output_subdir)
                        if not os.path.isdir(output_path):
                            os.mkdir(output_path)
                        for i in range(layer_shape[3]):
                            output_layer = _tensor[0, :, :, i]
                            output_layer_min = output_layer.min()
                            output_layer_max = output_layer.max()
                            output_layer_range = output_layer_max - output_layer_min
                            if output_layer_range < 1e-8:
                                output_layer[:,:] = 0
                            else:
                                output_layer = (output_layer - output_layer_min) / output_layer_range * 255
                            output_layer = output_layer.astype(np.uint8)
                            cv2.imwrite('{}/{}.png'.format(output_path, i), output_layer)




if __name__ == '__main__':
    tf.app.run()
