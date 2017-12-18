#!/usr/bin/python

import tensorflow as tf
import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from nets.res50_disp_net import Res50DispNet
from data.data_loader import instance_generator
import data.image_process as image_process
from config import cfg
from utils.config_utils import cfg_from_file, print_config
from utils import log_helper



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'res50', """ which model to use """)
tf.app.flags.DEFINE_string('ckpt_path', '', """ path to restore model ckpt """)
tf.app.flags.DEFINE_string('cfg_file', '', """ experimental config file """)
tf.app.flags.DEFINE_string('sample_path', '', """ path to sample image """)
tf.app.flags.DEFINE_string('stereo_path', '', """ path to stereo image """)
tf.app.flags.DEFINE_string('output_path', '', """ path to disparity image """)
tf.app.flags.DEFINE_string('recon_path', '', """ path to output reconstructed demo configure """)
tf.app.flags.DEFINE_boolean('use_avg', True, """ whether to use moving average model """)
tf.app.flags.DEFINE_boolean('do_pp', True, """ whether to do post processing """)
tf.app.flags.DEFINE_boolean('do_stereo', False, """ whether to do prediction using stereo pair """)
tf.app.flags.DEFINE_integer('resize_ratio', 0, """ output iamges resize ratio """)

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def main(args):

    if FLAGS.cfg_file:
        print('loading config setting')
        cfg_from_file(FLAGS.cfg_file, cfg)

    do_pp = FLAGS.do_pp
    if FLAGS.do_stereo:
        do_pp = False
        cfg.DO_STEREO = True
    else:
        cfg.DO_STEREO = False

    cfg.BATCH_SIZE = 1
    if do_pp:
        cfg.BATCH_SIZE = 2

    print_config(cfg)

    output_path = FLAGS.output_path
    if output_path != '':
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

    logger = log_helper.get_logger()
    do_recon = FLAGS.recon_path != ''
    if do_recon:
        if FLAGS.stereo_path == '':
            logger.error("to do reconstruction, stereo_path has to be set!")
            sys.exit(-1)
        recon_path = FLAGS.recon_path
        if not os.path.isdir(recon_path):
            os.makedirs(recon_path)
    stereo_path = FLAGS.stereo_path

    if FLAGS.model == 'res50':
        model = Res50DispNet(cfg, logger)
    else:
        logger.error('wrong model type: {}'.format(FLAGS.model))
        sys.exit(-1)

    if FLAGS.use_avg:
        # get moving avg
        variable_averages = tf.train.ExponentialMovingAverage(cfg.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    else:
        saver = tf.train.Saver(model.all_variables)

    with tf.Session() as sess:
        # restore model
        logger.info("restoring model ......")
        saver.restore(sess, FLAGS.ckpt_path)
        total_time_elapsed = 0.0

        aspect_ratio = float(cfg.IMAGE_WIDTH) / cfg.IMAGE_HEIGHT
        for image, fname in instance_generator(FLAGS.sample_path, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT,
                                               do_pp, stereo_path, cfg.DO_STEREO, do_recon):
            if cfg.DO_STEREO or do_recon:
                sample_name = fname[0]
                stereo_name = fname[1]
                logger.info("inference for {} & {}".format(fname[0], fname[1]))
                feed_dict = {
                    model.left_image: image[0],
                    model.right_image: image[1]
                }
                fname = sample_name
            else:
                logger.info("inference for {}".format(fname))
                feed_dict = {
                    model.left_image: image
                }

            begin_ts = time.time()

            if not do_recon:
                pre_disp = sess.run(model.left_disparity[0], feed_dict=feed_dict)
            else:
                pre_disp, recon, recon_diff = sess.run([model.left_disparity[0],
                                                        model.left_reconstruction[0],
                                                        model.left_recon_diff[0]],
                                                        feed_dict=feed_dict)
                recon = recon[0,:,:,:]
                recon_diff = recon_diff[0,:,:,:]

                #print pre_disp.shape
                #print recon.shape
                #print recon_diff.shape

            end_ts = time.time()
            logger.info("cost time: {} s".format(end_ts - begin_ts))
            total_time_elapsed += end_ts - begin_ts

            if do_pp:
                disp = post_process_disparity(pre_disp.squeeze())
            else:
                disp = pre_disp[0].squeeze()

            if FLAGS.resize_ratio != 0 and FLAGS.resize_ratio != 1:
                disp = cv2.resize(disp, (FLAGS.resize_ratio*cfg.IMAGE_WIDTH, FLAGS.resize_ratio*cfg.IMAGE_HEIGHT),
                                  interpolation=cv2.INTER_LINEAR)


            # output disparity
            if output_path != '':
                if do_pp:
                    output_fname = output_path + "/pp_" + os.path.basename(fname)
                else:
                    output_fname = output_path + "/" + os.path.basename(fname)

                plt.imsave(output_fname, disp, cmap=plt.cm.gray)

            if recon_path is not None:
                o_image = cv2.resize(image[0][0],
                                     (FLAGS.resize_ratio*cfg.IMAGE_WIDTH, FLAGS.resize_ratio*cfg.IMAGE_HEIGHT),
                                     interpolation=cv2.INTER_LINEAR)
                o_recon = cv2.resize(recon,
                                     (FLAGS.resize_ratio*cfg.IMAGE_WIDTH, FLAGS.resize_ratio*cfg.IMAGE_HEIGHT),
                                     interpolation=cv2.INTER_LINEAR)
                o_diff = cv2.resize(recon_diff,
                                    (FLAGS.resize_ratio*cfg.IMAGE_WIDTH, FLAGS.resize_ratio*cfg.IMAGE_HEIGHT),
                                    interpolation=cv2.INTER_LINEAR)

                whole_fig = plt.figure(figsize=(int(aspect_ratio*8), 8))
                gs = gridspec.GridSpec(2, 2)
                a = plt.subplot(gs[0, 0])
                b = plt.subplot(gs[1, 0])
                c = plt.subplot(gs[0, 1])
                d = plt.subplot(gs[1, 1])

                a.imshow(o_image)
                a.set_title('raw_image')
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)

                b.imshow(disp, cmap=plt.cm.gray)
                b.set_title('disparity')
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)

                c.imshow(o_recon)
                c.set_title('reconstruct')
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)

                d.imshow(o_diff)
                d.set_title('recon_diff')
                #plt.tight_layout()
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)

                output_fname = recon_path + "/" + os.path.basename(fname)
                plt.savefig(output_fname)

                # for release memory
                plt.clf()
                plt.close()

        print("total time elapsed: {} s".format(total_time_elapsed))


if __name__ == '__main__':
    tf.app.run()
