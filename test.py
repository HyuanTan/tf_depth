#!/usr/bin/python

import tensorflow as tf
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from nets.res50_disp_net import Res50DispNet
from data.data_loader import instance_label_generator
from config import cfg
from utils.config_utils import cfg_from_file, print_config
from utils.metrics import depth_metrics
from utils import log_helper



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'res50', """ which model to use """)
tf.app.flags.DEFINE_string('ckpt_path', '', """ path to restore model ckpt """)
tf.app.flags.DEFINE_string('cfg_file', '', """ experimental config file """)
tf.app.flags.DEFINE_string('sample_path', '', """ path to sample image """)
tf.app.flags.DEFINE_string('label_path', '', """ path to labels """)
tf.app.flags.DEFINE_string('stereo_path', '', """ path to stereo image """)
tf.app.flags.DEFINE_string('base_path', '', """ path to baseline """)
tf.app.flags.DEFINE_string('output_path', '', """ path to disparity image """)
tf.app.flags.DEFINE_boolean('use_avg', True, """ whether to use moving average model """)
tf.app.flags.DEFINE_boolean('do_pp', True, """ whether to do post processing """)

KITTI_BASE = 0.54
KITTI_FOCAL = dict()
KITTI_FOCAL[1242] = 721.5377
KITTI_FOCAL[1241] = 718.856
KITTI_FOCAL[1224] = 707.0493
KITTI_FOCAL[1238] = 718.3351

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def checkArgs():
    print('Inference model: {}'.format(FLAGS.model))
    print("Sample image path: {}".format(FLAGS.sample_path))
    print("Label image path: {}".format(FLAGS.label_path))
    print("Prediction output path: {}".format(FLAGS.output_path))
    print("Modle checkpoint path: {}".format(FLAGS.ckpt_path))
    print("Config file: {}".format(FLAGS.cfg_file))


def main(args):
    checkArgs()

    if FLAGS.cfg_file:
        print('loading config setting')
        cfg_from_file(FLAGS.cfg_file, cfg)
    if FLAGS.stereo_path != '':
        cfg.DO_STEREO = True
    else:
        cfg.DO_STEREO = False

    base_path = None
    title_str = "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('ratio', 'abs_rel_i', 'sq_rel_i', 'rmse_i', 'rmse_log_i', 'd1_all_i', 'a1_i', 'a2_i', 'a3_i', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'd1_all', 'a1', 'a2', 'a3')
    if FLAGS.base_path != '':
        base_path = FLAGS.base_path
        title_str = "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('ratio', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'd1_all', 'a1', 'a2', 'a3', 'abs_rel_b', 'sq_rel_b', 'rmse_b', 'rmse_log_b', 'd1_all_b', 'a1_b', 'a2_b', 'a3_b')


    stereo_path = FLAGS.stereo_path if cfg.DO_STEREO else None

    cfg.BATCH_SIZE = 1
    if FLAGS.do_pp and not cfg.DO_STEREO:
        cfg.BATCH_SIZE = 2

    print_config(cfg)

    if FLAGS.output_path != '':
        output_path = FLAGS.output_path
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

    logger = log_helper.get_logger()
    if FLAGS.model == 'res50':
        model = Res50DispNet(cfg, logger)
    else:
        logger.error('wrong model type: {}'.format(FLAGS.model))
        sys.exit(-1)

    # get moving avg
    if FLAGS.use_avg:
        variable_averages = tf.train.ExponentialMovingAverage(cfg.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    else:
        saver = tf.train.Saver(model.all_variables)

    total_time_elapsed = 0
    with tf.Session() as sess:
        # restore model
        logger.info("restoring model ......")
        saver.restore(sess, FLAGS.ckpt_path)

        rate_list     = []
        rmse_inter_list     = []
        rmse_log_inter_list = []
        abs_rel_inter_list  = []
        sq_rel_inter_list   = []
        d1_all_inter_list   = []
        a1_inter_list       = []
        a2_inter_list       = []
        a3_inter_list       = []
        rmse_list     = []
        rmse_log_list = []
        abs_rel_list  = []
        sq_rel_list   = []
        d1_all_list   = []
        a1_list       = []
        a2_list       = []
        a3_list       = []


        for image, label, fname in instance_label_generator(FLAGS.sample_path, FLAGS.label_path, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT,
                                                            FLAGS.do_pp, stereo_path, base_path=base_path):
            if cfg.DO_STEREO:
                sample_name = fname[0]
                stereo_name = fname[1]
                logger.info("testing for {} & {}".format(fname[0], fname[1]))
                feed_dict = {
                    model.left_image: image[0],
                    model.right_image: image[1]
                }
                fname = sample_name
            else:
                logger.info("testing for {}".format(fname))
                if base_path is None:
                    feed_dict = {
                        model.left_image: image
                    }
                else:
                    feed_dict = {
                        model.left_image: image[0]
                    }

            begin_ts = time.time()

            pre_disp = sess.run(model.left_disparity[0], feed_dict=feed_dict)

            end_ts = time.time()
            logger.info("cost time: {} s".format(end_ts - begin_ts))
            total_time_elapsed += end_ts - begin_ts

            if FLAGS.do_pp and not cfg.DO_STEREO:
                disp = post_process_disparity(pre_disp.squeeze())
            else:
                disp = pre_disp[0].squeeze()

            base_disp = None if base_path is None else image[-1]

            width = label.shape[1]
            focal = KITTI_FOCAL[width]
            base = KITTI_BASE
            rate, d1_all_inter, abs_rel_inter, sq_rel_inter, rmse_inter, rmse_log_inter, a1_inter, a2_inter, a3_inter, d1_all, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_metrics(label, disp, focal, base, base_disp)

            print(title_str)
            print("{:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(rate, abs_rel_inter, sq_rel_inter, rmse_inter, rmse_log_inter, d1_all_inter, a1_inter, a2_inter, a3_inter, abs_rel, sq_rel, rmse, rmse_log, d1_all, a1, a2, a3))

            rate_list.append(rate)
            rmse_inter_list.append(rmse_inter)
            rmse_log_inter_list.append(rmse_log_inter)
            abs_rel_inter_list.append(abs_rel_inter)
            sq_rel_inter_list.append(sq_rel_inter)
            d1_all_inter_list.append(d1_all_inter)
            a1_inter_list.append(a1_inter)
            a2_inter_list.append(a2_inter)
            a3_inter_list.append(a3_inter)
            rmse_list.append(rmse)
            rmse_log_list.append(rmse_log)
            abs_rel_list.append(abs_rel)
            sq_rel_list.append(sq_rel)
            d1_all_list.append(d1_all)
            a1_list.append(a1)
            a2_list.append(a2)
            a3_list.append(a3)

            # output_image to verify
            if FLAGS.output_path != '':
                if FLAGS.do_pp and not cfg.DO_STEREO:
                    output_fname = output_path + "/pp_" + os.path.basename(fname)
                else:
                    output_fname = output_path + "/" + os.path.basename(fname)
                plt.imsave(output_fname, disp, cmap=plt.cm.gray)

        rate_mean = np.array(rate_list).mean()
        rmse_inter_mean = np.array(rmse_inter_list).mean()
        rmse_log_inter_mean = np.array(rmse_log_inter_list).mean()
        abs_rel_inter_mean = np.array(abs_rel_inter_list).mean()
        sq_rel_inter_mean = np.array(sq_rel_inter_list).mean()
        d1_all_inter_mean = np.array(d1_all_inter_list).mean()
        a1_inter_mean = np.array(a1_inter_list).mean()
        a2_inter_mean = np.array(a2_inter_list).mean()
        a3_inter_mean = np.array(a3_inter_list).mean()
        rmse_mean = np.array(rmse_list).mean()
        rmse_log_mean = np.array(rmse_log_list).mean()
        abs_rel_mean = np.array(abs_rel_list).mean()
        sq_rel_mean = np.array(sq_rel_list).mean()
        d1_all_mean = np.array(d1_all_list).mean()
        a1_mean = np.array(a1_list).mean()
        a2_mean = np.array(a2_list).mean()
        a3_mean = np.array(a3_list).mean()

        print("============total metric============")
        print(title_str)
        print("{:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(rate_mean, abs_rel_inter_mean, sq_rel_inter_mean, rmse_inter_mean, rmse_log_inter_mean, d1_all_inter_mean, a1_inter_mean, a2_inter_mean, a3_inter_mean, abs_rel_mean, sq_rel_mean, rmse_mean, rmse_log_mean, d1_all_mean, a1_mean, a2_mean, a3_mean))

        print("total time elapsed: {} s".format(total_time_elapsed))


if __name__ == '__main__':
    tf.app.run()
