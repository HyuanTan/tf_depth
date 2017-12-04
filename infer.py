#!/usr/bin/python

import tensorflow as tf
import os
import time
import cv2
import numpy as np

from nets.sq_net import SQSegNet
from nets.erf_net import ERFSegNet
from data.data_loader import instance_generator
import data.image_process as image_process
from config import cfg
from utils.training_utils import get_train_op, add_summaries
from utils.config_utils import cfg_from_file, print_config
from utils.metrics import per_class_iu
from utils import log_helper



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'sq', """ which model to use """)
tf.app.flags.DEFINE_string('ckpt_path', '', """ path to restore model ckpt """)
tf.app.flags.DEFINE_string('cfg_file', '', """ experimental config file """)
tf.app.flags.DEFINE_string('sample_path', '', """ path to sample image """)
tf.app.flags.DEFINE_string('output_path', '', """ path to validate label image """)
tf.app.flags.DEFINE_string('mask_path', '', """ path to validate label image """)
tf.app.flags.DEFINE_boolean('restore_avg', True, """ whether to restore moving avg version """)



def checkArgs():
    print('Inference model: {}'.format(FLAGS.model))
    print("Sample image path: {}".format(FLAGS.sample_path))
    print("Prediction output path: {}".format(FLAGS.output_path))
    print("Mask output path: {}".format(FLAGS.mask_path))
    print("Modle checkpoint path: {}".format(FLAGS.ckpt_path))
    print("Config file: {}".format(FLAGS.cfg_file))

def genPredProb(image, num_classes):
    """ store label data to colored image """
    im = np.zeros((image.shape[0], image.shape[1], 3))
    for i in range(num_classes - 1):
        im[:,:,0] += image[:,:,i+1] * cfg.COLOR_CODE[i][0]
        im[:,:,1] += image[:,:,i+1] * cfg.COLOR_CODE[i][1]
        im[:,:,2] += image[:,:,i+1] * cfg.COLOR_CODE[i][2]
        #r = np.zeros_like(prob_img)
        #g = prob_img.copy() * 255
        #b = np.zeros_like(prob_img)
        #rgb = np.zeros((image.shape[0], image.shape[1], 3))
        #rgb[:,:,0] = r
        #rgb[:,:,1] = g
        #rgb[:,:,2] = b
    im = np.uint8(im)
    return im


def main(args):
    checkArgs()

    if FLAGS.cfg_file:
        print('loading config setting')
        cfg_from_file(FLAGS.cfg_file, cfg)
    cfg.BATCH_SIZE = 1
    print_config(cfg)

    output_path = FLAGS.output_path
    mask_path = FLAGS.mask_path
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if not os.path.isdir(mask_path):
        os.makedirs(mask_path)

    logger = log_helper.get_logger()
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


    with tf.Session() as sess:
        # restore model
        logger.info("restoring model ......")
        saver.restore(sess, FLAGS.ckpt_path)
        total_time_elapsed = 0.0

        for image, fname in instance_generator(FLAGS.sample_path):
            logger.info("predicting for {}".format(fname))

            begin_ts = time.time()
            feed_dict = {
                model.input_image: [image],
                model.train_phase: False
            }

            prediction = sess.run(model.output, feed_dict=feed_dict)
            end_ts = time.time()
            logger.info("cost time: {} s".format(end_ts - begin_ts))
            total_time_elapsed += end_ts - begin_ts

            # output_image to verify
            output_fname = output_path + "/" + os.path.basename(fname)
            pred_prob = genPredProb(prediction[0], cfg.NUM_CLASSES)
            ret = cv2.imwrite(output_fname, pred_prob)
            if not ret:
                logger.error('writing image to {} failed!'.format(output_fname))
                sys.exit(-1)

            # masking image
            mask_fname = mask_path + "/" + os.path.basename(fname)
            r, g, b = cv2.split(image.astype(np.uint8))
            cv_img = cv2.merge([b, g, r])
            masked = image_process.prob_mask(cv_img, pred_prob)
            ret = cv2.imwrite(mask_fname, masked)
            if not ret:
                logger.error('writing image to {} failed!'.format(output_fname))
                sys.exit(-1)

        print("total time elapsed: {} s".format(total_time_elapsed))


if __name__ == '__main__':
    tf.app.run()
