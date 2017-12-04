#!/usr/bin/python

import tensorflow as tf
import os
import time

import numpy as np
import cv2

from data.data_loader import TFLoadingPipeline
from config import cfg
from utils import log_helper



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cfg_file', '', """ experimental config file """)
tf.app.flags.DEFINE_string('sample_path', '', """ path to sample image """)
tf.app.flags.DEFINE_string('label_path', '', """ path to label image """)
tf.app.flags.DEFINE_string('output_dir', '', """ path to output image """)



def main(args):
    if FLAGS.cfg_file:
        print('loading config setting')
        cfg_from_file(FLAGS.cfg_file, cfg)
    cfg.MAX_STEP = 50
    cfg.BATCH_SIZE = 1
    cfg.TRAIN_QUEUE_CAPACITY = 10

    if not os.path.isdir(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    logger = log_helper.get_logger()

    data_pipeline = TFLoadingPipeline(cfg, logger, shuffle=True)
    data_pipeline.setup(FLAGS.sample_path, FLAGS.label_path, cfg.BATCH_SIZE, cfg.TRAIN_QUEUE_CAPACITY)

    with tf.Session() as sess:
        data_pipeline.start(sess)
        for step in xrange(cfg.MAX_STEP):
            image_batch, label_batch = data_pipeline.load_batch()

            logger.info('output {}th image for validation'.format(step))
            out_fname = '{}/{}.png'.format(FLAGS.output_dir, step)
            image = image_batch[0].astype(np.uint8)
            r, g, b = cv2.split(image)
            image = cv2.merge((b,g,r))
            label = label_batch[0].astype(np.uint8)
            mask = image.copy()
            mask[:,:,1][label[:,:,0]>0] = 255
            overlay = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
            cv2.imwrite(out_fname, overlay)

        data_pipeline.shutdown()


if __name__ == '__main__':
    tf.app.run()
