#!/usr/bin/python

import tensorflow as tf
import sys
import os
import time
import numpy as np

from nets.res50_disp_net import Res50DispNet
from config import cfg
from utils.config_utils import cfg_from_file, print_config
from utils import log_helper



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'res50', """ which model to use """)
tf.app.flags.DEFINE_string('cfg_file', '', """ experimental config file """)


def main(args):

    if FLAGS.cfg_file:
        print('loading config setting')
        cfg_from_file(FLAGS.cfg_file, cfg)
    print_config(cfg)

    logger = log_helper.get_logger()
    logger.info("show information about {}:".format(FLAGS.model))
    if FLAGS.model == 'res50':
        model = Res50DispNet(cfg, logger)
    else:
        logger.error('wrong model type: {}'.format(FLAGS.model))
        sys.exit(-1)


if __name__ == '__main__':
    tf.app.run()
