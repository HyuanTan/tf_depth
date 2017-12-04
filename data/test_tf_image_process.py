#!/usr/bin/python

import tensorflow as tf
import os
import sys
import random

import numpy as np
import skimage
import skimage.io

from tf_image_process import *

if __name__ == '__main__':
    fname = sys.argv[1]
    t_h = int(sys.argv[4])
    t_w = int(sys.argv[5])
    #img = np.array(skimage.io.imread(fname), np.uint8)
    img = tf.image.decode_png(tf.read_file(fname))
    img_shape = tf.shape(img)
    h = img_shape[0]
    w = img_shape[1]
    h_f = tf.cast(h, tf.float32)
    w_f = tf.cast(w, tf.float32)

    # do cropping
    do_crop = (h_f / w_f) > (float(t_h)/t_w)
    crop_h = tf.cast(float(t_h) / t_w * w_f, tf.int32)
    offset_h = h - crop_h
    img_crop = tf.cond(do_crop, lambda: tf.image.crop_to_bounding_box(img, offset_h, 0, crop_h, w), lambda: img)
    img_resize  = tf.image.resize_images(img_crop,  [t_h, t_w], tf.image.ResizeMethod.AREA)
    tf.summary.image('crop_img', tf.expand_dims(img_crop, 0), 1)
    tf.summary.image('resize_img', tf.expand_dims(img_resize, 0), 1)
    tf.summary.scalar('crop_h', crop_h)
    tf.summary.scalar('offset_h', offset_h)


    #img_ph = tf.placeholder(tf.uint8, shape=[h,w,3])
    #img_ph = tf.image.convert_image_dtype(img_ph,  tf.float32)
    tf.summary.image('origin', tf.expand_dims(img, 0), 1)

    img_flip = flip_image(img, 'h')
    tf.summary.image('flip_img', tf.expand_dims(img_flip, 0), 1)

    random_bright = tf.random_uniform([], -0.5, 0.5)
    img_bright = adjust_brightness(img, random_bright)
    tf.summary.image('bright_img', tf.expand_dims(img_bright, 0), 1)
    tf.summary.scalar('bright', random_bright)

    random_contrast = tf.random_uniform([], 0.2, 1.8)
    img_contrast = adjust_contrast(img, random_contrast)
    tf.summary.image('contrast_img', tf.expand_dims(img_contrast, 0), 1)
    tf.summary.scalar('contrast', random_contrast)

    random_saturation = tf.random_uniform([], 0.2, 1.8)
    img_saturation = adjust_saturation(img, random_saturation)
    tf.summary.image('saturation_img', tf.expand_dims(img_saturation, 0), 1)
    tf.summary.scalar('saturation', random_saturation)

    random_hue = tf.random_uniform([], -0.1, 0.1)
    img_hue = adjust_hue(img, random_hue)
    tf.summary.image('hue_img', tf.expand_dims(img_hue, 0), 1)
    tf.summary.scalar('hue', random_hue)

    #random_gamma = tf.random_uniform([], 0.4, 1.6)
    #img_gamma = adjust_gamma(img_ph, random_gamma)
    #tf.summary.image('gamma_img', img_gamma, 1)
    #tf.summary.scalar('gamma', random_gamma)

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        #feed_dict = {img_ph: img}
        in_shape = img_shape.eval(session=sess)
        print("input shape: [" + ','.join([str(x) for x in in_shape]) + ']')
        summary_writer = tf.summary.FileWriter(sys.argv[2], sess.graph)

        for i in range(int(sys.argv[3])):
            #sess.run([random_bright, random_contrast, random_saturation, random_hue, img_flip, img_bright, img_contrast, img_saturation, img_hue])
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, i)


