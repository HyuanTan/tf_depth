#!/usr/bin/python

import tensorflow as tf

def flip_image(img, mode='h'):
    if mode == 'h':
        out_img = tf.image.flip_left_right(img)
    elif mode == 'v':
        out_img = tf.image.flip_up_down(img)
    else:
        out_img = img

    return out_img

def adjust_brightness(img, delta):
    return tf.image.adjust_brightness(img, delta)

def adjust_contrast(img, factor):
    return tf.image.adjust_contrast(img, factor)

def adjust_saturation(img, factor):
    return tf.image.adjust_saturation(img, factor)

def adjust_hue(img, delta):
    return tf.image.adjust_hue(img, delta)

def adjust_gamma(img, gamma):
    return tf.image.adjust_gamma(img, gamma)
