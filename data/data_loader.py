#!/usr/bin/python

import tensorflow as tf
import os
import sys
import random

import numpy as np
import skimage
import skimage.io
import cv2

def get_filename_list(sample_path, label_path):
    if not sample_path:
        image_filenames = None
    else:
        sample_dir = os.path.dirname(sample_path)
        sample_infile = open(sample_path, 'r')
        image_filenames = []
        for line in sample_infile:
            line = line.strip()
            image_filenames.append('{}/{}'.format(sample_dir, line))

    if not label_path:
        label_filenames = None
    else:
        label_dir = os.path.dirname(label_path)
        label_infile = open(label_path, 'r')
        label_filenames = []
        for line in label_infile:
            line = line.strip()
            label_filenames.append('{}/{}'.format(label_dir, line))

    return image_filenames, label_filenames

def instance_generator(sample_path, width, height, do_pp=True, stereo_path=None, do_stereo=False, do_recon=True):
    image_fnames, _ = get_filename_list(sample_path, None)

    if not do_stereo:
        if not do_recon:
            for fname in image_fnames:
                img = cv2.imread(fname)
                shape = img.shape
                if shape[0] != height or shape[1] != width:
                    img = cv2.resize(img, (width, height), 0, 0, cv2.INTER_AREA)
                b,g,r = cv2.split(img)
                img = cv2.merge([r,g,b])
                img = img.astype(np.float32) / 255.0
                if do_pp:
                    img_flip = np.fliplr(img)
                    img_batch = np.stack([img, img_flip], axis=0)
                else:
                    img_batch = np.expand_dims(img, axis=0)

                yield (img_batch, fname)
        else:
            stereo_fnames, _ = get_filename_list(stereo_path, None)
            for fname, sfname in zip(image_fnames, stereo_fnames):
                img = cv2.imread(fname)
                simg = cv2.imread(sfname)
                shape = img.shape
                if shape[0] != height or shape[1] != width:
                    img = cv2.resize(img, (width, height), 0, 0, cv2.INTER_AREA)
                    simg = cv2.resize(simg, (width, height), 0, 0, cv2.INTER_AREA)

                b,g,r = cv2.split(img)
                img = cv2.merge([r,g,b])
                img = img.astype(np.float32) / 255.0

                b,g,r = cv2.split(simg)
                simg = cv2.merge([r,g,b])
                simg = simg.astype(np.float32) / 255.0

                if do_pp:
                    img_flip = np.fliplr(img)
                    img_batch = np.stack([img, img_flip], axis=0)

                    # we don't care about the flipped reconstruction
                    simg_batch = np.stack([simg, simg], axis=0)
                else:
                    img_batch = np.expand_dims(img, axis=0)
                    simg_batch = np.expand_dims(simg, axis=0)

                yield ((img_batch, simg_batch), (fname, sfname))
    else:
        stereo_fnames, _ = get_filename_list(stereo_path, None)
        for fname, sfname in zip(image_fnames, stereo_fnames):
            img = cv2.imread(fname)
            simg = cv2.imread(sfname)
            shape = img.shape
            if shape[0] != height or shape[1] != width:
                img = cv2.resize(img, (width, height), 0, 0, cv2.INTER_AREA)
                simg = cv2.resize(simg, (width, height), 0, 0, cv2.INTER_AREA)

            b,g,r = cv2.split(img)
            img = cv2.merge([r,g,b])
            img = img.astype(np.float32) / 255.0

            b,g,r = cv2.split(simg)
            simg = cv2.merge([r,g,b])
            simg = simg.astype(np.float32) / 255.0

            # for stereo, no need to post process, since the flipping
            # would cause the swapping of the left/right in stereo pair
            img_batch = np.expand_dims(img, axis=0)
            simg_batch = np.expand_dims(simg, axis=0)

            yield ((img_batch, simg_batch), (fname, sfname))

def instance_label_generator(sample_path, label_path, width, height, do_pp=True, stereo_path=None):
    image_fnames, label_fnames = get_filename_list(sample_path, label_path)

    if stereo_path is None:
        for fname, label_fname in zip(image_fnames, label_fnames):
            img = cv2.imread(fname)
            shape = img.shape
            if shape[0] != height or shape[1] != width:
                img = cv2.resize(img, (width, height), 0, 0, cv2.INTER_AREA)
            b,g,r = cv2.split(img)
            img = cv2.merge([r,g,b])
            img = img.astype(np.float32) / 255.0
            if do_pp:
                img_flip = np.fliplr(img)
                img_batch = np.stack([img, img_flip], axis=0)
            else:
                img_batch = np.expand_dims(img, axis=0)

            label = cv2.imread(label_fname, -1)
            label = label.astype(np.float32) / 256      # for KITTI disp image
            yield (img_batch, label, fname)
    else:
        stereo_fnames, _ = get_filename_list(stereo_path, None)
        for fname, sfname, label_fname in zip(image_fnames, stereo_fnames, label_fnames):
            img = cv2.imread(fname)
            simg = cv2.imread(sfname)
            shape = img.shape
            if shape[0] != height or shape[1] != width:
                img = cv2.resize(img, (width, height), 0, 0, cv2.INTER_AREA)
                simg = cv2.resize(simg, (width, height), 0, 0, cv2.INTER_AREA)

            b,g,r = cv2.split(img)
            img = cv2.merge([r,g,b])
            img = img.astype(np.float32) / 255.0

            b,g,r = cv2.split(simg)
            simg = cv2.merge([r,g,b])
            simg = simg.astype(np.float32) / 255.0

            # for stereo, no need to post process, since the flipping
            # would cause the swapping of the left/right in stereo pair
            img_batch = np.expand_dims(img, axis=0)
            simg_batch = np.expand_dims(simg, axis=0)

            label = cv2.imread(label_fname, -1)
            label = label.astype(np.float32) / 256      # for KITTI disp image
            yield ((img_batch, simg_batch), label, (fname, sfname))


class LoadingPipeline(object):

    def __init__(self, cfg, logger, shuffle=True):
        self._shuffle = shuffle
        self._logger = logger
        self._cfg = cfg

    def setup(self, sample_path, label_path, batch_size, augment_suffix_list=None):
        self._logger.info('setting up the data loading pipeline')
        image_fnames, label_fnames = get_filename_list(sample_path, label_path)
        self._fname_list = zip(image_fnames, label_fnames)
        self._fname_queue = []
        self._batch_size = batch_size
        self._augment_list = []
        if augment_suffix_list:
            self._augment_list.append('')
            with open(augment_suffix_list, 'r') as f:
                for line in f:
                    line = line.strip()
                    self._augment_list.append(line)

    def setup_validate(self, sample_path, label_path, batch_size):
        self._logger.info('setting up the loading pipeline for validation data')
        image_fnames, label_fnames = get_filename_list(sample_path, label_path)
        self._val_fname_list = zip(image_fnames, label_fnames)
        self._val_fname_queue = []
        self._val_batch_size = batch_size

    def load_label(self, fname):
        im = np.array(skimage.io.imread(fname), np.int32)
        im = im[..., np.newaxis]

        return im

    def load_instance(self, fname, augment_list=None):
        if not augment_list:
            im = np.array(skimage.io.imread(fname), np.float32)
        else:
            aug_idx = np.random.randint(len(augment_list))
            suffix = augment_list[aug_idx]
            name, ext = os.path.splitext(fname)
            fname = '{}{}{}'.format(name, suffix, ext)
            im = np.array(skimage.io.imread(fname), np.float32)

        return im

    def load_batch(self):
        sample_list = []
        label_list = []

        if len(self._fname_queue) < self._batch_size:
            if self._shuffle:
                random.shuffle(self._fname_list)
            self._fname_queue += self._fname_list

        for i in xrange(self._batch_size):
            sample_fname, label_fname = self._fname_queue.pop(0)
            sample = self.load_instance(sample_fname, self._augment_list)
            label = self.load_label(label_fname)
            sample_list.append(sample)
            label_list.append(label)

        sample_batch = np.stack(sample_list)
        label_batch = np.stack(label_list)

        return sample_batch, label_batch

    def load_validate_batch(self):
        sample_list = []
        label_list = []

        if len(self._val_fname_queue) < self._val_batch_size:
            self._val_fname_queue += self._val_fname_list

        for i in xrange(self._val_batch_size):
            sample_fname, label_fname = self._val_fname_queue.pop(0)
            sample = self.load_instance(sample_fname)
            label = self.load_label(label_fname)
            sample_list.append(sample)
            label_list.append(label)

        sample_batch = np.stack(sample_list)
        label_batch = np.stack(label_list)

        return sample_batch, label_batch
       

class TFLoadingPipeline(object):

    def __init__(self, cfg, logger):

        self._logger = logger
        self._cfg = cfg

        self._sess = None
        self._coord = None
        self._threads = None

        self._aspect_ratio = float(self._cfg.IMAGE_HEIGHT) / self._cfg.IMAGE_WIDTH

    def setup(self, sample_path, label_path, batch_size, capacity,
              augment=False, shuffle=True, num_threads=1):
        self._logger.info('setting up the tf data loading pipeline')
        image_batch, label_batch = \
                self._gen_sample_label_batch(batch_size, sample_path, label_path, capacity,
                                                augment, shuffle, num_threads)
        self._image_batch = image_batch
        self._label_batch = label_batch

    def setup_validate(self, val_sample_path, val_label_path, val_batch_size, capacity,
                       augment=False, shuffle=False, num_threads=1):
        self._logger.info('setting up the tf loading pipeline for validation data')
        val_image_batch, val_label_batch = \
                self._gen_sample_label_batch(val_batch_size, val_sample_path, val_label_path,
                                                capacity, augment, shuffle, num_threads)
        self._val_image_batch = val_image_batch
        self._val_label_batch = val_label_batch

    def load_batch(self):
        image_batch, label_batch = self._sess.run([self._image_batch, self._label_batch])
        return image_batch, label_batch

    def load_validate_batch(self):
        val_image_batch, val_label_batch = self._sess.run([self._val_image_batch, self._val_label_batch])
        return val_image_batch, val_label_batch


    def _gen_sample_label_batch(self, batch_size, sample_path, label_path, capacity,
                                   augment=False, shuffle=True, num_threads=1):
        image_fnames, label_fnames = get_filename_list(sample_path, label_path)
        image_fname_tensor = tf.convert_to_tensor(image_fnames, dtype=tf.string)
        label_fname_tensor = tf.convert_to_tensor(label_fnames, dtype=tf.string)

        filename_queue = tf.train.slice_input_producer([image_fname_tensor, label_fname_tensor],
                                                       shuffle=shuffle)

        image_filename = filename_queue[0]
        label_filename = filename_queue[1]

        image_bytes = tf.image.decode_png(tf.read_file(image_filename), channels=3)
        label_bytes = tf.image.decode_png(tf.read_file(label_filename), channels=3)

        # image & label should have same shape
        # here we don't deal with the case that image & label have different shapes
        ori_shape = tf.shape(image_bytes)
        ori_h = ori_shape[0]
        ori_w = ori_shape[1]
        ori_h_f = tf.cast(ori_h, tf.float32)
        ori_w_f = tf.cast(ori_w, tf.float32)

        image = tf.image.convert_image_dtype(image_bytes, tf.float32)
        label = tf.image.convert_image_dtype(label_bytes, tf.float32)

        # only crop the upper part of the image
        # if the original aspect ratio is larger than the target, do cropping
        do_crop = (ori_h_f / ori_w_f) > self._aspect_ratio
        crop_h = tf.cast(self._aspect_ratio * ori_w_f, tf.int32)
        offset_h = ori_h - crop_h
        image = tf.cond(do_crop, lambda: tf.image.crop_to_bounding_box(image, offset_h, 0, crop_h, ori_w), lambda: image)
        label = tf.cond(do_crop, lambda: tf.image.crop_to_bounding_box(label, offset_h, 0, crop_h, ori_w), lambda: label)

        # do resizing
        image  = tf.image.resize_images(image,  [self._cfg.IMAGE_HEIGHT, self._cfg.IMAGE_WIDTH], tf.image.ResizeMethod.AREA)
        label  = tf.image.resize_images(label,  [self._cfg.IMAGE_HEIGHT, self._cfg.IMAGE_WIDTH], tf.image.ResizeMethod.AREA)

        if augment:
            # flip horizontally
            do_flip = tf.random_uniform([], 0, 1)
            tmp_img = image
            image  = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(label), lambda: image)
            label  = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(tmp_img),  lambda: label)

            # adjust brightness
            do_bright = tf.random_uniform([], 0, 1)
            bri_factor = tf.random_uniform([], -0.3, 0.3)
            image  = tf.cond(do_bright > 0.5, lambda: tf.image.adjust_brightness(image, bri_factor), lambda: image)
            label  = tf.cond(do_bright > 0.5, lambda: tf.image.adjust_brightness(label, bri_factor), lambda: label)

            # adjust contrast
            do_contrast = tf.random_uniform([], 0, 1)
            con_factor = tf.random_uniform([], 0.2, 1.8)
            image  = tf.cond(do_contrast > 0.5, lambda: tf.image.adjust_contrast(image, con_factor), lambda: image)
            label  = tf.cond(do_contrast > 0.5, lambda: tf.image.adjust_contrast(label, con_factor), lambda: label)

            # adjust saturation
            do_saturation = tf.random_uniform([], 0, 1)
            sat_factor = tf.random_uniform([], 0.2, 1.8)
            image  = tf.cond(do_saturation > 0.5, lambda: tf.image.adjust_saturation(image, sat_factor), lambda: image)
            label  = tf.cond(do_saturation > 0.5, lambda: tf.image.adjust_saturation(label, sat_factor), lambda: label)

            # adjust hue
            do_hue = tf.random_uniform([], 0, 1)
            hue_factor = tf.random_uniform([], -0.1, 0.1)
            image  = tf.cond(do_hue > 0.5, lambda: tf.image.adjust_hue(image, hue_factor), lambda: image)
            label  = tf.cond(do_hue > 0.5, lambda: tf.image.adjust_hue(label, hue_factor), lambda: label)


        self._logger.info('Filling queue with %d images before starting the pipeline. '
                          'This will take a few minutes.' % capacity)

        if shuffle:
            image_batch, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity + 3 * batch_size,
                min_after_dequeue=capacity)
        else:
            image_batch, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity + 3 * batch_size)

        return image_batch, label_batch

    def start(self, sess):
        if self._sess:
            self._logger.warn('Loading pipeline can only start once!')
            return
        self._logger.info('start data loading pipeline')
        self._sess = sess
        self._coord = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(sess=sess, coord=self._coord)

    def shutdown(self):
        self._logger.info('shut down data loading pipeline')
        self._coord.request_stop()
        self._coord.join(self._threads)

    @property
    def image_batch(self):
        return self._image_batch

    @property
    def label_batch(self):
        return self._label_batch

    @property
    def val_image_batch(self):
        return self._val_image_batch

    @property
    def val_label_batch(self):
        return self._val_label_batch

