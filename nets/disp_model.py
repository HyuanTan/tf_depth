import tensorflow as tf
import numpy as np
from base_model import BaseModel
from utils.bilinear_sampler import bilinear_sampler_1d_h

# input: mono left image
# output: multi-scale disparity images
# supervision: multi-scale reconstruction between left & right images

class DispModel(BaseModel):
    def __init__(self, cfg, logger):
        super(DispModel, self).__init__(cfg, logger)

    def _add_input_layer(self):
        image_h = self._cfg.IMAGE_HEIGHT
        image_w = self._cfg.IMAGE_WIDTH
        image_c = self._cfg.IMAGE_DEPTH
        batch_size = self._cfg.BATCH_SIZE
        with tf.name_scope("input"):
            self._left = tf.placeholder(tf.float32, shape=[batch_size,image_h,image_w,image_c], name='left')
            self._right = tf.placeholder(tf.float32, shape=[batch_size,image_h,image_w,image_c], name='right')
            self._is_training = tf.placeholder_with_default(False, (), name='is_training')

    def _add_output_layer(self):
        with tf.name_scope('output'):
            # disparities
            self._disp_left = [tf.expand_dims(d[:,:,:,0], 3, name='disp_left{}'.format(i)) for i,d in enumerate(self._disp_pyramid)]
            self._disp_right = [tf.expand_dims(d[:,:,:,1], 3, name='disp_right{}'.format(i)) for i,d in enumerate(self._disp_pyramid)]

    def _add_loss_layer(self):
        # supervision
        self._left_pyramid = self._scale_pyramid(self._left)
        self._right_pyramid = self._scale_pyramid(self._right)

        # reconstructed images
        with tf.variable_scope('reconstruction'):
            self._left_recon  = [self._generate_image_left(self._right_pyramid[i], self._disp_left[i], name='left{}'.format(i))  for i in range(self._cfg.SCALE_NUM)]
            self._right_recon = [self._generate_image_right(self._left_pyramid[i], self._disp_right[i], name='right{}'.format(i)) for i in range(self._cfg.SCALE_NUM)]

        # lr consistency
        if self._cfg.USE_CONSIST:
            with tf.variable_scope('left-right'):
                right_to_left_disp = [self._generate_image_left(self._disp_right[i], self._disp_left[i], name='right_to_left{}'.format(i))  for i in range(self._cfg.SCALE_NUM)]
                left_to_right_disp = [self._generate_image_right(self._disp_left[i], self._disp_right[i], name='left_to_right{}'.format(i)) for i in range(self._cfg.SCALE_NUM)]

        # disparity smoothness
        with tf.variable_scope('smoothness'):
            disp_left_smoothness  = self._get_disparity_smoothness(self._disp_left,  self._left_pyramid, name='left')
            disp_right_smoothness = self._get_disparity_smoothness(self._disp_right, self._right_pyramid, name='right')

        # start calculating losses
        self._losses = []

        # IMAGE RECONSTRUCTION
        # L1
        with tf.variable_scope('recon_l1_loss'):
            self._l1_left = [tf.abs(self._left_recon[i] - self._left_pyramid[i], name='left_diff{}'.format(i)) for i in range(self._cfg.SCALE_NUM)]
            l1_reconstruction_loss_left = [tf.reduce_mean(l, name='left{}'.format(i)) for i,l in enumerate(self._l1_left)]
            self._l1_right = [tf.abs(self._right_recon[i] - self._right_pyramid[i], name='right_diff{}'.format(i)) for i in range(self._cfg.SCALE_NUM)]
            l1_reconstruction_loss_right = [tf.reduce_mean(l, name='right{}'.format(i)) for i,l in enumerate(self._l1_right)]

        self._losses += l1_reconstruction_loss_left + l1_reconstruction_loss_right
        image_l1_loss = tf.add_n(l1_reconstruction_loss_left + l1_reconstruction_loss_right)

        # SSIM
        if self._cfg.USE_SSIM:
            with tf.variable_scope('recon_ssim_loss'):
                ssim_left = [self._SSIM( self._left_recon[i],  self._left_pyramid[i]) for i in range(self._cfg.SCALE_NUM)]
                ssim_loss_left = [tf.reduce_mean(s, name='left{}'.format(i)) for i,s in enumerate(ssim_left)]
                ssim_right = [self._SSIM(self._right_recon[i], self._right_pyramid[i]) for i in range(self._cfg.SCALE_NUM)]
                ssim_loss_right = [tf.reduce_mean(s, name='right{}'.format(i)) for i,s in enumerate(ssim_right)]

            self._losses += ssim_loss_left + ssim_loss_right
            image_ssim_loss = tf.add_n(ssim_loss_left + ssim_loss_right)

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness_loss'):
            smooth_left_loss  = [tf.multiply(tf.reduce_mean(tf.abs(disp_left_smoothness[i])) + tf.reduce_mean(tf.abs(disp_left_smoothness[i+self._cfg.SCALE_NUM])),
                                             1.0/2**i, name='left{}'.format(i)) for i in range(self._cfg.SCALE_NUM)]
            smooth_right_loss = [tf.multiply(tf.reduce_mean(tf.abs(disp_right_smoothness[i])) + tf.reduce_mean(tf.abs(disp_right_smoothness[i+self._cfg.SCALE_NUM])),
                                             1.0/2**i, name='right{}'.format(i)) for i in range(self._cfg.SCALE_NUM)]

        self._losses += smooth_left_loss + smooth_right_loss

        # CONSISTENCY
        if self._cfg.USE_CONSIST:
            with tf.variable_scope('consistency_loss'):
                consist_left_loss  = [tf.reduce_mean(tf.abs(right_to_left_disp[i] - self._disp_left[i]), name='left{}'.format(i)) for i in range(self._cfg.SCALE_NUM)]
                consist_right_loss = [tf.reduce_mean(tf.abs(left_to_right_disp[i] - self._disp_right[i]), name='right{}'.format(i)) for i in range(self._cfg.SCALE_NUM)]

            self._losses += consist_left_loss + consist_right_loss

        # MODEL LOSS
        with tf.variable_scope('model_loss'):
            # REGULARIZATION LOSS
            # currently only regularization losses are in collection "self._loss_collection"
            if self._cfg.WEIGHT_DECAY is not None:
                regularize_loss = tf.add_n(tf.get_collection(self._loss_collection), name='reg_loss')
                self._losses.append(regularize_loss)
                self._model_losses.append(regularize_loss)

            # WEIGTHED SUM IMAGE LOSS
            if self._cfg.USE_SSIM:
                image_loss_right = [self._cfg.ALPHA_IMAGE_LOSS * ssim_loss_right[i] + (1 - self._cfg.ALPHA_IMAGE_LOSS) * l1_reconstruction_loss_right[i] for i in range(self._cfg.SCALE_NUM)]
                image_loss_left  = [self._cfg.ALPHA_IMAGE_LOSS * ssim_loss_left[i]  + (1 - self._cfg.ALPHA_IMAGE_LOSS) * l1_reconstruction_loss_left[i]  for i in range(self._cfg.SCALE_NUM)]

                image_loss = tf.add_n(image_loss_left + image_loss_right, name='image_loss')
            else:
                image_loss = tf.add_n(l1_reconstruction_loss_left + l1_reconstruction_loss_right, name='image_loss')

            tf.add_to_collection(self._loss_collection, image_loss)

            # WEIGTHED SUM SMOOTHNESS LOSS
            smooth_loss = tf.add_n(smooth_left_loss + smooth_right_loss, name='smooth_loss')
            tf.add_to_collection(self._loss_collection, self._cfg.SMOOTH_LOSS_WEIGHT * smooth_loss)

            # WEIGTHED SUM CONSISTENCY LOSS
            if self._cfg.USE_CONSIST:
                consist_loss = tf.add_n(consist_left_loss + consist_right_loss, name='consist_loss')
                tf.add_to_collection(self._loss_collection, self._cfg.CONSIST_LOSS_WEIGHT * consist_loss)

            # TOTAL LOSS
            self._total_loss = tf.add_n(tf.get_collection(self._loss_collection), name='total_loss')

        self._losses += [image_loss, smooth_loss, consist_loss, self._total_loss]
        self._model_losses = [image_loss, image_l1_loss, image_ssim_loss, smooth_loss, consist_loss, self._total_loss]

    def _generate_image_left(self, img, disp, name=None):
        return bilinear_sampler_1d_h(img, -disp, name=name)

    def _generate_image_right(self, img, disp, name=None):
        return bilinear_sampler_1d_h(img, disp, name=name)

    def _gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def _gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def _get_disparity_smoothness(self, disp, pyramid, name='disparity_smooth'):
        disp_gradients_x = [self._gradient_x(d) for d in disp]
        disp_gradients_y = [self._gradient_y(d) for d in disp]

        image_gradients_x = [self._gradient_x(img) for img in pyramid]
        image_gradients_y = [self._gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [tf.multiply(disp_gradients_x[i], weights_x[i], name=name+'_x{}'.format(i)) for i in range(self._cfg.SCALE_NUM)]
        smoothness_y = [tf.multiply(disp_gradients_y[i], weights_y[i], name=name+'_y{}'.format(i)) for i in range(self._cfg.SCALE_NUM)]
        return smoothness_x + smoothness_y

    def _SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID')
        mu_y = tf.nn.avg_pool(y, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID')

        sigma_x  = tf.nn.avg_pool(x ** 2, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID') - mu_x ** 2
        sigma_y  = tf.nn.avg_pool(y ** 2, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID') - mu_y ** 2
        sigma_xy = tf.nn.avg_pool(x * y, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def _scale_pyramid(self, img):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(self._cfg.SCALE_NUM - 1):
            ratio = 2 ** (i + 1)
            nh = h / ratio
            nw = w / ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    @property
    def layers(self):
        return self._layers

    @property
    def left_image(self):
        return self._left

    @property
    def right_image(self):
        return self._right

    @property
    def left_pyramid(self):
        return self._left_pyramid

    @property
    def right_pyramid(self):
        return self._right_pyramid

    @property
    def train_phase(self):
        return self._is_training

    @property
    def left_disparity(self):
        return self._disp_left

    @property
    def right_disparity(self):
        return self._disp_right

    @property
    def left_reconstruction(self):
        return self._left_recon

    @property
    def right_reconstruction(self):
        return self._right_recon

    @property
    def left_recon_diff(self):
        return self._l1_left

    @property
    def right_recon_diff(self):
        return self._l1_right

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def model_losses(self):
        return self._model_losses

    @property
    def losses(self):
        return self._losses

    @property
    def loss_collection(self):
        return self._loss_collection
