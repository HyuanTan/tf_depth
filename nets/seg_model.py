import tensorflow as tf
import numpy as np
from base_model import BaseModel

# input: mono image
# output: probability image with same shape of input image

class SegModel(BaseModel):
    def __init__(self, cfg, logger):
        super(SegModel, self).__init__(cfg, logger)

    def _add_input_layer(self):
        image_h = self._cfg.IMAGE_HEIGHT
        image_w = self._cfg.IMAGE_WIDTH
        image_c = self._cfg.IMAGE_DEPTH
        batch_size = self._cfg.BATCH_SIZE
        with tf.name_scope("input"):
            self._input_image = tf.placeholder(tf.float32, shape=[batch_size,image_h,image_w,image_c], name='image')
            self._is_training = tf.placeholder_with_default(False, (), name='is_training')

        self._label_image = tf.placeholder(tf.int32, shape=[None, image_h, image_w, 1], name='label')


    def _add_output_layer(self):
        with tf.name_scope("output"):
            self._segment_output = tf.nn.softmax(self._segment_logit, name="prob")

    def _add_loss_layer(self):
        num_class = self._cfg.NUM_CLASSES
        with tf.name_scope('model_loss'):
            regularize_loss = tf.add_n(tf.get_collection(self._loss_collection), name='reg_loss')
            logits = tf.reshape(self._segment_logit, (-1, num_class))
            weighted = self._cfg.LABEL_WEIGHTED
            if weighted:
                class_weight = np.array(self._cfg.CLASS_WEIGHT, dtype=np.float32)
                epsilon = tf.constant(value=1e-10)
                logits = logits + epsilon

                # consturct one-hot label array
                label_flat = tf.reshape(self._label_image, (-1, 1))
                labels = tf.reshape(tf.one_hot(label_flat, depth=num_class), (-1, num_class))

                softmax = tf.nn.softmax(logits)
                cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), class_weight), axis=[1])
                cross_entropy_loss = tf.reduce_mean(cross_entropy, name='segment_loss')
            else:
                labels = tf.reshape(labels, [-1])

                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits, labels, name='cross_entropy_per_example')
                cross_entropy_loss = tf.reduce_mean(cross_entropy, name='segment_loss')

            tf.add_to_collection(self._loss_collection, cross_entropy_loss)

            self._total_loss = tf.add_n(tf.get_collection(self._loss_collection), name='total_loss')
            self._losses = [self._total_loss, cross_entropy_loss, regularize_loss]

    @property
    def input_image(self):
        return self._input_image

    @property
    def train_phase(self):
        return self._is_training

    @property
    def label(self):
        return self._label_image

    @property
    def output(self):
        return self._segment_output

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def losses(self):
        return self._losses
        #losses = tf.get_collection(self._loss_collection)
        #return losses + self._total_loss

    @property
    def loss_collection(self):
        return self._loss_collection

