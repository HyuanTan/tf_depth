import tensorflow as tf
import numpy as np
import math

##################################
######### Initializers ###########
##################################

def msra_initializer():
    """ [ref] K. He et.al 2015: arxiv:1502.01852 
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        k = shape[0] * shape[1]        # kernel size
        d = shape[3]                   # filter number

        stddev = math.sqrt(2. / (k**2 * d))
        return tf.truncated_normal(shape, stddev=stddev, dtype=dtype)

    return _initializer

def orthogonal_initializer(scale = 1.1):
    """ From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (np.prod(shape[:-1]), shape[-1])
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape) #this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)

    return _initializer

def bilinear_initializer():

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        width = shape[0]
        heigh = shape[1]
        fw = width/2.0
        fh = heigh/2.0
        cw = (width - 1) / (2.0 * fw)
        ch = (heigh - 1) / (2.0 * fh)

        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / fw - cw)) * (1 - abs(y / fh - ch))
                bilinear[x, y] = value

        weights = np.zeros(shape)
        for i in range(shape[2]):
            for j in range(shape[3]):
                weights[:, :, i, j] = bilinear

        return tf.constant(weights, dtype=dtype)

    return _initializer


##################################




##################################
####### Varible Helpers ##########
##################################

def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory."""

    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def variable_with_l2_reg(name, shape, initializer, wd=None, 
                         use_cpu=False, loss_collection='losses'):
    """Helper to create an initialized Variable with weight decay."""

    if use_cpu:
        var = variable_on_cpu(
                  name,
                  shape,
                  initializer)
    else:
        var = tf.get_variable(name, shape, initializer=initializer)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_decay_loss')
        tf.add_to_collection(loss_collection, weight_decay)
    return var


##################################
