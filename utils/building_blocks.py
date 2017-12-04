import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

import sys
import math

from variable_utils import variable_on_cpu, variable_with_l2_reg, orthogonal_initializer, msra_initializer, bilinear_initializer
import log_helper

logger = log_helper.get_logger()

####################################################################
##################### Layers #######################################
####################################################################

def batch_norm_layer(inputT, is_training, decay, name, act_fn=None):
    return tf.contrib.layers.batch_norm(inputT, decay=decay, center=True, scale=True, epsilon=1e-5,
                                        activation_fn=act_fn, is_training=is_training, updates_collections=None,
                                        scope=name.split('/')[-1]+"_bn")


def conv_layer_with_bn(inputT, shape, stride=1, initializer=orthogonal_initializer(), wd=2e-4,
                       use_bias=True, use_bn=False, is_training=None, bn_decay=0.9,
                       act_fn=tf.nn.relu, use_cpu=False, loss_collection='losses', name=None):
    # bias is meaningless when using batch normalization
    if use_bn:
        use_bias = False

    out_channel = shape[3]
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name, 'conv') as scope:
        kernel = variable_with_l2_reg('weights', shape=shape, initializer=initializer,
                                      wd=wd, use_cpu=use_cpu, loss_collection=loss_collection)
        conv = tf.nn.conv2d(inputT, kernel, strides, padding='SAME')

        if use_bias:
            if use_cpu:
                biases = variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
            else:
                biases = tf.get_variable('biases', [out_channel], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.bias_add(conv, biases)

        if use_bn:
            if is_training is None:
                logger.error("is_training is None when use bn layer!")
                sys.exit(-1)
            conv = batch_norm_layer(conv, is_training, bn_decay, scope.name)

        if act_fn is not None:
            conv = act_fn(conv)

    return conv

def dilated_conv_layer_with_bn(inputT, shape, rate, initializer=orthogonal_initializer(), wd=2e-4,
                               use_bias=True, use_bn=False, is_training=None, bn_decay=0.9,
                               act_fn=tf.nn.relu, use_cpu=False, loss_collection='losses', name=None):
    # bias is meaningless when using batch normalization
    if use_bn:
        use_bias = False

    out_channel = shape[3]
    with tf.variable_scope(name, 'dilate') as scope:
        kernel = variable_with_l2_reg('weights', shape=shape, initializer=initializer,
                                      wd=wd, use_cpu=use_cpu, loss_collection=loss_collection)
        dilated = tf.nn.atrous_conv2d(inputT, kernel, rate, padding='SAME')

        if use_bias:
            if use_cpu:
                biases = variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
            else:
                biases = tf.get_variable('biases', [out_channel], initializer=tf.constant_initializer(0.0))

            dilated = tf.nn.bias_add(dilated, biases)

        if use_bn:
            if is_training is None:
                logger.error("is_training is None when use bn layer!")
                sys.exit(-1)
            dilated = batch_norm_layer(dilated, is_training, bn_decay, scope.name)

        if act_fn is not None:
            dilated = act_fn(dilated)

    return dilated

def deconv_layer_with_bn(inputT, shape, output_shape, stride=2, initializer=bilinear_initializer(), 
                         wd=None, use_bias=False, use_bn=False, is_training=None, bn_decay=0.9,
                         act_fn=None, use_cpu=False, loss_collection='losses', name=None):
    # bias is meaningless when using batch normalization
    if use_bn:
        use_bias = False

    # output_shape = [b, w, h, c]
    out_channel = shape[2]
    strides = [1, stride, stride, 1]

    with tf.variable_scope(name, 'deconv') as scope:
        kernel = variable_with_l2_reg('weights', shape=shape, initializer=initializer,
                                      wd=wd, use_cpu=use_cpu, loss_collection=loss_collection)
        deconv = tf.nn.conv2d_transpose(inputT, kernel, output_shape,
                                        strides=strides, padding='SAME')

        if use_bias:
            if use_cpu:
                biases = variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
            else:
                biases = tf.get_variable('biases', [out_channel], initializer=tf.constant_initializer(0.0))

            deconv = tf.nn.bias_add(deconv, biases)

        if use_bn:
            if is_training is None:
                logger.error("is_training is None when use bn layer!")
                sys.exit(-1)
            deconv = batch_norm_layer(deconv, is_training, bn_decay, scope.name)

        if act_fn is not None:
            deconv = act_fn(deconv)

    return deconv

def get_deconv_output_fmap_shape(ori_img_h, ori_img_w, downsample_pow):
    fmap_h = ori_img_h
    fmap_w = ori_img_w
    for i in range(downsample_pow-1):
        fmap_h = math.ceil(fmap_h / 2.0)
        fmap_w = math.ceil(fmap_w / 2.0)
    return (int(fmap_h), int(fmap_w))

def upsample_layer(inputT, out_h, out_w, name=None):
    lname = name if name is not None else 'upsample'
    return tf.image.resize_nearest_neighbor(inputT, [out_h, out_w], name=lname)

def upconv_layer_with_bn(inputT, shape, out_h, out_w, stride=1, initializer=orthogonal_initializer(),
                         wd=2e-4, use_bias=True, use_bn=False, is_training=None, bn_decay=0.9,
                         act_fn=tf.nn.relu, use_cpu=False, loss_collection='losses', name=None):
    with tf.variable_scope(name, 'upconv') as scope:
        upsample = upsample_layer(inputT, out_h, out_w)
        upconv = conv_layer_with_bn(upsample, shape, initialize=initialize, wd=wd,
                                    use_bias=use_bias, use_bn=use_bn, is_training=is_training, bn_decay=bn_decay,
                                    act_fn=act_fn, use_cpu=use_cpu, loss_collection=loss_collection)

    return upconv


####################################################################



####################################################################
#################### Modules #######################################
####################################################################

def fire(inputT, s1x1, e1x1, e3x3, initializer=orthogonal_initializer(), wd=2e-4,
         use_bn=False, is_training=None, bn_decay=0.9, act_fn=tf.nn.relu,
         use_cpu=False, loss_collection="losses", name='fire'):
    in_shape = inputT.get_shape().as_list()
    logger.info("{} input shape: [{}]".format(name, ','.join([str(x) for x in in_shape])))
    in_channel = in_shape[3]
    with tf.variable_scope(name, 'fire', [inputT]) as scope:
        sq1x1 = conv_layer_with_bn(inputT, [1, 1, in_channel, s1x1], 1, initializer, wd=wd, act_fn=act_fn,
                                   use_cpu=use_cpu, loss_collection=loss_collection, name='squeeze1x1')

        if use_bn:   # use activation after batch normalization and do not use bias when using bn
            ex1x1 = conv_layer_with_bn(sq1x1, [1, 1, s1x1, e1x1], 1, initializer, wd=wd, use_bias=False,
                               act_fn=None, use_cpu=use_cpu, loss_collection=loss_collection, name='expand1x1')
            ex3x3 = conv_layer_with_bn(sq1x1, [3, 3, s1x1, e3x3], 1, initializer, wd=wd, use_bias=False, 
                               act_fn=None, use_cpu=use_cpu, loss_collection=loss_collection, name='expand3x3')
        else:
            ex1x1 = conv_layer_with_bn(sq1x1, [1, 1, s1x1, e1x1], 1, initializer, wd=wd, act_fn=act_fn,
                                       use_cpu=use_cpu, loss_collection=loss_collection, name='expand1x1')
            ex3x3 = conv_layer_with_bn(sq1x1, [3, 3, s1x1, e3x3], 1, initializer, wd=wd, act_fn=act_fn,
                                       use_cpu=use_cpu, loss_collection=loss_collection, name='expand3x3')

        tensor = tf.concat([ex1x1, ex3x3], 3)

        if use_bn:
            tensor = batch_norm_layer(tensor, is_training, bn_decay, scope.name, act_fn=act_fn)

    return tensor

def parallel_dilate(inputT, out_ksize, rates, initializer=orthogonal_initializer(), wd=2e-4,
                    act_fn=tf.nn.relu, use_cpu=False, loss_collection='losses', name='para_dilate'):
    in_shape = inputT.get_shape().as_list()
    logger.info("{} input shape: [{}]".format(name, ','.join([str(x) for x in in_shape])))
    in_channel = in_shape[3]

    with tf.variable_scope(name, 'para_dilate', [inputT]) as scope:
        dilate_list = []
        for r in rates:
            dilated = dilated_conv_layer_with_bn(inputT, [3, 3 , in_channel, out_ksize], r, initializer, 
                                                 wd, use_bias=False, act_fn=act_fn, use_cpu=use_cpu,
                                                 loss_collection=loss_collection, name='dilate{}'.format(r))
            dilate_list.append(dilated)

        tensor = tf.add_n(dilate_list, name='dilate_total')

    return tensor
                                     
def refinement(inputT, skip_layer, out_ksize, skip_out_ksize,
               initializer=orthogonal_initializer(), wd=2e-4,
               use_bn=False, is_training=None, bn_decay=0.9, act_fn=tf.nn.relu,
               use_cpu=False, loss_collection="losses", name='refine'):
    in_shape = inputT.get_shape().as_list()
    logger.info("{} input shape: [{}]".format(name, ','.join([str(x) for x in in_shape])))
    in_channel = in_shape[3]

    skip_shape = skip_layer.get_shape().as_list()
    logger.info("{} skip layer shape: [{}]".format(name, ','.join([str(x) for x in skip_shape])))
    skip_channel = skip_shape[3]

    with tf.variable_scope(name, 'refine', [inputT, skip_layer]) as scope:
        skip_refine = conv_layer_with_bn(skip_layer, [3,3,skip_channel,skip_out_ksize], 1, initializer,
                                         wd=wd, use_bias=False, act_fn=act_fn, use_cpu=use_cpu,
                                         loss_collection=loss_collection, name='refine_skip')

        tensor = tf.concat([skip_refine, inputT], 3)

        k_shape = [3, 3, in_channel + skip_out_ksize, out_ksize]
        tensor = conv_layer_with_bn(tensor, k_shape, 1, initializer, wd=wd,
                                    use_bn=use_bn, is_training=is_training, bn_decay=bn_decay,
                                    act_fn=tf.nn.elu, loss_collection=loss_collection, name='refine_total')

    return tensor

def merge_skip(inputs, out_ksize, initializer=orthogonal_initializer(), wd=2e-4,
               use_bn=False, is_training=None, bn_decay=0.9, act_fn=tf.nn.relu,
               use_cpu=False, loss_collection="losses", name='merge_skip'):
    in_channel = 0
    for i,inputT in enumerate(inputs):
        in_shape = inputT.get_shape().as_list()
        logger.info("{} input#{} shape: [{}]".format(name, i+1, ','.join([str(x) for x in in_shape])))
        in_channel += in_shape[3]

    with tf.variable_scope(name, 'merge_skip', inputs) as scope:
        tensor = tf.concat(inputs, 3)

        k_shape = [3, 3, in_channel, out_ksize]
        tensor = conv_layer_with_bn(tensor, k_shape, 1, initializer, wd=wd, use_bn=use_bn,
                                    is_training=is_training, bn_decay=bn_decay, act_fn=tf.nn.elu,
                                    loss_collection=loss_collection, name='conv')

    return tensor

def downsampler(inputT, out_knum, initializer=orthogonal_initializer(), wd=2e-4,
                use_bn=False, is_training=None, bn_decay=0.9, act_fn=tf.nn.relu,
                use_cpu=False, loss_collection="losses", name='downsampler'):
    # bias is meaningless when using batch normalization
    conv_bias = True
    if use_bn:
        conv_bias = False

    in_shape = inputT.get_shape().as_list()
    logger.info("{} input shape: [{}]".format(name, ','.join([str(x) for x in in_shape])))
    in_knum = in_shape[3]
    conv_knum = out_knum - in_knum
    with tf.variable_scope(name, 'downsampler', [inputT]) as scope:
        # maxpooling part
        pool = tf.nn.max_pool(inputT, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool')

        # convolution part
        conv = conv_layer_with_bn(inputT, [3, 3, in_knum, conv_knum], 2, initializer, wd=wd, use_bias=conv_bias,
                                  act_fn=None, use_cpu=use_cpu, loss_collection=loss_collection, name='conv')

        # concat
        tensor = tf.concat([pool, conv], 3)

        if use_bn:
            if is_training is None:
                logger.error("is_training is None when use bn layer!")
                sys.exit(-1)
            tensor = batch_norm_layer(tensor, is_training, bn_decay, scope.name, act_fn=act_fn)
        elif act_fn is not None:
            tensor = act_fn(tensor)

    return tensor

def res_nonbt1d(inputT, ksize, dilate=False, dilate_rate=None, initializer=orthogonal_initializer(),
                wd=2e-4, use_bn=False, is_training=None, bn_decay=0.9, use_dropout=False, drop_rate=0.5,
                act_fn=tf.nn.relu, use_cpu=False, loss_collection="losses", name='nonbt1d'):
    in_shape = inputT.get_shape().as_list()
    logger.info("{} input shape: [{}]".format(name, ','.join([str(x) for x in in_shape])))
    knum = in_shape[3]
    kshape1 = [1, ksize, knum, knum]
    kshape2 = [ksize, 1, knum, knum]

    with tf.variable_scope(name, 'nonbt1d', [inputT]) as scope:
        # main stream
        main_stream = conv_layer_with_bn(inputT, kshape1, initializer=initializer, wd=wd, act_fn=act_fn,
                                         use_cpu=use_cpu, loss_collection=loss_collection, name='1dconv1')

        main_stream = conv_layer_with_bn(main_stream, kshape2, initializer=initializer, wd=wd,
                                         use_bn=use_bn, is_training=is_training, bn_decay=bn_decay, act_fn=act_fn,
                                         use_cpu=use_cpu, loss_collection=loss_collection, name='1dconv2')

        if dilate:
            if dilate_rate is None:
                logger.error("dilate_rate is None when use dilate convolution!")
                sys.exit(-1)

            main_stream = dilated_conv_layer_with_bn(main_stream, kshape1, dilate_rate, initializer=initializer, wd=wd,
                                act_fn=act_fn, use_cpu=use_cpu, loss_collection=loss_collection, name='dilate_1dconv3')

            main_stream = dilated_conv_layer_with_bn(main_stream, kshape2, dilate_rate, initializer=initializer, wd=wd,
                                        use_bn=use_bn, is_training=is_training, bn_decay=bn_decay, act_fn=None,
                                        use_cpu=use_cpu, loss_collection=loss_collection, name='dilate_1dconv4')
        else:
            main_stream = conv_layer_with_bn(main_stream, kshape1, initializer=initializer, wd=wd, act_fn=act_fn,
                                             use_cpu=use_cpu, loss_collection=loss_collection, name='1dconv3')

            main_stream = conv_layer_with_bn(main_stream, kshape2, initializer=initializer, wd=wd,
                                             use_bn=use_bn, is_training=is_training, bn_decay=bn_decay, act_fn=None,
                                             use_cpu=use_cpu, loss_collection=loss_collection, name='1dconv4')

        if use_dropout:
            main_stream = tf.contrib.layers.dropout(main_stream, 1-drop_rate, is_training=self._is_training,
                                                    scope='drop')

        # short cut: identity map
        tensor = tf.add(main_stream, inputT, name='merge_idmap')
        if act_fn is not None:
            tensor = act_fn(tensor)

        return tensor

def resbottle(inputT, s1x1, n3x3, e1x1, stride, initializer=orthogonal_initializer(), wd=2e-4,
              use_bn=False, is_training=None, bn_decay=0.9, use_dropout=False, drop_rate=0.5,
              act_fn=tf.nn.relu, use_cpu=False, loss_collection="losses", name='resbottle'):
    in_shape = inputT.get_shape().as_list()
    logger.info("{} input shape: [{}]".format(name, ','.join([str(x) for x in in_shape])))
    in_knum = in_shape[3]
    kshape1 = [1, 1, in_knum, s1x1]
    kshape2 = [3, 3, s1x1, n3x3]
    kshape3 = [1, 1, n3x3, e1x1]
    do_proj = (in_knum != e1x1) or (stride > 1)

    with tf.variable_scope(name, 'resbottle', [inputT]) as scope:
        # main stream
        sq1x1 = conv_layer_with_bn(inputT, kshape1, initializer=initializer, wd=wd, use_bn=use_bn,
                                   is_training=is_training, bn_decay=bn_decay, act_fn=act_fn,
                                   use_cpu=use_cpu, loss_collection=loss_collection, name='squeeze1x1')

        conv3x3 = conv_layer_with_bn(sq1x1, kshape2, stride, initializer=initializer, wd=wd,
                         use_bn=use_bn, is_training=is_training, bn_decay=bn_decay, act_fn=act_fn,
                         use_cpu=use_cpu, loss_collection=loss_collection, name='conv3x3')

        ex1x1 = conv_layer_with_bn(conv3x3, kshape3, initializer=initializer, wd=wd, use_bn=use_bn,
                                   is_training=is_training, bn_decay=bn_decay, act_fn=None,
                                   use_cpu=use_cpu, loss_collection=loss_collection, name='expand1x1')

        if use_dropout:
            main_stream = tf.contrib.layers.dropout(ex1x1, 1-drop_rate, is_training=self._is_training,
                                                    scope='drop')
        else:
            main_stream = ex1x1

        # short cut
        if do_proj:
            kshape_p = [1, 1, in_knum, e1x1]
            shortcut = conv_layer_with_bn(inputT, kshape_p, stride, initializer=initializer, wd=wd,
                       act_fn=None, use_cpu=use_cpu, loss_collection=loss_collection, name='shortcut')
        else:
            shortcut = inputT
        tensor = tf.add(main_stream, shortcut, name='merge_shortcut')
        if act_fn is not None:
            tensor = act_fn(tensor)

        return tensor, sq1x1, conv3x3, ex1x1

def resbottlev2(inputT, s1x1, n3x3, e1x1, stride, initializer=orthogonal_initializer(), wd=2e-4,
                use_bn=False, is_training=None, bn_decay=0.9, act_fn=tf.nn.relu,
                use_cpu=False, loss_collection="losses", name='resbottle'):
    """
    full pre-activation, in shortcut branch, no BN & act_fn at all,
    which means the last operation in main stream in each resblock should be weight-matmul/bias-add
    ref: Identity Mappings in Deep Residual Networks
    """
    in_shape = inputT.get_shape().as_list()
    logger.info("{} input shape: [{}]".format(name, ','.join([str(x) for x in in_shape])))
    in_knum = in_shape[3]
    kshape1 = [1, 1, in_knum, s1x1]
    kshape2 = [3, 3, s1x1, n3x3]
    kshape3 = [1, 1, n3x3, e1x1]
    do_proj = (in_knum != e1x1) or (stride > 1)

    with tf.variable_scope(name, 'resbottle', [inputT]) as scope:
        # main stream
        # pre-activation
        if use_bn:
            pre_activ = batch_norm_layer(inputT, is_training, bn_decay, 'pre_act', act_fn=act_fn)
        else:
            pre_activ = act_fn(inputT)

        # squeezing
        sq1x1 = conv_layer_with_bn(inputT, kshape1, initializer=initializer, wd=wd, use_bn=use_bn,
                                   is_training=is_training, bn_decay=bn_decay, act_fn=act_fn,
                                   use_cpu=use_cpu, loss_collection=loss_collection, name='squeeze1x1')

        # convolution
        conv3x3 = conv_layer_with_bn(sq1x1, kshape2, stride, initializer=initializer, wd=wd,
                         use_bn=use_bn, is_training=is_training, bn_decay=bn_decay, act_fn=act_fn,
                         use_cpu=use_cpu, loss_collection=loss_collection, name='conv3x3')

        # expansion
        ex1x1 = conv_layer_with_bn(conv3x3, kshape3, initializer=initializer, wd=wd, act_fn=None,
                                   use_cpu=use_cpu, loss_collection=loss_collection, name='expand1x1')

        # short cut
        if do_proj:
            kshape_p = [1, 1, in_knum, e1x1]
            shortcut = conv_layer_with_bn(inputT, kshape_p, stride, initializer=initializer, wd=wd,
                       act_fn=None, use_cpu=use_cpu, loss_collection=loss_collection, name='shortcut')
        else:
            shortcut = inputT

        tensor = tf.add(main_stream, shortcut, name='merge_shortcut')

        return tensor

       
####################################################################




