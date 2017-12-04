import tensorflow as tf
from seg_model import SegModel
from utils.building_blocks import conv_layer_with_bn, deconv_layer_with_bn, get_deconv_output_fmap_shape, fire, parallel_dilate, refinement
from utils.variable_utils import orthogonal_initializer, msra_initializer


class SQSegNet(SegModel):
    def __init__(self, cfg, logger):
        super(SQSegNet, self).__init__(cfg, logger)

    def _build_encoder(self, inputT):
        cfg = self._cfg

        in_shape = inputT.get_shape().as_list()
        self._logger.info("encoder input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        skip_layers = {}

        # Encoder is based on SqueezeNet 1.1

        # conv1
        net = conv_layer_with_bn(inputT, [3,3,in_shape[3],64], 2, initializer=msra_initializer(),
                                 wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN, is_training=self._is_training, 
                                 bn_decay=cfg.BN_DECAY, act_fn=tf.nn.elu, loss_collection=self._loss_collection,
                                 name='conv1')

        skip_layers['down2x'] = net

        # pool1
        net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='pool1')

        # fire2
        net = fire(net, 16, 64, 64, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, 
                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                   act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='fire2')

        # fire3
        net = fire(net, 16, 64, 64, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, 
                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                   act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='fire3')

        skip_layers['down4x'] = net

        # pool3
        net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='pool3')

        # fire4
        net = fire(net, 32, 128, 128, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                   act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='fire4')

        # fire5
        net = fire(net, 32, 128, 128, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                   act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='fire5')

        skip_layers['down8x'] = net

        # pool5
        net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='pool5')

        # fire6
        net = fire(net, 48, 192, 192, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                   act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='fire6')

        # fire7
        net = fire(net, 48, 192, 192, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                   act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='fire7')

        # fire8
        net = fire(net, 64, 256, 256, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                   act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='fire8')

        # fire9
        net = fire(net, 64, 256, 256, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                   act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='fire9')

        if cfg.USE_DROPOUT:
            net = tf.contrib.layers.dropout(net, 1-cfg.DROP_RATE, is_training=self._is_training, scope='drop10')

        return net, skip_layers

    def _build_segmenter_base(self, inputT):
        cfg = self._cfg

        in_shape = inputT.get_shape().as_list()
        self._logger.info("segmenter base input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        # parallel dilated module
        net = parallel_dilate(inputT, 256, [2,4,8,16], initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                              act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='para_dilate')

        if cfg.USE_DROPOUT:
            net = tf.contrib.layers.dropout(net, 1-cfg.DROP_RATE, is_training=self._is_training, scope='drop_segbase')

        return net

    def _build_segmenter(self, inputT, skip_layers):
        cfg = self._cfg

        in_shape = inputT.get_shape().as_list()
        self._logger.info("segmenter input shape: [" + ','.join([str(x) for x in in_shape]) + ']')


        # upsampled by deconvolution
        # deconv1
        #skip_shape = tf.shape(skip_layer)
        #deconv_out_shape = tf.stack([skip_shape[0], skip_shape[1], skip_shape[2], 256])

        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 4)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 256]

        net = deconv_layer_with_bn(inputT, [2, 2, 256, 256], deconv_out_shape, 2, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.elu, loss_collection=self._loss_collection,
                                   name="deconv1")

        # refine1
        skip_layer = skip_layers['down8x']
        net = refinement(net, skip_layer, 128, 256, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                         use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                         act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='refine1')

        # deconv2
        #skip_shape = tf.shape(skip_layer)
        #deconv_out_shape = tf.stack([skip_shape[0], skip_shape[1], skip_shape[2], 128])

        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 128]

        net = deconv_layer_with_bn(net, [2, 2, 128, 128], deconv_out_shape, 2, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.elu, loss_collection=self._loss_collection,
                                   name="deconv2")

        # refine2
        skip_layer = skip_layers['down4x']
        net = refinement(net, skip_layer, 64, 128, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                         use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                         act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='refine2')

        # deconv3
        #skip_shape = tf.shape(skip_layer)
        #deconv_out_shape = tf.stack([skip_shape[0], skip_shape[1], skip_shape[2], 64])

        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 2)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 64]

        net = deconv_layer_with_bn(net, [2, 2, 64, 64], deconv_out_shape, 2, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.elu, loss_collection=self._loss_collection,
                                   name="deconv3")

        # refine3
        skip_layer = skip_layers['down2x']
        net = refinement(net, skip_layer, 32, 64, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                         use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                         act_fn=tf.nn.elu, loss_collection=self._loss_collection, name='refine3')

        # deconv4
        #batch_size = tf.shape(inputT)[0]
        #fmap_w = cfg.IMAGE_WIDTH
        #fmap_h = cfg.IMAGE_HEIGHT

        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 1)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 32]

        #deconv_out_shape = tf.stack([batch_size, fmap_w, fmap_h, 32])

        net = deconv_layer_with_bn(net, [2, 2, 32, 32], deconv_out_shape, 2, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.elu, loss_collection=self._loss_collection,
                                   name="deconv4")

        if cfg.USE_DROPOUT:
            net = tf.contrib.layers.dropout(net, 1-cfg.DROP_RATE, is_training=self._is_training, scope='drop_segment')

        net_shape = net.get_shape().as_list()
        self._logger.info("classifier input shape: [" + ','.join([str(x) for x in net_shape]) + ']')

        conv_classifier = conv_layer_with_bn(net, [1, 1, 32, cfg.NUM_CLASSES], wd=cfg.WEIGHT_DECAY,
                                             initializer=msra_initializer(), act_fn=None, 
                                             loss_collection=self._loss_collection, name='conv_classifier')

        return conv_classifier

    def _add_forward_layer(self):
        if self._cfg.SUBTRACT_CHANNEL_MEAN:
            red, green, blue = tf.split(self._input_image, 3, 3)
            input_data = tf.concat([
                                red - self._cfg.BGR_MEAN[2],
                                green - self._cfg.BGR_MEAN[1],
                                blue - self._cfg.BGR_MEAN[0]
                            ], 3)
        else:
            input_data = self._input_image

        forward_layer, skip_layers = self._build_encoder(input_data)
        forward_layer = self._build_segmenter_base(forward_layer)
        self._segment_logit = self._build_segmenter(forward_layer, skip_layers)
        
        


