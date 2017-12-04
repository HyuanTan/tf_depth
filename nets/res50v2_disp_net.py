import tensorflow as tf
from disp_model import DispModel
from utils.building_blocks import deconv_layer_with_bn, get_deconv_output_fmap_shape, resbottlev2, upsample_layer, upconv_layer_with_bn, conv_layer_with_bn, merge_skip
from utils.variable_utils import orthogonal_initializer, msra_initializer


class Res50v2DispNet(DispModel):
    def __init__(self, cfg, logger):
        super(Res50DispNet, self).__init__(cfg, logger)

    def _build_encoder(self, inputT):
        cfg = self._cfg

        in_shape = inputT.get_shape().as_list()
        self._logger.info("encoder input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        skip_layers = []

        # conv7X7_1 H/2
        self._logger.info("l1_conv input shape: [" + ','.join([str(x) for x in in_shape]) + ']')
        net = conv_layer_with_bn(inputT, [7,7,in_shape[3],64], 2, initializer=msra_initializer(),
                                 wd=cfg.WEIGHT_DECAY, loss_collection=self._loss_collection,
                                 name='l1_conv')

        self._layers.append(net)

        skip_layers.append(net)

        # maxpool_2 H/4
        in_shape = net.get_shape().as_list()
        self._logger.info("l2_pool input shape: [{}]".format(','.join([str(x) for x in in_shape])))
        net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='l2_pool')

        self._layers.append(net)

        skip_layers.append(net)

        # resbottle_3 H/4
        net = resbottlev2(net, 64, 64, 256, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l3_resbottle')
        self._layers.append(net)

        # resbottle_4 H/4
        net = resbottlev2(net, 64, 64, 256, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l4_resbottle')
        self._layers.append(net)

        # resbottle_5 H/8
        net = resbottlev2(net, 64, 64, 256, 2, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l5_resbottle')
        self._layers.append(net)

        skip_layers.append(net)

        # resbottle_6 H/8
        net = resbottlev2(net, 128, 128, 512, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l6_resbottle')
        self._layers.append(net)

        # resbottle_7 H/8
        net = resbottlev2(net, 128, 128, 512, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l7_resbottle')
        self._layers.append(net)

        # resbottle_8 H/8
        net = resbottlev2(net, 128, 128, 512, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l8_resbottle')
        self._layers.append(net)

        # resbottle_9 H/16
        net = resbottlev2(net, 128, 128, 512, 2, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l9_resbottle')
        self._layers.append(net)

        skip_layers.append(net)

        # resbottle_10 H/16
        net = resbottlev2(net, 256, 256, 1024, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l10_resbottle')
        self._layers.append(net)

        # resbottle_11 H/16
        net = resbottlev2(net, 256, 256, 1024, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l11_resbottle')
        self._layers.append(net)

        # resbottle_12 H/16
        net = resbottlev2(net, 256, 256, 1024, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l12_resbottle')
        self._layers.append(net)

        # resbottle_13 H/16
        net = resbottlev2(net, 256, 256, 1024, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l13_resbottle')
        self._layers.append(net)

        # resbottle_14 H/16
        net = resbottlev2(net, 256, 256, 1024, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l14_resbottle')
        self._layers.append(net)

        # resbottle_15 H/32
        net = resbottlev2(net, 256, 256, 1024, 2, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l15_resbottle')
        self._layers.append(net)

        skip_layers.append(net)

        # resbottle_16 H/32
        net = resbottlev2(net, 512, 512, 2048, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l16_resbottle')
        self._layers.append(net)

        # resbottle_17 H/32
        net = resbottlev2(net, 512, 512, 2048, 1, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l17_resbottle')
        self._layers.append(net)

        # resbottle_18 H/64
        net = resbottlev2(net, 512, 512, 2048, 2, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                        use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                        use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE, act_fn=tf.nn.elu,
                        loss_collection=self._loss_collection, name='l18_resbottle')
        self._layers.append(net)

        return net, skip_layers


    def _build_decoder(self, inputT, skip_layers):
        cfg = self._cfg
        down_pow = len(skip_layers) + 1 # downsample 6x
        self._disp_pyramid = [None] * cfg.SCALE_NUM
        updisp = None

        in_shape = inputT.get_shape().as_list()
        self._logger.info("decoder input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        # upsample_19
        self._logger.info("l19_deconv input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 512]

        net = deconv_layer_with_bn(inputT, [2, 2, 512, 2048], deconv_out_shape, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.elu, loss_collection=self._loss_collection,
                                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                                   name="l19_deconv")   # H/32
        down_pow -= 1 # downsample 5x
        self._layers.append(net)

        # merge_skip_20
        skip_layer = skip_layers[down_pow-1] if down_pow > 0 else None
        inputs = [net]
        if skip_layer is not None:
            inputs.append(skip_layer)
        if updisp is not None:
            inputs.append(updisp)
        net = merge_skip(inputs, 512, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                         act_fn=tf.nn.elu, loss_collection=self._loss_collection, use_bn=cfg.USE_BN,
                         is_training=self._is_training, bn_decay=cfg.BN_DECAY, name="l20_merge_skip")
        self._layers.append(net)

        if down_pow < cfg.SCALE_NUM:
            scale_disp = self._get_disp(net, down_pow)
            self._disp_pyramid[down_pow] = scale_disp
            self._layers.append(scale_disp)
            if down_pow > 0:
                up_h, up_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
                updisp = upsample_layer(scale_disp, up_h, up_w, 'updisp_s{}'.format(down_pow-1))

        # upsample_21
        in_shape = net.get_shape().as_list()
        self._logger.info("l21_deconv input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 256]

        net = deconv_layer_with_bn(net, [2, 2, 256, 512], deconv_out_shape, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.elu, loss_collection=self._loss_collection,
                                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                                   name="l21_deconv")   # H/16
        down_pow -= 1 # downsample 4x
        self._layers.append(net)

        # merge_skip_22
        skip_layer = skip_layers[down_pow-1] if down_pow > 0 else None
        inputs = [net]
        if skip_layer is not None:
            inputs.append(skip_layer)
        if updisp is not None:
            inputs.append(updisp)
        net = merge_skip(inputs, 256, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                         act_fn=tf.nn.elu, loss_collection=self._loss_collection, use_bn=cfg.USE_BN,
                         is_training=self._is_training, bn_decay=cfg.BN_DECAY, name="l22_merge_skip")
        self._layers.append(net)

        if down_pow < cfg.SCALE_NUM:
            scale_disp = self._get_disp(net, down_pow)
            self._disp_pyramid[down_pow] = scale_disp
            self._layers.append(scale_disp)
            if down_pow > 0:
                up_h, up_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
                updisp = upsample_layer(scale_disp, up_h, up_w, 'updisp_s{}'.format(down_pow-1))

        # upsample_23
        in_shape = net.get_shape().as_list()
        self._logger.info("l23_deconv input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 128]

        net = deconv_layer_with_bn(net, [2, 2, 128, 256], deconv_out_shape, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.elu, loss_collection=self._loss_collection,
                                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                                   name="l23_deconv")   # H/8
        down_pow -= 1 # downsample 3x
        self._layers.append(net)

        # merge_skip_24
        skip_layer = skip_layers[down_pow-1] if down_pow > 0 else None
        inputs = [net]
        if skip_layer is not None:
            inputs.append(skip_layer)
        if updisp is not None:
            inputs.append(updisp)
        net = merge_skip(inputs, 128, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                         act_fn=tf.nn.elu, loss_collection=self._loss_collection, use_bn=cfg.USE_BN,
                         is_training=self._is_training, bn_decay=cfg.BN_DECAY, name="l24_merge_skip")
        self._layers.append(net)

        if down_pow < cfg.SCALE_NUM:
            scale_disp = self._get_disp(net, down_pow)
            self._disp_pyramid[down_pow] = scale_disp
            self._layers.append(scale_disp)
            if down_pow > 0:
                up_h, up_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
                updisp = upsample_layer(scale_disp, up_h, up_w, 'updisp_s{}'.format(down_pow-1))

        # upsample_25
        in_shape = net.get_shape().as_list()
        self._logger.info("l25_deconv input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 64]

        net = deconv_layer_with_bn(net, [2, 2, 64, 128], deconv_out_shape, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.elu, loss_collection=self._loss_collection,
                                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                                   name="l25_deconv")   # H/4
        down_pow -= 1 # downsample 2x
        self._layers.append(net)

        # merge_skip_26
        skip_layer = skip_layers[down_pow-1] if down_pow > 0 else None
        inputs = [net]
        if skip_layer is not None:
            inputs.append(skip_layer)
        if updisp is not None:
            inputs.append(updisp)
        net = merge_skip(inputs, 64, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                         act_fn=tf.nn.elu, loss_collection=self._loss_collection, use_bn=cfg.USE_BN,
                         is_training=self._is_training, bn_decay=cfg.BN_DECAY, name="l26_merge_skip")
        self._layers.append(net)

        if down_pow < cfg.SCALE_NUM:
            scale_disp = self._get_disp(net, down_pow)
            self._disp_pyramid[down_pow] = scale_disp
            self._layers.append(scale_disp)
            if down_pow > 0:
                up_h, up_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
                updisp = upsample_layer(scale_disp, up_h, up_w, 'updisp_s{}'.format(down_pow-1))

        # upsample_27
        in_shape = net.get_shape().as_list()
        self._logger.info("l27_deconv input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 32]

        net = deconv_layer_with_bn(net, [2, 2, 32, 64], deconv_out_shape, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.elu, loss_collection=self._loss_collection,
                                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                                   name="l27_deconv")   # H/2
        down_pow -= 1 # downsample 1x
        self._layers.append(net)

        # merge_skip_28
        skip_layer = skip_layers[down_pow-1] if down_pow > 0 else None
        inputs = [net]
        if skip_layer is not None:
            inputs.append(skip_layer)
        if updisp is not None:
            inputs.append(updisp)
        net = merge_skip(inputs, 32, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                         act_fn=tf.nn.elu, loss_collection=self._loss_collection, use_bn=cfg.USE_BN,
                         is_training=self._is_training, bn_decay=cfg.BN_DECAY, name="l28_merge_skip")
        self._layers.append(net)

        if down_pow < cfg.SCALE_NUM:
            scale_disp = self._get_disp(net, down_pow)
            self._disp_pyramid[down_pow] = scale_disp
            self._layers.append(scale_disp)
            if down_pow > 0:
                up_h, up_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
                updisp = upsample_layer(scale_disp, up_h, up_w, 'updisp_s{}'.format(down_pow-1))

        # upsample_29
        in_shape = net.get_shape().as_list()
        self._logger.info("l29_deconv input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 16]

        net = deconv_layer_with_bn(net, [2, 2, 16, 32], deconv_out_shape, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.elu, loss_collection=self._loss_collection,
                                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                                   name="l29_deconv")   # H
        down_pow -= 1 # downsample 0x
        self._layers.append(net)

        # merge_skip_30
        skip_layer = skip_layers[down_pow-1] if down_pow > 0 else None
        inputs = [net]
        if skip_layer is not None:
            inputs.append(skip_layer)
        if updisp is not None:
            inputs.append(updisp)
        net = merge_skip(inputs, 16, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                         act_fn=tf.nn.elu, loss_collection=self._loss_collection, use_bn=cfg.USE_BN,
                         is_training=self._is_training, bn_decay=cfg.BN_DECAY, name="l30_merge_skip")
        self._layers.append(net)

        if down_pow < cfg.SCALE_NUM:
            scale_disp = self._get_disp(net, down_pow)
            self._disp_pyramid[down_pow] = scale_disp
            self._layers.append(scale_disp)
            if down_pow > 0:
                up_h, up_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, down_pow)
                updisp = upsample_layer(scale_disp, up_h, up_w, 'updisp_s{}'.format(down_pow-1))

    def _add_forward_layer(self):
        self._layers = []

        if self._cfg.DO_STEREO:
            input_data = tf.concat([self._left, self._right], 3)
        else:
            input_data = self._left

        forward_layer, skip_layers = self._build_encoder(input_data)
        self._build_decoder(forward_layer, skip_layers)
        
    def _get_disp(self, inputT, scale):
        cfg = self._cfg

        in_shape = inputT.get_shape().as_list()
        self._logger.info("disp_s{} input shape: [".format(scale) + ','.join([str(x) for x in in_shape]) + ']')

        net = conv_layer_with_bn(inputT, [3,3,in_shape[3],2], initializer=msra_initializer(),
                                 wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.sigmoid,
                                 loss_collection=self._loss_collection, name='disp_s{}'.format(scale))
        disp = cfg.DISP_FACTOR * net

        return disp

       


