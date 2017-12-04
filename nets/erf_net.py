import tensorflow as tf
from seg_model import SegModel
from utils.building_blocks import deconv_layer_with_bn, get_deconv_output_fmap_shape, downsampler, res_nonbt1d
from utils.variable_utils import orthogonal_initializer, msra_initializer


class ERFSegNet(SegModel):
    def __init__(self, cfg, logger):
        super(ERFSegNet, self).__init__(cfg, logger)

    def _build_encoder(self, inputT):
        cfg = self._cfg

        in_shape = inputT.get_shape().as_list()
        self._logger.info("encoder input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        # downsampler1
        net = downsampler(inputT, 16, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          loss_collection=self._loss_collection, name='l1_downsampler')
        # downsampler2
        net = downsampler(net, 64, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          loss_collection=self._loss_collection, name='l2_downsampler')
        # nonbt1d_3
        net = res_nonbt1d(net, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l3_nonbt1d')
        # nonbt1d_4
        net = res_nonbt1d(net, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l4_nonbt1d')
        # nonbt1d_5
        net = res_nonbt1d(net, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l5_nonbt1d')
        # nonbt1d_6
        net = res_nonbt1d(net, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l6_nonbt1d')
        # nonbt1d_7
        net = res_nonbt1d(net, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l7_nonbt1d')
        # downsampler8
        net = downsampler(net, 128, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          loss_collection=self._loss_collection, name='l8_downsampler')
        # nonbt1d_9
        net = res_nonbt1d(net, 3, True, 2, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l9_nonbt1d')
        # nonbt1d_10
        net = res_nonbt1d(net, 3, True, 4, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l10_nonbt1d')
        # nonbt1d_11
        net = res_nonbt1d(net, 3, True, 8, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l11_nonbt1d')
        # nonbt1d_12
        net = res_nonbt1d(net, 3, True, 16, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l12_nonbt1d')
        # nonbt1d_13
        net = res_nonbt1d(net, 3, True, 2, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l13_nonbt1d')
        # nonbt1d_14
        net = res_nonbt1d(net, 3, True, 4, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l14_nonbt1d')
        # nonbt1d_15
        net = res_nonbt1d(net, 3, True, 8, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l15_nonbt1d')
        # nonbt1d_16
        net = res_nonbt1d(net, 3, True, 16, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l16_nonbt1d')

        return net


    def _build_segmenter(self, inputT):
        cfg = self._cfg

        in_shape = inputT.get_shape().as_list()
        self._logger.info("segmenter input shape: [" + ','.join([str(x) for x in in_shape]) + ']')

        # upsampled by deconvolution
        # deconv17
        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 64]

        net = deconv_layer_with_bn(inputT, [2, 2, 64, 128], deconv_out_shape, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.relu, loss_collection=self._loss_collection,
                                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                                   name="l17_deconv")

        # nonbt1d_18
        net = res_nonbt1d(net, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l18_nonbt1d')
        # nonbt1d_19
        net = res_nonbt1d(net, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l19_nonbt1d')

        # deconv20
        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 2)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, 32]

        net = deconv_layer_with_bn(net, [2, 2, 32, 64], deconv_out_shape, initializer=msra_initializer(),
                                   wd=cfg.WEIGHT_DECAY, act_fn=tf.nn.relu, loss_collection=self._loss_collection,
                                   use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                                   name="l20_deconv")

        # nonbt1d_21
        net = res_nonbt1d(net, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l21_nonbt1d')
        # nonbt1d_22
        net = res_nonbt1d(net, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                          use_dropout=cfg.USE_DROPOUT, drop_rate=cfg.DROP_RATE,
                          loss_collection=self._loss_collection, name='l22_nonbt1d')

        # deconv23: classifier
        fmap_h, fmap_w = get_deconv_output_fmap_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 1)
        deconv_out_shape = [cfg.BATCH_SIZE, fmap_h, fmap_w, cfg.NUM_CLASSES]

        classifier = deconv_layer_with_bn(net, [2, 2, cfg.NUM_CLASSES, 32], deconv_out_shape,
                                          initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, 
                                          use_bn=cfg.USE_BN, is_training=self._is_training, bn_decay=cfg.BN_DECAY,
                                          loss_collection=self._loss_collection, name="l23_classifier")

        return classifier

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

        forward_layer = self._build_encoder(input_data)
        self._segment_logit = self._build_segmenter(forward_layer)
        
        


