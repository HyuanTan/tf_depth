from easydict import EasyDict as edict
import numpy as np

cfg = edict()

# training process config
cfg.MOVING_AVERAGE_DECAY = 0.999     # The decay to use for the moving average.

cfg.INIT_LEARNING_RATE = 1e-3      # Initial learning rate.
cfg.LR_DECAY = 'piecewise'
cfg.LR_DECAY_STEP = 54800
cfg.LR_DECAY_FACTOR = [0.5, 0.25]
cfg.LR_DECAY_BOUNDARY = [0.6, 0.8]

cfg.BATCH_SIZE = 8
cfg.TRAIN_QUEUE_CAPACITY = 200
cfg.NUM_EXAMPLES = 21364
cfg.NUM_EPOCH = 50

cfg.DO_VALIDATE = True

cfg.VAL_BATCH_SIZE = cfg.BATCH_SIZE
cfg.EVAL_QUEUE_CAPACITY = 50
cfg.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200

cfg.WEIGHT_DECAY = None

cfg.USE_BN = True
cfg.BN_DECAY = 0.999

cfg.USE_DROPOUT = False
cfg.DROP_RATE = 0.3

cfg.GRAD_SUMMARY = True

# model params
cfg.SCALE_NUM = 4
cfg.DISP_FACTOR = 0.3

cfg.USE_SSIM = True
cfg.ALPHA_IMAGE_LOSS = 0.85

cfg.SMOOTH_LOSS_WEIGHT = 0.1

cfg.USE_CONSIST = True
cfg.CONSIST_LOSS_WEIGHT = 1.0

cfg.DO_STEREO = False

# image config
cfg.IMAGE_HEIGHT = 192
cfg.IMAGE_WIDTH = 576
cfg.IMAGE_DEPTH = 3


