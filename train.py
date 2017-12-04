#!/usr/bin/python

import tensorflow as tf
import sys
import os
import time
import numpy as np
np.set_printoptions(threshold=np.nan)

from nets.res50_disp_net import Res50DispNet
from data.data_loader import TFLoadingPipeline, LoadingPipeline
from config import cfg
from utils.training_utils import get_train_op, add_summaries, add_metric_summary
from utils.config_utils import cfg_from_file, print_config
#from utils.metrics import per_class_iu, per_class_iu_from_hist, get_class_hist
from utils import log_helper



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'res50', """ which model to use """)
tf.app.flags.DEFINE_string('ckpt_path', './log', """ dir to store ckpt """)
tf.app.flags.DEFINE_string('cfg_file', '', """ experimental config file """)
tf.app.flags.DEFINE_string('sample_path', '', """ path to sample image """)
tf.app.flags.DEFINE_string('label_path', '', """ path to label image """)
tf.app.flags.DEFINE_string('val_sample_path', '', """ path to validate sample image """)
tf.app.flags.DEFINE_string('val_label_path', '', """ path to validate label image """)
tf.app.flags.DEFINE_boolean('debug', False, """ if in debug mode """)



def checkArgs():
    print('Training model: {}'.format(FLAGS.model))
    print("Sample image path: {}".format(FLAGS.sample_path))
    print("Label image path: {}".format(FLAGS.label_path))
    print("Validation sample path: {}".format(FLAGS.val_sample_path))
    print("Validation label path: {}".format(FLAGS.val_label_path))
    print("Model checkpoint path: {}".format(FLAGS.ckpt_path))
    print("Config file: {}".format(FLAGS.cfg_file))

def main(args):
    checkArgs()

    if FLAGS.cfg_file:
        print('loading config setting')
        cfg_from_file(FLAGS.cfg_file, cfg)
    if FLAGS.debug:
        cfg.BATCH_SIZE = 2
        cfg.NUM_EPOCH = 1
        cfg.TRAIN_QUEUE_CAPACITY = 10
        cfg.DO_VALIDATE = False
        print('set to DEBUG mode')
    print_config(cfg)

    if cfg.DO_VALIDATE:
        val_step = np.ceil(float(cfg.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL) / cfg.VAL_BATCH_SIZE).astype(np.int32)
    steps_per_epoch = np.ceil(float(cfg.NUM_EXAMPLES) / cfg.BATCH_SIZE).astype(np.int32)
    max_step = steps_per_epoch * cfg.NUM_EPOCH
    if FLAGS.debug:
        max_step = 1000
    ckpt_path = FLAGS.ckpt_path
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    logger = log_helper.get_logger()
    if FLAGS.model == 'res50':
        model = Res50DispNet(cfg, logger)
    else:
        logger.error('wrong model type: {}'.format(FLAGS.model))
        sys.exit(-1)

    data_pipeline = TFLoadingPipeline(cfg, logger)
    data_pipeline.setup(FLAGS.sample_path, FLAGS.label_path, cfg.BATCH_SIZE,
                        cfg.TRAIN_QUEUE_CAPACITY, augment=True)
    if cfg.DO_VALIDATE:
        data_pipeline.setup_validate(FLAGS.val_sample_path, FLAGS.val_label_path,
                                     cfg.VAL_BATCH_SIZE, cfg.EVAL_QUEUE_CAPACITY)

    global_step = tf.Variable(0, trainable=False)

    train_op, grads = get_train_op(cfg, model.total_loss, global_step, max_step)

    if FLAGS.debug:
        nan_checks = []
        for grad, var in grads:
            nan_checks.append(tf.is_nan(grad))

        layers = model.layers[-8:]
        layers += model.left_reconstruction + model.right_reconstruction + model.right_to_left_disp + model.left_to_right_disp + model.left_disp_smoothness + model.right_disp_smoothness + model.left_disp_smoothness + model.right_disp_smoothness
        if cfg.WEIGHT_DECAY is not None:
            _, img_loss, img_l1_loss, img_ssim_loss, smth_loss, cons_loss, total_loss = model.model_losses
        else:
            img_loss, img_l1_loss, img_ssim_loss, smth_loss, cons_loss, total_loss = model.model_losses
        layer_grads_total = tf.gradients(total_loss, layers)
        layer_grads_img = tf.gradients(img_loss, layers)
        layer_grads_img_l1 = tf.gradients(img_l1_loss, layers)
        layer_grads_img_ssim = tf.gradients(img_ssim_loss, layers)
        layer_grads_smth = tf.gradients(smth_loss, layers)
        layer_grads_cons = tf.gradients(cons_loss, layers)
        layer_n_grads = zip(layers, layer_grads_img, layer_grads_img_l1, layer_grads_img_ssim, layer_grads_smth, layer_grads_cons, layer_grads_total)

    saver = tf.train.Saver(var_list=model.all_variables, max_to_keep=10)

    # add summaries
    images = [model.left_image, model.right_image, model.left_disparity[0], model.right_disparity[0],
              model.left_reconstruction[0], model.right_reconstruction[0], model.left_recon_diff[0],
              model.right_recon_diff[0]]
    add_summaries(images, 'image', cfg)
    add_summaries(model.trainables, 'hist')
    add_summaries(model.bn_variables, 'hist')
    add_summaries(model.bn_mean_variance, 'hist')
    add_summaries(model.losses, 'scala')
    summary_op = tf.summary.merge_all()

    if cfg.DO_VALIDATE:
        # add evaluation metric summaries
        val_loss_summary_op, val_loss_summary_ph = add_metric_summary('val_loss')
        metric_summary_op = tf.summary.merge([val_loss_summary_op])

    with tf.Session() as sess:
        total_start_time = time.time()
        summary_writer = tf.summary.FileWriter(ckpt_path, sess.graph)

        # initialize
        logger.info('initializing model params...')
        sess.run(model.initializer)

        data_pipeline.start(sess)
        start_time = time.time()
        for step in xrange(max_step):

            image_batch, label_batch = data_pipeline.load_batch()
            feed_dict = {
                model.left_image: image_batch,
                model.train_phase: True,
                model.right_image: label_batch
            }

            # run training
            if FLAGS.debug:
                nan_check_res = sess.run(nan_checks, feed_dict=feed_dict)
                for i,res in enumerate(nan_check_res[::-1]):
                    if res.any():
                        grad, var = grads[::-1][i]
                        print('{}\'s gradients has nan'.format(var.op.name))
                        grad_res, var_res = sess.run([grad, var], feed_dict=feed_dict)
                        print grad_res.shape
                        print var_res.shape
                        print('========================================')
                        print(grad_res)
                        print('========================================')
                        print(var_res)
                        print('========================================')
                        #print('all grads & vars:')
                        #print('========================================')
                        #for grad, var in grads:
                            #grad_res, var_res = sess.run([grad, var], feed_dict=feed_dict)
                            #print('{} gradient'.format(var.op.name))
                            #print(grad_res)
                            #print('========================================')
                            #print(var.op.name)
                            #print(var_res)
                        print('all layers & grads:')
                        for var, grad_img, grad_img_l1, grad_img_ssim, grad_smth, grad_cons, grad_ttl in layer_n_grads:
                            print(var.op.name)
                            var_res = sess.run(var, feed_dict=feed_dict)
                            print(var_res.shape)
                            print(var_res)
                            print('========================================')
                            if grad_img is not None:
                                grad_img_res = sess.run(grad_img, feed_dict=feed_dict)
                                print('{} gradient img'.format(var.op.name))
                                print(grad_img_res.shape)
                                print(grad_img_res)
                                print('========================================')
                            if grad_img_l1 is not None:
                                grad_img_l1_res = sess.run(grad_img_l1, feed_dict=feed_dict)
                                print('{} gradient img l1'.format(var.op.name))
                                print(grad_img_l1_res.shape)
                                print(grad_img_l1_res)
                                print('========================================')
                            if grad_img_ssim is not None:
                                grad_img_ssim_res = sess.run(grad_img_ssim, feed_dict=feed_dict)
                                print('{} gradient img ssim'.format(var.op.name))
                                print(grad_img_ssim_res.shape)
                                print(grad_img_ssim_res)
                                print('========================================')

                            if grad_smth is not None:
                                grad_smth_res = sess.run(grad_smth, feed_dict=feed_dict)
                                print('{} gradient smth'.format(var.op.name))
                                print(grad_smth_res.shape)
                                print(grad_smth_res)
                                print('========================================')
                            if grad_cons is not None:
                                grad_cons_res = sess.run(grad_cons, feed_dict=feed_dict)
                                print('{} gradient cons'.format(var.op.name))
                                print(grad_cons_res.shape)
                                print(grad_cons_res)
                                print('========================================')
                            if grad_ttl is not None:
                                grad_ttl_res = sess.run(grad_ttl, feed_dict=feed_dict)
                                print('{} gradient total'.format(var.op.name))
                                print(grad_ttl_res.shape)
                                print(grad_ttl_res)
                                print('========================================')

                        print('all losses:')
                        for loss in model.losses:
                            loss_res = sess.run(loss, feed_dict=feed_dict)
                            print('{}: {}'.format(loss.op.name, loss_res))

                        sys.exit(-1)

            _, loss, summary_str = sess.run([train_op, model.total_loss, summary_op],
                                            feed_dict=feed_dict)

            if FLAGS.debug:
                summary_writer.add_summary(summary_str, step)

            # every 10 step, output metrics
            if step and step % 10 == 0:
                #pred, loss = sess.run([model.output, model.total_loss], feed_dict=feed_dict)

                duration = time.time() - start_time
                logger.info('step {}: {} sec elapsed, loss = {}'.format(step, duration, loss))

                start_time = time.time()
            # every 100 step, do validation & write summary
            if not FLAGS.debug and step and step % 100 == 0:
                # write summary
                #summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

                if cfg.DO_VALIDATE:
                    logger.info('start validating......')
                    total_val_loss = 0.0
                    for test_step in range(val_step):
                        val_image_batch, val_label_batch = data_pipeline.load_validate_batch()
                        val_feed_dict = {
                            model.left_image: val_image_batch,
                            model.right_image: val_label_batch
                        }

                        val_loss = sess.run(model.total_loss, feed_dict=val_feed_dict)
                        total_val_loss += val_loss

                    avg_loss = total_val_loss / val_step
                    logger.info("val loss: {}".format(avg_loss))

                    metric_summary_str = sess.run(metric_summary_op,
                                                  feed_dict={val_loss_summary_ph: avg_loss})
                    summary_writer.add_summary(metric_summary_str, step)

                    logger.info(" end validating.... ")

            # every 10000 steps, save the model checkpoint
            if (step and step % 10000 == 0) or (step + 1) == max_step:
                checkpoint_path = os.path.join(ckpt_path, '{}_model.ckpt'.format(FLAGS.model))
                saver.save(sess, checkpoint_path, global_step=step)

        # Done training
        logger.info('training complete')
        data_pipeline.shutdown()
        total_end_time = time.time()
        logger.info('total time elapsed: {} h'.format((total_end_time - total_start_time)/3600.0))


if __name__ == '__main__':
    tf.app.run()
