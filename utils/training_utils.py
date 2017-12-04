#!/usr/bin/python

import tensorflow as tf

def get_train_op(cfg, total_loss, global_step, max_step=None):
    init_lr = cfg.INIT_LEARNING_RATE

    if cfg.LR_DECAY == 'exponential':
        lr = tf.train.exponential_decay(init_lr, global_step,
                cfg.LR_DECAY_STEP, cfg.LR_DECAY_FACTOR, staircase=True, name='learning_rate')
    elif cfg.LR_DECAY == 'piecewise':
        boundaries = [int(x * max_step) for x in cfg.LR_DECAY_BOUNDARY]
        values = [init_lr] + [init_lr * x for x in cfg.LR_DECAY_FACTOR]
        lr = tf.train.piecewise_constant(global_step, boundaries, values, name='learning_rate')
    else:
        lr = init_lr

    # Compute gradients.
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histogram summaries for gradients.
    if cfg.GRAD_SUMMARY:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cfg.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Track the moving average of bn mean/stdvar
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops + [apply_gradient_op, variables_averages_op]):
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op, grads

def add_summaries(var_list, mode, cfg=None):
    summary_ops = []
    for var in var_list:
        if mode == 'scala':
            summary_ops.append(tf.summary.scalar(var.op.name, var))
        elif mode == 'hist':
            summary_ops.append(tf.summary.histogram(var.op.name, var))
        elif mode == 'image':
            max_show = cfg.BATCH_SIZE if cfg else 3
            summary_ops.append(tf.summary.image(var.op.name, var, max_show))

    return summary_ops

def add_metric_summary(name):
    metric_summary_ph = tf.placeholder(tf.float32)
    summary_op = tf.summary.scalar(name, metric_summary_ph)
    return summary_op, metric_summary_ph

