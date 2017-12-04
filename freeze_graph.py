import os
import sys
import argparse

import tensorflow as tf
#from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util

from nets.sq_net import SQSegNet
from nets.erf_net import ERFSegNet
from config import cfg
from utils import log_helper
from utils.config_utils import cfg_from_file, print_config



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'erf', """ which model to use """)
tf.app.flags.DEFINE_string('ckpt_path', '', """ path to restore model ckpt """)
tf.app.flags.DEFINE_string('cfg_file', '', """ experimental config file """)
tf.app.flags.DEFINE_string('output_name', '', """ name of output graph """)
tf.app.flags.DEFINE_string('output_path', '', """ dir to output graph """)
tf.app.flags.DEFINE_boolean('restore_avg', True, """ whether to restore moving avg version """)
tf.app.flags.DEFINE_boolean('whole_graph_bin', False, """ whether binary format for output whole graph """)
tf.app.flags.DEFINE_boolean('infer_graph_bin', True, """ whether binary format for output infer graph """)

def main(args):
    if FLAGS.cfg_file:
        print('loading config setting')
        cfg_from_file(FLAGS.cfg_file, cfg)
    cfg.BATCH_SIZE = 1
    print_config(cfg)

    output_path = FLAGS.output_path
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    batch_size = 1
    image_h = cfg.IMAGE_HEIGHT
    image_w = cfg.IMAGE_WIDTH
    image_c = cfg.IMAGE_DEPTH
    output_name = FLAGS.output_name

    whole_graph_ext = 'pb' if FLAGS.whole_graph_bin else 'pbtxt'
    infer_graph_ext = 'pb' if FLAGS.infer_graph_bin else 'pbtxt'
    whole_graph_name = "{}_whole.{}".format(output_name, whole_graph_ext)
    infer_graph_name = "{}_infer.{}".format(output_name, whole_graph_ext)
    uff_graph_name = "{}_uff.{}".format(output_name, whole_graph_ext)
    output_graph_path = "{}/{}.{}".format(output_path, output_name, infer_graph_ext)
    output_uff_graph_path = "{}/{}_uff.{}".format(output_path, output_name, infer_graph_ext)
    print whole_graph_name
    print infer_graph_name
    print uff_graph_name
    print output_graph_path
    print output_uff_graph_path

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # Build graph
    logger = log_helper.get_logger()
    if FLAGS.model == 'sq':
        model = SQSegNet(cfg, logger)
    elif FLAGS.model == 'erf':
        model = ERFSegNet(cfg, logger)

    output_node_names = "output/prob"


    if FLAGS.restore_avg:
        # get moving avg
        variable_averages = tf.train.ExponentialMovingAverage(cfg.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    else:
        saver = tf.train.Saver(model.all_variables)
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        # Load checkpoint
        whole_graph_def = sess.graph.as_graph_def()

        # fix whole_graph_def for bn
        for node in whole_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        print("%d ops in the whole graph." % len(whole_graph_def.node))

        tf.train.write_graph(whole_graph_def, output_path,
                             whole_graph_name, as_text=not FLAGS.whole_graph_bin)

        infer_graph_def = graph_util.extract_sub_graph(whole_graph_def, output_node_names.split(","))
        print("%d ops in the infer graph." % len(infer_graph_def.node))

        tf.train.write_graph(infer_graph_def, output_path,
                             infer_graph_name, as_text=not FLAGS.whole_graph_bin)


        # fix infer_graph_def for bn for converstion to tensorRT uff
        for node in infer_graph_def.node:
            name_fields = node.name.split('/')
            if name_fields[-2] == 'batchnorm':
                if name_fields[-1] == 'add':
                    for index in xrange(len(node.input)):
                        if 'cond/Merge' in node.input[index]:
                            node.input[index] = '/'.join(name_fields[:-2] + ['moving_variance', 'read'])
                if name_fields[-1] == 'mul_2':
                    for index in xrange(len(node.input)):
                        if 'cond/Merge' in node.input[index]:
                            node.input[index] = '/'.join(name_fields[:-2] + ['moving_mean', 'read'])

        uff_graph_def = graph_util.extract_sub_graph(infer_graph_def, output_node_names.split(","))
        print("%d ops in the uff graph." % len(uff_graph_def.node))

        tf.train.write_graph(uff_graph_def, output_path,
                             uff_graph_name, as_text=not FLAGS.whole_graph_bin)

        saver.restore(sess, FLAGS.ckpt_path)

        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            whole_graph_def, # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        output_uff_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            infer_graph_def, # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        mode = "wb" if FLAGS.infer_graph_bin else "w"
        with tf.gfile.GFile(output_graph_path, mode) as f:
            if FLAGS.infer_graph_bin:
                f.write(output_graph_def.SerializeToString())
            else:
                f.write(str(output_graph_def))

        print("%d ops in the output graph." % len(output_graph_def.node))

        with tf.gfile.GFile(output_uff_graph_path, mode) as f:
            if FLAGS.infer_graph_bin:
                f.write(output_uff_graph_def.SerializeToString())
            else:
                f.write(str(output_uff_graph_def))

        print("%d ops in the output uff graph." % len(output_uff_graph_def.node))


if __name__ == '__main__':
    tf.app.run()

