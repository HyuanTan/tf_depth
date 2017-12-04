import tensorflow as tf
import cv2
import numpy as np

import os
import time

from config import cfg
from utils.config_utils import cfg_from_file, print_config
from data.data_loader import instance_generator
import data.image_process as image_process
from utils import log_helper

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('graph_name', '', """ path to graph pb """)
tf.app.flags.DEFINE_string('cfg_file', '', """ experimental config file """)
tf.app.flags.DEFINE_string('sample_path', '', """ path to sample image """)
tf.app.flags.DEFINE_string('output_path', '', """ path to validate label image """)
tf.app.flags.DEFINE_string('mask_path', '', """ path to validate label image """)
tf.app.flags.DEFINE_boolean('verbose', False, """ whether to log out the op information in graph """)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name=None, 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

def genPredProb(image, num_classes):
    """ store label data to colored image """
    im = np.zeros((image.shape[0], image.shape[1], 3))
    for i in range(num_classes - 1):
        im[:,:,0] += image[:,:,i+1] * cfg.COLOR_CODE[i][0]
        im[:,:,1] += image[:,:,i+1] * cfg.COLOR_CODE[i][1]
        im[:,:,2] += image[:,:,i+1] * cfg.COLOR_CODE[i][2]
        #r = np.zeros_like(prob_img)
        #g = prob_img.copy() * 255
        #b = np.zeros_like(prob_img)
        #rgb = np.zeros((image.shape[0], image.shape[1], 3))
        #rgb[:,:,0] = r
        #rgb[:,:,1] = g
        #rgb[:,:,2] = b
    im = np.uint8(im)
    return im

def main(args):

    if FLAGS.cfg_file:
        print('loading config setting')
        cfg_from_file(FLAGS.cfg_file, cfg)
    print_config(cfg)

    output_path = FLAGS.output_path
    mask_path = FLAGS.mask_path
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if not os.path.isdir(mask_path):
        os.makedirs(mask_path)

    image_h = cfg.IMAGE_HEIGHT
    image_w = cfg.IMAGE_WIDTH

    logger = log_helper.get_logger()

    # We use our "load_graph" function
    logger.info("accessing tf graph")
    graph = load_graph(FLAGS.graph_name)

    if FLAGS.verbose:
        # We can verify that we can access the list of operations in the graph
        for op in graph.get_operations():
            logger.info(op.name)
            # prefix/Placeholder/inputs_placeholder
            # ...
            # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
    input_img = graph.get_tensor_by_name('import/input/image:0')
    pred = graph.get_tensor_by_name('import/output/prob:0')

    # launch a Session
    with tf.Session(graph=graph) as sess:

        total_time_elapsed = 0.0

        for image, fname in instance_generator(FLAGS.sample_path):
            logger.info("predicting for {}".format(fname))

            begin_ts = time.time()
            feed_dict = {
                input_img: image[np.newaxis],
            }

            # Note: we didn't initialize/restore anything, everything is stored in the graph_def
            prediction = sess.run(pred, feed_dict=feed_dict)
            end_ts = time.time()
            logger.info("cost time: {} s".format(end_ts - begin_ts))
            total_time_elapsed += end_ts - begin_ts

            # output_image to verify
            output_fname = output_path + "/" + os.path.basename(fname)
            pred_img = np.reshape(prediction, (image_h, image_w, cfg.NUM_CLASSES))
            pred_prob = genPredProb(pred_img, cfg.NUM_CLASSES)
            ret = cv2.imwrite(output_fname, pred_prob)
            if not ret:
                logger.error('writing image to {} failed!'.format(output_fname))
                sys.exit(-1)

            # masking image
            mask_fname = mask_path + "/" + os.path.basename(fname)
            r, g, b = cv2.split(image.astype(np.uint8))
            cv_img = cv2.merge([b, g, r])
            masked = image_process.prob_mask(cv_img, pred_prob)
            ret = cv2.imwrite(mask_fname, masked)
            if not ret:
                logger.error('writing image to {} failed!'.format(output_fname))
                sys.exit(-1)

        print("total time elapsed: {} s".format(total_time_elapsed))

if __name__ == '__main__':
    tf.app.run()


