#!/usr/bin/python

import sys
import os
import argparse
import math

import cv2
import numpy as np

from data import image_process


def preprocess_label(path):
    dir = os.path.dirname(path)
    fd = open(path)
    for line in fd:
        line = line.strip()
        print(line)
        #name, ext = os.path.splitext(line)
        #outname = name + '_proc' + ext
        filename = dir + '/' + line
        #outfilename = dir + '/' + outname

        img = cv2.imread(filename)
        shape = (img.shape[0], img.shape[1], 1)
        img_proc = np.zeros(shape, np.uint8)
        img_proc[img[:,:,0]>0, 0] = img[img[:,:,0]>0, 0]
        cv2.imwrite(filename, img_proc)

def calc_class_weight(path, num_class, eps=1.02):
    #class_map = {}
    class_count = np.zeros((num_class))
    for fname in os.listdir(path):
        print fname
        img = cv2.imread(os.path.join(path, fname))
        class_count += np.bincount(img.flatten(), minlength=num_class)/3
        #print(img.shape)
        #class_map['p'] += (img>0).sum() / 3
        #class_map['n'] += (img==0).sum() / 3
    total_cnt = class_count.sum()
    class_freq = class_count / float(total_cnt)
    class_weight = 1. / np.log(class_freq + eps)

    print(class_count)
    print(class_freq)
    print(class_weight)

def crop_image(path, top_rate, bot_rate, left_rate, right_rate):
    dir = os.path.dirname(path)
    fd = open(path)
    cnt = 0
    for line in fd:
        line = line.strip()
        filename = dir + '/' + line
        #print(filename)
        img = cv2.imread(filename)
        if cnt == 0:
            shape = img.shape
            top = int(round(shape[0] * top_rate))
            bot = int(round(shape[0] * bot_rate))
            left = int(round(shape[1] * left_rate))
            right = int(round(shape[1] * right_rate))

            print("cropped height: [{}, {}), width: [{}, {})".format(top, bot, left, right))

        print("cropping {}".format(line))
        cropped = img[top:bot, left:right]
        
        cv2.imwrite(filename, cropped)
        cnt += 1

def resize_image(path, height_resize, width_resize):
    dir = os.path.dirname(path)
    fd = open(path)
    for line in fd:
        line = line.strip()
        filename = dir + '/' + line
        #print(filename)
        img = cv2.imread(filename)
        shape = img.shape
        if shape[0] != height_resize or shape[1] != width_resize:
            print("resizing {}, original shape is ({}, {})".format(line, shape[0], shape[1]))
            im = cv2.resize(img, (width_resize, height_resize), 0, 0, cv2.INTER_AREA)

            cv2.imwrite(filename, im)
        
def _msg_compressed_to_png_file(msg, path, flip):
    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    if flip:
        img = cv2.flip(img, -1)

    cv2.imwrite(path, img)

def _msg_compressed_show(msg, topic):
    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    win_name = topic.split('/')[1]
    if 'roof' not in win_name:
        img = cv2.flip(img, -1)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(1)

def rosbag_image_show(bag, topic):
    msgs = bag.read_messages(topics=[topic])
    for m in msgs:
        if m.topic == topic:
            img = _msg_compressed_show(m.message, topic)

def rosbag_image_save(bag, topic, output_dir, prefix='', rate=0.1):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    start_time = None
    msgs = bag.read_messages(topics=[topic])
    flip = False if 'roof' in topic else True
    for m in msgs:
        if m.topic == topic:
            if start_time is None:
                start_time = m.timestamp
            frame_number = int(((m.timestamp - start_time).to_sec() + (rate / 2.0)) / rate)
            print frame_number
            _msg_compressed_to_png_file(m.message, "%s/%s%04d.png" % (output_dir, prefix, frame_number), flip)

def histEqualize(inpath, outpath):
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    for fname in os.listdir(inpath):
        print(fname)
        filename = inpath + '/' + fname
        img = cv2.imread(filename)
        outimg = image_process.histEqualize(img)
        outname = outpath + '/' + fname
        cv2.imwrite(outname, outimg)

def augment_image(inpath, maskpath):
    # add noise
    for fname in os.listdir(inpath):
        name, ext = os.path.splitext(fname)
        filename = inpath + '/' + fname
        maskname = maskpath + '/' + fname

        print("augmenting {}".format(fname))

        #print("adding noise {}".format(fname))
        img = cv2.imread(filename)
        # gauss noise
        #noise_rate_list = [10, 20]
        #for rate in noise_rate_list:
            #outimg = image_process.add_noise('gauss', img, rate)
            #outname = name + '_noise_g{}'.format(rate) + ext
            #cv2.imwrite(inpath + '/' + outname, outimg)
            #outmaskname = maskpath + '/' + outname
            #os.system('cp {} {}'.format(maskname, outmaskname))

        # s&p noise
        #outimg = image_process.add_noise('s&p', img, 0.005)
        #outname = name + '_noise_sp' + ext
        #cv2.imwrite(inpath + '/' + outname, outimg)
        #outmaskname = maskpath + '/' + outname
        #os.system('cp {} {}'.format(maskname, outmaskname))

    #for fname in os.listdir(inpath):
        #name, ext = os.path.splitext(fname)
        #filename = inpath + '/' + fname
        #maskname = maskpath + '/' + fname

        #img = cv2.imread(filename)

        #ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        #print("histogram equalizing")
        #ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        #cvt = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        #outname = name + '_hist' + ext
        ##cvt = cvt.astype('uint8')
        #cv2.imwrite(inpath + '/' + outname, cvt)
        #outmaskname = maskpath + '/' + outname
        #os.system('cp {} {}'.format(maskname, outmaskname))


        shadow_rate_list = [0.1, 0.3, 0.5]
        patch_rate_list = [0.1, 0.3, 0.5]
        light_rate_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        for rate in light_rate_list:
            outimg = image_process.adjust_light(hls, rate)
            outname = name + '_hls_l{}'.format(rate) + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

        for i,rate in enumerate(shadow_rate_list):
            oname = name + '_shadow{}'.format(rate)
            # left
            outimg = image_process.add_random_shadow_v(hls, 'left', rate)
            outname = oname + '_left' + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

            # right
            outimg = image_process.add_random_shadow_v(hls, 'right', rate)
            outname = oname + '_right' + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

            # bottom
            outimg = image_process.add_random_shadow_h(hls, rate)
            outname = oname + '_bottom' + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

            # cross
            outimg = image_process.add_random_shadow_c(hls, rate)
            outname = oname + '_cross' + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

        #for i,rate in enumerate(patch_rate_list):
            #oname = name + '_shadow{}'.format(rate)
            ## fake line
            #outimg = image_process.add_fake_line(hls, rate)
            #outname = oname + '_line' + ext
            #cv2.imwrite(inpath + '/' + outname, outimg)
            #outmaskname = maskpath + '/' + outname
            #os.system('cp {} {}'.format(maskname, outmaskname))

            ## fake patch
            #outimg = image_process.add_fake_patch(hls, rate)
            #outname = oname + '_patch' + ext
            #cv2.imwrite(inpath + '/' + outname, outimg)
            #outmaskname = maskpath + '/' + outname
            #os.system('cp {} {}'.format(maskname, outmaskname))

        b_mean = img[:,:,0].mean()
        g_mean = img[:,:,1].mean()
        r_mean = img[:,:,2].mean()

        for i in range(-3, 4):
            if i == 0:
                continue
            # Hue
            outimg = image_process.adjust_hue(hls, i*10)
            outname = name + '_hls_h' + str(i*10) + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

            # Saturation
            outimg = image_process.adjust_saturation(hls, i*10)
            outname = name + '_hls_s' + str(i*10) + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))
            
            # Contrast
            factor = 1. + i * 0.3
            if factor > 0:
                outimg = image_process.adjust_contrast(img, factor, b_mean, g_mean, r_mean)
                outname = name + '_hls_c' + str(factor) + ext
                cv2.imwrite(inpath + '/' + outname, outimg)
                outmaskname = maskpath + '/' + outname
                os.system('cp {} {}'.format(maskname, outmaskname))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--func", type=str, help="functionality")
    parser.add_argument("--infile", type=str, help="input data file")
    parser.add_argument("--inpath", type=str, help="input data path")
    parser.add_argument("--maskpath", type=str, help="input mask data path")
    parser.add_argument("--outpath", type=str, help="output data path")
    parser.add_argument("--bag", type=str, help="ros bag")
    parser.add_argument("--topic", type=str, help="ros topic")
    parser.add_argument("--prefix", default='', type=str, help="bag extrated image prefix")
    parser.add_argument("--toprate", default=0.0, type=float, help="for image cropping")
    parser.add_argument("--botrate", default=1.0, type=float, help="for image cropping")
    parser.add_argument("--leftrate", default=0.0, type=float, help="for image cropping")
    parser.add_argument("--rightrate", default=1.0, type=float, help="for image cropping")
    parser.add_argument("--height", type=int, help="for image resizing")
    parser.add_argument("--width", type=int, help="for image resizing")
    parser.add_argument("--numclass", type=int, help="for calc class weight")
    parser.add_argument("--epsilon", default=1.02, type=float, help="for calc class weight")

    args = parser.parse_args()

    if args.func == 'label_proc':
        preprocess_label(args.infile)
    elif args.func == 'class_weight':
        calc_class_weight(args.inpath, args.numclass, args.epsilon)
    elif 'crop' == args.func:
        crop_image(args.infile, args.toprate, args.botrate, args.leftrate, args.rightrate)
    elif 'resize' == args.func:
        resize_image(args.infile, args.height, args.width)
    elif args.func == 'augment':
        augment_image(args.inpath, args.maskpath)
    elif args.func == 'hist':
        histEqualize(args.inpath, args.outpath)
    elif 'rosbag' in args.func:
        import rosbag
        from cv_bridge import CvBridge

        bridge = CvBridge()

        if args.func == 'rosbag_img_show':
            bag = rosbag.Bag(args.bag)
            rosbag_image_show(bag, args.topic)
        elif args.func == 'rosbag_img_save':
            bag = rosbag.Bag(args.bag)
            rosbag_image_save(bag, args.topic, args.outpath, args.prefix)

