#!/usr/bin/python

import sys
import os
import random

import cv2
import numpy as np


def transparent_mask(raw, mask):
    overlay = raw.copy()
    overlay[(mask>0)] = mask[(mask>0)]
    return cv2.addWeighted(raw, 0.5, overlay, 0.5, 0)

def prob_mask(raw, prob):
    return cv2.addWeighted(raw, 1.0, prob, 1.0, 0)

def histEqualize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    cvt = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    return cvt

def adjust_light(hls, rate):
    """ input image is in HLS color space
        output image is in RGB color space
    """
    tmp = hls.astype(np.float64)
    tmp[:,:,1] = tmp[:,:,1] * rate
    tmp = np.clip(tmp, 0, 255).astype(np.uint8)
    cvt = cv2.cvtColor(tmp, cv2.COLOR_HLS2RGB)

    return cvt

def adjust_hue(hls, offset):
    """ input image is in HLS color space
        output image is in RGB color space
    """
    tmp = hls.astype('int16')
    tmp[:,:,0] = (tmp[:,:,0] + offset + 180) % 180
    tmp = tmp.astype('uint8')
    cvt = cv2.cvtColor(tmp, cv2.COLOR_HLS2RGB)

    return cvt

def adjust_saturation(hls, offset):
    """ input image is in HLS color space
        output image is in RGB color space
    """
    tmp = hls.astype('int16')
    tmp[:,:,2] += offset
    #tmp[:,:,2] = np.where(tmp[:,:,2] > 255, 255, tmp[:,:,2])
    #tmp[:,:,2] = np.where(tmp[:,:,2] < 0, 0, tmp[:,:,2])
    #tmp = tmp.astype('uint8')
    tmp = np.clip(tmp, 0, 255).astype(np.uint8)
    cvt = cv2.cvtColor(tmp, cv2.COLOR_HLS2RGB)

    return cvt

def adjust_contrast(img, rate, b_mean, g_mean, r_mean):
    b, g, r = cv2.split(img.astype(np.float32))
    b = (b - b_mean) * rate + b_mean
    g = (g - g_mean) * rate + g_mean
    r = (r - r_mean) * rate + r_mean

    b = np.clip(b, 0, 255).astype('uint8')
    g = np.clip(g, 0, 255).astype('uint8')
    r = np.clip(r, 0, 255).astype('uint8')

    out = cv2.merge([b, g, r])
    return out

def add_random_shadow_v(image, side, shadow_rate):
    """ input image is in HLS color space
        output image is in RGB color space
    """
    height = image.shape[0]
    width = image.shape[1]
    margin = 0.3 * width
    top_y = width*np.random.uniform()
    top_x = 0
    bot_x = height
    if side == 'left':
        bot_y = margin + (width - margin)*np.random.uniform()
    elif side == 'right':
        bot_y = (width - margin)*np.random.uniform()

    image_hls = image.copy()
    shadow_mask = 0*image_hls[:,:,1]
    X_m, Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1

    if side == 'left':
        cond = shadow_mask==1
    elif side == 'right':
        cond = shadow_mask==0

    image_hls = image_hls.astype(np.float64)
    image_hls[:,:,1][cond] = image_hls[:,:,1][cond]*shadow_rate
    #image_hls[:,:,1] = np.where(image_hls[:,:,1] > 255, 255, image_hls[:,:,1])
    #image_hls[:,:,1] = np.where(image_hls[:,:,1] < 0, 0, image_hls[:,:,1])
    #image_hls = image_hls.astype(np.uint8)
    image_hls = np.clip(image_hls, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return out

def add_random_shadow_h(image, shadow_rate):
    """ input image is in HLS color space
        output image is in RGB color space
    """
    height = image.shape[0]
    width = image.shape[1]
    margin = 0.2 * height
    left_y = 0
    left_x = margin + (height - margin)*np.random.uniform()
    right_x = margin + (height - margin)*np.random.uniform()
    right_y = width

    image_hls = image.copy()
    shadow_mask = 0*image_hls[:,:,1]
    X_m, Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]]
    shadow_mask[((X_m-left_x)*(right_y-left_y) -(right_x - left_x)*(Y_m-left_y) >=0)]=1

    cond = shadow_mask==1

    image_hls = image_hls.astype(np.float64)
    image_hls[:,:,1][cond] = image_hls[:,:,1][cond]*shadow_rate
    #image_hls[:,:,1] = np.where(image_hls[:,:,1] > 255, 255, image_hls[:,:,1])
    #image_hls[:,:,1] = np.where(image_hls[:,:,1] < 0, 0, image_hls[:,:,1])
    #image_hls = image_hls.astype(np.uint8)
    image_hls = np.clip(image_hls, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return out

def add_random_shadow_c(image, shadow_rate):
    """ input image is in HLS color space
        output image is in RGB color space
    """
    height = image.shape[0]
    width = image.shape[1]
    margin = 0.2 * height
    left_y = 0
    left_x1 = margin + (height - margin)*np.random.uniform()
    left_x2 = margin + (height - margin)*np.random.uniform()
    right_x1 = margin + (height - margin)*np.random.uniform()
    right_x2 = margin + (height - margin)*np.random.uniform()
    right_y = width

    top_left_x = min(left_x1, left_x2)
    bot_left_x = max(left_x1, left_x2)
    top_right_x = min(right_x1, right_x2)
    bot_right_x = max(right_x1, right_x2)

    image_hls = image.copy()
    shadow_mask = 0*image_hls[:,:,1]
    X_m, Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]]
    shadow_mask[np.logical_and((X_m-top_left_x)*(right_y-left_y) -(top_right_x - top_left_x)*(Y_m-left_y) >=0,
                               (X_m-bot_left_x)*(right_y-left_y) -(bot_right_x - bot_left_x)*(Y_m-left_y) <=0)]=1

    cond = shadow_mask==1

    image_hls = image_hls.astype(np.float64)
    image_hls[:,:,1][cond] = image_hls[:,:,1][cond]*shadow_rate
    #image_hls[:,:,1] = np.where(image_hls[:,:,1] > 255, 255, image_hls[:,:,1])
    #image_hls[:,:,1] = np.where(image_hls[:,:,1] < 0, 0, image_hls[:,:,1])
    #image_hls = image_hls.astype(np.uint8)
    image_hls = np.clip(image_hls, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return out

def add_fake_line(image, shadow_rate, lw=5):
    """ input image is in HLS color space
        output image is in RGB color space
    """
    height = image.shape[0]
    width = image.shape[1]
    top_margin = 0.2 * height
    #bot_margin = 0.1 * height
    bot_left_margin = 0.2 * width
    bot_right_margin = 0.2 * width
    top_left_margin = 0.4 * width
    top_right_margin = 0.4 * width

    #top_x = top_margin + (height - top_margin - bot_margin) * np.random.uniform()
    top_x = top_margin
    top_left_y = top_left_margin + (width - top_left_margin - top_right_margin)*np.random.uniform()
    bot_left_y = bot_left_margin + (width - bot_left_margin - bot_right_margin)*np.random.uniform()
    bot_x = height

    top_right_y = top_left_y + lw
    bot_right_y = bot_left_y + lw

    image_hls = image.copy()
    shadow_mask = 0*image_hls[:,:,1]
    X_m, Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]]
    shadow_mask[np.logical_and(X_m >= top_x,
                np.logical_and((X_m-top_x)*(bot_left_y-top_left_y) - (bot_x - top_x)*(Y_m-top_left_y) <=0,
                               (X_m-top_x)*(bot_right_y-top_right_y) - (bot_x - top_x)*(Y_m-top_right_y) >=0))]=1

    cond = shadow_mask==1

    image_hls = image_hls.astype(np.float64)
    image_hls[:,:,1][cond] = image_hls[:,:,1][cond]*shadow_rate
    #image_hls[:,:,1] = np.where(image_hls[:,:,1] > 255, 255, image_hls[:,:,1])
    #image_hls[:,:,1] = np.where(image_hls[:,:,1] < 0, 0, image_hls[:,:,1])
    #image_hls = image_hls.astype(np.uint8)
    image_hls = np.clip(image_hls, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return out

def add_fake_patch(image, shadow_rate, pn=8):
    """ input image is in HLS color space
        output image is in RGB color space
    """
    height = image.shape[0]
    width = image.shape[1]
    top_margin = 0.2 * height
    #bot_margin = 0.1 * height
    left_margin = 0.2 * width
    right_margin = 0.2 * width

    # generate points and arrange them in clockwise order
    tmpy1 = np.random.uniform(left_margin, width - right_margin)
    tmpy2 = np.random.uniform(left_margin, width - right_margin)

    left_y = min(tmpy1, tmpy2)
    right_y = max(tmpy1, tmpy2)

    left_x = np.random.uniform(top_margin, height)
    right_x = np.random.uniform(top_margin, height)

    up_lb = max(left_x, right_x)
    down_ub = min(left_x, right_x)

    left_pt = [left_y, left_x]
    right_pt = [right_y, right_x]

    up_pts_num = np.random.randint(pn - 1)
    down_pts_num = pn - 2 - up_pts_num

    vertices = [left_pt]
    ub = top_margin
    lb = up_lb
    leftb = left_y
    rightb = right_y
    for i in range(up_pts_num):
        x = np.random.uniform(ub, lb)
        y = np.random.uniform(leftb, rightb)
        vertices.append([y,x])
        leftb = y
    vertices.append(right_pt)
    ub = down_ub
    lb = height
    leftb = left_y
    rightb = right_y
    for i in range(down_pts_num):
        x = np.random.uniform(ub, lb)
        y = np.random.uniform(leftb, rightb)
        vertices.append([y,x])
        rightb = y


    image_hls = image.copy()
    shadow_mask = 0*image_hls[:,:,1]

    cv2.fillConvexPoly(shadow_mask, np.array(vertices, dtype=np.int32), 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)

    cond = shadow_mask>0

    image_hls = image_hls.astype(np.float64)
    image_hls[:,:,1][cond] = image_hls[:,:,1][cond]*shadow_rate
    #image_hls[:,:,1] = np.where(image_hls[:,:,1] > 255, 255, image_hls[:,:,1])
    #image_hls[:,:,1] = np.where(image_hls[:,:,1] < 0, 0, image_hls[:,:,1])
    #image_hls = image_hls.astype(np.uint8)
    image_hls = np.clip(image_hls, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return out

def add_noise(noise_typ, image, p):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        stddev = p
        gauss = np.random.normal(0, stddev, (row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        out = image + gauss
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)

        return out

    elif noise_typ == "s&p":
        row,col,ch = image.shape
        rate = p
        sp_noise = np.random.randint(0, 256, (row, col, ch)).reshape(row, col, ch)
        white_thres = int(256 * (1-rate))
        black_thres = int(256 * rate)
        white = sp_noise >= white_thres
        black = sp_noise < black_thres
        out = image.copy()
        out[white] = 255
        out[black] = 0

        return out


