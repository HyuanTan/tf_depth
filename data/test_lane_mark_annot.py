#!/usr/bin/python

import sys
import os
import cv2
import numpy as np
import json
import math

Colors = [(0,0,255),    # single white solid
          (0,255,0),        # single white dotted
          (0,0,0),    # dual white solid
          (255,0,0),        # yellow solid
          (255,64,255), # yellow dotted
          (128,128,255),      # road shoulder
          (0,255,255)]        # frond vehicle rear

def labelling(fname, json_obj):

    #maskname = maskpath + '/' + fname
    print fname
    img = cv2.imread(fname)
    lines = json_obj['line']
    for line in lines:
        line_type = int(line['attribute']['type']) - 1
        if line_type == 2:
            print 'dual white solide!'
        if line_type > 4:
            continue
        points = []
        #print len(line['points'])
        dist = 0
        cnt = 0
        for pt in line['points']:
            if cnt > 0:
                dist += math.sqrt((pt['x'] - points[cnt-1][0])**2 + (pt['y'] - points[cnt-1][1])**2)

            points.append((pt['x'], pt['y']))
            cnt += 1
        cv2.polylines(img, np.int32([points]), False, Colors[line_type])
        if dist > 30:
            for pt in points:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 2, Colors[line_type], -1)
        else:
            print 'skip point drawing'

    return img

    #if len(points) > 0:
        #cv2.fillConvexPoly(label, np.array(points, dtype=np.int32), 255)

    #mask = img.copy()
    #mask[label>0,1] = label[label>0]
    #overlay = cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)
    #cv2.imwrite(labelname, label)
    #cv2.imwrite(maskname, overlay)

def labelling2(fname, json_obj):
    print fname
    img = cv2.imread('{}/{}'.format(inpath, fname))
    if img is None:
        print('no {}, return!'.format(fname))
        return
    label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    lanes = json_obj['line']
    for lane in lanes:
        line_type = int(lane['attribute']['type'])
        if line_type == 2:
            color = 1
        else:
            color = 2
        if line_type > 5:
            continue
        points = []
        #print len(line['points'])
        dist = 0
        cnt = 0
        for pt in lane['points']:
            if cnt > 0:
                dist += math.sqrt((pt['x'] - points[cnt-1][0])**2 + (pt['y'] - points[cnt-1][1])**2)

            points.append((pt['x'], pt['y']))
            cnt += 1

        if dist > 30:
            cv2.polylines(label, np.int32([points]), False, color)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    label = cv2.dilate(label, kernel, iterations = 1)
    overlay = img.copy()
    #print overlay[label==1].shape
    overlay[label==2,0] = Colors[0][0]
    overlay[label==2,1] = Colors[0][1]
    overlay[label==2,2] = Colors[0][2]
    overlay[label==1,0] = Colors[3][0]
    overlay[label==1,1] = Colors[3][1]
    overlay[label==1,2] = Colors[3][2]
    mask = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    cv2.imwrite('{}/{}'.format(labelpath, fname), label)
    cv2.imwrite('{}/{}'.format(maskpath, fname), mask)




if __name__ == '__main__':
    json_fname = sys.argv[1]
    inpath = sys.argv[2]
    labelpath = sys.argv[3]
    maskpath = sys.argv[4]
    #exclude = sys.argv[5]

    #exc_list = []
    #with open(exclude, 'r') as infile:
        #for line in infile:
            #exc_list.append(line.strip())

    if not os.path.isdir(labelpath):
        os.mkdir(labelpath)
    if not os.path.isdir(maskpath):
        os.mkdir(maskpath)

    lineno = 0

    with open(json_fname, 'r') as infile:
        for line in infile:
            if lineno <= 0:
                lineno += 1
                continue
            #if lineno > 2001:
                #break
            lineno += 1

            line = line.strip().split()
            json_obj = json.loads(line[2])
            fname = line[0]
            #fname = '{}/{}'.format(inpath, line[0])
            #img = draw(json_obj)
            img = labelling2(fname, json_obj)


