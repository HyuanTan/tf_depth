#!/usr/bin/python

import sys
import os
import cv2
import numpy as np
import json

json_fname = sys.argv[1]
inpath = sys.argv[2]
labelpath = sys.argv[3]
maskpath = sys.argv[4]

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

        label = np.zeros([1200,1920], dtype=np.uint8)
        points = []

        fields = line.strip().split('\t')

        fname = fields[0].split('\\')[3][4:]
        print fname
        in_fname = inpath + '/' + fname
        labelname = labelpath + '/' + fname
        maskname = maskpath + '/' + fname
        json_str = fields[2]
        json_obj = json.loads(json_str)
        lane_areas = json_obj['area']
        for area in lane_areas:
            if area['attribute']['type'] == 'current_lane':
                print len(area['points'])
                for pt in area['points']:
                    points.append([pt['x'], pt['y']])

        if len(points) > 0:
            cv2.fillConvexPoly(label, np.array(points, dtype=np.int32), 255)

        img = cv2.imread(in_fname)
        mask = img.copy()
        mask[label>0,1] = label[label>0]
        overlay = cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)
        cv2.imwrite(labelname, label)
        cv2.imwrite(maskname, overlay)




#cv2.namedWindow('foo')
#cv2.imshow('foo', background)
#while True:
    #key = cv2.waitKey(1) & 0xff
    #if key == ord('q'):
        #cv2.destroyAllWindows()
        #break




