#!/bin/bash

python test.py --model=sq --ckpt_path=./log/model.ckpt-331199 --sample_path=/home/dijiang/Work/data/bag_extracted/image/left/0415_0430_highway_train/val_sample.txt --label_path=/home/dijiang/Work/data/bag_extracted/image/left/0415_0430_highway_train/val_label.txt --output_path=./log/output_debug
#python test.py --debug --dbg_tname=conv_classifier/weights/ExponentialMovingAverage --model=sq --ckpt_path=./log/model.ckpt-331199 --sample_path=/home/dijiang/Work/data/bag_extracted/image/left/0415_0430_highway_train/val_sample.txt --label_path=/home/dijiang/Work/data/bag_extracted/image/left/0415_0430_highway_train/val_label.txt --output_path=./log/output_debug
