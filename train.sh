#!/bin/bash

python train.py --ckpt_path=./log --sample_path=/data4/kitti_raw_train/disp_train_left.txt --label_path=/data4/kitti_raw_train/disp_train_right.txt --val_sample_path=/data4/kitti_raw_train/disp_val_left.txt --val_label_path=/data4/kitti_raw_train/disp_val_right.txt
