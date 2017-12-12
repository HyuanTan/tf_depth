#!/bin/bash

python train.py --ckpt_path=./plusai_log --sample_path=/data4/bag_extracted/stereo_train/left.txt --label_path=/data4/bag_extracted/stereo_train/right.txt --val_sample_path=/data4/bag_extracted/stereo_train/left_val.txt --val_label_path=/data4/bag_extracted/stereo_train/right_val.txt
