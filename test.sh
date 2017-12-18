#!/bin/bash

#stereo
python test.py --cfg_file=./cfg_kitti.yaml --ckpt_path=./ckpt/res50_model_stereo.ckpt-179999 --sample_path=/data4/kitti_stereo_2015/eval_sample.txt --stereo_path=/data4/kitti_stereo_2015/eval_stereo.txt --label_path=/data4/kitti_stereo_2015/eval_label.txt --output_path=./res50_disp_stereo/

