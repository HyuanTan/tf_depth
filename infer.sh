#!/bin/bash

python infer.py --ckpt_path=./plusai_models/res50_model_stereo.ckpt-133549 --sample_path=/home/dijiang/Work/data/bag_extracted/image/stereo/demo_test/left.txt --stereo_path=/home/dijiang/Work/data/bag_extracted/image/stereo/demo_test/right.txt --recon_path=./model_inference/plusai_v1_stereo/recon --resize_ratio=2 --do_stereo=True

# mono
#python infer.py --ckpt_path=./plusai_models/res50_model_mono.ckpt-133549 --sample_path=/home/dijiang/Work/data/bag_extracted/image/stereo/demo_test/left.txt --stereo_path=/home/dijiang/Work/data/bag_extracted/image/stereo/demo_test/right.txt --recon_path=./model_inference/plusai_v1_mono/recon --resize_ratio=2
