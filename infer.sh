#!/bin/bash

python infer_with_pb.py --cfg_file=cfg_lane_mark.yaml --graph_name=./frozen_graph/test_uff.pb --sample_path=/home/dijiang/Work/data/bag_extracted/image/forward_center/20170714_300x640/tmp10.txt --output_path=./model_inference/test_uff_graph --mask_path=./model_inference/test_uff_graph_mask
