#!/bin/bash

. CONFIG

NET_G=${DATA_ROOT}/models/cub_keypoint_net_gen.t7
NET_T=${DATA_ROOT}/models/cub_net_txt.t7
NET_KP=${DATA_ROOT}/models/cub_net_keypoints.t7

net_gen=${NET_G} \
net_txt=${NET_T} \
net_kp=${NET_KP} \
data_root=${CUB_KP_META_DIR} \
img_dir=${CUB_IMG_DIR} \
th demo_cub_keypoints_gen.lua

