#!/bin/bash

. CONFIG

NET_G=${DATA_ROOT}/models/mhp_keypoint_net_gen.t7
NET_T=${DATA_ROOT}/models/mhp_net_txt.t7
NET_KP=${DATA_ROOT}/models/mhp_net_keypoints.t7

SUBSET=${MHP_SUBSET}

# Uncomment line below to sample text from all categories.
#SUBSET=""

net_gen=${NET_G} \
net_txt=${NET_T} \
net_kp=${NET_KP} \
data_root=${MHP_META_DIR} \
img_dir=${MHP_IMG_DIR} \
trainfiles=${SUBSET} \
fineSize=128 \
th demo_mhp_keypoints_gen.lua

