#!/bin/bash

. CONFIG

NET_G=${DATA_ROOT}/models/cub_bbox_net_gen.t7
NET_T=${DATA_ROOT}/models/cub_net_txt.t7

# Trans
net_gen=${NET_G} \
net_txt=${NET_T} \
txt_file=scripts/cub_captions_move.txt \
demo=trans \
th demo_cub_move_bbox.lua

# Stretch
net_gen=${NET_G} \
net_txt=${NET_T} \
txt_file=scripts/cub_captions_move.txt \
demo=stretch \
th demo_cub_move_bbox.lua

# Shrink
net_gen=${NET_G} \
net_txt=${NET_T} \
txt_file=scripts/cub_captions_move.txt \
demo=shrink \
th demo_cub_move_bbox.lua

