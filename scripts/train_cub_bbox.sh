
. CONFIG

# Use this to change which GPU you use.
ID=2
# Batch size
BS=16
# Generator size
NGF=128
# Discriminator size
NDF=128
# Number of captions to use per image
NC=4

display_id=724${ID} \
num_holdout=1 \
port=8000 \
display=1 \
gpu=${ID} \
dataset='cub_bbox_and_image' \
name="cub_bbox_nh1_stn_bs${BS}_ngf${NGF}_ndf${NDF}" \
img_dir=${CUB_IMG_DIR} \
data_root=${CUB_BBOX_META_DIR} \
niter=600 \
save_every=100 \
print_every=4 \
nThreads=6 \
checkpoint_dir=${CHECKPOINT_DIR} \
dbg=0 \
iou_thresh=0.6 \
num_elt=1 \
batchSize=${BS} \
ngf=${NGF} \
ndf=${NDF} \
numCaption=${NC} \
th main_cub_bbox.lua

