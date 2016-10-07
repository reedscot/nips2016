
. CONFIG

# Use this to change which GPU you use.
ID=1
# Batch size
BS=64
# Generator size
NGF=128
# Discriminator size
NDF=128

display_id=824${ID} \
num_holdout=1 \
port=8000 \
display=1 \
gpu=${ID} \
dataset='cub_keypoint_only' \
name="cub_nh1_kptxt2kp_bs${BS}_ngf${NGF}_ndf${NDF}" \
img_dir=${CUB_IMG_DIR} \
data_root=${CUB_KP_META_DIR} \
niter=200 \
save_every=10 \
print_every=10 \
nThreads=6 \
checkpoint_dir=${CHECKPOINT_DIR} \
dbg=0 \
num_elt=15 \
batchSize=${BS} \
ngf=${NGF} \
ndf=${NDF} \
th main_gen_keypoints.lua

