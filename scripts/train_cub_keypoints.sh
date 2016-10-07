
. CONFIG

# Use this to change which GPU you use.
ID=1
# Batch size
BS=16
# Generator size
NGF=128
# Discriminator size
NDF=128
# Number of captions to use per image
NC=4

KD=16
ZKP=0

num_holdout=1 \
display_id=724${ID} \
port=8000 \
display=1 \
gpu=${ID} \
dataset='cub_keypoint_and_image' \
keypoint_dim=${KD} \
name="cub_kp_nh1_z${ZKP}_kd${KD}_bs${BS}_ngf${NGF}_ndf${NDF}" \
img_dir=${CUB_IMG_DIR} \
data_root=${CUB_KP_META_DIR} \
niter=600 \
save_every=100 \
print_every=10 \
nThreads=8 \
checkpoint_dir=${CHECKPOINT_DIR} \
dbg=0 \
num_elt=15 \
batchSize=${BS} \
ngf=${NGF} \
ndf=${NDF} \
numCaption=4 \
zero_kp=${ZKP} \
th main_cub_keypoints.lua

