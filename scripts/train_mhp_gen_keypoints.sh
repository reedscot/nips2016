
. CONFIG

# Use this to change which GPU you use.
ID=2
# Batch size
BS=64
# Generator size
NGF=128
# Discriminator size
NDF=128
NC=1

display_id=824${ID} \
num_holdout=1 \
port=8000 \
display=1 \
gpu=${ID} \
trainfiles=${MHP_SUBSET} \
dataset='mhp_keypoint_only' \
name="mhp_nh1_kptxt2kp_bs${BS}_ngf${NGF}_ndf${NDF}" \
img_dir=${MHP_IMG_DIR} \
data_root=${MHP_META_DIR} \
niter=200 \
save_every=20 \
print_every=10 \
nThreads=6 \
checkpoint_dir=${CHECKPOINT_DIR} \
dbg=0 \
num_elt=17 \
batchSize=${BS} \
ngf=${NGF} \
ndf=${NDF} \
numCaption=${NC} \
th main_gen_keypoints.lua

