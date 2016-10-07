
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
NC=2

# Minimum score assigned to samples to update D
T=0.2
ACTG=relu
ACTD=elu
NZ=100
LR=0.0002
DROP=0.2

display_id=727${ID} \
num_holdout=1 \
port=8000 \
dropout=${DROP} \
display=1 \
gpu=${ID} \
dataset='mhp_keypoint_and_image' \
lr=${LR} \
fake_score_thresh=${T} \
activationG=${ACTG} \
activationD=${ACTD} \
nz=${NZ} \
trainfiles="" \
name="mhp_full_nh1_i128_kp_d${DROP}_nc${NC}_lr${LR}_bs${BS}_ngf${NGF}_ndf${NDF}_d${ACTD}" \
img_dir=${MHP_IMG_DIR} \
data_root=${MHP_META_DIR} \
niter=200 \
save_every=10 \
decay_every=20 \
print_every=10 \
nThreads=6 \
checkpoint_dir=${CHECKPOINT_DIR} \
dbg=0 \
num_elt=17 \
batchSize=${BS} \
loadSize=140 \
fineSize=128 \
ngf=${NGF} \
ndf=${NDF} \
init_t=${MHP_NET_TXT} \
numCaption=${NC} \
th main_mhp.lua

