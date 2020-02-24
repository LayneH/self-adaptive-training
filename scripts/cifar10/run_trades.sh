DATASET=cifar10
DATA_ROOT='~/datasets/CIFAR10'
ARCH=wrn34
LR=0.1
LR_SCHEDULE='step'
MILESTONES='75 90 100'
EPOCHS=100
BATCH_SIZE=128
LOSS=trades
BETA=6.0
EPS=0.031
STEP_SIZE=0.007
NUM_STEPS=10
EXP_NAME=${DATASET}/adv_${ARCH}_${LOSS}_b${BETA}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='0'

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main_adv.py --arch ${ARCH} --loss ${LOSS} --beta ${BETA} \
        --epsilon ${EPS} --step-size ${STEP_SIZE} --num-steps ${NUM_STEPS} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} --lr-milestones ${MILESTONES} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        >> ${LOG_FILE} 2>&1
