DATASET=cifar10
DATA_ROOT='~/datasets/CIFAR10'
ARCH=resnet18
LR=0.0001
WEIGHT_DECAY=0.0
OPTIMIZER='adam'
LR_SCHEDULE='step'
LR_GAMMA=0.1
LR_MILESTONES='4000'
EPOCHS=4000
BATCH_SIZE=128
LOSS=sat
ALPHA=0.9
ES=40
NOISE_RATE=0.1667
NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='noisy_set test_set'
GPU_ID='0'

### varying the width parameter of ResNet-18
for BW in {1..10} # {1..64}
do
    EXP_NAME=${DATASET}/dd_${ARCH}_${LOSS}_bw${BW}_${NOISE_TYPE}_r${NOISE_RATE}_${LR_SCHEDULE}_m${ALPHA}_p${ES}_$1
    SAVE_DIR=ckpts/${EXP_NAME}
    LOG_FILE=log/${EXP_NAME}.log
    echo ${EXP_NAME}

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python -u main.py --arch ${ARCH} --loss ${LOSS} \
            --dataset ${DATASET} --data-root ${DATA_ROOT} \
            --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
            --lr ${LR} --weight-decay ${WEIGHT_DECAY} \
            --sat-alpha ${ALPHA} --sat-es ${ES} \
            --optimizer ${OPTIMIZER} --base-width ${BW} \
            --lr-schedule ${LR_SCHEDULE} --lr-gamma ${LR_GAMMA} --lr-milestones ${LR_MILESTONES} \
            --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
            --save-dir ${SAVE_DIR} \
            >> ${LOG_FILE}  &
done
