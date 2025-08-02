# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="0"




PYTHON_PATH="/usr/bin/python"
TORCH_RUN_PATH="/usr/local/bin/torchrun"
PYTHON_PATH=$(which python)
TORCH_RUN_PATH=$(which torchrun)

NUM_WORKERS=8
SEED=49
BATCH_SIZE=4
DATASET_CONFIG="config/dataset/test_audio.json"
TRAINER_CONFIG="config/trainer/muq_25hz.json"
RUN_NAME='muq_25hz'

EXP_NAME="Xcodec2"
SAVE_DIR="exp/${RUN_NAME}"
mkdir -p ${SAVE_DIR}


if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_DISABLED=true
fi
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

cmd="${PYTHON_PATH} train.py \
    --num-gpus ${num_gpus} \
    --num-workers ${NUM_WORKERS} \
    --trainer-config ${TRAINER_CONFIG} \
    --seed ${SEED} \
    --batch-size ${BATCH_SIZE} \
    --project-name ${EXP_NAME} \
    --run-name ${RUN_NAME} \
    --dataset-config ${DATASET_CONFIG} \
    --trainer-config ${TRAINER_CONFIG} \
    --save-dir ${SAVE_DIR} \

"


current_time=$(date -u -d "+8 hours" +"%Y-%m-%d %H:%M:%S")
current_day=$(date -u -d "+8 hours" +"%m%d")
current_hour=$(date -u -d "+8 hours" +"%Y_%m%d_%H%M")

destination_path="${SAVE_DIR}/train_${current_day}.sh"
cat "$0" > "$destination_path"

echo "======== current time $current_time, python config: =========="
echo "$cmd"


eval $cmd
