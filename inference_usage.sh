TRAINER_CONFIG="config/trainer/muq_25hz.json"
CKPT_PATH="pretrained/muq_25hz.pth"
INPUT_FOLDER="test_audio"

python inference_usage.py \
    --ckpt ${CKPT_PATH} \
    --input-dir ${INPUT_FOLDER} \
    --trainer-config ${TRAINER_CONFIG} \



