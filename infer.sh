export TRANSFORMERS_CACHE=""
export http_proxy=""
export https_proxy=""



##########  muq 25hz ##############
TRAINER_CONFIG="config/trainer/muq_25hz.json"
CKPT_PATH="pretrained/muq_25hz.pth"
INPUT_FOLDER="test_audio"
OUTPUT_FOLDER="exp_infer/muq_25hz"
######################################


python inference.py \
    --ckpt ${CKPT_PATH} \
    --input-dir ${INPUT_FOLDER} \
    --trainer-config ${TRAINER_CONFIG} \
    --output-dir ${OUTPUT_FOLDER}



