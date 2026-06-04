#!/bin/bash
# train_grpo_rlpf_debug.sh
# debug 验证示例：默认 2 卡，可按实际环境调整

OUT_DIR=${OUT_DIR:-outputs/grpo_rlpf_debug}
PYTHON_BIN=${PYTHON_BIN:-python}
DEEPSPEED_INCLUDE=${DEEPSPEED_INCLUDE:-localhost:0,1}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-cfg/ds_config_debug.json}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-Qwen/Qwen1.5-1.8B}
DATA_FILE=${DATA_FILE:-data/corpus_dataset_with_reward_debug.parquet}
EVAL_DATA_FILE=${EVAL_DATA_FILE:-data/corpus_dataset_eval.parquet}
CKPT_DIR=${CKPT_DIR:-checkpoints}
CKPT_TAG=${CKPT_TAG:-tag-ff-80000}

mkdir -p "${OUT_DIR}"

nohup "${PYTHON_BIN}" run_deepspeed.py --include="${DEEPSPEED_INCLUDE}" train_grpo_rlpf.py \
  --deepspeed \
  --deepspeed_config "${DEEPSPEED_CONFIG}" \
  --pretrained_model  "${PRETRAINED_MODEL}" \
  --data_file         "${DATA_FILE}" \
  --eval_data_file    "${EVAL_DATA_FILE}" \
  --load_ckpt_dir     "${CKPT_DIR}" \
  --load_ckpt_tag     "${CKPT_TAG}" \
  --out_dir           "${OUT_DIR}" \
  --grpo_gamma   0.1  \
  --alpha_bce    0.5  \
  --alpha_ce     0.5  \
  --batch_size        2    \
  --epochs            1    \
  --lora_r            8    \
  --lora_alpha        16   \
  --k                 128  \
  --lr                1e-5 \
  --max_len           512  \
  --dropout_rate      0.5  \
  --warmup_step_num   20   \
  --ckpt_interval     2000 \
  --pr_threshold      0.1  \
  --pos_oversample    9    \
  --max_train_samples 50000 \
  --lr_scheduler_type cosine \
  > "${OUT_DIR}/train.log" 2>&1 &

echo "PID=$!"
echo "tail -f ${OUT_DIR}/train.log"
