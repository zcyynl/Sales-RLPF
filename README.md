# Salesloop-RLPF

Open-source preparation directory for the SalesLoop RLPF experiment pipeline.

### Training data

`train_grpo_rlpf.py` expects training data with at least:

- `corpus`
- `dataset`
- `label_30`
- `lock_label_str`
- `reward`

### Reward generation data

`score_for_reward.py` expects at least:

- `corpus`
- `dataset`
- `label_30`
- `dt`


## Example usage

### Reward generation

```bash
python src/score_for_reward.py \
  --data_file data/corpus_dataset.parquet \
  --ckpt_dir checkpoints \
  --ckpt_tag tag-ff-80000 \
  --pretrained_model Qwen/Qwen2.5-1.5B \
  --out_file data/corpus_dataset_with_reward.parquet
```

### Debug training

```bash
cd src
DATA_FILE=../data/corpus_dataset_with_reward_debug.parquet \
EVAL_DATA_FILE=../data/corpus_dataset_eval.parquet \
DEEPSPEED_CONFIG=../cfg/ds_config_debug.json \
./train_grpo_rlpf_debug.sh
```

### Full training

```bash
cd src
DATA_FILE=../data/corpus_dataset_with_reward.parquet \
EVAL_DATA_FILE=../data/corpus_dataset_eval.parquet \
DEEPSPEED_CONFIG=../cfg/ds_config_mixprecision_stage22.json \
./train_grpo_rlpf.sh
```
