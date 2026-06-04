"""
score_for_reward.py

用 tag-ff-80000 checkpoint 对 corpus_dataset_zcy.parquet 中 dataset=3 (August) 数据推理，
按天（dt）排名，计算 RLPF reward:
    R_i = label_30_i * 1 / log2(rank_i + 1)

输出：corpus_dataset_zcy_with_reward.parquet
  - 包含原始全量数据
  - dataset=3 行新增 score / rank / reward 三列
  - dataset=1,2 行 reward=0.0

用法（单卡）:
    python score_for_reward.py

用法（多卡，torchrun）:
    torchrun --nproc_per_node=8 score_for_reward.py

注意：必须使用原始架构（参数名 custom_linear_0 / custom_linear_1 / pairwise_head）
以确保正确加载 tag-ff-80000 的 task head 权重。
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from modelscope import AutoTokenizer, AutoModelForCausalLM
from layers_score import add_lora


# ══════════════════════════════════════════════════════════════
#  模型架构（与 train_0921_oral.sh / train_newdata_0811_pooling.py 完全一致）
#  参数名必须对齐才能正确加载 checkpoint
# ══════════════════════════════════════════════════════════════

class TaskLayer(nn.Module):
    """与 train_newdata_0811_pooling.py 中同名类保持一致（参数名不能改）"""
    def __init__(self, hidden_size, embedding_size=128, dropout_rate=0.5):
        super().__init__()
        self.custom_linear_0         = nn.Linear(hidden_size, embedding_size, dtype=torch.bfloat16)
        self.custom_linear_dropout   = nn.Dropout(p=dropout_rate)
        self.custom_linear_activation = nn.GELU()
        self.custom_linear_1         = nn.Linear(embedding_size, 1, dtype=torch.bfloat16)

    def forward(self, x):
        x = self.custom_linear_0(x)
        x = self.custom_linear_dropout(x)
        x = self.custom_linear_activation(x)
        return self.custom_linear_1(x), x          # (B,1), (B,emb)


class PairwiseRankHead(nn.Module):
    """仅用于加载 checkpoint，forward 在推理中不调用"""
    def __init__(self, hidden_size, emb_size=128, dropout=0.5):
        super().__init__()
        self.fc1  = nn.Linear(hidden_size, emb_size, dtype=torch.bfloat16)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(emb_size, 1, dtype=torch.bfloat16)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x)))).squeeze(-1)


class ScoringModel(nn.Module):
    """
    架构与 tag-ff-80000 训练时完全一致（含 pairwise_head，strict=True 加载）。
    推理只用 custom_linear 头的 score。
    """
    def __init__(self, pretrained_model_name, lora_r=8, lora_alpha=16, k=128, dropout_rate=0.5):
        super().__init__()
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            use_cache=False,
        )
        if lora_r > 0:
            add_lora(self.pretrained_model, lora_r, lora_alpha, pissa=False)
        self.custom_linear = TaskLayer(
            self.pretrained_model.config.hidden_size, k, dropout_rate)
        self.pairwise_head = PairwiseRankHead(
            self.pretrained_model.config.hidden_size, k, dropout_rate)

    @torch.no_grad()
    def get_score(self, input_ids, attention_mask):
        out       = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_h    = out.hidden_states[-1][:, -1, :]     # (B, H) last token
        score, _  = self.custom_linear(last_h)          # (B, 1)
        return score.squeeze(-1)                         # (B,)


# ══════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════

class CorpusDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts   = texts
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len - 1,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        pad_id   = torch.tensor([[self.tok.pad_token_id]])
        pad_mask = torch.tensor([[1]])
        input_ids      = torch.cat([enc.input_ids,      pad_id],   dim=1)
        attention_mask = torch.cat([enc.attention_mask, pad_mask], dim=1)
        return input_ids.squeeze(0), attention_mask.squeeze(0)


# ══════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════

def load_checkpoint(model, ckpt_dir, ckpt_tag):
    ckpt_path = os.path.join(ckpt_dir, ckpt_tag, "mp_rank_00_model_states.pt")
    print(f"[Score] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state['module'], strict=False)
    print(f"[Score] Missing={len(missing)}, Unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected (first 5): {unexpected[:5]}")
    return model


def run_inference(df_shard, model, tokenizer, max_len, batch_size, device, rank=0):
    """对 df_shard 推理，返回 np.array scores"""
    dataset = CorpusDataset(df_shard['corpus'].tolist(), tokenizer, max_len)
    sampler = None
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=sampler, num_workers=4, pin_memory=True,
    )
    scores = []
    model.eval()
    disable_tqdm = (rank != 0)
    for input_ids, attention_mask in tqdm(loader, desc=f"[GPU{rank}] Scoring", disable=disable_tqdm):
        s = model.get_score(input_ids.to(device), attention_mask.to(device))
        scores.append(s.float().cpu().numpy())
    return np.concatenate(scores, axis=0)


def compute_reward(day_df):
    """天内排名 + reward = label_30 * g(rank)"""
    scores  = day_df['score'].values
    order   = np.argsort(-scores)           # 降序
    ranks   = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    gain    = 1.0 / np.log2(ranks + 1)
    reward  = day_df['label_30'].values * gain
    out     = day_df.copy()
    out['rank']   = ranks
    out['reward'] = reward.astype(np.float32)
    return out


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",
        default="data/corpus_dataset.parquet")
    parser.add_argument("--ckpt_dir",
        default="checkpoints")
    parser.add_argument("--ckpt_tag",   default="tag-ff-80000")
    parser.add_argument("--pretrained_model",
        default="Qwen/Qwen1.5-1.8B")
    parser.add_argument("--out_file",
        default="data/corpus_dataset_with_reward.parquet")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--max_len",      type=int,   default=2000)
    parser.add_argument("--lora_r",       type=int,   default=8)
    parser.add_argument("--lora_alpha",   type=float, default=16.0)
    parser.add_argument("--k",            type=int,   default=128)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--max_samples",  type=int,   default=0,
        help="调试用：只取 dataset=3 的前 N 条，0 表示全量")
    args = parser.parse_args()

    # ── 分布式初始化（torchrun 时自动多卡，否则单卡）──
    use_dist = 'RANK' in os.environ
    if use_dist:
        dist.init_process_group(backend='nccl')
        local_rank  = int(os.environ['LOCAL_RANK'])
        rank        = int(os.environ['RANK'])
        world_size  = int(os.environ['WORLD_SIZE'])
    else:
        local_rank, rank, world_size = 0, 0, 1
    device = torch.device(f'cuda:{local_rank}')

    # ── 1. 加载数据（只推理 dataset=3）──
    if rank == 0:
        print(f"[Score] Loading data from {args.data_file}")
    df_full = pd.read_parquet(args.data_file)
    df_aug  = df_full[df_full['dataset'] == 3].reset_index(drop=False)   # 保留原 index
    if args.max_samples > 0:
        df_aug = df_aug.iloc[:args.max_samples].reset_index(drop=True)
        if rank == 0:
            print(f"[Score] DEBUG mode: truncated to {args.max_samples} samples")
    days    = sorted(df_aug['dt'].unique())
    if rank == 0:
        print(f"[Score] August rows: {len(df_aug)}, days: {days}")

    # ── 2. 构建模型 ──
    if rank == 0:
        print("[Score] Building model...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = ScoringModel(
        args.pretrained_model, args.lora_r, args.lora_alpha,
        args.k, args.dropout_rate)
    model = load_checkpoint(model, args.ckpt_dir, args.ckpt_tag)
    model = model.to(device)
    if rank == 0:
        print("[Score] Model ready.")

    # ── 3. 每个 rank 负责自己的分片推理 ──
    shard_indices = list(range(rank, len(df_aug), world_size))
    shard_df      = df_aug.iloc[shard_indices].reset_index(drop=True)
    shard_scores  = run_inference(shard_df, model, tokenizer,
                                  args.max_len, args.batch_size, device, rank)

    # ── 4. 汇总到 rank 0 ──
    if use_dist:
        gathered_indices = [None] * world_size
        gathered_scores  = [None] * world_size
        dist.all_gather_object(gathered_indices, shard_indices)
        dist.all_gather_object(gathered_scores,  shard_scores.tolist())
        if rank == 0:
            all_indices = []
            all_scores  = []
            for idx_list, sc_list in zip(gathered_indices, gathered_scores):
                all_indices.extend(idx_list)
                all_scores.extend(sc_list)
            # 按原始顺序重排
            order = np.argsort(all_indices)
            final_scores = np.array(all_scores)[order]
    else:
        final_scores = shard_scores

    # ── 5. rank 0 计算 reward 并保存 ──
    if rank == 0:
        df_aug = df_aug.reset_index(drop=True)
        df_aug['score'] = final_scores.astype(np.float32)

        print("[Score] Computing daily ranks and rewards...")
        df_aug = df_aug.groupby('dt', group_keys=False).apply(compute_reward)

        # 统计
        pos = df_aug['label_30'] == 1
        print(f"[Score] Positive samples: {pos.sum()}")
        print(f"  reward mean={df_aug.loc[pos,'reward'].mean():.4f}  "
              f"max={df_aug.loc[pos,'reward'].max():.4f}  "
              f"min={df_aug.loc[pos,'reward'].min():.6f}")
        print(f"  % reward>0 among positives: "
              f"{(df_aug.loc[pos,'reward']>0).mean()*100:.1f}%")

        # 写回全量 df
        df_full['score']  = np.nan
        df_full['rank']   = np.nan
        df_full['reward'] = np.float32(0.0)

        orig_idx = df_aug['index'].values       # 原始 df_full 中的行号
        df_full.loc[orig_idx, 'score']  = df_aug['score'].values
        df_full.loc[orig_idx, 'rank']   = df_aug['rank'].values
        df_full.loc[orig_idx, 'reward'] = df_aug['reward'].values

        print(f"[Score] Saving to {args.out_file} ...")
        df_full.to_parquet(args.out_file, index=False)
        print("[Score] Done.")

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
