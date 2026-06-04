"""
train_grpo_rlpf.py  [salesloop版]

基于 as_llr_original/src/train_grpo_rlpf.py，新增：
  - pydantic v2 兼容 patch（deepspeed 0.11.1 + pydantic 2.x）

其余逻辑不变：
- 训练数据：corpus_dataset_zcy_with_reward*.parquet 中 dataset=3（August，已带预计算 reward）
- 测试数据：corpus_dataset_zcy.parquet 中 dataset=2（July 测试集）
- 标签：label_30（30天锁单，对应 SalesLoop 论文 T=30）
- GRPO reward：来自 score_for_reward.py 预计算的 R_i = label_30_i * 1/log2(rank_i+1)
- warm-start：从 tag-ff-80000 加载 LLM+LoRA 权重，strict=False
"""

# ── pydantic v2 兼容 patch（必须在 import deepspeed 之前）─────────
import pydantic as _pydantic
if int(_pydantic.VERSION.split('.')[0]) >= 2:
    from pydantic.fields import FieldInfo as _FieldInfo
    if not hasattr(_FieldInfo, 'required'):
        _FieldInfo.required = property(lambda self: self.is_required())
# ──────────────────────────────────────────────────────────────────

import argparse
import os
import numpy as np
import deepspeed
from modelscope import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DistributedSampler, Dataset, DataLoader
import pandas as pd
from datetime import datetime

from pairdata import BalancedBatchSampler
from layers import add_lora, append_llama_pro_block_group
from deepspeed.utils import log_dist
from optimizer import WarmupExponentialLR, WarmupCosineLR


# ══════════════════════════════════════════════════════════════
#  Logging
# ══════════════════════════════════════════════════════════════

def append_logtxt(log_path, message):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{t}\t{message}\n")


def precision_recall_at_k(y_true_np, y_score_np, k):
    k = min(k, len(y_true_np))
    if k == 0:
        return 0.0, 0.0
    idx_topk    = np.argsort(-y_score_np)[:k]
    tp_in_k     = y_true_np[idx_topk].sum()
    precision_k = tp_in_k / k
    recall_k    = tp_in_k / y_true_np.sum() if y_true_np.sum() > 0 else 0.0
    return precision_k, recall_k


# ══════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════

class MyDatasetRLPF(Dataset):
    """
    读 corpus_dataset_zcy_with_reward*.parquet (dataset=3)。
    label_30 → lock_label (BCE head)
    lock_label_str → CE head
    reward → 预计算 R_i（由 score_for_reward.py 生成）
    """
    def __init__(self, data, tokenizer, max_token_length):
        self.corpus    = data['corpus'].values
        self.label     = data['label_30'].values           # RLPF: label_30
        self.label_str = data['lock_label_str'].values
        self.reward    = data['reward'].values.astype(np.float32)   # RLPF: 预计算reward
        self.tokenizer = tokenizer
        self.max_len   = max_token_length

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        enc = self.tokenizer(
            [self.corpus[index]],
            max_length=self.max_len - 1,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        pad_id   = torch.tensor([self.tokenizer.pad_token_id])
        pad_mask = torch.tensor([1])
        input_ids      = torch.hstack((enc.input_ids,      pad_id.repeat(enc.input_ids.shape[0], 1)))
        attention_mask = torch.hstack((enc.attention_mask, pad_mask.repeat(enc.attention_mask.shape[0], 1)))

        ce_target = self.tokenizer(
            self.label_str[index],
            return_tensors='pt', max_length=1, padding='max_length'
        ).input_ids[:, -1].squeeze(-1)

        return (
            input_ids.squeeze(0),
            attention_mask.squeeze(0),
            torch.tensor(self.label[index],  dtype=torch.bfloat16),
            ce_target,
            torch.tensor(self.reward[index], dtype=torch.float32),   # RLPF: reward
        )


class MyDatasetEval(Dataset):
    """用于评估的数据集，读 corpus_dataset_zcy.parquet dataset=2"""
    def __init__(self, data, tokenizer, max_token_length):
        self.corpus    = data['corpus'].values
        self.label     = data['label_30'].values
        self.label_str = data['lock_label_str'].values
        self.tokenizer = tokenizer
        self.max_len   = max_token_length

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        enc = self.tokenizer(
            [self.corpus[index]],
            max_length=self.max_len - 1,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        pad_id   = torch.tensor([self.tokenizer.pad_token_id])
        pad_mask = torch.tensor([1])
        input_ids      = torch.hstack((enc.input_ids,      pad_id.repeat(enc.input_ids.shape[0], 1)))
        attention_mask = torch.hstack((enc.attention_mask, pad_mask.repeat(enc.attention_mask.shape[0], 1)))
        ce_target = self.tokenizer(
            self.label_str[index],
            return_tensors='pt', max_length=1, padding='max_length'
        ).input_ids[:, -1].squeeze(-1)
        return (
            input_ids.squeeze(0),
            attention_mask.squeeze(0),
            torch.tensor(self.label[index], dtype=torch.bfloat16),
            ce_target,
        )


# ══════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════

def load_data_rlpf(reward_data_file, eval_data_file, world_size, pos_oversample=9, max_train_samples=0):
    """
    RLPF: 训练用 August 带 reward 数据，评估用 July 测试集。
    """
    # 训练：dataset=3（August，带 reward）
    df_train = pd.read_parquet(reward_data_file)
    df_train = df_train[df_train['dataset'] == 3].reset_index(drop=True)
    if max_train_samples > 0:
        df_train = df_train.sample(min(max_train_samples, len(df_train)), random_state=42).reset_index(drop=True)
    # lock_label_str 由 label_30 生成（若无则创建）
    if 'lock_label_str' not in df_train.columns:
        df_train['lock_label_str'] = df_train['label_30'].apply(lambda x: '是' if x == 1 else '否')
    log_dist(message=f"[Data] train(Aug) rows={len(df_train)}, "
                     f"pos_rate={df_train['label_30'].mean():.4%}, "
                     f"reward>0: {(df_train['reward']>0).sum()}", ranks=[0])

    # 正样本过采样（保证每个 batch 有足够正样本）
    if pos_oversample > 0:
        pos_df = df_train[df_train['label_30'] == 1].reset_index(drop=True)
        df_train = pd.concat([df_train] + [pos_df] * pos_oversample, ignore_index=True)

    n = (len(df_train) // world_size) * world_size
    df_train = df_train.sample(n, random_state=42).reset_index(drop=True)
    log_dist(message=f"[Data] after oversample train={len(df_train)}", ranks=[0])

    # 评估：dataset=2（July 测试集）
    df_eval = pd.read_parquet(eval_data_file)
    df_eval = df_eval[df_eval['dataset'] == 2].reset_index(drop=True)
    if 'lock_label_str' not in df_eval.columns:
        df_eval['lock_label_str'] = df_eval['label_30'].apply(lambda x: '是' if x == 1 else '否')
    log_dist(message=f"[Data] eval(Jul) rows={len(df_eval)}, "
                     f"pos_rate={df_eval['label_30'].mean():.4%}", ranks=[0])

    return df_train, df_eval


# ══════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════

class TaskLayer(nn.Module):
    def __init__(self, hidden_size, embedding_size=128, dropout_rate=0.5):
        super().__init__()
        self.fc0  = nn.Linear(hidden_size, embedding_size, dtype=torch.bfloat16)
        self.drop = nn.Dropout(p=dropout_rate)
        self.act  = nn.GELU()
        self.fc1  = nn.Linear(embedding_size, 1, dtype=torch.bfloat16)

    def forward(self, x):
        h = self.act(self.drop(self.fc0(x)))
        return self.fc1(h), h


class LanguageModelWithLinear(nn.Module):
    def __init__(self, pretrained_model_name, lora_r, lora_alpha,
                 llama_pro_group_size=-1, k=128, dropout_rate=0.5,
                 cl_tau=0, pissa=False,
                 grpo_gamma=0.1, alpha_bce=0.5, alpha_ce=0.5):
        super().__init__()
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            use_cache=False,
        )
        log_dist(message=f"{self.pretrained_model}", ranks=[0])
        if llama_pro_group_size > 0:
            append_llama_pro_block_group(self.pretrained_model, llama_pro_group_size)
        else:
            if lora_r > 0:
                add_lora(self.pretrained_model, lora_r, lora_alpha, pissa)
        self.custom_linear = TaskLayer(self.pretrained_model.config.hidden_size, k, dropout_rate)
        self.grpo_gamma    = grpo_gamma
        self.alpha_bce     = alpha_bce
        self.alpha_ce      = alpha_ce
        self.bce_loss_fn   = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss_fn    = nn.CrossEntropyLoss(reduction='sum')
        log_dist(message="LanguageModelWithLinear (GRPO-RLPF) COMPLETE", ranks=[0])

    # ── RLPF: 使用预计算 reward ─────────────────────────────────
    def grpo_listwise_loss(self, scores, rewards):
        """
        RLPF 版 Discriminative GRPO listwise loss。
        scores  : (B,)  模型当前 logit（无 sigmoid）
        rewards : (B,)  预计算的 R_i = label_30_i * 1/log2(global_rank_i + 1)
                         来自 score_for_reward.py，训练中固定，不依赖当前模型。
        """
        with torch.no_grad():
            mean_r     = rewards.mean()
            std_r      = rewards.std().clamp(min=1e-8)
            advantages = (rewards - mean_r) / std_r                        # 优势归一化
            target     = torch.softmax(advantages / self.grpo_gamma, dim=0)  # 目标分布

        log_probs = torch.log_softmax(scores.float(), dim=0)
        loss      = -(target * log_probs).sum()
        return loss

    def forward(self, input_ids, attention_mask, bce_target, ce_target, reward):
        outputs      = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden  = outputs.hidden_states[-1][:, -1, :]
        scores, _    = self.custom_linear(last_hidden)
        scores       = scores.squeeze(-1)
        ce_logits    = outputs.logits[:, -1, :]

        grpo_loss = self.grpo_listwise_loss(scores, reward)   # RLPF: 传入预计算 reward
        bce_loss  = self.bce_loss_fn(scores, bce_target)
        ce_loss   = self.ce_loss_fn(ce_logits, ce_target)
        loss      = grpo_loss + self.alpha_bce * bce_loss + self.alpha_ce * ce_loss

        return loss, scores, ce_logits, grpo_loss, bce_loss, ce_loss


# ══════════════════════════════════════════════════════════════
#  Evaluate（使用 MyDatasetEval，label_30 为目标）
# ══════════════════════════════════════════════════════════════

def evaluate(test_data, model_engine, tokenizer, max_len, ii, threshold=0.1):
    eval_logtxt_path = os.path.join(args.out_dir, "eval_log.txt")
    if model_engine.local_rank == 0 and not os.path.exists(eval_logtxt_path):
        with open(eval_logtxt_path, 'w', encoding='utf-8') as f:
            f.write("time\tstep\teval_auc\tprecision\trecall\tf1\tpos_num\tneg_num\ttopk\n")

    log_dist(message="----------- Evaluation (label_30) --------------", ranks=[0])
    rank = model_engine.local_rank

    dataset_test    = MyDatasetEval(test_data, tokenizer, max_len)
    dataset_sampler = DistributedSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, sampler=dataset_sampler)
    log_dist(message=f"Eval dataset size: {len(dataset_test)}", ranks=[0])

    y_true_test = torch.tensor([], device=rank)
    y_pred_test = torch.tensor([], device=rank)

    model_engine.eval()
    for batch_idx, (input_ids, attention_mask, label, label_str) in enumerate(dataloader_test):
        input_ids      = input_ids.to(rank)
        attention_mask = attention_mask.to(rank)
        label          = label.to(rank)
        label_str      = label_str.to(rank)
        # eval 时 reward 用 dummy 0（只需 scores）
        dummy_reward = torch.zeros(input_ids.shape[0], dtype=torch.float32, device=rank)
        with torch.no_grad():
            _, scores, _, _, _, _ = model_engine(
                input_ids, attention_mask, label, label_str, dummy_reward)
            y_pred = torch.sigmoid(scores).detach()
            y_true_test = torch.hstack((y_true_test, label))
            y_pred_test = torch.hstack((y_pred_test, y_pred))
        if batch_idx % 500 == 0:
            log_dist(message=f"eval step={batch_idx}", ranks=[0])

    y_true_all = [torch.zeros_like(y_true_test) for _ in range(model_engine.world_size)]
    y_pred_all = [torch.zeros_like(y_pred_test) for _ in range(model_engine.world_size)]
    torch.distributed.all_gather(y_true_all, y_true_test)
    torch.distributed.all_gather(y_pred_all, y_pred_test)

    y_true  = torch.cat(y_true_all, dim=0).detach().cpu().float().numpy()
    y_score = torch.cat(y_pred_all, dim=0).detach().cpu().float().numpy()

    AUC        = roc_auc_score(y_true=y_true, y_score=y_score)
    y_pred_lbl = (y_score > threshold).astype(int)
    P  = precision_score(y_true, y_pred_lbl, zero_division=0)
    R  = recall_score(y_true,    y_pred_lbl, zero_division=0)
    F1 = f1_score(y_true,        y_pred_lbl, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_lbl)

    K_LIST = [100, 500, 1000, 5000, 10000]
    topk_strs = []
    for k_val in K_LIST:
        p_k, r_k = precision_recall_at_k(y_true, y_score, k_val)
        topk_strs.append(f"P@{k_val}={p_k:.4f},R@{k_val}={r_k:.4f}")
    topk_info = " | ".join(topk_strs)

    log_dist(message=(
        f"\n\nEval AUC:{AUC:.6f}  P@{threshold}:{P:.4f}  R@{threshold}:{R:.4f}  F1:{F1:.4f}\n"
        f"Confusion matrix:\n{cm}\n[Top-k] {topk_info}"
    ), ranks=[0])

    if model_engine.local_rank == 0:
        if model_engine.monitor.tb_monitor is not None:
            sw = model_engine.monitor.tb_monitor.summary_writer
            sw.add_scalar("eval/auc",       AUC, ii)
            sw.add_scalar("eval/precision", P,   ii)
            sw.add_scalar("eval/recall",    R,   ii)
        pos_num = int(y_true.sum())
        neg_num = len(y_true) - pos_num
        txt = f"{ii}\t{AUC:.4f}\t{P:.4f}\t{R:.4f}\t{F1:.4f}\t{pos_num}\t{neg_num}\t{topk_info}"
        append_logtxt(eval_logtxt_path, txt)

    model_engine.train()
    log_dist(message="Evaluation Completed!", ranks=[0])


# ══════════════════════════════════════════════════════════════
#  Train
# ══════════════════════════════════════════════════════════════

def train(args, pretrained_model_name):
    threshold  = args.pr_threshold
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))

    os.makedirs(args.out_dir, exist_ok=True)
    logtxt_path = os.path.join(args.out_dir, "train_log.txt")
    if rank == 0 and not os.path.exists(logtxt_path):
        with open(logtxt_path, 'w', encoding='utf-8') as f:
            f.write("time\tepoch\tstep\tloss\tgrpo\tbce\tce\tauc\tpre\trec\tf1\ttopk\n")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # model
    new_model = LanguageModelWithLinear(
        pretrained_model_name,
        args.lora_r, args.lora_alpha,
        args.llama_pro_group_size,
        args.k, args.dropout_rate,
        args.cl_tau, args.pissa,
        grpo_gamma=args.grpo_gamma,
        alpha_bce=args.alpha_bce,
        alpha_ce=args.alpha_ce,
    )
    if rank == 0:
        print(new_model)

    # data：训练用 August reward 数据，评估用 July 测试集
    train_data, test_data = load_data_rlpf(
        args.data_file, args.eval_data_file, world_size,
        pos_oversample=args.pos_oversample,
        max_train_samples=args.max_train_samples)
    dataset1 = MyDatasetRLPF(train_data, tokenizer, args.max_len)

    # param groups
    params_g1, params_g2 = [], []
    for name, param in new_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("custom_linear"):
            params_g1.append(param)
        else:
            if args.llama_pro_group_size <= 0:
                if 'lora' not in name:
                    param.requires_grad = False
                else:
                    params_g2.append(param)
            else:
                if 'mlp.down_proj' in name or 'self_attn.o_proj' in name:
                    params_g1.append(param)
                else:
                    params_g2.append(param)

    lr2 = args.lr / 20.0
    optimizer_ = torch.optim.AdamW([
        {"params": params_g1, "lr": args.lr, "name": "task"},
        {"params": params_g2, "lr": lr2,     "name": "llm"},
    ], lr=args.lr)

    # RLPF: BalancedBatchSampler 保证每 batch 有正样本（reward>0 的样本）
    balanced_sampler = BalancedBatchSampler(dataset1, batch_size=args.batch_size)
    train_dataloader = DataLoader(dataset1, batch_sampler=balanced_sampler)

    total_steps = len(train_dataloader) * args.epochs
    if args.lr_scheduler_type == "cosine":
        lr_scheduler_ = WarmupCosineLR(
            optimizer=optimizer_, warmup_steps=args.warmup_step_num,
            total_steps=total_steps, min_lr_ratio=0.1)
    else:
        lr_scheduler_ = WarmupExponentialLR(
            optimizer=optimizer_, gamma=0.9999, warmup_step=args.warmup_step_num)

    parameters = filter(lambda p: p.requires_grad, new_model.parameters())
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args, model=new_model, optimizer=optimizer_,
        lr_scheduler=lr_scheduler_, model_parameters=parameters,
        training_data=dataset1,
    )

    # RLPF: warm-start，load_module_strict=False
    # task head 参数名不同（fc0/fc1 vs custom_linear_0/custom_linear_1），
    # 接受 task head 随机初始化，LLM+LoRA 权重正常加载
    if args.load_ckpt_dir and args.load_ckpt_tag:
        log_dist(message=f"[Warm-start] {args.load_ckpt_dir} tag={args.load_ckpt_tag} "
                         f"(load_module_strict=False)", ranks=[0])
        model_engine.load_checkpoint(
            load_dir=args.load_ckpt_dir,
            tag=args.load_ckpt_tag,
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
            load_module_strict=False,
        )
        log_dist(message="[Warm-start] LLM+LoRA weights loaded.", ranks=[0])

    for name, param in model_engine.named_parameters():
        if param.requires_grad:
            log_dist(message=f"trainable: {name}", ranks=[0])
    log_dist(message=f"steps per epoch: {len(train_dataloader)}", ranks=[0])

    device = model_engine.local_rank
    y_true_tensor = torch.tensor([], device=device)
    y_pred_tensor = torch.tensor([], device=device)
    ii = 0

    for epoch in range(args.epochs):
        log_dist(message=f"Epoch {epoch+1}/{args.epochs}", ranks=[0])

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids, attention_mask, label, label_str, reward = batch
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label          = label.to(device)
            label_str      = label_str.to(device)
            reward         = reward.to(device)        # RLPF: 预计算 reward

            loss, scores, _, grpo_loss, bce_loss, ce_loss = model_engine(
                input_ids, attention_mask, label, label_str, reward)

            y_pred = torch.sigmoid(scores).detach()
            y_true_tensor = torch.hstack((y_true_tensor, label))
            y_pred_tensor = torch.hstack((y_pred_tensor, y_pred))

            if ii % 10 == 0:
                log_dist(message=(
                    f"step={ii} loss={loss.item():.4f} "
                    f"grpo={grpo_loss.item():.4f} "
                    f"bce={bce_loss.item():.4f} "
                    f"ce={ce_loss.item():.4f} "
                    f"reward_mean={reward.mean().item():.4f}"
                ), ranks=[0])
                for pg in optimizer.param_groups:
                    log_dist(message=f"{pg['name']} lr={pg['lr']:.2e}", ranks=[0])
                if model_engine.monitor.tb_monitor is not None:
                    sw = model_engine.monitor.tb_monitor.summary_writer
                    sw.add_scalar("loss/total", loss.item(),      ii)
                    sw.add_scalar("loss/grpo",  grpo_loss.item(), ii)
                    sw.add_scalar("loss/bce",   bce_loss.item(),  ii)
                    sw.add_scalar("loss/ce",    ce_loss.item(),   ii)

            if (ii + 1) % 100 == 0 and model_engine.local_rank == 0:
                y_true_np = y_true_tensor.cpu().float().numpy()
                y_pred_np = y_pred_tensor.cpu().float().numpy()
                try:
                    auc = roc_auc_score(y_true_np, y_pred_np)
                except Exception:
                    auc = 0.0
                y_pred_lbl = (y_pred_np > 0.5).astype(int)
                pre = precision_score(y_true_np, y_pred_lbl, zero_division=0)
                rec = recall_score(y_true_np,    y_pred_lbl, zero_division=0)
                f1  = f1_score(y_true_np,        y_pred_lbl, zero_division=0)
                K_LIST = [100, 500, 1000, 5000, 10000]
                topk_strs = [
                    f"P@{k}={precision_recall_at_k(y_true_np, y_pred_np, k)[0]:.4f},"
                    f"R@{k}={precision_recall_at_k(y_true_np, y_pred_np, k)[1]:.4f}"
                    for k in K_LIST
                ]
                topk_info = " | ".join(topk_strs)
                log_dist(message=f"[Train Top-k step={ii}] {topk_info}", ranks=[0])
                pos_num = int(y_true_np.sum())
                neg_num = len(y_true_np) - pos_num
                txt = (f"{epoch}\t{ii}\t{loss.item():.4f}\t{grpo_loss.item():.4f}\t"
                       f"{bce_loss.item():.4f}\t{ce_loss.item():.4f}\t"
                       f"{auc:.4f}\t{pre:.4f}\t{rec:.4f}\t{f1:.4f}\t"
                       f"{pos_num}\t{neg_num}\t{topk_info}")
                append_logtxt(logtxt_path, txt)

            model_engine.backward(loss)
            model_engine.step()

            if ii > 0 and ii % args.ckpt_interval == 0:
                n_sample    = max(1, len(test_data) // 10)
                test_sample = test_data.sample(n_sample, random_state=42).reset_index(drop=True)
                evaluate(test_sample, model_engine, tokenizer, args.max_len, ii, threshold=threshold)
                model_engine.save_checkpoint(
                    save_dir=args.out_dir,
                    client_state={"loss": loss.item()},
                    tag=f"tag-ff-{ii}",
                    save_latest=True,
                )
            ii += 1

    model_engine.save_checkpoint(
        save_dir=args.out_dir,
        client_state={"loss": loss.item()},
        tag="tag-ff-final",
        save_latest=True,
    )
    evaluate(test_data, model_engine, tokenizer, args.max_len,
             len(train_dataloader) * args.epochs + 1, threshold=threshold)


# ══════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════

def add_argument():
    parser = argparse.ArgumentParser(description="train_grpo_rlpf")

    # paths
    parser.add_argument("--out_dir", default="outputs/grpo_rlpf", type=str)
    parser.add_argument("--data_file",
        default="data/corpus_dataset_with_reward.parquet",
        type=str, help="训练数据，需包含 dataset=3 样本与 reward 列")
    parser.add_argument("--eval_data_file",
        default="data/corpus_dataset_eval.parquet",
        type=str, help="评估数据，需包含 dataset=2 样本")
    parser.add_argument("--pretrained_model",
        default="Qwen/Qwen1.5-1.8B",
        type=str)

    # warm-start
    parser.add_argument("--load_ckpt_dir",  default="", type=str,
        help="tag-ff-80000 所在父目录")
    parser.add_argument("--load_ckpt_tag",  default="", type=str,
        help="如 tag-ff-80000")

    # GRPO
    parser.add_argument("--grpo_gamma", default=0.1,  type=float)
    parser.add_argument("--alpha_bce",  default=0.5,  type=float)
    parser.add_argument("--alpha_ce",   default=0.5,  type=float)

    # training
    parser.add_argument("--batch_size",           default=64,     type=int)
    parser.add_argument("--epochs",               default=3,      type=int)
    parser.add_argument("--lr",                   default=1e-5,   type=float)
    parser.add_argument("--ckpt_interval",        default=5000,   type=int)
    parser.add_argument("--warmup_step_num",      default=2000,   type=int)
    parser.add_argument("--local_rank",           default=-1,     type=int)
    parser.add_argument("--log-interval",         default=20,     type=int)
    parser.add_argument("--lora_r",               default=16,     type=int)
    parser.add_argument("--lora_alpha",           default=32.0,   type=float)
    parser.add_argument("--pissa",                default=False,  action="store_true")
    parser.add_argument("--max_len",              default=2000,   type=int)
    parser.add_argument("--dropout_rate",         default=0.5,    type=float)
    parser.add_argument("--k",                    default=128,    type=int)
    parser.add_argument("--cl_tau",               default=0,      type=float)
    parser.add_argument("--llama_pro_group_size", default=0,      type=int)
    parser.add_argument("--pr_threshold",         default=0.1,    type=float)
    parser.add_argument("--lr_scheduler_type",    default="cosine",
        choices=["cosine", "exponential"])
    parser.add_argument("--pos_oversample",       default=9,      type=int,
        help="正样本过采样倍数，保证 batch 内有正样本")
    parser.add_argument("--max_train_samples",    default=0,      type=int,
        help="截断训练集，0=不截断（验证用）")

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = add_argument()
    train(args, pretrained_model_name=args.pretrained_model)
