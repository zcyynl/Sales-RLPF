#!/usr/bin/env python3
"""
GRPO evaluation on fc_valid_data_v0806.pkl (48w test set)
Aligns with baseline_methods.md evaluation metrics and output format.
"""

import os
import sys
import argparse
import pickle
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from torch.utils.data import DataLoader

# Import from existing modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset0806_eval import data_processing, MyDataset
from train_grpo_rlpf import LanguageModelWithLinear, precision_recall_at_k

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_checkpoint(ckpt_dir):
    """Find checkpoint file in directory (handles ZeRO-2 naming and module/model wrapper)"""
    import glob

    # Patterns to try
    patterns = [
        os.path.join(ckpt_dir, "mp_rank_00_model_states.pt"),
        os.path.join(ckpt_dir, "model.pt"),
        os.path.join(ckpt_dir, "*.pt"),
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    files = list(set(files))
    if not files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    ckpt_path = files[0]
    print(f"[Info] Found checkpoint: {ckpt_path}")

    # Handle ZeRO-2 wrapped state dict
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Unwrap common wrappers
    for key in ["module", "model"]:
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break

    # Try common key names
    if "state_dict" in sd:
        sd = sd["state_dict"]

    # Remove lora_a/lora_b naming (LoRA params, will fail gracefully with strict=False)
    new_sd = {}
    for k, v in sd.items():
        new_sd[k] = v
    sd = new_sd

    return sd


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO model on fc_valid dataset")
    parser.add_argument("--data_file", type=str, default="data/fc_valid.pkl",
                        help="Path to evaluation pickle file")
    parser.add_argument("--ckpt_dir", type=str,
                        default="outputs/grpo_rlpf/tag-ff-final",
                        help="Checkpoint directory")
    parser.add_argument("--step_tag", type=str, default="50k", help="Step tag for logging")
    parser.add_argument("--pretrained_model", type=str, default="Qwen/Qwen1.5-1.8B",
                        help="Base pretrained model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--max_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--k", type=int, default=16, help="Context window k")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--topk", type=str, default="100,500,1000,5000,10000",
                        help="Top-K values for precision/recall (comma-separated)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GRPO Evaluation on fc_valid dataset")
    print(f"{'='*60}\n")

    # ─── Data loading ──────────────────────────────────────────────────────────
    print("[1/5] Loading test data...")
    _, test_df = data_processing(world_size=1, path=args.data_file)
    print(f"  Loaded {len(test_df):,} samples")
    print(f"  Positive samples: {test_df['label'].sum():,} ({100*test_df['label'].mean():.2f}%)")

    # ─── Tokenizer setup ───────────────────────────────────────────────────────
    print("\n[2/5] Initializing tokenizer...")
    from transformers import AutoTokenizer, AutoConfig
    config = AutoConfig.from_pretrained(args.pretrained_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        trust_remote_code=True,
        pad_token=config.eos_token,
        padding_side="left",  # Important: left padding for consideration models
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")

    # ─── Model initialization ──────────────────────────────────────────────────
    print("\n[3/5] Initializing model...")
    model = LanguageModelWithLinear(
        pretrained_model_name=args.pretrained_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        k=args.k,
        dropout_rate=args.dropout_rate,
    ).to(DEVICE)

    # Load checkpoint
    weight_file = find_checkpoint(args.ckpt_dir)
    model.load_state_dict(weight_file, strict=False)
    print(f"  Loaded checkpoint from {args.ckpt_dir}")

    model.eval()
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  LoRA params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ─── Build dataloader ─────────────────────────────────────────────────────
    print("\n[4/5] Building dataloader...")
    corpus = MyDataset(test_df, tokenizer, args.max_len)
    sampler = torch.utils.data.DistributedSampler(corpus, shuffle=False) if torch.distributed.is_initialized() else None
    dataloader = DataLoader(
        corpus,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total batches: {len(dataloader)}")

    # ─── Inference and evaluation ──────────────────────────────────────────────
    print("\n[5/5] Running inference...")

    all_prob = []
    all_label = []

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, label, label_str) in enumerate(dataloader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            label = label.to(DEVICE)

            # GRPO forward signature: (input_ids, attention_mask, bce_target, ce_target, reward)
            # For evaluation: use dummy reward (zero tensor)
            dummy_reward = torch.zeros(args.batch_size, dtype=torch.float32, device=DEVICE)

            # Forward pass - returns (loss, scores, ce_logits, grpo_loss, bce_loss, ce_loss)
            loss, scores, ce_logits, grpo_loss, bce_loss, ce_loss = model(
                input_ids, attention_mask, label, label_str, dummy_reward
            )

            # Sigmoid to get probabilities
            prob = torch.sigmoid(scores).squeeze(-1).cpu().numpy()
            all_prob.append(prob)
            all_label.extend(label.numpy())

            if (batch_idx + 1) % 100 == 0:
                print(f"    Progress: {batch_idx + 1}/{len(dataloader)}")

    y_prob = np.concatenate(all_prob)
    y_true = np.array(all_label)

    # ─── Compute metrics ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    # AUC
    auc = roc_auc_score(y_true, y_prob)
    print(f"\nAUC              : {auc:.6f}")

    # Classification metrics
    y_hat = (y_prob >= args.threshold).astype(int)
    P = precision_score(y_true, y_hat, zero_division=0)
    R = recall_score(y_true, y_hat, zero_division=0)
    F1 = f1_score(y_true, y_hat, zero_division=0)
    cm = confusion_matrix(y_true, y_hat)

    print(f"\nPrecision@50%     : {P:.4f}")
    print(f"Recall@50%       : {R:.4f}")
    print(f"F1               : {F1:.4f}")
    print(f"Confusion matrix :\n{cm}")

    # Top-K metrics
    topk_vals = [int(x) for x in args.topk.split(",")]
    print(f"\nTop-K Precision/Recall:")
    n_total = len(y_true)
    for k in topk_vals:
        idx_topk = np.argsort(y_prob)[-k:]
        y_true_topk = y_true[idx_topk]
        p_k = y_true_topk.mean()
        r_k = y_true_topk.sum() / y_true.sum() if y_true.sum() > 0 else 0
        print(f"  P@{k:5d}: {p_k:.4f} , R@{k:5d}: {r_k:.4f}")

    # Per-sample precision@100/500/1000
    print(f"\nPer-sample @ varying k:")
    for k in [100, 500, 1000]:
        p_k, r_k = precision_recall_at_k(y_true, y_prob, k)
        print(f"  P@{k}: {p_k:.4f} , R@{k}: {r_k:.4f}")

    print("\n" + "="*60)

    # Save prediction for later analysis
    save_path = args.ckpt_dir.replace("tag-ff-final", "grpo_fc_valid_pred").replace("final", "pred")
    os.makedirs(save_path, exist_ok=True)
    pred_file = os.path.join(save_path, f"step_{args.step_tag}_preds.npy")
    np.savez_compressed(
        pred_file,
        y_prob=y_prob,
        y_true=y_true,
        meta_df=test_df,
    )
    print(f"\n[Info] Saved predictions to: {pred_file}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
