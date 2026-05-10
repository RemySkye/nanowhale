#!/usr/bin/env python3
"""
nanowhale 1B MoE Pretraining Script

Stable pure-PyTorch training loop for the 1B-scale DeepSeek-V4 MoE model.
Verified on RTX 3080 Ti (12GB VRAM).

Key features:
- Pure PyTorch loop (no HF Trainer quirks)
- BF16 mixed precision
- Gradient clipping + checkpoints
- Safety guards for index errors
"""

import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast

from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM


def safe_forward(model, input_ids, labels):
    """Forward pass with error recovery for index issues."""
    try:
        outputs = model(input_ids=input_ids, labels=labels)
        return outputs.loss
    except RuntimeError as e:
        if "assert" in str(e).lower() or "index" in str(e).lower():
            print("  [WARN] Skipping problematic batch")
            return None
        raise


def main():
    parser = argparse.ArgumentParser(description="nanowhale 1B MoE Pretraining")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--data", type=str, default="data/processed/1b_moe_data_ready/final_train")
    parser.add_argument("--output", type=str, default="checkpoints/1b_moe_pretrain")
    args = parser.parse_args()

    print("=" * 60)
    print("nanowhale 1B MoE Pretraining")
    print("=" * 60)

    model_cfg = dict(
        vocab_size=128000,
        hidden_size=768,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=1,
        head_dim=128,
        qk_rope_head_dim=32,
        q_lora_rank=384,
        o_groups=4,
        o_lora_rank=192,
        moe_intermediate_size=2048,
        n_routed_experts=12,
        n_shared_experts=2,
        num_experts_per_tok=2,
        hc_mult=2,
        max_position_embeddings=4096,
    )

    print("[1] Building model...")
    config = DeepseekV4Config(**model_cfg)
    model = DeepseekV4ForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    print(f"    {total:,} parameters ({total/1e9:.2f}B)")

    if torch.cuda.is_available():
        model = model.cuda()
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    print("[2] Loading data...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_from_disk(args.data)
    subset = ds.select(range(min(len(ds), 800)))

    def tokenize(ex):
        ids = tokenizer.encode(ex["text"], truncation=True, max_length=args.max_len)
        return {"input_ids": ids, "labels": ids}

    tokenized = [tokenize(ex) for ex in subset]
    print(f"    {len(tokenized)} examples, max_len={args.max_len}")

    os.makedirs(args.output, exist_ok=True)

    print(f"[3] Training {args.steps} steps...")
    model.train()
    losses = []

    for step in range(args.steps):
        idx = step % len(tokenized)
        ids = tokenized[idx]

        if len(ids["input_ids"]) < 16:
            continue

        input_ids = torch.tensor([ids["input_ids"][:-1]], dtype=torch.long)
        labels = torch.tensor([ids["labels"][1:]], dtype=torch.long)

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = safe_forward(model, input_ids, labels)
            if loss is None:
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = safe_forward(model, input_ids, labels)
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 20 == 0:
            avg = sum(losses[-20:]) / len(losses[-20:])
            print(f"  Step {step+1:4d} | Loss: {avg:.4f}")

        if (step + 1) % 100 == 0:
            ckpt = os.path.join(args.output, f"step_{step+1}")
            os.makedirs(ckpt, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt, "model.pt"))
            tokenizer.save_pretrained(ckpt)
            print(f"  [Saved] {ckpt}")

    final = os.path.join(args.output, "final")
    os.makedirs(final, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final, "model.pt"))
    tokenizer.save_pretrained(final)

    print("=" * 60)
    print("DONE. Model saved to " + final)
    print(f"Average loss (last 30): {sum(losses[-30:])/30:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
