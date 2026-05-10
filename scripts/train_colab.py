#!/usr/bin/env python3
"""
nanowhale 1B MoE Pretraining — Colab Edition (H100/A100, 256k context)

Pure PyTorch training loop. No DeepSpeed dependency.
Model ~1.2B total / ~400M active — fits comfortably on 80GB GPUs.

Supports:
- Curriculum context length (4k → 8k → 32k → 64k → 128k → 256k)
- BF16 mixed precision
- torch.compile (massive speedup on H100)
- Selective QAT (experts only)
- Checkpointing + resume
"""

import os
import sys
import argparse
import yaml
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
from safetensors.torch import save_file, load_file

from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_curriculum_stage(step, curriculum):
    """Return target seq_len for current training step."""
    if not curriculum or not curriculum.get("enabled", False):
        return None
    stages = sorted(curriculum["stages"], key=lambda s: s["steps"])
    current = stages[0]["seq_len"]
    for i, s in enumerate(stages):
        if step >= s["steps"]:
            current = s["seq_len"]
    return current


def save_checkpoint(model, optimizer, scaler, step, path, tokenizer):
    """Save full training state."""
    os.makedirs(path, exist_ok=True)
    save_file(model.state_dict(), os.path.join(path, "model.safetensors"))
    ckpt = {
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, os.path.join(path, "optimizer.pt"))
    tokenizer.save_pretrained(path)
    print(f"  [Checkpoint] Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="nanowhale 1B Colab Pretraining")
    parser.add_argument("--config", default="configs/1b_moe_colab_256k.yaml")
    parser.add_argument("--output", default="/content/checkpoints/1b_moe_colab")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    os.makedirs(args.output, exist_ok=True)
    print("=" * 60)
    print("nanowhale 1B MoE — Colab Pretraining")
    print(f"Target context: {model_cfg['max_position_embeddings']:,} tokens")
    print("=" * 60)

    # --- Model ---
    print("[1] Building model...")
    config = DeepseekV4Config(**model_cfg)
    model = DeepseekV4ForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    print(f"    {total:,} params ({total/1e9:.2f}B)")

    if torch.cuda.is_available():
        model = model.cuda()
        scaler = torch.amp.GradScaler("cuda")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM free: {torch.cuda.memory_reserved(0)/1e9:.1f} GB")
    else:
        scaler = None
        print("    WARNING: No GPU found! Running on CPU.")

    # torch.compile for H100 speedup
    if train_cfg.get("torch_compile", False):
        try:
            model = torch.compile(model, mode=train_cfg["torch_compile_mode"])
            print("    torch.compile: ON")
        except Exception as e:
            print(f"    torch.compile: SKIPPED ({e})")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
        fused=True if torch.cuda.is_available() else False,
    )

    start_step = 0
    if args.resume:
        print(f"[Resume] Loading from {args.resume}")
        state = load_file(os.path.join(args.resume, "model.safetensors"))
        model.load_state_dict(state, strict=False)
        ckpt = torch.load(os.path.join(args.resume, "optimizer.pt"), map_location="cpu")
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"    Resuming at step {start_step}")

    # --- Tokenizer ---
    tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Data ---
    print("[2] Loading data...")
    data_path = train_cfg.get("dataset_path", "data/processed/1b_moe_reasoning/train_final")
    ds = load_from_disk(data_path)
    print(f"    {len(ds):,} examples")

    max_len = train_cfg.get("max_seq_length", 262144)

    def tokenize(ex):
        ids = tokenizer.encode(ex["text"], truncation=True, max_length=max_len)
        return {"input_ids": ids, "labels": ids}

    tokenized = [tokenize(ex) for ex in ds]
    print(f"    Tokenized {len(tokenized)} sequences")

    # --- Training loop ---
    max_steps = train_cfg.get("max_steps", 50000)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 16)
    log_every = train_cfg.get("logging_steps", 20)
    save_every = train_cfg.get("save_steps", 5000)
    curriculum = train_cfg.get("curriculum", None)

    print(f"[3] Training {max_steps - start_step} steps "
          f"(effective batch={train_cfg['per_device_train_batch_size'] * grad_accum})...")

    model.train()
    losses = []
    micro_step = 0
    accumulated_loss = 0.0

    for step in range(start_step, max_steps):
        # Curriculum context length
        cur_len = get_curriculum_stage(step, curriculum)
        if cur_len and cur_len != max_len:
            max_len = cur_len
            # Re-tokenize with new length
            def retok(ex):
                ids = tokenizer.encode(ex["text"], truncation=True, max_length=max_len)
                return {"input_ids": ids, "labels": ids}
            tokenized = [retok(ex) for ex in ds]
            print(f"  [Curriculum] Step {step}: seq_len -> {max_len}")

        idx = step % len(tokenized)
        ids = tokenized[idx]

        if len(ids["input_ids"]) < 16:
            continue

        actual_len = min(len(ids["input_ids"]) - 1, max_len)
        input_ids = torch.tensor([ids["input_ids"][:actual_len]], dtype=torch.long)
        labels = torch.tensor([ids["labels"][1:actual_len+1]], dtype=torch.long)

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            try:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            except RuntimeError as e:
                print(f"  [WARN] Skipping batch: {e}")
                continue

        loss = loss / grad_accum
        scaler.scale(loss).backward()
        accumulated_loss += loss.item()

        micro_step += 1
        if micro_step % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses.append(accumulated_loss * grad_accum)
        accumulated_loss = 0.0

        if (step + 1) % log_every == 0 and losses:
            avg = sum(losses[-log_every:]) / len(losses[-log_every:])
            print(f"  Step {step+1:5d} | Loss: {avg:.4f} | Seq: {max_len}")

        if (step + 1) % save_every == 0:
            ckpt_dir = os.path.join(args.output, f"step_{step+1}")
            save_checkpoint(model, optimizer, scaler, step + 1, ckpt_dir, tokenizer)

    # --- Final ---
    final_dir = os.path.join(args.output, "final")
    save_checkpoint(model, optimizer, scaler, max_steps, final_dir, tokenizer)

    print("=" * 60)
    print("DONE. Model saved to " + final_dir)
    if losses:
        print(f"Average loss (last 100): {sum(losses[-100:])/100:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
