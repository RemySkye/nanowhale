#!/usr/bin/env python3
"""
Expanded data preparation for nanowhale-1B-MoE (Colab-scale, 256k context)
Reasoning-heavy mix: general English + code + math + logic + long-context

100% public datasets — all verified available on HuggingFace Hub.
"""

import os
import argparse
from datasets import load_dataset, concatenate_datasets, Dataset

DEFAULT_TOKENIZER = "tokenizer"

# Reasonably-high-quality mix with reasoning emphasis
DATA_MIX = {
    # --- General English (still the backbone) ---
    "fineweb_2": {
        "name": "HuggingFaceFW/fineweb-2",
        "split": "train",
        "streaming": True,
        "weight": 0.30,
        "min_chars": 300,
        "purpose": "general-knowledge",
    },
    "c4_en": {
        "name": "allenai/c4",
        "subset": "en",
        "split": "train",
        "streaming": True,
        "weight": 0.20,
        "min_chars": 350,
        "purpose": "clean-web-text",
    },
    "finefineweb": {
        "name": "m-a-p/FineFineWeb",
        "split": "train",
        "streaming": True,
        "weight": 0.12,
        "min_chars": 400,
        "purpose": "curated-high-quality-web",
    },
    # --- Code ---
    "code_contests": {
        "name": "deepmind/code_contests",
        "split": "train",
        "streaming": False,
        "weight": 0.10,
        "min_chars": 128,
        "purpose": "competitive-coding-reasoning",
    },
    "mbpp": {
        "name": "google-research-datasets/mbpp",
        "split": "train",
        "streaming": True,
        "weight": 0.08,
        "min_chars": 64,
        "purpose": "python-reasoning",
    },
    # --- Math reasoning ---
    "gsm8k": {
        "name": "openai/gsm8k",
        "subset": "main",
        "split": "train",
        "streaming": False,
        "weight": 0.10,
        "min_chars": 64,
        "purpose": "math-reasoning",
    },
    # --- Long-context (packed from fineweb-2) ---
    "long_context": {
        "name": "HuggingFaceFW/fineweb-2",
        "split": "train",
        "streaming": True,
        "weight": 0.10,
        "pack_long": True,
        "purpose": "256k-long-context",
    },
}


def extract_text(sample, key):
    """Normalize text extraction across dataset formats."""
    if "text" in sample:
        return sample["text"]
    if "content" in sample:
        return sample["content"]
    if "prompt" in sample:
        return sample["prompt"]
    if "question" in sample:
        q = sample["question"]
        a = sample.get("answer", "") or sample.get("solution", "")
        return f"Q: {q}\nA: {a}"
    if "problem" in sample:
        p = sample["problem"]
        s = sample.get("solution", "") or sample.get("solutions", "")
        return f"Problem: {p}\nSolution: {s}"
    if "description" in sample:
        return sample["description"]
    if "messages" in sample:
        return " ".join(m.get("content", "") for m in sample["messages"])
    return str(sample)


def load_dataset_safe(name, **kwargs):
    """Safe dataset loading with fallback for config/subset."""
    try:
        return load_dataset(name, **kwargs)
    except Exception:
        # Try without subset/config
        kwargs.pop("name", None)
        kwargs.pop("subset", None)
        return load_dataset(name, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/1b_moe_reasoning")
    parser.add_argument("--max_per_dataset", type=int, default=15000)
    parser.add_argument("--pack_long_to", type=int, default=65536)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    all_examples = []

    for key, cfg in DATA_MIX.items():
        print(f"\n{'='*50}")
        print(f"[{key}] {cfg['purpose']} (weight={cfg['weight']})")
        print(f"{'='*50}")

        load_kwargs = {"split": cfg["split"]}
        if "subset" in cfg:
            load_kwargs["name"] = cfg["subset"]
        if cfg["streaming"]:
            load_kwargs["streaming"] = True

        try:
            ds = load_dataset(cfg["name"], **load_kwargs)
        except Exception as e:
            print(f"  [SKIP] Cannot load: {e}")
            continue

        min_chars = cfg.get("min_chars", 200)
        is_long = cfg.get("pack_long", False)
        samples = []

        for sample in ds:
            text = extract_text(sample, key)
            if not text or len(text) < min_chars:
                continue
            samples.append({"text": text})
            if len(samples) >= args.max_per_dataset:
                break

        if is_long:
            # Aggressively pack for long-context training
            packed = []
            buf = ""
            for s in samples:
                buf += "\n\n" + s["text"]
                if len(buf) > args.pack_long_to:
                    packed.append({"text": buf.strip()})
                    buf = ""
            if buf:
                packed.append({"text": buf.strip()})
            print(f"  Packed {len(samples)} -> {len(packed)} long-context examples")
            samples = packed

        print(f"  Kept {len(samples)} examples")
        all_examples.extend(samples)

        # Save per-source
        ds_out = Dataset.from_list(samples)
        ds_out.save_to_disk(os.path.join(args.output, key))

    # Shuffle and save final
    import random
    random.seed(42)
    random.shuffle(all_examples)

    final = Dataset.from_list(all_examples)
    final.save_to_disk(os.path.join(args.output, "final_train"))

    # Validation split
    val_size = max(1000, int(len(final) * 0.05))
    val = final.select(range(val_size))
    train = final.select(range(val_size, len(final)))

    val.save_to_disk(os.path.join(args.output, "val"))
    train.save_to_disk(os.path.join(args.output, "train_final"))

    print(f"\n{'='*50}")
    print(f"Done: {len(all_examples)} total examples")
    print(f"Train: {len(train)}, Val: {len(val)}")
    print(f"Saved to {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
