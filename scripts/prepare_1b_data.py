#!/usr/bin/env python3
"""
High-quality data preparation for nanowhale-1B-MoE (target: beat Qwen3.5-0.8B)
Focus: English general knowledge, coding, instruction following, agentic use,
strong reasoning, long-context capabilities (up to 128k).

Uses multiple sources + strict quality filtering.
Supports streaming, packing, and curriculum context length scheduling.
"""

import os
import argparse
import json
from typing import Iterator, Dict, Any

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import PreTrainedTokenizerFast

# 100% Public, high-quality mix (no gated datasets) for 1B MoE pretraining
# Focus: English general knowledge, reasoning, coding, instructions, long-context
DATA_MIX = {
    "fineweb_edu": {
        "name": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "streaming": True,
        "filters": {"score_min": 3.0, "min_token_count": 256},
        "weight": 0.55,
    },
    "cosmopedia": {
        "name": "HuggingFaceTB/cosmopedia",
        "name_config": "web_samples_v2",
        "split": "train",
        "streaming": True,
        "filters": {"min_len": 350},
        "weight": 0.25,
    },
    "long_context": {
        "name": "HuggingFaceFW/fineweb-edu",  # will be packed to 32k-64k+ during prep
        "split": "train",
        "streaming": True,
        "pack_long": True,
        "weight": 0.20,
    },
}

DEFAULT_TOKENIZER = "tokenizer"  # DeepSeek-V4 tokenizer already in repo


def filter_sample(sample: Dict[str, Any], dataset_key: str) -> bool:
    """Strict quality filter per dataset."""
    text = sample.get("text") or sample.get("content") or ""
    if len(text) < 128:
        return False

    if dataset_key == "fineweb_edu":
        score = sample.get("score", 0.0)
        tokens = sample.get("token_count", len(text) // 4)
        return score >= 3.0 and tokens >= 256

    if dataset_key == "cosmopedia":
        return len(text) >= 350

    return True


def make_long_context_samples(ds: Iterator[Dict], target_len: int = 32768) -> Iterator[Dict]:
    """Simple concatenation of multiple samples into long context examples for 64k+ training."""
    buffer = ""
    for sample in ds:
        text = sample.get("text", "") or sample.get("content", "")
        if not text:
            continue
        buffer += "\n\n" + text
        if len(buffer) > target_len:
            yield {"text": buffer.strip()}
            buffer = ""
    if buffer:
        yield {"text": buffer.strip()}


def load_and_filter_dataset(key: str, config: dict, tokenizer: PreTrainedTokenizerFast,
                            max_samples: int = None) -> list:
    """Load, filter, and optionally pack a dataset."""
    print(f"[Data] Loading {key} ...")
    streaming = config.get("streaming", True)
    ds = load_dataset(
        config["name"],
        config.get("name_config"),
        split=config.get("split", "train"),
        streaming=streaming,
    )

    out = []
    count = 0
    for s in ds:
        if filter_sample(s, key):
            out.append(s)
            count += 1
        if max_samples and count >= max_samples:
            break

    # For long-context data we pack aggressively
    if key == "long_context_docs" or config.get("pack_long", False):
        return list(make_long_context_samples(out, target_len=49152))

    return out


def create_1b_training_dataset(
    output_dir: str = "data/processed/1b_moe_data_64k",
    tokenizer_path: str = DEFAULT_TOKENIZER,
    total_tokens_target: int = 120_000_000,   # ~120B tokens for 1B MoE (dense equiv ~50B)
    seed: int = 42,
):
    """Main entry point: produce a high-quality packed dataset for 1B MoE pretraining."""
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Tokenizer vocab={tokenizer.vocab_size}")

    train_examples = []
    total_tokens = 0
    rng = __import__("random")
    rng.seed(seed)

    for dataset_key, cfg in DATA_MIX.items():
        rel_weight = cfg["weight"]
        target_for_ds = int(total_tokens_target * rel_weight)
        print(f"\n=== Processing {dataset_key} (weight={rel_weight}, target_tokens={target_for_ds:,}) ===")

        samples = load_and_filter_dataset(dataset_key, cfg, tokenizer, max_samples=100_000)

        # Simple packing to 8k–64k maximally for long context curriculum
        packed = []
        current = ""
        for sample in samples:
            text = sample.get("text") or sample.get("content") or str(sample)
            if len(current) + len(text) > 49152:   # Max pack length per example
                if current:
                    packed.append({"text": current.strip()})
                    current = text
                else:
                    current = text
            else:
                current = (current + "\n\n" + text).strip() if current else text

            if len(packed) > 5000:   # reasonable cap per dataset
                break

        # Save sample to disk
        ds_out = Dataset.from_list(packed)
        ds_out.save_to_disk(f"{output_dir}/{dataset_key}")
        print(f"Saved {len(packed)} examples for {dataset_key}")

        train_examples.extend(packed)

    # Final concatenated + shuffled dataset
    final_ds = Dataset.from_list(train_examples)
    final_ds = final_ds.shuffle(seed=seed)

    final_ds.save_to_disk(f"{output_dir}/final_train")
    print(f"\n✅ Final training dataset saved: {len(final_ds)} examples")
    print(f"Dataset location: {output_dir}/final_train")

    # Simple validation split (5%)
    val_size = max(2000, int(len(final_ds) * 0.05))
    val_ds = final_ds.select(range(val_size))
    train_final = final_ds.select(range(val_size, len(final_ds)))

    val_ds.save_to_disk(f"{output_dir}/val")
    train_final.save_to_disk(f"{output_dir}/train_final")

    print("Done. Ready for pretraining with 64k+ context.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/1b_moe_64k")
    parser.add_argument("--tokens", type=int, default=120_000_000)  # 120M tokens sample
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    args = parser.parse_args()

    create_1b_training_dataset(
        output_dir=args.output,
        tokenizer_path=args.tokenizer,
        total_tokens_target=args.tokens,
    )
