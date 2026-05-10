# nanowhale — DeepSeek-V4 MoE at 1B Scale

> **WORK IN PROGRESS — NOT READY FOR USE**
> Training stability still being refined. Do not use yet.

A DeepSeek-V4 architecture implementation scaled to ~1.2B parameters (~400M active via MoE sparsity).

- **Colab H100 / A100** — one-click notebook included
- **RTX 3080 Ti** — verified working with optimized settings

**Goal**: Match or beat dense models like Qwen3.5-0.8B with 2x faster inference using expert sparsity.

## Architecture

Full DeepSeek-V4 feature set at 1B scale:

- **Multi-Head Latent Attention (MLA)** — 16 heads, 1 KV head (MQA), head_dim=128 (32 RoPE + 96 NoPE), q_lora_rank=384
- **Mixture-of-Experts (MoE)** — 12 routed + 2 shared experts, top-2 routing, SwiGLU FFN (dim 2048)
- **Hyper-Connections** — hc_mult=2, Sinkhorn routing
- **Vocab**: 128,000 tokens (DeepSeek-V4 tokenizer)

| Parameter | Value |
|-----------|-------|
| Total params | ~1.2B |
| Active per token | ~400M (MoE top-2) |
| Hidden size | 768 |
| Layers | 16 |
| Context | 4,096 (extendable to 256k with curriculum) |

## Quick Start

### Google Colab (Recommended)

Open the notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RemySkye/nanowhale/blob/main/colab/nanowhale_1b_moe_colab.ipynb)

The notebook is **completely self-contained** — no external scripts needed. It includes the full model code, data preparation, training loop, and saving to Drive.

### Local Training (RTX 3080 Ti)

```bash
# Data prep (3 public datasets)
python scripts/prepare_1b_data.py --output data/processed/1b_moe_data_ready

# Training
python scripts/train_1b_pretrain.py --steps 500 --lr 1.5e-4 --max_len 512
```

## Dataset

**Colab (reasoning-heavy, 7 datasets)**: `scripts/prepare_colab_data.py`

| Dataset | Weight | Purpose |
|---------|--------|---------|
| HuggingFaceFW/fineweb-2 | 30% | General English knowledge |
| allenai/c4 | 20% | Clean web text |
| m-a-p/FineFineWeb | 12% | Curated high-quality web |
| deepmind/code_contests | 10% | Competitive coding reasoning |
| openai/gsm8k | 10% | Math reasoning |
| google-research-datasets/mbpp | 8% | Python problem solving |
| fineweb-2 long-pack | 10% | 64k+ context blocks |

**Local (fast, 3 datasets)**: `scripts/prepare_1b_data.py`

## Repo Structure

```
├── modeling_deepseek_v4.py          # DeepSeek-V4 model (MLA + MoE + HC)
├── configuration_deepseek_v4.py     # Model config class
├── 1B_MOE_QAT_SCALING_PLAN.md       # Full scaling plan & QAT strategy
├── colab/
│   └── nanowhale_1b_moe_colab.ipynb  # All-in-one self-contained notebook
├── configs/
│   ├── main_100m.yaml               # Original 110M config
│   ├── debug_1b_moe.yaml            # 1B debug config (3080 Ti)
│   └── 1b_moe_colab_256k.yaml       # Colab H100/A100 256k config
├── scripts/
│   ├── train_colab.py                # Colab pretraining (pure PyTorch)
│   ├── train_1b_pretrain.py          # Local pretraining (RTX 3080 Ti)
│   ├── train_pretrain.py             # Original 110M (SFTTrainer)
│   ├── train_sft.py                  # SFT fine-tuning
│   ├── prepare_colab_data.py         # Colab data prep (reasoning mix)
│   ├── prepare_1b_data.py            # Local data prep
│   ├── prepare_data.py               # Original data utilities
│   ├── chat.py / eval_smoke.py       # Inference & evaluation
│   ├── count_params.py / inspect_*.py # Analysis tools
│   └── upload_to_hub.py              # HF Hub upload
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Optimizations

**Colab (H100/A100)**:
- BF16 + torch.compile (big speedup on H100)
- No DeepSpeed needed (model fits easily in 80GB)
- Curriculum context: 4k to 256k
- AdamW fused + efficient attention (SDPA)

**Local (RTX 3080 Ti)**:
- BF16 mixed precision
- Gradient checkpointing
- Small batches with gradient accumulation

## Known Issues

- **bf16 NaN**: The model can produce NaN in bf16 at small scale. Training scripts use safety handling.
- **from_pretrained quirk**: Custom architecture may re-initialize weights. Use manual `load_state_dict`.
- **Token IDs must match vocab**: The config `vocab_size` MUST match the tokenizer (128,000). Mismatch causes CUDA index errors.

## License

MIT
