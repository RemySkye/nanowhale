# nanowhale 🐳 — DeepSeek-V4 MoE at 1B Scale

A DeepSeek-V4 architecture implementation scaled to **~1B parameters** (~400M active via MoE sparsity), trainable on a single **RTX 3080 Ti (12GB)**.

> **Goal**: Match or beat dense models like Qwen3.5-0.8B with 2× faster inference using expert sparsity.

## Architecture

Full DeepSeek-V4 feature set at 1B scale:

- **Multi-Head Latent Attention (MLA)** — 16 heads, 1 KV head (MQA), head_dim=128 (32 RoPE + 96 NoPE), q_lora_rank=384
- **Mixture-of-Experts (MoE)** — 12 routed + 2 shared experts, top-2 routing, SwiGLU FFN (dim 2048)
- **Hyper-Connections** — hc_mult=2, Sinkhorn routing
- **Vocab**: 128,000 tokens (DeepSeek-V4 tokenizer)

| Parameter | Value |
|-----------|-------|
| Total params | ~1.09B |
| Active per token | ~400M (MoE top-2) |
| Hidden size | 768 |
| Layers | 16 |
| Context | 4,096 (extendable to 64k+ with YaRN + CSA) |

## Quick Start

### Install

```bash
pip install torch transformers datasets safetensors pyyaml
```

### Data Preparation

```bash
python scripts/prepare_1b_data.py --output data/processed/1b_moe_data_ready
```

Uses public datasets only:
- HuggingFaceFW/fineweb-edu (55%) — high-quality educational English
- HuggingFaceTB/cosmopedia (25%) — synthetic textbooks & articles
- fineweb-edu long-pack (20%) — 32k–64k context blocks

### Pretraining

```bash
python scripts/train_1b_pretrain.py --steps 500 --lr 1.5e-4 --max_len 512
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--steps` | 500 | Training steps |
| `--lr` | 1.5e-4 | Learning rate |
| `--max_len` | 512 | Max sequence length |
| `--data` | `data/processed/1b_moe_data_ready/final_train` | Dataset path |
| `--output` | `checkpoints/1b_moe_pretrain` | Output directory |

## Repo Structure

```
├── modeling_deepseek_v4.py          # DeepSeek-V4 model (MLA + MoE + HC)
├── configuration_deepseek_v4.py     # Model config class
├── 1B_MOE_QAT_SCALING_PLAN.md       # Full scaling plan & QAT strategy
├── configs/
│   ├── main_100m.yaml               # Original 110M config
│   ├── debug_1b_moe.yaml            # 1B debug config
│   ├── 1b_moe_64k_qat.yaml          # Target 64k+ QAT config
│   └── deepspeed_zero3_3080ti.json  # ZeRO-3 optimized for 3080 Ti
├── scripts/
│   ├── train_1b_pretrain.py          # 1B pretraining
│   ├── train_pretrain.py             # Original 110M pretraining (SFTTrainer)
│   ├── train_sft.py                  # SFT fine-tuning
│   ├── prepare_1b_data.py            # 1B data preparation
│   ├── prepare_data.py               # Original data utilities
│   ├── chat.py                       # Interactive chat
│   ├── eval_smoke.py                 # Perplexity evaluation
│   ├── count_params.py               # Parameter counting
│   └── upload_to_hub.py              # Hub upload
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Memory Optimizations (for RTX 3080 Ti)

- BF16 mixed precision with `torch.amp`
- Gradient checkpointing
- DeepSpeed ZeRO-3 ready (`configs/deepspeed_zero3_3080ti.json`)
- torchao MXFP8 MoE expert quantization (planned, see scaling plan)
- Selective QAT for INT4 deployment (see `1B_MOE_QAT_SCALING_PLAN.md`)

## Known Issues

- **bf16 NaN**: The model can produce NaN in bf16 at small scale. Training scripts use safety handling.
- **from_pretrained quirk**: Custom architecture may re-initialize weights. Use manual `load_state_dict`.
- **Token IDs must match vocab**: The config `vocab_size` MUST match the tokenizer (128,000). Mismatch causes CUDA index errors.

## License

MIT
