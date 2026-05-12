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
- **Vocab**: 129,280 tokens (DeepSeek-V4 tokenizer)

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
│   ├── train_pretrain.py             # Original 100M (SFTTrainer)
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

## Precision & Stability

**IMPORTANT**: This model **must use fp32 precision** for training and inference due to numerical instability in the Hyper-Connections architecture at small scale. Mixed precision (bf16/fp16) causes NaN values.

### Why fp32?
- **Hyper-Connections numerical range**: The HC architecture produces values that overflow bf16's limited dynamic range
- **MoE routing stability**: Expert routing scores require full fp32 precision to avoid degenerate routing
- **Small scale effects**: At 1B scale, the model lacks the numerical stability of larger models

All training scripts and configs have been updated to use fp32 by default. See `BUG_FIXES.md` for details.

## Optimizations

**Colab (H100/A100)**:
- fp32 + torch.compile (big speedup on H100)
- No DeepSpeed needed (model fits easily in 80GB)
- Curriculum context: 4k to 256k
- AdamW fused + efficient attention (SDPA)

**Local (RTX 3080 Ti)**:
- fp32 full precision (no mixed precision)
- Gradient checkpointing
- Small batches with gradient accumulation

## Known Issues

- **fp32 required**: The model produces NaN in bf16/fp16. All scripts enforce fp32.
- **from_pretrained quirk**: Custom architecture may re-initialize weights. Use manual `load_state_dict`.
- **Token IDs must match vocab**: The config `vocab_size` MUST match the tokenizer (129,280). Mismatch causes CUDA index errors.
- **Large vocab overhead**: The 129K vocab embedding table consumes ~37% of parameters in 100M model.

## Changes Made (This Session)

1. **Fixed vocab_size mismatch** - Aligned all configs to 129,280 tokens
2. **Enforced fp32 precision** - All training/inference now uses fp32
3. **Added safetensors support** - Models save in secure, fast format
4. **Improved error handling** - NaN detection and recovery in training
5. **Updated documentation** - Clear explanation of fp32 requirement

## Testing

Before using, run the verification steps in `TESTING_GUIDE.md`:

```bash
# Quick smoke test
python scripts/count_params.py

# Test model creation and forward pass
python -c "
from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM
import torch
config = DeepseekV4Config(vocab_size=129280, hidden_size=768, num_hidden_layers=2)
model = DeepseekV4ForCausalLM(config)
input_ids = torch.randint(0, 129280, (1, 32))
outputs = model(input_ids=input_ids, labels=input_ids)
print(f'Loss: {outputs.loss.item():.4f}')
"
```

## License

MIT

## References

- Official nanowhale repository: https://github.com/huggingface/nanowhale
- DeepSeek-V4 paper: https://arxiv.org/abs/2412.10203
- Bug fixes documentation: `BUG_FIXES.md`
- Testing guide: `TESTING_GUIDE.md`
