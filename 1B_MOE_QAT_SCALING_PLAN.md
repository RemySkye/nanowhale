# nanowhale-1B-MoE: Full Scaling Plan to Beat Qwen3.5-0.8B

**Project Goal**: Scale the excellent DeepSeek-V4 based nanowhale architecture from ~110M to a ~1B parameter Mixture-of-Experts model with:
- **~400M active parameters** per token (2-2.5x inference speedup over dense 0.8-1B models)
- **64k–128k native context** (heavily optimized long-context training)
- **Near-lossless INT4 deployment** via high-quality Quantization-Aware Training (QAT) with >99% retained quality
- **Superior performance** to Qwen3.5-0.8B on reasoning, coding, instruction following, agentic tasks, and long-term memory benchmarks despite smaller active size.

**Hardware**: RTX 3080 Ti (12 GB VRAM, ~11 GB usable), 16-core CPU, 32 GB RAM. Training must be extremely memory- and speed-optimized. CPU offloading + async data loading allowed.

**Date of Plan**: 2026-05-09
**Base Architecture**: DeepSeek-V4 (MLA + MoE + Hyper-Connections + MTP)

---

## 1. Target Architecture for ~1B Total / ~400M Active

### Recommended Config (1B-MoE-64k+)

```python
model:
  # General
  vocab_size: 49152                    # Reduced from 129k → massive embedding savings (~75M params saved)
  hidden_size: 1152                    # Balanced density
  num_hidden_layers: 22
  max_position_embeddings: 131072      # 128k target (train up to 64k-128k with YaRN/NTK + CSA)
  rope_theta: 10000.0
  rope_scaling:                        # YaRN scaling for long context extrapolation
    type: "yarn"
    factor: 32.0
    original_max_position_embeddings: 4096
    beta_fast: 32
    beta_slow: 1.0

  # MLA (Multi-head Latent Attention) - memory efficient
  num_attention_heads: 16
  num_key_value_heads: 1               # MQA for speed
  head_dim: 128
  qk_rope_head_dim: 32                 # 32 RoPE + 96 NoPE
  q_lora_rank: 640
  o_groups: 4
  o_lora_rank: 320
  sliding_window: 4096                 # CSA base window
  compress_ratios: [0.5, 0.5, ...]     # Enable Compressed Sparse Attention layers for long context

  # MoE (target ~400M active)
  moe_intermediate_size: 3072
  n_routed_experts: 16
  n_shared_experts: 2
  num_experts_per_tok: 2               # Top-2 (excellent efficiency)
  scoring_func: "sqrtsoftplus"
  routed_scaling_factor: 1.0
  num_hash_layers: 4                   # Hash routing for first layers
  swiglu_limit: 10.0

  # Hyper-Connections (tuned for stability)
  hc_mult: 2                           # Reduced from 4 → huge stability + memory win
  hc_sinkhorn_iters: 3

  # Training stability
  rms_norm_eps: 1e-6
  initializer_range: 0.02
```

**Param Breakdown Estimate**:
- Total: ~1.02–1.08B
- Active per token: ~380–420M (MoE sparsity)
- Embeddings: ~50M (with reduced vocab)
- Experts (dense equiv):-dominant part
- Attention + HC + norms: controlled

**Why this beats dense Qwen3.5-0.8B**:
- MoE sparsity gives ~2× effective FLOPs per active param.
- MLA is far more efficient than standard GQA/MQA at long context.
- DeepSeek-style routing + HC gives better specialization and long-term memory than dense models.

---

## 2. Complete Optimization Stack (RTX 3080 Ti + 16c CPU Optimized)

### Memory (Fit 1B MoE on 11 GB)
1. **DeepSpeed ZeRO-3** (or FSDP2 `FULL_SHARD + cpu_offload`) — shards params, grads, optimizer states. Critical.
2. **bitsandbytes 8-bit AdamW / Lion** — ~50–65% optimizer state savings.
3. **Gradient Checkpointing** (selective, non-reentrant) + activation offloading to CPU.
4. **torchao MXFP8 MoE Training** (prototype but highly effective on Ampere):
   - Replaces `torch._grouped_mm` with `_to_mxfp8_then_scaled_grouped_mm`
   - 1.4–1.8× expert computation speedup and memory reduction.

Key optimizations for this hardware:

- **torch.compile** with "reduce-overhead" mode
- Use PyTorch 2.1+ SDPA (already in code)
- CPU offloading for optimizer states (compatible with ZeRO-3/DeepSpeed or Accelerate)
- Dataloader with num_workers=8-12 (we have 16 cores)
- Mixed precision: prefer bf16 where stable, fall back to fp16 with care for HC
- Reduce hc_mult=2 as recommended
- Smaller vocab to reduce embedding table footprint
- Curriculum context: start 2k → 8k → 32k → 64k with YaRN
- For 128k: rely on CSA + YaRN extrapolation

### QAT Pipeline for Near-Lossless INT4

Target: INT4 W4A8 (weight 4-bit, activation 8-bit) or full W4A4 with minimal degradation.

Steps:
1. Pretrain in bf16/MXFP8
2. QAT fine-tune (last 10-20% of training) using torchao int8/fp8 simulate or custom STE
3. Calibration: 100-200 samples of diverse data
4. Final conversion: GPTQ or AWQ on the QAT model for final INT4
5. Validation: Run eval before/after to ensure <1-2% perplexity increase and no generation issues

Quality safeguard: Use "smooth" quantization + per-token calibration + test on reasoning/coding benchmarks.

### Data Strategy (English + Reasoning + Coding + Long Context)

High-quality mix targeting <99% English, heavy on reasoning/coding + long context:

**Core mix for 1B MoE pretraining (target ~120B+ tokens for good 64k+ model)**

| Dataset                    | Purpose                        | Weight | Quality Focus                     |
|----------------------------|--------------------------------|--------|-----------------------------------|
| FineWeb-Edu                | General knowledge + long docs  | 45%    | High educational score (>3.0)     |
| Cosmopedia-2.0             | Synthetic textbooks       | 15%    | Long coherent articles            |
| The-Stack-v2 (Python/JS/TS)| Strong coding                  | 15%    | Clean, deduplicated code          |
| Magpie-Phi-3-Ultra         | Instruction + agentic          | 10%    | High-quality synthetic conversations |
| OpenMath-Instruct          | Math reasoning                 | 8%     | Step-by-step proofs               |
| Long synthetic + curated   | 32k-128k context training      | 7%     | Documents + multi-turn long chats |

All data heavily filtered for English quality, length, and educational value. Long-context examples are aggressively packed to 32k-64k blocks during preparation.

Use streaming where possible to keep memory low on RTX 3080 Ti + 32 GB RAM.

Now I'll create the markdown file and then perform data preparation.