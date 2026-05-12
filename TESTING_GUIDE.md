# Testing Guide for nanowhale

This guide provides step-by-step instructions to verify that the nanowhale codebase is working correctly after the bug fixes.

## Prerequisites

```bash
# Install required packages
pip install torch>=2.6.0 transformers>=5.0.0 datasets>=3.0.0 accelerate>=1.0.0 trl>=1.0.0 safetensors pyyaml huggingface_hub
```

## Step 1: Verify Installation and Model Creation

```bash
# Test that the model can be created and run a forward pass
python -c "
from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM
import torch

print('Testing model creation...')
config = DeepseekV4Config(
    vocab_size=129280,
    hidden_size=768,
    num_hidden_layers=2,
    num_attention_heads=4,
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
    max_position_embeddings=512
)
model = DeepseekV4ForCausalLM(config)
model.eval()

# Test forward pass
input_ids = torch.randint(0, 129280, (1, 32))
with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=input_ids)
print(f'✓ Model created successfully')
print(f'✓ Forward pass successful, loss: {outputs.loss.item():.4f}')
print(f'✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

## Step 2: Verify Tokenizer

```bash
# Test tokenizer loading and vocab size
python -c "
from transformers import PreTrainedTokenizerFast

print('Testing tokenizer...')
tokenizer = PreTrainedTokenizerFast.from_pretrained('tokenizer')
vocab_size = tokenizer.vocab_size
print(f'  Vocab size: {vocab_size}')

# Verify vocab size matches model config
assert vocab_size == 129280, f'Vocab size mismatch! Expected 129280, got {vocab_size}'
print('✓ Tokenizer loaded and verified (vocab_size=129280)')

# Test encoding/decoding
test_text = 'Hello, world! This is a test.'
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print(f'  Test encode/decode: \"{test_text}\" -> {len(encoded)} tokens -> \"{decoded}\"')
print('✓ Tokenizer encoding/decoding works')
"
```

## Step 3: Count Parameters for Various Configs

```bash
# Run the parameter counting script to verify different model sizes
python scripts/count_params.py
```

Expected output should show various configs from debug (~400K params) to ~800M params.

## Step 4: Test Data Preparation (Optional)

```bash
# Test data preparation with a tiny subset (requires internet)
python scripts/prepare_data.py
```

This will preview FineWeb-Edu and SmolTalk datasets.

## Step 5: Test Training Loop (Debug Mode)

```bash
# Run a tiny training loop to verify everything works
# This uses the debug config with 2 layers and 50 steps
python scripts/train_pretrain.py --config configs/debug.yaml --debug
```

Expected output:
- Model creation (~17M parameters)
- 50 training steps with decreasing loss
- Checkpoint saved to `checkpoints/pretrain_debug/final`

## Step 6: Test Evaluation

```bash
# First, ensure you have a trained model from Step 5, then:
python scripts/eval_smoke.py --model_path checkpoints/pretrain_debug/final
```

Expected output:
- Tokenizer loaded
- Model loaded
- Several text generations
- Perplexity computation

## Step 7: Test Chat Interface

```bash
# Interactive chat with the model (if you have a trained checkpoint)
python scripts/chat.py --model_path checkpoints/pretrain_debug/final --max_new_tokens 50
```

## Step 7: Verify Precision Settings

```bash
# Verify that fp32 is being used (not bf16/fp16)
python -c "
import torch
from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM

config = DeepseekV4Config(vocab_size=129280, hidden_size=768, num_hidden_layers=2)
model = DeepseekV4ForCausalLM(config)

# Check parameter dtype
param_dtype = next(model.parameters()).dtype
print(f'Model dtype: {param_dtype}')
assert param_dtype == torch.float32, f'Expected float32, got {param_dtype}'
print('✓ Model is using fp32 precision as required')
"
```

## Step 8: Test Safetensors Saving/Loading

```bash
# Test safetensors format
python -c "
import torch
from safetensors.torch import save_file, load_file
from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM
import tempfile
import os

print('Testing safetensors save/load...')
config = DeepseekV4Config(vocab_size=129280, hidden_size=64, num_hidden_layers=2)
model = DeepseekV4ForCausalLM(config)

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, 'model.safetensors')
    save_file(model.state_dict(), path)
    print(f'  Saved to {path}')
    
    # Load and verify
    loaded = load_file(path)
    print(f'  Loaded {len(loaded)} tensors')
    assert len(loaded) == len(model.state_dict()), 'Tensor count mismatch'
    print('✓ Safetensors save/load successful')
"
```

## Full Integration Test (Optional)

For a complete end-to-end test:

```bash
# 1. Prepare a small dataset
python scripts/prepare_1b_data.py --output data/test --tokens 1000000

# 2. Train for a few steps
python scripts/train_1b_pretrain.py --steps 10 --data data/test/final_train --output checkpoints/test

# 3. Evaluate
python scripts/eval_smoke.py --model_path checkpoints/test/final

# 4. Chat
echo -e "Hello\nWhat is 2+2?\nquit" | python scripts/chat.py --model_path checkpoints/test/final --max_new_tokens 20
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size or use gradient accumulation. The debug config uses minimal memory.

### Issue: NaN loss during training
**Solution**: This should not happen with fp32. If it does, check that all scripts are using the updated code. The `safe_forward` function will skip problematic batches.

### Issue: Vocab size mismatch errors
**Solution**: Ensure you're using `vocab_size=129280` in all configs. The tokenizer has 129,280 tokens.

### Issue: Model won't load from checkpoint
**Solution**: Try loading with `load_state_dict` manually instead of `from_pretrained`. The custom architecture sometimes has quirks.

## Performance Benchmarks (Expected)

On RTX 3080 Ti (12GB):
- Debug config (2 layers): ~50ms/step
- 1B config (16 layers): ~200-300ms/step with gradient checkpointing
- Memory usage: ~8-10GB for 1B config with batch_size=1

On H100 (80GB):
- Debug config: ~10ms/step with torch.compile
- 1B config: ~50-80ms/step with torch.compile

## Next Steps

After passing all tests:
1. Prepare full dataset with `scripts/prepare_1b_data.py`
2. Train with `scripts/train_1b_pretrain.py` (adjust steps/lr as needed)
3. Fine-tune with `scripts/train_sft.py`
4. Evaluate and upload to Hub with `scripts/upload_to_hub.py`

## Additional Resources

- Official repository: https://github.com/huggingface/nanowhale
- Bug fixes documentation: `BUG_FIXES.md`
- Scaling plan: `1B_MOE_QAT_SCALING_PLAN.md`
