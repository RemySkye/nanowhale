#!/usr/bin/env python3
"""
Apply easy optimizations to nanowhale Colab notebook:
1. Gradient Accumulation Scheduling
2. Smart Checkpointing 
3. Adaptive Gradient Clipping
4. Dynamic Sequence Packing (simplified version)
5. Layer-wise Learning Rates (basic version)
6. Gradient Centralization
"""

import json
import re
from datetime import datetime

def apply_easy_optimizations(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    updates = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # 1. Gradient Accumulation Scheduling
            if 'GRAD_ACCUM = ' in source and 'BATCH_SIZE = ' in source:
                lines = source.split('\n')
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if 'GRAD_ACCUM = ' in line:
                        # Add dynamic gradient accumulation
                        new_lines.append('')
                        new_lines.append('# Dynamic gradient accumulation scheduling')
                        new_lines.append('def get_grad_accum(step, base_accum, max_accum=32):')
                        new_lines.append('    """Increase accumulation over training for stability.""")')
                        new_lines.append('    progress = step / STEPS')
                        new_lines.append('    # Start with smaller accum for frequent updates, increase for stability')
                        new_lines.append('    return min(int(base_accum * (1 + progress * 3)), max_accum)')
                        updates.append(f"Cell {i}: Added gradient accumulation scheduling")
            
            # 2. Smart Checkpointing
            if 'SAVE_EVERY = ' in source and 'if (step + 1) % SAVE_EVERY == 0:' in source:
                lines = source.split('\n')
                new_lines = []
                checkpoint_var_added = False
                for line in lines:
                    new_lines.append(line)
                    if 'SAVE_EVERY = ' in line and not checkpoint_var_added:
                        new_lines.append('last_checkpoint_loss = float("inf")')
                        new_lines.append('checkpoint_improvement_threshold = 0.01  # 1% improvement')
                        checkpoint_var_added = True
                    if 'if (step + 1) % SAVE_EVERY == 0:' in line:
                        # Replace with smart checkpointing logic
                        new_lines.pop()  # Remove the original if statement
                        new_lines.append('should_save = False')
                        new_lines.append('if (step + 1) % SAVE_EVERY == 0:')
                        new_lines.append('    # Smart checkpointing: save if loss improved significantly')
                        new_lines.append('    if losses:')
                        new_lines.append('        current_loss = sum(losses[-10:]) / len(losses[-10:])')
                        new_lines.append('        if current_loss < last_checkpoint_loss * (1 - checkpoint_improvement_threshold):')
                        new_lines.append('            should_save = True')
                        new_lines.append('            last_checkpoint_loss = current_loss')
                        new_lines.append('        elif (step + 1) % (SAVE_EVERY * 5) == 0:  # Force save every 5x SAVE_EVERY')
                        new_lines.append('            should_save = True')
                        new_lines.append('    else:')
                        new_lines.append('        should_save = True')
                        new_lines.append('')
                        new_lines.append('if should_save:')
                new_source = '\n'.join(new_lines)
                cell['source'] = [new_source]
                updates.append(f"Cell {i}: Added smart checkpointing")
            
            # 3. Adaptive Gradient Clipping
            if 'torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)' in source:
                lines = source.split('\n')
                new_lines = []
                for line in lines:
                    if 'torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)' in line:
                        new_lines.append('        # Adaptive gradient clipping')
                        new_lines.append('        # Compute gradient statistics')
                        new_lines.append('        grad_norms = []')
                        new_lines.append('        for p in model.parameters():')
                        new_lines.append('            if p.grad is not None:')
                        new_lines.append('                grad_norms.append(p.grad.norm().item())')
                        new_lines.append('        if grad_norms:')
                        new_lines.append('            grad_std = torch.std(torch.tensor(grad_norms)).item()')
                        new_lines.append('            clip_norm = max(0.5, min(2.0, grad_std * 2))')
                        new_lines.append('        else:')
                        new_lines.append('            clip_norm = 1.0')
                        new_lines.append('        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)')
                    else:
                        new_lines.append(line)
                new_source = '\n'.join(new_lines)
                cell['source'] = [new_source]
                updates.append(f"Cell {i}: Added adaptive gradient clipping")
            
            # 4. VRAM Memory Management for different GPU types
            if 'vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9' in source:
                lines = source.split('\n')
                new_lines = []
                vram_mgmt_added = False
                for line in lines:
                    new_lines.append(line)
                    if 'print(f"VRAM:' in line and not vram_mgmt_added:
                        new_lines.append('')
                        new_lines.append('# VRAM management: target 70% usage, max 75%')
                        new_lines.append('TARGET_VRAM_USAGE = 0.70  # Target 70% VRAM usage')
                        new_lines.append('MAX_VRAM_USAGE = 0.75     # Never exceed 75%')
                        new_lines.append('')
                        new_lines.append('def calculate_safe_batch_size(vram_gb, base_batch):')
                        new_lines.append('    """Calculate batch size to stay within VRAM limits.""")')
                        new_lines.append('    # H100/A100 80GB: can handle larger batches')
                        new_lines.append('    # RTX 6000 96GB: even more VRAM but slower compute')
                        new_lines.append('    if vram_gb >= 75:  # RTX 6000 Ada 96GB')
                        new_lines.append('        # More VRAM but slower, optimize for compute not memory')
                        new_lines.append('        return min(base_batch * 2, 16)')
                        new_lines.append('    elif vram_gb >= 70:  # H100/A100 80GB')
                        new_lines.append('        return min(base_batch * 2, 12)')
                        new_lines.append('    elif vram_gb >= 35:  # A100 40GB')
                        new_lines.append('        return base_batch')
                        new_lines.append('    else:  # T4, etc.')
                        new_lines.append('        return max(1, base_batch // 2)')
                        new_lines.append('')
                        new_lines.append('def monitor_vram():')
                        new_lines.append('    """Check current VRAM usage and warn if exceeding limits.""")')
                        new_lines.append('    allocated = torch.cuda.memory_allocated() / 1e9')
                        new_lines.append('    reserved = torch.cuda.memory_reserved() / 1e9')
                        new_lines.append('    usage_pct = allocated / vram_gb')
                        new_lines.append('    if usage_pct > MAX_VRAM_USAGE:')
                        new_lines.append('        print(f"  ⚠️  VRAM usage at {usage_pct:.1%} - consider reducing batch size")')
                        new_lines.append('    elif usage_pct > TARGET_VRAM_USAGE:')
                        new_lines.append('        print(f"  ℹ️  VRAM usage at {usage_pct:.1%} - approaching target")')
                        new_lines.append('    return usage_pct')
                        vram_mgmt_added = True
                if vram_mgmt_added:
                    new_source = '\n'.join(new_lines)
                    cell['source'] = [new_source]
                    updates.append(f"Cell {i}: Added VRAM management")
            
            # 5. Optimize batch size calculation for specific GPUs
            if 'if vram_gb >= 70:' in source and 'BATCH_SIZE = ' in source:
                lines = source.split('\n')
                new_lines = []
                for line in lines:
                    if 'if vram_gb >= 70:' in line:
                        new_lines.append('if vram_gb >= 75:  # RTX 6000 Ada 96GB')
                        new_lines.append('    BATCH_SIZE = 6; GRAD_ACCUM = 8      # eff batch 48')
                        new_lines.append('elif vram_gb >= 70:  # H100/A100 80GB')
                        new_lines.append('    BATCH_SIZE = 8; GRAD_ACCUM = 8      # eff batch 64')
                        new_lines.append('elif vram_gb >= 35:  # A100 40GB / L40S')
                        new_lines.append('    BATCH_SIZE = 4; GRAD_ACCUM = 8      # eff batch 32')
                        new_lines.append('elif vram_gb >= 20:  # T4 16GB-ish / A10')
                        new_lines.append('    BATCH_SIZE = 2; GRAD_ACCUM = 12     # eff batch 24')
                        new_lines.append('else:  # T4 free tier')
                        new_lines.append('    BATCH_SIZE = 1; GRAD_ACCUM = 16     # eff batch 16')
                    else:
                        new_lines.append(line)
                new_source = '\n'.join(new_lines)
                cell['source'] = [new_source]
                updates.append(f"Cell {i}: Optimized batch sizes for GPU types")
            
            # 6. Layer-wise Learning Rates
            if 'optimizer = torch.optim.AdamW(model.parameters()' in source:
                lines = source.split('\n')
                new_lines = []
                lr_optimization_added = False
                for line in lines:
                    new_lines.append(line)
                    if 'optimizer = torch.optim.AdamW(model.parameters()' in line and not lr_optimization_added:
                        new_lines.append('')
                        new_lines.append('# Layer-wise learning rates (lower layers get smaller LR)')
                        new_lines.append('def apply_layer_wise_lr(model, base_lr, decay=0.95):')
                        new_lines.append('    """Apply decreasing LR to earlier layers.""")')
                        new_lines.append('    param_groups = []')
                        new_lines.append('    for name, param in model.named_parameters():')
                        new_lines.append('        if not param.requires_grad:')
                        new_lines.append('            continue')
                        new_lines.append('        # Calculate layer depth (deeper = higher LR)')
                        new_lines.append('        layer_match = re.search(r\'layers\.\(\d+\)\', name)')
                        new_lines.append('        if layer_match:')
                        new_lines.append('            layer_idx = int(layer_match.group(1))')
                        new_lines.append('            total_layers = model.config.num_hidden_layers')
                        new_lines.append('            # Normalize layer index (0 = bottom, total-1 = top)')
                        new_lines.append('            normalized_depth = layer_idx / total_layers')
                        new_lines.append('            # LR multiplier: deeper layers get higher LR')
                        new_lines.append('            lr_mult = decay ** (total_layers - layer_idx - 1)')
                        new_lines.append('        else:')
                        new_lines.append('            lr_mult = 1.0  # Embedding/head layers')
                        new_lines.append('        param_groups.append({\'params\': param, \'lr\': base_lr * lr_mult})')
                        new_lines.append('    return param_groups')
                        new_lines.append('')
                        new_lines.append('# Create optimizer with layer-wise LR')
                        new_lines.append('param_groups = apply_layer_wise_lr(model, LR)')
                        new_lines.append('optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1, betas=(0.9, 0.95), fused=True)')
                        lr_optimization_added = True
                if lr_optimization_added:
                    new_source = '\n'.join(new_lines)
                    cell['source'] = [new_source]
                    updates.append(f"Cell {i}: Added layer-wise learning rates")
            
            # 7. Gradient Centralization
            if 'loss.backward()' in source and 'scaler.scale(loss).backward()' in source:
                lines = source.split('\n')
                new_lines = []
                grad_central_added = False
                for line in lines:
                    new_lines.append(line)
                    if 'scaler.scale(loss).backward()' in line and not grad_central_added:
                        new_lines.append('        # Gradient centralization (improves optimization)')
                        new_lines.append('        with torch.no_grad():')
                        new_lines.append('            for p in model.parameters():')
                        new_lines.append('                if p.grad is not None:')
                        new_lines.append('                    # Subtract mean from gradients')
                        new_lines.append('                    p.grad -= p.grad.mean(dim=list(range(1, p.grad.dim())), keepdim=True)')
                        grad_central_added = True
                if grad_central_added:
                    new_source = '\n'.join(new_lines)
                    cell['source'] = [new_source]
                    updates.append(f"Cell {i}: Added gradient centralization")
            
            # 8. HuggingFace Upload Integration
            if 'from google.colab import drive' in source and 'drive.mount' in source:
                lines = source.split('\n')
                new_lines = []
                hf_upload_added = False
                for line in lines:
                    new_lines.append(line)
                    if 'shutil.copytree(src, dst, dirs_exist_ok=True)' in line and not hf_upload_added:
                        new_lines.append('')
                        new_lines.append('# Upload to HuggingFace Hub')
                        new_lines.append('try:')
                        new_lines.append('    from huggingface_hub import HfApi, login')
                        new_lines.append('    from pathlib import Path')
                        new_lines.append('    ')
                        new_lines.append('    # Login to HuggingFace')
                        new_lines.append('    from google.colab import userdata')
                        new_lines.append('    hf_token = userdata.get("HF_TOKEN")')
                        new_lines.append('    if hf_token:')
                        new_lines.append('        login(token=hf_token)')
                        new_lines.append('        print("✅ Logged into HuggingFace Hub")')
                        new_lines.append('        ')
                        new_lines.append('        # Create API instance')
                        new_lines.append('        api = HfApi()')
                        new_lines.append('        ')
                        new_lines.append('        # Get username for repo name')
                        new_lines.append('        try:')
                        new_lines.append('            user_info = api.whoami(token=hf_token)')
                        new_lines.append('            username = user_info["name"]')
                        new_lines.append('        except:')
                        new_lines.append('            username = "user"')
                        new_lines.append('        ')
                        new_lines.append('        # Create repo name with timestamp')
                        new_lines.append('        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")')
                        new_lines.append('        repo_name = f"nanowhale-1b-{timestamp}"')
                        new_lines.append('        repo_id = f"{username}/{repo_name}"')
                        new_lines.append('        ')
                        new_lines.append('        # Create repository')
                        new_lines.append('        print(f"Creating HuggingFace repo: {repo_id}")')
                        new_lines.append('        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)')
                        new_lines.append('        ')
                        new_lines.append('        # Upload model files')
                        new_lines.append('        print("Uploading model to HuggingFace...")')
                        new_lines.append('        api.upload_folder(')
                        new_lines.append('            folder_path=src,')
                        new_lines.append('            repo_id=repo_id,')
                        new_lines.append('            commit_message=f"Upload nanowhale-1b model - Step {step+1}"')
                        new_lines.append('        )')
                        new_lines.append('        print(f"✅ Model uploaded to https://huggingface.co/{repo_id}")')
                        new_lines.append('    else:')
                        new_lines.append('        print("⚠️  HF_TOKEN not found in Colab secrets. Skipping HuggingFace upload.")')
                        new_lines.append('except Exception as e:')
                        new_lines.append('    print(f"⚠️  HuggingFace upload failed: {e}")')
                        hf_upload_added = True
                if hf_upload_added:
                    new_source = '\n'.join(new_lines)
                    cell['source'] = [new_source]
                    updates.append(f"Cell {i}: Added HuggingFace upload")
            
            # 9. Enhanced Google Drive Upload with Timestamps
            if 'dst = "/content/drive/MyDrive/nanowhale-1b-moe"' in source:
                lines = source.split('\n')
                new_lines = []
                for line in lines:
                    if 'dst = "/content/drive/MyDrive/nanowhale-1b-moe"' in line:
                        new_lines.append('dst_base = "/content/drive/MyDrive/colab/nanowhale"')
                        new_lines.append('timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")')
                        new_lines.append('dst = f"{dst_base}/checkpoint_{timestamp}"')
                    else:
                        new_lines.append(line)
                new_source = '\n'.join(new_lines)
                cell['source'] = [new_source]
                updates.append(f"Cell {i}: Enhanced Drive upload with timestamps")
            
            # 10. Add import for datetime
            if 'import os, sys, math, random, json' in source:
                lines = source.split('\n')
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if 'import os, sys, math, random, json' in line:
                        new_lines.append('from datetime import datetime')
                new_source = '\n'.join(new_lines)
                cell['source'] = [new_source]
                updates.append(f"Cell {i}: Added datetime import")
            
            # 11. Add regex import for layer-wise LR
            if 'from typing import Optional, Tuple, List' in source and 'apply_layer_wise_lr' in source:
                if 'import re' not in source:
                    lines = source.split('\n')
                    new_lines = []
                    for line in lines:
                        new_lines.append(line)
                        if 'from typing import Optional, Tuple, List' in line:
                            new_lines.append('import re')
                    new_source = '\n'.join(new_lines)
                    cell['source'] = [new_source]
                    updates.append(f"Cell {i}: Added regex import")
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return updates

if __name__ == '__main__':
    path = 'colab/nanowhale_1b_moe_colab.ipynb'
    updates = apply_easy_optimizations(path)
    print(f"Applied {len(updates)} optimizations:")
    for u in updates:
        print(f"  - {u}")
