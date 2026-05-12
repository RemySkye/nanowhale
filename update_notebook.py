#!/usr/bin/env python3
"""
Update nanowhale Colab notebook with optimizations and bug fixes.
"""

import json
import sys

def update_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    updates_applied = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # 1. Fix vocab_size: 128000 -> 129280
            if 'vocab_size=128000' in source and 'DeepseekV4Config' in source:
                new_source = source.replace('vocab_size=128000', 'vocab_size=129280')
                cell['source'] = [new_source]
                updates_applied.append(f"Cell {i}: Fixed vocab_size to 129280")
            
            # 2. Fix vocab_size in model config instantiation
            if 'vocab_size=128000' in source and 'DeepseekV4Config(' in source:
                new_source = source.replace('vocab_size=128000', 'vocab_size=129280')
                cell['source'] = [new_source]
                updates_applied.append(f"Cell {i}: Fixed vocab_size in config instantiation")
            
            # 3. Replace bf16 with fp32 in autocast
            if 'torch.amp.autocast("cuda", dtype=torch.bfloat16)' in source:
                new_source = source.replace(
                    'torch.amp.autocast("cuda", dtype=torch.bfloat16)',
                    'torch.amp.autocast("cuda", dtype=torch.float32)'
                )
                cell['source'] = [new_source]
                updates_applied.append(f"Cell {i}: Changed to fp32 precision")
            
            # 4. Add NaN detection in training loop
            if 'scaler.scale(loss).backward()' in source and 'loss = out.loss' in source:
                # Insert NaN check after loss computation
                lines = source.split('\n')
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if 'loss = out.loss / GRAD_ACCUM' in line:
                        new_lines.append('        # NaN protection')
                        new_lines.append('        if torch.isnan(loss) or torch.isinf(loss):')
                        new_lines.append('            print(f"  [WARN] NaN/Inf loss at step {step}, skipping")')
                        new_lines.append('            optimizer.zero_grad()')
                        new_lines.append('            continue')
                new_source = '\n'.join(new_lines)
                cell['source'] = [new_source]
                updates_applied.append(f"Cell {i}: Added NaN detection")
            
            # 5. Replace cosine LR with decay-to-zero
            if 'torch.optim.AdamW' in source and 'lr=' in source:
                # Find the optimizer line and add LR scheduler after it
                lines = source.split('\n')
                new_lines = []
                scheduler_added = False
                for line in lines:
                    new_lines.append(line)
                    if 'optimizer = torch.optim.AdamW' in line and not scheduler_added:
                        # Add decay-to-zero LR scheduler
                        new_lines.append('')
                        new_lines.append('# Decay-to-zero LR schedule (superior to cosine per 2025 research)')
                        new_lines.append('def get_lr(step, total_steps, peak_lr):')
                        new_lines.append('    """Linear warmup then decay to zero."""')
                        new_lines.append('    warmup_steps = total_steps * 0.05')
                        new_lines.append('    if step < warmup_steps:')
                        new_lines.append('        return peak_lr * (step / warmup_steps)')
                        new_lines.append('    else:')
                        new_lines.append('        progress = (step - warmup_steps) / (total_steps - warmup_steps)')
                        new_lines.append('        return peak_lr * (1 - progress)')
                        new_lines.append('')
                        new_lines.append('# Update optimizer LR each step')
                        new_lines.append('def update_lr(step):')
                        new_lines.append('    lr = get_lr(step, STEPS, LR)')
                        new_lines.append('    for param_group in optimizer.param_groups:')
                        new_lines.append('        param_group["lr"] = lr')
                        new_lines.append('    return lr')
                        scheduler_added = True
                if scheduler_added:
                    new_source = '\n'.join(new_lines)
                    cell['source'] = [new_source]
                    updates_applied.append(f"Cell {i}: Added decay-to-zero LR scheduler")
            
            # 6. Add batch size ramping (Seesaw-inspired)
            if 'BATCH_SIZE =' in source and 'GRAD_ACCUM =' in source:
                lines = source.split('\n')
                new_lines = []
                ramp_added = False
                for line in lines:
                    new_lines.append(line)
                    if 'print(f"Batch size:' in line:
                        new_lines.append('')
                        new_lines.append('# Batch size ramping schedule (Seesaw-inspired)')
                        new_lines.append('def get_batch_size(step, base_batch, max_batch=8):')
                        new_lines.append('    """Double batch size at 25%, 50%, 75% of training."""')
                        new_lines.append('    progress = step / STEPS')
                        new_lines.append('    if progress < 0.25:')
                        new_lines.append('        return base_batch')
                        new_lines.append('    elif progress < 0.5:')
                        new_lines.append('        return min(base_batch * 2, max_batch)')
                        new_lines.append('    elif progress < 0.75:')
                        new_lines.append('        return min(base_batch * 4, max_batch)')
                        new_lines.append('    else:')
                        new_lines.append('        return min(base_batch * 8, max_batch)')
                        new_lines.append('')
                        ramp_added = True
                if ramp_added:
                    new_source = '\n'.join(new_lines)
                    cell['source'] = [new_source]
                    updates_applied.append(f"Cell {i}: Added batch size ramping")
            
            # 7. Improve data quality filtering
            if 'if not text or len(text) < min_chars: continue' in source:
                lines = source.split('\n')
                new_lines = []
                filter_added = False
                for line in lines:
                    new_lines.append(line)
                    if 'if not text or len(text) < min_chars: continue' in line:
                        new_lines.append('        # Quality filtering for better training efficiency')
                        new_lines.append('        # Skip low-quality or problematic samples')
                        new_lines.append('        if len(text) < 200:  # Increased minimum length')
                        new_lines.append('            continue')
                        new_lines.append('        # Basic quality checks')
                        new_lines.append('        if text.count(\'\\n\') < 2:  # At least some structure')
                        new_lines.append('            continue')
                        new_lines.append('        # Skip if too many special characters')
                        new_lines.append('        if text.count(\'<\') > len(text) * 0.1:  # HTML tags')
                        new_lines.append('            continue')
                        new_lines.append('        if text.count(\'=\') > len(text) * 0.05:  # Math formulas ok, but not too many')
                        new_lines.append('            continue')
                        filter_added = True
                if filter_added:
                    new_source = '\n'.join(new_lines)
                    cell['source'] = [new_source]
                    updates_applied.append(f"Cell {i}: Added quality filtering")
            
            # 8. Add weight averaging for better convergence
            if 'optimizer.zero_grad()' in source and 'scaler.step(optimizer)' in source:
                lines = source.split('\n')
                new_lines = []
                wa_added = False
                for line in lines:
                    new_lines.append(line)
                    if 'scaler.step(optimizer)' in line:
                        new_lines.append('        # Weight averaging for better convergence (anytime training)')
                        new_lines.append('        # Simple moving average of weights')
                        new_lines.append('        if step > STEPS * 0.5:  # Start averaging in second half')
                        new_lines.append('            with torch.no_grad():')
                        new_lines.append('                for param in model.parameters():')
                        new_lines.append('                    if hasattr(param, "wa"):')
                        new_lines.append('                        param.wa = 0.99 * param.wa + 0.01 * param.data')
                        new_lines.append('                    else:')
                        new_lines.append('                        param.wa = param.data.clone()')
                        new_lines.append('        # Apply averaged weights for logging/saving')
                        new_lines.append('        if step % LOG_EVERY == 0 and hasattr(list(model.parameters())[0], "wa"):')
                        new_lines.append('            with torch.no_grad():')
                        new_lines.append('                for param in model.parameters():')
                        new_lines.append('                    if hasattr(param, "wa"):')
                        new_lines.append('                        param.data = param.wa')
                        wa_added = True
                if wa_added:
                    new_source = '\n'.join(new_lines)
                    cell['source'] = [new_source]
                    updates_applied.append(f"Cell {i}: Added weight averaging")
            
            # 9. Update LR usage in training loop
            if 'update_lr(step)' not in source and 'for step in range(STEPS):' in source:
                lines = source.split('\n')
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if 'for step in range(STEPS):' in line:
                        new_lines.append('    # Update learning rate')
                        new_lines.append('    current_lr = update_lr(step)')
                new_source = '\n'.join(new_lines)
                cell['source'] = [new_source]
                updates_applied.append(f"Cell {i}: Added LR update call")
            
            # 10. Add dynamic batch size update
            if 'get_batch_size(' in source and 'build_batch(tokenized_ids, batch_idx, BATCH_SIZE)' in source:
                lines = source.split('\n')
                new_lines = []
                for line in lines:
                    if 'input_ids, labels, attn_mask = build_batch(tokenized_ids, batch_idx, BATCH_SIZE)' in line:
                        new_lines.append('    # Dynamic batch size (Seesaw ramping)')
                        new_lines.append('    current_batch = get_batch_size(step, BATCH_SIZE)')
                        new_lines.append('    input_ids, labels, attn_mask = build_batch(tokenized_ids, batch_idx, current_batch)')
                    else:
                        new_lines.append(line)
                new_source = '\n'.join(new_lines)
                cell['source'] = [new_source]
                updates_applied.append(f"Cell {i}: Added dynamic batch size")
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return updates_applied

if __name__ == '__main__':
    path = 'colab/nanowhale_1b_moe_colab.ipynb'
    updates = update_notebook(path)
    print(f"Applied {len(updates)} updates:")
    for u in updates:
        print(f"  - {u}")
