#!/usr/bin/env python3
"""
Fix remaining issues in the notebook:
1. Fix vocab size inconsistency (128000 -> 129280)
2. Fix training steps (25000 -> 5000)
3. Fix precision (bf16 -> fp32)
4. Add missing 're' import
5. Fix mismatched parentheses in cell 15
"""

import json
import re

def fix_notebook_issues(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    fixes = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            original_source = source
            
            # Fix 1: Vocab size - replace all 128000 with 129280
            if 'vocab_size=128000' in source:
                source = source.replace('vocab_size=128000', 'vocab_size=129280')
                fixes.append(f"Cell {i}: Fixed vocab size 128000 -> 129280")
            
            # Fix 2: Training steps - replace 25000 with 5000
            if 'STEPS = 25000' in source:
                source = source.replace('STEPS = 25000', 'STEPS = 5000')
                fixes.append(f"Cell {i}: Fixed training steps 25000 -> 5000")
            
            # Fix 3: Precision - replace bf16 with fp32
            if 'torch.bfloat16' in source:
                source = source.replace('torch.bfloat16', 'torch.float32')
                fixes.append(f"Cell {i}: Fixed precision bf16 -> fp32")
            
            # Fix 4: Add missing 're' import
            if 'from typing import Optional, Tuple, List' in source and 'import re' not in source:
                source = source.replace(
                    'from typing import Optional, Tuple, List',
                    'import re\nfrom typing import Optional, Tuple, List'
                )
                fixes.append(f"Cell {i}: Added missing 're' import")
            
            # Fix 5: Fix mismatched parentheses in cell 15
            if i == 15:  # This is the problematic cell
                # Look for the batch size print statement with mismatched parentheses
                if 'print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM} grad accum = effective {BATCH_SIZE * GRAD_ACCUM}\nTraining: 5000 steps with 8k context (Flash Attention + gradient checkpointing)")' in source:
                    # Fix the string - it's missing a closing quote
                    source = source.replace(
                        'print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM} grad accum = effective {BATCH_SIZE * GRAD_ACCUM}\nTraining: 5000 steps with 8k context (Flash Attention + gradient checkpointing)")',
                        'print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM} grad accum = effective {BATCH_SIZE * GRAD_ACCUM}")\nprint("Training: 5000 steps with 8k context (Flash Attention + gradient checkpointing)")'
                    )
                    fixes.append(f"Cell {i}: Fixed mismatched parentheses in print statement")
            
            # Update cell if changed
            if source != original_source:
                cell['source'] = [source]
    
    # Save the fixed notebook
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    return fixes

if __name__ == '__main__':
    path = 'colab/nanowhale_1b_moe_colab.ipynb'
    fixes = fix_notebook_issues(path)
    print(f"Applied {len(fixes)} fixes:")
    for fix in fixes:
        print(f"  - {fix}")
