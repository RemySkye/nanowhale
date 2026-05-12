#!/usr/bin/env python3
"""
Comprehensive validation of the Colab notebook.
"""

import json
import re

def validate_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    issues = []
    warnings = []
    
    print(f"📋 Validating: {path}")
    print(f"Total cells: {len(nb['cells'])} ({sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} code, {sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')} markdown)")
    print()
    
    # Check 1: Vocab size consistency
    vocab_sizes = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            matches = re.findall(r'vocab_size=(\d+)', source)
            vocab_sizes.extend(matches)
    
    if vocab_sizes:
        unique = set(vocab_sizes)
        if len(unique) > 1:
            issues.append(f"❌ Vocab size inconsistency: {unique}")
        elif '129280' not in unique:
            issues.append(f"❌ Vocab size should be 129280, found: {unique}")
        else:
            print("✅ Vocab size consistent: 129280")
    
    # Check 2: Max context length
    max_contexts = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            matches = re.findall(r'max_position_embeddings=(\d+)', source)
            max_contexts.extend(matches)
    
    if max_contexts:
        if '131072' not in set(max_contexts):
            warnings.append(f"⚠️ Max context should be 131072 (128k), found: {set(max_contexts)}")
        else:
            print("✅ Max context: 128k")
    
    # Check 3: Training steps
    steps_found = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'STEPS = ' in source:
                match = re.search(r'STEPS = (\d+)', source)
                if match and match.group(1) == '5000':
                    print("✅ Training steps: 5000")
                    steps_found = True
                else:
                    issues.append(f"❌ Training steps should be 5000, found: {match.group(1) if match else 'not found'}")
    
    # Check 4: Precision (fp32)
    bf16_found = False
    fp32_found = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'torch.bfloat16' in source:
                bf16_found = True
            if 'torch.float32' in source or 'dtype=torch.float32' in source:
                fp32_found = True
    
    if bf16_found:
        issues.append("❌ Found bf16 usage - should use fp32 for stability")
    elif fp32_found:
        print("✅ Using fp32 precision")
    
    # Check 5: Flash Attention
    flash_attention = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'torch.compile' in source and 'max-autotune' in source:
                flash_attention = True
                print("✅ Flash Attention enabled (torch.compile max-autotune)")
    
    # Check 6: Gradient checkpointing
    grad_ckpt = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'gradient_checkpointing = True' in source:
                grad_ckpt = True
                print("✅ Gradient checkpointing enabled")
    
    # Check 7: HuggingFace upload
    hf_upload = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'huggingface_hub' in source and 'upload_folder' in source:
                hf_upload = True
                print("✅ HuggingFace upload functionality present")
    
    # Check 8: Google Drive upload
    drive_upload = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'drive.mount' in source and 'shutil.copytree' in source:
                drive_upload = True
                print("✅ Google Drive upload functionality present")
    
    # Check 9: NaN protection
    nan_protection = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'torch.isnan' in source or 'torch.isinf' in source:
                nan_protection = True
                print("✅ NaN protection present")
    
    # Check 10: Learning rate scheduler
    lr_scheduler = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def get_lr(' in source or 'decay_to_zero' in source.lower():
                lr_scheduler = True
                print("✅ Decay-to-zero LR scheduler present")
    
    # Check 11: Batch size ramping
    batch_ramp = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def get_batch_size(' in source:
                batch_ramp = True
                print("✅ Batch size ramping present")
    
    # Check 12: Imports
    required_imports = ['datetime', 're']
    all_imports = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            all_imports.extend(re.findall(r'import (\w+)', source))
            all_imports.extend(re.findall(r'from (\w+)', source))
    
    for imp in required_imports:
        if imp not in all_imports:
            issues.append(f"❌ Missing required import: {imp}")
        else:
            print(f"✅ Import present: {imp}")
    
    # Check 13: Syntax validation (basic)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Check for common syntax errors
            if source.count('(') != source.count(')'):
                issues.append(f"❌ Cell {i}: Mismatched parentheses")
            if source.count('[') != source.count(']'):
                issues.append(f"❌ Cell {i}: Mismatched brackets")
            if '"""' in source and source.count('"""') % 2 != 0:
                issues.append(f"❌ Cell {i}: Unclosed triple quotes")
    
    print()
    print("=" * 60)
    if issues:
        print(f"❌ FOUND {len(issues)} ISSUES:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ NO ISSUES FOUND - Notebook appears valid!")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  {warning}")
    
    return len(issues) == 0

if __name__ == '__main__':
    path = 'colab/nanowhale_1b_moe_colab.ipynb'
    valid = validate_notebook(path)
    exit(0 if valid else 1)
