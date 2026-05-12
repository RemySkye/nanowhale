#!/usr/bin/env python3
"""
Fix the duplicate function definitions and extra closing parentheses in cell 15.
"""

import json

with open('colab/nanowhale_1b_moe_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell15 = nb['cells'][15]
source = ''.join(cell15['source'])

# The issue: there are duplicate get_batch_size definitions and duplicate optimizer lines
# Let's clean it up

lines = source.split('\n')
cleaned_lines = []
seen_batch_func = False
seen_optimizer_end = False
skip_until_empty = False

i = 0
while i < len(lines):
    line = lines[i]
    
    # Skip the duplicate get_batch_size definition
    if '# Batch size ramping schedule (Seesaw-inspired)' in line and seen_batch_func:
        # Skip this duplicate block until we hit the next section
        while i < len(lines) and ('# ---- torch.compile ----' not in lines[i]):
            i += 1
        continue
    
    # Track if we've seen the first get_batch_size
    if 'def get_batch_size(' in line:
        seen_batch_func = True
    
    # Fix the duplicate optimizer line and extra closing paren
    if 'betas=(0.9, 0.95), fused=True)' in line and seen_optimizer_end:
        # Skip this duplicate line
        i += 1
        continue
    
    if 'optimizer = torch.optim.AdamW(' in line:
        seen_optimizer_end = True
    
    # Skip standalone closing parenthesis that's extra
    if line.strip() == ')' and i > 0 and lines[i-1].strip().endswith(')'):
        # This is likely an extra closing paren
        i += 1
        continue
    
    cleaned_lines.append(line)
    i += 1

# Rejoin and update cell
new_source = '\n'.join(cleaned_lines)
cell15['source'] = [new_source]

# Save
with open('colab/nanowhale_1b_moe_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed duplicate definitions and extra parentheses in cell 15")
