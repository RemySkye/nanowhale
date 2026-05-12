#!/usr/bin/env python3
import json

with open('colab/nanowhale_1b_moe_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell15 = nb['cells'][15]
source = ''.join(cell15['source'])
lines = source.split('\n')

print("Lines with more closing than opening parens:")
for i, line in enumerate(lines):
    if line.count(')') > line.count('('):
        print(f"Line {i}: {line[:100]}")
