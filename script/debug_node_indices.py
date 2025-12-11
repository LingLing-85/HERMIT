#!/usr/bin/env python
"""
Debug script to check node indexing and data consistency
"""
import sys
sys.path.insert(0, '..')

from script.utils.data_util import loader
import torch

print("Loading CAIDA dataset...")
data = loader('caida')

print('\n' + '='*60)
print('Dataset Information')
print('='*60)
print(f'Number of nodes: {data["num_nodes"]:,}')
print(f'Number of time snapshots: {data["time_length"]}')

# Check edge indices
print('\n' + '='*60)
print('Edge Index Analysis')
print('='*60)

for i, edge_index in enumerate(data['edge_index_list'][:5]):
    max_node = edge_index.max().item()
    min_node = edge_index.min().item()
    print(f'Snapshot {i}: shape={edge_index.shape}, min={min_node}, max={max_node}')

# Check if max node index exceeds num_nodes
max_overall = max([e.max().item() for e in data['edge_index_list']])
print(f'\nOverall max node index: {max_overall:,}')
print(f'Declared num_nodes: {data["num_nodes"]:,}')

if max_overall >= data["num_nodes"]:
    print(f'\n⚠️  ERROR: Max node index ({max_overall}) >= num_nodes ({data["num_nodes"]})')
    print('This will cause index out of bounds errors!')
else:
    print(f'\n✓ OK: Max node index ({max_overall}) < num_nodes ({data["num_nodes"]})')

# Check test edges
print('\n' + '='*60)
print('Test Edge Analysis')
print('='*60)

if 'pedges' in data and len(data['pedges']) > 0:
    for i in range(min(3, len(data['pedges']))):
        if data['pedges'][i].numel() > 0:
            max_node = data['pedges'][i].max().item()
            print(f'Test pedges {i}: max node = {max_node}')
        
        if data['new_pedges'][i].numel() > 0:
            max_node = data['new_pedges'][i].max().item()
            print(f'Test new_pedges {i}: max node = {max_node}')

print('='*60)
