#!/usr/bin/env python
"""
Test script to verify CAIDA dataset loading
"""
import sys
sys.path.insert(0, '..')

from script.utils.data_util import loader

print("Testing CAIDA dataset loading...")
data = loader('caida')

print('\n' + '='*60)
print('✓ Dataset loaded successfully!')
print('='*60)
print(f'✓ Number of nodes: {data["num_nodes"]:,}')
print(f'✓ Number of time snapshots: {data["time_length"]}')
print(f'✓ First snapshot edges: {data["edge_index_list"][0].shape}')
print(f'✓ Last snapshot edges: {data["edge_index_list"][-1].shape}')
print('='*60)
