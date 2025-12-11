#!/usr/bin/env python
"""
Investigate why AUC=1.0 - check for data leakage
"""
import sys
sys.path.insert(0, '..')

from script.utils.data_util import loader
import torch

print("Loading CAIDA dataset...")
data = loader('caida')

print('\n' + '='*60)
print('Investigating AUC=1.0 Issue')
print('='*60)

# Check the structure of test edges
print(f'\nNumber of time snapshots: {data["time_length"]}')
print(f'Test length: 3')
print(f'Training snapshots: 0-{data["time_length"]-4}')
print(f'Test snapshots: {data["time_length"]-3}-{data["time_length"]-1}')

# Examine test edges
test_start = data["time_length"] - 3

print('\n' + '='*60)
print('Test Edge Statistics')
print('='*60)

for i in range(test_start, data["time_length"]):
    idx = i
    print(f'\nSnapshot {i}:')
    print(f'  Edge index shape: {data["edge_index_list"][idx].shape}')
    print(f'  Pos edges shape: {data["pedges"][idx].shape}')
    print(f'  Neg edges shape: {data["nedges"][idx].shape}')
    print(f'  New pos edges shape: {data["new_pedges"][idx].shape}')
    print(f'  New neg edges shape: {data["new_nedges"][idx].shape}')
    
    # Check if positive edges are actually in the graph
    edge_index = data["edge_index_list"][idx]
    pos_edges = data["pedges"][idx]
    
    # Convert to sets for comparison
    edge_set = set()
    for j in range(edge_index.shape[1]):
        edge_set.add((edge_index[0, j].item(), edge_index[1, j].item()))
    
    # Check how many positive test edges are in the training graph
    in_graph = 0
    for j in range(pos_edges.shape[1]):
        if (pos_edges[0, j].item(), pos_edges[1, j].item()) in edge_set:
            in_graph += 1
    
    print(f'  ⚠️  Pos edges in graph: {in_graph}/{pos_edges.shape[1]} ({100*in_graph/pos_edges.shape[1]:.1f}%)')

print('\n' + '='*60)
print('Analysis')
print('='*60)
print('If positive test edges are already in the training graph,')
print('the model can achieve AUC=1.0 by simply memorizing the graph!')
print('This is a DATA LEAKAGE problem.')
print('='*60)
