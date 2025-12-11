#!/usr/bin/env python
"""
Deep investigation of evaluation metrics and data quality
"""
import sys
sys.path.insert(0, '..')

from script.utils.data_util import loader
import torch
import numpy as np

print("Loading CAIDA dataset...")
data = loader('caida')

print('\n' + '='*60)
print('Deep Investigation of Evaluation Issues')
print('='*60)

# Get test snapshots
test_start = data["time_length"] - 3
print(f'\nTest snapshots: {test_start} to {data["time_length"]-1}')

for i in range(test_start, data["time_length"]):
    print(f'\n{"="*60}')
    print(f'Snapshot {i}')
    print(f'{"="*60}')
    
    edge_index = data["edge_index_list"][i]
    pos_edges = data["pedges"][i]
    neg_edges = data["nedges"][i]
    new_pos_edges = data["new_pedges"][i]
    new_neg_edges = data["new_nedges"][i]
    
    print(f'\nGraph edges: {edge_index.shape[1]} edges')
    print(f'  Min node: {edge_index.min().item()}, Max node: {edge_index.max().item()}')
    
    print(f'\nTest positive edges: {pos_edges.shape[1]} edges')
    if pos_edges.shape[1] > 0:
        print(f'  Sample: {pos_edges[:, :min(3, pos_edges.shape[1])].tolist()}')
        print(f'  Min node: {pos_edges.min().item()}, Max node: {pos_edges.max().item()}')
    
    print(f'\nTest negative edges: {neg_edges.shape[1]} edges')
    if neg_edges.shape[1] > 0:
        print(f'  Sample: {neg_edges[:, :min(3, neg_edges.shape[1])].tolist()}')
        print(f'  Min node: {neg_edges.min().item()}, Max node: {neg_edges.max().item()}')
    
    print(f'\nNew positive edges: {new_pos_edges.shape[1]} edges')
    if new_pos_edges.shape[1] > 0:
        print(f'  Sample: {new_pos_edges[:, :min(3, new_pos_edges.shape[1])].tolist()}')
        print(f'  Min node: {new_pos_edges.min().item()}, Max node: {new_pos_edges.max().item()}')
    
    print(f'\nNew negative edges: {new_neg_edges.shape[1]} edges')
    if new_neg_edges.shape[1] > 0:
        print(f'  Sample: {new_neg_edges[:, :min(3, new_neg_edges.shape[1])].tolist()}')
        print(f'  Min node: {new_neg_edges.min().item()}, Max node: {new_neg_edges.max().item()}')
    
    # Check for duplicates in positive edges
    pos_set = set()
    duplicates = 0
    for j in range(pos_edges.shape[1]):
        edge = (pos_edges[0, j].item(), pos_edges[1, j].item())
        if edge in pos_set:
            duplicates += 1
        pos_set.add(edge)
    
    if duplicates > 0:
        print(f'\n⚠️  WARNING: {duplicates} duplicate positive edges!')
    
    # Check if negative edges are actually in the graph (should NOT be)
    edge_set = set()
    for j in range(edge_index.shape[1]):
        edge_set.add((edge_index[0, j].item(), edge_index[1, j].item()))
    
    neg_in_graph = 0
    for j in range(neg_edges.shape[1]):
        if (neg_edges[0, j].item(), neg_edges[1, j].item()) in edge_set:
            neg_in_graph += 1
    
    if neg_in_graph > 0:
        print(f'\n⚠️  CRITICAL: {neg_in_graph}/{neg_edges.shape[1]} negative edges are IN the graph!')
        print('This means negative samples are actually positive - evaluation is broken!')

print('\n' + '='*60)
print('Summary')
print('='*60)
print('\nPossible issues:')
print('1. Very small test set (only 1-2 edges per snapshot)')
print('2. Negative sampling might be broken')
print('3. Data preprocessing might have issues')
print('='*60)
