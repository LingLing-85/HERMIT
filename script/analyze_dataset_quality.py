#!/usr/bin/env python
"""
Analyze the entire CAIDA dataset to understand data quality
"""
import sys
sys.path.insert(0, '..')

from script.utils.data_util import loader
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Loading CAIDA dataset...")
data = loader('caida')

print('\n' + '='*60)
print('CAIDA Dataset Quality Analysis')
print('='*60)

# Analyze edge counts over time
edge_counts = []
for i, edge_index in enumerate(data['edge_index_list']):
    edge_counts.append(edge_index.shape[1])

edge_counts = np.array(edge_counts)

print(f'\nTotal snapshots: {len(edge_counts)}')
print(f'Edge count statistics:')
print(f'  Mean: {edge_counts.mean():.1f}')
print(f'  Median: {np.median(edge_counts):.1f}')
print(f'  Min: {edge_counts.min()}')
print(f'  Max: {edge_counts.max()}')
print(f'  Std: {edge_counts.std():.1f}')

# Find snapshots with very few edges
threshold = 10
sparse_snapshots = np.where(edge_counts < threshold)[0]
print(f'\nSnapshots with < {threshold} edges: {len(sparse_snapshots)}')
if len(sparse_snapshots) > 0:
    print(f'  Indices: {sparse_snapshots[:20].tolist()}...' if len(sparse_snapshots) > 20 else f'  Indices: {sparse_snapshots.tolist()}')

# Show last 20 snapshots
print(f'\nLast 20 snapshots edge counts:')
for i in range(max(0, len(edge_counts)-20), len(edge_counts)):
    print(f'  Snapshot {i}: {edge_counts[i]} edges')

# Create visualization
plt.figure(figsize=(15, 5))
plt.plot(edge_counts)
plt.xlabel('Time Snapshot')
plt.ylabel('Number of Edges')
plt.title('CAIDA Dataset: Edge Count Over Time')
plt.grid(True, alpha=0.3)
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold} edges)')
plt.legend()
plt.tight_layout()
plt.savefig('../data/output/caida_edge_counts.png', dpi=150)
print(f'\n✓ Saved visualization to: data/output/caida_edge_counts.png')

# Recommendation
print('\n' + '='*60)
print('Recommendations')
print('='*60)

if len(sparse_snapshots) > 100:
    print('⚠️  WARNING: Many snapshots have very few edges!')
    print('   This suggests data quality issues in preprocessing.')
    print('\nSuggestions:')
    print('1. Check if CSV files are being read correctly')
    print('2. Verify node ID mapping is working properly')
    print('3. Consider filtering out snapshots with < 10 edges')
    print('4. Check if the original CSV data has this sparsity')
else:
    print('✓ Most snapshots have reasonable edge counts')
    print('  Only the last few snapshots are sparse')
    print('\nSuggestion: Use a different test set split')
    print('  e.g., use snapshots from the middle of the dataset')

print('='*60)
