#!/usr/bin/env python
"""
Debug preprocessing to find where edges are lost
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Test with first CSV file
csv_file = Path('/mnt/NewSSD/workshop2025/caida_dataset_rtt/caida_dataset_select/20150103.csv')

print(f"Testing file: {csv_file.name}")
print("="*60)

# Step 1: Read CSV
df = pd.read_csv(csv_file)
print(f"\n1. After reading CSV: {len(df)} rows")
print(f"   Columns: {df.columns.tolist()}")
print(f"   First few rows:")
print(df.head())

# Step 2: Check for non-numeric values
print(f"\n2. Checking for non-numeric values...")
source_numeric = pd.to_numeric(df['source'], errors='coerce')
target_numeric = pd.to_numeric(df['target'], errors='coerce')

source_na = source_numeric.isna().sum()
target_na = target_numeric.isna().sum()

print(f"   Non-numeric source values: {source_na}")
print(f"   Non-numeric target values: {target_na}")

# Step 3: Filter
df_filtered = df[pd.to_numeric(df['source'], errors='coerce').notna()]
df_filtered = df_filtered[pd.to_numeric(df_filtered['target'], errors='coerce').notna()]
print(f"\n3. After filtering: {len(df_filtered)} rows")

# Step 4: Convert to int
df_filtered['source'] = df_filtered['source'].astype(int)
df_filtered['target'] = df_filtered['target'].astype(int)
print(f"\n4. After converting to int: {len(df_filtered)} rows")

# Step 5: Check unique nodes
unique_sources = df_filtered['source'].nunique()
unique_targets = df_filtered['target'].nunique()
all_nodes = set(df_filtered['source'].unique()) | set(df_filtered['target'].unique())

print(f"\n5. Unique nodes:")
print(f"   Unique sources: {unique_sources}")
print(f"   Unique targets: {unique_targets}")
print(f"   Total unique nodes: {len(all_nodes)}")

# Step 6: Check what happens with node mapping
print(f"\n6. Node mapping simulation:")
print(f"   Original node IDs range: {min(all_nodes)} to {max(all_nodes)}")

# Create mapping like in preprocessing script
node_to_id = {int(node): idx for idx, node in enumerate(sorted(all_nodes))}
print(f"   Mapped node IDs range: 0 to {len(node_to_id)-1}")

# Map the edges
sources_mapped = df_filtered['source'].map(node_to_id).values
targets_mapped = df_filtered['target'].map(node_to_id).values

print(f"\n7. After mapping:")
print(f"   Number of edges: {len(sources_mapped)}")
print(f"   Sample edges (original): {list(zip(df_filtered['source'].head(), df_filtered['target'].head()))}")
print(f"   Sample edges (mapped): {list(zip(sources_mapped[:5], targets_mapped[:5]))}")

# Step 8: Create edge index (like in preprocessing)
edges_forward = np.stack([sources_mapped, targets_mapped], axis=0)
edges_backward = np.stack([targets_mapped, sources_mapped], axis=0)
edge_index = np.concatenate([edges_forward, edges_backward], axis=1)

print(f"\n8. Final edge index:")
print(f"   Shape: {edge_index.shape}")
print(f"   Expected: (2, {len(df_filtered)*2}) for undirected graph")

print("\n" + "="*60)
print("DIAGNOSIS:")
if edge_index.shape[1] == len(df_filtered) * 2:
    print("✓ Preprocessing logic is CORRECT")
    print("✗ Problem must be elsewhere!")
else:
    print(f"✗ Lost {len(df_filtered)*2 - edge_index.shape[1]} edges in preprocessing")
