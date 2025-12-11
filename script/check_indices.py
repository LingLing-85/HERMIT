import torch
import numpy as np

def check_indices():
    print("Loading processed data...")
    data_path = '../data/input/processed/caida/caida.pt'
    edge_index_list = torch.load(data_path)
    
    print(f"Loaded {len(edge_index_list)} snapshots")
    
    max_idx = 0
    min_idx = float('inf')
    
    for i, edge_index in enumerate(edge_index_list):
        # edge_index is [2, N]
        current_max = edge_index.max().item()
        current_min = edge_index.min().item()
        
        max_idx = max(max_idx, current_max)
        min_idx = min(min_idx, current_min)
        
        if i % 10 == 0:
            print(f"Snapshot {i}: min={current_min}, max={current_max}")
            
    print("="*40)
    print(f"Overall Min Index: {min_idx}")
    print(f"Overall Max Index: {max_idx}")
    
    # Check what data_util thinks
    from utils.make_edges_new import get_edges
    undirected_edges = get_edges(edge_index_list)
    
    # data_util.py logic:
    # num_nodes = int(np.max(np.hstack(undirected_edges))) + 1
    
    # Let's simulate this
    all_edges = np.hstack(undirected_edges)
    calculated_num_nodes = int(np.max(all_edges)) + 1
    
    print(f"Calculated num_nodes (from make_edges): {calculated_num_nodes}")
    
    if max_idx >= calculated_num_nodes:
        print(f"❌ CRITICAL ERROR: Max index {max_idx} >= num_nodes {calculated_num_nodes}")
    else:
        print(f"✓ Indices seem correct relative to calculated num_nodes")

if __name__ == "__main__":
    check_indices()
