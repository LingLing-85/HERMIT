
import torch
import numpy as np
import sys
import os
from tqdm import tqdm

# Add script directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'script'))

# CRITICAL: Set sys.argv BEFORE importing config to ensure dataset='caida'
sys.argv = ['calculate_baseline.py', '--dataset', 'caida']

from script.config import args
from script.utils.util import set_random
from script.utils.data_util import loader

# Ensure root directory is in sys.path
ROOT_DIR = '/home/ling/ling/HMPTGN_edge_feature'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

def load_caida_data():
    """Load CAIDA data"""
    print("Loading CAIDA data...")
    set_random(args.seed)
    
    # Context manager to temporarily change directory for loader
    original_cwd = os.getcwd()
    try:
        target_dir = os.path.join(ROOT_DIR, 'script')
        if os.path.exists(target_dir):
            os.chdir(target_dir)
            data = loader(dataset='caida')
    finally:
        os.chdir(original_cwd)

    stats_file = os.path.join(ROOT_DIR, 'data/input/processed/caida/rtt_stats.pt')
    stats = torch.load(stats_file)
    return data, stats

def denormalize_rtt(normalized_rtt, avg_min, avg_max):
    """Denormalize RTT"""
    log_rtt = normalized_rtt * (avg_max - avg_min) + avg_min
    rtt_ms = np.expm1(log_rtt)
    return rtt_ms

def calculate_baselines():
    data, stats = load_caida_data()
    
    total_snapshots = data['time_length']
    test_length = 60
    train_length = total_snapshots - test_length
    
    print(f"\nEvaluating Baselines on last {test_length} snapshots...")
    print(f"Test Range: {train_length} -> {total_snapshots}")
    
    # Store edges from previous snapshot for "Last Value" prediction
    prev_snapshot_rtt = {} # (src, dst) -> rtt_ms
    
    # Metrics accumulators
    last_value_errors = []
    historical_mean_errors = []
    
    # Initialize history with training data (optional, but good for mean)
    # For now, we'll just track history dynamically during test for simplicity
    # or just use "Last Value" which is the strongest trivial baseline
    
    for t in tqdm(range(train_length - 1, total_snapshots)):
        # Load current snapshot data
        weights = data['weights'][t]
        if weights is None: continue
        
        edge_index = data['edge_index_list'][t]
        
        # Get ground truth for this snapshot
        current_rtt_norm = weights[:, 0].numpy()
        current_rtt_ms = denormalize_rtt(current_rtt_norm, stats['avg_min'], stats['avg_max'])
        
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()
        
        # If we are in the test period, calculate errors
        if t >= train_length:
            for i in range(len(current_rtt_ms)):
                src, dst = src_nodes[i], dst_nodes[i]
                edge_key = (src, dst)
                true_val = current_rtt_ms[i]
                
                # Baseline 1: Last Value Prediction
                # "Predict that RTT will be the same as the last time we saw this link"
                if edge_key in prev_snapshot_rtt:
                    pred_val = prev_snapshot_rtt[edge_key]
                    last_value_errors.append(abs(pred_val - true_val))
                
                # (Can add more baselines here if needed)

        # Update history for next step
        # Note: We update AFTER prediction to prevent leakage
        for i in range(len(current_rtt_ms)):
            src, dst = src_nodes[i], dst_nodes[i]
            prev_snapshot_rtt[(src, dst)] = current_rtt_ms[i]

    # Report
    if last_value_errors:
        avg_mae = np.mean(last_value_errors)
        print(f"\n{'='*40}")
        print(f"BASELINE RESULTS (Last Value Predictor)")
        print(f"{'='*40}")
        print(f"Logic: Predict RTT[t] = RTT[t-1]")
        print(f"Baseline MAE: {avg_mae:.4f} ms")
        print(f"Your Model MAE: ~5-7 ms (Target to beat)")
        
        if avg_mae < 5.0:
            print("WARNING: Baseline is very strong. Your model might not be beating it.")
        else:
            print("Good! Your model is likely learning something useful beyond simple persistence.")
    else:
        print("Could not calculate baseline (no recurring edges found?)")

if __name__ == "__main__":
    calculate_baselines()
