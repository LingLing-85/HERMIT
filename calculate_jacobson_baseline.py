
import torch
import numpy as np
import sys
import os
from tqdm import tqdm

# Add script directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'script'))

# CRITICAL: Set sys.argv BEFORE importing config
sys.argv = ['calculate_jacobson_baseline.py', '--dataset', 'caida']

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

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # MAPE calculation (add epsilon to avoid division by zero)
    epsilon = 1e-10
    y_true_safe = np.maximum(y_true, epsilon)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    return mae, rmse, mape

def main():
    # 1. Setup
    np.random.seed(42)
    data, stats = load_caida_data()
    
    # Jacobson's Algorithm Parameters
    # Standard TCP uses alpha = 1/8 = 0.125
    ALPHA = 0.125
    
    # State: Dictionary to store SRTT for each edge
    # Key: (src, dst), Value: Current SRTT
    srtt_map = {}
    
    total_snapshots = data['time_length']
    test_length = 60
    train_length = total_snapshots - test_length

    # Metric Containers
    metrics = {
        'Existing': {'true': [], 'pred': []}, # Edges where we have an SRTT history
        'New': {'true': [], 'pred': []}      # Edges we see for the first time (Jacobson fails here)
    }

    print(f"Running Jacobson Baseline (alpha={ALPHA})...")

    # 2. Iterate through ALL snapshots (Training + Test)
    # Because Jacobson needs to "warm up" during training to have predictions for test
    start_eval_idx = train_length
    
    for t in tqdm(range(total_snapshots)):
        # Load current snapshot data
        weights = data['weights'][t]
        if weights is None: continue
        
        edge_index = data['edge_index_list'][t]
        
        # Get ground truth for this snapshot
        current_rtt_norm = weights[:, 0].numpy()
        current_rtt_ms = denormalize_rtt(current_rtt_norm, stats['avg_min'], stats['avg_max'])
        
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()
        
        # Process each edge
        for i in range(len(current_rtt_ms)):
            src, dst = min(src_nodes[i], dst_nodes[i]), max(src_nodes[i], dst_nodes[i])
            edge_key = (src, dst)
            rtt_sample = current_rtt_ms[i]
            
            # If we are in the TEST phase, record prediction error
            if t >= start_eval_idx:
                if edge_key in srtt_map:
                    # We have history -> Predict SRTT
                    pred_val = srtt_map[edge_key]
                    metrics['Existing']['true'].append(rtt_sample)
                    metrics['Existing']['pred'].append(pred_val)
                else:
                    # No history -> "New Link". Jacobson cannot predict.
                    # We record it just to show count, but error is technically undefined (or use simple heuristic)
                    metrics['New']['true'].append(rtt_sample)
                    # For New Links, Jacobson has NO prediction.
                    # Usually implemented as taking the first sample as SRTT.
                    # So error is N/A for this specific instant.
                    pass 

            # Update Step (Jacobson's Algorithm)
            if edge_key in srtt_map:
                # SRTT = (1 - alpha) * SRTT + alpha * sample
                srtt_map[edge_key] = (1 - ALPHA) * srtt_map[edge_key] + ALPHA * rtt_sample
            else:
                # First measurement becomes SRTT
                srtt_map[edge_key] = rtt_sample

    # 3. Report Results
    print("\n" + "="*60)
    print("JACOBSON BASELINE REPORT (alpha=0.125)")
    print("="*60)
    
    # Calculate Metrics for Existing Links
    y_true = np.array(metrics['Existing']['true'])
    y_pred = np.array(metrics['Existing']['pred'])
    
    if len(y_true) > 0:
        mae, rmse_ms, mape = calculate_metrics(y_true, y_pred)
        rmse_s = rmse_ms / 1000.0
        
        print(f"{'Category':<15} | {'Count':<10} | {'MAPE (%)':<10} | {'RMSE (s)':<10}")
        print("-" * 55)
        print(f"{'Existing':<15} | {len(y_true):<10} | {mape:<10.4f} | {rmse_s:<10.5f}")
        print("-" * 55)
    
    # Report New Links
    new_count = len(metrics['New']['true'])
    print(f"\nNew Links Count: {new_count}")
    print("Metric for New Links: N/A (Jacobson requires at least 1 historical sample)")
    print("="*60)

if __name__ == "__main__":
    main()
