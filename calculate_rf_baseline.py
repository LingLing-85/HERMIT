
import torch
import numpy as np
import sys
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add script directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'script'))

# CRITICAL: Set sys.argv BEFORE importing config
sys.argv = ['calculate_rf_baseline.py', '--dataset', 'caida']

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
    
    # Load Node Features
    feature_path = os.path.join(ROOT_DIR, 'data/input/processed/caida/features.pt')
    node_features = torch.load(feature_path).numpy()
    
    return data, stats, node_features

def denormalize_rtt(normalized_rtt, avg_min, avg_max):
    """Denormalize RTT"""
    log_rtt = normalized_rtt * (avg_max - avg_min) + avg_min
    rtt_ms = np.expm1(log_rtt)
    return rtt_ms

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    epsilon = 1e-10
    y_true_safe = np.maximum(y_true, epsilon)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    return mae, rmse, mape

def main():
    # 1. Setup
    np.random.seed(42)
    data, stats, node_features = load_caida_data()
    
    total_snapshots = data['time_length']
    test_length = 60
    train_length = total_snapshots - test_length

    # 2. Prepare Training Data
    # Random Forest Input: [Feature_Src, Feature_Dst]
    # Random Forest Output: Normalized RTT
    
    print(f"Preparing Training Data (sampling from snapshots 0 to {train_length})...")
    X_train = []
    y_train = []
    
    # Sampling strategy: Collect edges from training snapshots
    # To avoid OOM and waiting forever, we limit training samples
    MAX_TRAIN_SAMPLES = 500000 
    
    # Iterate through training snapshots
    # We stride to get temporal diversity
    stride = max(1, train_length // 100) 
    
    collected_count = 0
    snapshot_indices = list(range(0, train_length, stride))
    
    for t in tqdm(snapshot_indices):
        edge_index = data['edge_index_list'][t]
        weights = data['weights'][t]
        if weights is None: continue
        
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()
        targets = weights[:, 0].numpy() # Normalized RTT
        
        # Features
        src_feats = node_features[src_nodes]
        dst_feats = node_features[dst_nodes]
        
        # Concatenate [Src, Dst] or [Src, Dst, Src*Dst, |Src-Dst|]?
        # Let's keep it simple: [Src, Dst] (concatenated features)
        # Assuming node_features is [N, D]
        # X is [E, 2*D]
        batch_X = np.hstack([src_feats, dst_feats])
        
        X_train.append(batch_X)
        y_train.append(targets)
        
        collected_count += len(targets)
        if collected_count >= MAX_TRAIN_SAMPLES:
            break
            
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    # Subsample if needed
    if len(y_train) > MAX_TRAIN_SAMPLES:
        indices = np.random.choice(len(y_train), MAX_TRAIN_SAMPLES, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
    print(f"Training Random Forest on {len(y_train)} samples with feature dim {X_train.shape[1]}...")
    
    # Train RF
    # n_jobs=-1 uses all processors
    rf = RandomForestRegressor(n_estimators=50, max_depth=20, n_jobs=-1, random_state=42, verbose=1)
    rf.fit(X_train, y_train)
    print("Training Complete.")

    # 3. Predict on Test Data
    print(f"Evaluating on Test Set (Snapshots {train_length} to {total_snapshots})...")
    
    metrics = {
        'Global': {'true': [], 'pred': []},
        'Existing': {'true': [], 'pred': []},
        'New': {'true': [], 'pred': []}
    }
    
    # Track seen edges for New/Existing split
    seen_edges = set()
    # Build history first
    print("Building history set...")
    for t in range(train_length):
        edge_index = data['edge_index_list'][t]
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()
        for i in range(len(src_nodes)):
            if src_nodes[i] != dst_nodes[i]:
                src, dst = min(src_nodes[i], dst_nodes[i]), max(src_nodes[i], dst_nodes[i])
                seen_edges.add((src, dst))
                
    # Test Loop
    for t in tqdm(range(train_length, total_snapshots)):
        edge_index = data['edge_index_list'][t]
        weights = data['weights'][t]
        if weights is None: continue
        
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()
        targets_norm = weights[:, 0].numpy()
        
        src_feats = node_features[src_nodes]
        dst_feats = node_features[dst_nodes]
        X_test = np.hstack([src_feats, dst_feats])
        
        # Predict
        pred_norm = rf.predict(X_test)
        
        # Denormalize
        pred_ms = denormalize_rtt(pred_norm, stats['avg_min'], stats['avg_max'])
        true_ms = denormalize_rtt(targets_norm, stats['avg_min'], stats['avg_max'])
        
        # Categorize
        for i in range(len(true_ms)):
            src, dst = min(src_nodes[i], dst_nodes[i]), max(src_nodes[i], dst_nodes[i])
            edge_key = (src, dst)
            
            metrics['Global']['true'].append(true_ms[i])
            metrics['Global']['pred'].append(pred_ms[i])
            
            if edge_key in seen_edges:
                metrics['Existing']['true'].append(true_ms[i])
                metrics['Existing']['pred'].append(pred_ms[i])
            else:
                metrics['New']['true'].append(true_ms[i])
                metrics['New']['pred'].append(pred_ms[i])
                seen_edges.add(edge_key)

    # 4. Report
    print("\n" + "="*60)
    print("RANDOM FOREST BASELINE REPORT")
    print("="*60)
    print(f"{'Category':<15} | {'Count':<10} | {'MAE (ms)':<10} | {'MAPE (%)':<10} | {'RMSE (s)':<10}")
    print("-" * 70)
    for cat in ['Global', 'Existing', 'New']:
        y_true = np.array(metrics[cat]['true'])
        y_pred = np.array(metrics[cat]['pred'])
        count = len(y_true)
        if count > 0:
            mae, rmse_ms, mape = calculate_metrics(y_true, y_pred)
            rmse_s = rmse_ms / 1000.0
            print(f"{cat:<15} | {count:<10} | {mae:<10.4f} | {mape:<10.4f} | {rmse_s:<10.5f}")
        else:
             print(f"{cat:<15} | {0:<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
    print("-" * 70)
    print("="*60)

if __name__ == "__main__":
    main()
