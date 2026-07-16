
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
sys.argv = ['experiment_rf_comparison.py', '--dataset', 'caida']

from script.config import args
from script.utils.util import set_random
from script.utils.data_util import loader
from script.models.load_model import load_model
from script.inits import prepare
from script.loss import RTTRegressionHead
from script.hgcn.manifolds import PoincareBall, Hyperboloid

# Ensure root directory is in sys.path
ROOT_DIR = '/home/ling/ling/HMPTGN_edge_feature'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# CONFIGURATION
ANOMALY_THRESHOLD_MS = 200.0
MAX_TRAIN_SAMPLES = 500000

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
    stats = torch.load(stats_file, weights_only=False)
    return data, stats

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

def calculate_rigid_training_stats(data, train_length, stats):
    """
    Calculate node RTT statistics using ONLY training snapshots (0 to train_length-1).
    This avoids data leakage from the test set.
    """
    import pandas as pd
    print(f"Calculating Rigorous Training Stats (Snapshots 0-{train_length})...")
    
    # Collect all (node, rtt) pairs using lists initially but convert to numpy/pandas for aggregation
    # to avoid slow python dict loops
    all_node_indices = []
    all_rtt_values = []
    
    # Iterate through training snapshots
    for t in tqdm(range(train_length)):
        weights = data['weights'][t]
        edge_index = data['edge_index_list'][t]
        
        if weights is None: continue
        
        # weights[:, 0] is normalized Log RTT
        rtts = weights[:, 0].numpy()
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()
        
        # Concatenate src and dst nodes, and repeat rtt values
        # e.g. edge (u, v) with rtt R -> u has R, v has R
        current_nodes = np.concatenate([src_nodes, dst_nodes])
        current_rtts = np.concatenate([rtts, rtts])
        
        all_node_indices.append(current_nodes)
        all_rtt_values.append(current_rtts)
            
    # Combine all
    all_node_indices = np.concatenate(all_node_indices)
    all_rtt_values = np.concatenate(all_rtt_values)
    
    # Use pandas for fast groupby aggregation
    df = pd.DataFrame({'node': all_node_indices, 'rtt': all_rtt_values})
    node_stats = df.groupby('node')['rtt'].agg(['mean', 'std']).reset_index()
    
    # Compute global stats for cold start
    global_mean = all_rtt_values.mean()
    global_std = all_rtt_values.std()
    
    print(f"Global Training Mean (Norm): {global_mean:.4f}, Std: {global_std:.4f}")
    
    num_nodes = data['num_nodes']
    # Initialize with global stats
    rigid_features = np.zeros((num_nodes, 2), dtype=np.float32)
    rigid_features[:, 0] = global_mean
    rigid_features[:, 1] = global_std
    
    # Fill specific stats
    # node_stats has columns [node, mean, std]
    # We can use numpy indexing if node IDs are continuous integers
    nodes = node_stats['node'].values.astype(int)
    means = node_stats['mean'].values
    stds = node_stats['std'].fillna(0).values # std might be NaN if only 1 value
    
    # Safety filter for valid node IDs
    valid_mask = nodes < num_nodes
    nodes = nodes[valid_mask]
    means = means[valid_mask]
    stds = stds[valid_mask]
    
    rigid_features[nodes, 0] = means
    rigid_features[nodes, 1] = stds
            
    return rigid_features

def main():
    # 1. Setup
    np.random.seed(42)
    torch.manual_seed(42)
    
    data, stats = load_caida_data()
    
    # 2. Load Pre-trained HMPTGN Model
    print("Loading Best HMPTGN Model...")
    args.num_nodes = data['num_nodes']
    args.num_nodes = data['num_nodes']
    
    # Calculate Training Set Length
    total_snapshots = data['time_length']
    test_length = 145 # config.py: args.testlength
    val_length = 70   # config.py: args.vallength
    train_length = total_snapshots - test_length - val_length
    print(f"Split: Train={train_length}, Val={val_length}, Test={test_length}")

    # Use Rigorous Statistics (Training Set Only)
    # feature_path = os.path.join(ROOT_DIR, 'data/input/processed/caida/features.pt')
    # x_static = torch.load(feature_path).float().to(args.device)
    
    rigid_features_np = calculate_rigid_training_stats(data, train_length, stats)
    x_static = torch.from_numpy(rigid_features_np).float().to(args.device)
    
    # Only override nfeat if NOT using trainable features
    if not args.trainable_feat:
        args.nfeat = x_static.size(1)
    else:
        print(f"Using Trainable Features: nfeat={args.nfeat} (Static features used for RF only)")

    model = load_model(args).to(args.device)
    
    # Load weights
    model_path = os.path.join(ROOT_DIR, 'data/output/log/caida/HERMIT/best_model.pth')
    if not os.path.exists(model_path):
        # Fallback to legacy HMPTGN path
        model_path = os.path.join(ROOT_DIR, 'data/output/log/caida/HMPTGN/best_model.pth')
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return

    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Load RTT Head (MLP Decoder)
    print("Loading HMPTGN-MLP (RTT Head)...")
    manifold = PoincareBall() # Default for HMPTGN
    rtt_head = RTTRegressionHead(manifold, input_dim=args.nout, hidden_dim=args.rtt_hidden_dim, c=args.curvature).to(args.device)
    
    if 'loss_state_dict' in checkpoint:
        # We need to extract the 'rtt_head' part from the loss_state_dict
        # The keys in loss_state_dict are lik 'rtt_head.mlp.0.weight'
        # But our local rtt_head expects 'mlp.0.weight'
        loss_state_dict = checkpoint['loss_state_dict']
        rtt_head_state = {}
        found_weights = False
        for k, v in loss_state_dict.items():
            if k.startswith('rtt_head.'):
                rtt_head_state[k[len('rtt_head.'):]] = v
                found_weights = True
        
        if found_weights:
            rtt_head.load_state_dict(rtt_head_state)
            print("RTT Head weights loaded successfully.")
        else:
             print("Warning: RTT Head weights NOT found in loss_state_dict!")
    else:
        print("Warning: loss_state_dict NOT found in checkpoint!")
        
    rtt_head.eval()
    
    # Update for embedding generation loop
    # We need embeddings for everything up to end of test
    # But strictly speaking we only need 0..Train for RF training
    # And Test_Start..Test_End for Evaluation
    # But due to GRU, we must run sequentially from 0
    total_run_length = total_snapshots 

    
    # 3. Generate Embeddings (Forward Pass)
    print("Generating HMPTGN Embeddings for all snapshots...")
    embeddings_store = [] # [t] -> tensor(N, dim)
    
    with torch.no_grad():
        for t in tqdm(range(total_snapshots)): # Process all snapshots to keep GRU state correct
            # Use prepare() to handle graph setup exactly like training
            # Only need edge_index and x for embedding generation
            edge_index, _, _, _, _, _, _, _ = prepare(data, t)
            weights = data['weights'][t]
            
            # Forward pass
            # Forward pass
            # If using trainable features, pass None to x so model uses self.feat
            model_input_x = None if args.trainable_feat else x_static
            z = model(edge_index, model_input_x, weights.to(args.device) if weights is not None else None)
            
            # Store embeddings on CPU to save GPU memory
            embeddings_store.append(z.cpu())
            
            # Update hiddens for next step (Crucial for GRU)
            model.update_hiddens_all_with(z)

    # 4. Prepare Training Data for RF
    print(f"\nPreparing RF Training Data (sampling max {MAX_TRAIN_SAMPLES} edges)...")
    
    X_train_raw = [] # For Euclidean RF
    X_train_tan = [] # For Hyperbolic RF
    X_train_pure = [] # For Pure RF (Stats Only)
    y_train = []
    
    collected_count = 0
    stride = max(1, train_length // 100) # Sample ~100 snapshots
    snapshot_indices = list(range(0, train_length, stride))
    
    # Pre-calculate curvature for Log Map
    # HMPTGN uses model.c[2] for the output embeddings (from GRU)
    c_out = model.c[2]
    
    for t in tqdm(snapshot_indices):
        edge_index = data['edge_index_list'][t]
        weights = data['weights'][t]
        if weights is None: continue
        
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()
        targets = weights[:, 0].numpy() # Normalized RTT
        
        # Get embeddings for this snapshot
        z_t = embeddings_store[t].to(args.device) # Move back to GPU for math if needed, or keep CPU
        
        # Compute Tangent Space Embeddings (Hyperbolic RF inputs)
        # We process all nodes at once to be efficient
        z_tan_t = model.toTangentX(z_t, c_out).cpu()
        z_t = z_t.cpu() # Keep raw on CPU
        
        # Features
        src_raw = z_t[src_nodes].numpy()
        dst_raw = z_t[dst_nodes].numpy()
        
        src_tan = z_tan_t[src_nodes].numpy()
        dst_tan = z_tan_t[dst_nodes].numpy()
        
        # Concatenate [Src_Raw_Feat, Dst_Raw_Feat, Src_Emb, Dst_Emb]
        # X_static is on GPU, need to move relevant parts to CPU
        src_feats = x_static[src_nodes].cpu().numpy()
        dst_feats = x_static[dst_nodes].cpu().numpy()

        batch_X_raw = np.hstack([src_feats, dst_feats, src_raw, dst_raw])
        batch_X_tan = np.hstack([src_feats, dst_feats, src_tan, dst_tan])
        batch_X_pure = np.hstack([src_feats, dst_feats])
        
        X_train_raw.append(batch_X_raw)
        X_train_tan.append(batch_X_tan)
        X_train_pure.append(batch_X_pure)
        y_train.append(targets)
        
        collected_count += len(targets)
        if collected_count >= MAX_TRAIN_SAMPLES:
            break
            
    X_train_raw = np.vstack(X_train_raw)
    X_train_tan = np.vstack(X_train_tan)
    X_train_pure = np.vstack(X_train_pure)
    y_train = np.concatenate(y_train)
    
    # Subsample if necessary
    if len(y_train) > MAX_TRAIN_SAMPLES:
        indices = np.random.choice(len(y_train), MAX_TRAIN_SAMPLES, replace=False)
        X_train_raw = X_train_raw[indices]
        X_train_tan = X_train_tan[indices]
        X_train_pure = X_train_pure[indices]
        y_train = y_train[indices]

    print(f"Training Data Shape: {X_train_raw.shape}")

    # 5. Train Random Forests
    print("\nTraining Euclidean RF (Baseline)...")
    # Using stronger RF for high-dim input + ensuring features are seen
    # RF Hyperparameters: 120 estimators, max depth 30, max features 0.8
    rf_euc = RandomForestRegressor(n_estimators=120, max_depth=30, max_features=0.8, n_jobs=-1, random_state=42, verbose=1)
    rf_euc.fit(X_train_raw, y_train)
    
    print("\nTraining Hyperbolic RF (Tangent Space)...")
    rf_hyp = RandomForestRegressor(n_estimators=120, max_depth=30, max_features=0.8, n_jobs=-1, random_state=42, verbose=1)
    rf_hyp.fit(X_train_tan, y_train)

    print("\nTraining Pure RF (Stats Only)...")
    rf_pure = RandomForestRegressor(n_estimators=50, max_depth=20, n_jobs=-1, random_state=42, verbose=1)
    rf_pure.fit(X_train_pure, y_train)
    
    print("\nTraining Complete.")

    # 6. Evaluate
    print(f"\nEvaluating on Test Snapshots ({train_length} -> {total_snapshots})...")
    
    metrics = {
        'Global': {'euc': [], 'hyp': [], 'conc': [], 'mlp': [], 'pure': []},
        'Existing': {'euc': [], 'hyp': [], 'conc': [], 'mlp': [], 'pure': []},
        'New': {'euc': [], 'hyp': [], 'conc': [], 'mlp': [], 'pure': []},
        'Anomalous': {'euc': [], 'hyp': [], 'conc': [], 'mlp': [], 'pure': []}
    }
    
    # Build history
    seen_edges = set()
    for t in range(train_length):
        edge_index = data['edge_index_list'][t]
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()
        for i in range(len(src_nodes)):
            if src_nodes[i] != dst_nodes[i]:
                src, dst = min(src_nodes[i], dst_nodes[i]), max(src_nodes[i], dst_nodes[i])
                seen_edges.add((src, dst))
                
    # Evaluator loop
    # Evaluate on TEST set only (skip Validation)
    test_start_index = train_length + val_length
    print(f"\nEvaluating on Test Snapshots ({test_start_index} -> {total_snapshots})...")
    
    for t in tqdm(range(test_start_index, total_snapshots)):
        edge_index = data['edge_index_list'][t]
        weights = data['weights'][t]
        if weights is None: continue
        
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()
        targets_norm = weights[:, 0].numpy()
        
        # Prepare Features
        z_t = embeddings_store[t].to(args.device)
        # HMPTGN-MLP Prediction (z_t is still on GPU here)
        with torch.no_grad():
             # edge_index is on CPU, but that's fine for embedding lookup on GPU table
             pred_norm_mlp = rtt_head(z_t, edge_index.to(args.device)).cpu().numpy()

        z_tan_t = model.toTangentX(z_t, c_out).cpu()
        z_t = z_t.cpu()
        
        src_raw = z_t[src_nodes].numpy()
        dst_raw = z_t[dst_nodes].numpy()
        src_feats = x_static[src_nodes].cpu().numpy()
        dst_feats = x_static[dst_nodes].cpu().numpy()

        X_test_raw = np.hstack([src_feats, dst_feats, src_raw, dst_raw])
        
        src_tan = z_tan_t[src_nodes].numpy()
        dst_tan = z_tan_t[dst_nodes].numpy()
        X_test_tan = np.hstack([src_feats, dst_feats, src_tan, dst_tan])
        
        X_test_pure = np.hstack([src_feats, dst_feats]) # Pure Stats
        
        # Predict
        pred_norm_euc = rf_euc.predict(X_test_raw)
        pred_norm_hyp = rf_hyp.predict(X_test_tan)
        pred_norm_pure = rf_pure.predict(X_test_pure)

        # Denormalize
        pred_ms_euc = denormalize_rtt(pred_norm_euc, stats['avg_min'], stats['avg_max'])
        pred_ms_hyp = denormalize_rtt(pred_norm_hyp, stats['avg_min'], stats['avg_max'])
        pred_ms_pure = denormalize_rtt(pred_norm_pure, stats['avg_min'], stats['avg_max'])
        pred_ms_mlp = denormalize_rtt(pred_norm_mlp, stats['avg_min'], stats['avg_max'])
        true_ms = denormalize_rtt(targets_norm, stats['avg_min'], stats['avg_max'])
        
        for i in range(len(true_ms)):
            src, dst = min(src_nodes[i], dst_nodes[i]), max(src_nodes[i], dst_nodes[i])
            edge_key = (src, dst)
            val_euc = pred_ms_euc[i]
            val_hyp = pred_ms_hyp[i]
            val_pure = pred_ms_pure[i]
            val_mlp = pred_ms_mlp[i]
            val_true = true_ms[i]
            
            # Global
            metrics['Global']['euc'].append((val_true, val_euc))
            metrics['Global']['hyp'].append((val_true, val_hyp))
            metrics['Global']['conc'].append((val_true, val_euc))
            metrics['Global']['pure'].append((val_true, val_pure))
            metrics['Global']['mlp'].append((val_true, val_mlp))
            
            # Existing vs New
            if edge_key in seen_edges:
                metrics['Existing']['euc'].append((val_true, val_euc))
                metrics['Existing']['hyp'].append((val_true, val_hyp))
                metrics['Existing']['conc'].append((val_true, val_euc))
                metrics['Existing']['pure'].append((val_true, val_pure))
                metrics['Existing']['mlp'].append((val_true, val_mlp))
            else:
                metrics['New']['euc'].append((val_true, val_euc))
                metrics['New']['hyp'].append((val_true, val_hyp))
                metrics['New']['conc'].append((val_true, val_euc))
                metrics['New']['pure'].append((val_true, val_pure))
                metrics['New']['mlp'].append((val_true, val_mlp))
                seen_edges.add(edge_key)
                
            # Anomalous
            if val_true > ANOMALY_THRESHOLD_MS:
                metrics['Anomalous']['euc'].append((val_true, val_euc))
                metrics['Anomalous']['hyp'].append((val_true, val_hyp))
                metrics['Anomalous']['conc'].append((val_true, val_euc))
                metrics['Anomalous']['pure'].append((val_true, val_pure))
                metrics['Anomalous']['mlp'].append((val_true, val_mlp))

    # 7. Print Report
    print("\n" + "="*80)
    print("COMPARISON REPORT: HMPTGN-MLP vs HMPTGN-RF (Concatenated) vs Pure RF")
    print(f"Models: MLP (Native), Concatenated RF (Euc), Pure RF (Stats Only)")
    print(f"Anomalous Threshold: > {ANOMALY_THRESHOLD_MS} ms")
    print("="*80)
    
    print(f"{'Category':<15} | {'Model':<20} | {'Count':<8} | {'MAE (ms)':<10} | {'RMSE (ms)':<10} | {'MAPE (%)':<10}")
    print("-" * 100)
    
    categories = ['Global', 'Existing', 'New', 'Anomalous']
    
    for cat in categories:
        for model_key, label in [('mlp', 'HMPTGN-MLP'), ('conc', 'HMPTGN-RF (Concat)'), ('pure', 'Pure RF (Baseline)')]:
            pairs = metrics[cat][model_key]
            if not pairs:
                print(f"{cat:<15} | {label:<15} | {0:<8} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
                continue
                
            y_true = np.array([p[0] for p in pairs])
            y_pred = np.array([p[1] for p in pairs])
            
            mae, rmse, mape = calculate_metrics(y_true, y_pred)
            
            print(f"{cat:<15} | {label:<15} | {len(y_true):<8} | {mae:<10.4f} | {rmse:<10.4f} | {mape:<10.2f}")
        print("-" * 80)
        
if __name__ == "__main__":
    main()
