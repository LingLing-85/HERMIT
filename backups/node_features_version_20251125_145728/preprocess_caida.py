"""
Preprocessing script for CAIDA dataset
Converts CSV files to PyTorch format required by HMPTGN
"""
import os
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def load_csv_files(data_dir, verbose=True):
    """
    Load all CSV files from the CAIDA dataset directory
    
    Args:
        data_dir: Path to directory containing CSV files (e.g., 20150101.csv)
        verbose: Whether to print progress
    
    Returns:
        List of (date, dataframe) tuples sorted by date
    """
    import re
    
    # Get all CSV files
    all_csv_files = sorted(Path(data_dir).glob('*.csv'))
    
    # Filter to only include files with date format (YYYYMMDD.csv)
    # This excludes mapping.csv, caida_select_10.csv, etc.
    date_pattern = re.compile(r'^\d{8}\.csv$')
    csv_files = [f for f in all_csv_files if date_pattern.match(f.name)]
    
    print(f"Found {len(csv_files)} date-formatted CSV files to process")
    
    if len(csv_files) == 0:
        raise ValueError(f"No date-formatted CSV files found in {data_dir}")
    
    if verbose:
        print(f"Found {len(csv_files)} date-formatted CSV files")
        if len(all_csv_files) > len(csv_files):
            excluded = [f.name for f in all_csv_files if f not in csv_files]
            print(f"Excluded {len(excluded)} non-date files: {excluded[:5]}")
    
    data_list = []
    skipped_files = []
    
    # Feature accumulation dictionaries
    # node_id -> list of avg_rtt values
    node_rtt_avg = {}
    # node_id -> list of std_rtt values
    node_rtt_std = {}
    
    for csv_file in tqdm(csv_files, desc="Loading CSV files", disable=not verbose):
        date = csv_file.stem  # e.g., '20150101'
        try:
            # Try reading with header first
            df = pd.read_csv(csv_file)
            
            # Check if it has the expected columns
            if 'source' not in df.columns or 'target' not in df.columns:
                # Try reading without header
                df = pd.read_csv(csv_file, header=None, names=['source', 'target', 'weight', 'avg_rtt', 'std_rtt'])
            
            # Ensure required columns exist
            required_cols = ['source', 'target']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {csv_file}: Missing columns. Found {df.columns}")
                skipped_files.append(csv_file.name)
                continue
                
            # Extract features if available
            has_features = 'avg_rtt' in df.columns and 'std_rtt' in df.columns
            
            if has_features:
                # Accumulate features for source nodes
                for src, rtt, std in zip(df['source'], df['avg_rtt'], df['std_rtt']):
                    if src not in node_rtt_avg:
                        node_rtt_avg[src] = []
                        node_rtt_std[src] = []
                    node_rtt_avg[src].append(rtt)
                    node_rtt_std[src].append(std)
                
                # Accumulate features for target nodes (assuming undirected/symmetric properties for node features)
                for tgt, rtt, std in zip(df['target'], df['avg_rtt'], df['std_rtt']):
                    if tgt not in node_rtt_avg:
                        node_rtt_avg[tgt] = []
                        node_rtt_std[tgt] = []
                    node_rtt_avg[tgt].append(rtt)
                    node_rtt_std[tgt].append(std)

            # Filter out rows where source or target cannot be converted to int
            df = df[pd.to_numeric(df['source'], errors='coerce').notna()]
            df = df[pd.to_numeric(df['target'], errors='coerce').notna()]
            
            # Convert to int
            df['source'] = df['source'].astype(int)
            df['target'] = df['target'].astype(int)

            data_list.append((date, df))
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            skipped_files.append(csv_file.name)
            
    if verbose and skipped_files:
        print(f"Skipped {len(skipped_files)} files due to errors: {skipped_files[:5]}")
        
    return data_list, node_rtt_avg, node_rtt_std


def process_caida_data(data_dir, output_dir):
    """
    Process CAIDA data and save as .pt file
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_list, node_rtt_avg, node_rtt_std = load_csv_files(data_dir)
    
    if not data_list:
        raise ValueError("No valid data loaded")
        
    print(f"Processing {len(data_list)} time snapshots...")
    
    # Get all unique nodes and map to IDs
    all_nodes = set()
    for _, df in data_list:
        all_nodes.update(df['source'].unique())
        all_nodes.update(df['target'].unique())
        
    print(f"Total unique nodes: {len(all_nodes)}")
    
    # Create mapping
    node_to_id = {int(node): idx for idx, node in enumerate(sorted(all_nodes))}
    
    # Process edges
    edge_index_list = []
    
    for _, df in tqdm(data_list, desc="Processing snapshots"):
        # Map nodes to IDs
        src = df['source'].map(node_to_id).values
        dst = df['target'].map(node_to_id).values
        
        # Create edge index [2, num_edges]
        edge_index = np.stack([src, dst], axis=0)
        
        # Convert to tensor
        edge_index_tensor = torch.from_numpy(edge_index).long()
        edge_index_list.append(edge_index_tensor)
        
    # Process Features
    print("Processing node features...")
    num_nodes = len(all_nodes)
    features = torch.zeros((num_nodes, 2))
    
    for node_id, idx in node_to_id.items():
        if node_id in node_rtt_avg and node_rtt_avg[node_id]:
            avg_val = np.mean(node_rtt_avg[node_id])
            std_val = np.mean(node_rtt_std[node_id])
            features[idx, 0] = avg_val
            features[idx, 1] = std_val
    
    # Apply log transformation to handle skewed distribution
    # Add 1 to avoid log(0), then apply log
    features = torch.log1p(features)
    
    # Clip outliers (values beyond 3 std from mean)
    for col in range(features.shape[1]):
        col_data = features[:, col]
        mean = col_data.mean()
        std = col_data.std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        features[:, col] = torch.clamp(col_data, lower_bound, upper_bound)
    
    # Normalize features (Z-score normalization)
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    features = (features - mean) / (std + 1e-6)
    
    # Save processed data
    dataset_name = 'caida'
    save_dir = output_dir / dataset_name
    save_dir.mkdir(exist_ok=True)
    
    torch.save(edge_index_list, save_dir / f'{dataset_name}.pt')
    torch.save(features, save_dir / 'features.pt')
    
    print(f"✓ Saved preprocessed data to: {save_dir / f'{dataset_name}.pt'}")
    print(f"✓ Saved features to: {save_dir / 'features.pt'}")
    print(f"✓ Number of time snapshots: {len(edge_index_list)}")
    print(f"✓ Number of nodes: {len(all_nodes)}")
    
    avg_edges = sum(e.shape[1] for e in edge_index_list) / len(edge_index_list)
    print(f"✓ Average edges per snapshot: {avg_edges:.1f}")
    
    # Save node mapping for reference
    with open(save_dir / 'node_mapping.txt', 'w') as f:
        for node, idx in sorted(node_to_id.items(), key=lambda item: item[1]):
            f.write(f"{node}\t{idx}\n")
            
    print(f"✓ Saved node mapping to: {save_dir / 'node_mapping.txt'}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess CAIDA dataset for HMPTGN')
    parser.add_argument('--data_dir', type=str, 
                        default='/mnt/NewSSD/workshop2025/caida_dataset_rtt/caida_dataset_select',
                        help='Path to directory containing CSV files')
    parser.add_argument('--output_dir', type=str, 
                        default='../data/input/processed',
                        help='Path to output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CAIDA Dataset Preprocessing for HMPTGN")
    print("="*60)
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    process_caida_data(args.data_dir, args.output_dir)
    
    print("\n" + "="*60)
    print("Preprocessing completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
