import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for academic paper
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

# Set font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

def load_data():
    data_dir = Path('./data/input/processed/caida')
    print("Loading data...")
    weights = torch.load(data_dir / 'caida_weights.pt')
    stats = torch.load(data_dir / 'rtt_stats.pt')
    
    print("Processing a 1-million edge sample to match the exact scale of the original plot...")
    # Concatenate all snapshots, then randomly sample 1,000,000 edges
    all_rtt_norm = torch.cat([w[:, 0] for w in weights])
    
    # Set seed for reproducible exact look
    torch.manual_seed(42)
    indices = torch.randperm(len(all_rtt_norm))[:1000000]
    all_rtt_norm = all_rtt_norm[indices]
    
    # Denormalize
    avg_min = stats['avg_min']
    avg_max = stats['avg_max']
    
    rtt_log_val = all_rtt_norm * (avg_max - avg_min) + avg_min
    rtt_ms = torch.expm1(rtt_log_val)
    
    return rtt_ms, rtt_log_val  # Return tensors

def plot_distributions(rtt_ms, rtt_log_val):
    print(f"Plotting distributions using {len(rtt_ms)} points (Snapshot 0)...")
    
    if torch.is_tensor(rtt_ms):
        rtt_ms = rtt_ms.numpy()
        rtt_log_val = rtt_log_val.numpy()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot (a) Original RTT
    # Thicker bars -> fewer bins. User wanted "coarser/thicker" and "exactly 15 bars". 
    # 15 bins for 0-1500 range = 100ms per bin.
    # explicit binrange ensures bins are calculated over 0-1500, not the full data range.
    sns.histplot(rtt_ms, bins=15, binrange=(0, 1500), ax=ax1, color='#c0c0c0', edgecolor='black', kde=False)
    ax1.set_xlabel('RTT (ms)')
    ax1.set_ylabel('Count (Log Scale)')
    ax1.set_yscale('log')
    # Set ylim to strictly match "10~100000" if the data fits, or just ensuring the log scale looks similar.
    ax1.set_ylim(10, 1000000) 
    ax1.set_xlim(0, 1500)
    ax1.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400])
    ax1.set_title('')
    
    # Plot (b) Log-RTT
    sns.histplot(rtt_log_val, bins=100, ax=ax2, color='#c0c0c0', edgecolor='black', kde=True)
    ax2.set_xlabel('Log(RTT + 1)')
    ax2.set_ylabel('Count')
    ax2.set_title('')
    
    # Set main title
    fig.suptitle('RTT Distribution', fontsize=16, weight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) 
    
    # Add caption labels
    fig.text(0.25, 0.02, '(a)', ha='center', va='center', fontsize=14, weight='bold')
    fig.text(0.75, 0.02, '(b)', ha='center', va='center', fontsize=14, weight='bold')
    
    output_path = 'rtt_distribution_paper.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig('rtt_distribution_paper.pdf', bbox_inches='tight')
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    rtt_ms, rtt_log_val = load_data()
    plot_distributions(rtt_ms, rtt_log_val)
