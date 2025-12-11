import torch
import sys

def check_features():
    try:
        features = torch.load('../data/input/processed/caida/features.pt')
        print(f"Feature shape: {features.shape}")
        print(f"Min: {features.min().item()}")
        print(f"Max: {features.max().item()}")
        print(f"Mean: {features.mean().item()}")
        print(f"Std: {features.std().item()}")
        
        # Check for NaNs or Infs in the input features themselves
        if torch.isnan(features).any():
            print("❌ Features contain NaN values!")
        if torch.isinf(features).any():
            print("❌ Features contain Inf values!")
            
        # Check distribution
        print("Distribution (first 10 values of col 0):", features[:10, 0])
        print("Distribution (first 10 values of col 1):", features[:10, 1])
        
    except Exception as e:
        print(f"Error loading features: {e}")

if __name__ == "__main__":
    check_features()
