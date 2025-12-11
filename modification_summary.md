# HMPTGN Model Modifications for Multi-Channel Edge Features

This document details the changes made to the HMPTGN codebase to support 3-channel edge features (`rtt_avg`, `rtt_std`, `weight`) instead of a single scalar weight.

## 1. Objective
The goal was to allow the model to utilize multiple edge attributes simultaneously:
- **RTT Average**: Average round-trip time (latency).
- **RTT Standard Deviation**: Jitter/variability of latency.
- **Link Weight**: Topological weight (if available).

Instead of manually combining these into a single scalar during preprocessing (which loses information), we modified the model to accept all three and *learn* the optimal combination.

## 2. Preprocessing Changes (`script/utils/preprocess_caida.py`)

We updated the preprocessing logic to extract, normalize, and stack these features.

### Key Changes:
1.  **Global Statistics Calculation**:
    - We now compute the global minimum and maximum for `avg_rtt` and `std_rtt` across *all* snapshots before processing.
    - This ensures consistent normalization across time.

2.  **Log-Normalization**:
    - Applied `log1p` (log(x+1)) to RTT and Std features to handle their long-tail distribution (large range of values).
    - Normalized to `[0, 1]` using the global min/max.

3.  **Feature Stacking**:
    - Instead of a single scalar, we now save a **3-dimensional vector** for each edge: `[Normalized_RTT, Normalized_Std, Weight]`.
    - If `weight` is missing in the CSV, it defaults to 1.0.
    - If RTT is missing, it defaults to `[0.0, 0.0, 1.0]`.

```python
# Pseudo-code of new logic
rtt = normalize(log1p(raw_rtt))
std = normalize(log1p(raw_std))
w = raw_weight
feature_vector = [rtt, std, w]  # Shape: [Num_Edges, 3]
```

## 3. Model Architecture Changes (`script/models/HMPTGN.py`)

We modified the `HMPTGN` model to include an "adapter" layer that projects the 3-channel input down to the scalar weight expected by the core hyperbolic GCN layers.

### Key Changes:
1.  **Edge Encoder Layer**:
    - Added `self.edge_encoder = nn.Linear(3, 1)` in `__init__`.
    - This linear layer learns weights `w1, w2, w3` and bias `b` to combine the inputs:
      $$ \text{ScalarWeight} = \sigma(w_1 \cdot \text{RTT} + w_2 \cdot \text{Std} + w_3 \cdot \text{Weight} + b) $$

2.  **Forward Pass Adaptation**:
    - In `forward()`, we check if the input `weight` tensor has 3 dimensions.
    - If so, we pass it through `edge_encoder` and a `Sigmoid` activation.
    - The `Sigmoid` ensures the resulting weight is in the `(0, 1)` range, which is critical for numerical stability in the hyperbolic layers.

```python
# Code snippet from HMPTGN.py
if weight is not None and weight.dim() > 1 and weight.shape[-1] == 3:
    # [E, 3] -> [E, 1] -> [E]
    weight = self.edge_encoder(weight).squeeze(-1)
    # Sigmoid to ensure weights are in (0, 1) range
    weight = torch.sigmoid(weight)
```

## 4. Benefits
- **Learnable**: The model automatically determines the importance of latency vs. jitter vs. topology.
- **Robust**: Handles the inverse relationship (High RTT = Bad Connection) naturally. The model can learn a negative weight for RTT, effectively penalizing high-latency links.
- **Backward Compatible**: If a dataset only provides scalar weights, the model skips the encoder and works as before.
