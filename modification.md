# HMPTGN Codebase Modifications

This document tracks all modifications made to the original HMPTGN codebase to support the CAIDA dataset and improve training stability.

## 1. Data Preprocessing (`script/utils/preprocess_caida.py`)

**Status**: [NEW FILE]
**Purpose**: Convert CAIDA CSV files to PyTorch Geometric format.

**Key Features**:
- Loads CSV files with columns `source,target,time,weight,avg_rtt,std_rtt`
- Maps node IDs to consecutive integers (0 to N-1)
- **[CRITICAL FIX]** Filters input files to only include date-formatted files (e.g., `20150101.csv`), excluding metadata files like `mapping.csv` and `caida_select_10.csv` which caused data corruption.
- Saves processed data as `caida.pt` (list of edge indices) and `node_mapping.txt`.

## 2. Edge Generation (`script/utils/make_edges_new.py`)

**Status**: [MODIFIED]
**Purpose**: Generate positive/negative edges for training and testing.

**Modifications**:
1. **Numpy Compatibility**:
   - Replaced `np.long` (deprecated) with `np.int64`.
   ```python
   # Before
   edges_pos = np.vstack(np.divmod(perm, num_nodes)).transpose().astype(np.long)
   # After
   edges_pos = np.vstack(np.divmod(perm, num_nodes)).transpose().astype(np.int64)
   ```

2. **[CRITICAL FIX] Shape Handling**:
   - Fixed a bug where `get_edges` incorrectly transposed the input edge index `[2, N]`, causing the model to see only 2 edges per snapshot instead of ~180k.
   ```python
   # Before: Always transpose
   edge_index, _ = remove_self_loops(torch.from_numpy(np.array(edge_index_list[i])).transpose(1, 0))
   
   # After: Check shape first
   data = torch.from_numpy(np.array(edge_index_list[i]))
   if data.shape[0] != 2 and data.shape[1] == 2:
       data = data.transpose(1, 0)
   edge_index, _ = remove_self_loops(data)
   ```

## 3. Configuration (`script/config.py`)

**Status**: [MODIFIED]
**Purpose**: Define hyperparameters and dataset settings.

**Modifications**:
1. **Dataset Config**:
   - Added `caida` dataset support.
   - Set `testlength = 3` (last 3 snapshots for testing).
   - Set `trainable_feat = 1` (use trainable node embeddings).

2. **Stability Settings**:
   - **Learning Rate**: Reduced from `0.002` -> `0.001` -> `0.0001` (to fix NaN loss).
   - **Epsilon**: Increased from `1e-15` -> `1e-8` (to prevent division by zero in hyperbolic math).

## 4. Main Training Loop (`script/main.py`)

**Status**: [MODIFIED]
**Purpose**: Main training execution.

**Modifications**:
1. **Gradient Clipping**:
   - Added gradient clipping to prevent exploding gradients, a common cause of NaN loss in hyperbolic models.
   ```python
   epoch_loss.backward()
   torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Added
   optimizer.step()
   ```

## 5. Data Loader (`script/utils/data_util.py`)

**Status**: [MODIFIED]
**Purpose**: Load datasets.

**Modifications**:
- Added `caida` to the list of datasets loaded via `load_new_dataset`.

---

## Current Issues

Despite these fixes, the model still encounters **NaN Loss** during the first epoch. This suggests:
1. **Hyperbolic Space Instability**: The Poincare ball model is numerically sensitive.
2. **Initialization**: Initial embeddings might be too large or close to the boundary.
3. **Data Scale**: The CAIDA dataset (280k nodes) is much larger than original datasets (Enron: ~100 nodes), which might require different hyperparameters.
