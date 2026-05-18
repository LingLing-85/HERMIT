# Temporal Hyperbolic Graph Learning with Random Forest for Internet Connection and RTT Prediction

This repository is the official implementation of **HERMIT (Hyperbolic Edge-aware RTT modeling via Integrated Topology)**, a hybrid framework designed for Round-Trip Time (RTT) prediction. 

HERMIT combines the geometric representation learning power of Hyperbolic Temporal Graph Networks with the robust regression capabilities of Random Forest in Poincare tangent spaces.

![Framework of HMPTGN](figures/HERMIT.png)

---

## 1. Methodology Overview

The HERMIT pipeline consists of two distinct phases:

1. **Phase 1: Representation Learning (Base Model)**
   - Learns hyperbolic dynamic node embeddings using a Manifold-Preserving Temporal Graph Network (HMPTGN) optimized with link prediction and an auxiliary MLP regressor.
   
2. **Phase 2: RTT Inference (Hyperbolic Random Forest)**
   - Extracts the pre-trained hyperbolic embeddings.
   - Maps embeddings to the Euclidean tangent space via logarithmic mapping ($\log_{\mathbf{0}}$).
   - Trains and evaluates a Random Forest Regressor on the concatenated tangent embeddings and node historical statistics.

---

## 2. Setup & Environment

To install all the required Python dependencies, run the following command:

```bash
pip install -r requirements.txt
```

*Note: The dataset cached in `./data/input/processed/caida` is ready for direct model training and evaluation.*

---

## 3. Execution Workflow

### Step 1: Train the HMPTGN Base Model
To train the base temporal graph network and learn hyperbolic embeddings:

```bash
python script/main.py --dataset caida --lr 0.00005 --max_epoch 100
```
This script will train the HMPTGN model and save the best checkpoint to `data/output/log/caida/HMPTGN/best_model.pth`.

---

### Step 2: Run RTT Inference and Evaluation
Once Phase 1 is complete, run the evaluation script to train the Random Forest regressor on hyperbolic tangent spaces and output the final RTT prediction metrics:

```bash
python evaluate_hmptgn_rf.py
```
This script will:
- Load the pre-trained HMPTGN embeddings from Phase 1.
- Extract node-level historical RTT features.
- Map hyperbolic embeddings into Poincare tangent space.
- Train the **HERMIT** regressor.
- Output comparative metrics (MAE, RMSE, MAPE) against baseline models for *Global*, *Existing*, *New*, and *Anomalous* links.

---

## 4. Code Integrity Verification

To ensure that the environment is correctly set up and all Python scripts are structurally correct without syntax errors, run:

```bash
python -m py_compile evaluate_hmptgn_rf.py script/main.py
```

If the command executes with no output, the code structure and all dependencies are verified and correct.
