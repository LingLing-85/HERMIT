## 1. Overview

This is the Pytorch implementation for "Toward a Manifold-Preserving Temporal Graph Network in Hyperbolic Space (IJCAI24)"

Authors: Quan Le, Cuong Ta

Paper: [Toward a Manifold-Preserving Temporal Graph Network in Hyperbolic Space](https://www.ijcai.org/proceedings/2024/484)

![Framework of HMPTGN](figures/HMPTGN_framework.png)

## 2. Setup

### 2.1 Environment
`pip install -r requirements.txt`

### 2.2 Datasets
The data is cached in `./data/input/cached`.

## 3. Experiments
3.0 Go to the script at first

```cd ./script```

3.1 To run HMPTGN:

```python main.py --model=HMPTGN --dataset=enron10 --lr=0.002 --seed=998877```

3.1.1 To run HMPTGN on caida:
```cd script && nohup python -u main.py --dataset caida --lr 0.00005 --max_epoch 100 > ../data/output/log/caida/HMPTGN/training_formal_$(date +%Y%m%d_%H%M%S).log 2>&1 &```


3.2 Seed: 998877, 23456, 900.

3.3 Dataset choices: disease, enron10, dblp, uci, mathoverflow, fbw.

## 4. Baselines
For the baselines, please follow these repos and papers:
- [HGWaveNet](https://github.com/TaiLvYuanLiang/HGWaveNet)
- [HTGN](https://github.com/marlin-codes/HTGN)
- [VGRNN](https://github.com/VGraphRNN/VGRNN)
- [EvolveGCN](https://github.com/IBM/EvolveGCN)
- [DySAT](https://github.com/FeiGSSS/DySAT_pytorch)
- [DHGAT](https://doi.org/10.1016/j.neucom.2023.127038)

執行指令
```python script/main.py --dataset caida --lr 0.1 --max_epoch 10```
