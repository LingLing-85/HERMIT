# Definitions: Node Features, Edge Features, AUC, and New AUC

此文件整理目前 HMPTGN 模型在 CAIDA Dataset 上的特徵定義與評估指標計算方式。

## 1. Node Feature (節點特徵)

### 定義
Node Feature 表示每個節點在特定時間點的統計特徵。在目前的實作中，這是從該節點連接的所有 Edge 的 RTT (Round-Trip Time) 統計數據聚合而來。

### 內容與資料來源
- **資料來源**: 原始 CSV 檔案 (如 `20150101.csv`) 中的 `avg_rtt` 和 `std_rtt` 欄位。
- **計算方式**:
  1. 對於每個節點 (Node)，收集其所有相連 Edge (作為 Source 或 Target) 的 `avg_rtt` 和 `std_rtt`。
  2. 計算這些 Edge 數值的 **平均值 (Mean)**。
  3. **資料前處理 (Preprocessing)**:
     - **Log Transform**: 使用 `log1p` (log(x+1)) 進行轉換。
     - **Outlier Clipping**: 將數值限制在平均值 ± 3 倍標準差範圍內。
     - **Z-Score Normalization**: 進行標準化 (減去平均，除以標準差)，使其平均值為 0，標準差為 1。
- **最終維度**: 2 (Feature Dimension = 2)。
  - Index 0: Normalized Average RTT
  - Index 1: Normalized Standard Deviation of RTT

如果資料集中沒有這些特徵，則會使用可訓練的 Embedding (Trainable Features) 或 One-hot encoding。

---

## 2. Edge Feature (邊特徵)

### 定義
Edge Feature 表示每條連線 (Link) 上的具體屬性，用於輔助模型學習網路結構與動態變化。

### 內容與資料來源
- **資料來源**: 原始 CSV 檔案中的 `avg_rtt`, `std_rtt`, 和 `weight` 欄位。
- **計算方式**:
  1. **Log Transform**: 對 `avg_rtt` 與 `std_rtt` 進行 `log1p` 轉換。
  2. **Global Min-Max Normalization**:
     - 計算所有時間快照 (Snapshots) 中 RTT 的全局最大值與最小值。
     - 將數值縮放到 `[0, 1]` 範圍: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$。
  3. **Weight**: 直接使用原始權重 (若無則預設為 1.0)。
- **最終維度**: 3 (Feature Dimension = 3)。
  - Index 0: Normalized Log Average RTT
  - Index 1: Normalized Log Standard Deviation of RTT
  - Index 2: Weight

---

## 3. AUC (Area Under the Curve)

### 定義
標準的 Link Prediction 評估指標。衡量模型區分「真實存在的邊」與「隨機不存在的邊」的能力。

### 計算範圍與邏輯
- **Positive Edges (正樣本)**: 當前測試時間點 $t$ (Snapshot $t$) 中 **實際存在** 的所有邊。
- **Negative Edges (負樣本)**: 當前時間點 $t$ 中 **不存在** 的邊 (通過隨機負採樣產生)。
- **評估方式**:
  - 模型計算 Positive Edges 與 Negative Edges 的存在機率分數。
  - 使用 `sklearn.metrics.roc_auc_score` 計算 AUC。
  - AUC = 0.5 表示隨機猜測，AUC = 1.0 表示完美預測。

---

## 4. New AUC (New Link Prediction AUC)

### 定義
專注於 **新出現的連結 (Evolving Edges)** 的預測能力。這是為了評估模型是否能預測網路的「演化」，而不僅僅是記住舊有的結構。

### 計算範圍與邏輯
- **Positive Edges (正樣本)**: 
  - 定義為 $E_{new} = E_t \setminus E_{t-1}$。
  - 即：**在時間點 $t$ 存在，但在時間點 $t-1$ 不存在** 的邊。
  - 這些代表了網路中真正「新增」的連接。
- **Negative Edges (負樣本)**:
  - 與標準 AUC 相同，從當前時間點 $t$ 中隨機採樣的不存在邊。
- **評估意義**:
  - New AUC 通常比標準 AUC 更難，因為模型無法依賴上一時刻的記憶，必須捕捉節點的動態變化趨勢。

---

### 總結比較

| 指標 | 關注對象 (Positive Samples) | 意義 |
| :--- | :--- | :--- |
| **AUC** | $G_t$ 中的所有邊 | 預測當前網路結構的整體能力 (包含舊邊與新邊) |
| **New AUC** | 在 $G_t$ 出現但不在 $G_{t-1}$ 的邊 | 預測網路 **動態演化 (Evolution)** 與新連結生成的能力 |
