# HMPTGN Data Pipeline & Design Rationale

這份文件詳細整理了 HMPTGN 模型從資料輸入到訓練的完整流程，並解釋了關鍵的設計理念與技術實作細節（特別是記憶體效率部分）。

## 1. 資料流程總覽 (Data Pipeline Overview)

整個流程可以分為四個階段：預處理、載入、準備、模型運算。

### Step 1: 資料預處理 (Preprocessing)
*   **腳本**: `script/utils/preprocess_caida.py`
*   **輸入**: 原始 CSV 檔案 (包含 `source`, `target`, `avg_rtt`, `std_rtt`, `weight`)。
*   **動作**:
    1.  讀取每個時間點 (Snapshot) 的連線數據。
    2.  **特徵提取**: 提取 `avg_rtt`, `std_rtt`, `weight`。
    3.  **正規化**: 對 RTT 進行 Log 轉換與 Min-Max Normalization (縮放到 [0, 1])。
    4.  **打包**: 將邊特徵堆疊成形狀為 `[E, 3]` 的 Tensor (E 為邊數)。
*   **輸出**:
    *   `caida.pt`: 儲存邊的連接關係 (Adjacency List)。
    *   `caida_weights.pt`: 儲存邊的 3 維特徵。
    *   `features.pt`: 儲存節點特徵。

### Step 2: 資料載入 (Loading)
*   **腳本**: `script/utils/data_util.py` (`load_new_dataset` 函數)
*   **動作**:
    1.  從硬碟讀取 `.pt` 檔案。
    2.  將 `caida_weights.pt` 載入到記憶體 (RAM) 中，存放在 `data['weights']` 列表裡。
    3.  此時資料尚未進入 GPU。

### Step 3: 訓練準備 (Preparation)
*   **腳本**: `script/inits.py` (`prepare` 函數)
*   **動作**:
    1.  在訓練迴圈中，針對當前時間點 `t`。
    2.  從 `data['weights']` 取出對應的 Tensor。
    3.  **搬運**: 執行 `.to(device)`，將該時間點的數據搬移至 GPU 記憶體。

### Step 4: 模型運算 (Model Forward)
*   **腳本**: `script/models/HMPTGN.py` & `script/main.py`
*   **動作**:
    1.  模型接收 `edge_weight` (形狀 `[E, 3]`)。
    2.  透過 `edge_encoder` 將 3 維特徵壓縮為 1 維權重。
    3.  將計算出的權重用於圖卷積 (GCN) 運算。

---

## 2. 特徵工程與設計理念 (Feature Engineering & Rationale)

為什麼要這樣設計特徵？

### A. Edge Weight (邊權重)
**為什麼需要？**
在網路拓樸中，單純的「有連線(1)/沒連線(0)」不足以描述網路狀態。
*   **RTT (延遲)**: 區分高品質光纖與低品質路徑。
*   **Std (穩定度)**: 區分穩定連線與抖動嚴重的連線。
*   **Weight (頻率)**: 代表該時間區間內的連接次數，反映連線的活躍程度與重要性。

**模型如何使用？**
模型透過 `edge_encoder` (線性層) 自動學習這三個指標的權重組合：
$$ \text{Score} = w_1 \cdot \text{RTT} + w_2 \cdot \text{Std} + w_3 \cdot \text{Weight} $$
這讓模型能動態判斷什麼樣的連線才是「重要」的。

### B. Node Feature (節點特徵)
**為什麼 RTT 可以當節點特徵？**
雖然 RTT 是邊的屬性，但我們將一個節點所有連線的 RTT 取平均，作為該節點的 **「環境指標」**。
*   **低平均 RTT**: 暗示該節點可能是核心節點 (Core Router)，周邊連線品質好。
*   **高平均 RTT**: 暗示該節點可能是邊緣節點或位於網路品質較差的區域。
這提供了比單純 One-Hot Encoding 更豐富的初始資訊，幫助模型加速收斂。

---

## 3. 技術實作：為什麼不會 OOM？ (Why No OOM?)

您可能會擔心引入權重矩陣會導致記憶體不足 (Out Of Memory)，但本實作透過以下機制避免了此問題：

### 1. 稀疏矩陣儲存 (Sparse Representation)
我們**不使用**稠密矩陣 (Dense Matrix, $N \times N$) 來儲存權重。
*   **做法**: 採用 **Edge List** 格式，只儲存「實際存在的邊」。
*   **差異**:
    *   稠密矩陣: 需儲存 1 億個值 (假設 1萬個節點)。
    *   稀疏矩陣: 若只有 5 萬條邊，只需儲存 5 萬個值。
    *   **結果**: 記憶體消耗極低。

### 2. 時間切片處理 (Snapshot-based Processing)
*   **做法**: 訓練是按時間點 (Snapshot) 依序進行的。
*   **效益**: GPU 記憶體在同一時間只需要容納 **「一張圖」** 的數據。計算完當前時間點後，記憶體空間即可被釋放或重複利用於下一個時間點。

### 3. 低維度特徵 (Low Dimensionality)
*   **做法**: 邊特徵僅有 3 個維度 (`avg_rtt`, `std_rtt`, `weight`)。
*   **效益**: 相比於電腦視覺或 NLP 動輒數百數千維的特徵，這裡的額外記憶體開銷微乎其微。

### 4. `edge_encoder` 的降維作用
*   **做法**: 模型在第一層就透過 `edge_encoder` 將 `[E, 3]` 的特徵壓縮為 `[E, 1]` 的純量權重。
*   **效益**: 後續複雜的 GCN 運算 (矩陣乘法等) 都是基於 1 維權重進行，運算複雜度與原始無特徵版本幾乎相同。
