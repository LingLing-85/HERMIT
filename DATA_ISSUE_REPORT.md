# CAIDA 資料預處理問題診斷報告

## 🚨 嚴重問題發現

### 問題描述
預處理後的 CAIDA 資料集存在嚴重的資料量問題：

- **所有 1457 個時間步**平均只有 **2.7 條邊**
- 邊數範圍：2-4 條
- 這導致模型無法正常訓練

### 症狀
1. **AUC=1.0, AP=1.0**：測試集太小（只有 1-2 個樣本），隨機猜測都能達到完美分數
2. **NaN Loss**：訓練資料不足，模型無法學習
3. **訓練崩潰**：第 3 個 epoch 就出現 NaN

### 預期 vs 實際

| 指標 | 預期值 | 實際值 |
|------|--------|--------|
| 每個時間步的邊數 | 數千到數萬 | 2-4 |
| 總邊數 | 數百萬 | ~4000 |
| 節點數 | 277,071 | 277,071 ✓ |

節點數是正確的，但邊數嚴重不足！

## 可能原因

### 1. CSV 檔案本身就很稀疏
- 原始資料可能已經過濾或採樣
- 需要檢查原始 CSV 檔案的行數

### 2. 資料過濾太嚴格
預處理腳本中的過濾邏輯：
```python
# 過濾掉無法轉換為整數的行
df = df[pd.to_numeric(df['source'], errors='coerce').notna()]
df = df[pd.to_numeric(df['target'], errors='coerce').notna()]
```

可能過濾掉了太多資料。

### 3. CSV 格式問題
- 可能 CSV 檔案有多個 header 行
- 可能資料格式與預期不符

## 診斷步驟

請執行以下命令來診斷問題：

### 1. 檢查 CSV 檔案數量和大小
```bash
ls -lh /mnt/NewSSD/workshop2025/caida_dataset_rtt/caida_dataset_select/ | head -20
```

### 2. 檢查 CSV 檔案行數
```bash
wc -l /mnt/NewSSD/workshop2025/caida_dataset_rtt/caida_dataset_select/*.csv | head -20
```

### 3. 查看第一個 CSV 檔案的內容
```bash
head -30 /mnt/NewSSD/workshop2025/caida_dataset_rtt/caida_dataset_select/$(ls /mnt/NewSSD/workshop2025/caida_dataset_rtt/caida_dataset_select/ | head -1)
```

### 4. 檢查是否有資料被過濾掉
在預處理腳本中添加 debug 輸出：
```python
print(f"Before filtering: {len(df)} rows")
df = df[pd.to_numeric(df['source'], errors='coerce').notna()]
df = df[pd.to_numeric(df['target'], errors='coerce').notna()]
print(f"After filtering: {len(df)} rows")
```

## 下一步行動

1. **提供原始資料資訊**：執行上述診斷命令
2. **檢查資料格式**：確認 CSV 檔案格式是否正確
3. **修復預處理腳本**：根據診斷結果調整過濾邏輯
4. **重新預處理**：使用修復後的腳本重新處理資料
5. **驗證結果**：確保每個時間步有合理數量的邊

## 臨時解決方案

如果無法修復資料問題，可以考慮：
- 使用論文中提供的其他資料集（enron10, dblp, uci 等）
- 尋找其他 CAIDA 資料集來源
- 使用合成資料進行測試
