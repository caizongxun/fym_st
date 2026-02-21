# V2 模塊化交易系統

## 概述
模塊化的加密貨幣交易系統,整合 HuggingFace 數據源、特徵工程、標籤生成、雙模型訓練與共振-否決推論引擎,並提供完整的圖形化界面。

## 架構
```
v2/
├── gui_app.py                # Streamlit 圖形化界面
├── data_loader.py             # HuggingFace 數據載入器
├── feature_engineering.py    # 特徵計算模塊
├── label_generation.py        # 標籤生成模塊
├── model_trainer.py           # 雙模型訓練系統
├── inference_engine.py        # 共振-否決推論引擎
├── pipeline.py                # 完整管道整合
├── example_usage.py           # 基礎範例
├── example_data_pipeline.py   # 數據管道範例
├── example_model_training.py  # 模型訓練範例
├── models/                    # 訓練完成的模型儲存
├── requirements.txt           # 依賴套件清單
└── README.md                 # 文檔
```

## 快速開始

### 安裝依賴

```bash
cd v2
pip install -r requirements.txt
```

### 啟動 GUI 應用程式

```bash
streamlit run gui_app.py
```

瀏覽器會自動開啟 `http://localhost:8501`

## GUI 功能說明

### 📊 數據載入

**功能**
- 查看資料集資訊 (38個交易對)
- 選擇交易對和時間框架
- 從 HuggingFace 載入 OHLCV 數據
- 預覽數據與統計資訊

**操作步驟**
1. 在左側查看資料集資訊
2. 選擇交易對 (例: BTCUSDT)
3. 選擇時間框架 (例: 15m)
4. 點擊「載入數據」
5. 查看數據預覽與統計

### 🔧 特徵工程

**功能**
- 設定布林帶參數
- 設定樞紐點參數
- 計算 15 個技術指標
- 預覽特徵數據

**參數設定**
- 布林帶週期: 5-50 (預設 20)
- 標準差倍數: 1.0-3.0 (預設 2.0)
- 回溯週期: 50-200 (預設 100)
- 樞紐左側K線: 1-10 (預設 3)
- 樞紐右側K線: 1-10 (預設 3)

**輸出特徵**
- Bollinger Bands: basis, upper, lower, bandwidth, percentile
- 擠壓/擴張狀態: is_squeeze, is_expansion
- 均值回歸: z_score
- SMC 訊號: pivot points, sweeps, BOS

### 🎯 標籤生成

**功能**
- 設定 ATR 參數
- 設定停損/停利倍數
- 生成二元分類標籤
- 查看標籤統計

**參數設定**
- ATR 週期: 5-30 (預設 14)
- 停損 ATR 倍數: 0.5-3.0 (預設 1.5)
- 停利 ATR 倍數: 1.0-5.0 (預設 3.0)
- 前瞥 K 線數: 5-50 (預設 16)

**統計資訊**
- 做多/做空樣本總數
- 成功/失敗次數
- 成功率百分比

### 🧠 模型訓練

**功能**
- 訓練反彈預測模型 (Model A)
- 訓練趋勢過濾模型 (Model B)
- 查看訓練結果
- 查看特徵重要性

**訓練參數**
- 方向: long / short
- 樹數量: 100-1000 (預設 500)
- 學習率: 0.01-0.2 (預設 0.05)
- 最大深度: 3-15 (預設 7)
- 訓練集比例: 0.5-0.9 (預設 0.8)

**輸出結果**
- 訓練/測試 ROC-AUC
- 樣本數量
- Top 5 特徵重要性

**模型儲存**
- 反彈模型: `v2/models/bounce_{direction}_model.pkl`
- 過濾模型: `v2/models/filter_{direction}_model.pkl`

### 🚀 推論測試

**功能**
- 載入訓練完成的模型
- 設定決策閉值
- 執行雙模型推論
- 查看共振-否決結果

**閉值設定**
- 反彈閉值: 0.0-1.0 (預設 0.65)
- 過濾閉值: 0.0-1.0 (預設 0.40)

**決策邏輯**
```
P_bounce > 0.65 AND P_filter < 0.40 → ENTRY_APPROVED
P_bounce <= 0.65 → BOUNCE_WEAK
P_filter >= 0.40 → TREND_VETO
```

**統計資訊**
- 總樣本數
- 核准進場數
- 進場率 %
- 核准後成功率 %
- 平均 P_bounce / P_filter
- 訊號原因分佈

## 命令行範例

### 基礎特徵與標籤
```bash
python example_usage.py
```

### 完整數據管道
```bash
python example_data_pipeline.py
```

### 模型訓練與推論
```bash
python example_model_training.py
```

## 數據載入器

### CryptoDataLoader

```python
from data_loader import CryptoDataLoader

loader = CryptoDataLoader()
df = loader.load_klines('BTCUSDT', '15m')
```

**資料集**
- Repository: `zongowo111/v2-crypto-ohlcv-data`
- 38 個交易對
- 3 個時間框架 (15m, 1h, 1d)

## 特徵工程

### FeatureEngineer

```python
from feature_engineering import FeatureEngineer

fe = FeatureEngineer(bb_period=20, lookback=100)
df_features = fe.process_features(df)
```

**15 個特徵**
- Bollinger Bands (7)
- Mean Reversion (1)
- SMC Signals (7)

## 標籤生成

### LabelGenerator

```python
from label_generation import LabelGenerator

lg = LabelGenerator(atr_period=14, sl_atr_mult=1.5, tp_atr_mult=3.0)
df_labeled = lg.generate_labels(df_features)
```

**標籤邏輯**
- Label = 1: TP 先觸及
- Label = 0: SL 先觸及或超時

## 模型訓練

### ModelTrainer & TrendFilterTrainer

```python
from model_trainer import ModelTrainer, TrendFilterTrainer

# Model A
trainer_a = ModelTrainer(model_type='bounce')
results_a = trainer_a.train(df_train)
trainer_a.save_model('models/bounce_model.pkl')

# Model B
trainer_b = TrendFilterTrainer()
results_b = trainer_b.train(df_train)
trainer_b.save_model('models/filter_model.pkl')
```

## 推論引擎

### InferenceEngine

```python
from inference_engine import InferenceEngine

engine = InferenceEngine(
    bounce_model_path='models/bounce_model.pkl',
    filter_model_path='models/filter_model.pkl',
    bounce_threshold=0.65,
    filter_threshold=0.40
)

df_predictions = engine.predict_batch(df_test)
stats = engine.get_statistics(df_predictions)
```

**輸出**
- `p_bounce`: 反彈機率
- `p_filter`: 過濾機率
- `signal`: 0 或 1
- `reason`: ENTRY_APPROVED / BOUNCE_WEAK / TREND_VETO

## 效能指標

### 模型 A (反彈)
- 目標 ROC-AUC: > 0.70
- 精確度優先

### 模型 B (過濾)
- 目標 ROC-AUC: > 0.65
- 召回率優先

### 推論引擎
- 進場率: 15-25%
- 成功率: 55-70%
- 風險降低: 30-40%

## 防漏措施

- 時序切分 (無隨機打亂)
- 樞紐點位移確認
- 標籤僅使用未來價格
- 特徵排除 OHLC 與時間戳

## 下一步

1. 回測引擎 (滑點模擬)
2. 策略模塊 (倍數管理)
3. 實時交易機器人
4. 多時間框架整合
5. 自適應閉值優化
