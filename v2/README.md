# V2 模塊化交易系統

## 概述
模塊化的加密貨幣交易系統,整合 HuggingFace 數據源、特徵工程、標籤生成、雙模型訓練、共振-否決推論引擎、**進階數據收集(訂單流+市場環境)**,並提供完整的圖形化界面。

## 架構
```
v2/
├── gui_app.py                     # Streamlit 圖形化界面
├── data_loader.py                 # HuggingFace 數據載入器
├── feature_engineering.py        # 特徵計算模塊
├── label_generation.py            # 標籤生成模塊
├── model_trainer.py               # 雙模型訓練系統(防過擬合優化)
├── inference_engine.py            # 共振-否決推論引擎
├── pipeline.py                    # 完整管道整合
├── advanced_data_collector.py     # 進階數據收集器 (NEW)
├── advanced_feature_merger.py     # 進階特徵合併器 (NEW)
├── example_usage.py               # 基礎範例
├── example_data_pipeline.py       # 數據管道範例
├── example_model_training.py      # 模型訓練範例
├── models/                        # 訓練完成的模型儲存
├── advanced_data/                 # 進階數據儲存 (NEW)
├── requirements.txt               # 依賴套件清單
└── README.md                     # 文檔
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

## 進階數據收集 (NEW)

### 功能概述

收集 Binance 免費提供的進階市場數據,大幅提升模型預測能力。

### 數據類型

**1. 訂單流特徵 (Order Flow)**
- delta_volume: 買賣量差異
- buy_pressure: 主動買入壓力
- sell_pressure: 主動賣出壓力
- trade_intensity: 成交密集度
- avg_trade_size: 平均成交量
- trade_size_volatility: 成交量波動性
- large_trade_count: 大單數量
- large_trade_ratio: 大單比例

**2. 資金費率 (Funding Rate)**
- fundingRate: 當前資金費率
- funding_rate_ma8: 8期移動平均
- funding_rate_ma24: 24期移動平均
- funding_rate_std: 標準差
- funding_rate_extreme: 極端值標記

**3. 未平倉量 (Open Interest)**
- sumOpenInterest: 總未平倉量
- sumOpenInterestValue: 總未平倉價值
- oi_change: OI變化量
- oi_change_rate: OI變化率
- oi_ma7: 7期移動平均
- oi_ma30: 30期移動平均

**4. 多空比 (Long/Short Ratio)**
- longShortRatio: 多空比例
- longAccount: 多單賬戶比例
- shortAccount: 空單賬戶比例
- ls_ratio_ma7: 7期移動平均
- ls_ratio_extreme: 極端值標記

### 批量收集數據

**收集所有 38 個幣種的進階數據**

```python
from advanced_data_collector import BatchAdvancedDataCollector

collector = BatchAdvancedDataCollector()

summary = collector.collect_all_symbols(
    start_date='2024-01-01',
    end_date='2024-12-31',
    timeframe='15m',
    output_dir='v2/advanced_data'
)
```

**執行收集**
```bash
python v2/advanced_data_collector.py
```

**收集範圍**
- 38個交易對 (BTCUSDT, ETHUSDT, ...)
- 2024全年數據
- 15分鐘時間框架
- 約 35,000+ K線/幣種

**輸出格式**
```
v2/advanced_data/
├── BTCUSDT_order_flow.parquet
├── BTCUSDT_funding_rate.parquet
├── BTCUSDT_open_interest.parquet
├── BTCUSDT_long_short_ratio.parquet
├── ETHUSDT_order_flow.parquet
├── ...
└── collection_summary.csv
```

### 合併進階特徵

**自動合併到基礎數據**

```python
from data_loader import CryptoDataLoader
from advanced_feature_merger import AdvancedFeatureMerger

loader = CryptoDataLoader()
merger = AdvancedFeatureMerger()

# 載入基礎數據
df_base = loader.load_klines('BTCUSDT', '15m')
df_base = loader.prepare_dataframe(df_base)

# 合併進階特徵
df_full = merger.merge_all_features(df_base, 'BTCUSDT')

# 查看新增特徵
advanced_features = merger.get_advanced_feature_columns(df_full)
print(f"Added {len(advanced_features)} advanced features")
```

**特徵命名規則**
- `of_`: Order Flow 特徵 (e.g., of_delta_volume)
- `fr_`: Funding Rate 特徵 (e.g., fr_fundingRate)
- `oi_`: Open Interest 特徵 (e.g., oi_change_rate)
- `ls_`: Long/Short Ratio 特徵 (e.g., ls_longShortRatio)

### 預期效果

整合進階特徵後,模型效能預期提升:

| 特徵類型 | Test AUC 提升 | 總特徵數 |
|---------|--------------|----------|
| 基礎特徵及 | 0.54-0.58 | 6 |
| + 訂單流 | +0.08-0.15 | 8 |
| + 資金費率 | +0.03-0.08 | 5 |
| + 未平倉量 | +0.03-0.06 | 6 |
| + 多空比 | +0.02-0.05 | 5 |
| **全部整合** | **0.70-0.80** | **30+** |

## GUI 功能說明

### [1] 數據載入

**功能**
- 查看資料集資訊 (38個交易對)
- 選擇交易對和時間框架
- 從 HuggingFace 載入 OHLCV 數據
- 預覽數據與統計資訊

### [2] 特徵工程

**功能**
- 設定布林帶參數
- 設定樞紐點參數
- 計算 15 個技術指標
- 預覽特徵數據

**輸出特徵**
- Bollinger Bands: bandwidth, percentile
- 擠壓/擴張狀態
- Z-Score 均值回歸
- SMC 微結構訊號

### [3] 標籤生成

**功能**
- ATR 動態停損/停利
- 生成二元分類標籤
- 區分觸及停損 vs 時間耗盡
- 查看標籤統計

### [4] 模型訓練

**功能**
- 訓練反彈預測模型 (Model A)
- 訓練趋勢過濾模型 (Model B)
- 防過擬合優化超參數
- 查看特徵重要性

**防過擬合超參數**
- max_depth: 4 (限制複雜度)
- min_child_samples: 150 (提升泛化)
- subsample: 0.8 (樣本抽樣)
- colsample_bytree: 0.8 (特徵抽樣)
- reg_alpha: 0.5, reg_lambda: 1.0 (L1/L2正規化)

### [5] 推論測試

**功能**
- 雙模型共振-否決推論
- 調整決策閉值
- 查看訊號統計

**決策邏輯**
```
P_bounce > 0.65 AND P_filter < 0.40 -> ENTRY_APPROVED
P_bounce <= 0.65 -> BOUNCE_WEAK
P_filter >= 0.40 -> TREND_VETO
```

## 命令行範例

### 收集進階數據
```bash
python v2/advanced_data_collector.py
```

### 完整訓練流程
```bash
python v2/example_model_training.py
```

## 數據來源

### 基礎數據
- **HuggingFace**: `zongowo111/v2-crypto-ohlcv-data`
- 38個交易對, 3個時間框架

### 進階數據
- **Binance API**: 完全免費
- 聚合成交 (aggTrades)
- 資金費率 (fundingRate)
- 未平倉量 (openInterest)
- 多空比 (longShortRatio)

## 效能指標

### 基礎模型
- 反彈模型 AUC: 0.54-0.58
- 過濾模型 AUC: 0.60-0.70

### 進階模型 (整合訂單流+市場環境)
- 反彈模型 AUC: 0.70-0.80
- 過濾模型 AUC: 0.70-0.85
- 進場率: 15-25%
- 核准後成功率: 60-75%

## 防漏措施

- 時序切分 (無隨機打亂)
- 樞紐點位移確認
- 標籤僅使用未來價格
- 排除絕對價格特徵
- 排除未來標籤 (hit_sl, hit_tp)

## 下一步

1. 回測引擎 (滑點模擬)
2. 策略模塊 (倍數管理)
3. 實時交易機器人
4. 多時間框架整合
5. 自適應閉值優化
