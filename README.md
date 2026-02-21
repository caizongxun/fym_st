# 🎯 BB+NW 波段反轉交易系統 v2.0

**Bollinger Bands + Nadaraya-Watson Swing Reversal Trading System**

一套專為 15m 波段反轉交易設計的機構級 AI 交易系統。

---

## 🌟 系統特色

### 三層架構設計

```
觸發層 (Event Trigger)  →  特徵層 (Features)  →  AI 層 (Meta-Label)
     │                           │                        │
  BB + NW                    ADX + CVD              LightGBM
  觸碸軌道                   過濾特徵                判斷反彈
```

### 核心優勢

1. **無未來函數** (No Repaint)
   - Nadaraya-Watson 使用滾動視窗計算
   - 回測數據 = 實盤數據

2. **事件驅動抽樣**
   - 只在觸碸 BB/NW 軌道時啟動
   - 節省 85-98% 運算資源

3. **兩大防禁機制**
   - 防止單邊趨勢輾壓 (ADX + HTF EMA)
   - 辨識獵取流動性 (CVD 背離 + VWWA)

4. **單一強大模型**
   - 不需要多模型投票
   - LightGBM 自帶集成學習

---

## 🛠️ 系統架構

### 目錄結構

```
fym_st/
├── trading_system/
│   ├── core/
│   │   ├── feature_engineering.py    # 特徵工程 (含 NW, ADX, Bounce)
│   │   ├── event_filter.py           # BB/NW 觸碸過濾器
│   │   ├── data_loader.py            # 數據載入 (HF + Binance)
│   │   ├── model_trainer.py          # 模型訓練
│   │   ├── labeling.py               # Triple Barrier 標註
│   │   └── backtest_engine.py        # 回測引擎
│   │
│   ├── gui/
│   │   ├── pages/
│   │   │   ├── dashboard_page.py      # 控制台
│   │   │   ├── training_page.py       # 訓練頁面 (重新設計)
│   │   │   ├── backtesting_page.py    # 回測頁面 (重新設計)
│   │   │   ├── calibration_page.py    # 機率校準
│   │   │   └── live_prediction_page.py # 即時預測
│   │   └── __init__.py
│   │
│   └── app_main.py                  # Streamlit 主程式 (重新設計)
│
├── models/                         # 已訓練模型儲存處
├── data/                           # HuggingFace 數據庫
└── README.md                       # 本文件
```

### 核心模組

#### 1. FeatureEngineer (特徵工程)

```python
from core import FeatureEngineer

fe = FeatureEngineer()

# 建立 15m 特徵 (BB + NW + ADX + CVD)
df_15m = fe.build_features(
    df,
    include_microstructure=True,   # CVD, VWWA
    include_nw_envelope=True,       # NW 包絡線
    include_adx=True,               # ADX 趨勢強度
    include_bounce_features=False   # MTF 後再加
)

# MTF 合併
df_mtf = fe.merge_and_build_mtf_features(df_15m, df_1h)

# 加入波段反轉特徵
df_mtf = fe.add_bounce_confluence_features(df_mtf)
```

**特徵清單** (~80-100 個):
- BB 通道: `bb_middle`, `bb_upper`, `bb_lower`, `bb_width_pct`, `bb_position`
- NW 包絡線: `nw_middle`, `nw_upper`, `nw_lower`, `nw_width_pct`
- ADX 趨勢: `adx`, `plus_di`, `minus_di`
- CVD 流動性: `cvd_10`, `cvd_20`, `cvd_norm_10`, `divergence_score_10`
- VWWA: `vwwa_buy_signal`, `lower_wick_size`
- 反轉共振: `bb_pierce_lower`, `sweep_divergence_buy`, `trend_crush_risk_15m`
- MTF (1h): 所有特徵加上 `_1h` 後綴

#### 2. BBNW_BounceFilter (觸碸過濾器)

```python
from core.event_filter import BBNW_BounceFilter

filter = BBNW_BounceFilter(
    use_bb=True,                # 啟用 BB 觸發
    use_nw=True,                # 啟用 NW 觸發
    min_pierce_pct=0.001,       # 0.1% 誤差
    require_volume_surge=False  # 不強制要求爆量
)

df_filtered = filter.filter_events(df_mtf)
# 輸出: is_long_setup, is_short_setup, touch_type
```

**過濾結果**:
- 原始數據: 10,000 筆
- 過濾後: 500-1500 筆 (5-15%)
- 只保留觸碸軌道的極端事件

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

**主要依賴**:
- `streamlit` - GUI 界面
- `pandas`, `numpy` - 數據處理
- `lightgbm` - AI 模型
- `plotly` - 視覺化
- `python-binance` - Binance API
- `datasets` - HuggingFace 數據

### 2. 啟動系統

```bash
cd trading_system
streamlit run app_main.py
```

瀏覽器會自動打開: `http://localhost:8501`

### 3. 訓練第一個模型

1. **點擊左側選單**: 🧪 模型訓練

2. **配置參數**:
   - 交易對: BTCUSDT
   - 數據來源: HuggingFace (快速)
   - 只使用 2024 數據: ✅
   - NW 指標: h=8.0, mult=3.0
   - BB/NW 觸發: 全部啟用
   - TP/SL: 3.0 / 1.0
   - 最長持倉: 60 根 (15 小時)

3. **點擊 🚀 開始訓練**

4. **等待 10-15 分鐘**

### 4. 執行回測

1. **點擊左側選單**: 📊 回測分析

2. **選擇模型**: 刚才訓練的模型

3. **配置參數**:
   - 測試期間: 2024 全年 (OOS)
   - 機率門檻: 0.60
   - 初始資金: 10,000 USDT
   - 單筆仓位: 10%
   - 出場策略: 動態追蹤

4. **點擊 🚀 執行回測**

---

## 📊 效能指標

### 預期表現

| 指標 | 目標值 | 健康範圍 |
|------|----------|----------|
| 勝率 | 55-65% | 50-70% |
| 盈虧比 (R:R) | 2.5:1 | 2.0:1 - 4.0:1 |
| 盈虧因子 | 1.8+ | 1.5+ |
| 最大回撤 | < 25% | < 30% |
| 年化 ROI | 30%+ | 20%+ |
| 每月信號 | 15-30 個 | 10-40 個 |

### 關鍵特徵重要性 (Top 10)

1. `sweep_divergence_buy` - CVD 背離分數
2. `trend_crush_risk_1h` - 1h 趨勢風險
3. `bb_pierce_lower` - BB 下軌刺穿深度
4. `vwwa_buy_signal` - 下影線吸收率
5. `adx` - 趨勢強度
6. `cvd_norm_10` - 10 期標準化 CVD
7. `nw_pierce_lower` - NW 下軌刺穿深度
8. `bb_squeeze_ratio` - BB 壓縮比例
9. `ema_50_dist_1h` - 1h EMA50 距離
10. `volume_ratio` - 成交量爆量倍數

---

## 🛡️ 防禁機制詳解

### 1. 防止單邊趨勢輾壓

**問題場景**:
```
價格在主跌浪中觸碸 BB 下軌
→ 傳統策略: 做多 (預期反彈)
→ 實際: 繼續下跌被輾壓
```

**我們的解決方案**:

1. **ADX 過濾**:
   ```python
   if adx > 25 and adx_rising:
       # 走勢中，模型會輸出低機率 (< 0.30)
   ```

2. **HTF EMA 過濾**:
   ```python
   if abs(price - ema_50_1h) / ema_50_1h > 0.05:
       # 距離 1h EMA50 太遠，強趨勢
       # trend_crush_risk_1h 特徵會極高
   ```

3. **自動學習**:
   - LightGBM 會學習: 當 `adx > 30` 且 `trend_crush_risk_1h > 0.05` 時，觸碸下軌的標籤大多是 LOSS
   - 模型會自動給予低機率

### 2. 辨識獵取流動性

**問題場景**:
```
機構用長下影線刺穿下軌
→ 散戶止損被觸發
→ 機構大量接盤
→ 價格暴漲
```

**我們的解決方案**:

1. **CVD 背離偵測**:
   ```python
   # 價格下跌 5%，但 CVD 為正
   divergence_score = cvd_norm_10 - price_pct_10
   # divergence_score > 0.5 → 機構接盤
   ```

2. **VWWA 吸收率**:
   ```python
   lower_wick_ratio = lower_wick / body_size
   vwwa_buy_signal = lower_wick_ratio * volume_ratio
   # vwwa_buy_signal > 2.0 → 大量流動性被吸收
   ```

3. **組合判斷**:
   ```python
   if bb_pierce_lower > 0.005 and \
      sweep_divergence_buy > 0 and \
      vwwa_buy_signal > 2.0:
       # 完美的獵取流動性信號
       # 模型會輸出高機率 (> 0.75)
   ```

---

## 💻 程式範例

### 完整訓練流程

```python
from core import (
    CryptoDataLoader, FeatureEngineer, 
    TripleBarrierLabeling, ModelTrainer
)
from core.event_filter import BBNW_BounceFilter

# 1. 載入數據
loader = CryptoDataLoader()
df_15m = loader.load_klines('BTCUSDT', '15m')
df_1h = loader.load_klines('BTCUSDT', '1h')

# 2. 建立特徵
fe = FeatureEngineer()

df_15m_features = fe.build_features(
    df_15m,
    include_microstructure=True,
    include_nw_envelope=True,
    include_adx=True,
    include_bounce_features=False
)

df_1h_features = fe.build_features(
    df_1h,
    include_microstructure=True,
    include_nw_envelope=True,
    include_adx=True,
    include_bounce_features=False
)

# 3. MTF 合併
df_mtf = fe.merge_and_build_mtf_features(df_15m_features, df_1h_features)
df_mtf = fe.add_bounce_confluence_features(df_mtf)

# 4. 事件過濾
filter = BBNW_BounceFilter(
    use_bb=True,
    use_nw=True,
    min_pierce_pct=0.001
)
df_filtered = filter.filter_events(df_mtf)

print(f"過濾結果: {len(df_mtf)} → {len(df_filtered)} ({len(df_filtered)/len(df_mtf)*100:.1f}%)")

# 5. 標註
labeler = TripleBarrierLabeling(
    tp_multiplier=3.0,
    sl_multiplier=1.0,
    max_hold_bars=60
)
df_labeled = labeler.create_labels(df_filtered)

# 6. 訓練
trainer = ModelTrainer()
metrics = trainer.train(
    df_labeled,
    model_type='lightgbm',
    cv_folds=5,
    early_stopping_rounds=50
)

print(f"CV AUC: {metrics['cv_auc_mean']:.3f}")
print(f"CV Accuracy: {metrics['cv_accuracy_mean']:.3f}")

# 7. 儲存
trainer.save_model('BTCUSDT_15m_BB_NW_Bounce_v1.pkl')
```

### 實時預測

```python
# 載入模型
trainer = ModelTrainer()
trainer.load_model('BTCUSDT_15m_BB_NW_Bounce_v1.pkl')

# 獲取最新數據
df_latest = loader.fetch_latest_klines('BTCUSDT', '15m', days=1)

# 建立特徵 + 過濾
df_features = fe.build_features(df_latest, include_nw_envelope=True, include_adx=True)
df_filtered = filter.filter_events(df_features)

if len(df_filtered) > 0:
    # 預測
    probs = trainer.predict_proba(df_filtered)
    
    # 只保留高機率信號
    df_filtered['prob'] = probs
    signals = df_filtered[df_filtered['prob'] >= 0.60]
    
    print(f"發現 {len(signals)} 個交易信號!")
    print(signals[['open_time', 'close', 'is_long_setup', 'prob']])
else:
    print("無觸碸事件")
```

---

## ⚠️ 重要聲明

1. **風險警告**: 加密貨幣交易具有極高風險，可能導致全部資金損失
2. **無擔保**: 本系統不擔保任何盈利
3. **教育用途**: 僅供研究與學習使用
4. **先測試**: 建議先在模擬盤充分測試

---

## 📚 參考資源

### 學術論文
- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) - Marcos Lopez de Prado
- [Machine Learning for Algorithmic Trading](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715) - Stefan Jansen

### 技術文檔
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Triple Barrier Method](https://mlfinlab.readthedocs.io/en/latest/labeling/tb_meta_labeling.html)
- [Nadaraya-Watson Estimator](https://en.wikipedia.org/wiki/Kernel_regression)

### 市場數據
- [Binance API](https://binance-docs.github.io/apidocs/)
- [HuggingFace Crypto Datasets](https://huggingface.co/datasets)

---

## 🔗 聯絡資訊

- **項目位置**: [GitHub Repository](https://github.com/caizongxun/fym_st)
- **問題回報**: [Issues](https://github.com/caizongxun/fym_st/issues)

---

## 📜 授權聲明

MIT License

Copyright (c) 2026 BB+NW Swing Trading System

---

<p align="center">
  <b>BB+NW Swing Reversal System v2.0</b><br>
  Built with ❤️ for Swing Traders<br>
  <i>"Trade Smarter, Not Harder"</i>
</p>