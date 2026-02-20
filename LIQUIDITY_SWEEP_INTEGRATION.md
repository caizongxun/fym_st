# 流動性掃蕩系統整合指南

## 已完成的整合

### 1. 核心組件

#### LiquiditySweepDetector
```python
from trading_system.core import LiquiditySweepDetector

detector = LiquiditySweepDetector(
    lookback_period=50,
    wick_multiplier=2.0,
    oi_std_threshold=2.0
)
```

#### 擴充的 CryptoDataLoader
```python
from trading_system.core import CryptoDataLoader

loader = CryptoDataLoader()
df = loader.fetch_latest_klines(
    symbol='BTCUSDT',
    timeframe='1h',
    days=90,
    include_oi=True,          # 新增: OI 數據
    include_funding=True      # 新增: 資金費率
)
```

#### 擴充的 FeatureEngineer
```python
from trading_system.core import FeatureEngineer

fe = FeatureEngineer()
df_features = fe.build_features(
    df,
    include_liquidity_features=True  # 啟用流動性特徵
)
```

---

## 使用方式

### 方式 1: 獨立使用流動性掃蕩偵測

```python
from trading_system.core import CryptoDataLoader, LiquiditySweepDetector

# 1. 載入數據
loader = CryptoDataLoader()
df = loader.fetch_latest_klines(
    'BTCUSDT', '1h', days=90,
    include_oi=True, include_funding=True
)

# 2. 偵測流動性掃蕩
detector = LiquiditySweepDetector()
df_sweep = detector.detect_liquidity_sweep(df, direction='lower')
df_sweep = detector.calculate_sweep_features(df_sweep)

# 3. 篩選信號
signals = df_sweep[df_sweep['sweep_lower_signal']]
print(f"偵測到 {len(signals)} 個流動性掃蕩事件")
```

### 方式 2: 整合到訓練流程

```python
from trading_system.core import (
    CryptoDataLoader, FeatureEngineer,
    TripleBarrierLabeling, ModelTrainer,
    LiquiditySweepDetector
)

# 1. 載入數據 (含 OI)
loader = CryptoDataLoader()
df = loader.fetch_latest_klines(
    'BTCUSDT', '1h', days=365,
    include_oi=True, include_funding=True
)

# 2. 建立特徵 (含流動性特徵)
fe = FeatureEngineer()
df_features = fe.build_features(df, include_liquidity_features=True)

# 3. 偵測流動性掃蕩
detector = LiquiditySweepDetector()
df_sweep = detector.detect_liquidity_sweep(df_features, direction='lower')
df_sweep = detector.calculate_sweep_features(df_sweep)

# 4. 標籤
from trading_system.core import TripleBarrierLabeling
labeling = TripleBarrierLabeling(tp=3.0, sl=1.0)
df_labeled = labeling.label(df_sweep)

# 5. 增強掃蕩事件權重
df_labeled['sample_weight'] = 1.0
df_labeled.loc[df_labeled['sweep_lower_signal'], 'sample_weight'] = 3.0

# 6. 訓練模型
trainer = ModelTrainer()

# 選擇特徵 (包含流動性特徵)
features = [
    # 價格特徵
    'atr_pct', 'rsi_normalized', 'bb_position', 'bb_width_pct',
    
    # EMA 特徵
    'ema_9_dist', 'ema_21_dist', 'ema_9_21_ratio',
    
    # 流動性掃蕩特徵 (新增)
    'lower_wick_ratio', 'upper_wick_ratio',
    'oi_change_pct', 'oi_change_4h', 'oi_normalized',
    'cvd_slope_5', 'cvd_normalized',
    'dist_to_support_pct',
    
    # 其他
    'volume_ratio', 'volatility_20'
]

trainer.train(
    df_labeled,
    features=features,
    label='label',
    sample_weight='sample_weight'
)
```

### 方式 3: 回測時過濾流動性掃蕩事件

```python
from trading_system.core import (
    CryptoDataLoader, FeatureEngineer,
    ModelTrainer, Backtester,
    LiquiditySweepDetector
)

# 1. 載入數據
loader = CryptoDataLoader()
df = loader.fetch_latest_klines(
    'BTCUSDT', '1h', days=90,
    include_oi=True, include_funding=True
)

# 2. 建立特徵
fe = FeatureEngineer()
df_features = fe.build_features(df, include_liquidity_features=True)

# 3. 偵測流動性掃蕩
detector = LiquiditySweepDetector()
df_sweep = detector.detect_liquidity_sweep(df_features, direction='lower')
df_sweep = detector.calculate_sweep_features(df_sweep)

# 4. 預測
trainer = ModelTrainer()
trainer.load_model('model.pkl')
probabilities = trainer.predict_proba(df_sweep[features])
df_sweep['win_probability'] = probabilities

# 5. 篩選: 同時符合高機率和流動性掃蕩
signals = df_sweep[
    (df_sweep['win_probability'] > 0.65) &
    (df_sweep['sweep_lower_signal'] == True)
]

print(f"篩選後信號: {len(signals)}")

# 6. 回測
backtester = Backtester(
    initial_capital=10000,
    taker_fee=0.0006,
    maker_fee=0.0002,
    slippage=0.0005,
    risk_per_trade=0.01,
    leverage=10
)

results = backtester.run_backtest(
    signals,
    tp_multiplier=3.5,
    sl_multiplier=1.5
)

print(f"回測結果: {results['statistics']}")
```

---

## 快速測試

```bash
# 運行測試腳本
python test_liquidity_sweep.py
```

會輸出:
```
流動性掃蕩系統測試
===============================================================================
[1] 載入 BTCUSDT 1h 數據 (30天, 包含 OI & Funding Rate)...
   載入 720 筆 K 線
   OI 有效數據: 720/720 (100.0%)
   Funding Rate 有效數據: 720/720 (100.0%)

[2] 偵測做多信號 (掃蕩低點)...
   Liquidity Sweep (lower): 3/720 (0.4%)

[4] 結果分析
===============================================================================
築選漏斗:
   長下影線:       45/720 (6.2%)
   突破支撐位:     32/720 (4.4%)
   OI 銳減:         12/720 (1.7%)
   CVD 背離:        28/720 (3.9%)
   最終信號:       3/720 (0.4%)
```

---

## 新特徵列表

### 價格結構
- `lower_wick_ratio`: 下影線/實體比例
- `upper_wick_ratio`: 上影線/實體比例
- `dist_to_support_pct`: 距離支撐位%
- `dist_to_resistance_pct`: 距離壓力位%

### OI 相關
- `oi_change_pct`: OI 單根變化率
- `oi_change_1h`, `oi_change_4h`, `oi_change_24h`
- `oi_normalized`: OI 標準化值
- `open_interest`: 未平倉量

### CVD 相關
- `cvd`: 累計成交量差
- `cvd_slope_5`, `cvd_slope_10`: CVD 斜率
- `cvd_normalized`: CVD 標準化值

### 資金費率
- `funding_rate`: 8h 資金費率
- `funding_rate_ma_3`, `funding_rate_ma_7`

---

## 調參建議

### 保守模式 (高精度)
```python
detector = LiquiditySweepDetector(
    lookback_period=50,
    wick_multiplier=2.5,      # 更長影線
    oi_std_threshold=2.5,     # 更大 OI 銳減
    cvd_divergence_lookback=5
)
```
預期: 90天 2-5 個信號

### 激進模式 (高召回)
```python
detector = LiquiditySweepDetector(
    lookback_period=30,
    wick_multiplier=1.5,      # 較短影線
    oi_std_threshold=1.5,     # 較小 OI 變化
    cvd_divergence_lookback=15
)
```
預期: 90天 8-15 個信號

---

## 常見問題

**Q: OI 數據獲取失敗?**

A: Binance API 有時會限流,建議:
- 使用 1h 或 4h 時間框架
- 減少請求頻率
- 緩存數據到本地

**Q: 為什麼信號很少?**

A: 流動性掃蕩是稀有事件:
- 30天可能只有 1-2 個
- 建議使用 90 天以上
- 調低 `wick_multiplier` 和 `oi_std_threshold`

**Q: 如何驗證效果?**

A:
1. 獨立測試: `python test_liquidity_sweep.py`
2. 回測對比: 比較有/無流動性過濾的績效
3. 實盤小量測試

---

## 相關文檔

- 完整理論: `docs/LIQUIDITY_SWEEP_THEORY.md`
- 測試腳本: `test_liquidity_sweep.py`
- API 文檔: `trading_system/core/liquidity_sweep_detector.py`

---

## 授權

MIT License

---

**免責聲明**: 本系統僅供教育和研究用途。加密貨幣交易存在高風險,請謹慎使用。