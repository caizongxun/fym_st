# 第一階段:微觀結構特徵擴充 (Phase 1: Microstructure Feature Expansion)

## 目標

**透過微觀結構 (路線二) 將模型的阿爾法 (Alpha) 推至極限**

在 Binance 的標準 K 線數據中,已經原生包含了計算「訂單流 (Order Flow)」的最關鍵數據:
- `volume`: 總成交量
- `taker_buy_base_asset_volume`: 主動買盤成交量

我們不需要大幅修改爬蟲,只需在特徵工程模組 (FeatureEngineer) 中加入數學運算即可。

---

## 已完成的優化

### 1. 特徵精簡化

已將原有的 15+ 微觀結構特徵精簡至 **8 個核心特徵**:

#### 核心 8 特徵

1. **net_volume**: 淨主動成交量
   ```python
   net_volume = taker_buy_volume - taker_sell_volume
   ```

2. **cvd_10**: 短期 CVD (10 根 K 線)
   ```python
   cvd_10 = net_volume.rolling(10).sum()
   ```

3. **cvd_20**: 中期 CVD (20 根 K 線)
   ```python
   cvd_20 = net_volume.rolling(20).sum()
   ```

4. **cvd_norm_10**: 標準化 CVD (跨幣種可比)
   ```python
   cvd_norm_10 = cvd_10 / volume.rolling(10).sum()
   ```

5. **divergence_score_10**: **價格-CVD 背離分數** [核心特徵]
   ```python
   price_pct_10 = close.pct_change(10)
   divergence_score_10 = cvd_norm_10 - price_pct_10
   ```
   
   **邏輯**:
   - 價格下跌 (負值) + CVD 為正 (買盤強) = 極大正向背離
   - 代表底部有機構掛單吸收拋壓

6. **upper_wick_ratio**: 上影線/實體比例
   ```python
   upper_wick_ratio = upper_wick / body_size
   ```

7. **lower_wick_ratio**: 下影線/實體比例 (流動性掠奪)
   ```python
   lower_wick_ratio = lower_wick / body_size
   ```

8. **order_flow_imbalance**: 訂單流失衡比率 (-1 到 +1)
   ```python
   order_flow_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)
   ```

### 2. 其他特徵優化

- **Returns**: 20 → 10 (移除長期報酬)
- **EMA**: 移除 ema_50_dist 和 ema_21_50_ratio (冗餘)
- **Price Action**: 精簡至 3 個核心特徵

---

## 使用方式

### 方法 1: Streamlit GUI

```bash
cd trading_system
streamlit run app_main.py
```

1. 點擊 "模型訓練"
2. 選擇交易對: BTCUSDT
3. 時間框架: 1h
4. ☑️ **啟用微觀結構特徵** (include_microstructure)
5. 設定標籤參數:
   - TP: 3.0 ATR
   - SL: 1.0 ATR
6. 點擊 "開始訓練"

### 方法 2: Python 腳本

```python
from trading_system.core import (
    CryptoDataLoader, FeatureEngineer,
    TripleBarrierLabeling, ModelTrainer
)

# 1. 載入數據
loader = CryptoDataLoader()
df = loader.fetch_latest_klines('BTCUSDT', '1h', days=365)

# 2. 建立特徵 (含微觀結構)
fe = FeatureEngineer()
df_features = fe.build_features(
    df,
    include_microstructure=True  # 核心:預設開啟
)

# 3. 標籤
labeling = TripleBarrierLabeling(tp=3.0, sl=1.0)
df_labeled = labeling.label(df_features)

# 4. 選擇特徵 (包含 8 個微觀特徵)
features = [
    # 價格特徵
    'atr_pct', 'rsi_normalized', 'bb_position', 'bb_width_pct', 'vsr',
    
    # EMA 特徵
    'ema_9_dist', 'ema_21_dist', 'ema_9_21_ratio',
    
    # 微觀結構特徵 (新增)
    'net_volume',              # 1
    'cvd_10',                  # 2
    'cvd_20',                  # 3
    'cvd_norm_10',             # 4
    'divergence_score_10',     # 5 (核心)
    'upper_wick_ratio',        # 6
    'lower_wick_ratio',        # 7
    'order_flow_imbalance',    # 8
    
    # MACD
    'macd_normalized', 'macd_hist_normalized',
    
    # Returns
    'return_1', 'return_5', 'return_10',
    
    # Volume
    'volume_ratio', 'taker_buy_ratio',
    
    # Price Action
    'high_low_ratio', 'close_open_ratio', 'body_size',
    
    # Volatility
    'volatility_20', 'momentum_10'
]

# 5. 訓練模型
trainer = ModelTrainer()

trainer.train(
    df_labeled,
    features=features,
    label='label',
    test_size=0.2,
    sample_weight=None  # 或使用 'sample_weight' 欄位
)

# 6. 查看 Feature Importance
print("\n特徵重要性 Top 10:")
for feat, imp in sorted(trainer.feature_importance.items(), 
                        key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {feat}: {imp:.4f}")
```

---

## 預期成果

### 訓練指標

- **AUC**: 0.60+
- **Precision @ 0.60**: 58-60%
- **Feature Importance**: `divergence_score_10` 應該在 Top 5

### 關鍵特徵排名預期

1. `divergence_score_10` (新)
2. `atr_pct`
3. `lower_wick_ratio` (新)
4. `cvd_norm_10` (新)
5. `bb_width_pct`
6. `rsi_normalized`
7. `order_flow_imbalance` (新)
8. `volume_ratio`
9. `ema_9_dist`
10. `volatility_20`

---

## 特徵瘦身清單

若特徵總數接近 40 個,建議移除 Feature Importance 排名倒數的 10-15 個特徵:

### 建議移除的無效特徵

- 絕對均線距離 (非平穩)
- `bb_middle`, `bb_upper`, `bb_lower` (原始值)
- `volume_ma_20` (原始值)
- 長期報酬 (`return_20`)
- 冗餘的 EMA 比率

### 保留的核心特徵

- 所有相對/標準化特徵 (pct, ratio, normalized)
- 8 個微觀結構特徵
- ATR 相關
- BB 相關 (position, width)
- RSI, MACD

---

## 第一階段檢查清單

- [x] 優化 FeatureEngineer 加入 8 個核心微觀特徵
- [x] 精簡其他特徵避免過度擬合
- [x] 確保 include_microstructure=True 為預設
- [ ] 使用新特徵重新訓練模型
- [ ] 檢查 Feature Importance
- [ ] 移除低重要性特徵 (10-15 個)
- [ ] 再次訓練驗證 AUC 和 Precision

---

## 第二階段預覽

當第一階段完成後 (模型 AUC > 0.60, Precision @ 0.60 > 58%),將進入:

### 路線一: 樣本外盲測 (OOS Blind Test)

1. **鎖定 OOS 數據**:
   - 使用模型「完全沒見過」的數據
   - 例: 2023 下半年,或2026年新數據

2. **回測引擎驗證**:
   - 確保手續費正確 (Maker/Taker)
   - 確保滑點正確
   - 確保槓桿計算正確

3. **實戰參數**:
   - 初始資金: $10,000
   - 單筆風險: 1.5-2.0%
   - 機率門檻: 0.60
   - TP: 3.0 ATR, SL: 1.5 ATR
   - Maker: 0.02%, Taker: 0.06%, 滑點: 0.05%

---

## 相關文檔

- 理論基礎: `docs/LIQUIDITY_SWEEP_THEORY.md`
- 整合指南: `LIQUIDITY_SWEEP_INTEGRATION.md`
- 第二階段: `PHASE2_OOS_VALIDATION.md` (待建立)

---

## 問題排除

### Q: 特徵中沒有 taker_buy_base_asset_volume?

A: 確保使用 `fetch_latest_klines()` 而非 HuggingFace 數據集。Binance API 原生包含此欄位。

### Q: cvd_norm_10 為 NaN?

A: 正常,前 10 根 K 線無法計算。`build_features()` 會自動 `dropna()`。

### Q: Feature Importance 中 divergence_score_10 很低?

A: 可能需要:
1. 增加訓練數據量
2. 調整標籤參數 (TP/SL)
3. 使用樣本權重

---

**開始第一階段訓練**:

```bash
cd trading_system
streamlit run app_main.py
# 或
python examples/train_with_microstructure.py
```