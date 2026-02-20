# 流動性掠奪與微觀結構耗竭理論

## 理論核心

在加密貨幣市場中,機構資金 (Smart Money) 為了建立龐大部位,會主動推動價格去觸發散戶的停損單,獲取流動性。

**核心理念**: 不在突破時跟隨大眾,而是在大眾的停損被觸發且動能耗竭時,利用微觀數據背離進行狙擊。

---

## 三大支柱檢測

### 1. 價格行為: 假突破與流動性清掃

**條件**:
- 價格刺穿過去 20-50 根 K 線的高點/低點
- **關鍵**: K 線留下長影線 (影線 > 2x 實體)
- 收盤價回到突破點之上/之下

**意義**: 價格觸發停損後遭到強烈拒絕

```python
detector.detect_long_wick(df, direction='lower')  # 做多信號
detector.detect_support_resistance_breach(df)
```

---

### 2. 合約微觀結構: 未平倉量銳減 (OI Flush)

**條件**:
- 突破 K 線的 OI 下降 > 2σ (24h 標準差)

**意義**: 散戶爆倉/停損,燃料耗盡,原趨勢無法延續

```python
detector.detect_oi_flush(df)
```

**數據來源**: Binance Futures API
```python
loader.fetch_open_interest('BTCUSDT', '1h', days=90)
```

---

### 3. 訂單流背離: CVD 背離

**條件**:
- 做多: 價格新低 (LL) 但 CVD 較高低點 (HL)
- 做空: 價格新高 (HH) 但 CVD 較低高點 (LH)

**意義**: 機構在底部掛大量限價單吸收拋壓

```python
detector.detect_cvd_divergence(df, direction='lower')
```

**CVD 計算**:
```
CVD = Σ(Taker Buy Volume - Taker Sell Volume)
```

---

## 進場規則 (做多範例)

### 階段 1: 環境監控 (1h 級別)
- 尋找明顯支撐位 (Swing Low)

### 階段 2: 事件觸發 (15m 級別)
1. 15m 價格跌破支撐位
2. 迅速反彈,收盤價回到支撐之上 (長下影線)
3. 檢測該 K 線 OI 顯著下降
4. 檢測 CVD 底背離

### 階段 3: ML 機率審查
- 將事件特徵輸入 LightGBM 模型
- 預測成功反轉機率 > 0.65 才進場

### 階段 4: 執行與風控

**進場**:
- 下一根 K 線開盤
- 或在下影線 50% 回撤位掛限價單 (Maker 費率)

**止損 (SL)**:
- 掃蕩 K 線最低點 - 0.5 ATR
- 邏輯: 真實機構吸收,價格不該再破此點

**止盈 (TP)**:
- 對側流動性聚集區 (近期高點)
- R:R 通常 1:2.5 ~ 1:4

---

## 優勢對比舊系統

### 1. 手續費優化
- **舊**: 突破策略必須 Taker (0.06%)
- **新**: 左側確認右側進場,可用 Maker (0.02%)
- **節省**: 60% 手續費

### 2. 極致盈虧比
- **舊**: 突破後波動率大,SL 寬
- **新**: SL 在長影線尖端,精確且窄
- **結果**: 相同風險下開更大倉位

### 3. 數據維度降維打擊
- **舊**: 只有價格衍生指標 (MACD, RSI)
- **新**: 真實資金籌碼維度 (OI, CVD)
- **結果**: 盤整期預測能力大幅提升

---

## 使用方法

### 安裝依賴
```bash
pip install requests pandas numpy
```

### 快速開始

```python
from trading_system.core import CryptoDataLoader, LiquiditySweepDetector

# 1. 載入數據 (含 OI)
loader = CryptoDataLoader()
df = loader.fetch_latest_klines(
    symbol='BTCUSDT',
    timeframe='1h',
    days=90,
    include_oi=True,
    include_funding=True
)

# 2. 偵測流動性掃蕩
detector = LiquiditySweepDetector(
    lookback_period=50,
    wick_multiplier=2.0,
    oi_std_threshold=2.0
)

df_sweep = detector.detect_liquidity_sweep(df, direction='lower')

# 3. 計算特徵
df_sweep = detector.calculate_sweep_features(df_sweep)

# 4. 篩選信號
signals = df_sweep[df_sweep['sweep_lower_signal']]
print(f"偵測到 {len(signals)} 個流動性掃蕩信號")
```

### 運行測試
```bash
python test_liquidity_sweep.py
```

---

## 新特徵列表

### 價格結構
- `lower_wick_ratio`: 下影線/實體比例
- `upper_wick_ratio`: 上影線/實體比例
- `dist_to_support_pct`: 距離支撐位百分比
- `dist_to_resistance_pct`: 距離壓力位百分比

### OI 相關
- `oi_change_pct`: OI 單根變化率
- `oi_change_24h`: OI 24h 變化率
- `open_interest`: 未平倉量 (張)
- `open_interest_value`: 未平倉價值 (USD)

### CVD 相關
- `cvd`: 累計成交量差
- `cvd_slope`: CVD 5 根 K 線斜率
- `cvd_normalized`: CVD 標準化值

### 資金費率
- `funding_rate`: 8h 資金費率

---

## 下一階段: 整合到訓練流程

### 1. 修改標籤系統
將流動性掃蕩事件標記為高優先級訓練樣本:

```python
from trading_system.core import TripleBarrierLabeling

labeling = TripleBarrierLabeling(tp=3.0, sl=1.0)
df_labeled = labeling.label(df_sweep)

# 增強掃蕩事件的權重
df_labeled.loc[df_labeled['sweep_lower_signal'], 'sample_weight'] = 3.0
```

### 2. 訓練新模型
使用流動性掃蕩特徵訓練模型:

```python
from trading_system.core import ModelTrainer

trainer = ModelTrainer()
trainer.train(
    df_labeled,
    features=[
        'lower_wick_ratio', 'oi_change_pct', 'cvd_slope',
        'dist_to_support_pct', 'funding_rate',
        # ... 其他技術指標
    ]
)
```

### 3. 回測驗證
在回測系統中啟用流動性掃蕩過濾:

```python
from trading_system.core import Backtester

# 只在流動性掃蕩事件時進場
signals = df_sweep[
    (df_sweep['sweep_lower_signal']) &
    (df_sweep['win_probability'] > 0.65)
]

backtester = Backtester(...)
results = backtester.run_backtest(signals, ...)
```

---

## 參數調優建議

### 保守模式 (高精度)
```python
detector = LiquiditySweepDetector(
    lookback_period=50,
    wick_multiplier=2.5,      # 更長影線
    oi_std_threshold=2.5,     # 更大 OI 鈍減
    cvd_divergence_lookback=5 # 更短期背離
)
```

### 激進模式 (高召回)
```python
detector = LiquiditySweepDetector(
    lookback_period=30,
    wick_multiplier=1.5,      # 較短影線也接受
    oi_std_threshold=1.5,     # 較小 OI 變化
    cvd_divergence_lookback=15 # 更長期背離
)
```

---

## 常見問題

### Q1: OI 數據獲取失敗?
**A**: Binance Futures API 有時會限流,建議:
- 減少請求頻率
- 使用較長的時間框架 (1h, 4h)
- 緩存 OI 數據到本地

### Q2: 為什麼信號很少?
**A**: 流動性掃蕩是稀有事件 (月均 5-10 次),這是正常的:
- 30 天可能只有 1-2 個信號
- 建議使用 90 天以上數據
- 可調低 `wick_multiplier` 和 `oi_std_threshold`

### Q3: CVD 背離檢測不準?
**A**: CVD 對市場深度敏感,建議:
- 使用流動性好的幣種 (BTC, ETH)
- 調整 `cvd_divergence_lookback`
- 結合 RSI 背離雙重確認

---

## 授權

MIT License

---

## 貢獻

歡迎提交 Issue 和 PR!

聯繫: [Your Email]

---

**免責聲明**: 本系統僅供教育和研究用途。加密貨幣交易存在高風險,請謹慎使用。