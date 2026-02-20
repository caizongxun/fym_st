# 流動性掃蕩系統 - 快速開始

## 5 分鐘上手

### 1. 啟動 GUI

```bash
cd trading_system
streamlit run app_main.py
```

### 2. 選擇頁面

在左側導航選單點擊 **"流動性掃蕩分析"**

### 3. 配置參數

#### 基礎設定
- **交易對**: BTCUSDT
- **時間框架**: 1h
- **回測天數**: 90
- **信號方向**: lower (做多)
- **OI 數據**: ☑️
- **資金費率**: ☑️

#### 預設參數 (建議)
- **影線倍數**: 2.0
- **OI 銳減門檻**: 2.0σ
- **CVD 背離回期**: 10

### 4. 運行分析

點擊 **"運行分析"** 按鈕

### 5. 查看結果

系統會顯示:
- **築選漏斗**: 每個條件的符合數量
- **信號表格**: 所有流動性掃蕩事件詳情
- **K 線圖**: 帶有信號標記
- **OI 變化圖**: 未平倉量走勢
- **CVD 圖**: 成交量差走勢

---

## 常見情境

### 情境 1: 找不到信號

**原因**: 流動性掃蕩是稀有事件 (90天 2-10 個)

**解決**:
1. 增加天數: 90 → 180
2. 調低影線倍數: 2.0 → 1.5
3. 調低 OI 門檻: 2.0 → 1.5
4. 嘗試不同幣種: ETH, SOL

### 情境 2: OI 數據獲取失敗

**原因**: Binance API 限流

**解決**:
1. 稍後再試
2. 使用 1h 或 4h 時間框架
3. 減少回測天數

### 情境 3: 想整合 ML 模型

**步驟**:

1. 先訓練包含流動性特徵的模型:
   ```python
   fe = FeatureEngineer()
   df = fe.build_features(df, include_liquidity_features=True)
   ```

2. 在流動性掃蕩頁面:
   - 展開 **"模型整合"**
   - ☑️ **"使用 ML 模型過濾"**
   - 選擇你的模型文件
   - 設定機率門檻: 0.65

3. 運行分析後會顯示:
   ```
   模型過濾: 5 → 3 (門檻: 0.65)
   ```

---

## 進階使用

### 程式化使用

```python
from trading_system.core import (
    CryptoDataLoader, 
    LiquiditySweepDetector,
    FeatureEngineer,
    ModelTrainer
)

# 1. 載入數據
loader = CryptoDataLoader()
df = loader.fetch_latest_klines(
    'BTCUSDT', '1h', days=90,
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
df_sweep = detector.calculate_sweep_features(df_sweep)

# 3. 篩選信號
signals = df_sweep[df_sweep['sweep_lower_signal']]
print(f"偵測到 {len(signals)} 個信號")

# 4. 查看詳情
for idx, row in signals.iterrows():
    print(f"{row['open_time']}: ${row['close']:.2f}")
    print(f"  下影線: {row['lower_wick_ratio']:.2f}x")
    print(f"  OI 變化: {row['oi_change_pct']*100:.2f}%")
```

### 與回測系統整合

```python
from trading_system.core import Backtester

# 只在流動性掃蕩事件時進場
filtered_signals = df_sweep[
    (df_sweep['sweep_lower_signal']) &
    (df_sweep['win_probability'] > 0.65)  # 可選: ML 過濾
]

backtester = Backtester(
    initial_capital=10000,
    risk_per_trade=0.01,
    leverage=10
)

results = backtester.run_backtest(
    filtered_signals,
    tp_multiplier=3.5,
    sl_multiplier=1.5
)

print(f"總報酬: {results['statistics']['total_return']*100:.1f}%")
print(f"勝率: {results['statistics']['win_rate']*100:.1f}%")
```

---

## 參數調優建議

### 保守模式 (高精度, 低頻率)

```
影線倍數: 2.5
OI 門檻: 2.5σ
CVD 回期: 5

預期: 90天 2-5 個信號
勝率: 70%+
```

### 激進模式 (中等精度, 高頻率)

```
影線倍數: 1.5
OI 門檻: 1.5σ
CVD 回期: 15

預期: 90天 8-15 個信號
勝率: 55-60%
```

### 最佳實踐 (建議)

```
影線倍數: 2.0
OI 門檻: 2.0σ
CVD 回期: 10
ML 機率: > 0.65

預期: 90天 5-8 個信號
勝率: 65%+
```

---

## 實盤交易注意事項

### 進場策略

1. **第一選擇**: 在下影線 50% 回撤位掛限價單 (Maker)
2. **第二選擇**: 下一根 K 線開盤价市價單 (Taker)

### 停損設定

```
SL = 下影線最低點 - 0.5 ATR
```

### 止盈設定

```
TP = 近期高點 (對側流動性聚集區)
或

TP = 進場價 + 3.5 ATR
```

### 風險管理

- 單筆風險: 1-2% 資金
- 最大同時持倉: 3 筆
- 使用限價單節省手續費

---

## 常見問題

**Q: 為什麼我的信號很少?**

A: 流動性掃蕩是稀有事件,這是正常的。質量 > 數量。

**Q: OI 數據獲取失敗?**

A: Binance API 有時會限流,稍後再試或使用較長時間框架。

**Q: 如何驗證效果?**

A: 
1. 先回測 90-180 天數據
2. 比較有/無流動性過濾的績效
3. 實盤小量測試

**Q: 可以同時做空嗎?**

A: 可以,將 `direction` 設為 `upper`,系統會偵測高點掃蕩事件。

---

## 相關資源

- **完整理論**: `docs/LIQUIDITY_SWEEP_THEORY.md`
- **整合指南**: `LIQUIDITY_SWEEP_INTEGRATION.md`
- **更新日誌**: `CHANGELOG.md`
- **測試腳本**: `test_liquidity_sweep.py`
- **示例程式**: `examples/liquidity_sweep_example.py`

---

## 支持

遇到問題?
1. 查看 `docs/LIQUIDITY_SWEEP_THEORY.md`
2. 運行 `python test_liquidity_sweep.py`
3. 提交 GitHub Issue

---

**Happy Trading!** 🚀

---

**免責聲明**: 本系統僅供教育和研究用途。加密貨幣交易存在高風險,請謹慎使用。