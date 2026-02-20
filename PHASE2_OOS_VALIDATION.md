# 第二階段:樣本外盲測與實盤對接 (Phase 2: OOS Blind Test & Live Trading)

## 目標

**以最嚴格的樣本外盲測來驗證最終成果**

當第一階段完成 (模型 AUC > 0.60, Precision @ 0.60 > 58%) 後,進入最嚴肅的實戰檢驗環節。

若在極度嚴格的盲測環境下,系統仍能跑出平穩向上的資金曲線,且平均獲利確實大於平均虧損,
這便宣告系統已經正式研發完成,隨時可以串接 Binance API 轉入小資金實盤運行。

---

## 第二階段檢查清單

### 1. 鎖定樣本外數據 (Out-of-Sample Data)

- [ ] 確定模型訓練數據範圍 (例: 2024-01-01 ~ 2026-02-01)
- [ ] 保留模型「完全沒見過」的數據:
  - **選項 A**: 2023 年下半年數據
  - **選項 B**: 2026-02-01 之後的實時數據 (累積 1-2 週)
  - **選項 C**: 2024 年特定月份 (例: 7-9 月)
- [ ] 確保 OOS 數據至少 90 天 (約 2160 根 1h K線)

### 2. 更換為修復後的回測引擎

確保使用最新版本的 `Backtester` (已修復手續費、滑點、槓桿計算):

```python
from trading_system.core import Backtester

backtester = Backtester(
    initial_capital=10000.0,
    taker_fee=0.0006,      # 0.06% Taker
    maker_fee=0.0002,      # 0.02% Maker
    slippage=0.0005,       # 0.05% 滑點
    risk_per_trade=0.015,  # 1.5% 風險
    leverage=10            # 10x 槓桿
)
```

### 3. 設置實戰參數

#### 資金管理
- **初始資金**: $10,000 USDT
- **單筆風險**: 1.5% 或 2.0% 總資金
- **槓桿倍數**: 10x
- **最大同時持倉**: 3 筆

#### 信號篩選
- **機率門檻**: 0.60 (或模型表現最佳的高置信度區間)
- **事件過濾**: 啟用 EventFilter (若使用)
- **最小 ATR**: 過濾低波動性時段

#### 止盈止損
- **TP (Take Profit)**: 3.0 ATR
- **SL (Stop Loss)**: 1.5 ATR
- **Timeout**: 48 根 K線 (1h 約 2 天)

#### 費率設定
- **Maker 費率**: 0.0002 (0.02%)
- **Taker 費率**: 0.0006 (0.06%)
- **滑點**: 0.0005 (0.05%)

---

## 實施步驟

### 步驟 1: 準備 OOS 數據

```python
from trading_system.core import CryptoDataLoader

loader = CryptoDataLoader()

# 載入 OOS 數據 (模型從未見過)
df_oos = loader.fetch_latest_klines(
    symbol='BTCUSDT',
    timeframe='1h',
    days=90,  # 或指定日期範圍
    include_oi=True,
    include_funding=True
)

print(f"OOS 數據: {len(df_oos)} 筆")
print(f"範圍: {df_oos['open_time'].min()} ~ {df_oos['open_time'].max()}")
```

### 步驟 2: 建立特徵

```python
from trading_system.core import FeatureEngineer

fe = FeatureEngineer()
df_features = fe.build_features(
    df_oos,
    include_microstructure=True,  # 使用第一階段的特徵
    include_liquidity_features=True
)

print(f"特徵建立完成: {len(df_features)} 筆")
```

### 步驟 3: 載入訓練好的模型

```python
from trading_system.core import ModelTrainer

trainer = ModelTrainer()
trainer.load_model('model_20260220_phase1.pkl')  # 第一階段訓練的模型

print(f"模型加載: {trainer.model_metrics['auc']:.4f} AUC")
print(f"特徵數量: {len(trainer.feature_names)}")
```

### 步驟 4: 生成預測

```python
import pandas as pd
import numpy as np

# 選擇特徵
available_features = [f for f in trainer.feature_names if f in df_features.columns]
X_oos = df_features[available_features].fillna(0).replace([np.inf, -np.inf], 0)

# 預測
probabilities = trainer.predict_proba(X_oos)
df_features['win_probability'] = probabilities

# 篩選信號
signals = df_features[df_features['win_probability'] >= 0.60]

print(f"\n信號統計:")
print(f"  總 K線: {len(df_features)}")
print(f"  符合門檻: {len(signals)} ({100*len(signals)/len(df_features):.1f}%)")
print(f"  平均機率: {signals['win_probability'].mean():.3f}")
```

### 步驟 5: OOS 回測

```python
from trading_system.core import Backtester

# 初始化回測引擎
backtester = Backtester(
    initial_capital=10000.0,
    taker_fee=0.0006,
    maker_fee=0.0002,
    slippage=0.0005,
    risk_per_trade=0.015,
    leverage=10
)

# 運行回測
results = backtester.run_backtest(
    signals,
    tp_multiplier=3.0,
    sl_multiplier=1.5,
    direction=1  # 1=做多, -1=做空
)

# 查看結果
stats = results['statistics']

print("\n" + "="*60)
print("OOS 回測結果")
print("="*60)
print(f"\n基本指標:")
print(f"  交易次數: {stats['total_trades']}")
print(f"  勝率: {stats['win_rate']*100:.1f}%")
print(f"  總報酬: {stats['total_return']*100:.1f}%")
print(f"  年化報酬: {stats['total_return']*365/90*100:.1f}%")

print(f"\n風險指標:")
print(f"  盈虧比: {stats['profit_factor']:.2f}")
print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
print(f"  最大回撤: {stats['max_drawdown']*100:.1f}%")

print(f"\n交易細節:")
print(f"  平均獲利: {stats['avg_win']*100:.2f}%")
print(f"  平均虧損: {stats['avg_loss']*100:.2f}%")
print(f"  獲利/虧損比: {abs(stats['avg_win']/stats['avg_loss']):.2f}")
```

### 步驟 6: 視覺化分析

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

trades = results['trades']
equity_curve = results['equity_curve']

# 資金曲線圖
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('OOS 資金曲線', '回撤'),
    vertical_spacing=0.1,
    row_heights=[0.7, 0.3]
)

# 資金曲線
fig.add_trace(
    go.Scatter(
        x=list(range(len(equity_curve))),
        y=equity_curve,
        mode='lines',
        name='資金',
        line=dict(color='blue', width=2)
    ),
    row=1, col=1
)

# 回撤
running_max = pd.Series(equity_curve).cummax()
drawdown = (pd.Series(equity_curve) - running_max) / running_max * 100

fig.add_trace(
    go.Scatter(
        x=list(range(len(drawdown))),
        y=drawdown,
        mode='lines',
        name='回撤',
        line=dict(color='red'),
        fill='tozeroy'
    ),
    row=2, col=1
)

fig.update_xaxes(title_text="交易次數", row=2, col=1)
fig.update_yaxes(title_text="資金 ($)", row=1, col=1)
fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)

fig.update_layout(height=800, showlegend=True)
fig.show()
```

---

## 成功標準

### 最低要求

- ☑️ 總報酬 > 0% (盈利)
- ☑️ 勝率 > 50%
- ☑️ 盈虧比 > 1.5
- ☑️ 平均獲利 > 平均虧損
- ☑️ 最大回撤 < 20%

### 優秀標準

- ⭐ 總報酬 > 10% (90天)
- ⭐ 勝率 > 55%
- ⭐ 盈虧比 > 2.0
- ⭐ Sharpe > 1.0
- ⭐ 最大回撤 < 15%

### 卓越標準

- 🏆 總報酬 > 15% (90天)
- 🏆 勝率 > 60%
- 🏆 盈虧比 > 2.5
- 🏆 Sharpe > 1.5
- 🏆 最大回撤 < 10%

---

## 實盤對接準備

若 OOS 回測達到「最低要求」,可以開始準備實盤:

### 1. Binance API 設定

```python
import os
from binance.client import Client

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

client = Client(api_key, api_secret)

# 測試連接
account = client.get_account()
print(f"帳戶連接成功: {account['canTrade']}")
```

### 2. 小資金測試

- **建議起始資金**: $100-500 USDT
- **測試期**: 1-2 週
- **最大風險**: 單筆 1%

### 3. 實時監控

使用 Streamlit GUI 的 "即時預測" 頁面:

```bash
cd trading_system
streamlit run app_main.py
# 選擇 "即時預測"
```

---

## 常見問題

**Q: OOS 回測處於虛損狀態?**

A: 可能原因:
1. 模型過度擬合訓練數據
2. OOS 期間市場狀態與訓練期差異太大
3. 特徵工程有 lookahead bias

**解決**: 回到第一階段重新訓練

**Q: 交易頻率太低?**

A:
- 調低機率門檻: 0.60 → 0.55
- 增加交易對數量
- 使用較短時間框架 (15m)

**Q: 勝率高但盈虧比低?**

A:
- 提高 TP 倍數: 3.0 → 3.5 ATR
- 降低 SL 倍數: 1.5 → 1.0 ATR

---

## 相關文檔

- 第一階段: `PHASE1_MICROSTRUCTURE_TRAINING.md`
- 回測引擎: `trading_system/core/backtester.py`
- 即時預測: `trading_system/core/predictor.py`

---

**開始第二階段驗證**:

```bash
cd trading_system
python examples/oos_validation.py
```