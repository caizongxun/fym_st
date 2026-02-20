# 特徵平稩性指南 (Feature Stationarity Guide)

## 致命問題:為什麼 OOS 勝率從 55% 暴跌至 33%?

### 根本原因:資料工程問題 (Data Engineering)

您的模型邏輯與 CVD 理論沒有問題,**問題出在特徵的平稩性**。

---

## 致命元兇一:缺失特徵被補零 (Data Mismatch)

### 發生原因

```python
# 訓練時 (HuggingFace 數據)
df_train.columns = [
    'open', 'high', 'low', 'close', 'volume',
    'quote_asset_volume',  # ✅ 存在
    'number_of_trades',     # ✅ 存在
    'taker_buy_quote_asset_volume'  # ✅ 存在
]

# 回測時 (Binance API)
df_backtest.columns = [
    'open', 'high', 'low', 'close', 'volume',
    'taker_buy_base_asset_volume'  # 只有這個
    # quote_asset_volume: ❗ 缺失
    # number_of_trades: ❗ 缺失
]
```

### 災難後果

當回測引擎發現模型需要 `number_of_trades` 卻找不到時:

```python
# 模型期望
 X['number_of_trades'] = 15000  # 高活躍度
 
 # 實際收到
X['number_of_trades'] = 0  # 被強制補零!

# 決策樹規則
if number_of_trades < 5000:
    return probability = 0.20  # 完全錯誤!
```

**結果**: 機率分布被壓低,勝率崩盤。

---

## 致命元兇二:非平稩特徵導致模型「刻舶求劍」

### 問題案例

```python
# 2023 年訓練時
BTC = $30,000
df_train['bb_middle'] = 30000
df_train['volume'] = 1000 BTC

# 模型學到的規則
if bb_middle > 45000:
    return "超買" 

# 2026 年回測時
BTC = $90,000
df_backtest['bb_middle'] = 90000  # 完全超出訓練範圍!

# 模型失效
if bb_middle > 45000:  # 永遠 True
    return "超買"  # 錯誤!
```

### 災難後果

決策樹是基於數值大小進行切分的。**當未來的價格跨越了歷史區間,包含絕對價格的特徵會讓模型完全失效**。

---

## 解決方案:特徵大掃除

### 已封殺的危險特徵

#### 1. 絕對價格特徵

```python
forbidden = [
    'open',        # 絕對開盤價
    'high',        # 絕對最高價
    'low',         # 絕對最低價
    'close',       # 絕對收盤價
    'bb_middle',   # 絕對 BB 中軌
    'bb_upper',    # 絕對 BB 上軌
    'bb_lower',    # 絕對 BB 下軌
]
```

**為什麼**: BTC 價格從 $30K → $90K,範圍完全不同

#### 2. 絕對成交量特徵

```python
forbidden = [
    'volume',          # 絕對成交量
    'volume_ma_20',    # 絕對成交量 MA
    'taker_buy_base_asset_volume',  # 絕對主動買量
]
```

**為什麼**: 成交量隨時間變化,2023 與 2026 的成交量絕對值不可比

#### 3. API 不穩定欄位

```python
forbidden = [
    'quote_asset_volume',           # ❗ Binance API 可能缺失
    'taker_buy_quote_asset_volume', # ❗ 欄位名不一致
    'number_of_trades',             # ❗ HuggingFace 獨有
    'trades',                       # ❗ 名稱變體
]
```

**為什麼**: 不同數據源欄位不一致,導致被補零

---

## 保留的平稩特徵

### ✓ 比例特徵

```python
safe_features = [
    'bb_width_pct',     # BB 寬度 / BB 中軌 (比例)
    'volume_ratio',     # 當前成交量 / MA20 (比例)
    'taker_buy_ratio',  # 主動買 / 總成交量 (比例)
    'atr_pct',          # ATR / 價格 (百分比)
]
```

**為什麼安全**: 比例無論價格高低都一致 (0-2 範圍)

### ✓ 標準化特徵

```python
safe_features = [
    'rsi_normalized',       # (RSI - 50) / 50 (範圍 -1 到 +1)
    'cvd_norm_10',          # CVD / 總成交量 (標準化)
    'macd_normalized',      # MACD / 價格 (百分比)
    'oi_normalized',        # (OI - mean) / std (z-score)
]
```

**為什麼安全**: 標準化後範圍固定 (-3 到 +3)

### ✓ 距離比特徵

```python
safe_features = [
    'ema_9_dist',       # (價格 - EMA9) / EMA9 (百分比)
    'ema_21_dist',      # (價格 - EMA21) / EMA21
    'bb_position',      # (價格 - BB下) / (BB上 - BB下) (0-1)
]
```

**為什麼安全**: 相對距離,無論價格高低都有意義

### ✓ 影線比例

```python
safe_features = [
    'upper_wick_ratio',  # 上影線 / 實體
    'lower_wick_ratio',  # 下影線 / 實體
    'body_size',         # |收盤 - 開盤| / 價格
]
```

**為什麼安全**: 相對比例,範圍 0-5

### ✓ 微觀結構特徵 (核心)

```python
safe_features = [
    'net_volume',              # 淨主動成交量 (濾動視窗)
    'cvd_10',                  # CVD 10 期 (濾動)
    'cvd_norm_10',             # 標準化 CVD
    'divergence_score_10',     # 價格-CVD 背離 (核心!)
    'order_flow_imbalance',    # 訂單流失衡 (-1 到 +1)
]
```

**為什麼安全**: 使用濾動視窗與標準化,具備平稩性

---

## 實施步驟

### 步驟 1: 重新訓練 (使用修正後的訓練頁面)

```bash
cd trading_system
streamlit run app_main.py
```

1. 點擊 **"模型訓練"**
2. 選擇 BTCUSDT, 1h
3. TP: 3.0, SL: 1.0
4. 啟用事件過濾 (嚴格模式)
5. 點擊 **"開始訓練"**

### 步驟 2: 確認特徵清理

訓練時應該看到:

```
⚠️ 特徵大掃除:移除絕對值與 API 不穩定特徵
✅ 移除 12 個非平稩特徵: open, high, low, close, volume...
訓練數據: 5261 樣本, 25 特徵
```

**關鍵**: 特徵數應從 37 降至 20-25 個

### 步驟 3: 檢查特徵重要性

展開 **"保留的平稩特徵"** 查看列表,確認:

✅ 無 `open`, `high`, `low`, `close`
✅ 無 `volume`, `volume_ma_20`
✅ 無 `bb_middle`, `bb_upper`, `bb_lower`
✓ 有 `bb_width_pct`, `bb_position`
✓ 有 `cvd_norm_10`, `divergence_score_10`
✓ 有 `lower_wick_ratio`, `upper_wick_ratio`

### 步驟 4: OOS 回測 (無缺失特徵)

前往 **"回測分析"**:

1. 選擇新訓練的模型
2. 數據來源: **Binance API** (90 天)
3. 機率門檻: 0.55 或 0.60
4. 啟用事件過濾
5. 點擊 **"運行回測"**

### 步驟 5: 確認無警告

回測運行時應該看到:

```
模型特徵: 25 個
缺失特徵 (0):   # ✅ 必須為 0!
機率分布: min=0.320, mean=0.485, max=0.782
信號: 25 個 (門檻: 0.55)
```

**關鍵**: 缺失特徵必須為 **0**!

---

## 預期成果

### 訓練階段

| 指標 | 修正前 | 修正後 (預期) |
|------|---------|----------------|
| 特徵數 | 37 | 20-25 |
| AUC | 0.607 | 0.60-0.62 |
| 精確率 | 55.6% | 55-58% |
| 期望值 | 0.724R | 0.5-0.8R |

### OOS 回測階段

| 指標 | 修正前 | 修正後 (預期) |
|------|---------|----------------|
| 缺失特徵 | 3 個 | **0 個** |
| 勝率 | 33% | **50-60%** |
| 盈虧比 | 0.8 | **1.5-2.5** |
| 總報酬 | -15% | **+5% 到 +15%** |

---

## 成功標準

### 訓練階段

- ✅ 特徵數 20-25 個
- ✅ 無絕對值特徵
- ✅ AUC > 0.58
- ✅ 期望值 > 0.3R

### OOS 回測階段

- ✅ **缺失特徵 = 0** (最關鍵!)
- ✅ 勝率 > 50%
- ✅ 盈虧比 > 1.5
- ✅ 總報酬 > 0%
- ✅ 資金曲線平穩向上

---
## 常見問題

### Q: 為什麼不保留 `close` 去計算 `return`?

A: `return` 是相對變化 (`pct_change`),是平稩的。但 `close` 本身是絕對值,非平稩。

### Q: 為什麼 `atr` 也被移除?

A: `atr` 是絕對值。我們保留 `atr_pct = atr / close`,這是相對比例。

### Q: `cvd_10` 是絕對成交量,為什麼保留?

A: 因為使用了濾動視窗 (rolling 10),且配合 `cvd_norm_10` 標準化,具備平稩性。

### Q: 特徵數從 37 降至 25,模型會變弱嗎?

A: 不會!移除的是 **干擾特徵**,保留的都是 **有效特徵**。質量 > 數量。

### Q: 回測時仍有缺失特徵警告?

A: 確認你使用的是 **重新訓練** 後的模型。舊模型仍包含危險特徵。

---

## 核心原則

### ① 只使用比例/標準化特徵

**錯誤**: `if close > 50000`
**正確**: `if bb_position > 0.8`

### ② 避免 API 不穩定欄位

**錯誤**: 使用 `number_of_trades`
**正確**: 使用 `volume_ratio`

### ③ 確保跨數據源一致性

**錯誤**: HuggingFace 訓練 → Binance 回測 (欄位不同)
**正確**: 只使用兩者都有的欄位

### ④ 特徵少而精

**錯誤**: 37 個特徵 (包含冗餘)
**正確**: 20-25 個有效特徵

---

## 總結

這次特徵大掃除是系統邁向實盤前**最關鍵的資料清洗步驟**。

**完成後你將看到**:
1. ✅ 缺失特徵警告消失
2. ✅ 回測勝率回歸正常 (50%+)
3. ✅ 資金曲線平穩向上
4. ✅ 模型具備真正的泛化能力

**立即行動**:
```bash
cd trading_system
streamlit run app_main.py
# 點擊 "模型訓練" → 重新訓練
```

重新訓練後,再次運行 90 天 Binance API 回測,缺失特徵警告將會消失,資金曲線將會回歸正軌!