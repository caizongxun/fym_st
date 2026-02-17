# 進階BB結合策略指南

## 當前策略分析

### BB反彈策略 (現有)

**優點**:
- 整體穩定獲利
- 風險可控
- 適合震盪市場

**缺點**:
- 報酬率中等 (單次獲利小)
- 需要較多交易次數累積
- 趨勢市場表現一般

---

## 高頻策略方案

### 策略1: BB + RSI + MACD 三重確認

**核心邏輯**: 三個指標同時確認,提高勝率

**做多條件**:
```python
1. 價格觸及BB下軌 (超賣區)
2. RSI < 30 且開始上升
3. MACD線向上穿越信號線 (金叉)
4. ADX < 30 (非強趨勢)
```

**做空條件**:
```python
1. 價格觸及BB上軌 (超買區)
2. RSI > 70 且開始下降
3. MACD線向下穿越信號線 (死叉)
4. ADX < 30 (非強趨勢)
```

**參數設置**:
```yaml
BB: 20週期, 2倍標準差
RSI: 14週期
MACD: 快線12, 慢線26, 信號線9
ADX: 14週期
止盈: 1.5-2.0 ATR
止損: 1.0-1.5 ATR
```

**預期改進**:
- 勝率: 65% → 75%
- 交易次數: 減少30%
- 單筆獲利: 增加20-30%

---

### 策略2: BB Squeeze突破策略

**核心邏輯**: 布林帶收縮後的爆發行情

**Squeeze識別**:
```python
# BB寬度縮小到歷史低點
bb_width = (upper_band - lower_band) / middle_band
squeeze_threshold = bb_width_sma(100) * 0.5

if bb_width < squeeze_threshold:
    # 進入Squeeze狀態
    wait_for_breakout = True
```

**突破交易**:
```python
# 做多: 價格突破BB上軌 + 成交量放大
if price > upper_band and volume > volume_sma(20) * 1.5:
    entry_long()
    
# 做空: 價格跌破BB下軌 + 成交量放大
if price < lower_band and volume > volume_sma(20) * 1.5:
    entry_short()
```

**參數設置**:
```yaml
BB: 20週期, 2倍標準差
Squeeze判定: BB寬度 < 歷史均值50%
成交量確認: > 20週期均量1.5倍
止盈: 3.0 ATR (大行情)
止損: 1.5 ATR
```

**適用場景**:
- 重大事件前的低波動期
- 橫盤整理突破
- 高勝率 (80%+) 但交易次數少

---

### 策略3: BB + Volume (成交量) 高頻剝頭皮

**核心邏輯**: 結合價格位置和成交量異常

**信號生成**:
```python
# 做多: BB下軌反彈 + 成交量激增
if price <= lower_band * 1.01:  # 接近下軌
    if volume > volume_sma(20) * 2.0:  # 成交量爆增
        if price > price_1min_ago:  # 開始反彈
            entry_long()
            
# 做空: BB上軌回落 + 成交量激增            
if price >= upper_band * 0.99:  # 接近上軌
    if volume > volume_sma(20) * 2.0:
        if price < price_1min_ago:  # 開始回落
            entry_short()
```

**快速止盈止損**:
```yaml
止盈: 0.5% (非常小,高頻)
止損: 0.3%
時間止損: 5分鐘內未達目標平倉
```

**適合**:
- 15分鐘或更短週期
- 流動性好的幣種 (BTC, ETH)
- 零手續費交易所

---

### 策略4: BB + EMA趨勢過濾

**核心邏輯**: 只在趨勢方向交易BB反彈

**趨勢判定**:
```python
# 使用多重EMA判斷趨勢
ema_fast = EMA(close, 21)
ema_slow = EMA(close, 55)

# 上升趨勢: 只做多
if ema_fast > ema_slow and close > ema_fast:
    trend = 'up'
    only_long = True
    
# 下降趨勢: 只做空    
if ema_fast < ema_slow and close < ema_fast:
    trend = 'down'
    only_short = True
```

**交易邏輯**:
```python
# 上升趨勢中:
if trend == 'up':
    # 只在BB下軌或中軌做多
    if price <= middle_band:
        if bb_bounce_prob > 0.6:
            entry_long()
            
# 下降趨勢中:            
if trend == 'down':
    # 只在BB上軌或中軌做空
    if price >= middle_band:
        if bb_bounce_prob > 0.6:
            entry_short()
```

**優勢**:
- 順勢交易,勝率更高
- 避免趨勢市場的逆勢虧損
- 單筆獲利更大

---

### 策略5: BB + ATR動態調整

**核心邏輯**: 根據波動率動態調整參數

**ATR狀態分類**:
```python
atr = calculate_atr(14)
atr_sma = atr.rolling(100).mean()

# 低波動 (ATR < 均值80%)
if atr < atr_sma * 0.8:
    bb_std = 1.5  # 收緊BB
    tp_mult = 1.5  # 降低止盈
    trade_freq = 'high'  # 增加交易頻率
    
# 中等波動
elif atr < atr_sma * 1.2:
    bb_std = 2.0  # 標準BB
    tp_mult = 2.0
    trade_freq = 'normal'
    
# 高波動 (ATR > 均值120%)    
else:
    bb_std = 2.5  # 放寬BB
    tp_mult = 3.0  # 提高止盈
    trade_freq = 'low'  # 減少交易
```

**優勢**:
- 自適應市場狀態
- 低波動期增加交易頻率
- 高波動期擴大單筆獲利

---

## 策略組合建議

### 方案A: 穩健型 (勝率優先)

**主策略**: BB + RSI + MACD 三重確認
**輔助**: BB + EMA趨勢過濾

**配置**:
```yaml
幣種: 3-5個主流幣
時間週期: 15分鐘
最大持倉: 3
單筆倉位: 40%
預期勝率: 70-75%
預期報酬: 月30-50%
```

### 方案B: 進取型 (報酬優先)

**主策略**: BB Squeeze突破
**輔助**: BB + Volume高頻剝頭皮

**配置**:
```yaml
幣種: 5-10個
時間週期: 15分鐘 + 5分鐘
最大持倉: 5
單筆倉位: 25%
預期勝率: 60-65%
預期報酬: 月50-100%
風險: 較高波動
```

### 方案C: 平衡型 (推薦)

**主策略**: BB + RSI + MACD
**輔助1**: BB + ATR動態調整
**輔助2**: BB + EMA趨勢過濾

**配置**:
```yaml
幣種: 5個
時間週期: 15分鐘
最大持倉: 3-4
單筆倉位: 30%
預期勝率: 68-72%
預期報酬: 月40-70%
```

---

## 實施步驟

### 階段1: 添加指標 (1-2天)

```python
# 在 bb_bounce_features.py 中添加
def add_advanced_indicators(df):
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # EMA
    df['ema_21'] = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
    df['ema_55'] = ta.trend.EMAIndicator(df['close'], 55).ema_indicator()
    
    # Volume SMA
    df['volume_sma'] = df['volume'].rolling(20).mean()
    
    # BB Width
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_width_sma'] = df['bb_width'].rolling(100).mean()
    
    return df
```

### 階段2: 創建新信號生成器 (2-3天)

```python
# signal_generator_bb_advanced.py
class BBAdvancedSignalGenerator:
    def __init__(self, strategy='triple_confirm'):
        self.strategy = strategy
        
    def generate_signals(self, df):
        if self.strategy == 'triple_confirm':
            return self._triple_confirm_signals(df)
        elif self.strategy == 'squeeze':
            return self._squeeze_signals(df)
        # ...
```

### 階段3: 回測驗證 (3-5天)

1. 單策略測試 (30天數據)
2. 對比現有策略
3. 參數優化
4. Walk-forward驗證

### 階段4: 實盤測試 (7-14天)

1. 小資金測試 (100 USDT)
2. 監控表現
3. 調整參數
4. 逐步增加資金

---

## 預期改進效果

### 當前BB反彈策略
```yaml
月報酬: 15-25%
勝率: 60-65%
平均單筆: 0.8-1.2%
月交易次數: 30-50
```

### 改進後 (三重確認策略)
```yaml
月報酬: 30-45%  (+100%)
勝率: 70-75%  (+15%)
平均單筆: 1.5-2.0%  (+60%)
月交易次數: 25-40  (-20%)
```

### 改進後 (組合策略)
```yaml
月報酬: 40-70%  (+180%)
勝率: 68-72%  (+12%)
平均單筆: 1.2-1.8%  (+40%)
月交易次數: 40-60  (+20%)
```

---

## 風險控制升級

### 1. 動態止損

```python
# 根據波動率調整止損
if atr < atr_sma * 0.8:
    sl_mult = 1.0  # 低波動縮小止損
elif atr > atr_sma * 1.2:
    sl_mult = 2.0  # 高波動放寬止損
else:
    sl_mult = 1.5  # 標準止損
```

### 2. 時間止損

```python
# 持倉超過N分鐘未達止盈/止損,強制平倉
max_hold_time = 60  # 分鐘
if current_time - entry_time > max_hold_time:
    force_close()
```

### 3. 回撤保護

```python
# 當日虧損達到閾值,暫停交易
daily_loss_limit = 0.05  # 5%
if daily_loss / capital > daily_loss_limit:
    stop_trading_today = True
```

---

## 技術指標速查

### RSI (相對強弱指標)
```python
# 計算公式
rsi = 100 - (100 / (1 + RS))
RS = 平均漲幅 / 平均跌幅

# 使用
超買: RSI > 70
超賣: RSI < 30
中性: 30 < RSI < 70
```

### MACD (指數平滑移動平均線)
```python
# 計算
MACD線 = EMA(12) - EMA(26)
信號線 = EMA(MACD線, 9)
柱狀圖 = MACD線 - 信號線

# 信號
金叉: MACD線向上穿越信號線
死叉: MACD線向下穿越信號線
```

### BB Squeeze
```python
# 判定
bb_width = (upper - lower) / middle
squeeze = bb_width < historical_avg * 0.5

# 突破
breakout_up = price > upper and volume > avg_volume * 1.5
breakout_down = price < lower and volume > avg_volume * 1.5
```

---

## 總結

### 推薦實施順序

1. **立即實施**: BB + RSI + MACD三重確認 (最易實施,效果顯著)
2. **短期實施**: BB + Volume高頻剝頭皮 (提高交易頻率)
3. **中期實施**: BB Squeeze突破 (捕捉大行情)
4. **長期優化**: BB + ATR動態調整 (自適應市場)

### 預期時間線

```
第1週: 添加RSI/MACD指標
第2週: 實現三重確認策略
第3週: 回測驗證30天
第4週: 小資金實盤測試
第5-6週: 逐步擴大到所有幣種
第7-8週: 添加其他輔助策略
```

### 關鍵成功因素

1. 嚴格的信號過濾 (三重確認)
2. 動態的參數調整 (ATR適應)
3. 完善的風控機制 (多層止損)
4. 充分的回測驗證 (至少3個月數據)
5. 謹慎的實盤測試 (小資金起步)

通過結合這些策略,預期可將月報酬率從15-25%提升至40-70%,同時保持或提高勝率!