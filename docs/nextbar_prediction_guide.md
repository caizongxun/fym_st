# 下一根K棒高低點預測策略

## 核心思想

使用過去20根K棒的統計特徵，預測下一根K棒的最高價和最低價

## 交易邏輯

### 基本流程
```
1. 當前時間: 14:45, 當前close = $100
2. 模型預測下一根K棒 (14:45-15:00):
   - 預測 high_pct = +0.35%  -> 預測最高價 = $100.35
   - 預測 low_pct = -0.25%   -> 預測最低價 = $99.75

3. 當下一根K棒開盤時 (14:45), 掛單:
   - 做多限價單: $99.75 (Maker 0.02%)
   - 做空限價單: $100.35 (Maker 0.02%)

4. 成交後設定:
   - 若做多成交 @ $99.75:
     止盈 = $100.35 (預測高點)
     止損 = $99.55 (預測低點 - 0.2%)
   
   - 若做空成交 @ $100.35:
     止盈 = $99.75 (預測低點)
     止損 = $100.55 (預測高點 + 0.2%)
```

### 優勢
- 雙向掛限價單，都是 Maker 費率 (0.02%)
- 利用K棒內部波動
- 不需要預測方向
- 風報比清晰

## 快速開始

### 1. 訓練模型

```bash
python test_nextbar_prediction.py
```

預期輸出:
```
OOS測試結果:
  HIGH 預測誤差: 0.18%
  LOW 預測誤差: 0.16%
  區間預測誤差: 0.12%
```

### 2. 在代碼中使用

```python
from data.huggingface_loader import HuggingFaceKlineLoader
from models.train_nextbar_model import NextBarModelTrainer

# 載入數據
loader = HuggingFaceKlineLoader()
df = loader.load_klines('BTCUSDT', '15m')

# 訓練模型
trainer = NextBarModelTrainer(model_type='xgboost')
metrics = trainer.train_with_oos(
    df,
    max_range_pct=0.015,  # 過濾 > 1.5% 的異常波動
    oos_days=30,
    test_size=0.2
)

# 保存模型
trainer.save_model('BTCUSDT', prefix='v1')

# 載入模型進行預測
trainer.load_model('models/saved/BTCUSDT_v1_nextbar_xgboost.pkl')

# 提取特徵
from utils.nextbar_feature_extractor import NextBarFeatureExtractor
extractor = NextBarFeatureExtractor()
df_with_features = extractor.extract_features(df)

# 預測最後一根K棒
X = df_with_features[trainer.actual_feature_columns].iloc[-1:].fillna(0)
pred_high_pct = trainer.model_high.predict(X)[0]
pred_low_pct = trainer.model_low.predict(X)[0]

current_close = df.iloc[-1]['close']
pred_high = current_close * (1 + pred_high_pct)
pred_low = current_close * (1 + pred_low_pct)

print(f"當前價格: ${current_close}")
print(f"預測下一根K棒:")
print(f"  最高: ${pred_high:.2f} ({pred_high_pct*100:+.2f}%)")
print(f"  最低: ${pred_low:.2f} ({pred_low_pct*100:+.2f}%)")
```

## 特徵說明

### 核心特徵 (55+)

1. **歷史振幅統計**
   - 過去5/10/20根K棒的平均振幅
   - 振幅標準差
   - 當前振幅 vs 歷史振幅比率

2. **K棒內部結構**
   - 上影線/下影線比例
   - 實體比例
   - 過去平均上下影線

3. **ATR 相關**
   - ATR 百分比
   - ATR 比率 (vs 20根平均)
   - ATR 變化率 (1根/3根)

4. **價格位置**
   - 當前價在近10/20/50根高低點的位置
   - 距離近期高低點的距離

5. **成交量**
   - 成交量比率
   - 成交量與振幅的關係

6. **動量**
   - 3/5/10根K棒動量
   - 動量加速度

7. **指標**
   - BB 寬度和位置
   - RSI 及其變化
   - MACD

8. **歷史百分比特徵**
   - 過去5/10/20根K棒的平均 high_pct
   - 過去5/10/20根K棒的平均 low_pct

## 模型評估指標

### MAE (Mean Absolute Error)
平均絕對誤差，越小越好

```
優秀: MAE < 0.20%
合格: MAE < 0.30%
需優化: MAE > 0.30%
```

### RMSE (Root Mean Squared Error)
均方根誤差，懲罰大誤差

### 區間預測誤差
預測區間寬度與實際區間寬度的差距

## 回測策略

### 策略A: 雙向掛單

```python
每根K棒:
1. 預測下一根K棒 high/low
2. 掛做多 @ 預測_low * 0.999
3. 掛做空 @ 預測_high * 1.001
4. 成交後設定止盈止損

風控:
- 過濾預測區間太寬的機會 (> 0.8%)
- 同時只持有1單
- 止損 = 預測邊界 + 0.2%
```

### 策略B: 單向選擇

```python
如果預測區間不對稱:
- high_pct > abs(low_pct) * 1.5 -> 只做多
- abs(low_pct) > high_pct * 1.5 -> 只做空

優勢:
- 減少雙邊成交風險
- 更高勝率
```

### 策略C: 跟隨方向

```python
結合短期動量:
- 若 momentum_3 > 0 且 預測_high_pct > 0.3%:
  只掛做空 @ 預測_high
  
- 若 momentum_3 < 0 且 預測_low_pct < -0.3%:
  只掛做多 @ 預測_low

原理:
- 上漨中在高點做空
- 下跌中在低點做多
```

## 預期表現

### 模型精度
```
HIGH 預測 MAE: 0.15-0.25%
LOW 預測 MAE: 0.15-0.25%
區間覆蓋率: 70-80%
```

### 交易績效 (策略A - 雙向掛單)
```
每筆平均利潤: 0.3-0.5%
成交率: 40-60%
勝率: 55-65%
每天交易: 10-20筆
```

## 優化方向

1. **特徵工程**
   - 增加多時間框架特徵 (1h, 4h)
   - 訂單簿特徵
   - 資金費率特徵

2. **模型改進**
   - 嘗試 Quantile Regression (預測P90/P10)
   - Ensemble 多個模型
   - 動態權重調整

3. **回測改進**
   - 動態掛單偏移 (根據ATR)
   - 動態止盈止損
   - 組合信號過濾

## 注意事項

1. **數據品質**
   - 過濾異常波動 (> 1.5%)
   - 足夠的訓練數據 (> 50000根)

2. **模型過拆合**
   - 監控 Train vs OOS 誤差差距
   - 如果差距 > 0.05%，模型過拆合

3. **市場環境**
   - 高波動時期誤差增大
   - 考慮動態調整掛單偏移

4. **滑點**
   - 預留 0.1% 滑點空間
   - 實際交易成本可能高於預期

## 總結

下一根K棒高低點預測是一種**直接、具體**的預測方法，相比於預測方向：

- 優點: 更精確的進出場點、雙向交易、Maker費率
- 缺點: 成交率低、需要較高模型精度

建議先在模擬環境中測試，確認 MAE < 0.25% 且回測績效穩定後，再考慮實盤交易。