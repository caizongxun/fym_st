# 更新日誌

## v2.0.0 - 流動性掃蕩系統 (2026-02-20)

### 新增功能

#### 流動性掃蕩偵測系統
基於機構級市場微觀結構的全新進場理論:

- **LiquiditySweepDetector**: 核心偵測引擎
  - 價格行為: 假突破 + 長影線 (2x 實體)
  - OI 銳減: 未平倉量下降 > 2σ
  - CVD 背離: 成交量差背離

- **擴充的 CryptoDataLoader**:
  - 新增 `fetch_open_interest()`: 獲取 Binance Futures OI 數據
  - 新增 `fetch_funding_rate()`: 獲取資金費率歷史
  - 自動合併 OI/Funding Rate 到 K 線數據

- **擴充的 FeatureEngineer**:
  - 10+ 新流動性特徵
  - OI 相關: `oi_change_pct`, `oi_change_4h`, `oi_normalized`
  - CVD 相關: `cvd`, `cvd_slope_5`, `cvd_normalized`
  - 影線分析: `lower_wick_ratio`, `upper_wick_ratio`
  - 位置特徵: `dist_to_support_pct`, `dist_to_resistance_pct`

- **Streamlit GUI 新頁面**:
  - 流動性掃蕩分析頁面
  - 視覺化 K 線 + 信號標記
  - OI 變化圖表
  - CVD 走勢圖
  - ML 模型整合過濾

### 優勢

- **60% 手續費節省**: 左側進場使用 Maker 費率
- **精確停損**: 在影線尖端,R:R 1:2.5-1:4
- **非共線性數據**: 真實資金流維度 (OI + CVD)
- **適應性**: 盤整期也能運作

### 文檔

- `docs/LIQUIDITY_SWEEP_THEORY.md`: 完整理論說明
- `LIQUIDITY_SWEEP_INTEGRATION.md`: 整合指南
- 更新 `README.md`: 加入新系統介紹

### 測試和示例

- `test_liquidity_sweep.py`: 快速測試腳本
- `examples/liquidity_sweep_example.py`: 完整使用示例

### 使用方式

```bash
# 啟動 GUI
cd trading_system
streamlit run app_main.py

# 選擇 "流動性掃蕩分析" 頁面
```

```python
# 程式化使用
from trading_system.core import CryptoDataLoader, LiquiditySweepDetector

loader = CryptoDataLoader()
df = loader.fetch_latest_klines(
    'BTCUSDT', '1h', days=90,
    include_oi=True, include_funding=True
)

detector = LiquiditySweepDetector()
df_sweep = detector.detect_liquidity_sweep(df, direction='lower')
signals = df_sweep[df_sweep['sweep_lower_signal']]
```

### 修復

- 修正回測頁面特徵不匹配問題
- 修正 `exclude_cols` 過度排除特徵
- 增加機率分布診斷資訊

---

## v1.x - 原有功能

### 核心系統

- Automated Trading System (自動交易系統)
- Multi-Timeframe AI System (多時間框架 AI 系統)
- Triple Barrier Labeling (三重障礙標籤)
- Meta-Labeling (元標籤)
- Kelly Criterion Position Sizing (Kelly 標準仓位管理)
- Purged K-Fold CV (清除 K 折交叉驗證)
- Fractional Differentiation (分數差分)

### 特徵工程

- ATR, Bollinger Bands, RSI, MACD
- EMA 特徵, 成交量特徵
- 價格行為特徵
- 波動率特徵

### GUI 功能

- 模型訓練頁面
- 回測分析頁面
- 即時預測頁面
- 策略優化頁面
- 機率校準分析頁面
- 控制台頁面

---

## 計劃中功能

### v2.1 (計劃中)

- [ ] 多符號流動性掃蕩掃描
- [ ] 實時 OI 監控警示
- [ ] 流動性熱力圖
- [ ] 機構訂單流分析
- [ ] 整合到實盤交易系統

### v2.2 (計劃中)

- [ ] 自動參數優化
- [ ] 機器學習動態偵測
- [ ] 多時間框架流動性分析
- [ ] 資金費率異常警示

---

## 貢獻

歡迎提交 Issue 和 Pull Request!

聯繫: caizongxun@github

---

**免責聲明**: 本系統僅供教育和研究用途。加密貨幣交易存在高風險,請謹慎使用。