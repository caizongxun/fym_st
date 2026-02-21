# BB+NW Swing Reversal Trading System v2.0

**Bollinger Bands + Nadaraya-Watson Swing Reversal Trading System**

A modular, institutional-grade AI trading system designed for 15m swing reversal trading.

---

## System Features

### Three-Layer Architecture

```
Trigger Layer (Event Trigger)  ->  Feature Layer (Features)  ->  AI Layer (Meta-Label)
     |                                 |                              |
  BB + NW                         ADX + CVD                     LightGBM
  Touch Detection                 Filter Features               Judge Reversal
```

### Core Advantages

1. **No Future Function (No Repaint)**
   - Nadaraya-Watson uses rolling window calculation
   - Backtest data = Live data

2. **Event-Driven Sampling**
   - Only triggers when touching BB/NW channels
   - Saves 85-98% computational resources

3. **Two Protection Mechanisms**
   - Prevent trend crush (ADX + HTF EMA)
   - Detect liquidity sweeps (CVD divergence + VWWA)

4. **Single Powerful Model**
   - No need for model ensemble
   - LightGBM built-in ensemble learning

---

## Modular Architecture

### Directory Structure

```
fym_st/
├── trading_system/
│   ├── config/              # Configuration constants
│   │   └── constants.py
│   ├── core/                # Core engine modules
│   │   ├── feature_engineering.py
│   │   ├── event_filter.py
│   │   ├── data_loader.py
│   │   ├── model_trainer.py
│   │   ├── labeling.py
│   │   └── backtester.py
│   ├── workflows/           # Workflow orchestration
│   │   └── training_workflow.py
│   ├── gui/                 # User interface
│   │   ├── utils/
│   │   │   └── ui_components.py
│   │   └── pages/
│   └── app_main.py
│
├── models/                  # Saved models
├── data/                    # Data cache
├── scripts/                 # Utility scripts
├── README.md
├── ARCHITECTURE.md          # Detailed architecture docs
└── requirements.txt
```

### Key Modules

#### 1. FeatureEngineer (feature_engineering.py)

```python
from core import FeatureEngineer

fe = FeatureEngineer()

# Build 15m features (BB + NW + ADX + CVD)
df_15m = fe.build_features(
    df,
    include_microstructure=True,   # CVD, VWWA
    include_nw_envelope=True,       # NW Envelope
    include_adx=True,               # ADX Trend Strength
    include_bounce_features=False   # Add after MTF
)

# MTF merge
df_mtf = fe.merge_and_build_mtf_features(df_15m, df_1h)

# Add swing reversal features
df_mtf = fe.add_bounce_confluence_features(df_mtf)
```

**Feature List** (~80-100 features):
- BB Bands: `bb_middle`, `bb_upper`, `bb_lower`, `bb_width_pct`, `bb_position`
- NW Envelope: `nw_middle`, `nw_upper`, `nw_lower`, `nw_width_pct`
- ADX Trend: `adx`, `plus_di`, `minus_di`
- CVD Flow: `cvd_10`, `cvd_20`, `cvd_norm_10`, `divergence_score_10`
- VWWA: `vwwa_buy_signal`, `lower_wick_size`
- Reversal: `bb_pierce_lower`, `sweep_divergence_buy`, `trend_crush_risk_15m`
- MTF (1h): All features with `_1h` suffix

#### 2. BBNW_BounceFilter (event_filter.py)

```python
from core.event_filter import BBNW_BounceFilter

filter = BBNW_BounceFilter(
    use_bb=True,                # Enable BB trigger
    use_nw=True,                # Enable NW trigger
    min_pierce_pct=0.001,       # 0.1% tolerance
    require_volume_surge=False  # Don't require volume surge
)

df_filtered = filter.filter_events(df_mtf)
# Output: is_long_setup, is_short_setup, touch_type
```

**Filter Results**:
- Raw data: 10,000 bars
- Filtered: 500-1500 bars (5-15%)
- Only keeps extreme touch events

#### 3. TrainingWorkflow (workflows/training_workflow.py)

```python
from workflows import TrainingWorkflow, TrainingConfig

# Configure
config = TrainingConfig(
    symbol="BTCUSDT",
    use_2024_only=True,
    nw_h=8.0,
    nw_mult=3.0,
    tp_multiplier=3.0,
    sl_multiplier=1.0,
    max_hold_bars=60
)

# Run workflow
workflow = TrainingWorkflow(config)
result = workflow.run()

if result.success:
    print(f"Model: {result.model_path}")
    print(f"AUC: {result.metrics['cv_auc_mean']:.3f}")
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Main Dependencies**:
- `streamlit` - GUI interface
- `pandas`, `numpy` - Data processing
- `lightgbm` - AI model
- `plotly` - Visualization
- `python-binance` - Binance API
- `datasets` - HuggingFace data

### 2. Launch System

```bash
cd trading_system
streamlit run app_main.py
```

Browser will open: `http://localhost:8501`

### 3. Train First Model

1. **Click sidebar**: Model Training

2. **Configure parameters**:
   - Symbol: BTCUSDT
   - Data source: HuggingFace (fast)
   - Use 2024 data only: Yes
   - NW indicator: h=8.0, mult=3.0
   - BB/NW trigger: Enable all
   - TP/SL: 3.0 / 1.0
   - Max hold: 60 bars (15 hours)

3. **Click**: Start Training

4. **Wait**: 10-15 minutes

### 4. Run Backtest

1. **Click sidebar**: Backtesting Analysis

2. **Select model**: Recently trained model

3. **Configure parameters**:
   - Test period: 2024 Full Year (OOS)
   - Probability threshold: 0.60
   - Initial capital: 10,000 USDT
   - Position size: 10%
   - Exit strategy: Dynamic trailing

4. **Click**: Execute Backtest

---

## Performance Metrics

### Expected Performance

| Metric | Target | Healthy Range |
|--------|--------|---------------|
| Win Rate | 55-65% | 50-70% |
| Risk/Reward | 2.5:1 | 2.0:1 - 4.0:1 |
| Profit Factor | 1.8+ | 1.5+ |
| Max Drawdown | < 25% | < 30% |
| Annual ROI | 30%+ | 20%+ |
| Monthly Signals | 15-30 | 10-40 |

### Top 10 Features (Importance)

1. `sweep_divergence_buy` - CVD divergence score
2. `trend_crush_risk_1h` - 1h trend risk
3. `bb_pierce_lower` - BB lower band pierce depth
4. `vwwa_buy_signal` - Lower wick absorption
5. `adx` - Trend strength
6. `cvd_norm_10` - 10-period normalized CVD
7. `nw_pierce_lower` - NW lower band pierce depth
8. `bb_squeeze_ratio` - BB compression ratio
9. `ema_50_dist_1h` - 1h EMA50 distance
10. `volume_ratio` - Volume surge multiplier

---

## Protection Mechanisms Explained

### 1. Prevent Trend Crush

**Problem**:
```
Price touches BB lower band during strong downtrend
-> Traditional: Go long (expect bounce)
-> Reality: Continues down, gets crushed
```

**Our Solution**:

1. **ADX Filter**:
   ```python
   if adx > 25 and adx_rising:
       # Strong trend, model outputs low probability (< 0.30)
   ```

2. **HTF EMA Filter**:
   ```python
   if abs(price - ema_50_1h) / ema_50_1h > 0.05:
       # Too far from 1h EMA50, strong trend
       # trend_crush_risk_1h feature will be extreme
   ```

3. **Auto Learning**:
   - LightGBM learns: When `adx > 30` AND `trend_crush_risk_1h > 0.05`, lower band touches mostly result in LOSS
   - Model automatically assigns low probability

### 2. Detect Liquidity Sweeps

**Problem**:
```
Institutions use long lower wick to pierce lower band
-> Retail stop losses triggered
-> Institutions accumulate
-> Price rallies
```

**Our Solution**:

1. **CVD Divergence Detection**:
   ```python
   # Price makes new low 5%, but CVD is positive
   divergence_score = cvd_norm_10 - price_pct_10
   # divergence_score > 0.5 -> Institution buying
   ```

2. **VWWA Absorption**:
   ```python
   lower_wick_ratio = lower_wick / body_size
   vwwa_buy_signal = lower_wick_ratio * volume_ratio
   # vwwa_buy_signal > 2.0 -> Large liquidity absorbed
   ```

3. **Combined Judgment**:
   ```python
   if bb_pierce_lower > 0.005 and \
      sweep_divergence_buy > 0 and \
      vwwa_buy_signal > 2.0:
       # Perfect liquidity sweep signal
       # Model outputs high probability (> 0.75)
   ```

---

## Code Examples

### Complete Training Flow

```python
from core import (
    CryptoDataLoader, FeatureEngineer, 
    TripleBarrierLabeling, ModelTrainer
)
from core.event_filter import BBNW_BounceFilter

# 1. Load data
loader = CryptoDataLoader()
df_15m = loader.load_klines('BTCUSDT', '15m')
df_1h = loader.load_klines('BTCUSDT', '1h')

# 2. Build features
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

# 3. MTF merge
df_mtf = fe.merge_and_build_mtf_features(df_15m_features, df_1h_features)
df_mtf = fe.add_bounce_confluence_features(df_mtf)

# 4. Event filter
filter = BBNW_BounceFilter(
    use_bb=True,
    use_nw=True,
    min_pierce_pct=0.001
)
df_filtered = filter.filter_events(df_mtf)

print(f"Filter result: {len(df_mtf)} -> {len(df_filtered)} ({len(df_filtered)/len(df_mtf)*100:.1f}%)")

# 5. Label
labeler = TripleBarrierLabeling(
    tp_multiplier=3.0,
    sl_multiplier=1.0,
    max_hold_bars=60
)
df_labeled = labeler.create_labels(df_filtered)

# 6. Train
trainer = ModelTrainer()
metrics = trainer.train(
    df_labeled,
    model_type='lightgbm',
    cv_folds=5,
    early_stopping_rounds=50
)

print(f"CV AUC: {metrics['cv_auc_mean']:.3f}")
print(f"CV Accuracy: {metrics['cv_accuracy_mean']:.3f}")

# 7. Save
trainer.save_model('BTCUSDT_15m_BB_NW_Bounce_v1.pkl')
```

### Real-time Prediction

```python
# Load model
trainer = ModelTrainer()
trainer.load_model('BTCUSDT_15m_BB_NW_Bounce_v1.pkl')

# Get latest data
df_latest = loader.fetch_latest_klines('BTCUSDT', '15m', days=1)

# Build features + filter
df_features = fe.build_features(df_latest, include_nw_envelope=True, include_adx=True)
df_filtered = filter.filter_events(df_features)

if len(df_filtered) > 0:
    # Predict
    probs = trainer.predict_proba(df_filtered)
    
    # Only keep high probability signals
    df_filtered['prob'] = probs
    signals = df_filtered[df_filtered['prob'] >= 0.60]
    
    print(f"Found {len(signals)} trading signals!")
    print(signals[['open_time', 'close', 'is_long_setup', 'prob']])
else:
    print("No touch events")
```

---

## Remove Emojis (Optional)

If you prefer UI without emojis, run:

```bash
python scripts/remove_emojis.py
```

This will automatically remove all emojis from:
- GUI pages
- Core modules
- Workflows
- Config files

---

## Important Disclaimer

1. **Risk Warning**: Cryptocurrency trading carries extreme risk and may result in total loss of capital
2. **No Guarantee**: This system provides no guarantee of profitability
3. **Educational Purpose**: For research and learning only
4. **Your Responsibility**: All trading decisions are at your own risk
5. **Recommendation**: Test thoroughly on demo account first

---

## Resources

### Academic Papers
- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) - Marcos Lopez de Prado
- [Machine Learning for Algorithmic Trading](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715) - Stefan Jansen

### Technical Docs
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Triple Barrier Method](https://mlfinlab.readthedocs.io/en/latest/labeling/tb_meta_labeling.html)
- [Nadaraya-Watson Estimator](https://en.wikipedia.org/wiki/Kernel_regression)
- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed system architecture

### Market Data
- [Binance API](https://binance-docs.github.io/apidocs/)
- [HuggingFace Crypto Datasets](https://huggingface.co/datasets)

---

## Contact

- **Project**: [GitHub Repository](https://github.com/caizongxun/fym_st)
- **Issues**: [Report Issues](https://github.com/caizongxun/fym_st/issues)

---

## License

MIT License

Copyright (c) 2026 BB+NW Swing Trading System

---

<p align="center">
  <b>BB+NW Swing Reversal System v2.0</b><br>
  Built with precision for Swing Traders<br>
  <i>"Trade Smarter, Not Harder"</i>
</p>