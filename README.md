# FYM_ST - Advanced Automated Crypto Trading System

## 🚀 Two-Phase Development Strategy

**An ambitious professional approach: Push Alpha to the limit through microstructure (Phase 1), then validate with rigorous OOS blind testing (Phase 2)**

### Phase 1: Microstructure Feature Expansion (第一階段:微觀結構特徵擴充)

**Goal**: Maximize model Alpha through institutional-grade order flow analysis

Leveraging Binance's native K-line data (`taker_buy_base_asset_volume`), we've integrated **8 core microstructure features**:

1. **net_volume**: Net taker volume delta
2. **cvd_10**: Short-term Cumulative Volume Delta
3. **cvd_20**: Mid-term CVD
4. **cvd_norm_10**: Normalized CVD (cross-asset comparable)
5. **divergence_score_10**: Price-CVD divergence score **[Core Feature]**
6. **upper_wick_ratio**: Upper wick to body ratio
7. **lower_wick_ratio**: Lower wick to body ratio (liquidity sweep)
8. **order_flow_imbalance**: Order flow imbalance ratio (-1 to +1)

**Expected Results**:
- AUC: 0.60+
- Precision @ 0.60 threshold: 58-60%
- Feature Importance: `divergence_score_10` in Top 5

📚 **Documentation**: [`PHASE1_MICROSTRUCTURE_TRAINING.md`](PHASE1_MICROSTRUCTURE_TRAINING.md)

### Phase 2: Out-of-Sample Blind Test (第二階段:樣本外盲測)

**Goal**: Validate with strictest OOS data the model has never seen

Once Phase 1 completes (AUC > 0.60, Precision > 58%), proceed to rigorous validation:

**Testing Criteria**:
- **OOS Data**: 90+ days of unseen data (e.g., 2023 H2 or future 2026 data)
- **Initial Capital**: $10,000
- **Risk per Trade**: 1.5-2.0%
- **Probability Threshold**: 0.60
- **TP/SL**: 3.0 ATR / 1.5 ATR
- **Fees**: Maker 0.02%, Taker 0.06%, Slippage 0.05%

**Success Criteria**:
- ☑️ Total Return > 0%
- ☑️ Win Rate > 50%
- ☑️ Profit Factor > 1.5
- ☑️ Avg Win > Avg Loss
- ☑️ Max Drawdown < 20%

If successful, the system is production-ready for live trading with Binance API.

📚 **Documentation**: [`PHASE2_OOS_VALIDATION.md`](PHASE2_OOS_VALIDATION.md)

---

## Overview

FYM_ST is an institutional-grade automated cryptocurrency trading system implementing advanced quantitative methods from academic research and hedge fund practices. The system features:

1. **Automated Trading System**: Academic ML framework with meta-labeling and Kelly criterion
2. **Microstructure Analysis** (NEW): Institutional order flow and CVD divergence detection
3. **Liquidity Sweep Detection**: Market microstructure exhaustion identification
4. **Multi-Timeframe AI System**: High-frequency scalping with trend confirmation

## Core Systems

### 1. Institutional Microstructure Analysis (NEW - Phase 1)

**Revolutionary approach using native Binance K-line data to capture Smart Money intentions**

#### Theory: Order Flow & CVD Divergence

Institutional players leave footprints in the order book. By analyzing the difference between taker buy/sell volume (CVD), we can detect:

- **Bottom Accumulation**: Price drops but CVD rises (buyers absorbing sell pressure)
- **Top Distribution**: Price rises but CVD drops (sellers distributing to buyers)
- **Liquidity Sweeps**: Long wicks with OI flush (stop hunt)

#### 8 Core Microstructure Features

```python
# All calculated from Binance native data
taker_buy_volume = df['taker_buy_base_asset_volume']
taker_sell_volume = df['volume'] - taker_buy_volume

net_volume = taker_buy_volume - taker_sell_volume
cvd_10 = net_volume.rolling(10).sum()
cvd_norm_10 = cvd_10 / volume.rolling(10).sum()

price_pct_10 = close.pct_change(10)
divergence_score_10 = cvd_norm_10 - price_pct_10  # Key!
```

**Key Insight**: When `divergence_score_10` is highly positive (price down + CVD up), institutions are accumulating at the bottom.

🔥 **Quick Start Phase 1**:
```bash
cd trading_system
streamlit run app_main.py
# Select "模型訓練" and enable 微觀結構特徵
```

### 2. Liquidity Sweep Detection System

A complementary system focusing on OI and funding rate analysis:

#### Theory: Liquidity Sweep & Microstructure Exhaustion

Instead of traditional breakout strategies, this system identifies when institutional players (Smart Money) trigger retail stop-losses to accumulate positions:

1. **Price Action**: False breakout with long wick (2x body)
2. **OI Flush**: Open Interest drops >2σ (retail liquidation)
3. **CVD Divergence**: Cumulative Volume Delta shows absorption

#### Advantages

- **60% Fee Savings**: Left-side entry allows Maker orders vs Taker breakouts
- **Superior R:R**: Precise stop at wick extreme vs wide breakout stops
- **Non-Collinear Data**: OI + CVD adds real money flow dimension
- **Regime Adaptive**: Works in ranging markets where indicators fail

**Quick Start:**
```bash
python test_liquidity_sweep.py
python examples/liquidity_sweep_example.py
```

**Documentation**: 
- Theory: `docs/LIQUIDITY_SWEEP_THEORY.md`
- Integration: `LIQUIDITY_SWEEP_INTEGRATION.md`

### 3. Automated Trading System (`trading_system/`)

A quantitative trading framework implementing state-of-the-art machine learning methods:

#### Mathematical Framework

- **Triple Barrier Method**: Volatility-adjusted profit/loss targets using ATR
- **Meta-Labeling**: Two-layer signal filtering (primary signal + ML confirmation)
- **Fractional Differentiation**: Stationarity preservation with memory (d=0.4)
- **Purged K-Fold CV**: Time-series aware cross-validation preventing data leakage
- **Dynamic Kelly Criterion**: Probability-based position sizing with risk fraction
- **Microstructure Features** (NEW): 8 institutional order flow features

#### Features

- Modular architecture for maintainability
- Streamlit GUI for training, backtesting, and live prediction
- HuggingFace dataset integration (38 pairs, 3 timeframes)
- LightGBM with purged cross-validation
- Realistic backtesting with commission and slippage
- Real-time prediction using completed bars only

**Quick Start:**
```bash
cd trading_system
pip install -r requirements.txt
streamlit run app_main.py
```

### 4. Multi-Timeframe AI System (Original)

A sophisticated multi-model architecture for high-frequency trading:

#### Three-Model Architecture

1. **Trend Detection Model (1h)**: Market regime identification
2. **Volatility Prediction Model (15m)**: Volatility regime forecasting
3. **Reversal Detection Model (15m)**: High-probability entry points

**Quick Start:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Installation

```bash
git clone https://github.com/caizongxun/fym_st.git
cd fym_st
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
cd trading_system
pip install -r requirements.txt
```

## Configuration

Create `.env` file:

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
HUGGINGFACE_TOKEN=your_token  # Optional
```

## New Microstructure Features (Phase 1)

### Order Flow Features (Native Binance Data)

1. **net_volume**: Taker buy - taker sell volume
2. **cvd_10**: 10-period cumulative volume delta
3. **cvd_20**: 20-period CVD
4. **cvd_norm_10**: Normalized CVD (cross-asset comparable)
5. **divergence_score_10**: **Price-CVD divergence score** [Core]
6. **upper_wick_ratio**: Upper wick / body size
7. **lower_wick_ratio**: Lower wick / body size (liquidity sweep)
8. **order_flow_imbalance**: (Buy - Sell) / (Buy + Sell)

### OI & Funding Features (Binance Futures API)

- `oi_change_pct`, `oi_change_4h`: Open Interest changes
- `oi_normalized`: Standardized OI
- `funding_rate_ma_3`: Funding rate moving average
- `dist_to_support_pct`: Distance to support level

## Data Sources

- **Binance Spot API**: Real-time OHLCV + taker volume
- **Binance Futures API**: Open Interest + Funding Rate
- **HuggingFace Dataset**: `zongowo111/v2-crypto-ohlcv-data` (38 pairs, 3 timeframes)

## Supported Trading Pairs

```
AAVEUSDT  ADAUSDT   ALGOUSDT  ARBUSDT   ATOMUSDT  AVAXUSDT
BALUSDT   BATUSDT   BCHUSDT   BNBUSDT   BTCUSDT   COMPUSDT
CRVUSDT   DOGEUSDT  DOTUSDT   ENJUSDT   ENSUSDT   ETCUSDT
ETHUSDT   FILUSDT   GALAUSDT  GRTUSDT   IMXUSDT   KAVAUSDT
LINKUSDT  LTCUSDT   MANAUSDT  MATICUSDT MKRUSDT   NEARUSDT
OPUSDT    SANDUSDT  SNXUSDT   SOLUSDT   SPELLUSDT UNIUSDT
XRPUSDT   ZRXUSDT
```

## Risk Management

### Phase 1 Training
- Triple Barrier Labeling: TP=3.0 ATR, SL=1.0 ATR
- Sample weighting by return magnitude
- Feature selection: Remove low-importance features

### Phase 2 Validation
- Initial capital: $10,000
- Risk per trade: 1.5-2.0%
- Probability threshold: 0.60
- Maker fee: 0.02%, Taker: 0.06%, Slippage: 0.05%

## Key Principles

### Avoiding Lookahead Bias
- All predictions use completed bars only
- No access to current incomplete bar
- Time-series integrity in cross-validation

### Data Leakage Prevention
- Purged cross-validation
- Temporal train/test splits
- No shuffle in time-series data

### Two-Phase Validation
- **Phase 1**: Maximize Alpha with microstructure features
- **Phase 2**: Strict OOS blind test before live trading

## Usage Examples

### Phase 1: Train with Microstructure Features

```python
from trading_system.core import (
    CryptoDataLoader, FeatureEngineer,
    TripleBarrierLabeling, ModelTrainer
)

# Load data
loader = CryptoDataLoader()
df = loader.fetch_latest_klines('BTCUSDT', '1h', days=365)

# Build features with microstructure
fe = FeatureEngineer()
df_features = fe.build_features(
    df,
    include_microstructure=True  # Enable Phase 1 features
)

# Label
labeling = TripleBarrierLabeling(tp=3.0, sl=1.0)
df_labeled = labeling.label(df_features)

# Train
trainer = ModelTrainer()
trainer.train(
    df_labeled,
    features=[
        'atr_pct', 'rsi_normalized', 'bb_width_pct',
        'net_volume', 'cvd_10', 'cvd_norm_10',
        'divergence_score_10',  # Core microstructure feature
        'lower_wick_ratio', 'order_flow_imbalance'
    ]
)
```

### Phase 2: OOS Validation

```python
from trading_system.core import Backtester

# Load OOS data (unseen by model)
df_oos = loader.fetch_latest_klines('BTCUSDT', '1h', days=90)
df_oos_features = fe.build_features(df_oos, include_microstructure=True)

# Predict
probabilities = trainer.predict_proba(df_oos_features[features])
df_oos_features['win_probability'] = probabilities
signals = df_oos_features[df_oos_features['win_probability'] >= 0.60]

# Backtest
backtester = Backtester(
    initial_capital=10000,
    risk_per_trade=0.015,
    leverage=10
)
results = backtester.run_backtest(signals, tp_multiplier=3.0, sl_multiplier=1.5)

print(f"Total Return: {results['statistics']['total_return']*100:.1f}%")
print(f"Win Rate: {results['statistics']['win_rate']*100:.1f}%")
print(f"Profit Factor: {results['statistics']['profit_factor']:.2f}")
```

## Academic References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Hosking, J. R. M. (1981). Fractional Differencing
- Kelly, J. L. (1956). Information Rate Theory
- **Market Microstructure Theory** (Order Flow Analysis)
- **Cumulative Volume Delta** (CVD) - Institutional footprint detection

## Documentation

### Phase 1 & 2
- **Phase 1 Training**: [`PHASE1_MICROSTRUCTURE_TRAINING.md`](PHASE1_MICROSTRUCTURE_TRAINING.md)
- **Phase 2 Validation**: [`PHASE2_OOS_VALIDATION.md`](PHASE2_OOS_VALIDATION.md)

### Systems
- **Liquidity Sweep Theory**: `docs/LIQUIDITY_SWEEP_THEORY.md`
- **Integration Guide**: `LIQUIDITY_SWEEP_INTEGRATION.md`
- **Quick Start**: `QUICKSTART_LIQUIDITY_SWEEP.md`
- **Automated System**: `trading_system/README.md`

## Testing

```bash
# Phase 1: Train with microstructure features
cd trading_system
streamlit run app_main.py  # GUI
# OR
python examples/train_with_microstructure.py  # Script

# Phase 2: OOS validation
python examples/oos_validation.py

# Liquidity sweep detection
python test_liquidity_sweep.py
```

## Warning

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. 

**Two-Phase Requirement**: Do NOT proceed to live trading without completing both Phase 1 (model training with AUC > 0.60) and Phase 2 (OOS validation with positive returns).

## License

MIT License - See LICENSE file for details

## Author

Developed by [caizongxun](https://github.com/caizongxun)

## Contributing

Contributions welcome. Please open an issue or submit a pull request.

---

## What's New in v2.1

### Two-Phase Professional Development Strategy
- **Phase 1**: Microstructure feature expansion (8 core order flow features)
- **Phase 2**: Rigorous OOS blind testing framework

### Institutional Microstructure Analysis
- Native Binance taker volume analysis
- CVD (Cumulative Volume Delta) calculation
- Price-CVD divergence detection (core alpha source)
- Liquidity sweep wick analysis
- Order flow imbalance metrics

### Enhanced Feature Engineering
- Optimized from 15+ to 8 core microstructure features
- Removed redundant features to prevent overfitting
- Cross-asset normalized metrics
- Stationarity-preserving rolling windows

### Production-Ready Validation
- Strict OOS testing protocol
- Real-world fee simulation (Maker/Taker/Slippage)
- Multiple success criteria (Return/Win Rate/Profit Factor)
- Live trading readiness checklist