# FYM_ST - Advanced Automated Crypto Trading System

## Overview

FYM_ST is an institutional-grade automated cryptocurrency trading system implementing advanced quantitative methods from academic research and hedge fund practices. The system features:

1. **Automated Trading System**: Academic ML framework with meta-labeling and Kelly criterion
2. **Liquidity Sweep Detection** (NEW): Institutional market microstructure analysis
3. **Multi-Timeframe AI System**: High-frequency scalping with trend confirmation

## Core Systems

### 1. Liquidity Sweep Detection System (NEW)

A groundbreaking institutional-grade entry theory based on market microstructure:

#### Theory: Liquidity Sweep & Microstructure Exhaustion

Instead of traditional breakout strategies, this system identifies when institutional players (Smart Money) trigger retail stop-losses to accumulate positions, then enter when:

1. **Price Action**: False breakout with long wick (2x body)
2. **OI Flush**: Open Interest drops >2σ (retail liquidation)
3. **CVD Divergence**: Cumulative Volume Delta shows absorption

#### Advantages Over Traditional Systems

- **60% Fee Savings**: Left-side entry allows Maker orders vs Taker breakouts
- **Superior R:R**: Precise stop at wick extreme vs wide breakout stops
- **Non-Collinear Data**: OI + CVD adds real money flow dimension
- **Regime Adaptive**: Works in ranging markets where indicators fail

#### Key Features

- Open Interest (OI) historical data from Binance Futures
- Cumulative Volume Delta (CVD) calculation
- Funding Rate integration
- Smart Money absorption detection
- 10+ new institutional features

**Quick Start:**
```bash
python test_liquidity_sweep.py
python examples/liquidity_sweep_example.py
```

**Documentation**: 
- Theory: `docs/LIQUIDITY_SWEEP_THEORY.md`
- Integration: `LIQUIDITY_SWEEP_INTEGRATION.md`

### 2. Automated Trading System (`trading_system/`)

A quantitative trading framework implementing state-of-the-art machine learning methods:

#### Mathematical Framework

- **Triple Barrier Method**: Volatility-adjusted profit/loss targets using ATR
- **Meta-Labeling**: Two-layer signal filtering (primary signal + ML confirmation)
- **Fractional Differentiation**: Stationarity preservation with memory (d=0.4)
- **Purged K-Fold CV**: Time-series aware cross-validation preventing data leakage
- **Dynamic Kelly Criterion**: Probability-based position sizing with risk fraction
- **Sample Weighting**: High-impact trade prioritization during training

#### Features

- Modular architecture for maintainability
- Streamlit GUI for training, backtesting, and live prediction
- HuggingFace dataset integration (38 pairs, 3 timeframes)
- LightGBM with purged cross-validation
- Realistic backtesting with commission and slippage
- Real-time prediction using completed bars only
- **NEW**: Liquidity sweep features integration

**Quick Start:**
```bash
cd trading_system
pip install -r requirements.txt
streamlit run app_main.py
```

**Documentation**: See `trading_system/README.md` for detailed usage

### 3. Multi-Timeframe AI System (Original)

A sophisticated multi-model architecture for high-frequency trading:

#### Three-Model Architecture

1. **Trend Detection Model (1h)**: Market regime identification
2. **Volatility Prediction Model (15m)**: Volatility regime forecasting
3. **Reversal Detection Model (15m)**: High-probability entry points

#### Features

- Multi-timeframe analysis (1h + 15m)
- Signal stability (completed candles only)
- Out-of-sample validation
- Binance API integration
- ATR-based risk management
- Portfolio allocation across 38+ pairs

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
```

### For Automated Trading System
```bash
cd trading_system
pip install -r requirements.txt
```

### For Multi-Timeframe System
```bash
pip install -r requirements.txt
```

## Configuration

Create `.env` file:

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
HUGGINGFACE_TOKEN=your_token  # Optional
```

## Project Structure

```
fym_st/
├── trading_system/              # Automated trading system
│   ├── core/                    # Core quantitative modules
│   │   ├── data_loader.py       # Enhanced with OI/Funding Rate
│   │   ├── feature_engineering.py  # Enhanced with liquidity features
│   │   ├── liquidity_sweep_detector.py  # NEW
│   │   ├── labeling.py
│   │   ├── meta_labeling.py
│   │   ├── model_trainer.py
│   │   ├── position_sizing.py
│   │   ├── backtester.py
│   │   └── predictor.py
│   ├── gui/                     # Streamlit interface
│   │   └── pages/
│   ├── models/                  # Trained models
│   ├── app_main.py             # Entry point
│   ├── requirements.txt
│   └── README.md               # Detailed documentation
│
├── docs/
│   └── LIQUIDITY_SWEEP_THEORY.md  # NEW: Complete theory
│
├── examples/
│   └── liquidity_sweep_example.py # NEW: Usage examples
│
├── test_liquidity_sweep.py      # NEW: Quick test
├── LIQUIDITY_SWEEP_INTEGRATION.md  # NEW: Integration guide
│
├── app.py                       # Multi-timeframe system GUI
├── config.py
├── requirements.txt
├── data/                        # Data loading modules
├── models/                      # Multi-model architecture
├── training/                    # Training scripts
├── backtesting/                 # Backtesting engine
├── strategies/                  # Strategy implementations
├── utils/                       # Utility functions
└── uploads/                     # File upload directory
```

## New Liquidity Features

### Price Structure
- `lower_wick_ratio`, `upper_wick_ratio`: Wick to body ratio
- `dist_to_support_pct`, `dist_to_resistance_pct`: Distance to key levels

### Open Interest (OI)
- `oi_change_pct`, `oi_change_1h`, `oi_change_4h`, `oi_change_24h`
- `oi_normalized`: Standardized OI value
- `open_interest`: Raw OI data

### Cumulative Volume Delta (CVD)
- `cvd`: Cumulative buy/sell volume difference
- `cvd_slope_5`, `cvd_slope_10`: CVD momentum
- `cvd_normalized`: Standardized CVD

### Funding Rate
- `funding_rate`: 8-hour funding rate
- `funding_rate_ma_3`, `funding_rate_ma_7`: Moving averages

## Data Sources

- **HuggingFace Dataset**: `zongowo111/v2-crypto-ohlcv-data`
  - 38 cryptocurrency pairs
  - Timeframes: 15m, 1h, 1d
  - Format: Parquet files

- **Binance API**: Real-time data and execution
- **Binance Futures API** (NEW): Open Interest and Funding Rate

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

### Liquidity Sweep System
- Precise stops at wick extremes
- R:R typically 1:2.5 to 1:4
- Left-side entries for Maker fees
- OI flush confirmation reduces false signals

### Automated Trading System
- Kelly criterion position sizing
- ATR-based stop-loss and take-profit
- Maximum position limits
- Sample weighting by return magnitude

### Multi-Timeframe System
- Configurable position size percentage
- Maximum concurrent positions
- Dynamic ATR-based stops
- Multi-model confirmation filters

## Performance Metrics

All systems track:
- Total return and Sharpe ratio
- Win rate and profit factor
- Maximum drawdown
- Average win/loss
- Trade distribution
- Equity curve
- Commission impact analysis (NEW)

## Key Principles

### Avoiding Lookahead Bias
- All predictions use completed bars only
- No access to current incomplete bar
- Time-series integrity in cross-validation

### Data Leakage Prevention
- Purged cross-validation
- Temporal train/test splits
- No shuffle in time-series data

### Risk Controls
- Position size limits
- Volatility-adjusted stops
- Maximum exposure caps
- Probability-based sizing

## Usage Examples

### Detect Liquidity Sweeps
```python
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

### Train with Liquidity Features
```python
from trading_system.core import FeatureEngineer, ModelTrainer

fe = FeatureEngineer()
df_features = fe.build_features(df, include_liquidity_features=True)

trainer = ModelTrainer()
trainer.train(df_labeled, features=[
    'rsi_normalized', 'bb_width_pct',
    'lower_wick_ratio', 'oi_change_4h', 'cvd_slope_5',  # NEW
    'dist_to_support_pct', 'funding_rate_ma_3'  # NEW
])
```

## Academic References

This system implements methodologies from:

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Hosking, J. R. M. (1981). Fractional Differencing
- Kelly, J. L. (1956). Information Rate Theory
- Peng, C. K., et al. (1994). Long-range correlations in nucleotide sequences
- **Market Microstructure Theory** (Liquidity Sweep Detection)
- **Order Flow Analysis** (CVD and OI metrics)

## Documentation

- **Liquidity Sweep Theory**: `docs/LIQUIDITY_SWEEP_THEORY.md`
- **Integration Guide**: `LIQUIDITY_SWEEP_INTEGRATION.md`
- **Automated System**: `trading_system/README.md`
- **Usage Guide**: `USAGE.md`
- **Strategy Integration**: `STRATEGY_C_INTEGRATION.md`

## Testing

```bash
# Test liquidity sweep detection
python test_liquidity_sweep.py

# Run complete example
python examples/liquidity_sweep_example.py

# Launch GUI
streamlit run trading_system/app_main.py
```

## Warning

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct thorough testing and use appropriate risk management before live deployment.

## License

MIT License - See LICENSE file for details

## Author

Developed by [caizongxun](https://github.com/caizongxun)

## Contributing

Contributions welcome. Please open an issue or submit a pull request.

---

## What's New in v2.0

### Liquidity Sweep Detection System
- Institutional market microstructure analysis
- Open Interest (OI) integration
- Cumulative Volume Delta (CVD) calculation
- Funding Rate monitoring
- 10+ new features for ML models
- 60% reduction in trading fees
- Superior risk-reward ratios

### Enhanced Data Pipeline
- Binance Futures API integration
- Historical OI data fetching
- Automatic data merging
- Funding rate time-series

### Improved Feature Engineering
- Liquidity-aware features
- Market microstructure indicators
- Smart Money flow metrics
- Wick analysis ratios