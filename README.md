# FYM_ST - Advanced Automated Crypto Trading System

## Overview

FYM_ST is an institutional-grade automated cryptocurrency trading system implementing advanced quantitative methods from academic research and hedge fund practices. The system features two distinct trading engines:

1. **Automated Trading System** (New): Academic ML framework with meta-labeling and Kelly criterion
2. **Multi-Timeframe AI System**: High-frequency scalping with trend confirmation

## Core Systems

### 1. Automated Trading System (`trading_system/`)

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

**Quick Start:**
```bash
cd trading_system
pip install -r requirements.txt
streamlit run app_main.py
```

**Documentation**: See `trading_system/README.md` for detailed usage

### 2. Multi-Timeframe AI System (Original)

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
├── trading_system/              # NEW: Automated trading system
│   ├── core/                    # Core quantitative modules
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
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

## Data Sources

- **HuggingFace Dataset**: `zongowo111/v2-crypto-ohlcv-data`
  - 38 cryptocurrency pairs
  - Timeframes: 15m, 1h, 1d
  - Format: Parquet files

- **Binance API**: Real-time data and execution

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

Both systems track:
- Total return and Sharpe ratio
- Win rate and profit factor
- Maximum drawdown
- Average win/loss
- Trade distribution
- Equity curve

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

## Academic References

This system implements methodologies from:

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Hosking, J. R. M. (1981). Fractional Differencing
- Kelly, J. L. (1956). Information Rate Theory
- Peng, C. K., et al. (1994). Long-range correlations in nucleotide sequences

## Documentation

- **Automated System**: `trading_system/README.md`
- **Usage Guide**: `USAGE.md`
- **Strategy Integration**: `STRATEGY_C_INTEGRATION.md`

## Warning

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct thorough testing and use appropriate risk management before live deployment.

## License

MIT License - See LICENSE file for details

## Author

Developed by [caizongxun](https://github.com/caizongxun)

## Contributing

Contributions welcome. Please open an issue or submit a pull request.