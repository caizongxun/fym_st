# Automated Crypto Trading System

An academic-grade automated trading system for cryptocurrency markets, implementing state-of-the-art quantitative methods from leading hedge funds and research institutions.

## Core Features

### Mathematical Framework

- **Triple Barrier Method**: Rigorous target definition using volatility-adjusted barriers (ATR-based)
- **Fractional Differentiation**: Achieve stationarity while preserving memory properties (d=0.4)
- **Meta-Labeling**: Two-layer signal filtering to reduce false positives
- **Purged K-Fold Cross-Validation**: Eliminate data leakage in time series models
- **Dynamic Kelly Criterion**: Probability-based position sizing with configurable risk fraction
- **Sample Weighting**: Prioritize high-impact trades during model training

### System Architecture

```
trading_system/
├── core/                      # Core quantitative modules
│   ├── data_loader.py         # HuggingFace data integration
│   ├── feature_engineering.py # Technical indicators & transformations
│   ├── labeling.py            # Triple barrier labeling
│   ├── meta_labeling.py       # Primary signal generation
│   ├── model_trainer.py       # LightGBM with purged CV
│   ├── position_sizing.py     # Kelly criterion & risk management
│   ├── backtester.py          # Historical performance evaluation
│   └── predictor.py           # Real-time signal generation
├── gui/                       # Streamlit interface
│   └── pages/                 # Application pages
│       ├── dashboard_page.py
│       ├── training_page.py
│       ├── backtesting_page.py
│       └── live_prediction_page.py
├── models/                    # Trained model storage
└── app_main.py               # Application entry point
```

## Data Source

- **Provider**: HuggingFace Dataset
- **Dataset ID**: `zongowo111/v2-crypto-ohlcv-data`
- **Coverage**: 38 cryptocurrency pairs (BTCUSDT, ETHUSDT, etc.)
- **Timeframes**: 15m, 1h, 1d
- **Format**: Parquet files with OHLCV + volume metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Launch GUI Application

```bash
cd trading_system
streamlit run app_main.py
```

### Workflow

1. **Dashboard**: Preview data and system overview
2. **Training**: Train models with customizable parameters
3. **Backtesting**: Evaluate performance on historical data
4. **Live Prediction**: Generate real-time trading signals

### Training Configuration

- **TP Multiplier**: Take profit threshold (default: 2.5x ATR)
- **SL Multiplier**: Stop loss threshold (default: 1.5x ATR)
- **Max Holding Bars**: Maximum position duration (default: 24)
- **CV Splits**: Cross-validation folds (default: 5)
- **Purge Gap**: Time-series purge window (default: 24 bars)

### Position Sizing

$$f^* = \frac{bp - q}{b} \times \text{kelly fraction}$$

Where:
- $p$: Model win probability
- $q$: 1 - p
- $b$: Reward/risk ratio (TP/SL)
- Kelly fraction: Conservative multiplier (default: 0.5)

## Key Principles

### Avoiding Lookahead Bias

All predictions use **completed K-bars only**. Current incomplete bars are excluded to prevent data leakage during live trading.

### Time-Series Integrity

- Training/test splits respect temporal order
- Overlapping samples purged from training when in test window
- No shuffle during cross-validation

### Risk Management

- Per-trade position sizing via Kelly criterion
- Maximum position limits enforced
- Stop-loss and take-profit levels derived from market volatility (ATR)

## Performance Metrics

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Total Return**: Portfolio appreciation

## References

This system implements methodologies from:

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Hosking, J. R. M. (1981). Fractional Differencing
- Kelly, J. L. (1956). A New Interpretation of Information Rate

## License

MIT License - see LICENSE file

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct thorough testing and risk assessment before live deployment.