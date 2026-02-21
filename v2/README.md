# V2 Modular Trading System

## Overview
Modular automated trading system with feature engineering, machine learning models, and backtesting capabilities.

## Structure
```
v2/
├── feature_engineering.py  # Feature calculation module
├── data_loader.py          # Data acquisition module (TBD)
├── model.py                # ML model module (TBD)
├── strategy.py             # Trading strategy module (TBD)
├── backtest.py             # Backtesting engine (TBD)
└── README.md               # Documentation
```

## Feature Engineering Module

### FeatureEngineer Class
Calculates technical indicators based on 15-minute OHLCV data.

#### Parameters
- `bb_period`: Bollinger Bands period (default: 20)
- `bb_std`: Bollinger Bands standard deviation multiplier (default: 2)
- `lookback`: Lookback period for percentile calculation (default: 100)
- `pivot_left`: Left bars for pivot detection (default: 3)
- `pivot_right`: Right bars for pivot detection (default: 3)

#### Features Calculated

**Bollinger Bands Features**
- `basis`: 20-period SMA
- `upper`: Upper band (Basis + 2*Dev)
- `lower`: Lower band (Basis - 2*Dev)
- `bandwidth`: Channel width ratio
- `bandwidth_percentile`: Percentile rank within lookback period
- `is_squeeze`: Binary flag for squeeze state (percentile < 20%)
- `is_expansion`: Binary flag for expansion state (percentile > 80%)

**Mean Reversion Features**
- `z_score`: Price deviation from basis in standard deviations

**Smart Money Concepts (SMC) Features**
- `last_ph`: Last confirmed pivot high
- `last_pl`: Last confirmed pivot low
- `bear_sweep`: Bearish liquidity sweep signal
- `bull_sweep`: Bullish liquidity sweep signal
- `bull_bos`: Bullish break of structure
- `bear_bos`: Bearish break of structure

#### Usage Example
```python
from v2.feature_engineering import FeatureEngineer
import pandas as pd

# Load OHLCV data
df = pd.read_csv('data.csv')

# Initialize feature engineer
fe = FeatureEngineer(bb_period=20, lookback=100)

# Process features
df_features = fe.process_features(df)

# Get feature column names
feature_cols = fe.get_feature_columns()
```

## Anti-Lookahead Measures
All pivot points are shifted by `pivot_right` periods to ensure only confirmed historical data is used. Forward fill is applied to maintain continuity.

## Next Steps
1. Data loader module for Binance API integration
2. ML model training module
3. Strategy implementation module
4. Backtesting engine
5. Live trading bot
