# V2 Modular Trading System

## Overview
Modular automated trading system with feature engineering, label generation, machine learning models, and backtesting capabilities.

## Structure
```
v2/
├── feature_engineering.py  # Feature calculation module
├── label_generation.py      # Label generation module
├── example_usage.py         # Complete workflow example
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

## Label Generation Module

### LabelGenerator Class
Generates binary classification labels for Bollinger Bands mean reversion strategy using dynamic stop-loss and take-profit levels based on ATR.

#### Parameters
- `atr_period`: ATR calculation period (default: 14)
- `sl_atr_mult`: Stop-loss ATR multiplier (default: 1.5)
- `tp_atr_mult`: Take-profit ATR multiplier (default: 3.0)
- `lookahead_bars`: Forward-looking bars for label determination (default: 16, equals 4 hours on 15m timeframe)
- `lower_tolerance`: Tolerance for lower band touch detection (default: 1.001)
- `upper_tolerance`: Tolerance for upper band touch detection (default: 0.999)

#### Labeling Logic

**Entry Conditions**
- Long candidates: `Low <= Lower * 1.001`
- Short candidates: `High >= Upper * 0.999`

**Stop-Loss and Take-Profit Calculation**
- Long SL: `Entry - (ATR * 1.5)`
- Long TP: `Entry + (ATR * 3.0)`
- Short SL: `Entry + (ATR * 1.5)`
- Short TP: `Entry - (ATR * 3.0)`

**Label Assignment**
- Label = 1 (Success): TP hit first within lookahead period
- Label = 0 (Failure): SL hit first or neither hit (time exit)

#### Generated Columns
- `atr`: 14-period Average True Range
- `is_touching_lower`: Binary flag for lower band touch
- `is_touching_upper`: Binary flag for upper band touch
- `long_sl`, `long_tp`: Long position stop levels
- `short_sl`, `short_tp`: Short position stop levels
- `target_long`: Binary label for long positions
- `target_short`: Binary label for short positions

#### Usage Example
```python
from v2.label_generation import LabelGenerator

# Initialize label generator
lg = LabelGenerator(
    atr_period=14,
    sl_atr_mult=1.5,
    tp_atr_mult=3.0,
    lookahead_bars=16
)

# Generate labels
df_labeled = lg.generate_labels(df_features)

# Get label statistics
stats = lg.get_label_statistics(df_labeled)
print(stats)

# Prepare training data
df_train_long = lg.prepare_training_data(df_labeled, direction='long')
df_train_short = lg.prepare_training_data(df_labeled, direction='short')
```

## Complete Workflow Example

Run `example_usage.py` to see the full pipeline:

```bash
python v2/example_usage.py
```

This demonstrates:
1. Sample data generation
2. Feature engineering
3. Label generation
4. Training data preparation
5. Label statistics analysis

## Anti-Lookahead Measures
- Pivot points shifted by `pivot_right` periods
- Labels calculated using only future price data (forward-looking simulation)
- No data leakage between feature calculation and label generation

## Next Steps
1. Data loader module for Binance API integration
2. ML model training module
3. Strategy implementation module
4. Backtesting engine
5. Live trading bot
