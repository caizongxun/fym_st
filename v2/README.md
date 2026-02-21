# V2 Modular Trading System

## Overview
Modular automated trading system with HuggingFace data integration, feature engineering, label generation, and machine learning capabilities for cryptocurrency trading.

## Structure
```
v2/
├── data_loader.py           # HuggingFace data loader
├── feature_engineering.py  # Feature calculation module
├── label_generation.py      # Label generation module
├── pipeline.py              # Complete pipeline integration
├── example_usage.py         # Basic feature/label example
├── example_data_pipeline.py # Complete pipeline example
├── model.py                # ML model module (TBD)
├── strategy.py             # Trading strategy module (TBD)
├── backtest.py             # Backtesting engine (TBD)
└── README.md               # Documentation
```

## Data Loader Module

### CryptoDataLoader Class
Loads cryptocurrency OHLCV data from HuggingFace dataset.

#### Dataset Information
- **Repository**: `zongowo111/v2-crypto-ohlcv-data`
- **Symbols**: 38 cryptocurrency pairs (BTCUSDT, ETHUSDT, etc.)
- **Timeframes**: 15m, 1h, 1d
- **Total Files**: 114 parquet files

#### Data Structure
Path format: `klines/{SYMBOL}/{BASE}_{TIMEFRAME}.parquet`

Example:
- `klines/BTCUSDT/BTC_15m.parquet`
- `klines/ETHUSDT/ETH_1h.parquet`

#### Available Symbols (38)
```
AAVEUSDT, ADAUSDT, ALGOUSDT, ARBUSDT, ATOMUSDT, AVAXUSDT,
BALUSDT, BATUSDT, BCHUSDT, BNBUSDT, BTCUSDT, COMPUSDT,
CRVUSDT, DOGEUSDT, DOTUSDT, ENJUSDT, ENSUSDT, ETCUSDT,
ETHUSDT, FILUSDT, GALAUSDT, GRTUSDT, IMXUSDT, KAVAUSDT,
LINKUSDT, LTCUSDT, MANAUSDT, MATICUSDT, MKRUSDT, NEARUSDT,
OPUSDT, SANDUSDT, SNXUSDT, SOLUSDT, SPELLUSDT, UNIUSDT,
XRPUSDT, ZRXUSDT
```

#### Usage Example
```python
from v2.data_loader import CryptoDataLoader

# Initialize loader
loader = CryptoDataLoader()

# Load single symbol
df = loader.load_klines('BTCUSDT', '15m')

# Load multiple symbols
data = loader.load_multiple_symbols(['BTCUSDT', 'ETHUSDT'], '15m')

# Load all timeframes for one symbol
data = loader.load_all_timeframes('BTCUSDT')

# Get dataset info
info = loader.get_dataset_info()

# Filter by date range
df_filtered = loader.filter_date_range(df, '2024-01-01', '2024-12-31')
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

## Pipeline Module

### TradingPipeline Class
Integrates all modules into a seamless workflow.

#### Features
- Single symbol processing
- Batch processing for multiple symbols
- Automatic date filtering
- Label statistics reporting
- Training data preparation

#### Usage Example
```python
from v2.pipeline import TradingPipeline

# Initialize pipeline
pipeline = TradingPipeline(
    bb_period=20,
    atr_period=14,
    sl_atr_mult=1.5,
    tp_atr_mult=3.0,
    lookahead_bars=16
)

# Process single symbol
df_train, feature_cols = pipeline.prepare_training_data(
    symbol='BTCUSDT',
    timeframe='15m',
    direction='long',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Batch process multiple symbols
df_combined = pipeline.batch_process(
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    timeframe='15m',
    direction='long',
    start_date='2024-01-01'
)
```

## Complete Workflow Example

Run the complete pipeline example:

```bash
python v2/example_data_pipeline.py
```

This demonstrates:
1. Dataset information display
2. Single symbol processing with features and labels
3. Batch processing multiple symbols
4. Label statistics analysis
5. Training data preparation

## Installation Requirements

```bash
pip install pandas numpy huggingface-hub pyarrow
```

## Anti-Lookahead Measures
- Pivot points shifted by `pivot_right` periods
- Labels calculated using only future price data (forward-looking simulation)
- No data leakage between feature calculation and label generation

## Next Steps
1. ML model training module (XGBoost/LightGBM)
2. Strategy implementation module
3. Backtesting engine
4. Live trading bot
