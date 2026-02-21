# V2 Modular Trading System

## Overview
Modular automated trading system with HuggingFace data integration, feature engineering, label generation, dual-model training, and confluence-veto inference logic for cryptocurrency trading.

## Structure
```
v2/
├── data_loader.py             # HuggingFace data loader
├── feature_engineering.py    # Feature calculation module
├── label_generation.py        # Label generation module
├── model_trainer.py           # Dual-model training system
├── inference_engine.py        # Confluence-veto inference engine
├── pipeline.py                # Complete pipeline integration
├── example_usage.py           # Basic feature/label example
├── example_data_pipeline.py   # Data pipeline example
├── example_model_training.py  # Model training and inference example
├── models/                    # Trained model storage
├── strategy.py               # Trading strategy module (TBD)
├── backtest.py               # Backtesting engine (TBD)
└── README.md                 # Documentation
```

## Data Loader Module

### CryptoDataLoader Class
Loads cryptocurrency OHLCV data from HuggingFace dataset.

#### Dataset Information
- **Repository**: `zongowo111/v2-crypto-ohlcv-data`
- **Symbols**: 38 cryptocurrency pairs (BTCUSDT, ETHUSDT, etc.)
- **Timeframes**: 15m, 1h, 1d
- **Total Files**: 114 parquet files

#### Usage Example
```python
from v2.data_loader import CryptoDataLoader

loader = CryptoDataLoader()
df = loader.load_klines('BTCUSDT', '15m')
```

## Feature Engineering Module

### FeatureEngineer Class
Calculates 15 technical indicators including Bollinger Bands, Z-Score, and SMC features.

#### Features
- Bollinger Bands (basis, upper, lower, bandwidth, percentile)
- Squeeze/Expansion states
- Mean reversion (z_score)
- SMC signals (pivot points, sweeps, BOS)

## Label Generation Module

### LabelGenerator Class
Generates binary labels using dynamic ATR-based stop-loss and take-profit levels.

#### Parameters
- ATR period: 14
- SL multiplier: 1.5x ATR
- TP multiplier: 3.0x ATR
- Lookahead: 16 bars (4 hours on 15m)

#### Logic
- Label = 1: TP hit first
- Label = 0: SL hit first or timeout

## Model Training Module

### ModelTrainer Class
Trains bounce prediction model (Model A) using LightGBM classifier.

#### Features
- Automatic feature selection (excludes OHLC, timestamps)
- Time-series split (80/20 train/test)
- Class weight balancing
- Early stopping
- ROC-AUC evaluation
- Feature importance analysis

#### Usage Example
```python
from v2.model_trainer import ModelTrainer

trainer = ModelTrainer(
    model_type='bounce',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7
)

results = trainer.train(df_train, train_ratio=0.8)
trainer.save_model('v2/models/bb_bounce_model.pkl')
```

### TrendFilterTrainer Class
Trains trend filter model (Model B) to identify false breakouts.

#### Purpose
Learn "do not enter" signals by identifying:
- Strong trend continuation patterns
- Liquidity sweep failures
- False bounce setups

#### Label Logic
Inverts bounce labels: failure cases (label=0) become filter targets (label=1)

#### Usage Example
```python
from v2.model_trainer import TrendFilterTrainer

trainer = TrendFilterTrainer(
    n_estimators=500,
    learning_rate=0.05
)

results = trainer.train(df_train)
trainer.save_model('v2/models/trend_filter_model.pkl')
```

## Inference Engine Module

### InferenceEngine Class
Implements dual-model confluence-veto decision logic.

#### Architecture
```
Input Features
      │
      ├───────────> Model A (Bounce)
      │              │
      │              P_bounce
      │              │
      └───────────> Model B (Filter)
                     │
                     P_filter
                     │
               Confluence-Veto Logic
                     │
              Entry Signal (0 or 1)
```

#### Decision Thresholds
- **Bounce threshold**: 0.65 (P_bounce > 0.65)
- **Filter threshold**: 0.40 (P_filter < 0.40)

#### Decision Logic
```python
if P_bounce > 0.65 and P_filter < 0.40:
    signal = 1  # ENTRY_APPROVED
elif P_bounce <= 0.65:
    signal = 0  # BOUNCE_WEAK
elif P_filter >= 0.40:
    signal = 0  # TREND_VETO
```

#### Output Signals
- **ENTRY_APPROVED**: Both models agree (confluence)
- **BOUNCE_WEAK**: Model A confidence insufficient
- **TREND_VETO**: Model B blocks entry (veto power)

#### Usage Example
```python
from v2.inference_engine import InferenceEngine

engine = InferenceEngine(
    bounce_model_path='v2/models/bb_bounce_model.pkl',
    filter_model_path='v2/models/trend_filter_model.pkl',
    bounce_threshold=0.65,
    filter_threshold=0.40
)

# Single prediction
result = engine.predict_single(features)
print(result)
# {'p_bounce': 0.72, 'p_filter': 0.35, 'signal': 1, 'reason': 'ENTRY_APPROVED'}

# Batch prediction
df_predictions = engine.predict_batch(df_test)
stats = engine.get_statistics(df_predictions)
```

## Complete Workflow

### Training Pipeline
```bash
python v2/example_model_training.py
```

This executes:
1. Load BTCUSDT 15m data for 2024
2. Generate features and labels
3. Train Model A (bounce prediction)
4. Train Model B (trend filter)
5. Test inference engine on ETHUSDT
6. Display performance statistics

### Expected Output
```
Bounce Model Test AUC: 0.75+
Filter Model Test AUC: 0.70+
Entry Approval Rate: 15-25%
Approved Entry Success Rate: 55-70%
```

## Pipeline Module

### TradingPipeline Class
Integrates data loading, feature engineering, and label generation.

#### Usage
```python
from v2.pipeline import TradingPipeline

pipeline = TradingPipeline()

# Single symbol
df_train, features = pipeline.prepare_training_data(
    symbol='BTCUSDT',
    timeframe='15m',
    direction='long',
    start_date='2024-01-01'
)

# Batch processing
df_combined = pipeline.batch_process(
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    timeframe='15m',
    direction='long'
)
```

## Installation Requirements

```bash
pip install pandas numpy lightgbm scikit-learn joblib huggingface-hub pyarrow
```

## Anti-Lookahead Measures
- Time-series split (no random shuffling)
- Pivot points shifted by confirmation period
- Labels use only future price data
- Features exclude OHLC and timestamps
- Strict feature filtering in model training

## Model Performance Guidelines

### Bounce Model (Model A)
- Target ROC-AUC: > 0.70
- Precision focus: minimize false positives
- Recall balance: capture valid setups

### Filter Model (Model B)
- Target ROC-AUC: > 0.65
- High recall priority: catch dangerous conditions
- Veto power: blocks risky entries

### Inference Engine
- Entry rate: 15-25% of candidates
- Success rate: 55-70% of approved entries
- Risk reduction: 30-40% fewer losses vs. single model

## Next Steps
1. Backtesting engine with realistic slippage
2. Strategy module with position sizing
3. Live trading bot with risk management
4. Multi-timeframe model ensemble
5. Adaptive threshold optimization
