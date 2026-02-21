# BB+NW Swing Reversal Trading System - Architecture

## Overview

This document describes the modular architecture of the BB+NW Swing Reversal Trading System v2.0.

## Directory Structure

```
fym_st/
├── trading_system/
│   ├── config/                    # Configuration Module
│   │   ├── __init__.py
│   │   └── constants.py          # All system constants
│   │
│   ├── core/                     # Core Engine Module
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Data loading from HF/Binance
│   │   ├── feature_engineering.py # Feature computation
│   │   ├── event_filter.py       # BB/NW event filtering
│   │   ├── labeling.py           # Triple Barrier labeling
│   │   ├── model_trainer.py      # Model training with CV
│   │   ├── backtester.py         # Backtesting engine
│   │   └── predictor.py          # Real-time prediction
│   │
│   ├── workflows/                # Workflow Orchestration Module
│   │   ├── __init__.py
│   │   ├── training_workflow.py  # Complete training pipeline
│   │   ├── backtest_workflow.py  # Complete backtest pipeline
│   │   └── live_workflow.py      # Live trading pipeline
│   │
│   ├── gui/                      # GUI Module
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── ui_components.py  # Reusable UI components
│   │   │
│   │   ├── pages/
│   │   │   ├── __init__.py
│   │   │   ├── dashboard_page.py
│   │   │   ├── training_page.py
│   │   │   ├── backtesting_page.py
│   │   │   ├── calibration_page.py
│   │   │   ├── optimization_page.py
│   │   │   ├── liquidity_sweep_page.py
│   │   │   └── live_prediction_page.py
│   │   │
│   │   └── __init__.py
│   │
│   ├── models/                   # Saved models directory
│   ├── data/                     # Data cache directory
│   ├── logs/                     # Log files directory
│   └── app_main.py              # Main Streamlit app
│
├── README.md
├── ARCHITECTURE.md              # This file
└── requirements.txt
```

## Module Descriptions

### 1. Config Module (`config/`)

**Purpose**: Centralized configuration management

**Files**:
- `constants.py`: All system constants grouped by category
  - `SystemConfig`: Version, system name
  - `FeatureConfig`: BB, NW, ADX, ATR parameters
  - `FilterConfig`: Event filter thresholds
  - `ModelConfig`: Training parameters
  - `LabelConfig`: Triple Barrier parameters
  - `BacktestConfig`: Backtesting parameters
  - `PerformanceThresholds`: Target metrics
  - `DataConfig`: Data sources and date ranges
  - `UIConfig`: UI text (no emojis)
  - `PathConfig`: File paths

**Usage**:
```python
from config import FeatureConfig, FilterConfig

nw_h = FeatureConfig.NW_H
min_pierce = FilterConfig.MIN_PIERCE_PCT
```

### 2. Core Module (`core/`)

**Purpose**: Core trading system logic

**Key Classes**:
- `CryptoDataLoader`: Load data from HuggingFace/Binance
- `FeatureEngineer`: Calculate BB, NW, ADX, CVD features
- `BBNW_BounceFilter`: Filter BB/NW touch events
- `TripleBarrierLabeling`: Create labels for training
- `ModelTrainer`: Train LightGBM with PurgedKFold CV
- `Backtester`: Simulate trading strategy
- `RealtimePredictor`: Real-time signal generation

### 3. Workflows Module (`workflows/`)

**Purpose**: Orchestrate multi-step processes

**Key Classes**:
- `TrainingWorkflow`: Complete training pipeline
  - Load data
  - Build features
  - Filter events
  - Create labels
  - Train model
  - Save model

**Benefits**:
- Separation of concerns
- Reusable workflows
- Easy to test
- Clear error handling

**Usage**:
```python
from workflows import TrainingWorkflow, TrainingConfig

config = TrainingConfig(
    symbol="BTCUSDT",
    use_2024_only=True,
    tp_multiplier=3.0,
    sl_multiplier=1.0
)

workflow = TrainingWorkflow(config)
result = workflow.run()

if result.success:
    print(f"Model saved: {result.model_path}")
    print(f"AUC: {result.metrics['cv_auc_mean']:.3f}")
```

### 4. GUI Module (`gui/`)

**Purpose**: User interface

**Structure**:
- `utils/ui_components.py`: Reusable UI components
  - `UIComponents.render_header()`
  - `UIComponents.render_metrics_row()`
  - `UIComponents.render_parameter_section()`
  - `UIComponents.render_progress()`
  - `UIComponents.render_equity_curve()`

- `pages/`: Individual page implementations
  - Each page imports from workflows and core
  - Uses UIComponents for consistent UI
  - No emoji in any page

**Benefits**:
- Consistent UI/UX
- Reduced code duplication
- Easy to maintain

## Design Principles

### 1. Separation of Concerns
- Core logic separated from UI
- Configuration separated from implementation
- Workflows orchestrate, core executes

### 2. Single Responsibility
- Each module has one clear purpose
- Each class has one reason to change

### 3. Dependency Injection
- Configuration passed as parameters
- Easy to mock for testing

### 4. Fail Fast
- Validate inputs early
- Raise clear exceptions
- Log detailed error messages

### 5. Observability
- Logging at all levels
- Clear progress indicators
- Detailed error traces

## Data Flow

### Training Flow
```
┌─────────────────┐
│ training_page.py│
└────────┬────────┘
         │
         v
┌─────────────────────┐
│ TrainingWorkflow    │
│ ├─ load_data()      │
│ ├─ build_features() │
│ ├─ filter_events()  │
│ ├─ create_labels()  │
│ ├─ train_model()    │
│ └─ save_model()     │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Core Modules        │
│ ├─ CryptoDataLoader │
│ ├─ FeatureEngineer  │
│ ├─ BBNW_BounceFilter│
│ ├─ TripleBarrier    │
│ └─ ModelTrainer     │
└─────────────────────┘
```

### Backtesting Flow
```
┌──────────────────────┐
│ backtesting_page.py  │
└──────────┬───────────┘
           │
           v
┌──────────────────────┐
│ BacktestWorkflow     │
│ ├─ load_model()      │
│ ├─ load_test_data()  │
│ ├─ build_features()  │
│ ├─ filter_events()   │
│ ├─ predict()         │
│ └─ simulate_trades() │
└──────────┬───────────┘
           │
           v
┌──────────────────────┐
│ Core Modules         │
│ ├─ ModelTrainer      │
│ ├─ FeatureEngineer   │
│ ├─ BBNW_BounceFilter │
│ └─ Backtester        │
└──────────────────────┘
```

## Configuration Management

All configurations are centralized in `config/constants.py`. To modify system behavior:

1. **Feature Parameters**: Edit `FeatureConfig`
2. **Filter Settings**: Edit `FilterConfig`
3. **Model Training**: Edit `ModelConfig`
4. **Backtesting**: Edit `BacktestConfig`
5. **UI Text**: Edit `UIConfig`

## Error Handling

### Strategy
- Validate inputs at workflow level
- Core modules raise specific exceptions
- Workflows catch and log exceptions
- UI displays user-friendly error messages

### Example
```python
try:
    result = workflow.run()
    if result.success:
        st.success(result.message)
    else:
        st.error(result.message)
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    st.error(f"System error: {str(e)}")
```

## Testing Strategy

### Unit Tests
- Test each core module independently
- Mock external dependencies
- Test edge cases

### Integration Tests
- Test workflows end-to-end
- Use sample data
- Verify output format

### UI Tests
- Test page rendering
- Test user interactions
- Verify error handling

## Performance Optimization

### Data Loading
- Cache frequently used data
- Use efficient data formats (Parquet)
- Lazy loading where possible

### Feature Engineering
- Vectorized operations (NumPy/Pandas)
- Avoid loops where possible
- Pre-compute static features

### Model Training
- Use early stopping
- Limit feature count
- Use efficient algorithms (LightGBM)

## Future Enhancements

### Planned
1. Add `backtest_workflow.py`
2. Add `live_workflow.py`
3. Add unit tests
4. Add CLI interface
5. Add REST API

### Under Consideration
1. Database integration
2. Multi-symbol support
3. Portfolio management
4. Risk management module
5. Notification system

## Conclusion

This modular architecture provides:
- **Maintainability**: Easy to update individual components
- **Testability**: Each module can be tested independently
- **Scalability**: Easy to add new features
- **Clarity**: Clear separation of concerns
- **Reusability**: Components can be used in different contexts

For questions or suggestions, please open an issue on GitHub.