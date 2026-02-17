# FYM_ST - Advanced Multi-Timeframe AI Trading System

## Overview

FYM_ST is a sophisticated cryptocurrency trading system that combines multiple AI models with multi-timeframe analysis to identify high-probability trading opportunities. The system is specifically designed for high-frequency scalping on 15-minute timeframes while using 1-hour data for trend confirmation.

## Key Features

### Three-Model Architecture

1. **Trend Detection Model (1h)**
   - Identifies market regime: Strong Bullish / Weak Bullish / Ranging / Weak Bearish / Strong Bearish
   - Provides trend strength score (0-100)
   - Filters out counter-trend signals

2. **Volatility Prediction Model (15m)**
   - Forecasts upcoming volatility regime changes
   - Predicts trend initiation probability
   - Enables dynamic stop-loss adjustment

3. **Reversal Detection Model (15m)**
   - Identifies high-probability reversal points
   - Predicts support/resistance levels
   - Generates entry signals with confidence scores

### Trading System Features

- Multi-timeframe analysis (1h + 15m)
- Signal stability (uses only completed candles)
- Multiple cryptocurrency support (38+ pairs)
- Out-of-sample validation (1500 candles reserved)
- Comprehensive backtesting engine
- Real-time Binance data integration
- Advanced risk management (ATR-based stops)
- Portfolio allocation across multiple assets

### Backtesting Capabilities

- Configurable initial capital and leverage
- Dynamic position sizing
- Maximum concurrent positions limit
- ATR-based take-profit and stop-loss
- Binance contract fee structure
- Detailed trade history and metrics
- Equity curve visualization
- Multi-symbol portfolio testing

## Installation

```bash
# Clone the repository
git clone https://github.com/caizongxun/fym_st.git
cd fym_st

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
HUGGINGFACE_TOKEN=your_token_here  # Optional
```

## Usage

### Launch Web GUI

```bash
streamlit run app.py
```

The web interface provides:

- Model training controls
- Real-time signal monitoring
- Backtesting configuration
- Performance analytics
- Trade history viewer

### Training Models

1. Navigate to the "Model Training" tab
2. Select dataset parameters:
   - Training data size (default: 3000 candles)
   - OOS validation size (default: 1500 candles)
   - Symbols to train on
3. Click "Train All Models" or train individually
4. Monitor training progress and OOS metrics

### Backtesting

1. Navigate to the "Backtesting" tab
2. Configure parameters:
   - Initial capital (default: 10 USDT)
   - Leverage (1-50x)
   - Take-profit/Stop-loss (ATR multiples)
   - Position size (% of capital)
   - Max concurrent positions
   - Symbols to trade
   - Backtest period (days)
3. Run backtest and analyze results

## Project Structure

```
fym_st/
├── app.py                      # Streamlit web interface
├── config.py                   # Configuration settings
├── requirements.txt
├── README.md
├── data/
│   ├── __init__.py
│   ├── data_loader.py         # HuggingFace and Binance data loading
│   └── feature_engineer.py    # Technical indicator computation
├── models/
│   ├── __init__.py
│   ├── trend_model.py         # 1h trend detection
│   ├── volatility_model.py    # 15m volatility prediction
│   ├── reversal_model.py      # 15m reversal detection
│   └── saved/                 # Trained model files
├── training/
│   ├── __init__.py
│   ├── train_trend.py
│   ├── train_volatility.py
│   ├── train_reversal.py
│   └── labeling.py            # Label generation logic
├── backtesting/
│   ├── __init__.py
│   ├── engine.py              # Backtesting engine
│   └── metrics.py             # Performance calculations
└── utils/
    ├── __init__.py
    ├── signal_generator.py    # Entry/exit signal logic
    └── risk_manager.py        # Position sizing and stops
```

## Data Sources

- **Training**: HuggingFace dataset `zongowo111/v2-crypto-ohlcv-data`
  - 38 cryptocurrency pairs
  - Three timeframes: 15m, 1h, 1d
  - Historical data for model training

- **Backtesting/Live**: Binance API
  - Real-time market data
  - Accurate fee calculations
  - Production-ready data quality

## Risk Management

### Position Sizing
- Configurable position size as percentage of capital
- Maximum concurrent positions limit prevents overexposure
- Capital allocation across multiple symbols

### Stop Management
- ATR-based dynamic stops (e.g., 2x ATR)
- Wider stops in high volatility, tighter in low volatility
- Take-profit targets based on risk-reward ratio

### Entry Filters
- Multi-model confirmation required
- Volume confirmation
- Trend alignment check
- Reversal probability threshold (>75%)

## Performance Metrics

The system tracks:
- Total return and Sharpe ratio
- Win rate and profit factor
- Maximum drawdown
- Average trade duration
- Trade distribution by symbol
- Equity curve over time

## Warning

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly and use appropriate risk management.

## License

MIT License - See LICENSE file for details

## Author

Developed by [caizongxun](https://github.com/caizongxun)

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.