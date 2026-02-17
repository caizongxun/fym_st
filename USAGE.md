# FYM_ST Usage Guide

## Quick Start

### 1. Installation

```bash
git clone https://github.com/caizongxun/fym_st.git
cd fym_st
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file:

```env
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
```

### 3. Launch GUI

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

## Training Models

### Step 1: Select Symbol
Choose from 38 supported cryptocurrency pairs (e.g., BTCUSDT, ETHUSDT)

### Step 2: Configure Training
- Training Size: 3000 candles (default)
- OOS Validation: 1500 candles (reserved for validation)

### Step 3: Select Models
- Trend Detection (1h): Analyzes major trend direction
- Volatility Prediction (15m): Forecasts volatility changes
- Reversal Detection (15m): Identifies entry points

### Step 4: Start Training
Click "Start Training" and wait for completion. Models will be saved automatically.

### Training Tips
- Train on multiple symbols for diversification
- Re-train periodically (every 1-2 weeks) as market conditions change
- Check OOS metrics to ensure models generalize well
- Higher OOS accuracy indicates better performance on unseen data

## Running Backtests

### Capital Configuration
- **Initial Capital**: Starting balance (default: 10 USDT)
- **Leverage**: 1-50x (recommended: 5-10x for beginners)
- **Position Size**: Percentage of capital per trade (default: 95%)
- **Max Positions**: Maximum concurrent trades (default: 1)

### Risk Management
- **Take Profit**: ATR multiplier for profit target (default: 3x)
- **Stop Loss**: ATR multiplier for stop loss (default: 2x)

ATR (Average True Range) adapts stops to current volatility:
- High volatility = wider stops
- Low volatility = tighter stops

### Backtest Period
- Select number of days to test (7-365)
- Choose symbols to trade
- Data fetched from Binance API

### Interpreting Results

**Key Metrics:**
- **Win Rate**: >50% is good, >60% is excellent
- **Profit Factor**: >1.5 is profitable, >2.0 is strong
- **Max Drawdown**: Keep below 30% for safety
- **Sharpe Ratio**: >1.0 is acceptable, >2.0 is excellent

**Equity Curve:**
- Should trend upward consistently
- Avoid strategies with prolonged flat periods
- Check drawdown depth and recovery time

**Trade Distribution:**
- Verify trades are distributed across symbols
- Avoid over-concentration in single asset

## Live Signal Monitoring

### Real-Time Analysis
1. Select symbol to monitor
2. Click "Get Latest Signal"
3. System fetches live data from Binance
4. Analyzes using trained models
5. Displays current signal and predictions

### Signal Interpretation

**Signal Types:**
- **LONG**: Buy signal detected
- **SHORT**: Sell signal detected  
- **NONE**: No trade opportunity

**Signal Strength (0-100):**
- <50: Weak signal, avoid
- 50-70: Moderate signal, use with caution
- >70: Strong signal, high confidence

**Trend Context:**
- Strong Bull/Bear: Directional bias clear
- Weak Bull/Bear: Caution, trend weakening
- Ranging: Avoid or use mean-reversion strategy

**Volatility Regime:**
- High: Wider stops needed, bigger moves expected
- Medium: Normal trading conditions
- Low: Tighter stops, smaller profit targets

## Multi-Symbol Portfolio Strategy

### Capital Allocation
When trading multiple symbols with 10 USDT capital:
- System allocates capital across all positions
- Each position uses (Capital * Position Size) / Max Positions
- Example: 10 USDT, 2 max positions, 95% size = 4.75 USDT per position

### Symbol Selection
Recommended combinations:
- **Conservative**: BTC + ETH (major pairs)
- **Moderate**: BTC + ETH + BNB + SOL
- **Aggressive**: 5-8 mid-cap altcoins

### Diversification Benefits
- Reduces single-asset risk
- Captures opportunities across market
- Smoother equity curve

## Advanced Configuration

### Editing config.py

**Signal Thresholds:**
```python
MIN_REVERSAL_PROB = 0.75  # Lower for more signals
MIN_TREND_STRENGTH = 60    # Higher for stronger trends only
VOLUME_MULTIPLIER = 1.3    # Volume confirmation threshold
```

**Model Horizons:**
```python
TREND_HORIZON = 10      # Candles ahead for trend prediction
VOLATILITY_HORIZON = 5  # Candles ahead for volatility
REVERSAL_HORIZON = 10   # Candles ahead for reversal
```

**Trading Parameters:**
```python
DEFAULT_TP_ATR = 3.0  # More aggressive: 4-5x
DEFAULT_SL_ATR = 2.0  # Tighter: 1.5x, Wider: 2.5x
```

## Troubleshooting

### Issue: "Models not found"
**Solution**: Train models for the selected symbol first

### Issue: "Failed to load data"
**Solution**: 
- Check Binance API credentials in `.env`
- Verify internet connection
- Try different symbol

### Issue: "No trades in backtest"
**Solution**:
- Lower signal thresholds in `config.py`
- Increase backtest period
- Check if models are trained properly

### Issue: "High drawdown"
**Solution**:
- Reduce leverage
- Decrease position size
- Widen stop loss (higher ATR multiplier)
- Use more conservative signal thresholds

### Issue: "Too many signals"
**Solution**:
- Increase `MIN_REVERSAL_PROB` threshold
- Increase `MIN_TREND_STRENGTH` threshold
- Enable stricter volume confirmation

## Risk Warnings

1. **This is experimental software**: Thoroughly test before risking real capital
2. **Leverage is dangerous**: Can amplify both gains and losses
3. **Past performance ≠ future results**: Backtest results may not reflect live trading
4. **Market conditions change**: Re-train models regularly
5. **Start small**: Begin with minimum capital to validate strategy
6. **Monitor actively**: Even automated systems need supervision
7. **Exchange risks**: API keys can be compromised, use IP restrictions

## Performance Optimization

### Model Training
- Use more recent data for training (last 3-6 months)
- Increase OOS validation size for better generalization checks
- Train on correlated symbols if individual symbol lacks data

### Signal Quality
- Higher reversal probability threshold = fewer but better signals
- Volume confirmation reduces false signals
- Multi-timeframe alignment improves win rate

### Risk Management
- Never risk more than 2-5% per trade
- Use position sizing to scale risk appropriately
- Set maximum daily/weekly loss limits

## Data Management

### Training Data Source
- HuggingFace dataset: `zongowo111/v2-crypto-ohlcv-data`
- Contains 38 symbols with 15m, 1h, 1d timeframes
- Historical data suitable for model training

### Backtesting/Live Data
- Binance API: Real-time market data
- Ensures fee calculations match actual trading
- Use completed candles only (signal stability)

## Continuous Improvement

### Model Iteration
1. Review backtest performance
2. Identify losing streaks or drawdowns
3. Analyze failed trades (stopped out vs wrong direction)
4. Adjust thresholds or retrain with new data
5. Re-backtest to validate improvements

### Strategy Evolution
- Test different TP/SL ratios
- Experiment with position sizing rules
- Add filters (time of day, macro events)
- Combine with other indicators

## Support

For issues or questions:
- GitHub Issues: https://github.com/caizongxun/fym_st/issues
- Review code and documentation
- Test on paper trading first

## Next Steps

1. Train models on your preferred symbols
2. Run backtests with conservative parameters
3. Analyze results and adjust settings
4. Paper trade before live deployment
5. Start with minimal capital
6. Scale up gradually as confidence builds

Good luck and trade responsibly!