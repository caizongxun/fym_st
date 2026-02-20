# Strategy Optimization Guide

## Problem Analysis

Your current backtest results show:

- Win Rate: 36.7% (Too Low)
- Profit Factor: 0.70 (Below 1.0 = Losing)
- Average Win: $6.63 vs Average Loss: $5.50 (Poor Risk/Reward)
- Stop Loss Triggers: 1008 vs Take Profit: 535 (56% hit SL)
- Commission: $2,044 on $10,000 capital (20% eaten by fees)

## Root Causes

### 1. Low Signal Quality
The model is generating too many false signals with win probability below 65%.

### 2. Suboptimal TP/SL Ratio
Current 2.5:1.5 ratio might not match market behavior for this asset/timeframe.

### 3. Over-Trading
1774 trades means too frequent trading, resulting in excessive commission costs.

### 4. No Signal Filtering
Accepting all signals without additional technical confirmation.

## Solution: Use Strategy Optimization

### Step 1: Optimize Probability Threshold

**Goal**: Reduce false signals by increasing minimum win probability.

1. Go to "Strategy Optimization" page
2. Select "Probability Threshold" optimization
3. System will test thresholds: 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
4. Expected outcome:
   - Fewer trades (reduce commission impact)
   - Higher win rate (better signal quality)
   - Improved profit factor

**Recommendation**: Start with minimum probability 0.65-0.70

### Step 2: Optimize TP/SL Ratio

**Goal**: Find the risk/reward ratio that matches actual market volatility.

1. Select "TP/SL Ratio" optimization
2. System will test combinations:
   - TP: 2.0x, 2.5x, 3.0x, 3.5x, 4.0x ATR
   - SL: 1.0x, 1.25x, 1.5x, 1.75x, 2.0x ATR
3. Heatmap shows which combination gives best return
4. Expected outcome:
   - Better balance between TP and SL triggers
   - Improved profit factor
   - Potentially higher win rate

**Common Findings**:
- If SL triggers too often: Increase SL multiplier to 2.0x or reduce TP to 2.0x
- If timeout is high: Markets not volatile enough, consider different timeframe

### Step 3: Apply Signal Filters

**Goal**: Add technical confirmation to ML predictions.

1. Select "Signal Filters" optimization
2. System tests 4 configurations:
   - **No filters**: Baseline (your current state)
   - **Conservative**: High probability (0.70), strict filters
   - **Moderate**: Balanced approach (0.65), moderate filters
   - **Aggressive**: More signals (0.60), loose filters

3. Filters include:
   - Probability threshold
   - Volatility regime (VSR)
   - Trend alignment (EMA crossover)
   - RSI range (avoid overbought/oversold)
   - Volume confirmation
   - MACD confirmation

**Expected Outcome**:
- Conservative: Fewer trades, higher win rate, lower drawdown
- Moderate: Balanced risk/reward
- Aggressive: More trades, lower win rate but may catch more moves

## Recommended Workflow

### Phase 1: Quick Fix (Reduce Losses)

1. Run "Probability Threshold" optimization
2. Set minimum to 0.70 (only take high-confidence signals)
3. Re-run backtest
4. Expected: Win rate > 45%, fewer trades, positive return

### Phase 2: Fine-Tune (Maximize Returns)

1. Run "TP/SL Ratio" optimization with new probability threshold
2. Identify best TP/SL combination
3. Re-run backtest with optimized parameters
4. Expected: Profit factor > 1.2, max drawdown < 15%

### Phase 3: Add Filters (Reduce Risk)

1. Run "Signal Filters" optimization
2. Compare Conservative vs Moderate configurations
3. Choose based on your risk tolerance:
   - Conservative: Lower returns but safer
   - Moderate: Balanced approach
4. Expected: Sharpe ratio > 1.0, consistent returns

## Target Metrics

After optimization, aim for:

- **Win Rate**: 45-55%
- **Profit Factor**: > 1.5
- **Max Drawdown**: < 20%
- **Sharpe Ratio**: > 1.0
- **Average Win / Average Loss**: > 1.5
- **Number of Trades**: Reduce by 50-70% (quality over quantity)

## Example Optimization Results

### Before Optimization
```
Total Return: -18.60%
Win Rate: 36.70%
Profit Factor: 0.70
Total Trades: 1774
Commission: $2,044
```

### After Optimization (Expected)
```
Total Return: +15-30%
Win Rate: 48-52%
Profit Factor: 1.5-2.0
Total Trades: 400-600
Commission: $400-600
```

## Common Issues and Solutions

### Issue: Still losing after optimization
**Solution**: 
- Model may not be suitable for this asset/timeframe
- Try training on different timeframe (1h instead of 15m)
- Increase training data size
- Consider re-training with different label parameters

### Issue: Too few signals after filtering
**Solution**:
- Lower probability threshold slightly (0.65 instead of 0.70)
- Use "Moderate" instead of "Conservative" filters
- Consider multiple assets to increase opportunities

### Issue: Good backtest but poor forward testing
**Solution**:
- Likely overfitting - reduce number of features
- Use longer out-of-sample period
- Implement walk-forward optimization
- Monitor live performance and retrain periodically

## Advanced Tips

### 1. Multi-Timeframe Confirmation
Train models on both 15m and 1h, only take signals when both agree.

### 2. Position Sizing Adjustment
Reduce Kelly fraction from 0.5 to 0.3 for more conservative sizing.

### 3. Time-Based Filters
Avoid trading during low liquidity hours (typically better performance during active market hours).

### 4. Volatility Regime Adaptation
Use different TP/SL ratios based on current market volatility:
- High volatility: Wider stops (2.0x SL, 3.5x TP)
- Low volatility: Tighter stops (1.25x SL, 2.0x TP)

## Next Steps

1. Start optimization now in "Strategy Optimization" page
2. Test each optimization type
3. Record best parameters
4. Re-run full backtest with optimized settings
5. If profitable, proceed to paper trading
6. Monitor performance and adjust as needed

Remember: The goal is not to maximize backtest returns, but to find robust parameters that work across different market conditions.