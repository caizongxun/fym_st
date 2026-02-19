# 15m BB-SMC Confluence Oscillator Theory

## 1. Objective
To create a comprehensive 15-minute timeframe auxiliary indicator that filters noise and identifies high-probability trade setups by integrating **Bollinger Band statistical extremes (Mean Reversion)** with **Smart Money Concepts (Liquidity & Structure)**.

The core philosophy is: **Volatility provides the context (Regime), Statistical Deviation provides the bias (Pressure), and Market Structure provides the trigger (Timing).**

## 2. Theoretical Framework

### Component A: Volatility Regime (The Context)
*Purpose: To determine if the market is in a "Squeeze" (potential for explosive move) or "Expansion" (trend continuation/exhaustion).*

1.  **BB Width Normalization**:
    *   Formula: `BB_Width = (UpperBand - LowerBand) / Basis`
    *   **Normalization**: Calculate the *Percentile Rank* of the current `BB_Width` over a lookback period (e.g., 100 bars).
    *   **Logic**:
        *   `Rank < 20`: **Squeeze Mode**. Volatility is low. Expect a breakout. Mean reversion strategies are risky here if a breakout occurs, but valid if chopping.
        *   `Rank > 80`: **Expansion Mode**. Volatility is high. Trends are active, but exhaustion is possible.
        *   `20 <= Rank <= 80`: **Normal Mode**. Standard trading conditions.

### Component B: Mean Reversion Pressure (The Bias)
*Purpose: To quantify how "stretched" price is relative to its statistical mean.*

1.  **Z-Score / Modified %B**:
    *   Formula: `Z_Score = (Close - Basis) / StdDev`
    *   **Logic**:
        *   `Z > +2.0`: Statistically overbought (2 deviations above mean).
        *   `Z < -2.0`: Statistically oversold (2 deviations below mean).
        *   `Z` nearing 0: Price is at equilibrium (Basis).
2.  **Momentum of Return (Snap-back)**:
    *   Measure the *rate of change* of the Z-Score.
    *   If `Z` was > +2.0 and is now +1.5 within 1 bar, strong mean reversion momentum exists.

### Component C: SMC Micro-Structure (The Trigger)
*Purpose: To identify specific price action events that confirm the statistical bias.*

1.  **Swing Point Identification**:
    *   Use `ta.pivothigh` and `ta.pivotlow` with a short length (e.g., left=3, right=3) to define local structure.
2.  **Event Definitions**:
    *   **Liquidity Sweep (Turtle Soup)**:
        *   *Bearish Sweep*: Price makes a Higher High (breaks previous Pivot High) but closes *below* that previous High. (Wick sweep).
        *   *Bullish Sweep*: Price makes a Lower Low (breaks previous Pivot Low) but closes *above* that previous Low.
    *   **Break of Structure (BOS)**:
        *   *Bullish BOS*: Price breaks previous Pivot High and closes *above* it with a strong body.
        *   *Bearish BOS*: Price breaks previous Pivot Low and closes *below* it with a strong body.
    *   **Displacement Filter**:
        *   Valid BOS requires the candle body to be > 50% of the candle range (filtering wicks).

## 3. Signal Synthesis Logic (The Algorithm)

The indicator should output a **Confluence Score (-10 to +10)** based on the interaction of these components.

### Scenario 1: Mean Reversion Reversal (Contrarian)
*   **Condition**:
    1.  **Bias**: `Z-Score` is extreme (> +2 or < -2).
    2.  **Trigger**: Opposite **Liquidity Sweep** detected. (e.g., Z > +2 AND Bearish Sweep).
    3.  **Regime**: `BB Width Rank` is NOT in extreme Squeeze (to avoid catching a breakout) OR is in extreme Expansion (exhaustion).
*   **Signal**: High Probability Reversal.

### Scenario 2: Volatility Expansion Breakout (Trend Following)
*   **Condition**:
    1.  **Regime**: `BB Width Rank` was low (< 20) and is now rising (Expansion starting).
    2.  **Trigger**: Valid **BOS** in the direction of the break.
    3.  **Bias**: `Z-Score` confirms direction (e.g., Z crosses +1 upwards).
*   **Signal**: High Probability Breakout/Continuation.

## 4. Visualization Recommendations for Pine Script
1.  **Main Oscillator**: Plot the `Z-Score` as a histogram or line.
    *   Color code based on Regime: Grey for Squeeze, Bright for Expansion.
2.  **Signal Overlays**:
    *   Plot shapes (Triangle/Circle) on the oscillator when a **Sweep** or **BOS** coincides with the correct Z-Score zone.
    *   *Green Circle*: Oversold (Z < -2) + Bullish Sweep.
    *   *Red Circle*: Overbought (Z > +2) + Bearish Sweep.
3.  **Background Color**:
    *   Light Red/Green tint when Z-Score is extreme to highlight reversal zones.

## 5. Coding Instructions for AI
*   **Inputs**: `length` (20), `mult` (2.0) for BB; `pivot_len` (3) for SMC; `lookback` (100) for Percentile.
*   **Output**: A single sub-chart indicator.
*   **Optimization**: Ensure `ta.pivothigh/low` logic does not repaint (use confirmed pivots or handle lag appropriately for real-time alerts).
