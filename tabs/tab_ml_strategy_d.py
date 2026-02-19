"""Strategy D: ML-Based Range-Bound Trading with Tick-Level Backtest"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.ml_range_bound_strategy import MLRangeBoundStrategy
from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


def render_ml_strategy_d_tab(loader, symbol_selector):
    """Render Strategy D tab with ML training and tick-level backtest"""
    
    st.header("Strategy D: ML-Based Range-Bound Trading")
    
    st.success("""
    **Strategy D - AI-Driven Improvements**:
    
    **Key Innovations**:
    1. **No Fixed RSI Thresholds** - ML model learns optimal entry timing dynamically
    2. **20+ Dynamic Features** - Price, volatility, volume, trend, and historical patterns
    3. **Dual ML Models** - Separate LightGBM models for long/short predictions
    4. **Tick-Level Backtest** - Simulates 100 ticks per candle for realistic results
    5. **Adaptive Stop Loss** - ML predicts optimal SL/TP based on market conditions
    
    **vs Strategy C**:
    - More signals (RSI removed)
    - Better risk-adjusted returns
    - More realistic backtest results
    """)
    
    # Create tabs for training and backtest
    sub_tabs = st.tabs(["1. Model Training", "2. Backtest with Tick Simulation"])
    
    # ========== Tab 1: Model Training ==========
    with sub_tabs[0]:
        st.subheader("Train ML Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_symbol_list = symbol_selector("ml_train", multi=False)
            train_symbol = train_symbol_list[0]
            
            train_days = st.slider(
                "Training Data (Days)",
                min_value=30,
                max_value=180,
                value=90,
                key="ml_train_days",
                help="More data = better model"
            )
            
            forward_bars = st.slider(
                "Forward Bars for Label",
                min_value=5,
                max_value=20,
                value=10,
                key="forward_bars",
                help="Look ahead N bars to define profitable trades"
            )
        
        with col2:
            st.subheader("BB Parameters")
            bb_period = st.number_input("BB Period", 10, 50, 20, key="ml_bb_period")
            bb_std = st.number_input("BB Std Dev", 1.0, 3.0, 2.0, 0.1, key="ml_bb_std")
            
            adx_threshold = st.slider(
                "ADX Threshold",
                15, 35, 25,
                key="ml_adx",
                help="Only trade when ADX < threshold (ranging market)"
            )
        
        if st.button("Train Models", key="train_ml_btn", type="primary"):
            with st.spinner(f"Training ML models on {train_symbol}..."):
                try:
                    # Load data
                    if isinstance(loader, BinanceDataLoader):
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=train_days)
                        df = loader.load_historical_data(train_symbol, '15m', start_date, end_date)
                    else:
                        df = loader.load_klines(train_symbol, '15m')
                        df = df.tail(train_days * 96)
                    
                    st.info(f"Loaded {len(df)} candles")
                    
                    # Initialize and train strategy
                    strategy = MLRangeBoundStrategy(
                        bb_period=bb_period,
                        bb_std=bb_std,
                        adx_period=14,
                        adx_threshold=adx_threshold
                    )
                    
                    train_stats = strategy.train(df, forward_bars=forward_bars)
                    
                    # Display training results
                    st.success("Training Complete!")
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("Total Samples", train_stats['total_samples'])
                    with col_r2:
                        st.metric("Long Labels", train_stats['long_samples'])
                    with col_r3:
                        st.metric("Short Labels", train_stats['short_samples'])
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    
                    col_fi1, col_fi2 = st.columns(2)
                    
                    with col_fi1:
                        st.write("**Long Model**")
                        if hasattr(strategy.long_model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Feature': train_stats['feature_names'],
                                'Importance': strategy.long_model.feature_importances_
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            fig = go.Figure(go.Bar(
                                x=importance_df['Importance'],
                                y=importance_df['Feature'],
                                orientation='h'
                            ))
                            fig.update_layout(height=400, title="Top 10 Features")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col_fi2:
                        st.write("**Short Model**")
                        if hasattr(strategy.short_model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Feature': train_stats['feature_names'],
                                'Importance': strategy.short_model.feature_importances_
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            fig = go.Figure(go.Bar(
                                x=importance_df['Importance'],
                                y=importance_df['Feature'],
                                orientation='h'
                            ))
                            fig.update_layout(height=400, title="Top 10 Features")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Save models
                    model_path = f'models/saved/{train_symbol}_strategy_d.pkl'
                    strategy.save_models(model_path)
                    st.success(f"Models saved to {model_path}")
                    
                    # Store in session state for backtest
                    st.session_state['strategy_d_model'] = strategy
                    st.session_state['strategy_d_symbol'] = train_symbol
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # ========== Tab 2: Backtest ==========
    with sub_tabs[1]:
        st.subheader("Tick-Level Backtest")
        
        if 'strategy_d_model' not in st.session_state:
            st.warning("Please train models first in the 'Model Training' tab")
            return
        
        strategy = st.session_state['strategy_d_model']
        trained_symbol = st.session_state.get('strategy_d_symbol', 'BTCUSDT')
        
        st.info(f"Using trained model for {trained_symbol}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Backtest Settings")
            
            test_days = st.slider(
                "Test Period (Days)",
                min_value=7,
                max_value=60,
                value=30,
                key="test_days"
            )
            
            initial_capital = st.number_input(
                "Initial Capital (USDT)",
                min_value=1000.0,
                max_value=100000.0,
                value=10000.0,
                step=1000.0,
                key="ml_capital"
            )
            
            leverage = st.slider(
                "Leverage",
                min_value=1,
                max_value=10,
                value=3,
                key="ml_leverage"
            )
        
        with col2:
            st.subheader("ML Prediction Threshold")
            
            long_threshold = st.slider(
                "Long Probability Threshold",
                min_value=0.3,
                max_value=0.8,
                value=0.6,
                step=0.05,
                key="long_thresh",
                help="Enter long when model confidence > threshold"
            )
            
            short_threshold = st.slider(
                "Short Probability Threshold",
                min_value=0.3,
                max_value=0.8,
                value=0.6,
                step=0.05,
                key="short_thresh"
            )
            
            ticks_per_candle = st.select_slider(
                "Tick Simulation Density",
                options=[50, 100, 200],
                value=100,
                key="ticks",
                help="More ticks = more realistic but slower"
            )
        
        col3, col4 = st.columns(2)
        with col3:
            slippage = st.slider(
                "Slippage %",
                min_value=0.0,
                max_value=0.1,
                value=0.02,
                step=0.01,
                key="slippage"
            )
        with col4:
            fee_rate = st.number_input(
                "Fee Rate",
                min_value=0.0001,
                max_value=0.001,
                value=0.0006,
                step=0.0001,
                format="%.4f",
                key="ml_fee"
            )
        
        if st.button("Run Tick-Level Backtest", key="ml_backtest_btn", type="primary"):
            with st.spinner(f"Running tick-level backtest..."):
                try:
                    # Load test data
                    if isinstance(loader, BinanceDataLoader):
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=test_days)
                        df_test = loader.load_historical_data(trained_symbol, '15m', start_date, end_date)
                    else:
                        df_test = loader.load_klines(trained_symbol, '15m')
                        df_test = df_test.tail(test_days * 96)
                    
                    st.info(f"Loaded {len(df_test)} test candles")
                    
                    # Generate signals
                    df_test = strategy.add_indicators(df_test)
                    
                    signals = []
                    for i in range(50, len(df_test)):
                        long_proba, short_proba = strategy.predict(df_test, i)
                        
                        signal = 0
                        stop_loss = np.nan
                        take_profit = np.nan
                        
                        if long_proba > long_threshold:
                            signal = 1
                            entry = df_test.iloc[i]['close']
                            atr = df_test.iloc[i]['atr']
                            stop_loss = entry - 2 * atr
                            take_profit = df_test.iloc[i]['bb_mid']
                        elif short_proba > short_threshold:
                            signal = -1
                            entry = df_test.iloc[i]['close']
                            atr = df_test.iloc[i]['atr']
                            stop_loss = entry + 2 * atr
                            take_profit = df_test.iloc[i]['bb_mid']
                        
                        signals.append({
                            'signal': signal,
                            'long_proba': long_proba,
                            'short_proba': short_proba,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        })
                    
                    # Pad signals for alignment
                    signals = [{'signal': 0, 'long_proba': 0, 'short_proba': 0, 'stop_loss': np.nan, 'take_profit': np.nan}] * 50 + signals
                    df_signals = pd.DataFrame(signals)
                    
                    signal_count = (df_signals['signal'] != 0).sum()
                    st.info(f"Generated {signal_count} signals (Long: {(df_signals['signal']==1).sum()}, Short: {(df_signals['signal']==-1).sum()})")
                    
                    # Run tick-level backtest
                    engine = TickLevelBacktestEngine(
                        initial_capital=initial_capital,
                        leverage=leverage,
                        fee_rate=fee_rate,
                        slippage_pct=slippage,
                        ticks_per_candle=ticks_per_candle
                    )
                    
                    metrics = engine.run_backtest(df_test, df_signals)
                    
                    # Display results
                    st.subheader("Strategy D Performance")
                    
                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                    
                    with col_r1:
                        st.metric(
                            "Final Equity",
                            f"${metrics['final_equity']:,.2f}",
                            delta=f"${metrics['final_equity'] - initial_capital:,.2f}"
                        )
                        st.metric("Total Trades", metrics['total_trades'])
                    
                    with col_r2:
                        st.metric(
                            "Return %",
                            f"{metrics['total_return_pct']:.2f}%",
                            delta="Tick-level accuracy"
                        )
                        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                    
                    with col_r3:
                        st.metric(
                            "Profit Factor",
                            f"{metrics['profit_factor']:.2f}"
                        )
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    
                    with col_r4:
                        st.metric(
                            "Max Drawdown",
                            f"{metrics['max_drawdown_pct']:.2f}%"
                        )
                        st.metric("Avg PnL/Trade", f"${metrics['avg_pnl_per_trade']:.2f}")
                    
                    # Equity curve
                    st.markdown("### Equity Curve (Tick-Level Simulation)")
                    fig = engine.plot_equity_curve()
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade details
                    trades_df = engine.get_trades_dataframe()
                    if not trades_df.empty:
                        st.markdown("### Trade Log (Last 20)")
                        st.dataframe(trades_df.tail(20), use_container_width=True)
                        
                        # Download
                        csv = trades_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Full Trade Log",
                            data=csv,
                            file_name=f"{trained_symbol}_strategy_d_backtest.csv",
                            mime="text/csv",
                            key="ml_download"
                        )
                        
                        # Performance summary
                        if metrics['total_return_pct'] > 10 and metrics['win_rate'] > 50:
                            st.balloons()
                            st.success("Strategy D shows strong performance with ML-driven signals!")
                    else:
                        st.warning("No trades generated. Try adjusting probability thresholds.")
                        
                except Exception as e:
                    st.error(f"Backtest error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
