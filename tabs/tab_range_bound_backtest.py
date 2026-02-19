# -*- coding: utf-8 -*-
"""
Strategy C: Range-Bound Backtest Tab
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.range_bound_signal_generator import RangeBoundSignalGenerator
from backtesting.range_bound_engine import RangeBoundBacktestEngine
from data.binance_loader import BinanceDataLoader


def render_range_bound_backtest_tab(loader, symbol_selector):
    """Render Strategy C backtest tab"""
    
    st.header("Strategy C: Range-Bound Trading")
    
    st.success("""
    **Strategy C - Improved Range-Bound Strategy**:
    
    **Core Logic**:
    1. Range confirmation: ADX < 25 (non-trending market)
    2. Long signals:
       - Price <= BB lower band
       - RSI < 30 (oversold)
       - Volume shrinking (panic selling)
    3. Short signals:
       - Price >= BB upper band
       - RSI > 70 (overbought)
       - Volume shrinking (FOMO buying)
    4. Exit: Price returns to BB middle or ATR-based stop loss
    
    **Advantages**:
    - Multiple filters improve win rate
    - ATR dynamic stop loss adapts to market volatility
    - Improves original BB reversal strategy's low win rate
    """)
    
    st.markdown("### Strategy Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Settings")
        rb_symbol_list = symbol_selector("range_bound", multi=False)
        rb_symbol = rb_symbol_list[0]
        
        rb_days = st.slider(
            "Backtest Days",
            min_value=30,
            max_value=180,
            value=60,
            key="rb_days"
        )
        
        rb_capital = st.number_input(
            "Initial Capital (USDT)",
            min_value=100.0,
            max_value=100000.0,
            value=10000.0,
            step=100.0,
            key="rb_capital"
        )
        
        rb_leverage = st.slider(
            "Leverage",
            min_value=1,
            max_value=20,
            value=1,
            key="rb_leverage"
        )
    
    with col2:
        st.subheader("Strategy Parameters")
        
        adx_threshold = st.slider(
            "ADX Threshold (Range Confirmation)",
            min_value=15,
            max_value=35,
            value=25,
            key="rb_adx",
            help="Confirm ranging market when ADX < threshold"
        )
        
        rsi_oversold = st.slider(
            "RSI Oversold",
            min_value=20,
            max_value=35,
            value=30,
            key="rb_rsi_low"
        )
        
        rsi_overbought = st.slider(
            "RSI Overbought",
            min_value=65,
            max_value=80,
            value=70,
            key="rb_rsi_high"
        )
        
        volume_threshold = st.slider(
            "Volume Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.1,
            key="rb_vol",
            help="Volume < MA * threshold = shrinking"
        )
    
    st.markdown("### Stop Loss & Take Profit")
    col3, col4 = st.columns(2)
    
    with col3:
        use_atr_stops = st.radio(
            "Stop Loss Mode",
            options=[True, False],
            format_func=lambda x: "ATR Dynamic Stop" if x else "Fixed Percentage Stop",
            key="rb_stop_mode"
        )
    
    with col4:
        if use_atr_stops:
            atr_multiplier = st.slider(
                "ATR Multiplier",
                min_value=1.0,
                max_value=4.0,
                value=2.0,
                step=0.5,
                key="rb_atr_mult"
            )
            fixed_stop_pct = 0.02
        else:
            atr_multiplier = 2.0
            fixed_stop_pct = st.slider(
                "Fixed Stop Loss %",
                min_value=0.01,
                max_value=0.05,
                value=0.02,
                step=0.005,
                format_func=lambda x: f"{x*100:.1f}%",
                key="rb_fixed_stop"
            )
        
        target_rr = st.slider(
            "Target Risk-Reward Ratio",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.5,
            key="rb_rr",
            help="Risk:Reward = 1:this value"
        )
    
    if st.button("Run Backtest", key="rb_backtest_btn", type="primary"):
        with st.spinner(f"Backtesting {rb_symbol}..."):
            try:
                # Load data
                if isinstance(loader, BinanceDataLoader):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=rb_days)
                    df = loader.load_historical_data(rb_symbol, '15m', start_date, end_date)
                else:
                    df = loader.load_klines(rb_symbol, '15m')
                    df = df.tail(rb_days * 96)
                
                st.info(f"Loaded {len(df)} candles")
                
                # Generate signals
                generator = RangeBoundSignalGenerator(
                    adx_threshold=adx_threshold,
                    rsi_oversold=rsi_oversold,
                    rsi_overbought=rsi_overbought,
                    volume_threshold=volume_threshold,
                    use_atr_stops=use_atr_stops,
                    atr_multiplier=atr_multiplier,
                    fixed_stop_pct=fixed_stop_pct,
                    target_rr=target_rr
                )
                
                df_signals = generator.generate_signals(df)
                
                signal_count = (df_signals['signal'] != 0).sum()
                st.info(f"Generated {signal_count} signals (Long: {(df_signals['signal']==1).sum()}, Short: {(df_signals['signal']==-1).sum()})")
                
                # Run backtest
                engine = RangeBoundBacktestEngine(
                    initial_capital=rb_capital,
                    leverage=rb_leverage,
                    fee_rate=0.0006
                )
                
                signals_dict = {rb_symbol: df_signals}
                metrics = engine.run_backtest(signals_dict)
                
                # Display results
                st.subheader("Backtest Results")
                
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                
                with col_r1:
                    st.metric(
                        "Final Equity",
                        f"${metrics['final_equity']:,.2f}",
                        delta=f"${metrics['final_equity'] - rb_capital:,.2f}"
                    )
                    st.metric("Total Trades", metrics['total_trades'])
                
                with col_r2:
                    st.metric(
                        "Return %",
                        f"{metrics['total_return_pct']:.2f}%",
                        delta="vs 0.18% original"
                    )
                    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                
                with col_r3:
                    st.metric(
                        "Profit Factor",
                        f"{metrics['profit_factor']:.2f}",
                        delta="target >1.5"
                    )
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                
                with col_r4:
                    st.metric(
                        "Max Drawdown",
                        f"{metrics['max_drawdown_pct']:.2f}%"
                    )
                    avg_duration = metrics['avg_duration_min']
                    st.metric("Avg Hold (min)", f"{avg_duration:.0f}" if avg_duration else "N/A")
                
                # Equity curve
                st.markdown("### Equity Curve")
                fig = engine.plot_equity_curve()
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade details
                trades_df = engine.get_trades_dataframe()
                if not trades_df.empty:
                    st.markdown("### Trade Details (Last 20)")
                    display_cols = ['Entry Time', 'Exit Time', 'Direction', 'Entry Price', 'Exit Price',
                                  'PnL (USDT)', 'PnL %', 'Exit Reason', 'Hold Time (min)']
                    st.dataframe(trades_df[display_cols].tail(20), use_container_width=True)
                    
                    # Download button
                    csv = trades_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Trade Log CSV",
                        data=csv,
                        file_name=f"{rb_symbol}_strategy_c_backtest.csv",
                        mime="text/csv",
                        key="rb_download"
                    )
                    
                    # Performance analysis
                    if metrics['total_return_pct'] > 5 and metrics['win_rate'] > 40:
                        st.balloons()
                        st.success("Strategy C performs excellently! Win rate and profit factor both improved significantly")
                else:
                    st.warning("No trades generated, please adjust parameters")
                    
            except Exception as e:
                st.error(f"Backtest error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
