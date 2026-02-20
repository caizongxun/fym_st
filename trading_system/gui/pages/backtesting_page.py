import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import (
    CryptoDataLoader, FeatureEngineer, ModelTrainer,
    KellyCriterion, RiskManager, Backtester, RealtimePredictor
)

def render():
    st.title("Backtesting")
    
    st.markdown("""
    Test your trained model on historical data with realistic trading conditions:
    - Commission and slippage simulation
    - Dynamic position sizing via Kelly Criterion
    - Performance metrics and equity curve
    """)
    
    st.markdown("---")
    
    with st.expander("Backtest Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_files = [f for f in os.listdir("trading_system/models") if f.endswith('.pkl')] if os.path.exists("trading_system/models") else []
            
            if len(model_files) == 0:
                st.warning("No trained models found. Train a model first.")
                return
            
            model_file = st.selectbox("Select Model", model_files)
            
            loader = CryptoDataLoader()
            symbol = st.selectbox("Test Symbol", loader.get_available_symbols(), index=10)
            timeframe = st.selectbox("Timeframe", loader.get_available_timeframes(), index=1)
        
        with col2:
            initial_capital = st.number_input("Initial Capital", value=10000.0, step=1000.0)
            commission_rate = st.number_input("Commission Rate", value=0.001, step=0.0001, format="%.4f")
            slippage = st.number_input("Slippage", value=0.0005, step=0.0001, format="%.4f")
        
        with col3:
            tp_multiplier = st.number_input("TP Multiplier", value=2.5, step=0.1)
            sl_multiplier = st.number_input("SL Multiplier", value=1.5, step=0.1)
            kelly_fraction = st.number_input("Kelly Fraction", value=0.5, step=0.1)
    
    if st.button("Run Backtest", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Loading model...")
            progress_bar.progress(10)
            trainer = ModelTrainer()
            trainer.load_model(model_file)
            
            status_text.text("Loading data...")
            progress_bar.progress(20)
            df = loader.load_klines(symbol, timeframe)
            
            status_text.text("Building features...")
            progress_bar.progress(35)
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.build_features(df)
            
            status_text.text("Generating predictions...")
            progress_bar.progress(50)
            kelly = KellyCriterion(tp_multiplier, sl_multiplier, kelly_fraction)
            predictor = RealtimePredictor(trainer, feature_engineer, kelly)
            predictions = predictor.predict_from_completed_bars(df_features)
            
            signals = predictions[predictions['signal'] == 1].copy()
            st.info(f"Generated {len(signals)} trading signals")
            
            if len(signals) == 0:
                st.warning("No signals generated. Try adjusting parameters.")
                return
            
            status_text.text("Running backtest...")
            progress_bar.progress(70)
            backtester = Backtester(initial_capital, commission_rate, slippage)
            results = backtester.run_backtest(signals, tp_multiplier=tp_multiplier, sl_multiplier=sl_multiplier)
            
            progress_bar.progress(100)
            status_text.text("Backtest complete")
            
            st.success("Backtest completed successfully")
            
            stats = results['statistics']
            trades_df = results['trades']
            
            st.markdown("### Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{stats['total_return']*100:.2f}%")
                st.metric("Total Trades", stats['total_trades'])
            with col2:
                st.metric("Win Rate", f"{stats['win_rate']*100:.2f}%")
                st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{stats['max_drawdown']*100:.2f}%")
                st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
            with col4:
                st.metric("Avg Win", f"${stats['avg_win']:.2f}")
                st.metric("Avg Loss", f"${stats['avg_loss']:.2f}")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Equity Curve", "Drawdown"),
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(trades_df))),
                    y=trades_df['capital'],
                    mode='lines',
                    name='Capital',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            trades_df['cumulative_return'] = (trades_df['capital'] - initial_capital) / initial_capital
            trades_df['drawdown'] = trades_df['cumulative_return'] - trades_df['cumulative_return'].cummax()
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(trades_df))),
                    y=trades_df['drawdown'] * 100,
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Trade Number", row=2, col=1)
            fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            
            fig.update_layout(height=700, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Trade History")
            st.dataframe(trades_df.tail(50), use_container_width=True)
            
        except Exception as e:
            st.error(f"Backtest failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())