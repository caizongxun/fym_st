import streamlit as st
import pandas as pd
import os
import sys
import plotly.graph_objects as go
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import (
    CryptoDataLoader, FeatureEngineer, ModelTrainer,
    KellyCriterion, RealtimePredictor
)

def render():
    st.title("Live Prediction")
    
    st.markdown("""
    Generate real-time predictions using completed K-bars.
    
    Important: Predictions use ONLY completed bars to avoid lookahead bias.
    """)
    
    st.markdown("---")
    
    with st.expander("Prediction Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            model_files = [f for f in os.listdir("trading_system/models") if f.endswith('.pkl')] if os.path.exists("trading_system/models") else []
            
            if len(model_files) == 0:
                st.warning("No trained models found. Train a model first.")
                return
            
            model_file = st.selectbox("Select Model", model_files)
            
            loader = CryptoDataLoader()
            symbol = st.selectbox("Symbol", loader.get_available_symbols(), index=10)
            timeframe = st.selectbox("Timeframe", loader.get_available_timeframes(), index=1)
        
        with col2:
            tp_multiplier = st.number_input("TP Multiplier", value=2.5, step=0.1)
            sl_multiplier = st.number_input("SL Multiplier", value=1.5, step=0.1)
            kelly_fraction = st.number_input("Kelly Fraction", value=0.5, step=0.1)
            lookback_bars = st.number_input("Lookback Bars", value=200, step=50)
    
    col1, col2 = st.columns(2)
    with col1:
        run_prediction = st.button("Get Latest Signal", type="primary")
    with col2:
        show_all_signals = st.button("Show All Recent Signals")
    
    if run_prediction or show_all_signals:
        try:
            with st.spinner("Loading model and data..."):
                trainer = ModelTrainer()
                trainer.load_model(model_file)
                
                df = loader.load_klines(symbol, timeframe)
                df = df.tail(int(lookback_bars) + 100)
                
                feature_engineer = FeatureEngineer()
                kelly = KellyCriterion(tp_multiplier, sl_multiplier, kelly_fraction)
                predictor = RealtimePredictor(trainer, feature_engineer, kelly)
            
            if run_prediction:
                latest_signal = predictor.get_latest_signal(df)
                
                if latest_signal is None:
                    st.warning("No trading signal detected at current bar")
                else:
                    st.success("Trading Signal Detected")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Timestamp", latest_signal['timestamp'].strftime("%Y-%m-%d %H:%M"))
                        st.metric("Price", f"${latest_signal['price']:.2f}")
                    with col2:
                        st.metric("Win Probability", f"{latest_signal['win_probability']*100:.1f}%")
                        st.metric("Position Size", f"{latest_signal['position_size']*100:.1f}%")
                    with col3:
                        st.metric("ATR", f"{latest_signal['atr']:.4f}")
                        st.metric("RSI", f"{latest_signal['rsi']:.1f}")
                    
                    tp_price = latest_signal['price'] + (tp_multiplier * latest_signal['atr'])
                    sl_price = latest_signal['price'] - (sl_multiplier * latest_signal['atr'])
                    
                    st.markdown("### Trade Levels")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Entry", f"${latest_signal['price']:.2f}")
                    with col2:
                        st.metric("Take Profit", f"${tp_price:.2f}", f"+{((tp_price/latest_signal['price']-1)*100):.2f}%")
                    with col3:
                        st.metric("Stop Loss", f"${sl_price:.2f}", f"{((sl_price/latest_signal['price']-1)*100):.2f}%")
            
            if show_all_signals:
                predictions = predictor.predict_from_completed_bars(df)
                recent_signals = predictions[predictions['signal'] == 1].tail(20)
                
                if len(recent_signals) == 0:
                    st.warning("No signals found in recent data")
                else:
                    st.info(f"Found {len(recent_signals)} signals in last {lookback_bars} bars")
                    
                    display_cols = ['open_time', 'close', 'win_probability', 'position_size', 'atr', 'rsi', 'macd']
                    display_df = recent_signals[display_cols].copy()
                    display_df['win_probability'] = (display_df['win_probability'] * 100).round(1)
                    display_df['position_size'] = (display_df['position_size'] * 100).round(1)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    fig = go.Figure()
                    
                    lookback_df = df.tail(int(lookback_bars))
                    
                    fig.add_trace(go.Candlestick(
                        x=lookback_df['open_time'],
                        open=lookback_df['open'],
                        high=lookback_df['high'],
                        low=lookback_df['low'],
                        close=lookback_df['close'],
                        name='Price'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=recent_signals['open_time'],
                        y=recent_signals['close'],
                        mode='markers',
                        marker=dict(size=10, color='green', symbol='triangle-up'),
                        name='Signals'
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} {timeframe} with Signals",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())