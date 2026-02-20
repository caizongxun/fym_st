import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import CryptoDataLoader

def render():
    st.title("Trading System Dashboard")
    
    st.markdown("""
    ## System Overview
    
    This automated trading system implements academic-grade quantitative methods:
    
    - **Triple Barrier Labeling**: Mathematically rigorous target definition
    - **Meta-Labeling**: Two-layer filtering to reduce false signals
    - **Fractional Differentiation**: Stationarity without losing memory
    - **Purged K-Fold CV**: Prevent data leakage in time series
    - **Dynamic Kelly Criterion**: Probability-based position sizing
    - **Sample Weighting**: Focus on high-impact trades
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Available Symbols", "38")
    
    with col2:
        st.metric("Timeframes", "3 (15m, 1h, 1d)")
    
    with col3:
        st.metric("Data Source", "HuggingFace")
    
    st.markdown("---")
    
    st.subheader("Quick Data Preview")
    
    loader = CryptoDataLoader()
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Symbol", loader.get_available_symbols(), index=10)
    with col2:
        timeframe = st.selectbox("Timeframe", loader.get_available_timeframes())
    
    if st.button("Load Preview"):
        with st.spinner("Loading data..."):
            try:
                df = loader.load_klines(symbol, timeframe)
                
                st.success(f"Loaded {len(df)} rows")
                
                st.dataframe(df.tail(20), use_container_width=True)
                
                fig = go.Figure(data=[go.Candlestick(
                    x=df['open_time'].tail(100),
                    open=df['open'].tail(100),
                    high=df['high'].tail(100),
                    low=df['low'].tail(100),
                    close=df['close'].tail(100)
                )])
                
                fig.update_layout(
                    title=f"{symbol} {timeframe} - Last 100 Bars",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")