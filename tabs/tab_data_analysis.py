import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from ui.selectors import symbol_selector

def render_data_analysis_tab(loader):
    """
    Tab 1: 多時間框架數據分析
    
    功能:
    - 同時顯示 5m, 15m, 1h K線
    - 技術指標視覺化 (不用來篩選)
    - 成交量分析
    - 波動率統計
    """
    st.header("步驟 1: 多時間框架數據分析")
    
    st.info("""
    **系統架構**: Ensemble RL-Transformer
    
    這是第一步: 了解不同時間框架的價格行為
    - 5m: 精確進場點
    - 15m: 主交易時間框架
    - 1h: 趨勢判斷
    
    技術指標**只用來觀察**，不用來篩選交易
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 選擇幣種
        symbols = symbol_selector(loader, "data_analysis", multi=False)
        symbol = symbols[0]
        
        # 選擇時間範圍
        if isinstance(loader, BinanceDataLoader):
            days = st.slider(
                "時間範圍 (天)",
                min_value=1,
                max_value=30,
                value=7,
                key="data_days"
            )
        else:
            st.info("使用 HuggingFace 最近 7 天數據")
            days = 7
    
    with col2:
        # 顯示指標選項
        show_rsi = st.checkbox("顯示 RSI", value=True, key="show_rsi")
        show_macd = st.checkbox("顯示 MACD", value=True, key="show_macd")
        show_bb = st.checkbox("顯示布林帶", value=True, key="show_bb")
        show_volume = st.checkbox("顯示成交量", value=True, key="show_volume")
    
    if st.button("載入數據", key="load_data", type="primary"):
        with st.spinner(f"正在載入 {symbol} 數據..."):
            try:
                # 載入三個時間框架
                if isinstance(loader, BinanceDataLoader):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    df_5m = loader.load_historical_data(symbol, '5m', start_date, end_date)
                    df_15m = loader.load_historical_data(symbol, '15m', start_date, end_date)
                    df_1h = loader.load_historical_data(symbol, '1h', start_date, end_date)
                else:
                    df_5m = loader.load_klines(symbol, '5m').tail(days * 288)
                    df_15m = loader.load_klines(symbol, '15m').tail(days * 96)
                    df_1h = loader.load_klines(symbol, '1h').tail(days * 24)
                
                st.success(f"載入成功: 5m={len(df_5m)} | 15m={len(df_15m)} | 1h={len(df_1h)} 根K線")
                
                # 計算技術指標
                for df in [df_5m, df_15m, df_1h]:
                    # RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    
                    # MACD
                    exp1 = df['close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd'] = exp1 - exp2
                    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    
                    # 布林帶
                    df['bb_mid'] = df['close'].rolling(window=20).mean()
                    df['bb_std'] = df['close'].rolling(window=20).std()
                    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
                    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
                
                # 顯示圖表
                st.subheader("多時間框架視覺化")
                
                # 建立 3x1 子圖
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('1小時 K線', '15分鐘 K線', '5分鐘 K線'),
                    row_heights=[0.33, 0.33, 0.34]
                )
                
                # 1h K線
                fig.add_trace(
                    go.Candlestick(
                        x=df_1h.index,
                        open=df_1h['open'],
                        high=df_1h['high'],
                        low=df_1h['low'],
                        close=df_1h['close'],
                        name='1h'
                    ),
                    row=1, col=1
                )
                
                # 15m K線
                fig.add_trace(
                    go.Candlestick(
                        x=df_15m.index,
                        open=df_15m['open'],
                        high=df_15m['high'],
                        low=df_15m['low'],
                        close=df_15m['close'],
                        name='15m'
                    ),
                    row=2, col=1
                )
                
                # 5m K線
                fig.add_trace(
                    go.Candlestick(
                        x=df_5m.index,
                        open=df_5m['open'],
                        high=df_5m['high'],
                        low=df_5m['low'],
                        close=df_5m['close'],
                        name='5m'
                    ),
                    row=3, col=1
                )
                
                # 布林帶
                if show_bb:
                    for i, df in enumerate([df_1h, df_15m, df_5m], 1):
                        fig.add_trace(
                            go.Scatter(x=df.index, y=df['bb_upper'], 
                                     line=dict(color='gray', width=1, dash='dash'),
                                     showlegend=False, name='BB Upper'),
                            row=i, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=df.index, y=df['bb_lower'], 
                                     line=dict(color='gray', width=1, dash='dash'),
                                     showlegend=False, name='BB Lower'),
                            row=i, col=1
                        )
                
                fig.update_layout(
                    height=900,
                    xaxis_rangeslider_visible=False,
                    title_text=f"{symbol} 多時間框架分析"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 指標分析
                if show_rsi or show_macd:
                    st.subheader("技術指標 (15分鐘)")
                    
                    col1, col2 = st.columns(2)
                    
                    if show_rsi:
                        with col1:
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(
                                x=df_15m.index,
                                y=df_15m['rsi'],
                                mode='lines',
                                name='RSI'
                            ))
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                            fig_rsi.update_layout(title="RSI", height=300)
                            st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    if show_macd:
                        with col2:
                            fig_macd = go.Figure()
                            fig_macd.add_trace(go.Scatter(
                                x=df_15m.index,
                                y=df_15m['macd'],
                                mode='lines',
                                name='MACD'
                            ))
                            fig_macd.add_trace(go.Scatter(
                                x=df_15m.index,
                                y=df_15m['signal'],
                                mode='lines',
                                name='Signal'
                            ))
                            fig_macd.update_layout(title="MACD", height=300)
                            st.plotly_chart(fig_macd, use_container_width=True)
                
                # 統計資訊
                st.subheader("波動率統計")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    volatility_1h = df_1h['close'].pct_change().std() * 100
                    st.metric("1h 波動率", f"{volatility_1h:.2f}%")
                
                with col2:
                    volatility_15m = df_15m['close'].pct_change().std() * 100
                    st.metric("15m 波動率", f"{volatility_15m:.2f}%")
                
                with col3:
                    volatility_5m = df_5m['close'].pct_change().std() * 100
                    st.metric("5m 波動率", f"{volatility_5m:.2f}%")
                
                # 價格範圍
                st.subheader("價格範圍")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "1h 高/低",
                        f"${df_1h['high'].max():.2f} / ${df_1h['low'].min():.2f}"
                    )
                
                with col2:
                    st.metric(
                        "15m 高/低",
                        f"${df_15m['high'].max():.2f} / ${df_15m['low'].min():.2f}"
                    )
                
                with col3:
                    st.metric(
                        "5m 高/低",
                        f"${df_5m['high'].max():.2f} / ${df_5m['low'].min():.2f}"
                    )
                
                st.success("數據分析完成! 下一步: Tab 2 特徵工程")
                
            except Exception as e:
                st.error(f"載入失敗: {str(e)}")
                import traceback
                st.code(traceback.format_exc())