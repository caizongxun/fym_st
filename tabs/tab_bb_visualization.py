import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from ui.selectors import symbol_selector

def render_bb_visualization_tab(loader):
    """
    Tab 1: BB 通道視覺化
    觀察價格在 BB 上下軌的行為
    """
    st.header("步驟 1: BB 通道視覺化")
    
    st.info("""
    **策略核心**: 精準捕捉 BB 反轉點
    
    觀察重點:
    - 價格碰到上軌後是否反彈 (做空機會)
    - 價格碰到下軌後是否反彈 (做多機會)
    - 強趨勢時價格會沿著軌道走 (不要交易)
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        symbols = symbol_selector(loader, "bb_viz", multi=False)
        symbol = symbols[0]
        
        if isinstance(loader, BinanceDataLoader):
            days = st.slider("天數", 1, 30, 7, key="bb_days")
        else:
            st.info("使用 HuggingFace 最近 7 天數據")
            days = 7
    
    with col2:
        bb_period = st.number_input("BB 週期", 10, 50, 20, key="bb_period")
        bb_std = st.number_input("BB 標準差", 1.0, 3.0, 2.0, 0.1, key="bb_std")
        
        show_touch_points = st.checkbox("標記碰觸點", True, key="show_touch")
    
    if st.button("載入數據", key="load_bb_data", type="primary"):
        with st.spinner(f"正在載入 {symbol} 數據..."):
            try:
                if isinstance(loader, BinanceDataLoader):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    df = loader.load_historical_data(symbol, '15m', start_date, end_date)
                else:
                    df = loader.load_klines(symbol, '15m').tail(days * 96)
                
                # 計算 BB
                df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
                df['bb_std'] = df['close'].rolling(window=bb_period).std()
                df['bb_upper'] = df['bb_mid'] + bb_std * df['bb_std']
                df['bb_lower'] = df['bb_mid'] - bb_std * df['bb_std']
                
                # 檢測碰觸上下軌
                df['touch_upper'] = df['high'] >= df['bb_upper']
                df['touch_lower'] = df['low'] <= df['bb_lower']
                
                # 檢測反彈 (碰觸後 5 根 K 棒內是否反向移動 > 0.3%)
                df['bounce_from_upper'] = False
                df['bounce_from_lower'] = False
                
                for i in range(len(df) - 5):
                    if df['touch_upper'].iloc[i]:
                        future_low = df['low'].iloc[i+1:i+6].min()
                        if (df['high'].iloc[i] - future_low) / df['high'].iloc[i] > 0.003:
                            df.loc[df.index[i], 'bounce_from_upper'] = True
                    
                    if df['touch_lower'].iloc[i]:
                        future_high = df['high'].iloc[i+1:i+6].max()
                        if (future_high - df['low'].iloc[i]) / df['low'].iloc[i] > 0.003:
                            df.loc[df.index[i], 'bounce_from_lower'] = True
                
                # 繪圖
                fig = go.Figure()
                
                # K 線
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='價格'
                ))
                
                # BB 通道
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['bb_upper'],
                    line=dict(color='red', width=1),
                    name='上軌'
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['bb_mid'],
                    line=dict(color='gray', width=1, dash='dash'),
                    name='中軌'
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['bb_lower'],
                    line=dict(color='green', width=1),
                    name='下軌'
                ))
                
                # 標記碰觸點
                if show_touch_points:
                    upper_bounces = df[df['bounce_from_upper']]
                    lower_bounces = df[df['bounce_from_lower']]
                    
                    fig.add_trace(go.Scatter(
                        x=upper_bounces.index,
                        y=upper_bounces['high'],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        name='上軌反彈'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=lower_bounces.index,
                        y=lower_bounces['low'],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        name='下軌反彈'
                    ))
                
                fig.update_layout(
                    title=f"{symbol} BB 通道分析",
                    xaxis_title="時間",
                    yaxis_title="價格",
                    height=600,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 統計分析
                st.subheader("反彈統計")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    upper_touches = df['touch_upper'].sum()
                    st.metric("碰上軌次數", upper_touches)
                
                with col2:
                    upper_bounces = df['bounce_from_upper'].sum()
                    upper_bounce_rate = upper_bounces / upper_touches * 100 if upper_touches > 0 else 0
                    st.metric("上軌反彈次數", upper_bounces)
                    st.caption(f"反彈率: {upper_bounce_rate:.1f}%")
                
                with col3:
                    lower_touches = df['touch_lower'].sum()
                    st.metric("碰下軌次數", lower_touches)
                
                with col4:
                    lower_bounces = df['bounce_from_lower'].sum()
                    lower_bounce_rate = lower_bounces / lower_touches * 100 if lower_touches > 0 else 0
                    st.metric("下軌反彈次數", lower_bounces)
                    st.caption(f"反彈率: {lower_bounce_rate:.1f}%")
                
                st.success("視覺化完成! 下一步: Tab 2 訓練反轉預測模型")
                
            except Exception as e:
                st.error(f"載入失敗: {str(e)}")
                import traceback
                st.code(traceback.format_exc())