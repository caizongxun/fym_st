import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from ui.selectors import symbol_selector

def render_bb_visualization_tab(loader):
    """
    Tab 1: BB 反轉標籤視覺化
    顯示模型將要學習的反轉點標籤
    """
    st.header("步驟 1: BB 反轉標籤視覺化")
    
    st.info("""
    **標籤定義 (相對波動)**: 
    - 上軌反轉: 碰上軌後,未來下跌幅度 > 前 N 根平均波動的 X 倍
    - 下軌反轉: 碰下軌後,未來上漲幅度 > 前 N 根平均波動的 X 倍
    
    這樣可以過濾趨勢中的正常波動,只保留真正的反轉點
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        symbols = symbol_selector(loader, "bb_viz", multi=False)
        symbol = symbols[0]
        
        if isinstance(loader, BinanceDataLoader):
            days = st.slider("天數", 1, 30, 14, key="bb_days")
        else:
            st.info("使用 HuggingFace 最近 14 天數據")
            days = 14
    
    with col2:
        bb_period = st.number_input("BB 週期", 10, 50, 20, key="bb_period")
        bb_std = st.number_input("BB 標準差", 1.0, 3.0, 2.0, 0.1, key="bb_std")
        
    col3, col4 = st.columns([1, 1])
    
    with col3:
        lookback_period = st.number_input(
            "歷史波動期數 (N)",
            5, 50, 20,
            key="lookback_period",
            help="計算前 N 根 K 棒的平均波動"
        )
    
    with col4:
        volatility_multiplier = st.number_input(
            "波動倍數 (X)",
            1.0, 5.0, 2.0, 0.5,
            key="volatility_multiplier",
            help="未來反轉幅度需 > 歷史波動的 X 倍"
        )
    
    if st.button("載入數據", key="load_bb_data", type="primary"):
        with st.spinner(f"正在載入 {symbol} 數據..."):
            try:
                if isinstance(loader, BinanceDataLoader):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    df = loader.load_historical_data(symbol, '15m', start_date, end_date)
                else:
                    df = loader.load_klines(symbol, '15m').tail(days * 96)
                
                st.success(f"載入 {len(df)} 根 K 線")
                
                # 計算 BB
                df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
                df['bb_std'] = df['close'].rolling(window=bb_period).std()
                df['bb_upper'] = df['bb_mid'] + bb_std * df['bb_std']
                df['bb_lower'] = df['bb_mid'] - bb_std * df['bb_std']
                
                # 計算歷史平均波動率 (ATR 概念)
                df['price_range'] = df['high'] - df['low']
                df['avg_historical_volatility'] = df['price_range'].rolling(window=lookback_period).mean()
                df['volatility_threshold'] = df['avg_historical_volatility'] * volatility_multiplier
                
                # 定義碰觸上下軌
                df['touch_upper'] = df['high'] >= df['bb_upper'] * 0.999
                df['touch_lower'] = df['low'] <= df['bb_lower'] * 1.001
                
                # 計算未來 5 根 K 棒的價格變化 (絕對值)
                df['future_5bar_min'] = df['low'].shift(-5).rolling(window=5, min_periods=1).min()
                df['future_5bar_max'] = df['high'].shift(-5).rolling(window=5, min_periods=1).max()
                
                # 計算未來反轉幅度 (絕對值)
                df['future_drop'] = df['high'] - df['future_5bar_min']
                df['future_rise'] = df['future_5bar_max'] - df['low']
                
                # 定義反轉標籤 (相對波動)
                # 上軌反轉: 未來下跌 > 歷史平均波動 * 倍數
                df['label_upper_reversal'] = (
                    df['touch_upper'] & 
                    (df['future_drop'] > df['volatility_threshold'])
                )
                
                # 下軌反轉: 未來上漲 > 歷史平均波動 * 倍數
                df['label_lower_reversal'] = (
                    df['touch_lower'] & 
                    (df['future_rise'] > df['volatility_threshold'])
                )
                
                # 統計
                upper_touches = df['touch_upper'].sum()
                upper_reversals = df['label_upper_reversal'].sum()
                lower_touches = df['touch_lower'].sum()
                lower_reversals = df['label_lower_reversal'].sum()
                
                # 繪圖
                fig = go.Figure()
                
                # K 線
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='價格',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ))
                
                # BB 通道
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['bb_upper'],
                    line=dict(color='rgba(255,0,0,0.5)', width=2),
                    name='BB 上軌'
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['bb_mid'],
                    line=dict(color='gray', width=1, dash='dash'),
                    name='BB 中軌'
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['bb_lower'],
                    line=dict(color='rgba(0,255,0,0.5)', width=2),
                    name='BB 下軌'
                ))
                
                # 標記上軌反轉點
                upper_reversal_points = df[df['label_upper_reversal']]
                if len(upper_reversal_points) > 0:
                    fig.add_trace(go.Scatter(
                        x=upper_reversal_points.index,
                        y=upper_reversal_points['high'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red',
                            line=dict(color='darkred', width=2)
                        ),
                        name=f'上軌反轉 ({len(upper_reversal_points)})',
                        text=[f"反轉點<br>時間: {idx}<br>價格: {row['high']:.2f}<br>未來下跌: {row['future_drop']:.2f}<br>波動閾值: {row['volatility_threshold']:.2f}" 
                              for idx, row in upper_reversal_points.iterrows()],
                        hoverinfo='text'
                    ))
                
                # 標記下軌反轉點
                lower_reversal_points = df[df['label_lower_reversal']]
                if len(lower_reversal_points) > 0:
                    fig.add_trace(go.Scatter(
                        x=lower_reversal_points.index,
                        y=lower_reversal_points['low'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green',
                            line=dict(color='darkgreen', width=2)
                        ),
                        name=f'下軌反轉 ({len(lower_reversal_points)})',
                        text=[f"反轉點<br>時間: {idx}<br>價格: {row['low']:.2f}<br>未來上漲: {row['future_rise']:.2f}<br>波動閾值: {row['volatility_threshold']:.2f}" 
                              for idx, row in lower_reversal_points.iterrows()],
                        hoverinfo='text'
                    ))
                
                fig.update_layout(
                    title=f"{symbol} BB 反轉標籤 (相對波動法)",
                    xaxis_title="時間",
                    yaxis_title="價格",
                    height=700,
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 統計分析
                st.subheader("反轉標籤統計")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("碰上軌次數", upper_touches)
                
                with col2:
                    upper_reversal_rate = (upper_reversals / upper_touches * 100) if upper_touches > 0 else 0
                    st.metric(
                        "上軌反轉標籤", 
                        upper_reversals,
                        delta=f"{upper_reversal_rate:.1f}%"
                    )
                
                with col3:
                    st.metric("碰下軌次數", lower_touches)
                
                with col4:
                    lower_reversal_rate = (lower_reversals / lower_touches * 100) if lower_touches > 0 else 0
                    st.metric(
                        "下軌反轉標籤", 
                        lower_reversals,
                        delta=f"{lower_reversal_rate:.1f}%"
                    )
                
                # 波動分析
                st.subheader("波動分析")
                col1, col2 = st.columns(2)
                
                with col1:
                    avg_volatility = df['avg_historical_volatility'].mean()
                    st.metric(
                        f"平均歷史波動 ({lookback_period}根)",
                        f"{avg_volatility:.2f}"
                    )
                
                with col2:
                    avg_threshold = df['volatility_threshold'].mean()
                    st.metric(
                        f"平均反轉閾值 ({volatility_multiplier}x)",
                        f"{avg_threshold:.2f}"
                    )
                
                # 詳細數據表
                st.subheader("反轉點詳細數據")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(upper_reversal_points) > 0:
                        st.write("上軌反轉點 (做空信號)")
                        upper_display = upper_reversal_points[[
                            'high', 'bb_upper', 'future_drop', 'volatility_threshold'
                        ]].copy()
                        upper_display.columns = ['價格', 'BB上軌', '未來下跌', '波動閾值']
                        upper_display['倍數'] = (upper_display['未來下跌'] / upper_display['波動閾值']).round(2)
                        st.dataframe(upper_display.tail(10), use_container_width=True)
                    else:
                        st.info("無上軌反轉點")
                
                with col2:
                    if len(lower_reversal_points) > 0:
                        st.write("下軌反轉點 (做多信號)")
                        lower_display = lower_reversal_points[[
                            'low', 'bb_lower', 'future_rise', 'volatility_threshold'
                        ]].copy()
                        lower_display.columns = ['價格', 'BB下軌', '未來上漲', '波動閾值']
                        lower_display['倍數'] = (lower_display['未來上漲'] / lower_display['波動閾值']).round(2)
                        st.dataframe(lower_display.tail(10), use_container_width=True)
                    else:
                        st.info("無下軌反轉點")
                
                st.success(
                    f"標籤生成完成! 共 {upper_reversals + lower_reversals} 個反轉點\n"
                    f"使用相對波動法,過濾了趨勢中的正常波動"
                )
                
            except Exception as e:
                st.error(f"載入失敗: {str(e)}")
                import traceback
                st.code(traceback.format_exc())