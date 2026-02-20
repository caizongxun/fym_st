import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import (
    CryptoDataLoader, LiquiditySweepDetector, 
    FeatureEngineer, ModelTrainer
)

def render():
    st.title("流動性掃蕩分析")
    
    st.markdown("""
    ### 機構級市場微觀結構分析
    
    此系統偵測 **Smart Money** 觸發散戶停損單的流動性掃蕩事件。
    
    **三大支柱**:
    1. 價格行為: 假突破 + 長影線 (2x 實體)
    2. OI 銳減: 未平倉量下降 > 2σ
    3. CVD 背離: 成交量差背離
    """)
    
    st.markdown("---")
    
    # 配置
    with st.expander("數據配置", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loader = CryptoDataLoader()
            symbol = st.selectbox(
                "交易對", 
                loader.get_available_symbols(), 
                index=10  # BTCUSDT
            )
            timeframe = st.selectbox(
                "時間框架", 
                loader.get_available_timeframes(), 
                index=1  # 1h
            )
        
        with col2:
            days = st.number_input(
                "回測天數", 
                value=90, 
                min_value=30, 
                max_value=365, 
                step=30
            )
            direction = st.radio(
                "信號方向",
                ["lower (做多)", "upper (做空)"],
                help="lower=偵測低點掃蕩(做多), upper=偵測高點掃蕩(做空)"
            )
            direction_key = 'lower' if 'lower' in direction else 'upper'
        
        with col3:
            include_oi = st.checkbox("OI 數據", value=True)
            include_funding = st.checkbox("資金費率", value=True)
    
    with st.expander("偵測參數", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            lookback_period = st.slider(
                "支撐/壓力回期", 
                min_value=20, 
                max_value=100, 
                value=50, 
                step=10
            )
            wick_multiplier = st.slider(
                "影線倍數", 
                min_value=1.0, 
                max_value=3.0, 
                value=2.0, 
                step=0.1,
                help="影線長度 > 倍數 x 實體"
            )
        
        with col2:
            oi_std_threshold = st.slider(
                "OI 銳減門檻 (σ)", 
                min_value=1.0, 
                max_value=3.0, 
                value=2.0, 
                step=0.1,
                help="OI 下降 > N 倍標準差"
            )
            cvd_divergence_lookback = st.slider(
                "CVD 背離回期", 
                min_value=5, 
                max_value=20, 
                value=10, 
                step=1
            )
    
    # 模式選擇
    with st.expander("模型整合 (可選)", expanded=False):
        use_model = st.checkbox("使用 ML 模型過濾", value=False)
        
        if use_model:
            model_files = [f for f in os.listdir("models") if f.endswith('.pkl')] if os.path.exists("models") else []
            
            if len(model_files) > 0:
                model_file = st.selectbox("選擇模型", sorted(model_files, reverse=True))
                probability_threshold = st.slider(
                    "機率門檻", 
                    min_value=0.5, 
                    max_value=0.8, 
                    value=0.65, 
                    step=0.01
                )
            else:
                st.warning("未找到模型文件")
                use_model = False
    
    if st.button("運行分析", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. 載入數據
            status_text.text("載入數據...")
            progress_bar.progress(10)
            
            df = loader.fetch_latest_klines(
                symbol=symbol,
                timeframe=timeframe,
                days=int(days),
                include_oi=include_oi,
                include_funding=include_funding
            )
            
            st.success(f"載入 {len(df)} 筆數據, {df['open_time'].min()} ~ {df['open_time'].max()}")
            
            # 2. 偵測流動性掃蕩
            status_text.text("偵測流動性掃蕩...")
            progress_bar.progress(30)
            
            detector = LiquiditySweepDetector(
                lookback_period=lookback_period,
                wick_multiplier=wick_multiplier,
                oi_std_threshold=oi_std_threshold,
                cvd_divergence_lookback=cvd_divergence_lookback
            )
            
            df_sweep = detector.detect_liquidity_sweep(df, direction=direction_key)
            df_sweep = detector.calculate_sweep_features(df_sweep)
            
            # 統計每個條件
            has_wick = df_sweep[f'sweep_{direction_key}_wick'].sum()
            has_breach = df_sweep[f'sweep_{direction_key}_breach'].sum()
            has_oi_flush = df_sweep[f'sweep_{direction_key}_oi_flush'].sum()
            has_cvd_div = df_sweep[f'sweep_{direction_key}_cvd_div'].sum()
            final_signals = df_sweep[f'sweep_{direction_key}_signal'].sum()
            
            st.markdown("### 築選漏斗")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "長影線", 
                    has_wick, 
                    delta=f"{100*has_wick/len(df):.1f}%"
                )
            with col2:
                st.metric(
                    "突破點位", 
                    has_breach, 
                    delta=f"{100*has_breach/len(df):.1f}%"
                )
            with col3:
                st.metric(
                    "OI 銳減", 
                    has_oi_flush, 
                    delta=f"{100*has_oi_flush/len(df):.1f}%"
                )
            with col4:
                st.metric(
                    "CVD 背離", 
                    has_cvd_div, 
                    delta=f"{100*has_cvd_div/len(df):.1f}%"
                )
            with col5:
                st.metric(
                    "最終信號", 
                    final_signals, 
                    delta=f"{100*final_signals/len(df):.1f}%"
                )
            
            # 3. 模型過濾 (可選)
            if use_model and len(model_files) > 0:
                status_text.text("模型過濾...")
                progress_bar.progress(60)
                
                trainer = ModelTrainer()
                trainer.load_model(model_file)
                
                fe = FeatureEngineer()
                df_features = fe.build_features(df_sweep, include_liquidity_features=True)
                
                available_features = [f for f in trainer.feature_names if f in df_features.columns]
                X_pred = df_features[available_features].fillna(0).replace([np.inf, -np.inf], 0)
                
                probabilities = trainer.predict_proba(X_pred)
                df_features['win_probability'] = probabilities
                
                df_sweep = df_features.copy()
                
                # 雙重篩選
                signals = df_sweep[
                    (df_sweep[f'sweep_{direction_key}_signal']) &
                    (df_sweep['win_probability'] >= probability_threshold)
                ]
                
                st.info(f"模型過濾: {final_signals} → {len(signals)} (門檻: {probability_threshold})")
            else:
                signals = df_sweep[df_sweep[f'sweep_{direction_key}_signal']]
            
            progress_bar.progress(80)
            
            # 4. 顯示結果
            st.markdown("### 流動性掃蕩事件")
            
            if len(signals) > 0:
                st.success(f"偵測到 {len(signals)} 個流動性掃蕩事件")
                
                # 表格顯示
                display_cols = [
                    'open_time', 'close', 'low' if direction_key == 'lower' else 'high',
                    f'{"lower" if direction_key == "lower" else "upper"}_wick_ratio'
                ]
                
                if 'oi_change_pct' in signals.columns:
                    display_cols.append('oi_change_pct')
                if 'cvd_slope_5' in signals.columns:
                    display_cols.append('cvd_slope_5')
                if 'dist_to_support_pct' in signals.columns:
                    display_cols.append('dist_to_support_pct')
                if use_model and 'win_probability' in signals.columns:
                    display_cols.append('win_probability')
                
                available_display_cols = [c for c in display_cols if c in signals.columns]
                
                display_df = signals[available_display_cols].copy()
                display_df['open_time'] = display_df['open_time'].dt.strftime('%Y-%m-%d %H:%M')
                
                # 重命名欄位
                rename_map = {
                    'open_time': '時間',
                    'close': '收盤',
                    'low': '最低',
                    'high': '最高',
                    'lower_wick_ratio': '下影線比',
                    'upper_wick_ratio': '上影線比',
                    'oi_change_pct': 'OI變化%',
                    'cvd_slope_5': 'CVD斜率',
                    'dist_to_support_pct': '距支撐%',
                    'win_probability': '機率'
                }
                
                display_df = display_df.rename(columns=rename_map)
                st.dataframe(display_df, use_container_width=True)
                
                # 圖表
                st.markdown("### K 線圖 + 信號")
                
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=(f"{symbol} {timeframe}", "OI 變化", "CVD"),
                    vertical_spacing=0.08,
                    row_heights=[0.6, 0.2, 0.2]
                )
                
                # K 線
                fig.add_trace(
                    go.Candlestick(
                        x=df_sweep['open_time'],
                        open=df_sweep['open'],
                        high=df_sweep['high'],
                        low=df_sweep['low'],
                        close=df_sweep['close'],
                        name='K線'
                    ),
                    row=1, col=1
                )
                
                # 信號標記
                for idx, row in signals.iterrows():
                    if direction_key == 'lower':
                        fig.add_annotation(
                            x=row['open_time'],
                            y=row['low'],
                            text="▲",
                            showarrow=False,
                            font=dict(size=20, color='green'),
                            row=1, col=1
                        )
                    else:
                        fig.add_annotation(
                            x=row['open_time'],
                            y=row['high'],
                            text="▼",
                            showarrow=False,
                            font=dict(size=20, color='red'),
                            row=1, col=1
                        )
                
                # OI 變化
                if 'oi_change_pct' in df_sweep.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_sweep['open_time'],
                            y=df_sweep['oi_change_pct'] * 100,
                            mode='lines',
                            name='OI 變化%',
                            line=dict(color='orange')
                        ),
                        row=2, col=1
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                
                # CVD
                if 'cvd' in df_sweep.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_sweep['open_time'],
                            y=df_sweep['cvd'],
                            mode='lines',
                            name='CVD',
                            line=dict(color='blue')
                        ),
                        row=3, col=1
                    )
                
                fig.update_xaxes(title_text="時間", row=3, col=1)
                fig.update_yaxes(title_text="價格", row=1, col=1)
                fig.update_yaxes(title_text="%", row=2, col=1)
                fig.update_yaxes(title_text="CVD", row=3, col=1)
                
                fig.update_layout(height=900, showlegend=True, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # 下載
                csv = signals.to_csv(index=False)
                st.download_button(
                    label="下載 CSV",
                    data=csv,
                    file_name=f"liquidity_sweep_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("未偵測到流動性掃蕩事件")
                
                st.info("""
                **建議**:
                1. 增加回測天數 (90 → 180)
                2. 調低影線倍數 (2.0 → 1.5)
                3. 調低 OI 門檻 (2.0 → 1.5)
                4. 使用不同時間框架 (1h → 15m)
                """)
            
            progress_bar.progress(100)
            status_text.text("完成")
            
        except Exception as e:
            st.error(f"錯誤: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # 說明
    with st.expander("理論說明", expanded=False):
        st.markdown("""
        ### 流動性掃蕩與微觀結構耗竭理論
        
        #### 核心理念
        機構資金為了建立龐大部位,會主動推價格觸發散戶停損單,獲取流動性。
        
        #### 三大支柱
        
        **1. 價格行為**
        - 價格刺穿過去 N 根 K 線的高低點
        - K 線留下長影線 (影線 > 2x 實體)
        - 收盤價回到突破點之上/之下
        
        **2. OI 銳減**
        - 突破 K 線的 OI 下降 > 2σ (24h)
        - 代表散戶爆倉/停損
        - 燃料耗盡,趨勢無法延續
        
        **3. CVD 背離**
        - 做多: 價格新低 (LL) 但 CVD 較高低點 (HL)
        - 做空: 價格新高 (HH) 但 CVD 較低高點 (LH)
        - 機構在關鍵位置大量限價單吸收
        
        #### 優勢
        
        - **手續費**: 節省 60% (Maker vs Taker)
        - **盈虧比**: 1:2.5 ~ 1:4 (精確停損)
        - **數據維度**: 真實資金籌碼 (OI + CVD)
        - **適應性**: 盤整期也能運作
        
        #### 相關文檔
        
        - 完整理論: `docs/LIQUIDITY_SWEEP_THEORY.md`
        - 整合指南: `LIQUIDITY_SWEEP_INTEGRATION.md`
        """)