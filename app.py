import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from io import StringIO

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from utils.bb_reversal_detector import BBReversalDetector
from utils.bb_reversal_features import BBReversalFeatureExtractor
from models.train_bb_reversal_model import BBReversalModelTrainer
from backtesting.engine import BacktestEngine

st.set_page_config(page_title="AI 加密貨幣交易儀表板", layout="wide")
st.title("🚀 AI 加密貨幣交易儀表板 - BB反轉系統")

st.sidebar.title("設定")
data_source = st.sidebar.radio(
    "資料源",
    ["HuggingFace (38幣)", "Binance API (即時)"],
    help="HuggingFace: 離線資料,快速穩定\nBinance: 即時資料,需網絡"
)

if data_source == "HuggingFace (38幣)":
    loader = HuggingFaceKlineLoader()
    st.sidebar.success("使用HuggingFace離線資料")
else:
    loader = BinanceDataLoader()
    st.sidebar.info("使用Binance即時資料")

st.sidebar.success("""
**BB反轉系統**

🎯 核心功能:
- 過濾走勢中觸碰
- 確認有效反轉
- 智能標記反轉點
- LightGBM訓練

✨ 特點:
- 只學習有效反轉
- 過濾假突破
- 走勢自動判斷
- 高準確率預測
""")

def calculate_atr(df_signals):
    high_low = df_signals['high'] - df_signals['low']
    high_close = abs(df_signals['high'] - df_signals['close'].shift(1))
    low_close = abs(df_signals['low'] - df_signals['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    atr = atr.bfill().fillna(df_signals['close'] * 0.02)
    return atr

def symbol_selector(key_prefix: str, multi: bool = False, default_symbols: list = None):
    if data_source == "HuggingFace (38幣)":
        symbol_groups = HuggingFaceKlineLoader.get_symbol_groups()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selection_mode = st.radio(
                "選擇模式",
                ["熱門Top10", "按分類", "手動輸入"],
                key=f"{key_prefix}_mode"
            )
        
        with col2:
            if selection_mode == "熱門Top10":
                top_symbols = HuggingFaceKlineLoader.get_top_symbols(10)
                if multi:
                    selected = st.multiselect(
                        "選擇幣種",
                        top_symbols,
                        default=default_symbols or top_symbols[:2],
                        key=f"{key_prefix}_top"
                    )
                else:
                    selected = [st.selectbox(
                        "選擇幣種",
                        top_symbols,
                        key=f"{key_prefix}_top_single"
                    )]
            
            elif selection_mode == "按分類":
                category = st.selectbox(
                    "選擇分類",
                    list(symbol_groups.keys()),
                    key=f"{key_prefix}_category"
                )
                symbols_in_category = symbol_groups[category]
                
                if multi:
                    selected = st.multiselect(
                        f"{category} 幣種",
                        symbols_in_category,
                        default=default_symbols or symbols_in_category[:2],
                        key=f"{key_prefix}_cat_multi"
                    )
                else:
                    selected = [st.selectbox(
                        f"{category} 幣種",
                        symbols_in_category,
                        key=f"{key_prefix}_cat_single"
                    )]
            
            else:
                if multi:
                    text_input = st.text_area(
                        "輸入幣種 (逗號分隔)",
                        value=",".join(default_symbols) if default_symbols else "BTCUSDT,ETHUSDT",
                        key=f"{key_prefix}_manual",
                        height=100
                    )
                    selected = [s.strip().upper() for s in text_input.split(',') if s.strip()]
                else:
                    selected = [st.text_input(
                        "輸入幣種",
                        value="BTCUSDT",
                        key=f"{key_prefix}_manual_single"
                    ).strip().upper()]
        
        return selected
    
    else:
        if multi:
            text_input = st.text_area(
                "交易對 (逗號分隔)",
                value="BTCUSDT,ETHUSDT",
                key=f"{key_prefix}_binance"
            )
            return [s.strip().upper() for s in text_input.split(',') if s.strip()]
        else:
            return [st.text_input(
                "交易對",
                value="BTCUSDT",
                key=f"{key_prefix}_binance_single"
            ).strip().upper()]

tabs = st.tabs(["BB反轉視覺化", "BB反轉訓練"])

with tabs[0]:
    st.header("BB反轉點視覺化")
    
    st.info("""
    **BB觸碰反轉定義**:
    1. 價格觸碰BB上軌/下軌
    2. 過濾走勢中的觸碰 (假突破)
    3. 隨後N根K線出現有效反轉
    4. 確認回到BB中軌附近
    
    **標記說明**:
    - 紅色三角: 上軌觸碰後下跌反轉 (做空機會)
    - 綠色三角: 下軌觸碰後上漨反轉 (做多機會)
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        viz_symbols = symbol_selector("bb_viz", multi=False)
        viz_symbol = viz_symbols[0]
        viz_days = st.slider("顯示天數", min_value=3, max_value=30, value=7, key="viz_days")
        viz_candles = viz_days * 96
    
    with col2:
        st.subheader("參數設定")
        bb_period = st.number_input("BB周期", min_value=10, max_value=50, value=20, key="bb_period")
        bb_std = st.number_input("BB標準差", min_value=1.0, max_value=3.0, value=2.0, step=0.5, key="bb_std")
        touch_threshold = st.slider("觸碰閾值 (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="touch_threshold") / 100
        min_reversal = st.slider("最小反轉幅度 (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="min_reversal") / 100
    
    if st.button("生成BB反轉點圖表", key="gen_bb_viz", type="primary"):
        with st.spinner(f"載入 {viz_symbol} 數據..."):
            try:
                df = loader.load_klines(viz_symbol, '15m')
                df = df.tail(viz_candles)
                
                detector = BBReversalDetector(
                    bb_period=bb_period,
                    bb_std=bb_std,
                    touch_threshold=touch_threshold,
                    reversal_confirm_candles=5,
                    min_reversal_pct=min_reversal,
                    trend_filter_enabled=True,
                    trend_lookback=10,
                    require_middle_return=True
                )
                
                df_result = detector.detect_reversals(df)
                
                st.subheader(f"{viz_symbol} BB反轉點分析")
                
                stats = detector.get_statistics(df_result)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("總反轉點", stats['total_reversals'])
                with col2:
                    st.metric("上軌反轉", f"{stats['upper_reversals']} ({stats['upper_success_rate']:.1f}%)")
                with col3:
                    st.metric("下軌反轉", f"{stats['lower_reversals']} ({stats['lower_success_rate']:.1f}%)")
                
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("平均反轉幅度", f"{stats['avg_reversal_pct']:.2%}")
                    st.metric("總拒絕數", stats['total_rejected'])
                with col5:
                    if 'rejection_reasons' in stats and stats['rejection_reasons']:
                        st.write("拒絕原因:")
                        for reason, count in stats['rejection_reasons'].items():
                            st.text(f"- {reason}: {count}")
                
                fig = detector.plot_reversals(df_result, n_candles=viz_candles, title=f"{viz_symbol} BB反轉點檢測")
                st.plotly_chart(fig, use_container_width=True)
                
                if len(detector.reversals) > 0:
                    st.subheader("反轉點明細")
                    reversals_df = pd.DataFrame(detector.reversals)
                    reversals_df['reversal_pct'] = reversals_df['reversal_pct'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(reversals_df[['time', 'type', 'reversal_type', 'touch_price', 'target_price', 'reversal_pct']], use_container_width=True)
                else:
                    st.warning("沒有檢測到符合條件的反轉點，請降低最小反轉幅度或調整參數")
                    
            except Exception as e:
                st.error(f"錯誤: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

with tabs[1]:
    st.header("BB反轉點模型訓練")
    
    st.success("""
    **訓練原理**:
    1. 使用BB反轉檢測器築選有效反轉點
    2. 過濾走勢中的假突破
    3. 確認價格回到中軌附近
    4. 只學習真正有效的反轉
    
    **標籤定義**:
    - 上軌反轉 -> 做空 (0)
    - 下軌反轉 -> 做多 (1)
    """)
    
    train_symbols = symbol_selector("bb_train", multi=False)
    train_symbol = train_symbols[0]
    
    train_candles = st.number_input(
        "訓練K棒數量",
        min_value=10000,
        max_value=50000,
        value=20000,
        step=5000,
        key="train_candles",
        help="建議至少20000根以獲取足夠的有效反轉點"
    )
    
    st.caption(f"預估訓練時間: 約1-2分鐘")
    
    if st.button("開始訓練BB反轉模型", key="train_bb_btn", type="primary"):
        with st.spinner(f"正在訓練 {train_symbol} BB反轉模型..."):
            try:
                # 載入數據
                df = loader.load_klines(train_symbol, '15m')
                df = df.tail(train_candles)
                
                st.info(f"載入 {len(df)} 根K棒")
                
                # 特徵提取
                extractor = BBReversalFeatureExtractor(
                    bb_period=20,
                    bb_std=2.0,
                    rsi_period=14
                )
                
                df_processed = extractor.process(df, create_labels=True)
                
                # 獲取反轉點統計
                reversal_stats = extractor.get_reversal_statistics()
                
                st.info(f"特徵工程完成: {len(df_processed)} 有效樣本")
                st.info(f"檢測到 {reversal_stats['total_reversals']} 個有效反轉點")
                st.info(f"上軌反轉: {reversal_stats['upper_reversals']} | 下軌反轉: {reversal_stats['lower_reversals']}")
                st.info(f"拒絕無效觸碰: {reversal_stats['total_rejected']}")
                
                if reversal_stats['total_reversals'] < 50:
                    st.error(f"反轉點數量太少: {reversal_stats['total_reversals']}, 建議增加訓練數據或降低最小反轉幅度")
                    st.stop()
                
                # 獲取訓練數據
                X, y = extractor.get_training_data(df_processed)
                
                st.info(f"訓練樣本: {len(X)} (做多:{(y==1).sum()}, 做空:{(y==0).sum()})")
                
                # 訓練模型
                trainer = BBReversalModelTrainer(model_dir='models/saved')
                metrics = trainer.train(X, y)
                trainer.save_model(prefix=train_symbol)
                
                st.success(f"{train_symbol} BB反轉模型訓練完成!")
                st.info(f"模型保存至: models/saved/{train_symbol}_bb_reversal_lgb.pkl")
                
                # 顯示指標
                col1, col2 = st.columns(2)
                with col1:
                    accuracy = metrics['accuracy']
                    if accuracy >= 0.70:
                        st.success(f"準確率: {accuracy:.2%}")
                    elif accuracy >= 0.60:
                        st.info(f"準確率: {accuracy:.2%}")
                    else:
                        st.warning(f"準確率: {accuracy:.2%}")
                
                with col2:
                    st.metric("訓練樣本", len(X))
                
                # 特徵重要性
                importance = trainer.get_feature_importance(extractor.get_feature_columns(), top_n=15)
                st.subheader("Top 15 重要特徵")
                st.dataframe(importance, use_container_width=True)
                
                if accuracy < 0.60:
                    st.warning("建議: 準確率偏低，請增加訓練數據或調整參數")
                elif accuracy >= 0.70:
                    st.balloons()
                    st.success("準確率優異! 可以開始回測")
                
            except Exception as e:
                st.error(f"訓練失敗: {str(e)}")
                import traceback
                st.code(traceback.format_exc())