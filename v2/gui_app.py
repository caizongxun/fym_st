import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

from data_loader import CryptoDataLoader
from feature_engineering import FeatureEngineer
from label_generation import LabelGenerator
from pipeline import TradingPipeline
from model_trainer import ModelTrainer, TrendFilterTrainer
from inference_engine import InferenceEngine


st.set_page_config(
    page_title="V2 交易系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 V2 模塊化交易系統")

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = CryptoDataLoader()

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = TradingPipeline()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 數據載入",
    "🔧 特徵工程",
    "🎯 標籤生成",
    "🧠 模型訓練",
    "🚀 推論測試"
])

with tab1:
    st.header("📊 數據載入")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("資料集資訊")
        info = st.session_state.data_loader.get_dataset_info()
        st.metric("交易對數量", info['total_symbols'])
        st.metric("時間框架", len(info['timeframes']))
        st.metric("總檔案數", info['total_files'])
        
        with st.expander("查看所有交易對"):
            for symbol in info['symbols']:
                st.text(symbol)
    
    with col2:
        st.subheader("載入數據")
        
        col2_1, col2_2, col2_3 = st.columns(3)
        
        with col2_1:
            symbol = st.selectbox(
                "選擇交易對",
                info['symbols'],
                key='load_symbol'
            )
        
        with col2_2:
            timeframe = st.selectbox(
                "選擇時間框架",
                info['timeframes'],
                key='load_timeframe'
            )
        
        with col2_3:
            st.write("")
            st.write("")
            if st.button("📂 載入數據", use_container_width=True):
                with st.spinner('載入中...'):
                    try:
                        df = st.session_state.data_loader.load_klines(symbol, timeframe)
                        df_prepared = st.session_state.data_loader.prepare_dataframe(df)
                        st.session_state.df_raw = df_prepared
                        st.success(f"成功載入 {len(df_prepared)} 筆數據")
                    except Exception as e:
                        st.error(f"載入失敗: {str(e)}")
        
        if 'df_raw' in st.session_state:
            st.subheader("數據預覽")
            
            df_display = st.session_state.df_raw.copy()
            st.dataframe(df_display.head(100), use_container_width=True, height=300)
            
            st.subheader("數據統計")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("總筆數", len(df_display))
            with col_stat2:
                st.metric("起始時間", df_display['timestamp'].min().strftime('%Y-%m-%d'))
            with col_stat3:
                st.metric("結束時間", df_display['timestamp'].max().strftime('%Y-%m-%d'))
            with col_stat4:
                st.metric("平均價格", f"{df_display['close'].mean():.2f}")

with tab2:
    st.header("🔧 特徵工程")
    
    if 'df_raw' not in st.session_state:
        st.warning("請先在「數據載入」頁面載入數據")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("參數設定")
            
            bb_period = st.number_input("布林帶週期", 5, 50, 20)
            bb_std = st.number_input("標準差倍數", 1.0, 3.0, 2.0, 0.1)
            lookback = st.number_input("回溯週期", 50, 200, 100)
            pivot_left = st.number_input("樞紐左側K線", 1, 10, 3)
            pivot_right = st.number_input("樞紐右側K線", 1, 10, 3)
            
            if st.button("⚙️ 計算特徵", use_container_width=True):
                with st.spinner('計算中...'):
                    try:
                        fe = FeatureEngineer(
                            bb_period=bb_period,
                            bb_std=bb_std,
                            lookback=lookback,
                            pivot_left=pivot_left,
                            pivot_right=pivot_right
                        )
                        st.session_state.df_features = fe.process_features(st.session_state.df_raw)
                        st.session_state.feature_engineer = fe
                        st.success(f"特徵計算完成: {len(st.session_state.df_features)} 筆")
                    except Exception as e:
                        st.error(f"計算失敗: {str(e)}")
        
        with col2:
            if 'df_features' in st.session_state:
                st.subheader("特徵數據預覽")
                
                feature_cols = st.session_state.feature_engineer.get_feature_columns()
                display_cols = ['timestamp', 'close'] + feature_cols[:5]
                
                st.dataframe(
                    st.session_state.df_features[display_cols].head(50),
                    use_container_width=True,
                    height=300
                )
                
                st.subheader("特徵列表")
                col_feat1, col_feat2 = st.columns(2)
                with col_feat1:
                    st.write(feature_cols[:8])
                with col_feat2:
                    st.write(feature_cols[8:])

with tab3:
    st.header("🎯 標籤生成")
    
    if 'df_features' not in st.session_state:
        st.warning("請先在「特徵工程」頁面計算特徵")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("參數設定")
            
            atr_period = st.number_input("ATR週期", 5, 30, 14)
            sl_mult = st.number_input("停損ATR倍數", 0.5, 3.0, 1.5, 0.1)
            tp_mult = st.number_input("停利ATR倍數", 1.0, 5.0, 3.0, 0.1)
            lookahead = st.number_input("前瞥K線數", 5, 50, 16)
            
            if st.button("🎯 生成標籤", use_container_width=True):
                with st.spinner('生成中...'):
                    try:
                        lg = LabelGenerator(
                            atr_period=atr_period,
                            sl_atr_mult=sl_mult,
                            tp_atr_mult=tp_mult,
                            lookahead_bars=lookahead
                        )
                        st.session_state.df_labeled = lg.generate_labels(st.session_state.df_features)
                        st.session_state.label_generator = lg
                        
                        stats = lg.get_label_statistics(st.session_state.df_labeled)
                        st.session_state.label_stats = stats
                        
                        st.success("標籤生成完成")
                    except Exception as e:
                        st.error(f"生成失敗: {str(e)}")
        
        with col2:
            if 'df_labeled' in st.session_state:
                st.subheader("標籤統計")
                
                stats = st.session_state.label_stats
                
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.write("**做多樣本**")
                    if 'long_total' in stats:
                        st.metric("總數", stats['long_total'])
                        st.metric("成功", stats['long_success'])
                        st.metric("失敗", stats['long_fail'])
                        st.metric("成功率", f"{stats['long_success_rate']:.2f}%")
                    else:
                        st.info("無做多樣本")
                
                with col_stat2:
                    st.write("**做空樣本**")
                    if 'short_total' in stats:
                        st.metric("總數", stats['short_total'])
                        st.metric("成功", stats['short_success'])
                        st.metric("失敗", stats['short_fail'])
                        st.metric("成功率", f"{stats['short_success_rate']:.2f}%")
                    else:
                        st.info("無做空樣本")
                
                st.subheader("標籤數據預覽")
                display_cols = ['timestamp', 'close', 'lower', 'upper', 'atr', 
                               'is_touching_lower', 'is_touching_upper', 
                               'target_long', 'target_short']
                available_cols = [col for col in display_cols if col in st.session_state.df_labeled.columns]
                st.dataframe(
                    st.session_state.df_labeled[available_cols].head(50),
                    use_container_width=True,
                    height=300
                )

with tab4:
    st.header("🧠 模型訓練")
    
    if 'df_labeled' not in st.session_state:
        st.warning("請先在「標籤生成」頁面生成標籤")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("訓練參數")
            
            direction = st.selectbox("方向", ['long', 'short'])
            n_estimators = st.number_input("樹數量", 100, 1000, 500, 50)
            learning_rate = st.number_input("學習率", 0.01, 0.2, 0.05, 0.01)
            max_depth = st.number_input("最大深度", 3, 15, 7)
            train_ratio = st.number_input("訓練集比例", 0.5, 0.9, 0.8, 0.05)
            
            st.write("---")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🎯 訓練反彈模型", use_container_width=True):
                    with st.spinner('訓練中...'):
                        try:
                            df_train = st.session_state.label_generator.prepare_training_data(
                                st.session_state.df_labeled,
                                direction=direction
                            )
                            
                            trainer = ModelTrainer(
                                model_type='bounce',
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth
                            )
                            
                            results = trainer.train(df_train, train_ratio=train_ratio)
                            
                            os.makedirs('v2/models', exist_ok=True)
                            trainer.save_model(f'v2/models/bounce_{direction}_model.pkl')
                            
                            st.session_state.bounce_results = results
                            st.success("反彈模型訓練完成")
                        except Exception as e:
                            st.error(f"訓練失敗: {str(e)}")
            
            with col_btn2:
                if st.button("🚫 訓練過濾模型", use_container_width=True):
                    with st.spinner('訓練中...'):
                        try:
                            df_train = st.session_state.label_generator.prepare_training_data(
                                st.session_state.df_labeled,
                                direction=direction
                            )
                            
                            trainer = TrendFilterTrainer(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth
                            )
                            
                            results = trainer.train(df_train, train_ratio=train_ratio)
                            
                            os.makedirs('v2/models', exist_ok=True)
                            trainer.save_model(f'v2/models/filter_{direction}_model.pkl')
                            
                            st.session_state.filter_results = results
                            st.success("過濾模型訓練完成")
                        except Exception as e:
                            st.error(f"訓練失敗: {str(e)}")
        
        with col2:
            st.subheader("訓練結果")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.write("**反彈模型**")
                if 'bounce_results' in st.session_state:
                    results = st.session_state.bounce_results
                    st.metric("訓練 AUC", f"{results['train_auc']:.4f}")
                    st.metric("測試 AUC", f"{results['test_auc']:.4f}")
                    st.metric("訓練樣本", results['train_samples'])
                    st.metric("測試樣本", results['test_samples'])
                    
                    st.write("**特徵重要性 Top 5**")
                    st.dataframe(
                        results['feature_importance'].head(5),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("尚未訓練")
            
            with col_res2:
                st.write("**過濾模型**")
                if 'filter_results' in st.session_state:
                    results = st.session_state.filter_results
                    st.metric("訓練 AUC", f"{results['train_auc']:.4f}")
                    st.metric("測試 AUC", f"{results['test_auc']:.4f}")
                    st.metric("訓練樣本", results['train_samples'])
                    st.metric("測試樣本", results['test_samples'])
                    
                    st.write("**特徵重要性 Top 5**")
                    st.dataframe(
                        results['feature_importance'].head(5),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("尚未訓練")

with tab5:
    st.header("🚀 推論測試")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("模型選擇")
        
        direction_infer = st.selectbox("方向", ['long', 'short'], key='infer_direction')
        
        bounce_path = f'v2/models/bounce_{direction_infer}_model.pkl'
        filter_path = f'v2/models/filter_{direction_infer}_model.pkl'
        
        if os.path.exists(bounce_path) and os.path.exists(filter_path):
            st.success("模型檔案存在")
            
            st.subheader("閉值設定")
            bounce_threshold = st.slider("反彈閉值", 0.0, 1.0, 0.65, 0.05)
            filter_threshold = st.slider("過濾閉值", 0.0, 1.0, 0.40, 0.05)
            
            if st.button("🚀 執行推論", use_container_width=True):
                if 'df_labeled' not in st.session_state:
                    st.error("請先生成標籤數據")
                else:
                    with st.spinner('推論中...'):
                        try:
                            engine = InferenceEngine(
                                bounce_model_path=bounce_path,
                                filter_model_path=filter_path,
                                bounce_threshold=bounce_threshold,
                                filter_threshold=filter_threshold
                            )
                            
                            df_test = st.session_state.label_generator.prepare_training_data(
                                st.session_state.df_labeled,
                                direction=direction_infer
                            )
                            
                            df_predictions = engine.predict_batch(df_test)
                            stats = engine.get_statistics(df_predictions)
                            
                            st.session_state.df_predictions = df_predictions
                            st.session_state.inference_stats = stats
                            
                            st.success("推論完成")
                        except Exception as e:
                            st.error(f"推論失敗: {str(e)}")
        else:
            st.error(f"模型檔案不存在\n{bounce_path}\n{filter_path}")
            st.info("請先在「模型訓練」頁面訓練模型")
    
    with col2:
        if 'inference_stats' in st.session_state:
            st.subheader("推論統計")
            
            stats = st.session_state.inference_stats
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("總樣本", stats['total_samples'])
            with col_stat2:
                st.metric("核准進場", stats['entry_approved'])
            with col_stat3:
                st.metric("進場率", f"{stats['entry_rate']:.2f}%")
            with col_stat4:
                if 'approved_success_rate' in stats:
                    st.metric("核准後成功率", f"{stats['approved_success_rate']:.2f}%")
            
            col_stat5, col_stat6 = st.columns(2)
            with col_stat5:
                st.metric("平均 P_bounce", f"{stats['avg_p_bounce']:.4f}")
            with col_stat6:
                st.metric("平均 P_filter", f"{stats['avg_p_filter']:.4f}")
            
            st.subheader("訊號原因分佈")
            reason_df = pd.DataFrame([
                {'reason': k, 'count': v} 
                for k, v in stats['reason_counts'].items()
            ])
            st.dataframe(reason_df, use_container_width=True, hide_index=True)
            
            st.subheader("推論結果預覽")
            display_cols = ['timestamp', 'close', 'p_bounce', 'p_filter', 
                           'signal', 'reason', 'target']
            available_cols = [col for col in display_cols if col in st.session_state.df_predictions.columns]
            st.dataframe(
                st.session_state.df_predictions[available_cols].head(50),
                use_container_width=True,
                height=300
            )
        else:
            st.info("請先執行推論")

st.sidebar.header("關於")
st.sidebar.info(
    """
    **V2 模塊化交易系統**
    
    功能模塊:
    - 數據載入 (HuggingFace)
    - 特徵工程 (15個指標)
    - 標籤生成 (ATR動態)
    - 模型訓練 (LightGBM)
    - 推論測試 (共振-否決)
    
    版本: 2.0.0
    """
)

st.sidebar.header("快速操作")
if st.sidebar.button("清除所有緩存"):
    for key in list(st.session_state.keys()):
        if key not in ['data_loader', 'pipeline']:
            del st.session_state[key]
    st.sidebar.success("緩存已清除")
    st.rerun()
