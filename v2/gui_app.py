import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import glob
import time

from data_loader import CryptoDataLoader
from feature_engineering import FeatureEngineer
from label_generation import LabelGenerator
from pipeline import TradingPipeline
from model_trainer import ModelTrainer, TrendFilterTrainer
from inference_engine import InferenceEngine
from advanced_data_collector import BatchAdvancedDataCollector, BinanceAdvancedDataCollector
from advanced_feature_merger import AdvancedFeatureMerger


st.set_page_config(
    page_title="V2 交易系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("V2 模塊化交易系統")

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = CryptoDataLoader()

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = TradingPipeline()

if 'batch_collector' not in st.session_state:
    st.session_state.batch_collector = BatchAdvancedDataCollector()

if 'feature_merger' not in st.session_state:
    st.session_state.feature_merger = AdvancedFeatureMerger()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 [1] 數據載入",
    "🔄 [2] 資金費率收集",
    "🛠️ [3] 特徵工程",
    "🏷️ [4] 標籤生成",
    "🤖 [5] 模型訓練",
    "🎯 [6] 推論測試",
    "☁️ [7] HF 上傳"
])

# ============================================================
# Tab 1: 數據載入
# ============================================================
with tab1:
    st.header("數據載入")
    
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
            if st.button("載入數據", use_container_width=True):
                with st.spinner('載入中...'):
                    try:
                        df = st.session_state.data_loader.load_klines(symbol, timeframe)
                        df_prepared = st.session_state.data_loader.prepare_dataframe(df)
                        st.session_state.df_raw = df_prepared
                        st.session_state.current_symbol = symbol
                        st.session_state.current_timeframe = timeframe
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

# ============================================================
# Tab 2: 資金費率收集
# ============================================================
with tab2:
    st.header("🔄 資金費率收集")
    
    st.info(
        "💰 **資金費率 (Funding Rate)**\n"
        "- Binance 期貨每 8 小時一筆，具備 2019～今完整歷史\n"
        "- 是唯一可用于訓練的進階特徵 (其他皆僅 30 天)\n"
        "- 收集完成後在 [7] HF 上傳展頁將其備份到 HuggingFace"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("收集參數")
        
        collection_mode = st.radio(
            "收集模式",
            ["單一幣種", f"批量收集 ({len(st.session_state.batch_collector.hf_symbols)}個)"],
            key='collection_mode'
        )
        
        if collection_mode == "單一幣種":
            symbols_to_collect = [st.selectbox(
                "選擇幣種",
                st.session_state.batch_collector.hf_symbols,
                key='adv_symbol'
            )]
        else:
            symbols_to_collect = st.session_state.batch_collector.hf_symbols
            with st.expander(f"查看幣種清單 ({len(symbols_to_collect)}個)"):
                for sym in symbols_to_collect:
                    st.text(sym)
        
        output_dir = st.text_input(
            "輸出目錄",
            value='v2/advanced_data',
            key='adv_output_dir'
        )
        
        st.write("---")
        
        if st.button("🚀 開始收集資金費率", use_container_width=True, type="primary"):
            st.session_state.collection_started = True
            st.rerun()
    
    with col2:
        if st.session_state.get('collection_started', False):
            st.subheader("收集進度")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            summary_data = []
            
            collector = BinanceAdvancedDataCollector()
            os.makedirs(output_dir, exist_ok=True)
            
            for idx, symbol in enumerate(symbols_to_collect):
                progress = (idx + 1) / len(symbols_to_collect)
                progress_bar.progress(progress)
                status_text.text(f"正在處理 {symbol} ({idx+1}/{len(symbols_to_collect)})...")
                
                try:
                    start_time = collector.get_earliest_available_time(symbol)
                    df = collector.get_funding_rate(symbol, start_time)
                    
                    if not df.empty:
                        filepath = os.path.join(output_dir, f"{symbol}_funding_rate.parquet")
                        df.to_parquet(filepath, index=False)
                        summary_data.append({
                            '幣種': symbol,
                            '筆數': f"{len(df):,}",
                            '起始': df['timestamp'].min().strftime('%Y-%m-%d'),
                            '結束': df['timestamp'].max().strftime('%Y-%m-%d'),
                            '狀態': '✅ 成功'
                        })
                    else:
                        summary_data.append({
                            '幣種': symbol, '筆數': '0',
                            '起始': '-', '結束': '-', '狀態': '⚠️ 無數據'
                        })
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    summary_data.append({
                        '幣種': symbol, '筆數': '0',
                        '起始': '-', '結束': '-',
                        '狀態': f'❌ {str(e)[:20]}'
                    })
            
            progress_bar.progress(1.0)
            status_text.text("✅ 收集完成!")
            st.session_state.collection_summary = pd.DataFrame(summary_data)
            st.session_state.collection_started = False
            st.rerun()
        
        if 'collection_summary' in st.session_state:
            st.subheader("📋 收集摘要")
            st.dataframe(
                st.session_state.collection_summary,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            df_sum = st.session_state.collection_summary
            success = (df_sum['狀態'] == '✅ 成功').sum()
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("成功收集", f"{success}/{len(df_sum)}")
            with col_m2:
                st.metric("可上傳檔案", f"{success} 個 .parquet")
            
            st.info("💡 收集完成後，前往 [7] HF 上傳展頁進行備份")

# ============================================================
# Tab 3: 特徵工程
# ============================================================
with tab3:
    st.header("特徵工程")
    
    if 'df_raw' not in st.session_state:
        st.warning("⚠️ 請先在 [1]數據載入 頁面載入數據")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("參數設定")
            
            bb_period  = st.number_input("布林帶週期", 5, 50, 20)
            bb_std     = st.number_input("標準差倍數", 1.0, 3.0, 2.0, 0.1)
            lookback   = st.number_input("回溯週期", 50, 200, 100)
            pivot_left = st.number_input("樞紐左側K線", 1, 10, 3)
            pivot_right= st.number_input("樞紐右側K線", 1, 10, 3)
            
            st.write("---")
            
            merge_advanced = st.checkbox(
                "🔥 合併資金費率特徵",
                value=False,
                help="自動載入 funding_rate.parquet"
            )
            
            if st.button("計算特徵", use_container_width=True):
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
                        
                        base_features = len(fe.get_feature_columns())
                        
                        if merge_advanced:
                            if 'current_symbol' in st.session_state:
                                merger = st.session_state.feature_merger
                                st.session_state.df_features = merger.merge_for_training(
                                    st.session_state.df_features,
                                    st.session_state.current_symbol
                                )
                                adv_features = merger.get_training_feature_columns(st.session_state.df_features)
                                st.success(f"✅ 特徵計算完成: {len(st.session_state.df_features)} 筆 | 基礎: {base_features} | 資金費率: {len(adv_features)}")
                            else:
                                st.warning("⚠️ 請先載入數據")
                                st.success(f"✅ 基礎特徵計算完成: {len(st.session_state.df_features)} 筆 | 特徵數: {base_features}")
                        else:
                            st.success(f"✅ 基礎特徵計算完成: {len(st.session_state.df_features)} 筆 | 特徵數: {base_features}")
                    except Exception as e:
                        st.error(f"❌ 計算失敗: {str(e)}")
        
        with col2:
            if 'df_features' in st.session_state:
                st.subheader("特徵數據預覽")
                
                feature_cols = st.session_state.feature_engineer.get_feature_columns()
                adv_features = []
                if 'feature_merger' in st.session_state:
                    adv_features = st.session_state.feature_merger.get_training_feature_columns(
                        st.session_state.df_features
                    )
                
                all_features = feature_cols + adv_features
                display_cols = ['timestamp', 'close'] + all_features[:8]
                available_cols = [c for c in display_cols if c in st.session_state.df_features.columns]
                
                st.dataframe(
                    st.session_state.df_features[available_cols].head(50),
                    use_container_width=True,
                    height=300
                )
                
                st.subheader("📊 特徵列表")
                col_feat1, col_feat2 = st.columns(2)
                with col_feat1:
                    st.write("**基礎特徵**")
                    for feat in feature_cols:
                        st.text(f"• {feat}")
                with col_feat2:
                    if adv_features:
                        st.write(f"**資金費率特徵 ({len(adv_features)}個)**")
                        for feat in adv_features:
                            st.text(f"• {feat}")
                    else:
                        st.info("未載入資金費率特徵")

# ============================================================
# Tab 4: 標籤生成
# ============================================================
with tab4:
    st.header("標籤生成")
    
    if 'df_features' not in st.session_state:
        st.warning("⚠️ 請先在 [3]特徵工程 頁面計算特徵")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("參數設定")
            
            atr_period = st.number_input("ATR週期", 5, 30, 14)
            sl_mult    = st.number_input("停損ATR倍數", 0.5, 3.0, 1.5, 0.1)
            tp_mult    = st.number_input("停利ATR倍數", 1.0, 5.0, 3.0, 0.1)
            lookahead  = st.number_input("前瞥K線數", 5, 50, 16)
            
            if st.button("生成標籤", use_container_width=True):
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
                        st.success("✅ 標籤生成完成")
                    except Exception as e:
                        st.error(f"❌ 生成失敗: {str(e)}")
        
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

# ============================================================
# Tab 5: 模型訓練
# ============================================================
with tab5:
    st.header("模型訓練")
    
    if 'df_labeled' not in st.session_state:
        st.warning("⚠️ 請先在 [4]標籤生成 頁面生成標籤")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("訓練參數")
            
            direction    = st.selectbox("方向", ['long', 'short'])
            n_estimators = st.number_input("樹數量", 100, 1000, 300, 50)
            learning_rate= st.number_input("學習率", 0.001, 0.1, 0.01, 0.001, format="%.3f")
            max_depth    = st.number_input("最大深度", 3, 15, 4)
            train_ratio  = st.number_input("訓練集比例", 0.5, 0.9, 0.8, 0.05)
            
            st.write("---")
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("訓練反彈模型", use_container_width=True):
                    with st.spinner('訓練中...'):
                        try:
                            df_train = st.session_state.label_generator.prepare_training_data(
                                st.session_state.df_labeled, direction=direction)
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
                            st.success("✅ 反彈模型訓練完成")
                        except Exception as e:
                            st.error(f"❌ 訓練失敗: {str(e)}")
            
            with col_btn2:
                if st.button("訓練過濾模型", use_container_width=True):
                    with st.spinner('訓練中...'):
                        try:
                            df_train = st.session_state.label_generator.prepare_training_data(
                                st.session_state.df_labeled, direction=direction)
                            trainer = TrendFilterTrainer(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth
                            )
                            results = trainer.train(df_train, train_ratio=train_ratio)
                            os.makedirs('v2/models', exist_ok=True)
                            trainer.save_model(f'v2/models/filter_{direction}_model.pkl')
                            st.session_state.filter_results = results
                            st.success("✅ 過濾模型訓練完成")
                        except Exception as e:
                            st.error(f"❌ 訓練失敗: {str(e)}")
        
        with col2:
            st.subheader("訓練結果")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.write("**反彈模型**")
                if 'bounce_results' in st.session_state:
                    r = st.session_state.bounce_results
                    st.metric("訓練 AUC", f"{r['train_auc']:.4f}")
                    st.metric("測試 AUC", f"{r['test_auc']:.4f}")
                    st.metric("訓練樣本", r['train_samples'])
                    st.metric("測試樣本", r['test_samples'])
                    st.write("**特徵重要性 Top 10**")
                    st.dataframe(r['feature_importance'].head(10),
                                 use_container_width=True, hide_index=True, height=300)
                else:
                    st.info("尚未訓練")
            
            with col_res2:
                st.write("**過濾模型**")
                if 'filter_results' in st.session_state:
                    r = st.session_state.filter_results
                    st.metric("訓練 AUC", f"{r['train_auc']:.4f}")
                    st.metric("測試 AUC", f"{r['test_auc']:.4f}")
                    st.metric("訓練樣本", r['train_samples'])
                    st.metric("測試樣本", r['test_samples'])
                    st.write("**特徵重要性 Top 10**")
                    st.dataframe(r['feature_importance'].head(10),
                                 use_container_width=True, hide_index=True, height=300)
                else:
                    st.info("尚未訓練")

# ============================================================
# Tab 6: 推論測試
# ============================================================
with tab6:
    st.header("推論測試")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("模型選擇")
        direction_infer = st.selectbox("方向", ['long', 'short'], key='infer_direction')
        
        bounce_path = f'v2/models/bounce_{direction_infer}_model.pkl'
        filter_path = f'v2/models/filter_{direction_infer}_model.pkl'
        
        if os.path.exists(bounce_path) and os.path.exists(filter_path):
            st.success("✅ 模型檔案存在")
            
            st.subheader("閾値設定")
            bounce_threshold = st.slider("反彈閾値", 0.0, 1.0, 0.65, 0.05)
            filter_threshold = st.slider("過濾閾値", 0.0, 1.0, 0.40, 0.05)
            
            if st.button("執行推論", use_container_width=True):
                if 'df_labeled' not in st.session_state:
                    st.error("❌ 請先生成標籤數據")
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
                                st.session_state.df_labeled, direction=direction_infer)
                            df_predictions = engine.predict_batch(df_test)
                            stats = engine.get_statistics(df_predictions)
                            st.session_state.df_predictions = df_predictions
                            st.session_state.inference_stats = stats
                            st.success("✅ 推論完成")
                        except Exception as e:
                            st.error(f"❌ 推論失敗: {str(e)}")
        else:
            st.error("❌ 模型檔案不存在")
            st.info("請先在 [5]模型訓練 頁面訓練模型")
    
    with col2:
        if 'inference_stats' in st.session_state:
            st.subheader("推論統計")
            stats = st.session_state.inference_stats
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1: st.metric("總樣本", stats['total_samples'])
            with col_s2: st.metric("核准進場", stats['entry_approved'])
            with col_s3: st.metric("進場率", f"{stats['entry_rate']:.2f}%")
            with col_s4:
                if 'approved_success_rate' in stats:
                    st.metric("核准後成功率", f"{stats['approved_success_rate']:.2f}%")
            
            st.subheader("推論結果預覽")
            display_cols = ['timestamp', 'close', 'p_bounce', 'p_filter', 'signal', 'reason', 'target']
            available_cols = [col for col in display_cols if col in st.session_state.df_predictions.columns]
            st.dataframe(
                st.session_state.df_predictions[available_cols].head(50),
                use_container_width=True, height=300
            )
        else:
            st.info("請先執行推論")

# ============================================================
# Tab 7: HF 上傳
# ============================================================
with tab7:
    st.header("☁️ HuggingFace 上傳")
    
    st.info(
        "📂 **上傳結構**\n"
        "```\n"
        "klines/\n"
        "├── BTCUSDT/\n"
        "│   ├── [K線檔案]  ← 已在 HF\n"
        "│   └── BTCUSDT_funding_rate.parquet  ← 本次上傳\n"
        "├── ETHUSDT/\n"
        "│   └── ETHUSDT_funding_rate.parquet\n"
        "...\n"
        "```"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("上傳設定")
        
        hf_token = st.text_input(
            "HuggingFace Token",
            type="password",
            help="從 https://huggingface.co/settings/tokens 獲取",
            key='hf_token'
        )
        
        hf_repo = st.text_input(
            "Repository ID",
            value="zongowo111/v2-crypto-ohlcv-data",
            key='hf_repo'
        )
        
        data_dir = st.text_input(
            "資金費率目錄",
            value="v2/advanced_data",
            key='hf_data_dir'
        )
        
        commit_msg = st.text_input(
            "Commit 訊息",
            value=f"Add funding rate data - {datetime.now().strftime('%Y-%m-%d')}",
            key='hf_commit_msg'
        )
        
        st.write("---")
        st.caption("⚠️ 使用單次 commit 整包上傳，遠少 API 速率限制")
        
        if st.button("🚀 一鍵上傳到 HuggingFace", use_container_width=True, type="primary"):
            if not hf_token:
                st.error("❌ 請輸入 HuggingFace Token")
            elif not os.path.exists(data_dir):
                st.error(f"❌ 目錄不存在: {data_dir}")
            else:
                with st.spinner('準備上傳...'):
                    try:
                        from huggingface_hub import HfApi, CommitOperationAdd
                        
                        api = HfApi(token=hf_token)
                        
                        # 找出所有 funding_rate.parquet
                        parquet_files = glob.glob(os.path.join(data_dir, "*_funding_rate.parquet"))
                        
                        if not parquet_files:
                            st.error(f"❌ 在 {data_dir} 找不到 *_funding_rate.parquet 檔案")
                            st.info("請先在 [2] 資金費率收集 展頁執行收集")
                        else:
                            # 準備批次上傳操作 (CommitOperationAdd)
                            operations = []
                            file_map = []
                            
                            for fp in sorted(parquet_files):
                                filename = os.path.basename(fp)
                                # BTCUSDT_funding_rate.parquet -> BTCUSDT
                                symbol = filename.replace('_funding_rate.parquet', '')
                                path_in_repo = f"klines/{symbol}/{filename}"
                                operations.append(
                                    CommitOperationAdd(
                                        path_in_repo=path_in_repo,
                                        path_or_fileobj=fp
                                    )
                                )
                                file_map.append({'symbol': symbol, 'path': path_in_repo})
                            
                            st.write(f"📎 準備上傳 **{len(operations)}** 個檔案...")
                            
                            # 單次 commit 整包上傳
                            result = api.create_commit(
                                repo_id=hf_repo,
                                repo_type="dataset",
                                operations=operations,
                                commit_message=commit_msg
                            )
                            
                            st.session_state.upload_result = {
                                'files': file_map,
                                'commit_url': result.commit_url
                            }
                            
                            st.success(f"✅ 成功上傳 {len(operations)} 個檔案")
                            st.markdown(f"**Commit URL:** [{result.commit_url}]({result.commit_url})")
                    
                    except ImportError:
                        st.error("❌ 請先安裝: pip install huggingface_hub")
                    except Exception as e:
                        st.error(f"❌ 上傳失敗: {str(e)}")
    
    with col2:
        st.subheader("📁 检查本地檔案")
        
        data_dir_check = st.text_input(
            "目錄路徑",
            value="v2/advanced_data",
            key='data_dir_check'
        )
        
        if os.path.exists(data_dir_check):
            parquet_files = glob.glob(os.path.join(data_dir_check, "*_funding_rate.parquet"))
            
            st.metric("funding_rate.parquet 檔案數", len(parquet_files))
            
            if parquet_files:
                file_info = []
                total_size = 0
                for f in sorted(parquet_files):
                    size_mb = os.path.getsize(f) / (1024 * 1024)
                    total_size += size_mb
                    symbol = os.path.basename(f).replace('_funding_rate.parquet', '')
                    # 嘗試讀取筆數
                    try:
                        df_preview = pd.read_parquet(f, columns=['timestamp'])
                        records = len(df_preview)
                        t_min = pd.read_parquet(f, columns=['timestamp'])['timestamp'].min().strftime('%Y-%m-%d')
                        t_max = pd.read_parquet(f, columns=['timestamp'])['timestamp'].max().strftime('%Y-%m-%d')
                    except Exception:
                        records = '?'
                        t_min = t_max = '-'
                    
                    file_info.append({
                        '幣種': symbol,
                        '筆數': f"{records:,}" if isinstance(records, int) else records,
                        '起始': t_min,
                        '結束': t_max,
                        '大小 (MB)': f"{size_mb:.2f}"
                    })
                
                st.dataframe(
                    pd.DataFrame(file_info),
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                st.metric("總大小 (MB)", f"{total_size:.2f}")
        else:
            st.warning(f"⚠️ 目錄不存在: {data_dir_check}")
        
        if 'upload_result' in st.session_state:
            st.subheader("📋 上傳記錄")
            r = st.session_state.upload_result
            st.write(f"已上傳 {len(r['files'])} 個檔案")
            st.markdown(f"[Commit 連結]({r['commit_url']})")
            df_uploaded = pd.DataFrame(r['files'])
            st.dataframe(df_uploaded, use_container_width=True, hide_index=True)

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("關於")
st.sidebar.info(
    """
    **V2 模塊化交易系統**
    
    功能模塊:
    - 📊 數據載入 (HuggingFace)
    - 🔄 資金費率收集 (Binance API)
    - 🛠️ 特徵工程 (BB + SMC + POC + FVG)
    - 🏷️ 標籤生成 (ATR 動態)
    - 🤖 模型訓練 (LightGBM 防過擬合)
    - 🎯 推論測試 (共振-否決)
    - ☁️ HF 上傳 (整包 commit)
    
    版本: 2.2.0
    """
)

st.sidebar.header("快速操作")
if st.sidebar.button("🗑️ 清除所有緩存"):
    for key in list(st.session_state.keys()):
        if key not in ['data_loader', 'pipeline', 'batch_collector', 'feature_merger']:
            del st.session_state[key]
    st.sidebar.success("✅ 緩存已清除")
    st.rerun()
