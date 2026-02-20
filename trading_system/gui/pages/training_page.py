import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import (
    CryptoDataLoader, FeatureEngineer, TripleBarrierLabeling,
    MetaLabeling, ModelTrainer
)

def render():
    st.title("模型訓練")
    
    st.markdown("""
    使用進階機器學習訓練交易模型:
    - **時間平移三重屏障** - 修正時間洩漏
    - **MAE 追蹤** - 記錄最大不利偏移
    - **樣本質量加權** - 快速低回撤的交易權重更高
    - **機率校準** - Isotonic Regression
    """)
    
    st.markdown("---")
    
    with st.expander("訓練配置", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            loader = CryptoDataLoader()
            symbol = st.selectbox("訓練交易對", loader.get_available_symbols(), index=10)
            timeframe = st.selectbox("時間框架", loader.get_available_timeframes(), index=1)
            
            tp_multiplier = st.number_input("止盈倍數 (ATR)", value=3.0, step=0.5, 
                                           help="建議: 3.0-4.0")
            sl_multiplier = st.number_input("止損倍數 (ATR)", value=1.5, step=0.25,
                                           help="建議: 1.5-2.0")
        
        with col2:
            max_holding_bars = st.number_input("最大持倉根數", value=24, step=1)
            n_splits = st.number_input("交叉驗證折數", value=5, step=1, min_value=3, max_value=10)
            embargo_pct = st.number_input("禁止區百分比", value=0.01, step=0.01, format="%.3f")
            use_calibration = st.checkbox("啟用機率校準", value=True)
    
    with st.expander("進階配置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            slippage = st.number_input("滑點", value=0.001, step=0.0001, format="%.4f",
                                      help="模擬實際交易的價格滑點")
            time_decay_lambda = st.number_input("時間衰減系數", value=2.0, step=0.5,
                                               help="控制持倉時間懲罰強度")
        with col2:
            quality_alpha = st.number_input("質量權重系數", value=2.0, step=0.5,
                                           help="高質量樣本的權重放大倍數")
    
    with st.expander("模型超參數", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            max_depth = st.number_input("最大深度", value=6, min_value=3, max_value=10)
            learning_rate = st.number_input("學習率", value=0.05, step=0.01, format="%.3f")
            n_estimators = st.number_input("樹的數量", value=200, step=50)
        with col2:
            min_child_weight = st.number_input("最小子節點權重", value=3, min_value=1, max_value=10)
            subsample = st.number_input("子樣本比例", value=0.8, step=0.1, format="%.2f")
            colsample_bytree = st.number_input("特徵採樣比例", value=0.8, step=0.1, format="%.2f")
    
    model_name = st.text_input("模型名稱", value=f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    
    if st.button("開始訓練", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("載入數據...")
            progress_bar.progress(5)
            df = loader.load_klines(symbol, timeframe)
            st.info(f"載入 {len(df)} 筆數據,時間範圍: {df['open_time'].min()} 至 {df['open_time'].max()}")
            
            status_text.text("建立特徵...")
            progress_bar.progress(15)
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.build_features(df)
            st.info(f"已建立 {len(df_features.columns)} 個特徵")
            
            status_text.text("應用進階三重屏障標記...")
            progress_bar.progress(25)
            labeler = TripleBarrierLabeling(
                tp_multiplier=tp_multiplier,
                sl_multiplier=sl_multiplier,
                max_holding_bars=int(max_holding_bars),
                slippage=slippage,
                time_decay_lambda=time_decay_lambda,
                quality_weight_alpha=quality_alpha
            )
            df_labeled = labeler.apply_triple_barrier(df_features)
            
            positive_count = (df_labeled['label'] == 1).sum()
            negative_count = (df_labeled['label'] == 0).sum()
            positive_pct = positive_count / len(df_labeled) * 100
            
            avg_weight_pos = df_labeled[df_labeled['label'] == 1]['sample_weight'].mean()
            avg_weight_neg = df_labeled[df_labeled['label'] == 0]['sample_weight'].mean()
            
            st.info(f"標籤分布: {positive_pct:.1f}% 正樣本 ({positive_count} 勝, {negative_count} 負)")
            st.info(f"樣本權重 - 正類: {avg_weight_pos:.2f}, 負類: {avg_weight_neg:.2f}")
            
            status_text.text("準備訓練數據...")
            progress_bar.progress(35)
            
            exclude_cols = [
                'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume',
                'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume',
                'taker_buy_base_asset_volume',
                'label', 'label_return', 'hit_time', 'exit_type', 'sample_weight', 'mae_ratio',
                'exit_price', 'exit_bars', 'return', 'ignore'
            ]
            
            feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]
            feature_cols = [col for col in feature_cols 
                          if df_labeled[col].dtype in ['int64', 'float64', 'bool', 'int32', 'float32']]
            
            X = df_labeled[feature_cols].copy()
            y = df_labeled['label'].copy()
            sample_weights = df_labeled['sample_weight'].values
            
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            for col in X.select_dtypes(include=['bool']).columns:
                X[col] = X[col].astype(int)
            
            st.info(f"訓練數據: {len(X)} 樣本, {len(feature_cols)} 特徵")
            
            status_text.text(f"Purged K-Fold 交叉驗證訓練...")
            if use_calibration:
                st.info("機率校準: 已啟用")
            progress_bar.progress(50)
            
            trainer = ModelTrainer(use_calibration=use_calibration)
            
            scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
            
            params = {
                'max_depth': int(max_depth),
                'learning_rate': float(learning_rate),
                'n_estimators': int(n_estimators),
                'min_child_weight': int(min_child_weight),
                'subsample': float(subsample),
                'colsample_bytree': float(colsample_bytree),
                'scale_pos_weight': scale_pos_weight
            }
            
            cv_metrics = trainer.train_with_purged_kfold(
                X, y,
                sample_weights=sample_weights,
                n_splits=int(n_splits),
                embargo_pct=float(embargo_pct),
                params=params
            )
            
            progress_bar.progress(90)
            status_text.text("保存模型...")
            trainer.save_model(model_name)
            
            progress_bar.progress(100)
            status_text.text("訓練完成")
            
            st.success(f"模型已訓練並保存: {model_name}")
            
            st.markdown("### 交叉驗證結果")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("準確率", f"{cv_metrics.get('cv_val_accuracy', 0):.4f} ± {cv_metrics.get('cv_val_accuracy_std', 0):.4f}")
            with col2:
                st.metric("AUC", f"{cv_metrics.get('cv_val_auc', 0):.4f} ± {cv_metrics.get('cv_val_auc_std', 0):.4f}")
            with col3:
                st.metric("精確率", f"{cv_metrics.get('cv_val_precision', 0):.4f}")
            with col4:
                st.metric("召回率", f"{cv_metrics.get('cv_val_recall', 0):.4f}")
            
            auc = cv_metrics.get('cv_val_auc', 0)
            acc = cv_metrics.get('cv_val_accuracy', 0)
            recall = cv_metrics.get('cv_val_recall', 0)
            
            if auc >= 0.65 and recall >= 0.25:
                st.success("模型表現優秀! AUC >= 0.65 且召回率良好")
            elif auc >= 0.60:
                st.info("模型表現合格,可進行回測")
            elif auc < 0.55:
                st.error("模型 AUC < 0.55,預測能力不佳")
            
            if acc > 0.85:
                st.warning("準確率 > 85%,可能存在數據洩漏")
            
            if recall < 0.1:
                st.warning("召回率 < 10%,模型過於保守")
            
            st.markdown("### 特徵重要性 (前 20 名)")
            feature_importance = trainer.get_feature_importance()
            st.dataframe(feature_importance.head(20), use_container_width=True)
            
            st.markdown("### 下一步")
            if auc >= 0.60:
                st.info("""
                1. 前往 **機率校準分析** 檢查校準效果
                2. 前往 **回測分析** 測試策略績效
                3. 前往 **策略優化** 尋找最佳參數
                """)
            else:
                st.warning("""
                模型表現不佳,建議:
                1. 調整 TP/SL 倍數
                2. 修改時間框架或交易對
                3. 檢查特徵工程
                """)
            
        except Exception as e:
            st.error(f"訓練失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())