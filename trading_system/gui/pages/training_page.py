import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import (
    CryptoDataLoader, FeatureEngineer, TripleBarrierLabeling,
    MetaLabeling, ModelTrainer, EventFilter
)

def render():
    st.title("模型訓練")
    
    st.markdown("""
    使用進階機器學習訓練交易模型:
    - **嚴格事件過濾** - 保留10-15%有效樣本 (關鍵)
    - **禁用權重失衡** - 強制 1:1 平衡,避免濾發信號
    - **時間平移標記** - 修正時間洩漏
    - **樣本質量加權** - 快速低回撤權重高
    """)
    
    st.markdown("---")
    
    with st.expander("訓練配置", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            loader = CryptoDataLoader()
            symbol = st.selectbox("訓練交易對", loader.get_available_symbols(), index=10)
            timeframe = st.selectbox("時間框架", loader.get_available_timeframes(), index=1)
            
            tp_multiplier = st.number_input("止盈倍數 (ATR)", value=2.5, step=0.5, 
                                           help="建議: 2.5-3.0,必須 > 止損")
            sl_multiplier = st.number_input("止損倍數 (ATR)", value=1.5, step=0.25,
                                           help="建議: 1.5-2.0")
        
        with col2:
            max_holding_bars = st.number_input("最大持倉根數", value=24, step=1)
            n_splits = st.number_input("交叉驗證折數", value=5, step=1, min_value=3, max_value=10)
            embargo_pct = st.number_input("禁止區百分比", value=0.01, step=0.01, format="%.3f")
            use_calibration = st.checkbox("啟用機率校準", value=True)
    
    with st.expander("嚴格事件過濾", expanded=True):
        st.markdown("""
        **三重確認 (AND 邏輯)**:
        1. 成交量爆增 > 1.5倍
        2. 價格突破 20期高/低點
        3. 波動率從壓縮區爆發
        
        **目標**: 保留10-15%樣本
        """)
        
        use_event_filter = st.checkbox("啟用嚴格過濾", value=True,
                                       help="強烈建議啟用")
        
        col1, col2 = st.columns(2)
        with col1:
            min_volume_ratio = st.number_input("最小成交量比率", value=1.5, step=0.1,
                                              help="> 1.5倏 = 150%")
            min_vsr = st.number_input("最小波動率比率", value=1.0, step=0.1)
        with col2:
            use_strict = st.checkbox("嚴格模式 (AND)", value=True,
                                    help="必須同時滿足所有條件")
            bb_squeeze = st.number_input("BB壓縮門檻", value=0.5, step=0.1,
                                        help="0.5 = 低於中位數")
    
    with st.expander("進階配置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            slippage = st.number_input("滑點", value=0.001, step=0.0001, format="%.4f")
            time_decay_lambda = st.number_input("時間衰減系數", value=2.0, step=0.5)
            quality_alpha = st.number_input("質量權重系數", value=2.0, step=0.5)
        with col2:
            use_class_weight = st.checkbox("使用類別權重平衡", value=False,
                                          help="交易中不建議啟用,會導致濾發信號")
    
    with st.expander("模型超參數 (防過擬合)", expanded=False):
        st.info("數據減少到8000筆後,需要限制模型複雜度")
        col1, col2 = st.columns(2)
        with col1:
            max_depth = st.number_input("最大深度", value=4, min_value=3, max_value=6,
                                       help="數據少時使用4-5")
            learning_rate = st.number_input("學習率", value=0.02, step=0.01, format="%.3f",
                                           help="降低到0.01-0.02")
            n_estimators = st.number_input("樹的數量", value=300, step=50,
                                          help="300-500")
        with col2:
            min_child_weight = st.number_input("最小子節點權重", value=5, min_value=3, max_value=10,
                                              help="增加到5-7")
            subsample = st.number_input("子樣本比例", value=0.7, step=0.1, format="%.2f",
                                       help="降低到0.7-0.8")
            colsample_bytree = st.number_input("特徵採樣比例", value=0.6, step=0.1, format="%.2f",
                                              help="降低到0.6-0.7")
    
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
            progress_bar.progress(10)
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.build_features(df)
            st.info(f"已建立 {len(df_features.columns)} 個特徵")
            
            # 嚴格事件過濾
            if use_event_filter:
                status_text.text("應用嚴格事件過濾...")
                progress_bar.progress(15)
                
                event_filter = EventFilter(
                    use_strict_mode=use_strict,
                    min_volume_ratio=min_volume_ratio,
                    min_vsr=min_vsr,
                    bb_squeeze_threshold=bb_squeeze,
                    lookback_period=20
                )
                
                df_filtered = event_filter.filter_events(df_features)
                
                filter_ratio = len(df_filtered) / len(df)
                st.info(f"事件過濾: {len(df)} → {len(df_filtered)} ({100*filter_ratio:.1f}%)")
                
                if filter_ratio > 0.25:
                    st.warning(f"過濾後仍保留 {filter_ratio*100:.0f}%,建議提高 min_volume_ratio")
                elif filter_ratio < 0.05:
                    st.warning(f"過濾過嚴只保留 {filter_ratio*100:.1f}%,可能樣本不足")
                
                df_features = df_filtered
            else:
                st.error("未啟用事件過濾器! AUC 可能無法超過 0.55")
            
            status_text.text("應用三重屏障標記...")
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
                'exit_price', 'exit_bars', 'return', 'ignore',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_std',
                'volume_ma_20'
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
            
            status_text.text(f"Purged K-Fold 訓練...")
            progress_bar.progress(50)
            
            trainer = ModelTrainer(use_calibration=use_calibration)
            
            # 關鍵修正: 強制 scale_pos_weight = 1.0 (禁用權重失衡)
            if use_class_weight:
                scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
                st.warning(f"已啟用類別權重: {scale_pos_weight:.2f} (可能導致濾發信號)")
            else:
                scale_pos_weight = 1.0
                st.info("類別權重: 1.0 (禁用失衡,避免濾發信號)")
            
            params = {
                'max_depth': int(max_depth),
                'learning_rate': float(learning_rate),
                'n_estimators': int(n_estimators),
                'min_child_weight': int(min_child_weight),
                'subsample': float(subsample),
                'colsample_bytree': float(colsample_bytree),
                'scale_pos_weight': scale_pos_weight,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
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
                auc_val = cv_metrics.get('cv_val_auc', 0)
                auc_delta = auc_val - 0.5
                st.metric("AUC", f"{auc_val:.4f} ± {cv_metrics.get('cv_val_auc_std', 0):.4f}",
                         delta=f"+{auc_delta:.4f}" if auc_delta > 0 else f"{auc_delta:.4f}")
            with col3:
                st.metric("精確率", f"{cv_metrics.get('cv_val_precision', 0):.4f}")
            with col4:
                st.metric("召回率", f"{cv_metrics.get('cv_val_recall', 0):.4f}")
            
            auc = cv_metrics.get('cv_val_auc', 0)
            prec = cv_metrics.get('cv_val_precision', 0)
            recall = cv_metrics.get('cv_val_recall', 0)
            
            if auc >= 0.60 and prec >= 0.45:
                st.success(f"模型優秀! AUC={auc:.3f}, 精確率={prec:.1%}")
            elif auc >= 0.56 and prec >= 0.40:
                st.info(f"模型合格! AUC={auc:.3f}, 精確率={prec:.1%}")
            elif auc < 0.55:
                st.error(f"AUC {auc:.3f} < 0.55,預測能力不佳")
            
            if recall > 0.70 and prec < 0.40:
                st.warning(f"召回率過高({recall:.1%})但精確率低({prec:.1%}),模型濾發信號")
                st.info("建議: 關閉類別權重平衡")
            
            st.markdown("### 特徵重要性 (前 20 名)")
            feature_importance = trainer.get_feature_importance()
            st.dataframe(feature_importance.head(20), use_container_width=True)
            
        except Exception as e:
            st.error(f"訓練失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())