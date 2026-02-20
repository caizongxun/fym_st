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
    st.title("Model Training")
    
    st.markdown("""
    Train a machine learning model with:
    - Triple Barrier Method for labeling
    - **Automatic Probability Calibration** (NEW)
    - Purged K-Fold Cross-Validation
    - XGBoost with optimized hyperparameters
    """)
    
    st.markdown("---")
    
    with st.expander("Training Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            loader = CryptoDataLoader()
            symbol = st.selectbox("Training Symbol", loader.get_available_symbols(), index=10)
            timeframe = st.selectbox("Timeframe", loader.get_available_timeframes(), index=1)
            
            tp_multiplier = st.number_input("Take Profit Multiplier (ATR)", value=4.0, step=0.5, 
                                           help="優化建議: 4.0")
            sl_multiplier = st.number_input("Stop Loss Multiplier (ATR)", value=2.0, step=0.25,
                                           help="優化建議: 2.0")
        
        with col2:
            max_holding_bars = st.number_input("Max Holding Bars", value=24, step=1,
                                              help="最大持倉時間(根K線)")
            n_splits = st.number_input("CV Splits", value=5, step=1, min_value=3, max_value=10)
            embargo_pct = st.number_input("Embargo %", value=0.01, step=0.01, format="%.3f",
                                         help="Purged K-Fold 的空白期百分比")
            use_calibration = st.checkbox("Enable Probability Calibration", value=True,
                                         help="校準模型預測的機率,強烈建議啟用")
    
    with st.expander("Model Hyperparameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            max_depth = st.number_input("Max Depth", value=6, min_value=3, max_value=10)
            learning_rate = st.number_input("Learning Rate", value=0.05, step=0.01, format="%.3f")
            n_estimators = st.number_input("N Estimators", value=200, step=50)
        with col2:
            min_child_weight = st.number_input("Min Child Weight", value=3, min_value=1, max_value=10)
            subsample = st.number_input("Subsample", value=0.8, step=0.1, format="%.2f")
            colsample_bytree = st.number_input("Colsample Bytree", value=0.8, step=0.1, format="%.2f")
    
    model_name = st.text_input("Model Name", value=f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    
    if st.button("Start Training", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Loading data...")
            progress_bar.progress(5)
            df = loader.load_klines(symbol, timeframe)
            st.info(f"Loaded {len(df)} rows of data from {df['open_time'].min()} to {df['open_time'].max()}")
            
            status_text.text("Building features...")
            progress_bar.progress(15)
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.build_features(df)
            st.info(f"Built {len(df_features.columns)} features")
            
            status_text.text("Applying triple barrier labeling...")
            progress_bar.progress(25)
            labeler = TripleBarrierLabeling(
                tp_multiplier=tp_multiplier,
                sl_multiplier=sl_multiplier,
                max_holding_bars=int(max_holding_bars)
            )
            df_labeled = labeler.apply_triple_barrier(df_features)
            
            positive_count = (df_labeled['label'] == 1).sum()
            negative_count = (df_labeled['label'] == 0).sum()
            positive_pct = positive_count / len(df_labeled) * 100
            st.info(f"Label distribution: {positive_pct:.1f}% positive ({positive_count} wins, {negative_count} losses)")
            
            if positive_pct < 20 or positive_pct > 80:
                st.warning(f"Imbalanced labels: {positive_pct:.1f}% positive. Consider adjusting TP/SL multipliers.")
            
            status_text.text("Preparing training data...")
            progress_bar.progress(35)
            
            feature_cols = [col for col in df_labeled.columns if col not in [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'label', 'exit_type', 'exit_price', 'exit_bars', 'return'
            ]]
            
            X = df_labeled[feature_cols].copy()
            y = df_labeled['label'].copy()
            
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            st.info(f"Training data: {len(X)} samples with {len(feature_cols)} features")
            
            status_text.text(f"Training model with Purged K-Fold CV ({n_splits} splits)...")
            if use_calibration:
                st.info("Probability calibration: ENABLED")
            progress_bar.progress(50)
            
            trainer = ModelTrainer(use_calibration=use_calibration)
            
            params = {
                'max_depth': int(max_depth),
                'learning_rate': float(learning_rate),
                'n_estimators': int(n_estimators),
                'min_child_weight': int(min_child_weight),
                'subsample': float(subsample),
                'colsample_bytree': float(colsample_bytree)
            }
            
            cv_metrics = trainer.train_with_purged_kfold(
                X, y,
                n_splits=int(n_splits),
                embargo_pct=float(embargo_pct),
                params=params
            )
            
            progress_bar.progress(90)
            
            status_text.text("Saving model...")
            trainer.save_model(model_name)
            
            progress_bar.progress(100)
            status_text.text("Training complete")
            
            st.success(f"Model trained and saved as: {model_name}")
            
            st.markdown("### Cross-Validation Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CV Accuracy", f"{cv_metrics.get('cv_val_accuracy', 0):.4f} ± {cv_metrics.get('cv_val_accuracy_std', 0):.4f}")
            with col2:
                st.metric("CV AUC", f"{cv_metrics.get('cv_val_auc', 0):.4f} ± {cv_metrics.get('cv_val_auc_std', 0):.4f}")
            with col3:
                st.metric("CV Precision", f"{cv_metrics.get('cv_val_precision', 0):.4f}")
            with col4:
                st.metric("CV Recall", f"{cv_metrics.get('cv_val_recall', 0):.4f}")
            
            st.markdown("### Feature Importance (Top 20)")
            feature_importance = trainer.get_feature_importance()
            st.dataframe(feature_importance.head(20), use_container_width=True)
            
            st.markdown("### Next Steps")
            st.info("""
            1. Go to **Calibration Analysis** to check probability calibration
            2. Go to **Strategy Optimization** to find best parameters
            3. Go to **Backtesting** to test with TP=4.0, SL=2.0
            """)
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())