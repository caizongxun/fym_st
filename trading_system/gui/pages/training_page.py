import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import (
    CryptoDataLoader, FeatureEngineer, TripleBarrierLabeling,
    MetaLabeling, ModelTrainer, KellyCriterion
)

def render():
    st.title("Model Training")
    
    st.markdown("""
    Train a machine learning model using:
    - Triple Barrier Method for labeling
    - Meta-Labeling for signal filtering
    - Purged K-Fold Cross-Validation
    - Sample weighting by absolute returns
    """)
    
    st.markdown("---")
    
    with st.expander("Training Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            loader = CryptoDataLoader()
            symbol = st.selectbox("Training Symbol", loader.get_available_symbols(), index=10)
            timeframe = st.selectbox("Timeframe", loader.get_available_timeframes(), index=1)
            
            tp_multiplier = st.number_input("Take Profit Multiplier (ATR)", value=2.5, step=0.1)
            sl_multiplier = st.number_input("Stop Loss Multiplier (ATR)", value=1.5, step=0.1)
        
        with col2:
            max_holding_bars = st.number_input("Max Holding Bars", value=24, step=1)
            n_splits = st.number_input("CV Splits", value=5, step=1)
            purge_gap = st.number_input("Purge Gap", value=24, step=1)
            use_sample_weights = st.checkbox("Use Sample Weights", value=True)
    
    model_name = st.text_input("Model Name", value=f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    
    if st.button("Start Training", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Loading data...")
            progress_bar.progress(10)
            df = loader.load_klines(symbol, timeframe)
            st.info(f"Loaded {len(df)} rows of data")
            
            status_text.text("Building features...")
            progress_bar.progress(25)
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.build_features(df)
            st.info(f"Built {len(df_features.columns)} features")
            
            status_text.text("Applying triple barrier labeling...")
            progress_bar.progress(40)
            labeler = TripleBarrierLabeling(
                tp_multiplier=tp_multiplier,
                sl_multiplier=sl_multiplier,
                max_holding_bars=int(max_holding_bars)
            )
            df_labeled = labeler.apply_triple_barrier(df_features)
            
            positive_pct = (df_labeled['label'] == 1).sum() / len(df_labeled) * 100
            st.info(f"Label distribution: {positive_pct:.1f}% positive")
            
            status_text.text("Preparing training data...")
            progress_bar.progress(55)
            trainer = ModelTrainer()
            X, y, feature_cols = trainer.prepare_training_data(df_labeled)
            
            sample_weights = None
            if use_sample_weights:
                status_text.text("Calculating sample weights...")
                sample_weights = labeler.calculate_sample_weights(df_labeled)
            
            status_text.text("Training model with purged cross-validation...")
            progress_bar.progress(70)
            
            cv_results = trainer.train_with_purged_cv(
                X, y,
                sample_weights=sample_weights,
                n_splits=int(n_splits),
                purge_gap=int(purge_gap)
            )
            
            progress_bar.progress(90)
            
            status_text.text("Saving model...")
            trainer.save_model(model_name)
            
            progress_bar.progress(100)
            status_text.text("Training complete")
            
            st.success("Model trained successfully")
            
            st.markdown("### Training Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean CV Accuracy", f"{cv_results['mean_score']:.4f}")
            with col2:
                st.metric("Std CV Accuracy", f"{cv_results['std_score']:.4f}")
            with col3:
                st.metric("CV Folds", len(cv_results['cv_scores']))
            
            st.markdown("### Fold Scores")
            fold_df = pd.DataFrame({
                'Fold': range(1, len(cv_results['cv_scores']) + 1),
                'Accuracy': cv_results['cv_scores']
            })
            st.dataframe(fold_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())