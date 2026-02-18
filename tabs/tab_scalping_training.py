import streamlit as st
import os
from datetime import datetime, timedelta

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from models.train_scalping_model import ScalpingModelTrainer
from ui.selectors import symbol_selector

def render_scalping_training_tab(loader):
    """
    剝頭皮模型訓練 Tab (OOS + 多模型)
    
    Args:
        loader: 數據加載器 (HuggingFaceKlineLoader 或 BinanceDataLoader)
    """
    st.header("剝頭皮模型訓練 (OOS + 多模型)")
    
    st.success("""
    **剝頭皮策略 - OOS驗證 + 多模型選擇**:
    - 不依賴BB通道，使用40+多維技術指標
    - 預測未來5根K線內價格變動 ≥ 0.3%
    - 三分類: 做多(1) / 做空(0) / 觀望(2)
    
    **OOS驗證**:
    - 訓練集: 主要訓練數據
    - 驗證集: 訓練集內的20%，用於Early Stopping
    - OOS測試集: 最後30天，完全獨立的測試數據
    
    **模型選擇**:
    - LightGBM: 快速，適合大數據
    - XGBoost: 穩定，泛化能力強
    - CatBoost: 對類別特徵友好
    - Ensemble: 組合LightGBM+XGBoost投票
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        scalp_train_symbols = symbol_selector(loader, "scalp_train", multi=False)
        scalp_train_symbol = scalp_train_symbols[0]
        
        scalp_train_candles = st.number_input(
            "訓練K棒數量",
            min_value=10000,
            max_value=50000,
            value=30000,
            step=5000,
            key="scalp_train_candles",
            help="OOS之前的數據用於訓練"
        )
        
        scalp_oos_days = st.number_input(
            "OOS測試天數",
            min_value=7,
            max_value=60,
            value=30,
            step=7,
            key="scalp_oos_days",
            help="最後N天作為OOS測試集"
        )
        
        target_pct = st.slider(
            "目標利潤 (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
            key="scalp_target_pct"
        ) / 100
    
    with col2:
        st.subheader("模型與參數")
        
        model_type = st.selectbox(
            "模型類型",
            ["lightgbm", "xgboost", "catboost", "ensemble"],
            key="scalp_model_type",
            help="LightGBM: 快 | XGBoost: 穩定 | CatBoost: 類別友好 | Ensemble: 投票"
        )
        
        lookforward = st.number_input(
            "預測K線數",
            min_value=3,
            max_value=10,
            value=5,
            key="scalp_lookforward",
            help="預測未來N根K線內的價格變動"
        )
        
        risk_reward = st.slider(
            "風報比過濾",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            key="scalp_rr",
            help="上漨空間必須 > 下跌空間 * 風報比"
        )
    
    st.caption(
        f"訓練: {scalp_train_candles}根 | OOS: {scalp_oos_days}天 | "
        f"模型: {model_type.upper()} | 目標: {target_pct*100:.1f}% | "
        f"預測: {lookforward}根K線 | 風報比: {risk_reward}"
    )
    
    if st.button("開始OOS訓練", key="train_scalp_oos_btn", type="primary"):
        with st.spinner(f"正在訓練 {scalp_train_symbol} 剝頭皮模型 (OOS模式)..."):
            try:
                # 載入數據
                if isinstance(loader, BinanceDataLoader):
                    oos_candles = scalp_oos_days * 96
                    total_candles = scalp_train_candles + oos_candles
                    days_needed = (total_candles / 96) + 5
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days_needed)
                    df_full = loader.load_historical_data(
                        scalp_train_symbol, '15m', start_date, end_date
                    )
                else:
                    df_full = loader.load_klines(scalp_train_symbol, '15m')
                    oos_candles = scalp_oos_days * 96
                
                st.info(f"載入 {len(df_full)} 根K線")
                
                # 訓練模型 (OOS模式)
                trainer = ScalpingModelTrainer(
                    model_dir='models/saved',
                    model_type=model_type
                )
                trainer.label_generator.target_pct = target_pct
                trainer.label_generator.lookforward = lookforward
                trainer.label_generator.risk_reward_ratio = risk_reward
                
                metrics = trainer.train_with_oos(
                    df_full,
                    target_pct=target_pct,
                    lookforward=lookforward,
                    oos_days=scalp_oos_days
                )
                
                # 顯示結果
                st.subheader("訓練結果")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("訓練集準確率", f"{metrics['train_accuracy']:.2%}")
                with col2:
                    st.metric("驗證集準確率", f"{metrics['val_accuracy']:.2%}")
                with col3:
                    st.metric("OOS準確率", f"{metrics['oos_accuracy']:.2%}")
                with col4:
                    val_oos_gap = metrics['val_accuracy'] - metrics['oos_accuracy']
                    st.metric("泛化差距 (Val-OOS)", f"{val_oos_gap:.2%}")
                
                # 樣本數
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("訓練樣本", metrics['train_samples'])
                with col2:
                    st.metric("驗證樣本", metrics['val_samples'])
                with col3:
                    st.metric("OOS樣本", metrics['oos_samples'])
                
                # 特徵重要性 (只有單一模型才有)
                if model_type != 'ensemble' and metrics.get('feature_importance') is not None:
                    st.subheader("Top 20 重要特徵")
                    importance_df = metrics['feature_importance']
                    st.dataframe(importance_df, use_container_width=True)
                
                # 保存模型
                trainer.save_model(
                    scalp_train_symbol,
                    prefix=f'scalping_{model_type}_oos'
                )
                model_filename = f"{scalp_train_symbol}_scalping_{model_type}_oos_scalping_{model_type}.pkl"
                st.success(f"模型已保存: models/saved/{model_filename}")
                
                # 評估模型
                if metrics['oos_accuracy'] >= 0.60 and val_oos_gap < 0.10:
                    st.balloons()
                    st.success(
                        "OOS準確率 ≥ 60% 且泛化良好! 可以進行回測"
                    )
                elif metrics['oos_accuracy'] >= 0.55:
                    st.info(
                        "模型表現尚可，建議試試不同模型或調整參數"
                    )
                else:
                    st.warning(
                        f"OOS準確率偏低: {metrics['oos_accuracy']:.2%}，建議:"
                    )
                    st.write("1. 增加訓練數據")
                    st.write("2. 調整目標利潤/預測K線數")
                    st.write("3. 嘗試不同模型 (XGBoost/Ensemble)")
                    st.write("4. 調高風報比過濾")
                    
            except Exception as e:
                st.error(f"訓練失敗: {str(e)}")
                import traceback
                st.code(traceback.format_exc())