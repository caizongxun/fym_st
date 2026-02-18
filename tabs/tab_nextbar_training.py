import streamlit as st
import os
from datetime import datetime, timedelta

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from models.train_nextbar_model import NextBarModelTrainer
from ui.selectors import symbol_selector

def render_nextbar_training_tab(loader):
    """
    下一根K棒預測訓練 Tab
    
    Args:
        loader: 數據加載器
    """
    st.header("下一根K棒高低點預測 - 訓練")
    
    st.success("""
    **策略說明**:
    - 使用過去20根K棒預測下一根K棒的最高價和最低價
    - 在預測低點掛做多限價單 (Maker 0.02%)
    - 在預測高點掛做空限價單 (Maker 0.02%)
    - 成交後以另一邊預測價作為止盈
    
    **模型輸出**:
    - high_pct: 下一根K棒最高價相對當前close的百分比
    - low_pct: 下一根K棒最低價相對當前close的百分比
    
    **目標**: MAE < 0.25% (優秀), < 0.30% (合格)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        nextbar_symbols = symbol_selector(loader, "nextbar_train", multi=False)
        nextbar_symbol = nextbar_symbols[0]
        
        if isinstance(loader, BinanceDataLoader):
            nextbar_days = st.slider(
                "訓練天數",
                min_value=30,
                max_value=180,
                value=90,
                key="nextbar_train_days"
            )
        else:
            st.info("使用HuggingFace全部數據")
            nextbar_days = None
        
        max_range_pct = st.slider(
            "過濾異常波動 (%)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            key="nextbar_max_range",
            help="過濾區間 > 此值的K棒"
        ) / 100
    
    with col2:
        st.subheader("模型設定")
        
        model_type = st.selectbox(
            "模型類型",
            ["xgboost", "lightgbm"],
            key="nextbar_model_type",
            help="XGBoost: 穩定 | LightGBM: 快速"
        )
        
        oos_days = st.number_input(
            "OOS測試天數",
            min_value=7,
            max_value=60,
            value=30,
            key="nextbar_oos_days"
        )
        
        test_size = st.slider(
            "驗證集比例",
            min_value=0.1,
            max_value=0.3,
            value=0.2,
            step=0.05,
            key="nextbar_test_size"
        )
    
    st.caption(
        f"訓練: {nextbar_symbol} | OOS: {oos_days}天 | "
        f"模型: {model_type.upper()} | 過濾: >{max_range_pct*100:.1f}%"
    )
    
    if st.button("開始訓練", key="train_nextbar_btn", type="primary"):
        with st.spinner(f"正在訓練 {nextbar_symbol} 下一根K棒預測模型..."):
            try:
                # 載入數據
                if isinstance(loader, BinanceDataLoader):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=nextbar_days)
                    df = loader.load_historical_data(
                        nextbar_symbol, '15m', start_date, end_date
                    )
                else:
                    df = loader.load_klines(nextbar_symbol, '15m')
                
                st.info(f"載入 {len(df)} 根K線")
                
                # 訓練模型
                trainer = NextBarModelTrainer(model_type=model_type)
                
                metrics = trainer.train_with_oos(
                    df,
                    max_range_pct=max_range_pct,
                    oos_days=oos_days,
                    test_size=test_size
                )
                
                # 顯示結果
                st.subheader("訓練結果")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "HIGH 訓練MAE",
                        f"{metrics['high_train_mae']*100:.3f}%"
                    )
                    st.metric(
                        "HIGH 驗證MAE",
                        f"{metrics['high_val_mae']*100:.3f}%"
                    )
                    st.metric(
                        "HIGH OOS MAE",
                        f"{metrics['high_oos_mae']*100:.3f}%"
                    )
                
                with col2:
                    st.metric(
                        "LOW 訓練MAE",
                        f"{metrics['low_train_mae']*100:.3f}%"
                    )
                    st.metric(
                        "LOW 驗證MAE",
                        f"{metrics['low_val_mae']*100:.3f}%"
                    )
                    st.metric(
                        "LOW OOS MAE",
                        f"{metrics['low_oos_mae']*100:.3f}%"
                    )
                
                with col3:
                    st.metric(
                        "HIGH OOS RMSE",
                        f"{metrics['high_oos_rmse']*100:.3f}%"
                    )
                    st.metric(
                        "LOW OOS RMSE",
                        f"{metrics['low_oos_rmse']*100:.3f}%"
                    )
                    st.metric(
                        "區間 OOS MAE",
                        f"{metrics['range_oos_mae']*100:.3f}%"
                    )
                
                # 樣本數
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("訓練樣本", metrics['train_samples'])
                with col2:
                    st.metric("驗證樣本", metrics['val_samples'])
                with col3:
                    st.metric("OOS樣本", metrics['oos_samples'])
                
                # 評估模型
                high_mae = metrics['high_oos_mae'] * 100
                low_mae = metrics['low_oos_mae'] * 100
                avg_mae = (high_mae + low_mae) / 2
                
                st.subheader("模型評估")
                
                if avg_mae < 0.20:
                    st.success(
                        f"優秀! 平均MAE = {avg_mae:.3f}% < 0.20%, "
                        "可以用於實盤交易"
                    )
                elif avg_mae < 0.30:
                    st.info(
                        f"合格! 平均MAE = {avg_mae:.3f}% < 0.30%, "
                        "建議先回測驗證"
                    )
                else:
                    st.warning(
                        f"需要優化! 平均MAE = {avg_mae:.3f}% > 0.30%"
                    )
                    st.write("建議:")
                    st.write("1. 增加訓練數據")
                    st.write("2. 降低 max_range_pct 過濾參數")
                    st.write("3. 嘗試另一種模型")
                
                # 重要特徵
                st.subheader("TOP 10 重要特徵 (HIGH)")
                st.dataframe(
                    metrics['feature_importance_high'].head(10),
                    use_container_width=True
                )
                
                st.subheader("TOP 10 重要特徵 (LOW)")
                st.dataframe(
                    metrics['feature_importance_low'].head(10),
                    use_container_width=True
                )
                
                # 保存模型
                trainer.save_model(nextbar_symbol, prefix='nextbar_v1')
                model_filename = f"{nextbar_symbol}_nextbar_v1_nextbar_{model_type}.pkl"
                st.success(f"模型已保存: models/saved/{model_filename}")
                
            except Exception as e:
                st.error(f"訓練失敗: {str(e)}")
                import traceback
                st.code(traceback.format_exc())