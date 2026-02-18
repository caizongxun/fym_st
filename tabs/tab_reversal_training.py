import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import plotly.graph_objects as go

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from ui.selectors import symbol_selector

def render_reversal_training_tab(loader):
    st.header("步驟 2: BB 反轉模型訓練")
    
    st.info("""
    **精簡策略**: 只用最相關的特徵
    - BB 位置和寬度
    - 波動率比值
    - RSI
    - 成交量
    - K 棒形態
    - 連續碰觸次數
    """)
    
    # 參數設定
    st.subheader("訓練參數")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbols = symbol_selector(loader, "reversal_train", multi=False)
        symbol = symbols[0]
        
        if isinstance(loader, BinanceDataLoader):
            train_days = st.slider("訓練天數", 30, 180, 90, key="train_days")
        else:
            st.info("使用 HuggingFace 最近 90 天")
            train_days = 90
    
    with col2:
        lookback_period = st.number_input(
            "歷史波動期數",
            5, 50, 20,
            key="train_lookback"
        )
    
    with col3:
        volatility_multiplier = st.number_input(
            "波動倍數",
            1.0, 5.0, 2.0, 0.5,
            key="train_multiplier"
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        bb_period = st.number_input("BB 週期", 10, 50, 20, key="train_bb_period")
    
    with col5:
        bb_std = st.number_input("BB 標準差", 1.0, 3.0, 2.0, 0.1, key="train_bb_std")
    
    with col6:
        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05, key="test_size")
    
    # 模型超參數
    with st.expander("進階設定: XGBoost 超參數"):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_depth = st.slider("max_depth", 3, 10, 5, key="max_depth", help="樹深度,降低可防止過擬合")
            n_estimators = st.slider("n_estimators", 50, 300, 100, 50, key="n_estimators")
        with col2:
            learning_rate = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.2], 0.1, key="lr")
            min_child_weight = st.slider("min_child_weight", 1, 10, 1, key="min_child", help="增加可提高召回率")
        with col3:
            subsample = st.slider("subsample", 0.5, 1.0, 0.8, 0.1, key="subsample")
            colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.1, key="colsample")
    
    if st.button("開始訓練", key="start_training", type="primary"):
        with st.spinner(f"正在訓練 {symbol} 反轉模型..."):
            try:
                # 1. 載入數據
                st.write("步驟 1/5: 載入數據...")
                if isinstance(loader, BinanceDataLoader):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=train_days)
                    df = loader.load_historical_data(symbol, '15m', start_date, end_date)
                else:
                    df = loader.load_klines(symbol, '15m').tail(train_days * 96)
                
                st.success(f"載入 {len(df)} 根 K 線")
                
                # 2. 生成標籤
                st.write("步驟 2/5: 生成標籤...")
                df = generate_labels(df, bb_period, bb_std, lookback_period, volatility_multiplier)
                
                upper_labels = df['label_upper_reversal'].sum()
                lower_labels = df['label_lower_reversal'].sum()
                
                st.info(f"上軌反轉標籤: {upper_labels} | 下軌反轉標籤: {lower_labels}")
                
                if upper_labels < 10 or lower_labels < 10:
                    st.warning("標籤數量過少! 建議降低波動倍數或增加訓練天數")
                    return
                
                # 3. 提取精簡特徵
                st.write("步驟 3/5: 提取精簡特徵...")
                df = extract_simple_features(df, bb_period)
                
                # 4. 訓練上軌模型
                st.write("步驟 4/5: 訓練上軌反轉模型...")
                xgb_params = {
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'n_estimators': n_estimators,
                    'min_child_weight': min_child_weight,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree
                }
                upper_model, upper_metrics, upper_importance = train_model(
                    df[df['touch_upper']].copy(),
                    'label_upper_reversal',
                    test_size,
                    xgb_params
                )
                
                # 5. 訓練下軌模型
                st.write("步驟 5/5: 訓練下軌反轉模型...")
                lower_model, lower_metrics, lower_importance = train_model(
                    df[df['touch_lower']].copy(),
                    'label_lower_reversal',
                    test_size,
                    xgb_params
                )
                
                # 顯示結果
                st.success("訓練完成!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("上軌反轉模型")
                    display_metrics(upper_metrics)
                    st.plotly_chart(plot_feature_importance(upper_importance, "上軌"), use_container_width=True)
                
                with col2:
                    st.subheader("下軌反轉模型")
                    display_metrics(lower_metrics)
                    st.plotly_chart(plot_feature_importance(lower_importance, "下軌"), use_container_width=True)
                
                # 保存模型
                if st.button("保存模型", key="save_models"):
                    save_model_package(symbol, upper_model, lower_model, 
                                     bb_period, bb_std, lookback_period, 
                                     volatility_multiplier, upper_metrics, lower_metrics)
                
            except Exception as e:
                st.error(f"訓練失敗: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

def generate_labels(df, bb_period, bb_std, lookback_period, volatility_multiplier):
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_std * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - bb_std * df['bb_std']
    
    df['price_range'] = df['high'] - df['low']
    df['avg_historical_volatility'] = df['price_range'].rolling(window=lookback_period).mean()
    df['volatility_threshold'] = df['avg_historical_volatility'] * volatility_multiplier
    
    df['touch_upper'] = df['high'] >= df['bb_upper'] * 0.999
    df['touch_lower'] = df['low'] <= df['bb_lower'] * 1.001
    
    df['future_5bar_min'] = df['low'].shift(-5).rolling(window=5, min_periods=1).min()
    df['future_5bar_max'] = df['high'].shift(-5).rolling(window=5, min_periods=1).max()
    df['future_drop'] = df['high'] - df['future_5bar_min']
    df['future_rise'] = df['future_5bar_max'] - df['low']
    
    df['label_upper_reversal'] = (
        df['touch_upper'] & 
        (df['future_drop'] > df['volatility_threshold'])
    ).astype(int)
    
    df['label_lower_reversal'] = (
        df['touch_lower'] & 
        (df['future_rise'] > df['volatility_threshold'])
    ).astype(int)
    
    return df

def extract_simple_features(df, bb_period):
    """只提取最相關的特徵"""
    # BB 位置特徵 (最重要)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['dist_to_upper_pct'] = (df['bb_upper'] - df['close']) / df['close'] * 100
    df['dist_to_lower_pct'] = (df['close'] - df['bb_lower']) / df['close'] * 100
    
    # BB 寬度 (波動率)
    df['bb_width_pct'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    df['bb_width_ratio'] = df['bb_width_pct'] / df['bb_width_pct'].rolling(50).mean()
    
    # 當前波動率比值
    df['volatility_ratio'] = df['price_range'] / df['avg_historical_volatility']
    
    # RSI (超買超賣)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 成交量異常
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # K 棒形態
    df['body_size_pct'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open'] * 100
    df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open'] * 100
    
    # 連續碰觸 (重要!)
    df['touch_count_5'] = df['touch_upper'].rolling(5).sum() + df['touch_lower'].rolling(5).sum()
    
    return df

def train_model(df, label_col, test_size, xgb_params):
    """訓練模型 - 精簡特徵版"""
    feature_cols = [
        # BB 位置 (3)
        'bb_position', 'dist_to_upper_pct', 'dist_to_lower_pct',
        # BB 寬度 (2)
        'bb_width_pct', 'bb_width_ratio',
        # 波動率 (1)
        'volatility_ratio',
        # RSI (1)
        'rsi',
        # 成交量 (1)
        'volume_ratio',
        # K 棒形態 (3)
        'body_size_pct', 'upper_wick_pct', 'lower_wick_pct',
        # 連續碰觸 (1)
        'touch_count_5',
        # 歷史波動 (2)
        'avg_historical_volatility', 'volatility_threshold'
    ]  # 總共 14 個特徵
    
    df_clean = df[feature_cols + [label_col]].dropna()
    X = df_clean[feature_cols]
    y = df_clean[label_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 計算 scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        max_depth=xgb_params['max_depth'],
        learning_rate=xgb_params['learning_rate'],
        n_estimators=xgb_params['n_estimators'],
        min_child_weight=xgb_params['min_child_weight'],
        subsample=xgb_params['subsample'],
        colsample_bytree=xgb_params['colsample_bytree'],
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = (y_pred == y_test).mean() * 100
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # 特徵重要性
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'report': report,
        'confusion_matrix': cm
    }
    
    return model, metrics, importance

def display_metrics(metrics):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AUC", f"{metrics['auc']:.3f}")
    with col2:
        st.metric("准確率", f"{metrics['accuracy']:.1f}%")
    
    st.text("混淆矩陣:")
    cm = metrics['confusion_matrix']
    st.text(f"TN: {cm[0,0]}  FP: {cm[0,1]}")
    st.text(f"FN: {cm[1,0]}  TP: {cm[1,1]}")
    
    with st.expander("詳細分類報告"):
        st.text(metrics['report'])

def plot_feature_importance(importance, title):
    fig = go.Figure(go.Bar(
        x=importance['importance'],
        y=importance['feature'],
        orientation='h'
    ))
    fig.update_layout(
        title=f"{title}模型 - 特徵重要性",
        height=400,
        xaxis_title="重要性",
        yaxis_title="特徵"
    )
    return fig

def save_model_package(symbol, upper_model, lower_model, bb_period, bb_std, 
                       lookback_period, volatility_multiplier, upper_metrics, lower_metrics):
    model_dir = 'models/saved'
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{symbol}_bb_reversal_{timestamp}.pkl"
    model_path = os.path.join(model_dir, model_name)
    
    model_package = {
        'upper_model': upper_model,
        'lower_model': lower_model,
        'feature_columns': upper_model.get_booster().feature_names,
        'params': {
            'bb_period': bb_period,
            'bb_std': bb_std,
            'lookback_period': lookback_period,
            'volatility_multiplier': volatility_multiplier
        },
        'metrics': {
            'upper': upper_metrics,
            'lower': lower_metrics
        }
    }
    
    joblib.dump(model_package, model_path)
    st.success(f"模型已保存: {model_name}")