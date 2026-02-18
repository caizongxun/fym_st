import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from ui.selectors import symbol_selector

def render_reversal_training_tab(loader):
    """
    Tab 2: BB 反轉模型訓練
    訓練上軌和下軌反轉預測模型
    """
    st.header("步驟 2: BB 反轉模型訓練")
    
    st.info("""
    **訓練流程**:
    1. 設定標籤參數 (歷史波動期數, 波動倍數)
    2. 載入數據並生成標籤
    3. 提取特徵 (距離軌道, 波動率, RSI, 成交量等)
    4. 分開訓練上軌模型和下軌模型
    5. 評估模型效果
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
            key="train_lookback",
            help="計算前 N 根 K 棒的平均波動"
        )
    
    with col3:
        volatility_multiplier = st.number_input(
            "波動倍數",
            1.0, 5.0, 2.0, 0.5,
            key="train_multiplier",
            help="未來反轉幅度需 > 歷史波動的 X 倍"
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        bb_period = st.number_input("BB 週期", 10, 50, 20, key="train_bb_period")
    
    with col5:
        bb_std = st.number_input("BB 標準差", 1.0, 3.0, 2.0, 0.1, key="train_bb_std")
    
    with col6:
        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05, key="test_size")
    
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
                
                # 2. 計算 BB 和標籤
                st.write("步驟 2/5: 生成標籤...")
                df = generate_labels(df, bb_period, bb_std, lookback_period, volatility_multiplier)
                
                upper_labels = df['label_upper_reversal'].sum()
                lower_labels = df['label_lower_reversal'].sum()
                
                st.info(f"上軌反轉標籤: {upper_labels} | 下軌反轉標籤: {lower_labels}")
                
                if upper_labels < 10 or lower_labels < 10:
                    st.warning("標籤數量過少! 建議降低波動倍數或增加訓練天數")
                    return
                
                # 3. 提取特徵
                st.write("步驟 3/5: 提取特徵...")
                df = extract_features(df)
                
                # 4. 訓練上軌模型
                st.write("步驟 4/5: 訓練上軌反轉模型...")
                upper_model, upper_metrics = train_model(
                    df[df['touch_upper']].copy(),
                    'label_upper_reversal',
                    test_size
                )
                
                # 5. 訓練下軌模型
                st.write("步驟 5/5: 訓練下軌反轉模型...")
                lower_model, lower_metrics = train_model(
                    df[df['touch_lower']].copy(),
                    'label_lower_reversal',
                    test_size
                )
                
                # 顯示結果
                st.success("訓練完成!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("上軌反轉模型")
                    st.metric("測試集 AUC", f"{upper_metrics['auc']:.3f}")
                    st.metric("測試集准確率", f"{upper_metrics['accuracy']:.1f}%")
                    st.text("分類報告:")
                    st.text(upper_metrics['report'])
                
                with col2:
                    st.subheader("下軌反轉模型")
                    st.metric("測試集 AUC", f"{lower_metrics['auc']:.3f}")
                    st.metric("測試集准確率", f"{lower_metrics['accuracy']:.1f}%")
                    st.text("分類報告:")
                    st.text(lower_metrics['report'])
                
                # 保存模型
                if st.button("保存模型", key="save_models"):
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
                
            except Exception as e:
                st.error(f"訓練失敗: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

def generate_labels(df, bb_period, bb_std, lookback_period, volatility_multiplier):
    """生成反轉標籤"""
    # 計算 BB
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_std * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - bb_std * df['bb_std']
    
    # 計算歷史波動
    df['price_range'] = df['high'] - df['low']
    df['avg_historical_volatility'] = df['price_range'].rolling(window=lookback_period).mean()
    df['volatility_threshold'] = df['avg_historical_volatility'] * volatility_multiplier
    
    # 碰觸軌道
    df['touch_upper'] = df['high'] >= df['bb_upper'] * 0.999
    df['touch_lower'] = df['low'] <= df['bb_lower'] * 1.001
    
    # 未來變化
    df['future_5bar_min'] = df['low'].shift(-5).rolling(window=5, min_periods=1).min()
    df['future_5bar_max'] = df['high'].shift(-5).rolling(window=5, min_periods=1).max()
    df['future_drop'] = df['high'] - df['future_5bar_min']
    df['future_rise'] = df['future_5bar_max'] - df['low']
    
    # 生成標籤
    df['label_upper_reversal'] = (
        df['touch_upper'] & 
        (df['future_drop'] > df['volatility_threshold'])
    ).astype(int)
    
    df['label_lower_reversal'] = (
        df['touch_lower'] & 
        (df['future_rise'] > df['volatility_threshold'])
    ).astype(int)
    
    return df

def extract_features(df):
    """提取特徵"""
    # 距離 BB 軌道
    df['dist_to_upper_pct'] = (df['bb_upper'] - df['close']) / df['close'] * 100
    df['dist_to_lower_pct'] = (df['close'] - df['bb_lower']) / df['close'] * 100
    df['dist_to_mid_pct'] = (df['close'] - df['bb_mid']) / df['close'] * 100
    
    # BB 寬度
    df['bb_width_pct'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    
    # 波動率
    df['volatility_ratio'] = df['price_range'] / df['avg_historical_volatility']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 成交量比率
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # K 棒特徵
    df['body_size'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open'] * 100
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open'] * 100
    
    return df

def train_model(df, label_col, test_size):
    """訓練模型"""
    # 特徵欄位
    feature_cols = [
        'dist_to_upper_pct', 'dist_to_lower_pct', 'dist_to_mid_pct',
        'bb_width_pct', 'volatility_ratio', 'rsi', 'volume_ratio',
        'body_size', 'upper_shadow', 'lower_shadow',
        'avg_historical_volatility', 'volatility_threshold'
    ]
    
    # 準備數據
    df_clean = df[feature_cols + [label_col]].dropna()
    X = df_clean[feature_cols]
    y = df_clean[label_col]
    
    # 分割訓練/測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 訓練 XGBoost
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 評估
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = (y_pred == y_test).mean() * 100
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'report': report
    }
    
    return model, metrics