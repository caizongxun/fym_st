import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
import plotly.graph_objects as go

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from ui.selectors import symbol_selector

def render_reversal_training_tab(loader):
    st.header("步驟 2: BB 反轉模型訓練")
    
    st.info("""
    **訓練邏輯**: 使用前 N 根已完成 K 棒的資料,預測當前 K 棒觸碰 BB 軌道時是否會反彈
    - 特徵: 基於歷史 K 棒 (不包含當前)
    - 標籤: 當前 K 棒是否觸碰 + 未來是否反彈
    """)
    
    training_mode = st.radio(
        "訓練模式",
        ["手動設定超參數", "超參數搜索 (貝葉斯優化)"],
        horizontal=True
    )
    
    if training_mode == "手動設定超參數":
        render_manual_training(loader)
    else:
        render_bayesian_training(loader)

def render_manual_training(loader):
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
        lookback_candles = st.number_input(
            "歷史 K 棒數量", 20, 200, 100, 10,
            key="lookback_candles",
            help="使用前 N 根已完成 K 棒計算特徵"
        )
    with col3:
        future_candles = st.number_input(
            "未來觀察期", 3, 10, 5,
            key="future_candles",
            help="檢查未來 N 根 K 棒是否反彈"
        )
    
    col4, col5, col6 = st.columns(3)
    with col4:
        bb_period = st.number_input("BB 週期", 10, 50, 20, key="train_bb_period")
    with col5:
        bb_std = st.number_input("BB 標準差", 1.0, 3.0, 2.0, 0.1, key="train_bb_std")
    with col6:
        volatility_multiplier = st.number_input(
            "反彈幅度倍數", 1.0, 5.0, 2.0, 0.5,
            key="train_multiplier",
            help="反彈幅度 > 歷史波動 * 此倍數"
        )
    
    col7, col8 = st.columns(2)
    with col7:
        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05, key="test_size")
    with col8:
        probability_threshold = st.slider("機率閾值", 0.30, 0.60, 0.45, 0.05, key="prob_threshold")
    
    with st.expander("XGBoost 超參數"):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_depth = st.slider("max_depth", 3, 10, 6, key="max_depth")
            n_estimators = st.slider("n_estimators", 50, 300, 200, 50, key="n_estimators")
        with col2:
            learning_rate = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.2], 0.05, key="lr")
            min_child_weight = st.slider("min_child_weight", 1, 10, 1, key="min_child")
        with col3:
            gamma = st.slider("gamma", 0.0, 1.0, 0.1, 0.1, key="gamma")
            scale_pos_weight_multiplier = st.slider("scale_pos_weight 倍數", 0.5, 3.0, 1.5, 0.1, key="scale_multiplier")
    
    if st.button("開始訓練", key="start_training", type="primary"):
        train_manual_model(loader, symbol, train_days, lookback_candles, future_candles,
                          bb_period, bb_std, volatility_multiplier, test_size,
                          max_depth, learning_rate, n_estimators, min_child_weight,
                          gamma, scale_pos_weight_multiplier, probability_threshold)

def render_bayesian_training(loader):
    if not OPTUNA_AVAILABLE:
        st.error("需要安裝 Optuna 套件: pip install optuna")
        return
    
    st.subheader("訓練參數")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbols = symbol_selector(loader, "bayes_train", multi=False)
        symbol = symbols[0]
        if isinstance(loader, BinanceDataLoader):
            train_days = st.slider("訓練天數", 30, 180, 90, key="bayes_days")
        else:
            train_days = 90
    
    with col2:
        lookback_candles = st.number_input("歷史 K 棒數量", 20, 200, 100, 10, key="bayes_lookback")
    with col3:
        future_candles = st.number_input("未來觀察期", 3, 10, 5, key="bayes_future")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        bb_period = st.number_input("BB 週期", 10, 50, 20, key="bayes_bb_period")
    with col5:
        bb_std = st.number_input("BB 標準差", 1.0, 3.0, 2.0, 0.1, key="bayes_bb_std")
    with col6:
        volatility_multiplier = st.number_input("反彈幅度倍數", 1.0, 5.0, 2.0, 0.5, key="bayes_multiplier")
    
    col7, col8, col9 = st.columns(3)
    with col7:
        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05, key="bayes_test_size")
    with col8:
        n_trials = st.slider("試驗次數", 20, 100, 50, 10, key="bayes_trials")
    with col9:
        probability_threshold = st.slider("機率閾值", 0.30, 0.60, 0.45, 0.05, key="bayes_prob_threshold")
    
    if st.button("開始貝葉斯優化", key="start_bayes", type="primary"):
        train_with_bayesian(loader, symbol, train_days, lookback_candles, future_candles,
                           bb_period, bb_std, volatility_multiplier, test_size,
                           n_trials, probability_threshold)

# ========== 核心訓練邏輯 ==========

def train_manual_model(loader, symbol, train_days, lookback_candles, future_candles,
                      bb_period, bb_std, volatility_multiplier, test_size,
                      max_depth, learning_rate, n_estimators, min_child_weight,
                      gamma, scale_pos_weight_multiplier, probability_threshold):
    with st.spinner(f"正在訓練 {symbol}..."):
        try:
            df = load_data(loader, symbol, train_days)
            df = generate_labels_new(df, bb_period, bb_std, lookback_candles, 
                                    future_candles, volatility_multiplier)
            df = extract_features_new(df, bb_period, lookback_candles)
            
            xgb_params = {
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'min_child_weight': min_child_weight,
                'gamma': gamma,
                'scale_pos_weight_multiplier': scale_pos_weight_multiplier
            }
            
            upper_result = train_model(
                df[df['touch_upper']].copy(),
                'label_upper_reversal',
                test_size, xgb_params, probability_threshold
            )
            
            lower_result = train_model(
                df[df['touch_lower']].copy(),
                'label_lower_reversal',
                test_size, xgb_params, probability_threshold
            )
            
            # 儲存到 session_state
            st.session_state['trained_models'] = {
                'upper': upper_result,
                'lower': lower_result,
                'symbol': symbol,
                'params': {
                    'bb_period': bb_period,
                    'bb_std': bb_std,
                    'lookback_candles': lookback_candles,
                    'future_candles': future_candles,
                    'volatility_multiplier': volatility_multiplier,
                    'probability_threshold': probability_threshold
                }
            }
            
            display_results(upper_result, lower_result)
            
        except Exception as e:
            st.error(f"訓練失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def train_with_bayesian(loader, symbol, train_days, lookback_candles, future_candles,
                       bb_period, bb_std, volatility_multiplier, test_size,
                       n_trials, probability_threshold):
    with st.spinner("執行貝葉斯優化..."):
        try:
            df = load_data(loader, symbol, train_days)
            df = generate_labels_new(df, bb_period, bb_std, lookback_candles, 
                                    future_candles, volatility_multiplier)
            df = extract_features_new(df, bb_period, lookback_candles)
            
            st.write("優化上軌模型...")
            upper_best = bayesian_optimize(
                df[df['touch_upper']].copy(),
                'label_upper_reversal',
                test_size, n_trials, probability_threshold
            )
            
            st.write("優化下軌模型...")
            lower_best = bayesian_optimize(
                df[df['touch_lower']].copy(),
                'label_lower_reversal',
                test_size, n_trials, probability_threshold
            )
            
            # 儲存到 session_state
            st.session_state['trained_models'] = {
                'upper': upper_best,
                'lower': lower_best,
                'symbol': symbol,
                'params': {
                    'bb_period': bb_period,
                    'bb_std': bb_std,
                    'lookback_candles': lookback_candles,
                    'future_candles': future_candles,
                    'volatility_multiplier': volatility_multiplier,
                    'probability_threshold': probability_threshold
                }
            }
            
            st.success("優化完成!")
            display_bayesian_results(upper_best, lower_best)
            
        except Exception as e:
            st.error(f"優化失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ========== 新的標籤生成邏輯 ==========

def generate_labels_new(df, bb_period, bb_std, lookback_candles, future_candles, volatility_multiplier):
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_std * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - bb_std * df['bb_std']
    
    df['price_range'] = df['high'] - df['low']
    df['historical_volatility'] = df['price_range'].rolling(window=lookback_candles).mean()
    df['reversal_threshold'] = df['historical_volatility'] * volatility_multiplier
    
    df['bb_upper_prev'] = df['bb_upper'].shift(1)
    df['bb_lower_prev'] = df['bb_lower'].shift(1)
    
    df['touch_upper'] = df['high'] >= df['bb_upper_prev'] * 0.999
    df['touch_lower'] = df['low'] <= df['bb_lower_prev'] * 1.001
    
    df['future_min'] = df['low'].shift(-1).rolling(window=future_candles).min()
    df['future_max'] = df['high'].shift(-1).rolling(window=future_candles).max()
    
    df['future_drop'] = df['high'] - df['future_min']
    df['future_rise'] = df['future_max'] - df['low']
    
    df['label_upper_reversal'] = (
        df['touch_upper'] & 
        (df['future_drop'] > df['reversal_threshold'].shift(1))
    ).astype(int)
    
    df['label_lower_reversal'] = (
        df['touch_lower'] & 
        (df['future_rise'] > df['reversal_threshold'].shift(1))
    ).astype(int)
    
    return df

def extract_features_new(df, bb_period, lookback_candles):
    df['bb_position'] = (
        (df['close'].shift(1) - df['bb_lower'].shift(1)) / 
        (df['bb_upper'].shift(1) - df['bb_lower'].shift(1))
    )
    
    df['dist_to_upper_pct'] = (
        (df['bb_upper'].shift(1) - df['close'].shift(1)) / 
        df['close'].shift(1) * 100
    )
    
    df['dist_to_lower_pct'] = (
        (df['close'].shift(1) - df['bb_lower'].shift(1)) / 
        df['close'].shift(1) * 100
    )
    
    bb_width = df['bb_upper'].shift(1) - df['bb_lower'].shift(1)
    df['bb_width_pct'] = bb_width / df['bb_mid'].shift(1) * 100
    df['bb_width_ratio'] = bb_width / bb_width.rolling(50).mean()
    
    df['volatility_ratio'] = (
        df['price_range'].shift(1) / 
        df['historical_volatility'].shift(1)
    )
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].shift(1)
    
    df['volume_ratio'] = (
        df['volume'].shift(1) / 
        df['volume'].rolling(window=20).mean().shift(1)
    )
    
    df['body_size_pct'] = (
        abs(df['close'].shift(1) - df['open'].shift(1)) / 
        df['open'].shift(1) * 100
    )
    
    df['upper_wick_pct'] = (
        (df['high'].shift(1) - df[['open', 'close']].shift(1).max(axis=1)) / 
        df['open'].shift(1) * 100
    )
    
    df['lower_wick_pct'] = (
        (df[['open', 'close']].shift(1).min(axis=1) - df['low'].shift(1)) / 
        df['open'].shift(1) * 100
    )
    
    df['touch_count_5'] = (
        df['touch_upper'].shift(1).rolling(5).sum() + 
        df['touch_lower'].shift(1).rolling(5).sum()
    )
    
    return df

def get_feature_cols():
    return [
        'bb_position', 'dist_to_upper_pct', 'dist_to_lower_pct',
        'bb_width_pct', 'bb_width_ratio', 'volatility_ratio',
        'rsi', 'volume_ratio',
        'body_size_pct', 'upper_wick_pct', 'lower_wick_pct',
        'touch_count_5'
    ]

# ========== 模型訓練 ==========

def train_model(df, label_col, test_size, xgb_params, prob_threshold):
    feature_cols = get_feature_cols()
    df_clean = df[feature_cols + [label_col]].dropna()
    
    X, y = df_clean[feature_cols], df_clean[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    scale_pos_weight = (
        (y_train == 0).sum() / (y_train == 1).sum()
    ) * xgb_params['scale_pos_weight_multiplier']
    
    model = xgb.XGBClassifier(
        max_depth=xgb_params['max_depth'],
        learning_rate=xgb_params['learning_rate'],
        n_estimators=xgb_params['n_estimators'],
        min_child_weight=xgb_params['min_child_weight'],
        gamma=xgb_params['gamma'],
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= prob_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'metrics': calculate_metrics(y_test, y_pred, y_pred_proba, cm),
        'importance': importance
    }

def bayesian_optimize(df, label_col, test_size, n_trials, prob_threshold):
    feature_cols = get_feature_cols()
    df_clean = df[feature_cols + [label_col]].dropna()
    X, y = df_clean[feature_cols], df_clean[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.15),
            'n_estimators': trial.suggest_int('n_estimators', 100, 250, step=50),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'gamma': trial.suggest_float('gamma', 0.0, 0.3),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.2, 2.0),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42
        }
        model = xgb.XGBClassifier(**params)
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1', n_jobs=-1).mean()
        return score
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_model = xgb.XGBClassifier(**best_params, objective='binary:logistic', eval_metric='auc', random_state=42)
    best_model.fit(X_train, y_train)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= prob_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'best_params': best_params,
        'best_score': study.best_value,
        'model': best_model,
        'metrics': calculate_metrics(y_test, y_pred, y_pred_proba, cm),
        'study': study,
        'importance': importance
    }

# ========== 輔助函數 ==========

def load_data(loader, symbol, train_days):
    if isinstance(loader, BinanceDataLoader):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=train_days)
        df = loader.load_historical_data(symbol, '15m', start_date, end_date)
    else:
        df = loader.load_klines(symbol, '15m').tail(train_days * 96)
    return df

def calculate_metrics(y_test, y_pred, y_pred_proba, cm):
    tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
    return {
        'accuracy': (y_pred == y_test).mean() * 100,
        'auc': roc_auc_score(y_test, y_pred_proba),
        'precision': (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0,
        'recall': (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0,
        'f1': (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0,
        'confusion_matrix': cm,
        'report': classification_report(y_test, y_pred, zero_division=0)
    }

def display_results(upper_result, lower_result):
    st.success("訓練完成!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("上軌模型")
        display_metrics(upper_result['metrics'])
        st.plotly_chart(plot_importance(upper_result['importance'], "上軌"), use_container_width=True)
    
    with col2:
        st.subheader("下軌模型")
        display_metrics(lower_result['metrics'])
        st.plotly_chart(plot_importance(lower_result['importance'], "下軌"), use_container_width=True)
    
    # 使用 form 來解決狀態問題
    if 'trained_models' in st.session_state:
        if st.button("保存模型", key="save_manual_models", type="primary"):
            save_trained_models()

def display_bayesian_results(upper_best, lower_best):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("上軌最佳參數")
        st.json(upper_best['best_params'])
        st.metric("Best Score", f"{upper_best['best_score']:.3f}")
        display_metrics(upper_best['metrics'])
        st.plotly_chart(plot_importance(upper_best['importance'], "上軌"), use_container_width=True)
        with st.expander("優化歷史"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=[t.value for t in upper_best['study'].trials],
                mode='lines+markers'
            ))
            fig.update_layout(title="優化進度", xaxis_title="Trial", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("下軌最佳參數")
        st.json(lower_best['best_params'])
        st.metric("Best Score", f"{lower_best['best_score']:.3f}")
        display_metrics(lower_best['metrics'])
        st.plotly_chart(plot_importance(lower_best['importance'], "下軌"), use_container_width=True)
        with st.expander("優化歷史"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=[t.value for t in lower_best['study'].trials],
                mode='lines+markers'
            ))
            fig.update_layout(title="優化進度", xaxis_title="Trial", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)
    
    if 'trained_models' in st.session_state:
        if st.button("保存模型", key="save_bayes_models", type="primary"):
            save_trained_models()

def display_metrics(metrics):
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("AUC", f"{metrics['auc']:.3f}")
    with col2: st.metric("准確率", f"{metrics['accuracy']:.1f}%")
    with col3: st.metric("F1", f"{metrics['f1']:.3f}")
    col1, col2 = st.columns(2)
    with col1: st.metric("精確率", f"{metrics['precision']:.1f}%")
    with col2: st.metric("召回率", f"{metrics['recall']:.1f}%")
    cm = metrics['confusion_matrix']
    st.text(f"TN:{cm[0,0]} FP:{cm[0,1]} | FN:{cm[1,0]} TP:{cm[1,1]}")

def plot_importance(importance, title):
    fig = go.Figure(go.Bar(
        x=importance['importance'],
        y=importance['feature'],
        orientation='h'
    ))
    fig.update_layout(title=f"{title}特徵重要性", height=350)
    return fig

def save_trained_models():
    if 'trained_models' not in st.session_state:
        st.error("沒有可保存的模型")
        return
    
    model_data = st.session_state['trained_models']
    symbol = model_data['symbol']
    params = model_data['params']
    
    model_dir = 'models/saved'
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{symbol}_bb_{timestamp}.pkl"
    model_path = os.path.join(model_dir, model_name)
    
    try:
        joblib.dump({
            'upper_model': model_data['upper']['model'],
            'lower_model': model_data['lower']['model'],
            'params': {
                'bb_period': int(params['bb_period']),
                'bb_std': float(params['bb_std']),
                'lookback_candles': int(params['lookback_candles']),
                'future_candles': int(params['future_candles']),
                'volatility_multiplier': float(params['volatility_multiplier']),
                'probability_threshold': float(params['probability_threshold'])
            },
            'metrics': {
                'upper': model_data['upper']['metrics'],
                'lower': model_data['lower']['metrics']
            }
        }, model_path)
        
        st.success(f"模型已保存: {model_name}")
        st.info(f"完整路徑: {os.path.abspath(model_path)}")
        
        # 檢查檔案是否存在
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            st.success(f"確認檔案存在, 大小: {file_size / 1024:.2f} KB")
        else:
            st.error("檔案保存失敗!")
        
    except Exception as e:
        st.error(f"保存失敗: {str(e)}")
        import traceback
        st.code(traceback.format_exc())