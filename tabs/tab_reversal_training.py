import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, make_scorer
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
    
    training_mode = st.radio(
        "訓練模式",
        ["手動設定超參數", "超參數搜索 (網格搜索)", "超參數搜索 (貝葉斯優化)"],
        horizontal=True
    )
    
    if training_mode == "手動設定超參數":
        render_manual_training(loader)
    elif training_mode == "超參數搜索 (網格搜索)":
        render_grid_search_training(loader)
    else:
        render_bayesian_training(loader)

def render_manual_training(loader):
    st.info("手動調整所有超參數並訓練模型")
    
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
        lookback_period = st.number_input("歷史波動期數", 5, 50, 20, key="train_lookback")
    with col3:
        volatility_multiplier = st.number_input("波動倍數", 1.0, 5.0, 2.0, 0.5, key="train_multiplier")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        bb_period = st.number_input("BB 週期", 10, 50, 20, key="train_bb_period")
    with col5:
        bb_std = st.number_input("BB 標準差", 1.0, 3.0, 2.0, 0.1, key="train_bb_std")
    with col6:
        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05, key="test_size")
    
    with st.expander("進階設定: XGBoost 超參數"):
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
    
    with st.expander("進階設定: 機率閾值", expanded=True):
        probability_threshold = st.slider("機率閾值", 0.25, 0.65, 0.40, 0.05, key="prob_threshold")
    
    if st.button("開始訓練", key="start_training", type="primary"):
        train_manual_model(loader, symbol, train_days, bb_period, bb_std, lookback_period, 
                          volatility_multiplier, test_size, max_depth, learning_rate, n_estimators,
                          min_child_weight, gamma, scale_pos_weight_multiplier, probability_threshold)

def render_grid_search_training(loader):
    st.warning("""
    **重要**: 網格搜索只優化 XGBoost 超參數，不包含機率閾值和 scale_pos_weight。
    如果需要提高召回率，建議使用手動訓練模式並調整機率閾值。
    """)
    
    st.subheader("訓練參數")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbols = symbol_selector(loader, "grid_train", multi=False)
        symbol = symbols[0]
        if isinstance(loader, BinanceDataLoader):
            train_days = st.slider("訓練天數", 30, 180, 90, key="grid_days")
        else:
            train_days = 90
    
    with col2:
        lookback_period = st.number_input("歷史波動期數", 5, 50, 20, key="grid_lookback")
    with col3:
        volatility_multiplier = st.number_input("波動倍數", 1.0, 5.0, 2.0, 0.5, key="grid_multiplier")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        bb_period = st.number_input("BB 週期", 10, 50, 20, key="grid_bb_period")
    with col5:
        bb_std = st.number_input("BB 標準差", 1.0, 3.0, 2.0, 0.1, key="grid_bb_std")
    with col6:
        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05, key="grid_test_size")
    
    st.subheader("搜索範圍")
    
    search_type = st.radio("搜索粒度", ["快速搜索", "完整搜索"])
    
    if search_type == "快速搜索":
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 2],
            'gamma': [0, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'scale_pos_weight': [1.5, 2.0, 2.5]
        }
        st.info(f"將測試 {3*2*2*2*2*3} = 144 種組合")
    else:
        param_grid = {
            'max_depth': [4, 5, 6, 7],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [150, 200],
            'min_child_weight': [1, 2, 3],
            'gamma': [0, 0.1],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.8],
            'scale_pos_weight': [1.5, 2.0, 2.5]
        }
        st.info(f"將測試 {4*2*2*3*2*2*3} = 576 種組合")
    
    optimization_target = st.selectbox(
        "優化目標",
        ["Recall (召回率)", "F1 Score (平衡)", "Precision (精確率)", "AUC"]
    )
    
    probability_threshold = st.slider(
        "機率閾值 (搜索後使用)",
        0.25, 0.65, 0.35, 0.05,
        key="grid_prob_threshold",
        help="搜索完成後用此閾值評估模型"
    )
    
    if st.button("開始網格搜索", key="start_grid_search", type="primary"):
        train_with_grid_search(loader, symbol, train_days, bb_period, bb_std, lookback_period,
                              volatility_multiplier, test_size, param_grid, optimization_target, probability_threshold)

def render_bayesian_training(loader):
    if not OPTUNA_AVAILABLE:
        st.error("需要安裝 Optuna 套件!")
        st.code("pip install optuna")
        return
    
    st.info("""
    **貝葉斯優化 (Optuna)**: 智能搜索最佳超參數
    - 同時優化 XGBoost 參數和 scale_pos_weight
    - 比網格搜索快很多
    """)
    
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
        lookback_period = st.number_input("歷史波動期數", 5, 50, 20, key="bayes_lookback")
    with col3:
        volatility_multiplier = st.number_input("波動倍數", 1.0, 5.0, 2.0, 0.5, key="bayes_multiplier")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        bb_period = st.number_input("BB 週期", 10, 50, 20, key="bayes_bb_period")
    with col5:
        bb_std = st.number_input("BB 標準差", 1.0, 3.0, 2.0, 0.1, key="bayes_bb_std")
    with col6:
        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05, key="bayes_test_size")
    
    st.subheader("搜索設定")
    
    n_trials = st.slider("試驗次數", 20, 100, 50, 10)
    
    optimization_target = st.selectbox(
        "優化目標",
        ["Recall (召回率)", "F1 Score (平衡)", "Precision (精確率)", "AUC"]
    )
    
    probability_threshold = st.slider(
        "機率閾值 (搜索後使用)",
        0.25, 0.65, 0.35, 0.05,
        key="bayes_prob_threshold"
    )
    
    st.info(f"將執行 {n_trials} 次試驗")
    
    if st.button("開始貝葉斯優化", key="start_bayes", type="primary"):
        train_with_bayesian_optimization(loader, symbol, train_days, bb_period, bb_std, lookback_period,
                                        volatility_multiplier, test_size, n_trials, optimization_target, probability_threshold)

# ========== 訓練函數 ==========

def train_manual_model(loader, symbol, train_days, bb_period, bb_std, lookback_period, 
                      volatility_multiplier, test_size, max_depth, learning_rate, n_estimators,
                      min_child_weight, gamma, scale_pos_weight_multiplier, probability_threshold):
    with st.spinner(f"正在訓練 {symbol}..."):
        try:
            df = load_and_prepare_data(loader, symbol, train_days, bb_period, bb_std, lookback_period, volatility_multiplier)
            xgb_params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'n_estimators': n_estimators, 'min_child_weight': min_child_weight, 'gamma': gamma, 'scale_pos_weight_multiplier': scale_pos_weight_multiplier}
            upper_result = train_model_with_threshold(df[df['touch_upper']].copy(), 'label_upper_reversal', test_size, xgb_params, probability_threshold)
            lower_result = train_model_with_threshold(df[df['touch_lower']].copy(), 'label_lower_reversal', test_size, xgb_params, probability_threshold)
            display_training_results(upper_result, lower_result, symbol, bb_period, bb_std, lookback_period, volatility_multiplier, probability_threshold)
        except Exception as e:
            st.error(f"訓練失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def train_with_grid_search(loader, symbol, train_days, bb_period, bb_std, lookback_period,
                          volatility_multiplier, test_size, param_grid, optimization_target, probability_threshold):
    with st.spinner(f"正在執行網格搜索..."):
        try:
            df = load_and_prepare_data(loader, symbol, train_days, bb_period, bb_std, lookback_period, volatility_multiplier)
            st.write("正在搜索上軌模型最佳參數...")
            upper_best = grid_search_model(df[df['touch_upper']].copy(), 'label_upper_reversal', test_size, param_grid, optimization_target, probability_threshold)
            st.write("正在搜索下軌模型最佳參數...")
            lower_best = grid_search_model(df[df['touch_lower']].copy(), 'label_lower_reversal', test_size, param_grid, optimization_target, probability_threshold)
            st.success("網格搜索完成!")
            display_search_results(upper_best, lower_best, "網格搜索")
        except Exception as e:
            st.error(f"搜索失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def train_with_bayesian_optimization(loader, symbol, train_days, bb_period, bb_std, lookback_period,
                                    volatility_multiplier, test_size, n_trials, optimization_target, probability_threshold):
    with st.spinner(f"正在執行貝葉斯優化..."):
        try:
            df = load_and_prepare_data(loader, symbol, train_days, bb_period, bb_std, lookback_period, volatility_multiplier)
            st.write("正在優化上軌模型參數...")
            upper_best = bayesian_optimize_model(df[df['touch_upper']].copy(), 'label_upper_reversal', test_size, n_trials, optimization_target, probability_threshold)
            st.write("正在優化下軌模型參數...")
            lower_best = bayesian_optimize_model(df[df['touch_lower']].copy(), 'label_lower_reversal', test_size, n_trials, optimization_target, probability_threshold)
            st.success("貝葉斯優化完成!")
            display_bayesian_results(upper_best, lower_best)
        except Exception as e:
            st.error(f"優化失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ========== 輔助函數 ==========

def load_and_prepare_data(loader, symbol, train_days, bb_period, bb_std, lookback_period, volatility_multiplier):
    if isinstance(loader, BinanceDataLoader):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=train_days)
        df = loader.load_historical_data(symbol, '15m', start_date, end_date)
    else:
        df = loader.load_klines(symbol, '15m').tail(train_days * 96)
    df = generate_labels(df, bb_period, bb_std, lookback_period, volatility_multiplier)
    df = extract_simple_features(df, bb_period)
    return df

def grid_search_model(df, label_col, test_size, param_grid, optimization_target, probability_threshold):
    feature_cols = get_feature_cols()
    df_clean = df[feature_cols + [label_col]].dropna()
    X, y = df_clean[feature_cols], df_clean[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    scoring = get_scoring_metric(optimization_target)
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= probability_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    return {'best_params': grid_search.best_params_, 'best_score': grid_search.best_score_, 'model': best_model, 'metrics': calculate_metrics(y_test, y_pred, y_pred_proba, cm)}

def bayesian_optimize_model(df, label_col, test_size, n_trials, optimization_target, probability_threshold):
    feature_cols = get_feature_cols()
    df_clean = df[feature_cols + [label_col]].dropna()
    X, y = df_clean[feature_cols], df_clean[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 3.0),
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': 42
        }
        model = xgb.XGBClassifier(**params)
        score = cross_val_score(model, X_train, y_train, cv=3, scoring=get_scoring_metric(optimization_target), n_jobs=-1).mean()
        return score
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    best_model = xgb.XGBClassifier(**best_params, objective='binary:logistic', eval_metric='auc', random_state=42)
    best_model.fit(X_train, y_train)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= probability_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    return {'best_params': best_params, 'best_score': study.best_value, 'model': best_model, 'metrics': calculate_metrics(y_test, y_pred, y_pred_proba, cm), 'study': study}

def get_feature_cols():
    return ['bb_position', 'dist_to_upper_pct', 'dist_to_lower_pct', 'bb_width_pct', 'bb_width_ratio', 'volatility_ratio', 'rsi', 'volume_ratio', 'body_size_pct', 'upper_wick_pct', 'lower_wick_pct', 'touch_count_5', 'avg_historical_volatility', 'volatility_threshold']

def get_scoring_metric(optimization_target):
    if optimization_target == "F1 Score (平衡)": return 'f1'
    elif optimization_target == "Recall (召回率)": return 'recall'
    elif optimization_target == "Precision (精確率)": return 'precision'
    else: return 'roc_auc'

def calculate_metrics(y_test, y_pred, y_pred_proba, cm):
    tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
    return {'accuracy': (y_pred == y_test).mean() * 100, 'auc': roc_auc_score(y_test, y_pred_proba), 'precision': (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0, 'recall': (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0, 'f1': (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0, 'confusion_matrix': cm, 'report': classification_report(y_test, y_pred, zero_division=0)}

def display_search_results(upper_best, lower_best, title):
    st.subheader(f"{title}結果")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**上軌最佳參數**")
        st.json(upper_best['best_params'])
        st.metric("CV Score", f"{upper_best['best_score']:.3f}")
        st.write("**測試集表現**:")
        display_detailed_metrics(upper_best['metrics'])
    with col2:
        st.write("**下軌最佳參數**")
        st.json(lower_best['best_params'])
        st.metric("CV Score", f"{lower_best['best_score']:.3f}")
        st.write("**測試集表現**:")
        display_detailed_metrics(lower_best['metrics'])

def display_bayesian_results(upper_best, lower_best):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("上軌最佳參數")
        st.json(upper_best['best_params'])
        st.metric("Best Score", f"{upper_best['best_score']:.3f}")
        st.write("**測試集表現**:")
        display_detailed_metrics(upper_best['metrics'])
        with st.expander("優化歷史"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=[t.value for t in upper_best['study'].trials], mode='lines+markers'))
            fig.update_layout(title="優化進度", xaxis_title="Trial", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("下軌最佳參數")
        st.json(lower_best['best_params'])
        st.metric("Best Score", f"{lower_best['best_score']:.3f}")
        st.write("**測試集表現**:")
        display_detailed_metrics(lower_best['metrics'])
        with st.expander("優化歷史"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=[t.value for t in lower_best['study'].trials], mode='lines+markers'))
            fig.update_layout(title="優化進度", xaxis_title="Trial", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)

# 保留原有函數的簡化版本
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
    df['label_upper_reversal'] = (df['touch_upper'] & (df['future_drop'] > df['volatility_threshold'])).astype(int)
    df['label_lower_reversal'] = (df['touch_lower'] & (df['future_rise'] > df['volatility_threshold'])).astype(int)
    return df

def extract_simple_features(df, bb_period):
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['dist_to_upper_pct'] = (df['bb_upper'] - df['close']) / df['close'] * 100
    df['dist_to_lower_pct'] = (df['close'] - df['bb_lower']) / df['close'] * 100
    df['bb_width_pct'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    df['bb_width_ratio'] = df['bb_width_pct'] / df['bb_width_pct'].rolling(50).mean()
    df['volatility_ratio'] = df['price_range'] / df['avg_historical_volatility']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['body_size_pct'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open'] * 100
    df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open'] * 100
    df['touch_count_5'] = df['touch_upper'].rolling(5).sum() + df['touch_lower'].rolling(5).sum()
    return df

def train_model_with_threshold(df, label_col, test_size, xgb_params, prob_threshold):
    feature_cols = get_feature_cols()
    df_clean = df[feature_cols + [label_col]].dropna()
    X, y = df_clean[feature_cols], df_clean[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    scale_pos_weight = ((y_train == 0).sum() / (y_train == 1).sum()) * xgb_params['scale_pos_weight_multiplier']
    model = xgb.XGBClassifier(max_depth=xgb_params['max_depth'], learning_rate=xgb_params['learning_rate'], n_estimators=xgb_params['n_estimators'], min_child_weight=xgb_params['min_child_weight'], gamma=xgb_params['gamma'], subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, objective='binary:logistic', eval_metric='auc', random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= prob_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    importance = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    return {'model': model, 'metrics': calculate_metrics(y_test, y_pred, y_pred_proba, cm), 'importance': importance, 'pr_curve': {'precision': precision, 'recall': recall, 'thresholds': thresholds}}

def display_training_results(upper_result, lower_result, symbol, bb_period, bb_std, lookback_period, volatility_multiplier, probability_threshold):
    st.success("訓練完成!")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("上軌模型")
        display_detailed_metrics(upper_result['metrics'])
        st.plotly_chart(plot_feature_importance(upper_result['importance'], "上軌"), use_container_width=True)
    with col2:
        st.subheader("下軌模型")
        display_detailed_metrics(lower_result['metrics'])
        st.plotly_chart(plot_feature_importance(lower_result['importance'], "下軌"), use_container_width=True)
    if st.button("保存模型", key="save_models"):
        save_model_package(symbol, upper_result['model'], lower_result['model'], bb_period, bb_std, lookback_period, volatility_multiplier, probability_threshold, upper_result['metrics'], lower_result['metrics'])

def display_detailed_metrics(metrics):
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("AUC", f"{metrics['auc']:.3f}")
    with col2: st.metric("准確率", f"{metrics['accuracy']:.1f}%")
    with col3: st.metric("F1", f"{metrics['f1']:.3f}")
    col1, col2 = st.columns(2)
    with col1: st.metric("精確率", f"{metrics['precision']:.1f}%")
    with col2: st.metric("召回率", f"{metrics['recall']:.1f}%")
    cm = metrics['confusion_matrix']
    st.text(f"TN:{cm[0,0]} FP:{cm[0,1]} | FN:{cm[1,0]} TP:{cm[1,1]}")

def plot_feature_importance(importance, title):
    fig = go.Figure(go.Bar(x=importance['importance'], y=importance['feature'], orientation='h'))
    fig.update_layout(title=f"{title}特徵重要性", height=350)
    return fig

def save_model_package(symbol, upper_model, lower_model, bb_period, bb_std, lookback_period, volatility_multiplier, prob_threshold, upper_metrics, lower_metrics):
    model_dir = 'models/saved'
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{symbol}_bb_{timestamp}.pkl"
    joblib.dump({'upper_model': upper_model, 'lower_model': lower_model, 'params': {'bb_period': bb_period, 'bb_std': bb_std, 'lookback_period': lookback_period, 'volatility_multiplier': volatility_multiplier, 'probability_threshold': prob_threshold}, 'metrics': {'upper': upper_metrics, 'lower': lower_metrics}}, os.path.join(model_dir, model_name))
    st.success(f"模型已保存: {model_name}")