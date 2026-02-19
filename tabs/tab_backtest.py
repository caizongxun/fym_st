import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from ui.selectors import symbol_selector

def render_backtest_tab(loader):
    st.header("步驟 3: 模型回測")
    
    st.info("""
    回測邏輯:
    1. 每根 K 棒開盤時預測一次
    2. 檢查這根 K 棒是否觸碰 BB 軌道
    3. 觸碰時使用預測機率決定是否交易
    4. 成交價格使用觸碰價格 (保守) 或下根 K 棒開盤價 (實際)
    """)
    
    model_file = select_model()
    if not model_file:
        return
    
    try:
        model_package = joblib.load(os.path.join('models/saved', model_file))
        
        if not isinstance(model_package, dict):
            st.error("模型格式錯誤! 請重新訓練並保存模型")
            return
        
        if 'upper_model' not in model_package or 'lower_model' not in model_package:
            st.error("模型檔案不完整! 請重新訓練")
            return
        
        if 'params' not in model_package:
            st.error("模型缺少參數資訊! 請重新訓練")
            return
        
        st.success(f"已載入模型: {model_file}")
        
        with st.expander("模型參數"):
            st.json(model_package['params'])
        
    except Exception as e:
        st.error(f"載入模型失敗: {str(e)}")
        st.warning("請確保使用最新版本訓練的模型")
        return
    
    st.subheader("回測參數")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbols = symbol_selector(loader, "backtest", multi=False)
        symbol = symbols[0]
        if isinstance(loader, BinanceDataLoader):
            test_days = st.slider("回測天數", 7, 60, 30, key="test_days")
        else:
            st.info("使用 HuggingFace 最近 30 天")
            test_days = 30
    
    with col2:
        initial_capital = st.number_input("初始資金 (USDT)", 1000, 100000, 10000, 1000)
    
    with col3:
        position_size_pct = st.slider("每次交易資金比例 (%)", 5, 50, 10, 5)
    
    col4, col5 = st.columns(2)
    
    with col4:
        execution_mode = st.selectbox(
            "成交價格模式",
            ["觸碰價 (保守)", "下根開盤價 (實際)"],
            help="保守: 假設在觸碰價格成交 | 實際: 使用下根 K 棒開盤價"
        )
    
    with col5:
        fee_rate = st.number_input("手續費率 (%)", 0.0, 0.2, 0.05, 0.01)
    
    st.subheader("止損止盈設定")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stop_loss_mode = st.selectbox(
            "止損模式",
            ["固定百分比", "ATR 倍數"],
            help="固定: 使用固定百分比 | ATR: 基於市場波動率動態調整"
        )
    
    with col2:
        if stop_loss_mode == "固定百分比":
            stop_loss_value = st.number_input("止損 (%)", 0.5, 5.0, 2.0, 0.5)
        else:
            stop_loss_value = st.number_input("ATR 倍數", 1.0, 5.0, 2.0, 0.5)
            atr_period = st.number_input("ATR 週期", 7, 21, 14)
    
    with col3:
        take_profit_mode = st.selectbox(
            "止盈模式",
            ["固定百分比", "ATR 倍數"],
        )
    
    col4, col5 = st.columns(2)
    
    with col4:
        if take_profit_mode == "固定百分比":
            take_profit_value = st.number_input("止盈 (%)", 1.0, 10.0, 3.0, 0.5)
        else:
            take_profit_value = st.number_input("ATR 倍數 (TP)", 1.0, 8.0, 3.0, 0.5)
    
    with col5:
        slippage = st.number_input("滑點 (%)", 0.0, 0.5, 0.02, 0.01)
    
    st.subheader("進場過濾條件")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_trend_filter = st.checkbox("啟用趋勢過濾", value=True, help="只在趋勢方向一致時交易")
        if use_trend_filter:
            trend_ma_period = st.number_input("趋勢 MA 週期", 20, 200, 100)
    
    with col2:
        use_volatility_filter = st.checkbox("啟用波動率過濾", value=True, help="只在波動率適中時交易")
        if use_volatility_filter:
            min_bb_width = st.number_input("最小 BB 寬度 (%)", 0.5, 3.0, 1.0, 0.1)
    
    with col3:
        use_prob_boost = st.checkbox("啟用機率加權", value=True, help="高機率信號增加倉位")
        if use_prob_boost:
            prob_boost_threshold = st.slider("加權門檻", 0.5, 0.9, 0.7, 0.05)
    
    if st.button("開始回測", type="primary"):
        filters = {
            'use_trend_filter': use_trend_filter,
            'trend_ma_period': trend_ma_period if use_trend_filter else None,
            'use_volatility_filter': use_volatility_filter,
            'min_bb_width': min_bb_width if use_volatility_filter else None,
            'use_prob_boost': use_prob_boost,
            'prob_boost_threshold': prob_boost_threshold if use_prob_boost else None
        }
        
        stop_loss_config = {
            'mode': stop_loss_mode,
            'value': stop_loss_value,
            'atr_period': atr_period if stop_loss_mode == "ATR 倍數" else None
        }
        
        take_profit_config = {
            'mode': take_profit_mode,
            'value': take_profit_value
        }
        
        run_backtest(
            loader, symbol, test_days, model_package,
            initial_capital, position_size_pct,
            execution_mode, stop_loss_config, take_profit_config,
            fee_rate, slippage, filters
        )

def select_model():
    model_dir = 'models/saved'
    if not os.path.exists(model_dir):
        st.warning("尚無保存的模型, 請先訓練模型")
        return None
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not model_files:
        st.warning("尚無保存的模型")
        return None
    
    model_files.sort(reverse=True)
    selected = st.selectbox("選擇模型", model_files)
    return selected

def run_backtest(loader, symbol, test_days, model_package,
                initial_capital, position_size_pct,
                execution_mode, stop_loss_config, take_profit_config,
                fee_rate, slippage, filters):
    with st.spinner("正在回測..."):
        try:
            df = load_backtest_data(loader, symbol, test_days)
            df = prepare_backtest_data(df, model_package['params'], stop_loss_config, filters)
            
            results = backtest_strategy(
                df, model_package,
                initial_capital, position_size_pct,
                execution_mode, stop_loss_config, take_profit_config,
                fee_rate, slippage, filters
            )
            
            display_backtest_results(results, df)
            
        except Exception as e:
            st.error(f"回測失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def load_backtest_data(loader, symbol, test_days):
    if isinstance(loader, BinanceDataLoader):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_days + 10)
        df = loader.load_historical_data(symbol, '15m', start_date, end_date)
    else:
        df = loader.load_klines(symbol, '15m').tail((test_days + 10) * 96)
    return df

def prepare_backtest_data(df, params, stop_loss_config, filters):
    bb_period = params['bb_period']
    bb_std = params['bb_std']
    lookback_candles = params.get('lookback_candles', 20)
    
    # BB 指標
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_std * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - bb_std * df['bb_std']
    
    # 歷史波動
    df['price_range'] = df['high'] - df['low']
    df['historical_volatility'] = df['price_range'].rolling(window=lookback_candles).mean()
    
    # ATR
    if stop_loss_config['mode'] == "ATR 倍數":
        atr_period = stop_loss_config['atr_period']
        df['atr'] = calculate_atr(df, atr_period)
    
    # 趋勢過濾
    if filters.get('use_trend_filter'):
        ma_period = filters['trend_ma_period']
        df['trend_ma'] = df['close'].rolling(window=ma_period).mean()
    
    # BB 寬度
    df['bb_width_pct'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_mid']) * 100
    
    # 觸碰檢測
    df['bb_upper_prev'] = df['bb_upper'].shift(1)
    df['bb_lower_prev'] = df['bb_lower'].shift(1)
    df['touch_upper'] = df['high'] >= df['bb_upper_prev'] * 0.999
    df['touch_lower'] = df['low'] <= df['bb_lower_prev'] * 1.001
    
    # 特徵
    df = extract_features_for_backtest(df, bb_period)
    
    return df

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def extract_features_for_backtest(df, bb_period):
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

def backtest_strategy(df, model_package, initial_capital, position_size_pct,
                     execution_mode, stop_loss_config, take_profit_config,
                     fee_rate, slippage, filters):
    upper_model = model_package['upper_model']
    lower_model = model_package['lower_model']
    prob_threshold = model_package['params']['probability_threshold']
    
    feature_cols = [
        'bb_position', 'dist_to_upper_pct', 'dist_to_lower_pct',
        'bb_width_pct', 'bb_width_ratio', 'volatility_ratio',
        'rsi', 'volume_ratio',
        'body_size_pct', 'upper_wick_pct', 'lower_wick_pct',
        'touch_count_5'
    ]
    
    capital = initial_capital
    position = None
    trades = []
    equity_curve = []
    filtered_signals = 0
    
    df_clean = df.dropna(subset=feature_cols)
    
    for i in range(len(df_clean)):
        row = df_clean.iloc[i]
        current_time = row.name
        
        features = row[feature_cols].values.reshape(1, -1)
        upper_prob = upper_model.predict_proba(features)[0, 1]
        lower_prob = lower_model.predict_proba(features)[0, 1]
        
        if position is not None:
            exit_signal, exit_price, exit_reason = check_exit(
                position, row, stop_loss_config, take_profit_config
            )
            
            if exit_signal:
                pnl, pnl_pct = close_position(
                    position, exit_price, fee_rate, slippage
                )
                capital += pnl
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'probability': position['probability']
                })
                
                position = None
        
        if position is None:
            if row['touch_upper'] and upper_prob >= prob_threshold:
                if apply_filters(row, 'SHORT', filters, filtered_signals):
                    entry_price = get_entry_price(
                        row, 'upper', execution_mode, df_clean, i
                    )
                    
                    if entry_price is not None:
                        adjusted_size = adjust_position_size(
                            position_size_pct, upper_prob, filters
                        )
                        position = open_position(
                            'SHORT', entry_price, current_time,
                            capital, adjusted_size, fee_rate, slippage,
                            upper_prob
                        )
                else:
                    filtered_signals += 1
            
            elif row['touch_lower'] and lower_prob >= prob_threshold:
                if apply_filters(row, 'LONG', filters, filtered_signals):
                    entry_price = get_entry_price(
                        row, 'lower', execution_mode, df_clean, i
                    )
                    
                    if entry_price is not None:
                        adjusted_size = adjust_position_size(
                            position_size_pct, lower_prob, filters
                        )
                        position = open_position(
                            'LONG', entry_price, current_time,
                            capital, adjusted_size, fee_rate, slippage,
                            lower_prob
                        )
                else:
                    filtered_signals += 1
        
        current_equity = capital
        if position is not None:
            unrealized_pnl = calculate_unrealized_pnl(position, row['close'])
            current_equity += unrealized_pnl
        
        equity_curve.append({
            'time': current_time,
            'equity': current_equity
        })
    
    if position is not None:
        final_row = df_clean.iloc[-1]
        pnl, pnl_pct = close_position(
            position, final_row['close'], fee_rate, slippage
        )
        capital += pnl
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': final_row.name,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': final_row['close'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': 'FORCED_CLOSE',
            'probability': position['probability']
        })
    
    return {
        'trades': pd.DataFrame(trades),
        'equity_curve': pd.DataFrame(equity_curve),
        'final_capital': capital,
        'initial_capital': initial_capital,
        'filtered_signals': filtered_signals
    }

def apply_filters(row, direction, filters, filtered_count):
    if filters.get('use_trend_filter'):
        if direction == 'LONG' and row['close'] < row['trend_ma']:
            return False
        if direction == 'SHORT' and row['close'] > row['trend_ma']:
            return False
    
    if filters.get('use_volatility_filter'):
        if row['bb_width_pct'] < filters['min_bb_width']:
            return False
    
    return True

def adjust_position_size(base_size, probability, filters):
    if filters.get('use_prob_boost'):
        threshold = filters['prob_boost_threshold']
        if probability >= threshold:
            return min(base_size * 1.5, 50)
    return base_size

def get_entry_price(row, direction, execution_mode, df, current_idx):
    if '保守' in execution_mode:
        if direction == 'upper':
            return row['bb_upper_prev']
        else:
            return row['bb_lower_prev']
    else:
        if current_idx + 1 < len(df):
            next_row = df.iloc[current_idx + 1]
            return next_row['open']
        else:
            return None

def open_position(direction, entry_price, entry_time, capital, position_size_pct,
                 fee_rate, slippage, probability):
    position_value = capital * (position_size_pct / 100)
    
    if direction == 'LONG':
        adjusted_price = entry_price * (1 + slippage / 100)
    else:
        adjusted_price = entry_price * (1 - slippage / 100)
    
    quantity = position_value / adjusted_price
    fee = position_value * (fee_rate / 100)
    
    return {
        'direction': direction,
        'entry_price': adjusted_price,
        'entry_time': entry_time,
        'quantity': quantity,
        'entry_fee': fee,
        'probability': probability
    }

def check_exit(position, row, stop_loss_config, take_profit_config):
    entry_price = position['entry_price']
    direction = position['direction']
    
    if stop_loss_config['mode'] == "ATR 倍數":
        stop_distance = row['atr'] * stop_loss_config['value']
        stop_loss_pct = (stop_distance / entry_price) * 100
    else:
        stop_loss_pct = stop_loss_config['value']
    
    if take_profit_config['mode'] == "ATR 倍數":
        tp_distance = row['atr'] * take_profit_config['value']
        take_profit_pct = (tp_distance / entry_price) * 100
    else:
        take_profit_pct = take_profit_config['value']
    
    if direction == 'LONG':
        if row['low'] <= entry_price * (1 - stop_loss_pct / 100):
            return True, entry_price * (1 - stop_loss_pct / 100), 'STOP_LOSS'
        elif row['high'] >= entry_price * (1 + take_profit_pct / 100):
            return True, entry_price * (1 + take_profit_pct / 100), 'TAKE_PROFIT'
    else:
        if row['high'] >= entry_price * (1 + stop_loss_pct / 100):
            return True, entry_price * (1 + stop_loss_pct / 100), 'STOP_LOSS'
        elif row['low'] <= entry_price * (1 - take_profit_pct / 100):
            return True, entry_price * (1 - take_profit_pct / 100), 'TAKE_PROFIT'
    
    return False, None, None

def close_position(position, exit_price, fee_rate, slippage):
    entry_price = position['entry_price']
    quantity = position['quantity']
    direction = position['direction']
    
    if direction == 'LONG':
        adjusted_exit = exit_price * (1 - slippage / 100)
    else:
        adjusted_exit = exit_price * (1 + slippage / 100)
    
    if direction == 'LONG':
        pnl = quantity * (adjusted_exit - entry_price)
    else:
        pnl = quantity * (entry_price - adjusted_exit)
    
    exit_fee = quantity * adjusted_exit * (fee_rate / 100)
    pnl -= (position['entry_fee'] + exit_fee)
    
    pnl_pct = (pnl / (quantity * entry_price)) * 100
    
    return pnl, pnl_pct

def calculate_unrealized_pnl(position, current_price):
    entry_price = position['entry_price']
    quantity = position['quantity']
    direction = position['direction']
    
    if direction == 'LONG':
        return quantity * (current_price - entry_price)
    else:
        return quantity * (entry_price - current_price)

def display_backtest_results(results, df):
    st.success("回測完成!")
    
    trades_df = results['trades']
    equity_df = results['equity_curve']
    
    st.subheader("總體統計")
    col1, col2, col3, col4 = st.columns(4)
    
    total_return = results['final_capital'] - results['initial_capital']
    return_pct = (total_return / results['initial_capital']) * 100
    
    with col1:
        st.metric("初始資金", f"${results['initial_capital']:.2f}")
    with col2:
        st.metric("最終資金", f"${results['final_capital']:.2f}")
    with col3:
        st.metric("總損益", f"${total_return:.2f}")
    with col4:
        st.metric("報酬率", f"{return_pct:.2f}%")
    
    if len(trades_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        win_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = (len(win_trades) / len(trades_df)) * 100
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
        
        with col1:
            st.metric("交易次數", len(trades_df))
        with col2:
            st.metric("勝率", f"{win_rate:.1f}%")
        with col3:
            st.metric("平均獲利", f"${avg_win:.2f}")
        with col4:
            st.metric("平均虧損", f"${avg_loss:.2f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("過濾信號數", results.get('filtered_signals', 0))
        with col2:
            if len(win_trades) > 0 and avg_loss != 0:
                profit_factor = abs(avg_win * len(win_trades) / (avg_loss * len(trades_df[trades_df['pnl'] < 0])))
                st.metric("盈虧比", f"{profit_factor:.2f}")
        with col3:
            max_dd = calculate_max_drawdown(equity_df['equity'])
            st.metric("最大回撤", f"{max_dd:.2f}%")
        
        st.subheader("權益曲線")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df['time'],
            y=equity_df['equity'],
            mode='lines',
            name='權益'
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("交易記錄")
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.warning("無交易記錄")

def calculate_max_drawdown(equity_series):
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax * 100
    return drawdown.min()