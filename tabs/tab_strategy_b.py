"""策略B: SSL Hybrid + AI Filter"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


class SSLHybridStrategy:
    """
    SSL Hybrid + AI Filter
    
    原理:
    1. SSL Hybrid生成原始信號
    2. AI模型過濾假信號
    3. 只保留高品質信號
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
    
    def hma(self, series, period):
        """計算Hull Moving Average"""
        half_length = int(period / 2)
        sqrt_length = int(np.sqrt(period))
        
        wmaf = series.rolling(half_length).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        wmas = series.rolling(period).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        
        raw_hma = 2 * wmaf - wmas
        hma = raw_hma.rolling(sqrt_length).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        
        return hma
    
    def calculate_ssl_baseline(self, df, length=60, mult=0.2):
        """計算SSL Baseline"""
        df = df.copy()
        
        # Baseline (HMA)
        df['baseline'] = self.hma(df['close'], length)
        
        # Channel (ATR-based)
        df['tr'] = np.maximum(df['high'] - df['low'], 
                              np.maximum(abs(df['high'] - df['close'].shift(1)),
                                        abs(df['low'] - df['close'].shift(1))))
        df['range_ma'] = df['tr'].ewm(span=length).mean()
        df['upper_channel'] = df['baseline'] + df['range_ma'] * mult
        df['lower_channel'] = df['baseline'] - df['range_ma'] * mult
        
        return df
    
    def calculate_ssl1(self, df, length=60):
        """計算SSL1 - 主信號"""
        df = df.copy()
        
        # HMA of high and low
        df['ema_high'] = self.hma(df['high'], length)
        df['ema_low'] = self.hma(df['low'], length)
        
        # SSL Direction
        df['ssl_direction'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['close'] > df.iloc[i]['ema_high']:
                df.iloc[i, df.columns.get_loc('ssl_direction')] = 1
            elif df.iloc[i]['close'] < df.iloc[i]['ema_low']:
                df.iloc[i, df.columns.get_loc('ssl_direction')] = -1
            else:
                df.iloc[i, df.columns.get_loc('ssl_direction')] = df.iloc[i-1]['ssl_direction']
        
        df['ssl_down'] = np.where(df['ssl_direction'] < 0, df['ema_high'], df['ema_low'])
        
        return df
    
    def calculate_ssl2(self, df, length=5):
        """計算SSL2 - 確認信號"""
        df = df.copy()
        
        # Fast JMA (simplified as EMA)
        df['ma_high'] = df['high'].ewm(span=length).mean()
        df['ma_low'] = df['low'].ewm(span=length).mean()
        
        # SSL2 Direction
        df['ssl2_direction'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['close'] > df.iloc[i]['ma_high']:
                df.iloc[i, df.columns.get_loc('ssl2_direction')] = 1
            elif df.iloc[i]['close'] < df.iloc[i]['ma_low']:
                df.iloc[i, df.columns.get_loc('ssl2_direction')] = -1
            else:
                df.iloc[i, df.columns.get_loc('ssl2_direction')] = df.iloc[i-1]['ssl2_direction']
        
        df['ssl2_down'] = np.where(df['ssl2_direction'] < 0, df['ma_high'], df['ma_low'])
        
        return df
    
    def calculate_ssl_exit(self, df, length=15):
        """計算SSL Exit"""
        df = df.copy()
        
        df['exit_high'] = self.hma(df['high'], length)
        df['exit_low'] = self.hma(df['low'], length)
        
        df['exit_direction'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['close'] > df.iloc[i]['exit_high']:
                df.iloc[i, df.columns.get_loc('exit_direction')] = 1
            elif df.iloc[i]['close'] < df.iloc[i]['exit_low']:
                df.iloc[i, df.columns.get_loc('exit_direction')] = -1
            else:
                df.iloc[i, df.columns.get_loc('exit_direction')] = df.iloc[i-1]['exit_direction']
        
        df['ssl_exit'] = np.where(df['exit_direction'] < 0, df['exit_high'], df['exit_low'])
        
        return df
    
    def calculate_atr(self, df, period=14):
        """計算ATR"""
        df = df.copy()
        df['atr'] = df['tr'].rolling(period).mean()
        return df
    
    def calculate_ssl_features(self, df):
        """計算SSL所有特徵"""
        df = self.calculate_ssl_baseline(df)
        df = self.calculate_ssl1(df)
        df = self.calculate_ssl2(df)
        df = self.calculate_ssl_exit(df)
        df = self.calculate_atr(df)
        
        # 額外特徵
        df['distance_from_baseline'] = (df['close'] - df['baseline']) / df['atr']
        df['atr_percentile'] = df['atr'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)
        df['candle_size'] = abs(df['close'] - df['open']) / df['atr']
        df['in_channel'] = ((df['close'] > df['lower_channel']) & (df['close'] < df['upper_channel'])).astype(int)
        
        # SSL2 Continuation
        df['ssl2_buy_cont'] = ((df['close'] > df['baseline']) & (df['close'] > df['ssl2_down'])).astype(int)
        df['ssl2_sell_cont'] = ((df['close'] < df['baseline']) & (df['close'] < df['ssl2_down'])).astype(int)
        
        return df
    
    def generate_ssl_signals(self, df):
        """生成SSL原始信號"""
        signals = []
        
        for i in range(100, len(df)):
            r = df.iloc[i]
            prev = df.iloc[i-1]
            
            sig = 0
            reason = ""
            
            # Exit Signal (highest priority)
            if prev['close'] > prev['ssl_exit'] and r['close'] < r['ssl_exit']:
                sig = -2  # Force exit long
                reason = "EXIT_LONG"
            elif prev['close'] < prev['ssl_exit'] and r['close'] > r['ssl_exit']:
                sig = 2  # Force exit short
                reason = "EXIT_SHORT"
            
            # SSL2 Continuation Signals
            elif r['ssl2_buy_cont'] == 1 and prev['ssl2_buy_cont'] == 0:
                if r['distance_from_baseline'] < 2:  # Not too far
                    sig = 1
                    reason = "SSL2_BUY"
            
            elif r['ssl2_sell_cont'] == 1 and prev['ssl2_sell_cont'] == 0:
                if r['distance_from_baseline'] > -2:
                    sig = -1
                    reason = "SSL2_SELL"
            
            # Baseline Breakout
            elif prev['close'] < prev['upper_channel'] and r['close'] > r['upper_channel']:
                sig = 1
                reason = "BASELINE_BULL"
            
            elif prev['close'] > prev['lower_channel'] and r['close'] < r['lower_channel']:
                sig = -1
                reason = "BASELINE_BEAR"
            
            signals.append({
                'signal': sig,
                'reason': reason
            })
        
        empty = [{'signal': 0, 'reason': ''}] * 100
        return pd.DataFrame(empty + signals)
    
    def prepare_ml_features(self, df):
        """準備AI特徵"""
        feature_cols = [
            'distance_from_baseline',
            'atr_percentile',
            'candle_size',
            'in_channel',
            'ssl_direction',
            'ssl2_direction',
            'exit_direction',
            'ssl2_buy_cont',
            'ssl2_sell_cont'
        ]
        
        X = df[feature_cols].fillna(0)
        return X
    
    def train_filter_model(self, df, df_signals, lookahead=10, threshold=0.005):
        """訓練AI過濾模型"""
        # 重置索引以確保對齊
        df = df.reset_index(drop=True)
        df_signals = df_signals.reset_index(drop=True)
        
        # 準備特徵
        X = self.prepare_ml_features(df)
        
        # 生成標籤: 未來N期是否獲利
        y = []
        for i in range(len(df)):
            if i + lookahead >= len(df):
                y.append(0)
                continue
            
            sig = df_signals.iloc[i]['signal']
            if sig == 0:
                y.append(0)
                continue
            
            future_return = (df.iloc[i+lookahead]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']
            
            if sig > 0:  # Long signal
                y.append(1 if future_return > threshold else 0)
            else:  # Short signal
                y.append(1 if future_return < -threshold else 0)
        
        y = pd.Series(y)
        
        # 只訓練有信號的樣本
        signal_mask = df_signals['signal'] != 0
        X_train = X[signal_mask].reset_index(drop=True)
        y_train = y[signal_mask].reset_index(drop=True)
        
        if len(X_train) < 50:
            st.warning("信號太少,無法訓練AI")
            return None
        
        # 訓練
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_scaled, y_train)
        
        return self.model
    
    def filter_signals(self, df, df_signals, confidence_threshold=0.6):
        """用AI過濾信號"""
        if self.model is None:
            return df_signals
        
        df = df.reset_index(drop=True)
        df_signals = df_signals.reset_index(drop=True)
        
        X = self.prepare_ml_features(df)
        X_scaled = self.scaler.transform(X)
        
        # 預測信心度
        proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # 過濾
        filtered_signals = df_signals.copy()
        for i in range(len(filtered_signals)):
            if filtered_signals.iloc[i]['signal'] != 0:
                if proba[i] < confidence_threshold:
                    filtered_signals.iloc[i, filtered_signals.columns.get_loc('signal')] = 0
                    filtered_signals.iloc[i, filtered_signals.columns.get_loc('reason')] += "_FILTERED"
        
        return filtered_signals


def render_strategy_b_tab(loader, symbol_selector):
    st.header("策略 B: SSL Hybrid + AI")
    
    st.info("""
    **SSL Hybrid + AI Filter**:
    
    SSL Hybrid系統:
    - Baseline: 趨勢方向 (HMA60 + Channel)
    - SSL1: 主信號 (HMA交叉)
    - SSL2: 確認信號 (快速JMA5)
    - Exit: 出場信號 (HMA15)
    
    AI過濾:
    - 訓練XGBoost分類器
    - 過濾假信號 (震盪/假突破)
    - 只保留高信心度信號
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_b", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("回測天數", 7, 60, 30, key="test_b")
    
    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_b")
        leverage = st.slider("槓桿", 3, 10, 5, key="lev_b")
        position_pct = st.slider("倉位%", 40, 100, 70, 10, key="pos_b")
    
    with col3:
        st.markdown("**AI設定**")
        use_ai_filter = st.checkbox("AI過濾", value=True, key="ai_filter")
        confidence_threshold = st.slider("AI信心度", 0.5, 0.9, 0.65, 0.05, key="ai_conf")
        
        st.markdown("**風控**")
        stop_atr_mult = st.slider("止損ATR倍數", 1.0, 3.0, 1.5, 0.5, key="sl_b")
        tp_atr_mult = st.slider("止盈ATR倍數", 2.0, 5.0, 3.0, 0.5, key="tp_b")
    
    st.markdown("---")
    
    if st.button("執行SSL + AI策略", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("載入...")
            prog.progress(10)
            
            if isinstance(loader, BinanceDataLoader):
                end = datetime.now()
                start = end - timedelta(days=test_days + 30)
                df_test = loader.load_historical_data(symbol, '15m', start, end)
            else:
                df_test = loader.load_klines(symbol, '15m')
                df_test = df_test.tail((test_days + 30) * 96)
            
            df_test = df_test.reset_index(drop=True)
            
            st.success(f"{len(df_test)}根")
            prog.progress(20)
            
            stat.text("計算SSL...")
            strategy = SSLHybridStrategy()
            df_ssl = strategy.calculate_ssl_features(df_test)
            prog.progress(40)
            
            stat.text("生成信號...")
            df_signals = strategy.generate_ssl_signals(df_ssl)
            raw_count = (df_signals['signal'] != 0).sum()
            st.info(f"原始信號: {raw_count}")
            prog.progress(60)
            
            if use_ai_filter:
                stat.text("訓練AI...")
                model = strategy.train_filter_model(df_ssl, df_signals)
                
                if model is not None:
                    stat.text("過濾信號...")
                    df_signals = strategy.filter_signals(df_ssl, df_signals, confidence_threshold)
                    filtered_count = (df_signals['signal'] != 0).sum()
                    st.success(f"AI過濾後: {filtered_count} (過濾掉{raw_count-filtered_count}個)")
            
            prog.progress(70)
            
            # 設定止損止盈
            for i in range(len(df_signals)):
                if df_signals.iloc[i]['signal'] != 0:
                    r = df_ssl.iloc[i]
                    atr = r['atr']
                    
                    if df_signals.iloc[i]['signal'] > 0:
                        df_signals.at[i, 'stop_loss'] = r['close'] - (atr * stop_atr_mult)
                        df_signals.at[i, 'take_profit'] = r['close'] + (atr * tp_atr_mult)
                    else:
                        df_signals.at[i, 'stop_loss'] = r['close'] + (atr * stop_atr_mult)
                        df_signals.at[i, 'take_profit'] = r['close'] - (atr * tp_atr_mult)
                    
                    df_signals.at[i, 'position_size'] = position_pct / 100.0
                else:
                    df_signals.at[i, 'stop_loss'] = np.nan
                    df_signals.at[i, 'take_profit'] = np.nan
                    df_signals.at[i, 'position_size'] = 1.0
            
            final_count = (df_signals['signal'] != 0).sum()
            if final_count == 0:
                st.warning("無有效信號")
                return
            
            stat.text("回測...")
            engine = TickLevelBacktestEngine(capital, leverage, 0.0006, 0.01, 100)
            metrics = engine.run_backtest(df_ssl, df_signals)
            
            prog.progress(100)
            stat.text("完成")
            
            st.markdown("---")
            st.subheader("結果")
            
            c1, c2, c3, c4 = st.columns(4)
            
            pnl = metrics['final_equity'] - capital
            c1.metric("權益", f"${metrics['final_equity']:,.0f}", f"{pnl:+,.0f}")
            c1.metric("交易", metrics['total_trades'])
            
            ret = metrics['total_return_pct']
            monthly = ret * 30 / test_days
            c2.metric("總報酬", f"{ret:.1f}%")
            c2.metric("月化", f"{monthly:.1f}%")
            
            wr = metrics['win_rate']
            c3.metric("勝率", f"{wr:.1f}%")
            pf = metrics['profit_factor']
            c3.metric("盈虧比", f"{pf:.2f}")
            
            dd = metrics['max_drawdown_pct']
            c4.metric("回撤", f"{dd:.1f}%")
            c4.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")
            
            st.markdown("---")
            
            if wr >= 50 and pf >= 1.5:
                st.success("✅ SSL + AI策略有效!")
                st.balloons()
            elif ret > 0:
                st.info("⚠️ 有獲利")
            else:
                st.warning("❌ 調整參數")
            
            st.markdown("---")
            st.subheader("權益")
            fig = engine.plot_equity_curve()
            st.plotly_chart(fig, use_container_width=True)
            
            trades = engine.get_trades_dataframe()
            if not trades.empty:
                st.markdown("---")
                st.subheader("交易")
                
                wins = trades[trades['pnl_usdt'] > 0]
                losses = trades[trades['pnl_usdt'] < 0]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("贏", len(wins))
                c2.metric("輸", len(losses))
                if len(wins) > 0:
                    c3.metric("平均贏", f"${wins['pnl_usdt'].mean():.2f}")
                if len(losses) > 0:
                    c4.metric("平均輸", f"${losses['pnl_usdt'].mean():.2f}")
                
                st.dataframe(trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(30), use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_ssl_ai_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
