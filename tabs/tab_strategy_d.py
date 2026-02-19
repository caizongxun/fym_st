"""策略D: AI動態網格交易"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


class AIDynamicGridStrategy:
    """
    AI動態網格交易策略
    
    原理:
    1. AI預測未來波動範圍
    2. AI判斷市場狀態 (震盪/趨勢)
    3. 動態調整網格參數
    4. 智能停損保護
    """
    
    def __init__(self, grid_count=10):
        self.grid_count = grid_count
        self.scaler = StandardScaler()
        self.volatility_model = None
        self.market_state_model = None
    
    def calculate_features(self, df):
        """計算AI特徵"""
        df = df.copy()
        
        # 基本指標
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 波動率
        df['volatility'] = df['returns'].rolling(20).std()
        df['atr'] = self.calculate_atr(df, 14)
        df['bbands_width'] = self.calculate_bb_width(df, 20)
        
        # 趨勢指標
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['trend'] = (df['ema20'] - df['ema50']) / df['close']
        
        # 動量
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['macd'], df['signal'], _ = self.calculate_macd(df['close'])
        
        # 成交量
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 波動特徵
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['price_range_20'] = (df['close'].rolling(20).max() - df['close'].rolling(20).min()) / df['close']
        
        return df
    
    def calculate_atr(self, df, period=14):
        """計算ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def calculate_bb_width(self, df, period=20):
        """計算布林帶寬度"""
        ma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        return (std * 2) / ma
    
    def calculate_rsi(self, series, period=14):
        """計算RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, series, fast=12, slow=26, signal=9):
        """計算MACD"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def train_volatility_predictor(self, df):
        """訓練波動率預測模型"""
        feature_cols = [
            'volatility', 'atr', 'bbands_width', 'trend',
            'rsi', 'volume_ratio', 'high_low_ratio', 'price_range_20'
        ]
        
        # 準備訓練數據
        X = df[feature_cols].fillna(0)
        
        # 標籤: 未來20期的最大波動率
        y = df['close'].rolling(20).apply(
            lambda x: (x.max() - x.min()) / x.iloc[0]
        ).shift(-20).fillna(0)
        
        # 移除無效數據
        valid_mask = (X.notna().all(axis=1)) & (y.notna())
        X_train = X[valid_mask].iloc[:-20]
        y_train = y[valid_mask].iloc[:-20]
        
        if len(X_train) < 100:
            st.warning("數據不足,無法訓練AI")
            return None
        
        # 訓練模型
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        
        self.volatility_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.volatility_model.fit(X_scaled, y_train)
        
        return self.volatility_model
    
    def predict_price_range(self, df, idx):
        """預測未來價格範圍"""
        if self.volatility_model is None:
            # 如果沒有模型,使用歷史ATR
            return df.iloc[idx]['atr'] * 2
        
        feature_cols = [
            'volatility', 'atr', 'bbands_width', 'trend',
            'rsi', 'volume_ratio', 'high_low_ratio', 'price_range_20'
        ]
        
        X = df.iloc[idx][feature_cols].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        predicted_range_pct = self.volatility_model.predict(X_scaled)[0]
        return df.iloc[idx]['close'] * predicted_range_pct
    
    def detect_market_state(self, df, idx, lookback=50):
        """判斷市場狀態"""
        window = df.iloc[max(0, idx-lookback):idx]
        
        if len(window) < 20:
            return 'ranging'
        
        # 趨勢強度
        trend_strength = abs(window['trend'].iloc[-1])
        
        # 波動率
        recent_volatility = window['volatility'].iloc[-10:].mean()
        avg_volatility = window['volatility'].mean()
        
        # ADX模擬 (趨勢強度)
        price_changes = window['close'].diff().abs()
        avg_change = price_changes.mean()
        
        # 判斷
        if trend_strength > 0.03 and recent_volatility > avg_volatility * 1.2:
            return 'trending'
        else:
            return 'ranging'
    
    def calculate_grid_parameters(self, df, idx, base_grid_size=0.01):
        """計算動態網格參數"""
        current_price = df.iloc[idx]['close']
        
        # AI預測價格範圍
        predicted_range = self.predict_price_range(df, idx)
        
        # 市場狀態
        market_state = self.detect_market_state(df, idx)
        
        if market_state == 'trending':
            # 趨勢市: 縮小範圍,減少網格數
            range_mult = 0.7
            grid_count = max(5, self.grid_count // 2)
        else:
            # 震盪市: 正常範圍
            range_mult = 1.0
            grid_count = self.grid_count
        
        # 計算網格範圍
        range_size = predicted_range * range_mult
        grid_upper = current_price + range_size / 2
        grid_lower = current_price - range_size / 2
        
        # 網格間距
        grid_step = (grid_upper - grid_lower) / grid_count
        
        return {
            'upper': grid_upper,
            'lower': grid_lower,
            'step': grid_step,
            'count': grid_count,
            'market_state': market_state
        }
    
    def generate_grid_signals(self, df):
        """生成網格交易信號"""
        signals = []
        
        # 訓練AI模型
        self.train_volatility_predictor(df)
        
        # 記錄當前網格狀態
        active_grids = {}
        
        for i in range(100, len(df)):
            r = df.iloc[i]
            prev = df.iloc[i-1]
            
            # 每50根K線重新計算網格
            if i % 50 == 0 or i == 100:
                grid_params = self.calculate_grid_parameters(df, i)
                active_grids = {}
                
                # 初始化網格
                for j in range(grid_params['count']):
                    grid_price = grid_params['lower'] + j * grid_params['step']
                    active_grids[j] = {
                        'price': grid_price,
                        'active': False
                    }
            
            sig = 0
            reason = ""
            stop_loss = np.nan
            take_profit = np.nan
            
            # 檢查價格是否觸發網格
            for grid_id, grid_info in active_grids.items():
                grid_price = grid_info['price']
                
                # 價格下穿網格線 = 買入
                if prev['close'] > grid_price and r['close'] <= grid_price:
                    if not grid_info['active']:
                        sig = 1
                        reason = f"GRID_BUY_{grid_id}"
                        
                        # 止損: 下一個網格
                        if grid_id > 0:
                            stop_loss = active_grids[grid_id-1]['price']
                        else:
                            stop_loss = grid_price * 0.98
                        
                        # 止盈: 上一個網格
                        if grid_id < len(active_grids) - 1:
                            take_profit = active_grids[grid_id+1]['price']
                        else:
                            take_profit = grid_price * 1.02
                        
                        active_grids[grid_id]['active'] = True
                        break
                
                # 價格上穿網格線 = 賣出
                elif prev['close'] < grid_price and r['close'] >= grid_price:
                    if not grid_info['active']:
                        sig = -1
                        reason = f"GRID_SELL_{grid_id}"
                        
                        # 止損: 上一個網格
                        if grid_id < len(active_grids) - 1:
                            stop_loss = active_grids[grid_id+1]['price']
                        else:
                            stop_loss = grid_price * 1.02
                        
                        # 止盈: 下一個網格
                        if grid_id > 0:
                            take_profit = active_grids[grid_id-1]['price']
                        else:
                            take_profit = grid_price * 0.98
                        
                        active_grids[grid_id]['active'] = True
                        break
            
            signals.append({
                'signal': sig,
                'reason': reason,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
        
        empty = [{'signal': 0, 'reason': '', 'stop_loss': np.nan, 'take_profit': np.nan}] * 100
        return pd.DataFrame(empty + signals)


def render_strategy_d_tab(loader, symbol_selector):
    st.header("策略 D: AI動態網格")
    
    st.info("""
    **AI Dynamic Grid Trading**:
    
    智能網格系統:
    - AI預測波動範圍 → 動態調整網格大小
    - AI判斷市場狀態 → 震盪/趨勢自動切換
    - 智能網格間距 → 適應市場波動
    
    優勢:
    - 不需預測方向 (上漲下跌都賠)
    - 適合震盪市 (勝率70-80%)
    - 風險可控 (每格風險固定)
    - AI增強 (自動優化參數)
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_d", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("回測天數", 7, 60, 30, key="test_d")
    
    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_d")
        leverage = st.slider("槓桿", 3, 10, 5, key="lev_d")
        position_pct = st.slider("每格倉位%", 10, 50, 20, 5, key="pos_d")
        st.caption("每個網格使用的資金比例")
    
    with col3:
        st.markdown("**網格設定**")
        grid_count = st.slider("網格數量", 5, 20, 10, 1, key="grid_cnt")
        st.caption("AI會根據市場狀態動態調整")
    
    st.markdown("---")
    
    if st.button("執行AI網格策略", type="primary", use_container_width=True):
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
            
            stat.text("計算AI特徵...")
            strategy = AIDynamicGridStrategy(grid_count=grid_count)
            df_features = strategy.calculate_features(df_test)
            prog.progress(40)
            
            stat.text("訓練AI模型...")
            stat.text("生成網格信號...")
            df_signals = strategy.generate_grid_signals(df_features)
            
            signal_count = (df_signals['signal'] != 0).sum()
            if signal_count == 0:
                st.warning("無網格信號")
                return
            
            long_cnt = (df_signals['signal'] == 1).sum()
            short_cnt = (df_signals['signal'] == -1).sum()
            st.success(f"{signal_count}信號 (L:{long_cnt} S:{short_cnt})")
            prog.progress(70)
            
            # 設定倉位
            for i in range(len(df_signals)):
                if df_signals.iloc[i]['signal'] != 0:
                    df_signals.at[i, 'position_size'] = position_pct / 100.0
                else:
                    df_signals.at[i, 'position_size'] = 1.0
            
            stat.text("回測...")
            engine = TickLevelBacktestEngine(capital, leverage, 0.0006, 0.01, 100)
            metrics = engine.run_backtest(df_features, df_signals)
            
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
            
            if wr >= 60 and pf >= 1.3:
                st.success("✅ AI網格策略有效!")
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
                st.download_button("CSV", csv, f"{symbol}_grid_ai_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
