"""策略A: 純ML預測 - 不依賴傳統指標"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


class PureMLStrategy:
    """純ML預測策略 - 預測未來N根K線的價格方向"""
    
    def __init__(self, lookback=20, forward_bars=3):
        self.lookback = lookback
        self.forward_bars = forward_bars
        self.model_long = None
        self.model_short = None
        self.scaler = StandardScaler()
    
    def create_features(self, df):
        """創建純價格特徵 - 不用任何指標"""
        df = df.copy()
        
        # 價格變化率特徵
        for i in [1, 2, 3, 5, 10, 15, 20]:
            df[f'ret_{i}'] = df['close'].pct_change(i)
            df[f'high_ret_{i}'] = df['high'].pct_change(i)
            df[f'low_ret_{i}'] = df['low'].pct_change(i)
        
        # 波動率特徵
        for i in [5, 10, 20]:
            df[f'volatility_{i}'] = df['close'].pct_change().rolling(i).std()
            df[f'range_{i}'] = (df['high'] - df['low']) / df['close']
        
        # 成交量特徵
        df['volume_ret_1'] = df['volume'].pct_change(1)
        df['volume_ret_5'] = df['volume'].pct_change(5)
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # K線形態
        df['body'] = (df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # 價格位置
        for i in [10, 20, 50]:
            df[f'price_position_{i}'] = (df['close'] - df['close'].rolling(i).min()) / (df['close'].rolling(i).max() - df['close'].rolling(i).min())
        
        return df
    
    def create_labels(self, df):
        """創建標籤 - 未來是否上漲/下跌超過閾值"""
        df = df.copy()
        
        # 未來最高/最低價
        df['future_high'] = df['high'].shift(-self.forward_bars).rolling(self.forward_bars).max()
        df['future_low'] = df['low'].shift(-self.forward_bars).rolling(self.forward_bars).max()
        
        # 做多機會: 未來能上漲1%+
        df['long_target'] = ((df['future_high'] - df['close']) / df['close'] > 0.01).astype(int)
        
        # 做空機會: 未來會下跌1%+
        df['short_target'] = ((df['close'] - df['future_low']) / df['close'] > 0.01).astype(int)
        
        return df
    
    def train(self, df):
        """訓練模型"""
        df = self.create_features(df)
        df = self.create_labels(df)
        df = df.dropna()
        
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'open_time', 'long_target', 'short_target', 'future_high', 'future_low']]
        
        X = df[feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        
        y_long = df['long_target']
        y_short = df['short_target']
        
        # 訓練做多模型
        self.model_long = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=50, random_state=42)
        self.model_long.fit(X_scaled, y_long)
        
        # 訓練做空模型
        self.model_short = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=50, random_state=42)
        self.model_short.fit(X_scaled, y_short)
        
        return {
            'long_samples': int(y_long.sum()),
            'short_samples': int(y_short.sum()),
            'total_samples': len(df),
            'feature_cols': feature_cols
        }
    
    def predict(self, df, idx):
        """預測單個時間點"""
        df_test = self.create_features(df.iloc[:idx+1])
        feature_cols = [c for c in df_test.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'open_time']]
        
        X = df_test[feature_cols].iloc[-1:]
        X_scaled = self.scaler.transform(X)
        
        long_proba = self.model_long.predict_proba(X_scaled)[0][1]
        short_proba = self.model_short.predict_proba(X_scaled)[0][1]
        
        return long_proba, short_proba


def render_strategy_a_tab(loader, symbol_selector):
    st.header("策略 A: 純ML預測")
    
    st.info("""
    **純機器學習策略**:
    
    不使用任何傳統指標(RSI/MACD/BB等)
    
    特徵:
    - 價格變化率(1-20期)
    - 波動率(5-20期)
    - 成交量變化
    - K線形態
    - 價格相對位置
    
    預測: 未來3根K線能否獲利1%+
    
    優勢: 適應各種市場環境
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_a", multi=False)
        symbol = symbol_list[0]
        train_days = st.slider("訓練天數", 60, 180, 120, key="train")
        test_days = st.slider("回測天數", 7, 60, 30, key="test")
    
    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap")
        leverage = st.slider("槓桿", 3, 10, 5, key="lev")
        confidence = st.slider("信心度", 0.4, 0.8, 0.55, 0.05, key="conf")
    
    with col3:
        st.markdown("**風控**")
        target_pct = st.slider("目標獲利%", 0.5, 3.0, 1.0, 0.1, key="target")
        stop_pct = st.slider("止損%", 0.5, 3.0, 1.5, 0.1, key="stop")
        position_pct = st.slider("倉位%", 40, 100, 70, 10, key="pos")
    
    st.markdown("---")
    
    if st.button("執行純ML策略", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("1/4: 載入...")
            prog.progress(10)
            
            if isinstance(loader, BinanceDataLoader):
                end = datetime.now()
                start = end - timedelta(days=train_days + test_days)
                df_all = loader.load_historical_data(symbol, '15m', start, end)
            else:
                df_all = loader.load_klines(symbol, '15m')
                df_all = df_all.tail((train_days + test_days) * 96)
            
            split = len(df_all) - test_days * 96
            df_train = df_all.iloc[:split].copy()
            df_test = df_all.iloc[split:].copy()
            
            st.success(f"{len(df_train)}+{len(df_test)}")
            prog.progress(30)
            
            stat.text("2/4: 訓練ML...")
            strategy = PureMLStrategy(lookback=20, forward_bars=3)
            stats = strategy.train(df_train)
            st.success(f"L:{stats['long_samples']} S:{stats['short_samples']}")
            prog.progress(60)
            
            stat.text("3/4: 生成信號...")
            
            signals = []
            
            for i in range(50, len(df_test)):
                lp, sp = strategy.predict(df_test, i)
                r = df_test.iloc[i]
                
                sig = 0
                sl = np.nan
                tp = np.nan
                
                # 做多
                if lp > confidence and lp > sp:
                    sig = 1
                    entry = r['close']
                    sl = entry * (1 - stop_pct / 100)
                    tp = entry * (1 + target_pct / 100)
                
                # 做空
                elif sp > confidence and sp > lp:
                    sig = -1
                    entry = r['close']
                    sl = entry * (1 + stop_pct / 100)
                    tp = entry * (1 - target_pct / 100)
                
                signals.append({
                    'signal': sig,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'position_size': position_pct / 100.0,
                    'long_proba': lp,
                    'short_proba': sp
                })
            
            signals = [{'signal': 0, 'stop_loss': np.nan, 'take_profit': np.nan, 'position_size': 1.0, 'long_proba': 0, 'short_proba': 0}] * 50 + signals
            df_sig = pd.DataFrame(signals)
            
            cnt = (df_sig['signal'] != 0).sum()
            
            if cnt == 0:
                st.warning("無信號 - 降低信心度到 0.50")
                return
            
            long_cnt = (df_sig['signal'] == 1).sum()
            short_cnt = (df_sig['signal'] == -1).sum()
            st.success(f"{cnt}信號 (L:{long_cnt} S:{short_cnt})")
            prog.progress(80)
            
            stat.text("4/4: 回測...")
            engine = TickLevelBacktestEngine(capital, leverage, 0.0006, 0.01, 100)
            metrics = engine.run_backtest(df_test, df_sig)
            
            prog.progress(100)
            stat.text("完成")
            
            # Results
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
            
            if wr >= 50 and pf >= 1.2:
                st.success("✅ 策略有效")
                st.balloons()
            elif ret > 0:
                st.info("⚠️ 有獲利但需優化")
            else:
                st.warning("❌ 需調整參數")
            
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
                c3.metric("平均贏", f"${wins['pnl_usdt'].mean():.2f}" if len(wins)>0 else "$0")
                c4.metric("平均輸", f"${losses['pnl_usdt'].mean():.2f}" if len(losses)>0 else "$0")
                
                st.dataframe(trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(30), use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_ml_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
