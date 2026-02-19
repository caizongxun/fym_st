"""策略A: SMC v2 - 改進版"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


class SMCStrategy:
    """
    SMC v2 - 改進版
    
    改進:
    1. 簡化進場条件 (1個就够)
    2. 動態ATR止損
    3. 更大止盈目標
    4. 只做趨勢內交易
    """
    
    def __init__(self, lookback=50):
        self.lookback = lookback
    
    def calculate_atr(self, df, period=14):
        """計算ATR"""
        df = df.copy()
        
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(period).mean()
        
        return df
    
    def identify_trend(self, df):
        """識別趨勢 - EMA"""
        df = df.copy()
        
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema100'] = df['close'].ewm(span=100).mean()
        
        # 趨勢方向
        df['trend'] = 0
        df.loc[(df['ema20'] > df['ema50']) & (df['ema50'] > df['ema100']), 'trend'] = 1  # 上升
        df.loc[(df['ema20'] < df['ema50']) & (df['ema50'] < df['ema100']), 'trend'] = -1  # 下降
        
        # 趨勢強度
        df['trend_strength'] = abs(df['ema20'] - df['ema50']) / df['close']
        
        return df
    
    def identify_order_blocks(self, df):
        """識別Order Block - 大量K線"""
        df = df.copy()
        
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['range'] = (df['high'] - df['low']) / df['close']
        df['range_ma'] = df['range'].rolling(20).mean()
        
        # 看多OB: 大量 + 大振幅 + 上漨
        df['bullish_ob'] = 0
        df.loc[
            (df['volume'] > df['vol_ma'] * 1.5) &
            (df['range'] > df['range_ma'] * 1.2) &
            (df['close'] > df['open']),
            'bullish_ob'
        ] = 1
        
        # 看空OB: 大量 + 大振幅 + 下跌
        df['bearish_ob'] = 0
        df.loc[
            (df['volume'] > df['vol_ma'] * 1.5) &
            (df['range'] > df['range_ma'] * 1.2) &
            (df['close'] < df['open']),
            'bearish_ob'
        ] = 1
        
        return df
    
    def identify_fvg(self, df):
        """識別FVG - 價格缺口"""
        df = df.copy()
        
        df['bullish_fvg'] = 0
        df['bearish_fvg'] = 0
        
        for i in range(2, len(df)):
            # 看多FVG: K1高 < K3低
            if df.iloc[i-2]['high'] < df.iloc[i]['low']:
                gap_size = (df.iloc[i]['low'] - df.iloc[i-2]['high']) / df.iloc[i]['close']
                if gap_size > 0.002:  # 至少0.2%的gap
                    df.iloc[i-1, df.columns.get_loc('bullish_fvg')] = 1
            
            # 看空FVG: K1低 > K3高
            if df.iloc[i-2]['low'] > df.iloc[i]['high']:
                gap_size = (df.iloc[i-2]['low'] - df.iloc[i]['high']) / df.iloc[i]['close']
                if gap_size > 0.002:
                    df.iloc[i-1, df.columns.get_loc('bearish_fvg')] = 1
        
        return df
    
    def calculate_support_resistance(self, df):
        """計算支撑阻力"""
        df = df.copy()
        
        # 近20期高低點
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        
        # 相對位置
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'] + 1e-10)
        
        return df
    
    def generate_signals(self, df, min_conditions=1):
        """生成SMC信號 - 簡化版"""
        df = self.calculate_atr(df)
        df = self.identify_trend(df)
        df = self.identify_order_blocks(df)
        df = self.identify_fvg(df)
        df = self.calculate_support_resistance(df)
        
        signals = []
        
        for i in range(100, len(df)):
            r = df.iloc[i]
            
            sig = 0
            reason = ""
            sl_distance = 0
            tp_distance = 0
            
            # 做多條件
            long_cond = []
            
            # 1. 上升趨勢
            if r['trend'] == 1:
                long_cond.append("UP")
            
            # 2. 近期OB
            if df.iloc[max(0, i-5):i]['bullish_ob'].sum() > 0:
                long_cond.append("OB")
            
            # 3. FVG
            if r['bullish_fvg'] == 1:
                long_cond.append("FVG")
            
            # 4. 低位
            if r['price_position'] < 0.5:
                long_cond.append("LOW")
            
            # 5. 價格 > EMA20 (回調結束)
            if r['close'] > r['ema20'] and df.iloc[i-1]['close'] < df.iloc[i-1]['ema20']:
                long_cond.append("CROSS")
            
            # 做空條件
            short_cond = []
            
            # 1. 下降趨勢
            if r['trend'] == -1:
                short_cond.append("DOWN")
            
            # 2. 近期OB
            if df.iloc[max(0, i-5):i]['bearish_ob'].sum() > 0:
                short_cond.append("OB")
            
            # 3. FVG
            if r['bearish_fvg'] == 1:
                short_cond.append("FVG")
            
            # 4. 高位
            if r['price_position'] > 0.5:
                short_cond.append("HIGH")
            
            # 5. 價格 < EMA20
            if r['close'] < r['ema20'] and df.iloc[i-1]['close'] > df.iloc[i-1]['ema20']:
                short_cond.append("CROSS")
            
            # 決定信號
            if len(long_cond) >= min_conditions and r['trend'] == 1:
                sig = 1
                reason = "+".join(long_cond)
                # 動態ATR止損
                sl_distance = r['atr'] * 1.5
                tp_distance = r['atr'] * 3.0
            
            elif len(short_cond) >= min_conditions and r['trend'] == -1:
                sig = -1
                reason = "+".join(short_cond)
                sl_distance = r['atr'] * 1.5
                tp_distance = r['atr'] * 3.0
            
            signals.append({
                'signal': sig,
                'reason': reason,
                'sl_distance': sl_distance,
                'tp_distance': tp_distance,
                'atr': r['atr']
            })
        
        empty = [{'signal': 0, 'reason': '', 'sl_distance': 0, 'tp_distance': 0, 'atr': 0}] * 100
        return pd.DataFrame(empty + signals)


def render_strategy_a_tab(loader, symbol_selector):
    st.header("策略 A: SMC v2")
    
    st.info("""
    **SMC v2 - 改進版**:
    
    改進:
    1. 簡化進場 - 只需滿足N個條件
    2. 動態ATR止損 - 根據波動調整
    3. 更大止盈 - 3:1風報比
    4. 只做趨勢 - 上升做多/下降做空
    
    條件:
    - UP/DOWN: 趨勢方向 (EMA20>50>100)
    - OB: Order Block (大量K線)
    - FVG: Fair Value Gap (價格缺口)
    - LOW/HIGH: 相對位置
    - CROSS: 交叉確認
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_a", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("回測天數", 7, 60, 30, key="test")
    
    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap")
        leverage = st.slider("槓桿", 3, 10, 5, key="lev")
        position_pct = st.slider("倉位%", 40, 100, 70, 10, key="pos")
    
    with col3:
        st.markdown("**SMC設定**")
        min_conditions = st.slider("最少條件", 1, 3, 2, key="min_cond")
        st.caption("越低信號越多但品質下降")
        
        atr_sl_mult = st.slider("止損ATR倍數", 1.0, 3.0, 1.5, 0.5, key="sl_mult")
        atr_tp_mult = st.slider("止盈ATR倍數", 2.0, 5.0, 3.0, 0.5, key="tp_mult")
    
    st.markdown("---")
    
    if st.button("執行SMC v2", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("載入...")
            prog.progress(20)
            
            if isinstance(loader, BinanceDataLoader):
                end = datetime.now()
                start = end - timedelta(days=test_days + 20)
                df_test = loader.load_historical_data(symbol, '15m', start, end)
            else:
                df_test = loader.load_klines(symbol, '15m')
                df_test = df_test.tail((test_days + 20) * 96)
            
            st.success(f"{len(df_test)}根")
            prog.progress(40)
            
            stat.text("分析SMC...")
            strategy = SMCStrategy()
            df_sig = strategy.generate_signals(df_test, min_conditions)
            
            # 設定止損止盈
            for i in range(len(df_sig)):
                if df_sig.iloc[i]['signal'] != 0:
                    r = df_test.iloc[i]
                    atr = df_sig.iloc[i]['atr']
                    
                    if df_sig.iloc[i]['signal'] == 1:
                        df_sig.at[i, 'stop_loss'] = r['close'] - (atr * atr_sl_mult)
                        df_sig.at[i, 'take_profit'] = r['close'] + (atr * atr_tp_mult)
                    else:
                        df_sig.at[i, 'stop_loss'] = r['close'] + (atr * atr_sl_mult)
                        df_sig.at[i, 'take_profit'] = r['close'] - (atr * atr_tp_mult)
                    
                    df_sig.at[i, 'position_size'] = position_pct / 100.0
                else:
                    df_sig.at[i, 'stop_loss'] = np.nan
                    df_sig.at[i, 'take_profit'] = np.nan
                    df_sig.at[i, 'position_size'] = 1.0
            
            cnt = (df_sig['signal'] != 0).sum()
            
            if cnt == 0:
                st.warning("無信號 - 降低最少條件")
                return
            
            long_cnt = (df_sig['signal'] == 1).sum()
            short_cnt = (df_sig['signal'] == -1).sum()
            st.success(f"{cnt}信號 (L:{long_cnt} S:{short_cnt})")
            prog.progress(70)
            
            stat.text("回測...")
            engine = TickLevelBacktestEngine(capital, leverage, 0.0006, 0.01, 100)
            metrics = engine.run_backtest(df_test, df_sig)
            
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
            
            if wr >= 45 and pf >= 1.5:
                st.success("✅ SMC v2有效!")
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
                st.download_button("CSV", csv, f"{symbol}_smc_v2_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
