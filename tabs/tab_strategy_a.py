"""策略A: SMC (Smart Money Concepts) - 機構交易逻輯"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


class SMCStrategy:
    """
    SMC策略 - Smart Money Concepts
    
    核心概念:
    1. Order Block (OB) - 機構訂單區
    2. Fair Value Gap (FVG) - 公允價值缺口
    3. Break of Structure (BOS) - 市場結構破壞
    4. Change of Character (CHoCH) - 趨勢轉換
    
    逻輯:
    - 機構在OB區域建倉
    - FVG是回調進場點
    - BOS確認趨勢繼續
    """
    
    def __init__(self, lookback=50):
        self.lookback = lookback
    
    def identify_market_structure(self, df):
        """識別市場結構 - 高點/低點"""
        df = df.copy()
        
        # 高點 (Swing High)
        df['swing_high'] = False
        for i in range(5, len(df)-5):
            if df.iloc[i]['high'] == df.iloc[i-5:i+6]['high'].max():
                df.at[i, 'swing_high'] = True
        
        # 低點 (Swing Low)
        df['swing_low'] = False
        for i in range(5, len(df)-5):
            if df.iloc[i]['low'] == df.iloc[i-5:i+6]['low'].min():
                df.at[i, 'swing_low'] = True
        
        # 趨勢判斷: Higher Highs & Higher Lows = 上升
        df['trend'] = 0  # 0=中性, 1=上升, -1=下降
        
        highs = df[df['swing_high']]['high'].values
        lows = df[df['swing_low']]['low'].values
        
        if len(highs) >= 2 and len(lows) >= 2:
            # 簡化: 比較最近倴5個高低點
            recent_highs = highs[-5:] if len(highs) >= 5 else highs
            recent_lows = lows[-5:] if len(lows) >= 5 else lows
            
            if len(recent_highs) >= 2:
                hh = recent_highs[-1] > recent_highs[-2]
            else:
                hh = False
            
            if len(recent_lows) >= 2:
                hl = recent_lows[-1] > recent_lows[-2]
            else:
                hl = False
            
            if hh and hl:
                df['trend'] = 1  # 上升
            elif not hh and not hl:
                df['trend'] = -1  # 下降
        
        return df
    
    def identify_order_blocks(self, df):
        """識別Order Block - 大量後的K線"""
        df = df.copy()
        
        df['bullish_ob'] = False
        df['bearish_ob'] = False
        
        # 看空OB: 大量下跌前K線
        for i in range(10, len(df)-1):
            # 大量下跌
            big_drop = (df.iloc[i+1]['close'] < df.iloc[i]['close'] * 0.995) and \
                       (df.iloc[i+1]['volume'] > df.iloc[i-10:i+1]['volume'].mean() * 1.5)
            
            if big_drop:
                df.at[i, 'bearish_ob'] = True
        
        # 看多OB: 大量上漨前K線
        for i in range(10, len(df)-1):
            # 大量上漨
            big_rise = (df.iloc[i+1]['close'] > df.iloc[i]['close'] * 1.005) and \
                      (df.iloc[i+1]['volume'] > df.iloc[i-10:i+1]['volume'].mean() * 1.5)
            
            if big_rise:
                df.at[i, 'bullish_ob'] = True
        
        return df
    
    def identify_fvg(self, df):
        """識別Fair Value Gap - 3根K線間的缺口"""
        df = df.copy()
        
        df['bullish_fvg'] = False
        df['bearish_fvg'] = False
        
        for i in range(2, len(df)):
            # 看多FVG: K1高 < K3低 (中間有gap)
            if df.iloc[i-2]['high'] < df.iloc[i]['low']:
                df.at[i-1, 'bullish_fvg'] = True
            
            # 看空FVG: K1低 > K3高
            if df.iloc[i-2]['low'] > df.iloc[i]['high']:
                df.at[i-1, 'bearish_fvg'] = True
        
        return df
    
    def calculate_liquidity_zones(self, df):
        """計算流動性區域 - 高低點聚集"""
        df = df.copy()
        
        # 近20根的極值
        df['recent_high'] = df['high'].rolling(20).max()
        df['recent_low'] = df['low'].rolling(20).min()
        
        # 價格相對位置
        df['price_pct'] = (df['close'] - df['recent_low']) / (df['recent_high'] - df['recent_low'] + 1e-10)
        
        return df
    
    def generate_signals(self, df, use_ob=True, use_fvg=True, use_structure=True):
        """生成SMC交易信號"""
        df = self.identify_market_structure(df)
        df = self.identify_order_blocks(df)
        df = self.identify_fvg(df)
        df = self.calculate_liquidity_zones(df)
        
        signals = []
        
        for i in range(50, len(df)):
            r = df.iloc[i]
            prev = df.iloc[i-1]
            
            sig = 0
            reason = ""
            
            # 做多機會
            conditions_long = []
            
            if use_structure and r['trend'] == 1:
                conditions_long.append("UpTrend")
            
            if use_ob:
                # 查找近10根內的看多OB
                recent_ob = df.iloc[max(0, i-10):i]['bullish_ob'].any()
                if recent_ob:
                    conditions_long.append("BullOB")
            
            if use_fvg and r['bullish_fvg']:
                conditions_long.append("BullFVG")
            
            # 低位回調
            if r['price_pct'] < 0.4:
                conditions_long.append("LowZone")
            
            # 至少滿足2個條件
            if len(conditions_long) >= 2:
                sig = 1
                reason = "+".join(conditions_long)
            
            # 做空機會
            conditions_short = []
            
            if use_structure and r['trend'] == -1:
                conditions_short.append("DownTrend")
            
            if use_ob:
                recent_ob = df.iloc[max(0, i-10):i]['bearish_ob'].any()
                if recent_ob:
                    conditions_short.append("BearOB")
            
            if use_fvg and r['bearish_fvg']:
                conditions_short.append("BearFVG")
            
            # 高位回調
            if r['price_pct'] > 0.6:
                conditions_short.append("HighZone")
            
            if len(conditions_short) >= 2:
                if sig == 0:  # 避免衝突
                    sig = -1
                    reason = "+".join(conditions_short)
            
            signals.append({
                'signal': sig,
                'reason': reason,
                'trend': r['trend'],
                'price_pct': r['price_pct']
            })
        
        empty = [{'signal': 0, 'reason': '', 'trend': 0, 'price_pct': 0.5}] * 50
        return pd.DataFrame(empty + signals)


def render_strategy_a_tab(loader, symbol_selector):
    st.header("策略 A: SMC")
    
    st.info("""
    **SMC (Smart Money Concepts)**:
    
    機構交易逻輯:
    - Order Block: 機構大單區域
    - Fair Value Gap: 價格缺口
    - Market Structure: 趨勢結構
    - Liquidity Zones: 流動性區域
    
    進場:
    - 上升趨勢 + 看多OB + 低位回調
    - 下降趨勢 + 看空OB + 高位回調
    
    理念: 跟隨機構資金
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
        st.markdown("**風控**")
        stop_pct = st.slider("止損%", 0.5, 2.5, 1.2, 0.1, key="sl")
        target_pct = st.slider("止盈%", 0.8, 4.0, 2.0, 0.2, key="tp")
        
        st.markdown("**SMC功能**")
        use_ob = st.checkbox("Order Block", value=True, key="ob")
        use_fvg = st.checkbox("FVG", value=True, key="fvg")
        use_structure = st.checkbox("趨勢結構", value=True, key="struct")
    
    st.markdown("---")
    
    if st.button("執行SMC策略", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("載入...")
            prog.progress(20)
            
            if isinstance(loader, BinanceDataLoader):
                end = datetime.now()
                start = end - timedelta(days=test_days + 10)
                df_test = loader.load_historical_data(symbol, '15m', start, end)
            else:
                df_test = loader.load_klines(symbol, '15m')
                df_test = df_test.tail((test_days + 10) * 96)
            
            st.success(f"{len(df_test)}根")
            prog.progress(40)
            
            stat.text("分析SMC結構...")
            strategy = SMCStrategy()
            df_sig = strategy.generate_signals(df_test, use_ob, use_fvg, use_structure)
            
            # 加上止損止盈
            for i in range(len(df_sig)):
                if df_sig.iloc[i]['signal'] != 0:
                    r = df_test.iloc[i]
                    if df_sig.iloc[i]['signal'] == 1:
                        df_sig.at[i, 'stop_loss'] = r['close'] * (1 - stop_pct / 100)
                        df_sig.at[i, 'take_profit'] = r['close'] * (1 + target_pct / 100)
                    else:
                        df_sig.at[i, 'stop_loss'] = r['close'] * (1 + stop_pct / 100)
                        df_sig.at[i, 'take_profit'] = r['close'] * (1 - target_pct / 100)
                    df_sig.at[i, 'position_size'] = position_pct / 100.0
                else:
                    df_sig.at[i, 'stop_loss'] = np.nan
                    df_sig.at[i, 'take_profit'] = np.nan
                    df_sig.at[i, 'position_size'] = 1.0
            
            cnt = (df_sig['signal'] != 0).sum()
            
            if cnt == 0:
                st.warning("無SMC信號 - 嘗試調整參數或關閉某些功能")
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
            
            if wr >= 50 and pf >= 1.3:
                st.success("✅ SMC策略有效")
                st.balloons()
            elif ret > 0:
                st.info("⚠️ 有獲利但需優化")
            else:
                st.warning("❌ 需調整")
            
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
                
                # 顯示信號原因
                if 'reason' in df_sig.columns:
                    trades_with_reason = trades.copy()
                    trades_with_reason['reason'] = df_sig.loc[trades['entry_time']]['reason'].values if len(trades) <= len(df_sig) else ["N/A"] * len(trades)
                    st.dataframe(trades_with_reason[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(30), use_container_width=True)
                else:
                    st.dataframe(trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(30), use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_smc_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
