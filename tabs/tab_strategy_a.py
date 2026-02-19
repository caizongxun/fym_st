"""策略A: 訂單流失衡 - 利用成交量偵測買賣壓力"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


class OrderFlowStrategy:
    """
    訂單流失衡策略
    
    理論:
    1. 大量買單推高價格 -> 上漨動能
    2. 大量賣單壓低價格 -> 下跌動能
    3. 當買賣失衡時,在反轉前進場
    
    核心指標:
    - OBV (On Balance Volume)
    - 量價背離
    - 獲利回吐區
    """
    
    def __init__(self):
        pass
    
    def calculate_obv(self, df):
        """計算OBV - 累積成交量"""
        df = df.copy()
        
        # 基本OBV
        df['price_change'] = df['close'].diff()
        df['obv'] = 0.0
        
        for i in range(1, len(df)):
            if df.iloc[i]['price_change'] > 0:
                df.iloc[i, df.columns.get_loc('obv')] = df.iloc[i-1]['obv'] + df.iloc[i]['volume']
            elif df.iloc[i]['price_change'] < 0:
                df.iloc[i, df.columns.get_loc('obv')] = df.iloc[i-1]['obv'] - df.iloc[i]['volume']
            else:
                df.iloc[i, df.columns.get_loc('obv')] = df.iloc[i-1]['obv']
        
        # OBV變化率
        df['obv_change_5'] = df['obv'].pct_change(5)
        df['obv_change_10'] = df['obv'].pct_change(10)
        
        # OBV移動平均
        df['obv_ma_20'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = (df['obv'] - df['obv_ma_20']) / (df['obv_ma_20'].abs() + 1e-10)
        
        return df
    
    def calculate_volume_profile(self, df):
        """計算成交量分佈"""
        df = df.copy()
        
        # 成交量異常
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_std'] = df['volume'].rolling(20).std()
        df['volume_zscore'] = (df['volume'] - df['volume_ma']) / (df['volume_std'] + 1e-10)
        
        # 量價關係
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
        
        # 買賣壓力指標
        df['buy_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)) * df['volume']
        df['sell_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)) * df['volume']
        
        df['pressure_diff'] = df['buy_pressure'] - df['sell_pressure']
        df['pressure_ratio'] = df['buy_pressure'] / (df['sell_pressure'] + 1e-10)
        
        # 累積壓力
        df['cum_pressure_5'] = df['pressure_diff'].rolling(5).sum()
        df['cum_pressure_10'] = df['pressure_diff'].rolling(10).sum()
        
        return df
    
    def calculate_support_resistance(self, df):
        """計算支撐壓力區"""
        df = df.copy()
        
        # 近20根K線的高低點
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        
        # 價格相對位置
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'] + 1e-10)
        
        # 距離支撐/壓力
        df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close']
        df['dist_to_support'] = (df['close'] - df['support']) / df['close']
        
        return df
    
    def generate_signals(self, df, obv_threshold=0.02, pressure_threshold=0, volume_z=1.0):
        """生成交易信號"""
        df = self.calculate_obv(df)
        df = self.calculate_volume_profile(df)
        df = self.calculate_support_resistance(df)
        
        signals = []
        
        for i in range(50, len(df)):
            r = df.iloc[i]
            
            sig = 0
            reason = ""
            
            # 做多條件:
            # 1. OBV上升 (買盤增加)
            # 2. 累積買壓 > 賣壓
            # 3. 成交量異常增加
            # 4. 靠近支撐位
            
            long_obv = r['obv_change_5'] > obv_threshold
            long_pressure = r['cum_pressure_5'] > pressure_threshold
            long_volume = r['volume_zscore'] > volume_z
            long_position = r['price_position'] < 0.3  # 低位
            
            if long_obv and long_pressure and (long_volume or long_position):
                sig = 1
                reason = "OBV上升+買壓"
            
            # 做空條件:
            # 1. OBV下降 (賣盤增加)
            # 2. 累積賣壓 > 買壓
            # 3. 成交量異常增加
            # 4. 靠近壓力位
            
            short_obv = r['obv_change_5'] < -obv_threshold
            short_pressure = r['cum_pressure_5'] < -pressure_threshold
            short_volume = r['volume_zscore'] > volume_z
            short_position = r['price_position'] > 0.7  # 高位
            
            if short_obv and short_pressure and (short_volume or short_position):
                sig = -1
                reason = "OBV下降+賣壓"
            
            signals.append({
                'signal': sig,
                'reason': reason,
                'obv_trend': r['obv_change_5'],
                'pressure': r['cum_pressure_5'],
                'volume_z': r['volume_zscore'],
                'position': r['price_position']
            })
        
        # 前50筆無信號
        empty_signals = [{'signal': 0, 'reason': '', 'obv_trend': 0, 'pressure': 0, 'volume_z': 0, 'position': 0.5}] * 50
        
        return pd.DataFrame(empty_signals + signals)


def render_strategy_a_tab(loader, symbol_selector):
    st.header("策略 A: 訂單流失衡")
    
    st.info("""
    **訂單流失衡策略**:
    
    理論基礎:
    - 大量買單推高價格 (做多機會)
    - 大量賣單壓低價格 (做空機會)
    
    核心指標:
    - OBV (累積成交量)
    - 買賣壓力差值
    - 成交量異常
    - 支撐/壓力位置
    
    優勢: 捕捉主力資金動向
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
        st.markdown("**參數**")
        obv_thresh = st.slider("OBV門檻%", 1.0, 5.0, 2.0, 0.5, key="obv") / 100
        volume_z = st.slider("量能Z值", 0.5, 2.5, 1.0, 0.5, key="vz")
        stop_pct = st.slider("止損%", 0.5, 2.0, 1.0, 0.1, key="sl")
        target_pct = st.slider("止盈%", 0.5, 3.0, 1.5, 0.1, key="tp")
    
    st.markdown("---")
    
    if st.button("執行訂單流策略", type="primary", use_container_width=True):
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
            
            stat.text("計算訂單流...")
            strategy = OrderFlowStrategy()
            df_sig = strategy.generate_signals(df_test, obv_thresh, 0, volume_z)
            
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
                st.warning("無信號 - 調低OBV門檻")
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
            
            if wr >= 50 and pf >= 1.2:
                st.success("✅ 策略有效")
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
                
                st.dataframe(trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(30), use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_orderflow_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
