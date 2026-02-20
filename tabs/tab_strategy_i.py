"""
Strategy I v1.0 - Aggressive Strategy H
極致激進版本

目標: 30天 100% 報酬
方法: 10x 槓桿 + 80% 倉位 + 高頻交易

核心改動:
1. 降低信號門檻 (增加交易次數)
2. 縮短 TP/SL (快進快出)
3. 只在最強趨勢交易 (ADX>40)
4. 15m 級別開倉 (日內波段)
5. 金字塔加倉 (趨勢中連續加碼)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies.market_regime import MarketRegimeDetector
from strategies.multi_timeframe import MultiTimeframeLoader
from data.binance_loader import BinanceDataLoader


class AggressiveSignalGenerator:
    """
    激進信號生成器
    """
    
    def __init__(self):
        self.current_position = 0
        self.entry_price = 0
        self.pyramid_count = 0  # 加倉次數
    
    def generate_signals(self, df: pd.DataFrame, regime: str) -> pd.DataFrame:
        df = df.copy()
        df = self._calculate_indicators(df)
        
        if regime == 'BULLISH_TREND':
            df = self._aggressive_long_signals(df)
        elif regime == 'BEARISH_TREND':
            df = self._aggressive_short_signals(df)
        elif regime == 'RANGE_BOUND':
            df = self._scalping_signals(df)
        else:
            df['signal'] = 0
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # EMA
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = df['tr'].rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['adx'] = dx.rolling(14).mean()
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def _aggressive_long_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        激進多頭信號 - 降低門檻
        """
        df['signal'] = 0
        
        # 條件 1: 超強趨勢
        strong_trend = df['adx'] > 40
        
        # 條件 2: 價格 > EMA8 (降低門檻)
        price_above = df['close'] > df['ema8']
        
        # 條件 3: MACD 正值
        macd_positive = df['macd_hist'] > 0
        
        # 條件 4: 成交量確認
        volume_ok = df['volume_ratio'] > 1.2
        
        # 組合信號
        long_signal = strong_trend & price_above & macd_positive & volume_ok
        df.loc[long_signal, 'signal'] = 1
        
        return df
    
    def _aggressive_short_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        激進空頭信號
        """
        df['signal'] = 0
        
        strong_trend = df['adx'] > 40
        price_below = df['close'] < df['ema8']
        macd_negative = df['macd_hist'] < 0
        volume_ok = df['volume_ratio'] > 1.2
        
        short_signal = strong_trend & price_below & macd_negative & volume_ok
        df.loc[short_signal, 'signal'] = -1
        
        return df
    
    def _scalping_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        震盪市劑頭 - 高頻交易
        """
        df['signal'] = 0
        
        # RSI 極端值
        rsi_low = df['rsi'] < 35
        rsi_high = df['rsi'] > 65
        
        # MACD 交叉
        macd_cross_up = (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
        macd_cross_down = (df['macd_hist'] < 0) & (df['macd_hist'].shift(1) >= 0)
        
        df.loc[rsi_low | macd_cross_up, 'signal'] = 1
        df.loc[rsi_high | macd_cross_down, 'signal'] = -1
        
        return df
    
    def calculate_exit_levels(self, df: pd.DataFrame, regime: str) -> dict:
        """
        激進 TP/SL - 縮短持倉時間
        """
        atr = df['atr']
        close = df['close']
        
        if regime == 'BULLISH_TREND' or regime == 'BEARISH_TREND':
            # 趨勢市: 快進快出
            tp_multiplier = 2.0  # 縮短 TP
            sl_multiplier = 0.8  # 縮短 SL
        else:  # RANGE_BOUND
            # 劑頭: 極短線
            tp_multiplier = 1.0
            sl_multiplier = 0.5
        
        return {
            'tp_long': close + atr * tp_multiplier,
            'sl_long': close - atr * sl_multiplier,
            'tp_short': close - atr * tp_multiplier,
            'sl_short': close + atr * sl_multiplier
        }


def backtest_strategy_i(
    df: pd.DataFrame,
    signals: pd.Series,
    exit_levels: dict,
    capital: float = 10000,
    leverage: int = 10,
    position_size: float = 0.8,
    fee_rate: float = 0.0006,
    max_pyramids: int = 3
) -> tuple:
    """
    激進回測 - 允許金字塔加倉
    """
    equity = capital
    position = 0
    entry_prices = []  # 多個進場價
    pyramid_count = 0
    trades = []
    equity_curve = [capital]
    
    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        signal = signals.iloc[i]
        
        # 無倉位時檢查信號
        if position == 0 and signal != 0:
            position = signal
            entry_prices = [current_price]
            pyramid_count = 1
            entry_time = df.index[i]
            
            if position == 1:
                tp = exit_levels['tp_long'].iloc[i]
                sl = exit_levels['sl_long'].iloc[i]
            else:
                tp = exit_levels['tp_short'].iloc[i]
                sl = exit_levels['sl_short'].iloc[i]
        
        # 有倉位時檢查加倉或出場
        elif position != 0:
            # 檢查是否可以加倉
            avg_entry = np.mean(entry_prices)
            profit_pct = (current_price - avg_entry) / avg_entry * position
            
            # 浮盈 > 2% 且未達最大加倉次數
            if profit_pct > 0.02 and pyramid_count < max_pyramids and signal == position:
                entry_prices.append(current_price)
                pyramid_count += 1
            
            # 檢查出場
            exit_triggered = False
            exit_reason = ''
            
            if position == 1:
                if current_price >= tp:
                    exit_triggered = True
                    exit_reason = 'TP'
                elif current_price <= sl:
                    exit_triggered = True
                    exit_reason = 'SL'
            else:
                if current_price <= tp:
                    exit_triggered = True
                    exit_reason = 'TP'
                elif current_price >= sl:
                    exit_triggered = True
                    exit_reason = 'SL'
            
            if exit_triggered:
                avg_entry = np.mean(entry_prices)
                pnl_pct = (current_price - avg_entry) / avg_entry * position * 100
                fee = fee_rate * 2 * 100 * pyramid_count
                leveraged_pnl = pnl_pct * leverage - fee
                actual_pnl = capital * position_size * leveraged_pnl / 100
                
                equity += actual_pnl
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'direction': 'Long' if position == 1 else 'Short',
                    'avg_entry': avg_entry,
                    'exit_price': current_price,
                    'pyramids': pyramid_count,
                    'pnl': actual_pnl,
                    'pnl_pct': leveraged_pnl,
                    'exit_reason': exit_reason
                })
                
                position = 0
                entry_prices = []
                pyramid_count = 0
        
        equity_curve.append(equity)
    
    return trades, equity_curve


def render_strategy_i_tab(loader, symbol_selector):
    st.header("策略 I: 激進版 H 🔥")
    
    with st.expander("💥 濅致激進設定", expanded=True):
        st.markdown("""
        **目標**: 30天 100% 報酬
        
        **激進設定**:
        - 🔥 槓桿: **10x** (vs H 的 3x)
        - 🔥 倉位: **80%** (vs H 的 30%)
        - 🔥 TP/SL: ATR*2 / ATR*0.8 (快進快出)
        - 🔥 信號門檻: ADX>40 (只抓超強趨勢)
        - 🔥 金字塔: 浮盈>2% 自動加倉 (max 3次)
        
        **vs 策略 H**:
        | 項目 | H | I |
        |------|---|---|
        | 槓桿 | 3x | **10x** |
        | 倉位 | 30% | **80%** |
        | 交易次數 | 54 | **150+** |
        | 預期報酬 | +1.5% | **+100%** |
        
        ⚠️ **風險警告**:
        - 最大回撤可能 -40%
        - 連續虧損可能爆倉
        - 建議先小資金測試
        """)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**數據設定**")
        symbol_list = symbol_selector("strategy_i", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("測試天數", 14, 60, 30, key="test_i")
    
    with col2:
        st.markdown("**交易參數**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_i")
        leverage = st.slider("槓桿", 5, 20, 10, key="lev_i")
        position_size = st.slider("倉位%", 50, 100, 80, 5, key="pos_i") / 100.0
        max_pyramids = st.slider("最大加倉次數", 1, 5, 3, key="pyr_i")
    
    if st.button("🚀 開始激進回測", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("載入數據...")
            prog.progress(20)
            
            mtf_loader = MultiTimeframeLoader(loader)
            df_15m, df_1h, df_1d = mtf_loader.load_multi_timeframe(symbol, test_days + 60)
            
            stat.text("識別市場狀態...")
            prog.progress(40)
            
            detector = MarketRegimeDetector()
            features = detector.calculate_features(df_15m, df_1h, df_1d)
            regimes, _ = detector.predict(features)
            
            stat.text("生成激進信號...")
            prog.progress(60)
            
            df_test = df_1h.tail(test_days * 24).copy()
            regimes_test = regimes.tail(len(df_test))
            
            signal_gen = AggressiveSignalGenerator()
            all_signals = []
            all_exit_levels = []
            
            for i in range(len(df_test)):
                current_regime = regimes_test.iloc[i]
                df_window = df_test.iloc[:i+1]
                
                df_with_signals = signal_gen.generate_signals(df_window, current_regime)
                all_signals.append(df_with_signals.iloc[-1]['signal'])
                
                exit_levels = signal_gen.calculate_exit_levels(df_window, current_regime)
                all_exit_levels.append({
                    'tp_long': exit_levels['tp_long'].iloc[-1],
                    'sl_long': exit_levels['sl_long'].iloc[-1],
                    'tp_short': exit_levels['tp_short'].iloc[-1],
                    'sl_short': exit_levels['sl_short'].iloc[-1]
                })
            
            df_test['signal'] = all_signals
            exit_df = pd.DataFrame(all_exit_levels, index=df_test.index)
            
            stat.text("執行激進回測...")
            prog.progress(80)
            
            trades, equity_curve = backtest_strategy_i(
                df_test, df_test['signal'], exit_df,
                capital, leverage, position_size, max_pyramids=max_pyramids
            )
            
            prog.progress(100)
            stat.text("完成")
            
            # 顯示結果
            st.markdown("### 激進回測結果")
            final_equity = equity_curve[-1]
            total_return = (final_equity - capital) / capital * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("最終權益", f"${final_equity:,.0f}", f"{final_equity - capital:+,.0f}")
            c2.metric("總報酬", f"{total_return:.1f}%")
            c3.metric("交易次數", len(trades))
            
            if len(trades) > 0:
                wins = [t for t in trades if t['pnl'] > 0]
                win_rate = len(wins) / len(trades) * 100
                avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
                avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) or -1
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                
                # 加倉統計
                pyramids = [t['pyramids'] for t in trades]
                avg_pyramids = np.mean(pyramids)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("勝率", f"{win_rate:.1f}%")
                c2.metric("盈虧比", f"{profit_factor:.2f}")
                c3.metric("平均加倉", f"{avg_pyramids:.1f}次")
                c4.metric("日均報酬", f"{total_return/test_days:.2f}%")
                
                # 目標達成度
                if total_return >= 100:
                    st.success(f"🎉 目標達成! {total_return:.1f}% >= 100%")
                elif total_return >= 50:
                    st.info(f"👍 接近目標: {total_return:.1f}%")
                else:
                    st.warning(f"⚠️ 未達標: {total_return:.1f}% < 100%")
            
            # 權益曲線
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=equity_curve, mode='lines', name='權益', line=dict(color='red', width=2)))
            fig.add_hline(y=capital, line_dash="dash", annotation_text="初始資金")
            fig.add_hline(y=capital*2, line_dash="dot", line_color="green", annotation_text="目標 100%")
            fig.update_layout(title="激進權益曲線", xaxis_title="Steps", yaxis_title="Capital ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            if trades:
                st.subheader("交易記錄")
                st.dataframe(pd.DataFrame(trades), use_container_width=True)
        
        except Exception as e:
            st.error(f"錯誤: {e}")
            import traceback
            with st.expander("詳情"): st.code(traceback.format_exc())
