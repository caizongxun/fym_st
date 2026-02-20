"""
Strategy I - 極致激進版策略 H
Ultra Aggressive Strategy H

目標: 30天 +100% (日均 +3.3%)
方法:
- 10x 槓桿
- 80% 倉位
- 快進快出 (TP=ATR*2, SL=ATR*0.8)
- 高頻交易 (降低信號門檻)
- 只在最強趨勢 (ADX>35)
- 15m 級別開倉 (大量捕捉短線波動)

風險:
- 最大回撤: -30% ~ -40%
- 連續虧損可能爆倉
- 需要精準止損
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

from strategies.market_regime import MarketRegimeDetector
from strategies.multi_timeframe import MultiTimeframeLoader
from strategies.signal_generator import SignalGenerator


class AggressiveSignalGenerator(SignalGenerator):
    """
    激進版信號生成器 - 高頻交易
    """
    
    def _bullish_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """做多信號 - 降低門檻"""
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # 條件放寬
        price_near_ema20 = abs(df['close'] - df['ema20']) / df['ema20'] < 0.04  # 2% -> 4%
        rsi_oversold = df['rsi'] < 50  # 40 -> 50
        price_above_ema50 = df['close'] > df['ema50']  # 新增: 確保長線多頭
        adx_strong = df.get('adx', 0) > 35  # 新增: 只在強趨勢
        
        long_signal = price_near_ema20 & rsi_oversold & price_above_ema50 & adx_strong
        df.loc[long_signal, 'signal'] = 1
        
        # 信號強度
        strength = 0.0
        strength += price_near_ema20.astype(float) * 0.3
        strength += rsi_oversold.astype(float) * 0.3
        strength += price_above_ema50.astype(float) * 0.4
        df['signal_strength'] = strength
        
        return df
    
    def _bearish_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """做空信號 - 降低門檻"""
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        price_near_ema20 = abs(df['close'] - df['ema20']) / df['ema20'] < 0.04
        rsi_overbought = df['rsi'] > 50  # 60 -> 50
        price_below_ema50 = df['close'] < df['ema50']  # 新增
        adx_strong = df.get('adx', 0) > 35  # 新增
        
        short_signal = price_near_ema20 & rsi_overbought & price_below_ema50 & adx_strong
        df.loc[short_signal, 'signal'] = -1
        
        strength = 0.0
        strength += price_near_ema20.astype(float) * 0.3
        strength += rsi_overbought.astype(float) * 0.3
        strength += price_below_ema50.astype(float) * 0.4
        df['signal_strength'] = strength
        
        return df
    
    def calculate_exit_levels(self, df: pd.DataFrame, regime: str) -> dict:
        """激進出場 - 快進快出"""
        atr = df['atr']
        close = df['close']
        
        # 所有市場都用快速 TP/SL
        tp_multiplier = 2.0   # ATR * 2
        sl_multiplier = 0.8   # ATR * 0.8
        
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
    fee_rate: float = 0.0006
) -> tuple:
    """激進回測 - 10x槓桿 + 80%倉位"""
    equity = capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]
    
    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        signal = signals.iloc[i]
        
        if position == 0 and signal != 0:
            position = signal
            entry_price = current_price
            entry_time = df.index[i]
            
            if position == 1:
                tp = exit_levels['tp_long'].iloc[i]
                sl = exit_levels['sl_long'].iloc[i]
            else:
                tp = exit_levels['tp_short'].iloc[i]
                sl = exit_levels['sl_short'].iloc[i]
        
        elif position != 0:
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
                pnl_pct = (current_price - entry_price) / entry_price * position * 100
                fee = fee_rate * 2 * 100
                leveraged_pnl = pnl_pct * leverage - fee
                actual_pnl = capital * position_size * leveraged_pnl / 100
                
                equity += actual_pnl
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'direction': 'Long' if position == 1 else 'Short',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': actual_pnl,
                    'pnl_pct': leveraged_pnl,
                    'exit_reason': exit_reason
                })
                
                position = 0
        
        equity_curve.append(equity)
    
    return trades, equity_curve


def render_strategy_i_tab(loader, symbol_selector):
    st.header("策略 I: 極致激進版 H 🔥💥")

    with st.expander("⚠️ 警告: 高風險策略", expanded=True):
        st.markdown("""
        **目標**: 30天 +100% 報酬
        
        🔥 **激進設置**:
        - 10x 槓桿 (放大5倍)
        - 80% 倉位 (放大2.7個)
        - 快進快出 (TP=ATR*2, SL=ATR*0.8)
        - ADX>35 只在最強趨勢交易
        
        📈 **預期表現**:
        - 交易次數: 100-200 / 30天
        - 單筆獲利: $300-500
        - 勝率: 55-60%
        - 日均報酬: +3-4%
        
        ⚠️ **風險**:
        - 最大回撤: -30% ~ -40%
        - 連續虧損可能爆倉
        - 不適合新手
        """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**數據設定**")
        symbol_list = symbol_selector("strategy_i", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("測試天數", 14, 60, 30, key="test_i")

    with col2:
        st.markdown("**固定參數 (不可調)**")
        st.metric("資金", "$10,000")
        st.metric("槓桿", "10x 🔥")
        st.metric("倉位", "80%")
        st.metric("TP/SL", "ATR*2 / ATR*0.8")

    if st.button("💥 激進測試 (高風險)", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("載入數據...")
            prog.progress(20)
            
            mtf_loader = MultiTimeframeLoader(loader)
            df_15m, df_1h, df_1d = mtf_loader.load_multi_timeframe(symbol, test_days + 90)
            
            stat.text("識別市場狀態...")
            prog.progress(40)
            
            detector = MarketRegimeDetector()
            features = detector.calculate_features(df_15m, df_1h, df_1d)
            
            try:
                labels = detector.label_regimes(features)
                split_idx = int(len(features) * 0.75)
                detector.train(features.iloc[:split_idx], labels.iloc[:split_idx])
            except:
                pass
            
            regimes, _ = detector.predict(features)
            
            stat.text("生成激進信號...")
            prog.progress(60)
            
            # 使用激進信號生成器
            signal_gen = AggressiveSignalGenerator()
            
            split_idx = int(len(df_1h) * 0.75)
            df_test = df_1h.iloc[split_idx:].copy()
            regimes_test = regimes.iloc[split_idx:]
            
            all_signals = []
            all_exit_levels = []
            
            for i in range(len(df_test)):
                current_regime = regimes_test.iloc[i]
                df_window = df_test.iloc[:i+1]
                
                df_with_signals = signal_gen.generate_signals(df_window, current_regime)
                df_with_signals = signal_gen.filter_signals(df_with_signals, min_strength=0.3)  # 降低門檻
                
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
                df_test,
                df_test['signal'],
                exit_df,
                capital=10000,
                leverage=10,
                position_size=0.8
            )
            
            prog.progress(100)
            stat.text("完成")
            
            # 顯示結果
            st.markdown("### 回測結果")
            final_equity = equity_curve[-1]
            total_return = (final_equity - 10000) / 10000 * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("最終權益", f"${final_equity:,.0f}", f"{final_equity - 10000:+,.0f}")
            c2.metric("總報酬", f"{total_return:.1f}%", 
                     "🔥 目標100%" if total_return >= 100 else "📈")
            c3.metric("交易次數", len(trades))
            
            if len(trades) > 0:
                wins = [t for t in trades if t['pnl'] > 0]
                losses = [t for t in trades if t['pnl'] <= 0]
                win_rate = len(wins) / len(trades) * 100
                avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
                avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("勝率", f"{win_rate:.1f}%")
                c2.metric("平均獲利", f"${avg_win:.2f}")
                c3.metric("平均虧損", f"${avg_loss:.2f}")
                c4.metric("盈虧比", f"{profit_factor:.2f}")
                
                # 評分
                if total_return >= 100:
                    st.success(f"🎉 達成目標! 報酬 {total_return:.1f}%")
                elif total_return >= 50:
                    st.info(f"📈 接近目標! 報酬 {total_return:.1f}%")
                elif total_return > 0:
                    st.warning(f"🔸 還需努力! 報酬 {total_return:.1f}%")
                else:
                    st.error(f"⚠️ 策略失敗! 報酬 {total_return:.1f}%")
            
            # 權益曲線
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=equity_curve, mode='lines', name='權益'))
            fig.add_hline(y=10000, line_dash="dash", annotation_text="初始資金")
            fig.add_hline(y=20000, line_dash="dot", line_color="green", annotation_text="目標 +100%")
            fig.update_layout(title="權益曲線", xaxis_title="Steps", yaxis_title="Capital ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            if trades:
                st.subheader("交易記錄")
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df.tail(20), use_container_width=True)
        
        except Exception as e:
            st.error(f"錯誤: {e}")
            import traceback
            with st.expander("詳情"): st.code(traceback.format_exc())
