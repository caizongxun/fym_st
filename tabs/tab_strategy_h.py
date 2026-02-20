"""
Strategy H v1.0 - Hybrid Intelligent Trading System
混合智能交易系統

三層架構:
第 1 層: 市場狀態識別 (ML)
第 2 層: 交易信號生成 (指標 + ML)
第 3 層: 風控與執行 (RL Agent - 待建)

v1.0 功能:
- 多時間框架共振 (15m/1h/1d)
- 自動市場狀態識別
- 根據狀態切換策略 (做多/做空/網格)
- 量價確認
- 動態 TP/SL
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies.market_regime import MarketRegimeDetector
from strategies.multi_timeframe import MultiTimeframeLoader
from strategies.signal_generator import SignalGenerator
from data.binance_loader import BinanceDataLoader


def backtest_strategy_h(
    df: pd.DataFrame,
    signals: pd.Series,
    exit_levels: dict,
    capital: float = 10000,
    leverage: int = 3,
    position_size: float = 0.3,
    fee_rate: float = 0.0006
) -> tuple:
    """
    策略 H 回測
    """
    equity = capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]
    
    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        signal = signals.iloc[i]
        
        # 無倉位時檢查信號
        if position == 0 and signal != 0:
            position = signal  # 1 = 做多, -1 = 做空
            entry_price = current_price
            entry_time = df.index[i]
            
            # 記錄 TP/SL
            if position == 1:
                tp = exit_levels['tp_long'].iloc[i]
                sl = exit_levels['sl_long'].iloc[i]
            else:
                tp = exit_levels['tp_short'].iloc[i]
                sl = exit_levels['sl_short'].iloc[i]
        
        # 有倉位時檢查出場
        elif position != 0:
            exit_triggered = False
            exit_reason = ''
            
            # 檢查 TP/SL
            if position == 1:  # 做多
                if current_price >= tp:
                    exit_triggered = True
                    exit_reason = 'TP'
                elif current_price <= sl:
                    exit_triggered = True
                    exit_reason = 'SL'
            else:  # 做空
                if current_price <= tp:
                    exit_triggered = True
                    exit_reason = 'TP'
                elif current_price >= sl:
                    exit_triggered = True
                    exit_reason = 'SL'
            
            # 平倉
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


def render_strategy_h_tab(loader, symbol_selector):
    st.header("策略 H: 混合智能交易系統 v1.0 🤖")

    with st.expander("🌟 策略 H 核心優勢", expanded=True):
        st.markdown("""
        **三層智能架構**:
        
        🧠 **第 1 層: 市場狀態識別**
        - 多時間框架分析 (15m/1h/1d)
        - ML 自動識別 4 種市場: 上漲、下跌、震盪、高波
        
        🎯 **第 2 层: 自適應信號**
        - 上漲趨勢 → 只做多 (EMA20 回調 + RSI<40)
        - 下跌趨勢 → 只做空 (EMA20 反彈 + RSI>60)
        - 震盪整理 → 網格策略 (BB 上下軌)
        - 高波動 → 觀望不交易
        
        ✅ **核心優勢**:
        - 不再「只做空」，根據市場自動切換
        - 多時間框架共振，過濾假信號
        - 量價確認，提升勝率
        - 動態 TP/SL，適應不同市場
        
        🔥 **vs 策略 G (RL)**:
        - G: 黑盒，不知為何虧錢
        - H: 白盒，每個決策可解釋
        """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**數據設定**")
        symbol_list = symbol_selector("strategy_h", multi=False)
        symbol = symbol_list[0]
        
        train_days = st.slider("訓練天數", 90, 240, 120, key="train_h")
        test_days = st.slider("測試天數", 14, 60, 30, key="test_h")

    with col2:
        st.markdown("**交易參數**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_h")
        leverage = st.slider("槓桿", 1, 10, 3, key="lev_h")
        position_size = st.slider("倉位%", 10, 80, 30, 5, key="pos_h") / 100.0
        min_signal_strength = st.slider("最小信號強度", 0.3, 0.9, 0.5, 0.1, key="strength_h")

    if st.button("🚀 開始分析", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            # 第 1 步: 載入多時間框架數據
            stat.text("載入多時間框架數據...")
            prog.progress(10)
            
            mtf_loader = MultiTimeframeLoader(loader)
            total_days = train_days + test_days
            df_15m, df_1h, df_1d = mtf_loader.load_multi_timeframe(symbol, total_days)
            
            # 驗證數據
            validation = mtf_loader.validate_data(df_15m, df_1h, df_1d)
            if not all(validation.values()):
                st.warning(f"數據驗證: {validation}")
            
            st.info(f"15m: {len(df_15m)} 根 | 1h: {len(df_1h)} 根 | 1d: {len(df_1d)} 根")
            prog.progress(20)
            
            # 第 2 步: 市場狀態識別
            stat.text("識別市場狀態...")
            detector = MarketRegimeDetector()
            features = detector.calculate_features(df_15m, df_1h, df_1d)
            
            # 訓練模型
            labels = detector.label_regimes(features)
            split_idx = int(len(features) * (train_days / total_days))
            
            try:
                detector.train(features.iloc[:split_idx], labels.iloc[:split_idx])
                st.success("✅ ML 模型訓練完成")
            except:
                st.info("ℹ️ 使用規則基礎識別")
            
            regimes, regime_probas = detector.predict(features)
            prog.progress(40)
            
            # 第 3 步: 生成信號
            stat.text("生成交易信號...")
            signal_gen = SignalGenerator()
            
            # 分割訓練集和測試集
            df_test = df_1h.iloc[split_idx:].copy()
            regimes_test = regimes.iloc[split_idx:]
            
            # 為每個時間點生成信號
            all_signals = []
            all_exit_levels = []
            
            for i in range(len(df_test)):
                current_regime = regimes_test.iloc[i]
                df_window = df_test.iloc[:i+1]
                
                df_with_signals = signal_gen.generate_signals(df_window, current_regime)
                df_with_signals = signal_gen.filter_signals(
                    df_with_signals, 
                    min_strength=min_signal_strength
                )
                
                all_signals.append(df_with_signals.iloc[-1]['signal'])
                
                exit_levels = signal_gen.calculate_exit_levels(df_window, current_regime)
                all_exit_levels.append({
                    'tp_long': exit_levels['tp_long'].iloc[-1],
                    'sl_long': exit_levels['sl_long'].iloc[-1],
                    'tp_short': exit_levels['tp_short'].iloc[-1],
                    'sl_short': exit_levels['sl_short'].iloc[-1]
                })
            
            df_test['signal'] = all_signals
            df_test['regime'] = regimes_test.values
            
            exit_df = pd.DataFrame(all_exit_levels, index=df_test.index)
            
            prog.progress(60)
            
            # 第 4 步: 回測
            stat.text("執行回測...")
            trades, equity_curve = backtest_strategy_h(
                df_test,
                df_test['signal'],
                exit_df,
                capital,
                leverage,
                position_size
            )
            prog.progress(100)
            stat.text("完成")
            
            # 顯示結果
            st.markdown("### 市場狀態分析")
            regime_counts = regimes_test.value_counts()
            
            c1, c2, c3, c4 = st.columns(4)
            for regime_name in ['BULLISH_TREND', 'BEARISH_TREND', 'RANGE_BOUND', 'HIGH_VOLATILITY']:
                count = regime_counts.get(regime_name, 0)
                pct = count / len(regimes_test) * 100
                desc = detector.get_regime_description(regime_name)
                
                if regime_name == 'BULLISH_TREND':
                    c1.metric(f"{desc['emoji']} {desc['name']}", f"{pct:.1f}%")
                elif regime_name == 'BEARISH_TREND':
                    c2.metric(f"{desc['emoji']} {desc['name']}", f"{pct:.1f}%")
                elif regime_name == 'RANGE_BOUND':
                    c3.metric(f"{desc['emoji']} {desc['name']}", f"{pct:.1f}%")
                else:
                    c4.metric(f"{desc['emoji']} {desc['name']}", f"{pct:.1f}%")
            
            # 回測結果
            st.markdown("### 回測結果")
            final_equity = equity_curve[-1]
            total_return = (final_equity - capital) / capital * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("最終權益", f"${final_equity:,.0f}", f"{final_equity - capital:+,.0f}")
            c2.metric("總報酬", f"{total_return:.1f}%")
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
                
                # 檢查
                if profit_factor > 1.2:
                    st.success(f"✅ 盈虧比優秀: {profit_factor:.2f}")
                elif profit_factor > 0.8:
                    st.info(f"ℹ️ 盈虧比可接受: {profit_factor:.2f}")
                else:
                    st.warning(f"⚠️ 盈虧比偏低: {profit_factor:.2f}")
            
            # 權益曲線
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=equity_curve, mode='lines', name='權益'))
            fig.add_hline(y=capital, line_dash="dash", annotation_text="初始資金")
            fig.update_layout(title="權益曲線", xaxis_title="Steps", yaxis_title="Capital ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            # 交易明細
            if trades:
                st.subheader("交易記錄")
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"錯誤: {e}")
            import traceback
            with st.expander("詳情"): st.code(traceback.format_exc())
