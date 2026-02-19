"""策略A: SSL趨勢 + ATR動態止損"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from models.ml_range_bound_strategy import MLRangeBoundStrategy
from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


def calculate_ssl_channel(df, period=10):
    """Calculate SSL Channel"""
    df = df.copy()
    df['ssl_down'] = df['low'].rolling(window=period).mean()
    df['ssl_up'] = df['high'].rolling(window=period).mean()
    
    df['ssl_signal'] = 0
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['ssl_down'].iloc[i]:
            df.loc[df.index[i], 'ssl_signal'] = 1
        elif df['close'].iloc[i] < df['ssl_up'].iloc[i]:
            df.loc[df.index[i], 'ssl_signal'] = -1
        else:
            df.loc[df.index[i], 'ssl_signal'] = df['ssl_signal'].iloc[i-1]
    
    return df


def calculate_ema(df, periods=[20, 50]):
    """Calculate EMAs"""
    df = df.copy()
    for p in periods:
        df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    return df


def render_strategy_a_tab(loader, symbol_selector):
    st.header("策略 A: SSL趨勢回調策略")
    
    st.info("""
    **SSL趨勢 + 回調進場**:
    
    趨勢確認: SSL通道 + EMA20/50
    進場時機: 等待回調到EMA20
    止損: 1.5 ATR
    止盈: 3.0 ATR (2:1 盈虧比)
    
    優勢: 跟隨趨勢 + 回調低風險進場
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_a", multi=False)
        symbol = symbol_list[0]
        train_days = st.slider("訓練", 60, 180, 90, key="train")
        test_days = st.slider("回測", 7, 60, 30, key="test")
    
    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap")
        leverage = st.slider("槓桿", 3, 10, 5, key="lev")
        position_pct = st.slider("仓位%", 40, 100, 60, 10, key="pos")
    
    with col3:
        st.markdown("**參數**")
        ssl_period = st.number_input("SSL", 5, 30, 10, key="ssl")
        confidence = st.slider("信心度", 0.35, 0.65, 0.45, 0.05, key="conf")
        atr_period = st.number_input("ATR", 10, 30, 14, key="atr")
    
    with st.expander("進階設定"):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("**止損/止盈**")
            sl_atr = st.slider("止損 ATR", 1.0, 3.0, 1.5, 0.5, key="sl")
            tp_atr = st.slider("止盈 ATR", 2.0, 5.0, 3.0, 0.5, key="tp")
        
        with col_a2:
            st.markdown("**進場策略**")
            wait_pullback = st.checkbox("等待回調", value=True, key="pb",
                                       help="回調到EMA20附近再進場")
            pullback_range = st.slider("回調範圍%", 0.5, 3.0, 1.5, 0.5, key="pb_rng") if wait_pullback else 0
    
    st.markdown("---")
    
    if st.button("執行回測", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("1/5: 載入...")
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
            prog.progress(20)
            
            stat.text("2/5: 訓練ML...")
            strategy = MLRangeBoundStrategy(bb_period=20, bb_std=2.0, adx_period=14, adx_threshold=30)
            stats = strategy.train(df_train, forward_bars=5)
            st.success(f"L:{stats['long_samples']} S:{stats['short_samples']}")
            prog.progress(40)
            
            stat.text("3/5: 計算指標...")
            df_test = strategy.add_indicators(df_test)
            df_test = calculate_ssl_channel(df_test, period=ssl_period)
            df_test = calculate_ema(df_test, periods=[20, 50])
            st.success("指標完成")
            prog.progress(60)
            
            stat.text("4/5: 生成信號...")
            
            signals = []
            rejected_wrong_trend = 0
            rejected_no_pullback = 0
            rejected_low_prob = 0
            
            for i in range(50, len(df_test)):
                lp, sp = strategy.predict(df_test, i)
                r = df_test.iloc[i]
                
                sig = 0
                sl = np.nan
                tp = np.nan
                
                # 趨勢確認
                ssl_long = r['ssl_signal'] == 1
                ssl_short = r['ssl_signal'] == -1
                ema_long = r['ema_20'] > r['ema_50']
                ema_short = r['ema_20'] < r['ema_50']
                
                # 回調確認
                if wait_pullback:
                    dist_to_ema20 = abs(r['close'] - r['ema_20']) / r['close'] * 100
                    near_ema20 = dist_to_ema20 < pullback_range
                else:
                    near_ema20 = True
                
                # 做多: SSL多 + EMA多 + 回調EMA20 + ML看漨
                if ssl_long and ema_long and near_ema20 and lp > confidence:
                    sig = 1
                    entry = r['close']
                    atr = r['atr']
                    sl = entry - sl_atr * atr
                    tp = entry + tp_atr * atr
                
                # 做空: SSL空 + EMA空 + 回調EMA20 + ML看跌  
                elif ssl_short and ema_short and near_ema20 and sp > confidence:
                    sig = -1
                    entry = r['close']
                    atr = r['atr']
                    sl = entry + sl_atr * atr
                    tp = entry - tp_atr * atr
                
                else:
                    # 記錄拒絕
                    if (ssl_long and not ema_long) or (ssl_short and not ema_short):
                        rejected_wrong_trend += 1
                    elif wait_pullback and not near_ema20:
                        rejected_no_pullback += 1
                    elif (ssl_long and ema_long and lp <= confidence) or (ssl_short and ema_short and sp <= confidence):
                        rejected_low_prob += 1
                
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
                st.warning("無信號")
                st.info(f"""
                **拒絕**:
                - 趨勢不符: {rejected_wrong_trend}
                - 無回調: {rejected_no_pullback}
                - 機率低: {rejected_low_prob}
                
                建議: 關閉回調等待 或 降低信心度到 0.40
                """)
                return
            
            st.success(f"{cnt}信號 | 拒: 趨勢{rejected_wrong_trend} 回調{rejected_no_pullback} 機率{rejected_low_prob}")
            prog.progress(80)
            
            stat.text("5/5: 回測...")
            engine = TickLevelBacktestEngine(capital, leverage, 0.0006, 0.02, 100)
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
            st.subheader("評估")
            
            col_e1, col_e2, col_e3 = st.columns(3)
            
            with col_e1:
                if wr >= 50 and pf >= 1.5:
                    st.success("✅ 良好")
                elif wr >= 45:
                    st.info("⚠️ 中等")
                else:
                    st.warning("❌ 不佳")
            
            with col_e2:
                if dd > -30:
                    st.success("✅ 回撤OK")
                elif dd > -40:
                    st.info("⚠️ 回撤偏高")
                else:
                    st.error("❌ 回撤太大")
            
            with col_e3:
                if monthly >= 50:
                    st.success(f"🚀 {monthly:.0f}%")
                elif monthly >= 30:
                    st.info(f"👍 {monthly:.0f}%")
                else:
                    st.warning(f"⚠️ {monthly:.0f}%")
            
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
                
                st.dataframe(trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(20), use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_ssl_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
