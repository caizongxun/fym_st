"""策略A: SSL通道 + 動量突破"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from models.ml_range_bound_strategy import MLRangeBoundStrategy
from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


def calculate_ssl_channel(df, period=10):
    """Calculate SSL Channel indicator"""
    df = df.copy()
    
    # SSL Down = SMA of Low
    df['ssl_down'] = df['low'].rolling(window=period).mean()
    
    # SSL Up = SMA of High  
    df['ssl_up'] = df['high'].rolling(window=period).mean()
    
    # Determine SSL direction
    df['ssl_signal'] = 0
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['ssl_down'].iloc[i]:
            df.loc[df.index[i], 'ssl_signal'] = 1  # Bullish
        elif df['close'].iloc[i] < df['ssl_up'].iloc[i]:
            df.loc[df.index[i], 'ssl_signal'] = -1  # Bearish
        else:
            df.loc[df.index[i], 'ssl_signal'] = df['ssl_signal'].iloc[i-1]  # Keep previous
    
    return df


def render_strategy_a_tab(loader, symbol_selector):
    st.header("策略 A: SSL通道動量突破")
    
    st.info("""
    **SSL通道 + 動量策略**:
    
    進場:
    - 做多: 價格突破SSL通道上方 + ML確認看漲
    - 做空: 價格跌SSL通道下方 + ML確認看跌
    
    風控:
    - 止損: 1% (固定%)
    - 止盈: 2% (2:1 盈虧比)
    - 中等仓位 + 中等槓桿
    
    目標: 勝率55-65% | 月化 40-80%
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_a", multi=False)
        symbol = symbol_list[0]
        train_days = st.slider("訓練天數", 60, 180, 90, key="train")
        test_days = st.slider("回測天數", 7, 60, 30, key="test")
    
    with col2:
        st.markdown("**交易設定**")
        capital = st.number_input("資金 (USDT)", 1000.0, 100000.0, 10000.0, 1000.0, key="cap")
        leverage = st.slider("槓桿倍數", 3, 10, 6, key="lev")
        position_pct = st.slider("仓位%", 40, 100, 70, 10, key="pos")
    
    with col3:
        st.markdown("**策略參數**")
        ssl_period = st.number_input("SSL週期", 5, 30, 10, key="ssl")
        confidence = st.slider("ML信心度", 0.35, 0.65, 0.48, 0.02, key="conf")
        atr_period = st.number_input("ATR週期", 10, 30, 14, key="atr")
    
    with st.expander("進階: 風險管理"):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("**止損/止盈**")
            stop_loss_pct = st.slider("止損%", 0.5, 2.0, 1.0, 0.1, key="sl", help="固定%")
            take_profit_pct = st.slider("止盈%", 1.0, 4.0, 2.0, 0.5, key="tp", help="固定%")
        
        with col_a2:
            st.markdown("**過濾条件**")
            use_atr_filter = st.checkbox("使用ATR過濾", value=True, key="atr_flt",
                                        help="只在波動性足夠時交易")
            min_atr_pct = st.slider("最小ATR%", 0.1, 1.0, 0.3, 0.1, key="min_atr") if use_atr_filter else 0
    
    st.markdown("---")
    
    if st.button("執行SSL策略回測", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            # Load
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
            
            # Train ML
            stat.text("2/5: 訓練ML...")
            strategy = MLRangeBoundStrategy(bb_period=20, bb_std=2.0, adx_period=14, adx_threshold=30)
            stats = strategy.train(df_train, forward_bars=5)  # 短期預測
            st.success(f"L:{stats['long_samples']} S:{stats['short_samples']}")
            prog.progress(40)
            
            # Calculate SSL
            stat.text("3/5: 計算SSL通道...")
            df_test = strategy.add_indicators(df_test)
            df_test = calculate_ssl_channel(df_test, period=ssl_period)
            st.success("SSL計算完成")
            prog.progress(60)
            
            # Generate signals
            stat.text("4/5: 生成信號...")
            
            signals = []
            rejected_no_ssl_change = 0
            rejected_low_prob = 0
            rejected_low_atr = 0
            
            for i in range(50, len(df_test)):
                lp, sp = strategy.predict(df_test, i)
                r = df_test.iloc[i]
                prev = df_test.iloc[i-1]
                
                sig = 0
                sl = np.nan
                tp = np.nan
                
                # ATR過濾
                atr_pct = r['atr'] / r['close'] * 100
                atr_ok = not use_atr_filter or atr_pct >= min_atr_pct
                
                # SSL方向變化 (突破)
                ssl_change = r['ssl_signal'] != prev['ssl_signal']
                ssl_long = r['ssl_signal'] == 1
                ssl_short = r['ssl_signal'] == -1
                
                # 做多: SSL轉多 + ML看漨
                if ssl_long and lp > confidence and atr_ok:
                    sig = 1
                    entry = r['close']
                    sl = entry * (1 - stop_loss_pct / 100)
                    tp = entry * (1 + take_profit_pct / 100)
                
                # 做空: SSL轉空 + ML看跌
                elif ssl_short and sp > confidence and atr_ok:
                    sig = -1
                    entry = r['close']
                    sl = entry * (1 + stop_loss_pct / 100)
                    tp = entry * (1 - take_profit_pct / 100)
                
                else:
                    # 記錄拒絕
                    if not ssl_long and not ssl_short:
                        rejected_no_ssl_change += 1
                    elif (ssl_long and lp <= confidence) or (ssl_short and sp <= confidence):
                        rejected_low_prob += 1
                    elif not atr_ok:
                        rejected_low_atr += 1
                
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
                **拒絕原因**:
                - SSL無方向: {rejected_no_ssl_change}
                - ML機率低: {rejected_low_prob}
                - ATR不足: {rejected_low_atr}
                
                建議: 降低信心度到 0.40 或關ATR過濾
                """)
                return
            
            st.success(f"{cnt}信號 | 拒: SSL{rejected_no_ssl_change} 機率{rejected_low_prob} ATR{rejected_low_atr}")
            prog.progress(80)
            
            # Backtest
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
            
            # Eval
            st.markdown("---")
            st.subheader("評估")
            
            col_e1, col_e2, col_e3 = st.columns(3)
            
            with col_e1:
                if wr >= 60 and pf >= 1.5:
                    st.success("✅ 優秀: 勝率>60% 盈虧比>1.5")
                    st.balloons()
                elif wr >= 55:
                    st.success("👍 良好: 勝率>55%")
                elif wr >= 50:
                    st.info("⚠️ 中等: 勝率>50%")
                else:
                    st.warning("❌ 待優化")
            
            with col_e2:
                if dd > -30:
                    st.success(f"✅ 回撤<30%")
                elif dd > -40:
                    st.info(f"⚠️ 回撤<40%")
                else:
                    st.error(f"❌ 回撤>40%")
            
            with col_e3:
                if monthly >= 60:
                    st.success(f"🚀 月化>{monthly:.0f}%")
                elif monthly >= 40:
                    st.info(f"👍 月化>{monthly:.0f}%")
                elif monthly >= 20:
                    st.warning(f"⚠️ 月化={monthly:.0f}%")
                else:
                    st.error(f"❌ 月化<20%")
            
            # Chart
            st.markdown("---")
            st.subheader("權益曲線")
            fig = engine.plot_equity_curve()
            st.plotly_chart(fig, use_container_width=True)
            
            # Trades
            trades = engine.get_trades_dataframe()
            if not trades.empty:
                st.markdown("---")
                st.subheader("交易記錄")
                
                wins = trades[trades['pnl_usdt'] > 0]
                losses = trades[trades['pnl_usdt'] < 0]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("獲利", len(wins))
                c2.metric("虧損", len(losses))
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
