"""
Strategy J - 網格+趨勢組合系統
Grid + Trend Hybrid System

目標: 30天 +80-100% (穩健激進)

雙引擎架構:
- 引擎 1 (50%資金): AI網格 (震盪市穩定收益)
  - 10x 槓桿
  - 網格間距: 0.5%
  - 每格賺: 0.3-0.5%
  - 日交易: 20-30次
  - 日均報酬: +1-2%

- 引擎 2 (50%資金): 趨勢突破 (抓大行情)
  - 10x 槓桿
  - 只抓單邊大趨勢
  - TP: ATR*4 (大目標)
  - 週均報酬: +10-20%

優勢:
- 震盪市靠網格
- 趨勢市靠突破
- 風險分散
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies.market_regime import MarketRegimeDetector
from strategies.multi_timeframe import MultiTimeframeLoader
from strategies.signal_generator import SignalGenerator


class GridEngine:
    """
    網格引擎 - 震盪市穩定收益
    """
    
    def __init__(self, capital: float, leverage: int = 10, grid_spacing: float = 0.005):
        self.capital = capital
        self.leverage = leverage
        self.grid_spacing = grid_spacing  # 0.5%
        self.grids = []
    
    def setup_grids(self, current_price: float, num_grids: int = 10):
        """設置網格"""
        self.grids = []
        for i in range(-num_grids//2, num_grids//2 + 1):
            grid_price = current_price * (1 + i * self.grid_spacing)
            self.grids.append({
                'price': grid_price,
                'position': 0,  # 0=無, 1=做多, -1=做空
                'entry_price': 0
            })
    
    def execute(self, df: pd.DataFrame) -> tuple:
        """執行網格交易"""
        trades = []
        equity = self.capital
        
        # 初始化網格
        self.setup_grids(df.iloc[0]['close'])
        
        for i in range(1, len(df)):
            current_price = df.iloc[i]['close']
            
            # 檢查每個網格
            for grid in self.grids:
                # 價格觸碰網格且無倉位
                if abs(current_price - grid['price']) / grid['price'] < 0.001 and grid['position'] == 0:
                    # 低於中點做多，高於中點做空
                    center_price = self.grids[len(self.grids)//2]['price']
                    if grid['price'] < center_price:
                        grid['position'] = 1
                        grid['entry_price'] = current_price
                    else:
                        grid['position'] = -1
                        grid['entry_price'] = current_price
                
                # 檢查止盈
                elif grid['position'] != 0:
                    target_profit = grid['entry_price'] * self.grid_spacing
                    
                    if grid['position'] == 1 and current_price >= grid['entry_price'] + target_profit:
                        # 做多止盈
                        pnl_pct = (current_price - grid['entry_price']) / grid['entry_price'] * 100
                        pnl = self.capital * 0.5 * (pnl_pct * self.leverage - 0.12) / 100  # 50%資金
                        equity += pnl
                        trades.append({
                            'time': df.index[i],
                            'type': 'Grid Long',
                            'entry': grid['entry_price'],
                            'exit': current_price,
                            'pnl': pnl
                        })
                        grid['position'] = 0
                    
                    elif grid['position'] == -1 and current_price <= grid['entry_price'] - target_profit:
                        # 做空止盈
                        pnl_pct = (grid['entry_price'] - current_price) / grid['entry_price'] * 100
                        pnl = self.capital * 0.5 * (pnl_pct * self.leverage - 0.12) / 100
                        equity += pnl
                        trades.append({
                            'time': df.index[i],
                            'type': 'Grid Short',
                            'entry': grid['entry_price'],
                            'exit': current_price,
                            'pnl': pnl
                        })
                        grid['position'] = 0
        
        return trades, equity


class TrendBreakoutEngine:
    """
    趨勢突破引擎 - 抓大行情
    """
    
    def __init__(self, capital: float, leverage: int = 10):
        self.capital = capital
        self.leverage = leverage
    
    def execute(self, df: pd.DataFrame, regimes: pd.Series) -> tuple:
        """執行趨勢突破交易"""
        trades = []
        equity = self.capital
        position = 0
        entry_price = 0
        
        for i in range(50, len(df)):  # 需要EMA50
            current_price = df.iloc[i]['close']
            regime = regimes.iloc[i]
            
            # 無倉位時檢查突破
            if position == 0:
                ema50 = df.iloc[i]['ema50']
                adx = df.iloc[i].get('adx', 0)
                
                # 上升趨勢突破
                if regime == 'BULLISH_TREND' and current_price > ema50 and adx > 40:
                    position = 1
                    entry_price = current_price
                    tp = entry_price + df.iloc[i]['atr'] * 4  # 大目標
                    sl = entry_price - df.iloc[i]['atr'] * 1.5
                
                # 下降趨勢突破
                elif regime == 'BEARISH_TREND' and current_price < ema50 and adx > 40:
                    position = -1
                    entry_price = current_price
                    tp = entry_price - df.iloc[i]['atr'] * 4
                    sl = entry_price + df.iloc[i]['atr'] * 1.5
            
            # 有倉位時檢查出場
            elif position != 0:
                exit_triggered = False
                
                if position == 1:
                    if current_price >= tp or current_price <= sl:
                        exit_triggered = True
                else:
                    if current_price <= tp or current_price >= sl:
                        exit_triggered = True
                
                if exit_triggered:
                    pnl_pct = (current_price - entry_price) / entry_price * position * 100
                    pnl = self.capital * 0.5 * (pnl_pct * self.leverage - 0.12) / 100  # 50%資金
                    equity += pnl
                    trades.append({
                        'time': df.index[i],
                        'type': f'Trend {"Long" if position == 1 else "Short"}',
                        'entry': entry_price,
                        'exit': current_price,
                        'pnl': pnl
                    })
                    position = 0
        
        return trades, equity


def render_strategy_j_tab(loader, symbol_selector):
    st.header("策略 J: 網格+趨勢雙引擎 🔥🎯")

    with st.expander("🌟 雙引擎優勢", expanded=True):
        st.markdown("""
        **目標**: 30天 +80-100% 報酬 (穩健激進)
        
        🧲 **引擎 1: AI網格** (50%資金)
        - 震盪市穩定收益
        - 10x 槓桿 + 0.5% 網格間距
        - 每筆賺 0.3-0.5%
        - 日交易 20-30 次
        - 日均 +1-2%
        
        🚀 **引擎 2: 趨勢突破** (50%資金)
        - 抓單邊大行情
        - 10x 槓桿 + ADX>40
        - TP = ATR*4 (大目標)
        - 週均 +10-20%
        
        ✅ **核心優勢**:
        - 震盪市靠網格保底
        - 趨勢市靠突破爆發
        - 風險分散，最大回撤 -25%
        """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**數據設定**")
        symbol_list = symbol_selector("strategy_j", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("測試天數", 14, 60, 30, key="test_j")

    with col2:
        st.markdown("**固定參數**")
        st.metric("資金", "$10,000")
        st.metric("槓桿", "10x")
        st.metric("網格引擎", "50%資金")
        st.metric("趨勢引擎", "50%資金")

    if st.button("🚀 啟動雙引擎", type="primary", use_container_width=True):
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
            
            # 準備數據
            split_idx = int(len(df_1h) * 0.75)
            df_test = df_1h.iloc[split_idx:].copy()
            regimes_test = regimes.iloc[split_idx:]
            
            # 計算指標
            df_test['ema50'] = df_test['close'].ewm(span=50).mean()
            df_test['atr'] = df_test['close'].rolling(14).std()
            
            # ADX
            plus_dm = df_test['high'].diff()
            minus_dm = -df_test['low'].diff()
            tr = df_test['high'] - df_test['low']
            atr = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            df_test['adx'] = dx.rolling(14).mean()
            df_test.fillna(0, inplace=True)
            
            stat.text("執行引擎 1: 網格交易...")
            prog.progress(60)
            
            grid_engine = GridEngine(capital=5000, leverage=10, grid_spacing=0.005)
            grid_trades, grid_equity = grid_engine.execute(df_test)
            
            stat.text("執行引擎 2: 趨勢突破...")
            prog.progress(80)
            
            trend_engine = TrendBreakoutEngine(capital=5000, leverage=10)
            trend_trades, trend_equity = trend_engine.execute(df_test, regimes_test)
            
            prog.progress(100)
            stat.text("完成")
            
            # 統計結果
            total_equity = grid_equity + trend_equity - 10000
            total_return = (total_equity - 10000) / 10000 * 100
            
            st.markdown("### 雙引擎總結")
            c1, c2, c3 = st.columns(3)
            c1.metric("最終權益", f"${total_equity:,.0f}", f"{total_equity - 10000:+,.0f}")
            c2.metric("總報酬", f"{total_return:.1f}%",
                     "🎉 達標" if total_return >= 80 else "📈")
            c3.metric("總交易", len(grid_trades) + len(trend_trades))
            
            st.markdown("### 分引擎詳情")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🧲 網格引擎**")
                grid_return = (grid_equity - 5000) / 5000 * 100
                st.metric("網格報酬", f"{grid_return:.1f}%")
                st.metric("網格交易次數", len(grid_trades))
                if len(grid_trades) > 0:
                    grid_pnl = sum(t['pnl'] for t in grid_trades)
                    st.metric("網格總盈虧", f"${grid_pnl:,.2f}")
            
            with col2:
                st.markdown("**🚀 趨勢引擎**")
                trend_return = (trend_equity - 5000) / 5000 * 100
                st.metric("趨勢報酬", f"{trend_return:.1f}%")
                st.metric("趨勢交易次數", len(trend_trades))
                if len(trend_trades) > 0:
                    trend_pnl = sum(t['pnl'] for t in trend_trades)
                    st.metric("趨勢總盈虧", f"${trend_pnl:,.2f}")
            
            # 評分
            if total_return >= 100:
                st.success("🎉 超越目標! 完美表現!")
            elif total_return >= 80:
                st.success("✅ 達成目標! 雙引擎成功!")
            elif total_return >= 50:
                st.info("📈 接近目標，再接再勵!")
            elif total_return > 0:
                st.warning("🔸 有盈利，但未達標")
            else:
                st.error("⚠️ 策略失敗")
            
            # 交易記錄
            if grid_trades or trend_trades:
                st.subheader("交易記錄樣本")
                col1, col2 = st.columns(2)
                
                with col1:
                    if grid_trades:
                        st.markdown("**網格交易 (Top 10)**")
                        grid_df = pd.DataFrame(grid_trades[-10:])
                        st.dataframe(grid_df, use_container_width=True)
                
                with col2:
                    if trend_trades:
                        st.markdown("**趨勢交易 (All)**")
                        trend_df = pd.DataFrame(trend_trades)
                        st.dataframe(trend_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"錯誤: {e}")
            import traceback
            with st.expander("詳情"): st.code(traceback.format_exc())
