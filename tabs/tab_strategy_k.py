"""
Strategy K - RL Agent 激進改造版
Aggressive RL Agent Remake

目標: 30天 +100-150% (最高潛力)

改進點:
1. 槓桿: 5x → 10x
2. Reward 函數: 改為「日報酬率」
3. 允許多單重疊 (金字塔加倉)
4. 最大倉位: 200% (10x * 2倉)
5. 訓練目標: 不是「賺錢」而是「快速賺錢」

風險:
- 可能過擬合
- 可能爆倉 (-100%)
- 不可預測 (黑盒)
- 最高潛力 (+150%+)
"""

import streamlit as st
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import plotly.graph_objects as go

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from strategies.multi_timeframe import MultiTimeframeLoader


class AggressiveTradingEnv(gym.Env):
    """
    激進交易環境 - 金字塔加倉
    """
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000, leverage: int = 10):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.max_positions = 2  # 允許 2 個倉位重疊
        
        # Action: [hold, long1, long2, short1, short2, close_all]
        self.action_space = spaces.Discrete(6)
        
        # Observation: [price, indicators, positions]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 50  # 需要指標
        self.balance = self.initial_balance
        self.positions = []  # [{direction, entry_price, size}]
        self.trades = []
        self.initial_step_balance = self.balance
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        
        # 計算指標
        close_prices = self.df['close'].iloc[self.current_step-50:self.current_step+1]
        ema8 = close_prices.ewm(span=8).mean().iloc[-1]
        ema20 = close_prices.ewm(span=20).mean().iloc[-1]
        ema50 = close_prices.ewm(span=50).mean().iloc[-1]
        
        rsi_period = 14
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / (loss + 1e-8)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # 持倉狀態
        total_position_size = sum(p['size'] for p in self.positions)
        net_position = sum(p['direction'] * p['size'] for p in self.positions)  # 正=多, 負=空
        
        obs = np.array([
            row['close'] / 100000,  # 正規化價格
            (row['close'] - ema8) / row['close'],
            (row['close'] - ema20) / row['close'],
            (row['close'] - ema50) / row['close'],
            rsi / 100,
            row['volume'] / row['volume'],  # 正規化量
            self.balance / self.initial_balance,
            total_position_size,
            net_position,
            len(self.positions) / self.max_positions,
            (row['high'] - row['low']) / row['close'],  # ATR proxy
            row['close'] / row['open'] - 1,  # 當根K棒漲跌
            (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5],  # 5根動量
            (close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10],  # 10根動量
            (close_prices.iloc[-1] - close_prices.iloc[-20]) / close_prices.iloc[-20]   # 20根動量
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        fee = 0.0006
        position_size = 0.5  # 每個倉位 50%
        
        reward = 0
        
        # Action: 0=hold, 1=long1, 2=long2, 3=short1, 4=short2, 5=close_all
        if action == 1 and len(self.positions) < self.max_positions:  # Long 1個倉
            self.positions.append({
                'direction': 1,
                'entry_price': current_price,
                'size': position_size
            })
        
        elif action == 2 and len(self.positions) < self.max_positions:  # Long 加倉
            self.positions.append({
                'direction': 1,
                'entry_price': current_price,
                'size': position_size
            })
        
        elif action == 3 and len(self.positions) < self.max_positions:  # Short 1個倉
            self.positions.append({
                'direction': -1,
                'entry_price': current_price,
                'size': position_size
            })
        
        elif action == 4 and len(self.positions) < self.max_positions:  # Short 加倉
            self.positions.append({
                'direction': -1,
                'entry_price': current_price,
                'size': position_size
            })
        
        elif action == 5 and len(self.positions) > 0:  # 平所有倉
            for pos in self.positions:
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * pos['direction']
                pnl = self.initial_balance * pos['size'] * (pnl_pct * self.leverage - fee * 2)
                self.balance += pnl
                reward += pnl
                
                self.trades.append({
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'direction': pos['direction'],
                    'pnl': pnl
                })
            
            self.positions = []
        
        # 移動到下一步
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Reward 設計: 日報酬率 (鼓勵快速賺錢)
        step_return = (self.balance - self.initial_step_balance) / self.initial_step_balance * 100
        reward = step_return * 10  # 放大 reward
        self.initial_step_balance = self.balance
        
        # 爆倉懲罰
        if self.balance < self.initial_balance * 0.5:
            reward = -1000
            done = True
        
        return self._get_observation(), reward, done, False, {}
    
    def render(self):
        pass


def render_strategy_k_tab(loader, symbol_selector):
    st.header("策略 K: RL Agent 激進版 🤖🔥")

    with st.expander("⚠️ 極高風險警告", expanded=True):
        st.markdown("""
        **目標**: 30天 +100-150% (最高潛力)
        
        🤖 **RL Agent 改造**:
        - 10x 槓桿 (放大2倍)
        - 允許多倉重疊 (金字塔加倉)
        - 最大倉位 200% (2個 100%倉)
        - Reward = 日報酬率 (鼓勵快速賺錢)
        
        💡 **訓練目標**:
        - 不是「賺錢」
        - 而是「快速賺錢」
        - Agent 學會激進加倉
        
        ⚠️ **極高風險**:
        - 可能爆倉 (-100%)
        - 可能過擬合
        - 黑盒，不可預測
        - 但有最高潛力 (+150%+)
        """)

    if not SB3_AVAILABLE:
        st.error("需要安裝 stable-baselines3: pip install stable-baselines3")
        return

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**數據設定**")
        symbol_list = symbol_selector("strategy_k", multi=False)
        symbol = symbol_list[0]
        train_steps = st.slider("訓練步數", 10000, 100000, 50000, 10000, key="train_k")

    with col2:
        st.markdown("**固定參數**")
        st.metric("資金", "$10,000")
        st.metric("槓桿", "10x 🔥")
        st.metric("最大倉位", "200%")
        st.metric("Reward", "日報酬率")

    if st.button("🤖 訓練激進 Agent", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("載入數據...")
            prog.progress(10)
            
            mtf_loader = MultiTimeframeLoader(loader)
            df_15m, df_1h, df_1d = mtf_loader.load_multi_timeframe(symbol, 120)
            
            # 準備訓練數據
            df_train = df_1h.iloc[:int(len(df_1h)*0.75)].copy()
            df_test = df_1h.iloc[int(len(df_1h)*0.75):].copy()
            
            stat.text("建立激進環境...")
            prog.progress(20)
            
            env = DummyVecEnv([lambda: AggressiveTradingEnv(df_train, leverage=10)])
            
            stat.text(f"訓練 Agent ({train_steps} steps)...")
            prog.progress(30)
            
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=0.0005,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.95,  # 更重視近期 reward
                verbose=0
            )
            
            # 訓練
            for i in range(0, train_steps, train_steps//5):
                model.learn(total_timesteps=train_steps//5, reset_num_timesteps=False)
                prog.progress(30 + int(50 * (i+train_steps//5) / train_steps))
            
            stat.text("測試 Agent...")
            prog.progress(85)
            
            # 測試
            test_env = AggressiveTradingEnv(df_test, leverage=10)
            obs, _ = test_env.reset()
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                if done or truncated:
                    break
            
            prog.progress(100)
            stat.text("完成")
            
            # 結果
            final_balance = test_env.balance
            total_return = (final_balance - 10000) / 10000 * 100
            trades = test_env.trades
            
            st.markdown("### Agent 表現")
            c1, c2, c3 = st.columns(3)
            c1.metric("最終權益", f"${final_balance:,.0f}", f"{final_balance - 10000:+,.0f}")
            c2.metric("總報酬", f"{total_return:.1f}%",
                     "🎉" if total_return >= 100 else ("🔥" if total_return >= 50 else "📈"))
            c3.metric("交易次數", len(trades))
            
            if len(trades) > 0:
                wins = [t for t in trades if t['pnl'] > 0]
                losses = [t for t in trades if t['pnl'] <= 0]
                win_rate = len(wins) / len(trades) * 100
                avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
                avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric("勝率", f"{win_rate:.1f}%")
                c2.metric("平均獲利", f"${avg_win:.2f}")
                c3.metric("平均虧損", f"${avg_loss:.2f}")
            
            # 評分
            if total_return >= 150:
                st.success("🚀 神級表現! 超過 150%!")
            elif total_return >= 100:
                st.success("🎉 完美達標! +100%!")
            elif total_return >= 50:
                st.info("🔥 接近目標! +50%+")
            elif total_return > 0:
                st.warning("📈 有盈利但未達標")
            elif total_return > -50:
                st.warning("⚠️ 小幅虧損")
            else:
                st.error("💥 大幅虧損/爆倉")
            
            # 交易記錄
            if trades:
                st.subheader("Agent 交易記錄")
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df.tail(20), use_container_width=True)
        
        except Exception as e:
            st.error(f"錯誤: {e}")
            import traceback
            with st.expander("詳情"): st.code(traceback.format_exc())
