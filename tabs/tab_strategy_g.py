"""
Strategy G v1.2 - Deep Q-Learning with Enhanced State & Shaped Reward

核心理念:
不預測方向，直接學習「賺錢的行為」

v1.2 革命性改進:
1. 狀態空間擴充: 10維 → 17維
   - 新增市場狀態判斷 (趨勢/震盪/波動)
   - 新增 Agent 自我認知 (勝率/連虧/回撤)
2. 分階段 Reward: 持倉過程也給反饋
   - 方向對了 → 小獎勵
   - 止損拖延 → 持續懲罰
   - 浮盈不跑 → 貪婪懲罰
   - 連虧開倉 → 風控懲罰
3. 1h/1d週期: 相容 HuggingFace 資料集
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import random
import plotly.graph_objects as go

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("PyTorch 未安裝，使用簡化版 Q-Learning")

from data.binance_loader import BinanceDataLoader


class TradingEnvV2:
    """
    v1.2 增強交易環境
    """
    def __init__(self, df, capital=10000.0, leverage=3, fee_rate=0.0006, position_size=0.3):
        self.df = df.reset_index(drop=True)
        self.initial_capital = capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.position_size = position_size
        
        self._calculate_features()
        
        self.state_dim = 17  # v1.2: 擴充到 17 維
        self.action_dim = 4
        
        # v1.2: Agent 記憶
        self.trade_history = deque(maxlen=10)  # 最近 10 筆交易
        self.peak_capital = capital
        
        self.reset()
    
    def _calculate_features(self):
        df = self.df
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(abs(df['high'] - df['close'].shift(1)),
                       abs(df['low'] - df['close'].shift(1)))
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        # EMA
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema_dist'] = (df['close'] - df['ema20']) / (df['atr'] + 1e-8)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        
        # v1.2: 新增特徵
        # ADX (趨勢強度)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr_smooth = df['tr'].rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (tr_smooth + 1e-8))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (tr_smooth + 1e-8))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['adx'] = dx.rolling(14).mean()
        
        # BB
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # ATR 百分位 (波動狀態)
        df['atr_pct'] = df['atr'] / df['close']
        df['atr_percentile'] = df['atr_pct'].rolling(50).apply(lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5)
        
        df.fillna(0, inplace=True)
        self.df = df
    
    def reset(self, start_idx=60):
        self.current_step = start_idx
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.hold_time = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.trade_history.clear()
        self.consecutive_losses = 0
        
        return self._get_state()
    
    def _get_state(self):
        row = self.df.iloc[self.current_step]
        
        # 基礎倉位狀態 (3)
        position_encoded = self.position
        hold_time_norm = min(self.hold_time / 30.0, 1.0)
        
        if self.position != 0:
            pnl_ratio = (row['close'] - self.entry_price) / self.entry_price * self.position * 100
            pnl_ratio = np.clip(pnl_ratio / 10.0, -1.0, 1.0)
        else:
            pnl_ratio = 0
        
        # 市場特徵 (7)
        rsi_norm = row['rsi'] / 100.0
        macd_hist_norm = np.clip(row['macd_hist'] / (row['atr'] + 1e-8), -2, 2) / 2.0
        ema_dist_norm = np.clip(row['ema_dist'], -3, 3) / 3.0
        volume_ratio_norm = np.clip(row['volume_ratio'], 0, 3) / 3.0
        bb_pct_norm = np.clip(row['bb_pct'], 0, 1)
        
        roc_5 = (row['close'] - self.df.iloc[max(0, self.current_step - 5)]['close']) / (self.df.iloc[max(0, self.current_step - 5)]['close'] + 1e-8) * 100
        roc_5_norm = np.clip(roc_5 / 5.0, -1, 1)
        
        # v1.2: 新增市場狀態 (3)
        trend_strength = np.clip(row['adx'] / 50.0, 0, 1)  # ADX 正規化
        price_vs_ma20 = 1 if row['close'] > row['ema20'] else -1
        volatility_regime = np.clip(row['atr_percentile'], 0, 1)
        
        # v1.2: Agent 自我認知 (4)
        recent_win_rate = 0.5
        if len(self.trade_history) >= 3:
            wins = sum(1 for t in self.trade_history if t > 0)
            recent_win_rate = wins / len(self.trade_history)
        
        consecutive_losses_norm = min(self.consecutive_losses / 5.0, 1.0)
        
        capital_usage = 0
        if self.position != 0:
            capital_usage = self.position_size  # 已用倉位比例
        
        max_dd = 0
        if self.peak_capital > 0:
            max_dd = max(0, (self.peak_capital - self.capital) / self.peak_capital)
        max_dd_norm = min(max_dd, 1.0)
        
        state = np.array([
            # 倉位狀態 (3)
            position_encoded,
            hold_time_norm,
            pnl_ratio,
            # 市場特徵 (7)
            rsi_norm,
            macd_hist_norm,
            ema_dist_norm,
            volume_ratio_norm,
            bb_pct_norm,
            roc_5_norm,
            0,  # 保留位
            # 市場狀態 (3)
            trend_strength,
            price_vs_ma20,
            volatility_regime,
            # Agent 認知 (4)
            recent_win_rate,
            consecutive_losses_norm,
            capital_usage,
            max_dd_norm,
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """
        v1.2: 分階段 Reward
        """
        row = self.df.iloc[self.current_step]
        reward = 0
        done = False
        info = {}
        
        # Action 0: 開多倉
        if action == 0 and self.position == 0:
            self.position = 1
            self.entry_price = row['close']
            self.hold_time = 0
            reward = -0.01
            
            # v1.2: 連虧後開倉重罰
            if self.consecutive_losses >= 3:
                reward -= 1.0
        
        # Action 1: 開空倉
        elif action == 1 and self.position == 0:
            self.position = -1
            self.entry_price = row['close']
            self.hold_time = 0
            reward = -0.01
            
            if self.consecutive_losses >= 3:
                reward -= 1.0
        
        # Action 2: 平倉
        elif action == 2 and self.position != 0:
            exit_price = row['close']
            pnl_pct = (exit_price - self.entry_price) / self.entry_price * self.position * 100
            
            fee = self.fee_rate * 2 * 100
            leveraged_pnl = pnl_pct * self.leverage - fee
            actual_pnl = self.capital * self.position_size * leveraged_pnl / 100
            
            self.capital += actual_pnl
            if self.capital > self.peak_capital:
                self.peak_capital = self.capital
            
            # v1.2: 改進 Reward
            if leveraged_pnl > 0:
                # 獲利加權
                base_reward = leveraged_pnl / 10.0 * 1.5
                # 獎勵穩定小贏 (1-3%)
                if 1.0 < leveraged_pnl < 3.0:
                    base_reward += 1.0
                # 懲罰過度波動 (>5%)
                elif leveraged_pnl > 5.0:
                    base_reward -= (leveraged_pnl - 5.0) * 0.2
            else:
                # 虧損重罰 (學習快速止損)
                base_reward = leveraged_pnl / 10.0 * 2.5
                # 特別懲罰大虧 (>3%)
                if leveraged_pnl < -3.0:
                    base_reward -= abs(leveraged_pnl) * 0.3
            
            reward = base_reward
            
            self.total_trades += 1
            self.trade_history.append(actual_pnl)
            
            if actual_pnl > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
            
            info = {
                'trade': True,
                'pnl': actual_pnl,
                'pnl_pct': leveraged_pnl,
                'hold_time': self.hold_time
            }
            
            self.position = 0
            self.hold_time = 0
        
        # Action 3: 持有
        else:
            if self.position != 0:
                self.hold_time += 1
                
                # v1.2: 分階段反饋
                unrealized_pnl = (row['close'] - self.entry_price) / self.entry_price * self.position * 100 * self.leverage
                
                # 1. 方向對了 → 小獎勵
                if unrealized_pnl > 0.5:
                    reward += 0.05
                
                # 2. 止損拖延 → 持續懲罰
                if unrealized_pnl < -2.0:
                    reward -= 0.5 * min(self.hold_time / 5.0, 2.0)  # 拖越久罰越重
                
                # 3. 浮盈不跑 → 貪婪懲罰
                if unrealized_pnl > 4.0 and self.hold_time > 15:
                    reward -= 0.3
                
                # 4. 基礎持倉成本
                reward -= 0.002 * min(self.hold_time / 10.0, 1.0)
            else:
                reward = 0
        
        # 檢查結束
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            if self.position != 0:
                row_final = self.df.iloc[self.current_step]
                pnl_pct = (row_final['close'] - self.entry_price) / self.entry_price * self.position * 100
                fee = self.fee_rate * 2 * 100
                leveraged_pnl = pnl_pct * self.leverage - fee
                actual_pnl = self.capital * self.position_size * leveraged_pnl / 100
                self.capital += actual_pnl
                reward += leveraged_pnl / 10.0
        
        # 爆倉檢查
        if self.capital < self.initial_capital * 0.5:
            done = True
            reward = -10
        
        next_state = self._get_state() if not done else np.zeros(self.state_dim)
        
        return next_state, reward, done, info


class DQNAgentV2:
    """
    v1.2: 擴充網路以適應 17 維狀態
    """
    def __init__(self, state_dim, action_dim, lr=0.0001):
        if not TORCH_AVAILABLE:
            raise ImportError("需要安裝 PyTorch")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.90
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.batch_size = 64
        
        # v1.2: 加深網路 (17 → 128 → 64 → 4)
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, action_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_t)
            return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([x[0] for x in minibatch])
        actions = torch.LongTensor([x[1] for x in minibatch])
        rewards = torch.FloatTensor([x[2] for x in minibatch])
        next_states = torch.FloatTensor([x[3] for x in minibatch])
        dones = torch.FloatTensor([x[4] for x in minibatch])
        
        self.model.train()
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        self.model.eval()
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        self.model.train()
        loss = self.criterion(current_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()


def train_agent(env, agent, episodes=100):
    episode_rewards = []
    episode_capitals = []
    
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            if isinstance(agent, DQNAgentV2):
                agent.replay()
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        episode_capitals.append(env.capital)
        
        if (e + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_capital = np.mean(episode_capitals[-10:])
            st.text(f"Episode {e+1}/{episodes} | Avg Reward: {avg_reward:.2f} | Avg Capital: ${avg_capital:.0f}")
    
    return episode_rewards, episode_capitals


def backtest_agent(env, agent):
    state = env.reset()
    done = False
    trades = []
    equity_curve = [env.capital]
    
    while not done:
        action = agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)
        
        if info.get('trade'):
            trades.append({
                'step': env.current_step,
                'pnl': info['pnl'],
                'hold_time': info['hold_time']
            })
        
        equity_curve.append(env.capital)
        state = next_state
    
    return trades, equity_curve, env


def render_strategy_g_tab(loader, symbol_selector):
    st.header("策略 G: 強化學習 Agent v1.2 🚀")

    with st.expander("⚡ v1.2 革命性升級", expanded=True):
        st.markdown("""
        **v1.1 問題**: 盈虧比太差 (0.72)，平均虧損 > 平均獲利
        
        **v1.2 核心創新**:
        
        1️⃣ **狀態空間擴充**: 10維 → 17維
        - 市場狀態: 趨勢強度(ADX)、價格位置、波動狀態
        - Agent 自我認知: 近期勝率、連虧次數、資金使用、最大回撤
        
        2️⃣ **分階段 Reward**: 持倉過程也給反饋
        - ✅ 方向對了 → 小獎勵 (+0.05)
        - ❌ 止損拖延 → 持續懲罰 (-0.5 * 時間)
        - ❌ 浮盈不跑 → 貪婪懲罰 (-0.3)
        - ❌ 連虧開倉 → 風控懲罰 (-1.0)
        
        3️⃣ **對稱盈虧比 Reward**:
        - 獲利加權 1.5x
        - 虧損重罰 2.5x
        - 強制學習「大贏小輸」
        
        💡 **建議**: HuggingFace 用 1h，Binance API 可用 4h
        """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_g", multi=False)
        symbol = symbol_list[0]
        train_days = st.slider("訓練天數", 60, 240, 120, key="train_g")
        test_days = st.slider("測試天數", 14, 60, 30, key="test_g")
        
        # v1.2: 根據資料源調整選項
        if isinstance(loader, BinanceDataLoader):
            timeframe = st.selectbox("時間周期", ['1h', '4h'], index=1, key="tf_g")
        else:
            timeframe = st.selectbox("時間周期", ['15m', '1h', '1d'], index=1, key="tf_g")
            st.caption("💡 HuggingFace 不支援 4h，切換到 Binance API 可用")
        
        bars_per_day = {'15m': 96, '1h': 24, '4h': 6, '1d': 1}.get(timeframe, 24)

    with col2:
        st.markdown("**RL 參數**")
        episodes = st.slider("訓練輪數", 50, 200, 100, 10, key="ep_g")
        learning_rate = st.select_slider("學習率", [0.00005, 0.0001, 0.0005, 0.001], value=0.0001, key="lr_g")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_g")
        leverage = st.slider("槓桿", 1, 10, 3, key="lev_g")
        position_size = st.slider("倉位%", 10, 80, 30, 5, key="pos_g") / 100.0
        
        st.success("✨ v1.2: 17維狀態 + 分階段Reward")

    if st.button("🚀 訓練 v1.2 Agent", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("載入數據...")
            prog.progress(10)
            total_days = train_days + test_days + 5
            
            if isinstance(loader, BinanceDataLoader):
                end = datetime.now()
                df_all = loader.load_historical_data(symbol, timeframe, end - timedelta(days=total_days), end)
            else:
                df_all = loader.load_klines(symbol, timeframe)
                df_all = df_all.tail(total_days * bars_per_day)
            
            df_all = df_all.reset_index(drop=True)
            split_idx = int(len(df_all) * (train_days / (train_days + test_days)))
            df_train = df_all.iloc[:split_idx].reset_index(drop=True)
            df_test = df_all.iloc[split_idx:].reset_index(drop=True)
            
            st.info(f"訓練: {len(df_train)} 根 | 測試: {len(df_test)} 根")
            prog.progress(20)
            
            stat.text("初始化 v1.2 環境...")
            train_env = TradingEnvV2(df_train, capital, leverage, position_size=position_size)
            test_env = TradingEnvV2(df_test, capital, leverage, position_size=position_size)
            prog.progress(25)
            
            stat.text("創建 DQN v1.2 Agent...")
            if TORCH_AVAILABLE:
                agent = DQNAgentV2(train_env.state_dim, train_env.action_dim, lr=learning_rate)
            else:
                st.error("v1.2 需要 PyTorch，請安裝: pip install torch")
                return
            prog.progress(30)
            
            stat.text(f"訓練中 ({episodes} 輪)...")
            episode_rewards, episode_capitals = train_agent(train_env, agent, episodes)
            prog.progress(70)
            
            st.markdown("### 訓練過程")
            fig_train = go.Figure()
            fig_train.add_trace(go.Scatter(y=episode_capitals, mode='lines', name='權益'))
            fig_train.add_hline(y=capital, line_dash="dash", line_color="gray", annotation_text="初始資金")
            fig_train.update_layout(title="訓練輪權益變化", xaxis_title="Episode", yaxis_title="Capital ($)")
            st.plotly_chart(fig_train, use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("最終訓練權益", f"${episode_capitals[-1]:,.0f}")
            c2.metric("平均 Reward", f"{np.mean(episode_rewards[-10:]):.2f}")
            train_return = (episode_capitals[-1] - capital) / capital * 100
            c3.metric("訓練報酬", f"{train_return:.1f}%")
            
            stat.text("回測...")
            prog.progress(80)
            trades, equity_curve, final_env = backtest_agent(test_env, agent)
            prog.progress(100)
            stat.text("完成")
            
            st.markdown("### 回測結果")
            final_capital = equity_curve[-1]
            total_return = (final_capital - capital) / capital * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("最終權益", f"${final_capital:,.0f}", f"{final_capital - capital:+,.0f}")
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
            
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(y=equity_curve, mode='lines', name='權益', line=dict(color='blue')))
            fig_equity.add_hline(y=capital, line_dash="dash", line_color="gray", annotation_text="初始資金")
            fig_equity.update_layout(title="權益曲線", xaxis_title="Steps", yaxis_title="Capital ($)")
            st.plotly_chart(fig_equity, use_container_width=True)
            
            if trades:
                st.subheader("交易記錄")
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df.tail(20), use_container_width=True)
                
                # 過擬合檢查
                overfitting_ratio = episode_capitals[-1] / max(final_capital, 1)
                if overfitting_ratio > 5:
                    st.warning(f"⚠️ 過擬合風險: {overfitting_ratio:.1f}x")
                elif overfitting_ratio > 2:
                    st.info(f"ℹ️ 輕微過擬合: {overfitting_ratio:.1f}x")
                else:
                    st.success(f"✅ 泛化良好: {overfitting_ratio:.1f}x")
                    
                # v1.2: 盈虧比檢查
                if len(trades) > 10:
                    if profit_factor > 1.2:
                        st.success(f"✅ 盈虧比優秀: {profit_factor:.2f} (目標 >1.2)")
                    elif profit_factor > 0.8:
                        st.info(f"ℹ️ 盈虧比可接受: {profit_factor:.2f}")
                    else:
                        st.warning(f"⚠️ 盈虧比偏低: {profit_factor:.2f} (需改進)")
        
        except Exception as e:
            st.error(f"錯誤: {e}")
            import traceback
            with st.expander("詳情"): st.code(traceback.format_exc())
