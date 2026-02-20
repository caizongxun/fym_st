"""
Strategy G v1.1 - Deep Q-Learning Trading Agent (Anti-Overfitting)

核心理念:
不預測方向，直接學習「賺錢的行為」

v1.1 改進:
- 方案 2: 神經網路加入 Dropout (0.2) 降低過擬合
- 方案 3: Reward 設計改進，懲罰高波動交易
- 降低 gamma (0.95 → 0.90) 減少對未來的過度重視
- 慢速探索衰減 (0.995 → 0.99)
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


class TradingEnv:
    """
    模擬交易環境（符合 Gym 介面）
    """
    def __init__(self, df, capital=10000.0, leverage=3, fee_rate=0.0006, position_size=0.3):
        self.df = df.reset_index(drop=True)
        self.initial_capital = capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.position_size = position_size
        
        # 計算指標
        self._calculate_features()
        
        # 狀態維度
        self.state_dim = 10
        self.action_dim = 4
        
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
        
        # ATR變化
        df['atr_change'] = df['atr'].pct_change(5)
        
        df.fillna(0, inplace=True)
        self.df = df
    
    def reset(self, start_idx=60):
        self.current_step = start_idx
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.hold_time = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        return self._get_state()
    
    def _get_state(self):
        row = self.df.iloc[self.current_step]
        
        position_encoded = self.position
        hold_time_norm = min(self.hold_time / 30.0, 1.0)
        
        if self.position != 0:
            pnl_ratio = (row['close'] - self.entry_price) / self.entry_price * self.position * 100
            pnl_ratio = np.clip(pnl_ratio / 10.0, -1.0, 1.0)
        else:
            pnl_ratio = 0
        
        rsi_norm = row['rsi'] / 100.0
        macd_hist_norm = np.clip(row['macd_hist'] / row['atr'], -2, 2) / 2.0
        ema_dist_norm = np.clip(row['ema_dist'], -3, 3) / 3.0
        volume_ratio_norm = np.clip(row['volume_ratio'], 0, 3) / 3.0
        atr_change_norm = np.clip(row['atr_change'], -0.5, 0.5) * 2
        
        roc_5 = (row['close'] - self.df.iloc[self.current_step - 5]['close']) / self.df.iloc[self.current_step - 5]['close'] * 100
        roc_5_norm = np.clip(roc_5 / 5.0, -1, 1)
        
        state = np.array([
            position_encoded,
            hold_time_norm,
            pnl_ratio,
            rsi_norm,
            macd_hist_norm,
            ema_dist_norm,
            volume_ratio_norm,
            atr_change_norm,
            roc_5_norm,
            0
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """
        執行動作
        action: 0=開多, 1=開空, 2=平倉, 3=持有
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
        
        # Action 1: 開空倉
        elif action == 1 and self.position == 0:
            self.position = -1
            self.entry_price = row['close']
            self.hold_time = 0
            reward = -0.01
        
        # Action 2: 平倉
        elif action == 2 and self.position != 0:
            exit_price = row['close']
            pnl_pct = (exit_price - self.entry_price) / self.entry_price * self.position * 100
            
            fee = self.fee_rate * 2 * 100
            leveraged_pnl = pnl_pct * self.leverage - fee
            actual_pnl = self.capital * self.position_size * leveraged_pnl / 100
            
            self.capital += actual_pnl
            
            # 方案 3: Reward 設計改進
            base_reward = leveraged_pnl / 10.0
            
            # 懲罰高波動交易（鼓勵穩定獲利）
            if abs(leveraged_pnl) > 5.0:  # 盈虧 > 5%
                volatility_penalty = abs(leveraged_pnl) * 0.1  # 波動懲罰
                base_reward -= volatility_penalty
            
            # 獎勵穩定小贏（1-3%）
            if 1.0 < leveraged_pnl < 3.0:
                base_reward += 0.5
            
            reward = base_reward
            
            self.total_trades += 1
            if actual_pnl > 0:
                self.winning_trades += 1
            
            info = {
                'trade': True,
                'pnl': actual_pnl,
                'pnl_pct': leveraged_pnl,
                'hold_time': self.hold_time
            }
            
            self.position = 0
            self.hold_time = 0
        
        # Action 3 或其他: 持有
        else:
            if self.position != 0:
                self.hold_time += 1
                reward = -0.001 * min(self.hold_time / 10.0, 1.0)
                
                # 方案 3: 浮動盈虧過大懲罰加強
                unrealized_pnl = (row['close'] - self.entry_price) / self.entry_price * self.position * 100 * self.leverage
                if abs(unrealized_pnl) > 5.0:  # 浮盈/虧 > 5%
                    reward -= 1.0  # 鼓勵及時平倉
                elif unrealized_pnl < -3.0:  # 虧損 > 3%
                    reward -= 0.8  # 加強止損
            else:
                reward = 0
        
        # 檢查是否結束
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


class DQNAgent:
    """
    DQN Agent with Dropout (v1.1)
    """
    def __init__(self, state_dim, action_dim, lr=0.001):
        if not TORCH_AVAILABLE:
            raise ImportError("需要安裝 PyTorch")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.90  # v1.1: 降低對未來的重視
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # v1.1: 慢速衰減
        self.batch_size = 64
        
        # 方案 2: 神經網路加入 Dropout
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # v1.1: 新增
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # v1.1: 新增
            nn.Linear(64, action_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        # v1.1: 訓練時啟用 Dropout，推理時關閉
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
        
        self.model.train()  # 訓練模式
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        self.model.eval()  # 推理模式（計算 target）
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


class SimpleQLearningAgent:
    """
    簡化版 Q-Learning（不需要 PyTorch）
    """
    def __init__(self, state_dim, action_dim, lr=0.1):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # v1.1: 與 DQN 一致
        
        self.q_table = {}
    
    def _discretize_state(self, state):
        discrete = tuple(np.clip(np.round(state * 2), -2, 2).astype(int))
        return discrete
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        s = self._discretize_state(state)
        if s not in self.q_table:
            self.q_table[s] = np.zeros(self.action_dim)
        
        return np.argmax(self.q_table[s])
    
    def remember(self, state, action, reward, next_state, done):
        s = self._discretize_state(state)
        s_next = self._discretize_state(next_state)
        
        if s not in self.q_table:
            self.q_table[s] = np.zeros(self.action_dim)
        if s_next not in self.q_table:
            self.q_table[s_next] = np.zeros(self.action_dim)
        
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[s_next])
        
        self.q_table[s][action] += self.lr * (target - self.q_table[s][action])
    
    def replay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return 0


def train_agent(env, agent, episodes=50):
    """
    訓練 RL Agent
    """
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
            
            if isinstance(agent, DQNAgent):
                agent.replay()
            
            state = next_state
            total_reward += reward
        
        if isinstance(agent, SimpleQLearningAgent):
            agent.replay()
        
        episode_rewards.append(total_reward)
        episode_capitals.append(env.capital)
        
        if (e + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_capital = np.mean(episode_capitals[-10:])
            st.text(f"Episode {e+1}/{episodes} | Avg Reward: {avg_reward:.2f} | Avg Capital: ${avg_capital:.0f}")
    
    return episode_rewards, episode_capitals


def backtest_agent(env, agent):
    """
    用訓練好的 Agent 回測
    """
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
    st.header("策略 G: 強化學習 Agent v1.1")

    with st.expander("ℹ️ v1.1 抗過擬合改進", expanded=True):
        st.markdown("""
        **v1.0 問題**: 訓練權益 $70k vs 回測 $1k（過擬合）
        
        **v1.1 解決方案**:
        - 方案 2: 神經網路加入 Dropout (0.2)
        - 方案 3: Reward 設計改進
          - 懲罰高波動交易 (>5%)
          - 獎勵穩定小贏 (1-3%)
          - 加強浮動盈虧平倉懲罰
        - 降低 gamma (0.95 → 0.90)
        - 慢速探索衰減 (0.995 → 0.99)
        
        **預期效果**:
        - 訓練權益更穩定（不再爆張）
        - 回測效果接近訓練
        - 交易更謹慎（高品質信號）
        """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_g", multi=False)
        symbol = symbol_list[0]
        train_days = st.slider("訓練天數", 60, 180, 120, key="train_g")
        test_days = st.slider("測試天數", 14, 60, 30, key="test_g")
        timeframe = st.selectbox("時間周期", ['15m', '1h', '4h'], index=1, key="tf_g")
        bars_per_day = {'15m': 96, '1h': 24, '4h': 6}[timeframe]

    with col2:
        st.markdown("**RL 參數**")
        episodes = st.slider("訓練輪數", 20, 200, 100, 10, key="ep_g")  # v1.1: 預設 100
        learning_rate = st.select_slider("學習率", [0.0001, 0.001, 0.01, 0.1], value=0.0001, key="lr_g")  # v1.1: 預設 0.0001
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_g")
        leverage = st.slider("槓桿", 1, 10, 3, key="lev_g")
        position_size = st.slider("倉位%", 10, 80, 30, 5, key="pos_g") / 100.0
        
        agent_type = "DQN (w/ Dropout)" if TORCH_AVAILABLE else "Q-Learning"
        st.info(f"使用算法: {agent_type}")

    if st.button("訓練 RL Agent v1.1", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            # 載入數據
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
            
            # 創建環境
            stat.text("初始化環境...")
            train_env = TradingEnv(df_train, capital, leverage, position_size=position_size)
            test_env = TradingEnv(df_test, capital, leverage, position_size=position_size)
            prog.progress(25)
            
            # 創建 Agent
            stat.text(f"創建 {agent_type} Agent...")
            if TORCH_AVAILABLE:
                agent = DQNAgent(train_env.state_dim, train_env.action_dim, lr=learning_rate)
            else:
                agent = SimpleQLearningAgent(train_env.state_dim, train_env.action_dim, lr=learning_rate)
            prog.progress(30)
            
            # 訓練
            stat.text(f"訓練中 ({episodes} 輪)...")
            episode_rewards, episode_capitals = train_agent(train_env, agent, episodes)
            prog.progress(70)
            
            # 訓練結果
            st.markdown("### 訓練過程")
            fig_train = go.Figure()
            fig_train.add_trace(go.Scatter(y=episode_capitals, mode='lines', name='權益'))
            fig_train.add_hline(y=capital, line_dash="dash", line_color="gray", annotation_text="初始資金")
            fig_train.update_layout(title="訓練輪權益變化", xaxis_title="Episode", yaxis_title="Capital ($)")
            st.plotly_chart(fig_train, use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("最終訓練權益", f"${episode_capitals[-1]:,.0f}")
            c2.metric("平均 Reward", f"{np.mean(episode_rewards[-10:]):.2f}")
            train_vs_init = (episode_capitals[-1] - capital) / capital * 100
            c3.metric("訓練報酬", f"{train_vs_init:.1f}%")
            
            # 回測
            stat.text("回測...")
            prog.progress(80)
            trades, equity_curve, final_env = backtest_agent(test_env, agent)
            prog.progress(100)
            stat.text("完成")
            
            # 回測結果
            st.markdown("### 回測結果")
            final_capital = equity_curve[-1]
            total_return = (final_capital - capital) / capital * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("最終權益", f"${final_capital:,.0f}", f"{final_capital - capital:+,.0f}")
            c2.metric("總報酬", f"{total_return:.1f}%")
            c3.metric("交易次數", len(trades))
            
            if len(trades) > 0:
                wins = [t for t in trades if t['pnl'] > 0]
                win_rate = len(wins) / len(trades) * 100
                avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
                avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if len(trades) > len(wins) else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric("勝率", f"{win_rate:.1f}%")
                c2.metric("平均獲利", f"${avg_win:.2f}")
                c3.metric("平均虧損", f"${avg_loss:.2f}")
            
            # 權益曲線
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(y=equity_curve, mode='lines', name='權益', line=dict(color='blue')))
            fig_equity.add_hline(y=capital, line_dash="dash", line_color="gray", annotation_text="初始資金")
            fig_equity.update_layout(title="權益曲線", xaxis_title="Steps", yaxis_title="Capital ($)")
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # 交易明細
            if trades:
                st.subheader("交易記錄")
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df.tail(20), use_container_width=True)
                
                # v1.1: 過擬合檢查
                overfitting_ratio = episode_capitals[-1] / max(final_capital, 1)
                if overfitting_ratio > 5:
                    st.warning(f"⚠️ 過擬合風險: 訓練權益 / 回測權益 = {overfitting_ratio:.1f}x")
                elif overfitting_ratio > 2:
                    st.info(f"ℹ️ 輕微過擬合: 比率 = {overfitting_ratio:.1f}x（可接受）")
                else:
                    st.success(f"✅ 泛化良好: 比率 = {overfitting_ratio:.1f}x")
        
        except Exception as e:
            st.error(f"錯誤: {e}")
            import traceback
            with st.expander("詳情"): st.code(traceback.format_exc())
