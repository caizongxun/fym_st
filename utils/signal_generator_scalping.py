import pandas as pd
import numpy as np
import ta
from typing import Dict, Optional

class UltraScalpingSignalGenerator:
    """
    超高頻剩頭皮策略信號生成器
    
    策略原理:
    - 捕捉微小價格波動 (0.05-0.15%)
    - 快進快出,持倉時間 < 30分鐘
    - 目標: 每筆賺 0.1-0.2 USDT (含手續費)
    - 適用: 低波動/震盪市場
    
    信號條件:
    1. BB寬度縮小 (低波動)
    2. 價格觸及BB上/下軌
    3. 成交量激增 (確認反轉)
    4. 快速止盈/止損
    """
    
    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 volume_multiplier: float = 1.5,
                 quick_tp_pct: float = 0.15,
                 quick_sl_pct: float = 0.1,
                 min_bb_width: float = 0.01,
                 max_bb_width: float = 0.03):
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_multiplier = volume_multiplier
        self.quick_tp_pct = quick_tp_pct
        self.quick_sl_pct = quick_sl_pct
        self.min_bb_width = min_bb_width
        self.max_bb_width = max_bb_width
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(
            close=df['close'],
            window=self.bb_period,
            window_dev=self.bb_std
        )
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # BB位置
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # 成交量指標
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 價格動量
        df['price_change'] = df['close'].pct_change()
        df['price_volatility'] = df['price_change'].rolling(window=20).std()
        
        # RSI (輔助)
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        
        df = df.ffill().bfill()
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成超高頻剩頭皮信號
        
        逻輯:
        - 低波動 + BB邊界 + 成交量確認 = 入場
        - 賺到 0.1-0.2% 即離場
        """
        df = self.add_indicators(df)
        
        df['signal'] = 0
        df['signal_strength'] = 0.0
        df['quick_tp_price'] = 0.0
        df['quick_sl_price'] = 0.0
        
        # 做多條件: BB下軌 + 低波動 + 成交量確認
        long_conditions = (
            # 1. BB寬度在合理範圍 (不太窄不太寬)
            (df['bb_width'] > self.min_bb_width) &
            (df['bb_width'] < self.max_bb_width) &
            
            # 2. 價格接近下軌 (BB position < 0.15)
            (df['bb_position'] < 0.15) &
            
            # 3. 成交量激增 (確認反轉動能)
            (df['volume_ratio'] > self.volume_multiplier) &
            
            # 4. 價格開始回升 (前一根K棒下跌,當前K棒上漲)
            (df['price_change'].shift(1) < 0) &
            (df['price_change'] > 0)
        )
        
        # 做空條件: BB上軌 + 低波動 + 成交量確認
        short_conditions = (
            # 1. BB寬度在合理範圏
            (df['bb_width'] > self.min_bb_width) &
            (df['bb_width'] < self.max_bb_width) &
            
            # 2. 價格接近上軌 (BB position > 0.85)
            (df['bb_position'] > 0.85) &
            
            # 3. 成交量激增
            (df['volume_ratio'] > self.volume_multiplier) &
            
            # 4. 價格開始回落
            (df['price_change'].shift(1) > 0) &
            (df['price_change'] < 0)
        )
        
        # 設置信號
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        
        # 計算快速止盈止損價格
        df.loc[long_conditions, 'quick_tp_price'] = df.loc[long_conditions, 'close'] * (1 + self.quick_tp_pct / 100)
        df.loc[long_conditions, 'quick_sl_price'] = df.loc[long_conditions, 'close'] * (1 - self.quick_sl_pct / 100)
        
        df.loc[short_conditions, 'quick_tp_price'] = df.loc[short_conditions, 'close'] * (1 - self.quick_tp_pct / 100)
        df.loc[short_conditions, 'quick_sl_price'] = df.loc[short_conditions, 'close'] * (1 + self.quick_sl_pct / 100)
        
        # 信號強度 = BB位置 + 成交量 + 波動率
        df.loc[long_conditions, 'signal_strength'] = (
            (0.15 - df.loc[long_conditions, 'bb_position']) / 0.15 * 0.4 +
            np.clip((df.loc[long_conditions, 'volume_ratio'] - 1) / 2, 0, 1) * 0.3 +
            (1 - np.clip(df.loc[long_conditions, 'bb_width'] / self.max_bb_width, 0, 1)) * 0.3
        )
        
        df.loc[short_conditions, 'signal_strength'] = (
            (df.loc[short_conditions, 'bb_position'] - 0.85) / 0.15 * 0.4 +
            np.clip((df.loc[short_conditions, 'volume_ratio'] - 1) / 2, 0, 1) * 0.3 +
            (1 - np.clip(df.loc[short_conditions, 'bb_width'] / self.max_bb_width, 0, 1)) * 0.3
        )
        
        df['signal_strength'] = np.clip(df['signal_strength'], 0, 1)
        
        return df
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        signals = df[df['signal'] != 0]
        
        return {
            'total_signals': len(signals),
            'long_signals': len(signals[signals['signal'] == 1]),
            'short_signals': len(signals[signals['signal'] == -1]),
            'avg_signal_strength': signals['signal_strength'].mean() if len(signals) > 0 else 0,
            'signal_frequency': len(signals) / len(df) * 100,
            'avg_bb_width': df['bb_width'].mean(),
            'avg_volume_ratio': df['volume_ratio'].mean()
        }
    
    def calculate_expected_pnl(self, entry_price: float, leverage: float = 10, position_value: float = 50) -> Dict:
        """計算預期PNL (考慮手續費)"""
        maker_fee = 0.0002
        taker_fee = 0.0006
        
        # 進場手續費 (taker)
        entry_fee = position_value * taker_fee
        
        # 止盈離場 (maker, 使用limit order)
        tp_price = entry_price * (1 + self.quick_tp_pct / 100)
        tp_pnl = (tp_price - entry_price) / entry_price * position_value
        tp_fee = position_value * maker_fee
        tp_net = tp_pnl - entry_fee - tp_fee
        
        # 止損離場 (taker)
        sl_price = entry_price * (1 - self.quick_sl_pct / 100)
        sl_pnl = (sl_price - entry_price) / entry_price * position_value
        sl_fee = position_value * taker_fee
        sl_net = sl_pnl - entry_fee - sl_fee
        
        return {
            'entry_price': entry_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'tp_net_pnl': tp_net,
            'sl_net_pnl': sl_net,
            'risk_reward': abs(tp_net / sl_net) if sl_net != 0 else 0
        }


if __name__ == '__main__':
    print("超高頻剩頭皮策略測試")
    print("="*50)
    
    dates = pd.date_range('2024-01-01', periods=2000, freq='15min')
    np.random.seed(42)
    
    base_price = 50000
    prices = base_price + np.random.randn(2000).cumsum() * 50
    
    df = pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': prices + np.random.rand(2000) * 30,
        'low': prices - np.random.rand(2000) * 30,
        'close': prices + np.random.randn(2000) * 10,
        'volume': np.random.randint(100, 2000, 2000)
    })
    
    # 測試不同參數
    for tp_pct, sl_pct in [(0.1, 0.08), (0.15, 0.1), (0.2, 0.12)]:
        print(f"\n測試參數: TP={tp_pct}%, SL={sl_pct}%")
        
        generator = UltraScalpingSignalGenerator(
            quick_tp_pct=tp_pct,
            quick_sl_pct=sl_pct
        )
        
        df_signals = generator.generate_signals(df)
        summary = generator.get_signal_summary(df_signals)
        
        print(f"  信號數: {summary['total_signals']} (多:{summary['long_signals']}, 空:{summary['short_signals']})")
        print(f"  信號頻率: {summary['signal_frequency']:.2f}%")
        
        if summary['total_signals'] > 0:
            first_signal = df_signals[df_signals['signal'] != 0].iloc[0]
            pnl_calc = generator.calculate_expected_pnl(first_signal['close'], leverage=10, position_value=50)
            print(f"  預期TP PNL: {pnl_calc['tp_net_pnl']:.3f} USDT")
            print(f"  預期SL PNL: {pnl_calc['sl_net_pnl']:.3f} USDT")
            print(f"  風報比: {pnl_calc['risk_reward']:.2f}")