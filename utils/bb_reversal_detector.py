import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

class BBReversalDetector:
    """
    BB通道觸碰反轉點檢測器
    
    定義有效觸碰反轉:
    1. 價格觸碰或突破上軌/下軌
    2. 隨後N根K線內出現反向運動
    3. 反向運動幅度達到閾值
    """
    
    def __init__(self, 
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 touch_threshold: float = 0.001,  # 觸碰閾值 0.1%
                 reversal_confirm_candles: int = 3,  # 確認反轉的K線數
                 min_reversal_pct: float = 0.003):   # 最小反轉幅度 0.3%
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.touch_threshold = touch_threshold
        self.reversal_confirm_candles = reversal_confirm_candles
        self.min_reversal_pct = min_reversal_pct
    
    def calculate_bb(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        bb = ta.volatility.BollingerBands(
            df['close'], 
            window=self.bb_period, 
            window_dev=self.bb_std
        )
        
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def detect_touch_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        檢測觸碰點
        """
        df = df.copy()
        
        # 計算價格與上下軌的距離比例
        df['dist_to_upper'] = (df['bb_upper'] - df['high']) / df['bb_upper']
        df['dist_to_lower'] = (df['low'] - df['bb_lower']) / df['bb_lower']
        
        # 標記觸碰點
        df['touch_upper'] = df['dist_to_upper'] <= self.touch_threshold
        df['touch_lower'] = df['dist_to_lower'] <= self.touch_threshold
        
        # 標記突破點 (收盤價在軌道外)
        df['break_upper'] = df['close'] > df['bb_upper']
        df['break_lower'] = df['close'] < df['bb_lower']
        
        return df
    
    def check_reversal(self, df: pd.DataFrame, idx: int, touch_type: str) -> dict:
        """
        檢查在idx位置觸碰後是否發生反轉
        
        touch_type: 'upper' 或 'lower'
        """
        if idx >= len(df) - self.reversal_confirm_candles:
            return {'is_reversal': False}
        
        touch_price = df.iloc[idx]['close']
        
        # 檢查後續N根K線
        future_slice = df.iloc[idx+1:idx+1+self.reversal_confirm_candles]
        
        if touch_type == 'upper':
            # 觸碰上軌後,應該向下反轉
            # 找最低點
            min_price = future_slice['low'].min()
            reversal_pct = (touch_price - min_price) / touch_price
            
            # 檢查是否有效反轉
            if reversal_pct >= self.min_reversal_pct:
                return {
                    'is_reversal': True,
                    'reversal_type': 'down',
                    'reversal_pct': reversal_pct,
                    'touch_price': touch_price,
                    'target_price': min_price,
                    'reversal_candles': len(future_slice[future_slice['low'] == min_price])
                }
        
        elif touch_type == 'lower':
            # 觸碰下軌後,應該向上反轉
            # 找最高點
            max_price = future_slice['high'].max()
            reversal_pct = (max_price - touch_price) / touch_price
            
            # 檢查是否有效反轉
            if reversal_pct >= self.min_reversal_pct:
                return {
                    'is_reversal': True,
                    'reversal_type': 'up',
                    'reversal_pct': reversal_pct,
                    'touch_price': touch_price,
                    'target_price': max_price,
                    'reversal_candles': len(future_slice[future_slice['high'] == max_price])
                }
        
        return {'is_reversal': False}
    
    def detect_reversals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        檢測所有有效的反轉點
        """
        df = self.calculate_bb(df)
        df = self.detect_touch_points(df)
        
        # 初始化反轉標記
        df['reversal_point'] = ''
        df['reversal_valid'] = False
        df['reversal_pct'] = 0.0
        
        reversals = []
        
        for idx in range(len(df) - self.reversal_confirm_candles):
            row = df.iloc[idx]
            
            # 檢查上軌觸碰
            if row['touch_upper'] or row['break_upper']:
                reversal_info = self.check_reversal(df, idx, 'upper')
                if reversal_info['is_reversal']:
                    df.loc[df.index[idx], 'reversal_point'] = 'upper'
                    df.loc[df.index[idx], 'reversal_valid'] = True
                    df.loc[df.index[idx], 'reversal_pct'] = reversal_info['reversal_pct']
                    reversals.append({
                        'index': idx,
                        'time': row.get('open_time', idx),
                        'type': 'upper',
                        **reversal_info
                    })
            
            # 檢查下軌觸碰
            if row['touch_lower'] or row['break_lower']:
                reversal_info = self.check_reversal(df, idx, 'lower')
                if reversal_info['is_reversal']:
                    df.loc[df.index[idx], 'reversal_point'] = 'lower'
                    df.loc[df.index[idx], 'reversal_valid'] = True
                    df.loc[df.index[idx], 'reversal_pct'] = reversal_info['reversal_pct']
                    reversals.append({
                        'index': idx,
                        'time': row.get('open_time', idx),
                        'type': 'lower',
                        **reversal_info
                    })
        
        self.reversals = reversals
        return df
    
    def plot_reversals(self, df: pd.DataFrame, n_candles: int = 200, title: str = 'BB反轉點檢測'):
        """
        繪製K線圖並標記反轉點
        """
        # 只顯示最近N根K線
        df_plot = df.tail(n_candles).copy()
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, 'BB寬度')
        )
        
        # K線圖
        fig.add_trace(
            go.Candlestick(
                x=df_plot.index,
                open=df_plot['open'],
                high=df_plot['high'],
                low=df_plot['low'],
                close=df_plot['close'],
                name='K線',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # BB上軌
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['bb_upper'],
                name='BB上軌',
                line=dict(color='rgba(250, 128, 114, 0.5)', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # BB中軌
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['bb_middle'],
                name='BB中軌',
                line=dict(color='rgba(128, 128, 128, 0.5)', width=1)
            ),
            row=1, col=1
        )
        
        # BB下軌
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['bb_lower'],
                name='BB下軌',
                line=dict(color='rgba(100, 181, 246, 0.5)', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # 標記上軌反轉點 (紅色三角形)
        upper_reversals = df_plot[df_plot['reversal_point'] == 'upper']
        if len(upper_reversals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=upper_reversals.index,
                    y=upper_reversals['high'] * 1.002,  # 稍微往上偏移
                    mode='markers',
                    name='上軌反轉',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(color='darkred', width=1)
                    ),
                    text=[f"反轉: {x:.2%}" for x in upper_reversals['reversal_pct']],
                    hovertemplate='<b>上軌反轉點</b><br>反轉幅度: %{text}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 標記下軌反轉點 (綠色三角形)
        lower_reversals = df_plot[df_plot['reversal_point'] == 'lower']
        if len(lower_reversals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=lower_reversals.index,
                    y=lower_reversals['low'] * 0.998,  # 稍微往下偏移
                    mode='markers',
                    name='下軌反轉',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(color='darkgreen', width=1)
                    ),
                    text=[f"反轉: {x:.2%}" for x in lower_reversals['reversal_pct']],
                    hovertemplate='<b>下軌反轉點</b><br>反轉幅度: %{text}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # BB寬度
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['bb_width'] * 100,
                name='BB寬度 (%)',
                line=dict(color='purple', width=1)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text='時間', row=2, col=1)
        fig.update_yaxes(title_text='價格', row=1, col=1)
        fig.update_yaxes(title_text='BB寬度 (%)', row=2, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """
        統計反轉點數據
        """
        if not hasattr(self, 'reversals'):
            return {}
        
        upper_reversals = [r for r in self.reversals if r['type'] == 'upper']
        lower_reversals = [r for r in self.reversals if r['type'] == 'lower']
        
        stats = {
            'total_reversals': len(self.reversals),
            'upper_reversals': len(upper_reversals),
            'lower_reversals': len(lower_reversals),
            'avg_reversal_pct': np.mean([r['reversal_pct'] for r in self.reversals]) if self.reversals else 0,
            'avg_upper_reversal_pct': np.mean([r['reversal_pct'] for r in upper_reversals]) if upper_reversals else 0,
            'avg_lower_reversal_pct': np.mean([r['reversal_pct'] for r in lower_reversals]) if lower_reversals else 0,
        }
        
        # 計算觸碰成功率
        total_touches_upper = df['touch_upper'].sum() + df['break_upper'].sum()
        total_touches_lower = df['touch_lower'].sum() + df['break_lower'].sum()
        
        stats['upper_success_rate'] = (len(upper_reversals) / total_touches_upper * 100) if total_touches_upper > 0 else 0
        stats['lower_success_rate'] = (len(lower_reversals) / total_touches_lower * 100) if total_touches_lower > 0 else 0
        
        return stats


if __name__ == '__main__':
    print("BB反轉點檢測器測試")
    print("="*60)
    
    # 測試數據
    dates = pd.date_range('2024-01-01', periods=500, freq='15min')
    np.random.seed(42)
    
    base_price = 50000
    prices = base_price + np.random.randn(500).cumsum() * 100
    
    df = pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': prices + np.random.rand(500) * 100,
        'low': prices - np.random.rand(500) * 100,
        'close': prices + np.random.randn(500) * 50,
        'volume': np.random.randint(1000, 5000, 500)
    })
    
    detector = BBReversalDetector(
        bb_period=20,
        bb_std=2.0,
        touch_threshold=0.001,
        reversal_confirm_candles=3,
        min_reversal_pct=0.003
    )
    
    df_result = detector.detect_reversals(df)
    
    stats = detector.get_statistics(df_result)
    print(f"總反轉點: {stats['total_reversals']}")
    print(f"上軌反轉: {stats['upper_reversals']} (成功率: {stats['upper_success_rate']:.1f}%)")
    print(f"下軌反轉: {stats['lower_reversals']} (成功率: {stats['lower_success_rate']:.1f}%)")
    print(f"平均反轉幅度: {stats['avg_reversal_pct']:.2%}")