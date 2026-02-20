import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        pass
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def fractional_diff(self, series: pd.Series, d: float = 0.4, threshold: float = 0.01) -> pd.Series:
        weights = [1.0]
        k = 1
        while abs(weights[-1]) > threshold:
            weight = -weights[-1] * (d - k + 1) / k
            weights.append(weight)
            k += 1
        
        weights = np.array(weights[::-1])
        result = np.convolve(series.values, weights, mode='valid')
        result = pd.Series(result, index=series.index[len(weights)-1:])
        return result
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        result = df.copy()
        close = df['close']
        result['bb_middle'] = close.rolling(window=period).mean()
        result['bb_std'] = close.rolling(window=period).std()
        result['bb_upper'] = result['bb_middle'] + (std * result['bb_std'])
        result['bb_lower'] = result['bb_middle'] - (std * result['bb_std'])
        
        result['bb_position'] = (close - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-8)
        result['bb_width_pct'] = (result['bb_upper'] - result['bb_lower']) / (result['bb_middle'] + 1e-8)
        
        return result
    
    def calculate_vsr(self, df: pd.DataFrame, bb_period: int = 20, lookback: int = 50) -> pd.Series:
        df_bb = self.calculate_bollinger_bands(df, period=bb_period)
        bb_width = df_bb['bb_width_pct']
        vsr = bb_width / (bb_width.rolling(window=lookback).mean() + 1e-8)
        return vsr
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        result = df.copy()
        close = df['close']
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        result['macd_normalized'] = macd / (close + 1e-8)
        result['macd_signal'] = macd.ewm(span=signal, adjust=False).mean()
        result['macd_hist'] = macd - result['macd_signal']
        result['macd_hist_normalized'] = result['macd_hist'] / (close + 1e-8)
        return result
    
    def calculate_returns(self, df: pd.DataFrame, periods: list = [1, 5, 10]) -> pd.DataFrame:
        """簡化版報酬率特徵 (移除長期報酬避免冗餘)"""
        result = df.copy()
        for period in periods:
            result[f'return_{period}'] = df['close'].pct_change(period)
        return result
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        result['volume_ratio'] = df['volume'] / (result['volume_ma_20'] + 1e-8)
        
        # 基礎 taker_buy_ratio
        if 'taker_buy_base_asset_volume' in df.columns:
            result['taker_buy_ratio'] = df['taker_buy_base_asset_volume'] / (df['volume'] + 1e-8)
        elif 'taker_buy_volume' in df.columns:
            result['taker_buy_ratio'] = df['taker_buy_volume'] / (df['volume'] + 1e-8)
        else:
            result['taker_buy_ratio'] = 0.5
        
        return result
    
    def calculate_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        close = df['close']
        
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        
        # 只保留關鍵 EMA 特徵
        result['ema_9_dist'] = (close - ema_9) / (ema_9 + 1e-8)
        result['ema_21_dist'] = (close - ema_21) / (ema_21 + 1e-8)
        result['ema_9_21_ratio'] = ema_9 / (ema_21 + 1e-8)
        result['ema_cross'] = (ema_9 > ema_21).astype(int)
        
        return result
    
    def calculate_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        result['high_low_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
        result['close_open_ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
        result['body_size'] = abs(df['close'] - df['open']) / (df['close'] + 1e-8)
        
        return result
    
    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        機構級市場微觀結構特徵 (Order Flow & Microstructure)
        
        **核心理念**:
        在加密貨幣市場,機構資金會在關鍵位置掛大量限價單吸收零散的市價單。
        透過分析主動買賣盤的差異 (CVD) 與價格的背離,可以提前捕捉機構意圖。
        
        **8 個核心特徵**:
        1. net_volume: 淨主動成交量
        2. cvd_10: 短期 CVD (10 根)
        3. cvd_20: 中期 CVD (20 根)
        4. cvd_norm_10: 標準化 CVD (跨幣種可比)
        5. divergence_score_10: **價格-CVD 背離分數** (核心)
        6. upper_wick_ratio: 上影線/實體比例
        7. lower_wick_ratio: 下影線/實體比例 (流動性掠奪)
        8. order_flow_imbalance: 訂單流失衡比率
        
        這些特徵完全基於 Binance K線原生數據,無需額外 API。
        """
        result = df.copy()
        
        # 確認數據欄位
        taker_buy_col = None
        if 'taker_buy_base_asset_volume' in df.columns:
            taker_buy_col = 'taker_buy_base_asset_volume'
        elif 'taker_buy_volume' in df.columns:
            taker_buy_col = 'taker_buy_volume'
        
        if taker_buy_col is not None and 'volume' in df.columns:
            logger.info("[Phase 1] Adding institutional microstructure features...")
            
            # 1. 計算主動賣盤量 (Taker Sell Volume)
            taker_sell_volume = df['volume'] - df[taker_buy_col]
            
            # 2. 淨主動成交量 (Net Volume Delta)
            result['net_volume'] = df[taker_buy_col] - taker_sell_volume
            
            # 3. 短中期 CVD (Cumulative Volume Delta)
            # 使用滾動視窗確保時間序列平穩性
            result['cvd_10'] = result['net_volume'].rolling(window=10).sum()
            result['cvd_20'] = result['net_volume'].rolling(window=20).sum()
            
            # 4. 標準化 CVD 動能 (CVD Trend)
            # 用於跨時間框架與跨幣種比較
            total_vol_10 = df['volume'].rolling(window=10).sum() + 1e-8
            result['cvd_norm_10'] = result['cvd_10'] / total_vol_10
            
            # 5. 微觀背離指標 (Price-CVD Divergence) **[核心特徵]**
            # 邏輯: 價格變動與 CVD 變動的差值
            # 若價格下跌 (負值) 但 CVD 是正值 (買盤強),
            # 會產生極大的正向背離分數,代表底部有機構掛單吸收拋壓
            result['price_pct_10'] = df['close'].pct_change(10)
            result['divergence_score_10'] = result['cvd_norm_10'] - result['price_pct_10']
            
            # 6. 流動性掠奪引線特徵 (Liquidity Sweep Wick)
            # 計算上下影線相對於實體的比例,判斷是否為「拒絕突破」
            body_size = abs(df['close'] - df['open']) + 1e-8
            upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
            lower_wick = df[['open', 'close']].min(axis=1) - df['low']
            
            # 影線相對於實體的倍數
            result['upper_wick_ratio'] = upper_wick / body_size
            result['lower_wick_ratio'] = lower_wick / body_size
            
            # 7. 訂單流失衡比率 (Order Flow Imbalance Ratio)
            # 買賣壓力的相對強度 (-1 到 +1)
            result['order_flow_imbalance'] = (
                (df[taker_buy_col] - taker_sell_volume) / 
                (df[taker_buy_col] + taker_sell_volume + 1e-8)
            )
            
            logger.info("Added 8 core microstructure features")
        else:
            logger.warning("Taker volume data not available, skipping microstructure features")
        
        return result
    
    def calculate_liquidity_sweep_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        流動性掃蕩特徵 (OI & Funding Rate based)
        
        注意: 此函數專注於合約特有數據 (OI/Funding)
        訂單流特徵已在 add_microstructure_features 中處理
        """
        result = df.copy()
        
        # OI 特徵
        if 'open_interest' in df.columns:
            result['oi_change_pct'] = df['open_interest'].pct_change()
            result['oi_change_4h'] = df['open_interest'].pct_change(4)
            result['oi_normalized'] = (
                (df['open_interest'] - df['open_interest'].rolling(50).mean()) / 
                (df['open_interest'].rolling(50).std() + 1e-8)
            )
        
        # 資金費率
        if 'funding_rate' in df.columns:
            result['funding_rate_ma_3'] = df['funding_rate'].rolling(3).mean()
        
        return result
    
    def build_features(self, df: pd.DataFrame, 
                      use_fractional_diff: bool = False,
                      include_liquidity_features: bool = False,
                      include_microstructure: bool = True) -> pd.DataFrame:
        """
        建立所有特徵
        
        **第一階段重點**: include_microstructure=True (預設)
        這會加入 8 個機構級微觀結構特徵,大幅提升模型的 Alpha
        
        Args:
            df: 原始 OHLCV 數據
            use_fractional_diff: 是否使用分數差分
            include_liquidity_features: 是否包含 OI/Funding 特徵
            include_microstructure: 是否包含訂單流微觀結構特徵 (預設開啟)
        """
        logger.info(f"Building features for {len(df)} rows")
        result = df.copy()
        
        # ATR
        result['atr'] = self.calculate_atr(result)
        result['atr_pct'] = result['atr'] / (result['close'] + 1e-8)
        
        # Bollinger Bands
        result = self.calculate_bollinger_bands(result)
        result['vsr'] = self.calculate_vsr(result)
        
        # RSI
        result['rsi'] = self.calculate_rsi(result)
        result['rsi_normalized'] = (result['rsi'] - 50) / 50
        
        # MACD
        result = self.calculate_macd(result)
        
        # Returns (簡化)
        result = self.calculate_returns(result)
        
        # Volume
        result = self.calculate_volume_features(result)
        
        # EMA (精簡)
        result = self.calculate_ema_features(result)
        
        # Price Action (精簡)
        result = self.calculate_price_action(result)
        
        # ===== [第一階段核心] 機構級微觀結構特徵 =====
        if include_microstructure:
            result = self.add_microstructure_features(result)
        
        # OI/Funding 流動性特徵 (可選)
        if include_liquidity_features:
            logger.info("Adding OI/Funding liquidity features...")
            result = self.calculate_liquidity_sweep_features(result)
        
        # Fractional Diff (可選)
        if use_fractional_diff:
            result['price_frac_diff'] = self.fractional_diff(result['close'])
        
        # Volatility & Momentum
        result['volatility_20'] = result['close'].pct_change().rolling(window=20).std()
        result['momentum_10'] = result['close'].pct_change(10)
        
        result = result.dropna()
        logger.info(f"Features built, {len(result)} rows remaining after dropna")
        
        feature_cols = [col for col in result.columns if col not in 
                       ['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume',
                        'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume',
                        'taker_buy_base_asset_volume', 'label', 'label_return', 'ignore']]
        
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info(f"Feature list: {', '.join(feature_cols[:10])}... (+{len(feature_cols)-10} more)")
        
        return result