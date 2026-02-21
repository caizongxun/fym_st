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
    
    def add_mtf_alpha_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        新增 4 款 MTF 獨家 Alpha 特徵
        """
        result = df.copy()
        
        # 1. 跨週期波動率共振 (MVR)
        if 'vsr' in result.columns and 'vsr_1h' in result.columns:
            result['mvr'] = result['vsr'] / (result['vsr_1h'] + 1e-8)
        
        # 2. 訂單流分形背離 (CVD Fractal Divergence)
        if 'cvd_norm_10' in result.columns and 'cvd_norm_10_1h' in result.columns:
            result['cvd_fractal'] = result['cvd_norm_10'] - result['cvd_norm_10_1h']
        
        # 3. 影線成交量吸收率 (VWWA)
        min_open_close = result[['open', 'close']].min(axis=1)
        max_open_close = result[['open', 'close']].max(axis=1)
        high_low_range = result['high'] - result['low'] + 1e-8
        
        lower_wick_ratio = (min_open_close - result['low']) / high_low_range
        upper_wick_ratio = (result['high'] - max_open_close) / high_low_range
        
        if 'volume_ratio' in result.columns:
            result['vwwa_buy'] = lower_wick_ratio * result['volume_ratio']
            result['vwwa_sell'] = upper_wick_ratio * result['volume_ratio']
            
        # 4. 高維度趨勢衰竭計時器 (HTF Trend Exhaustion)
        if 'ema_cross_1h' in result.columns:
            trend_state = result['ema_cross_1h']
            # groupby 連續計數
            trend_age = trend_state.groupby((trend_state != trend_state.shift()).cumsum()).cumcount() + 1
            result['htf_trend_age_norm'] = trend_age / 24.0
            
        return result

    def merge_and_build_mtf_features(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        多時間框架合併: 將 1h 特徵無未來函數地對齊到 15m 數據上
        """
        logger.info("Merging 15m and 1h features for MTF Confluence System...")
        
        df_1h_copy = df_1h.copy()
        df_15m_copy = df_15m.copy()
        
        # 1. 創建 1h 的真實可見時間 (無未來函數)
        # 1h K 線在 open_time + 1 小時後才會收盤並確定特徵
        df_1h_copy['htf_close_time'] = df_1h_copy['open_time'] + pd.Timedelta(hours=1)
        
        # 2. 篩選 1h 的關鍵特徵並加上 _1h 綴詞
        # 只保留平穩特徵 + 必要的價格/EMA 特徵供後續計算
        cols_to_keep = [col for col in df_1h_copy.columns if col not in ['open_time', 'close_time', 'htf_close_time']]
        rename_dict = {col: f"{col}_1h" for col in cols_to_keep}
        df_1h_renamed = df_1h_copy.rename(columns=rename_dict)
        df_1h_renamed['htf_close_time'] = df_1h_copy['htf_close_time']
        
        # 確保時間排序 (merge_asof 必須)
        df_15m_copy = df_15m_copy.sort_values('open_time')
        df_1h_renamed = df_1h_renamed.sort_values('htf_close_time')
        
        # 3. merge_asof
        df_mtf = pd.merge_asof(
            df_15m_copy,
            df_1h_renamed,
            left_on='open_time',
            right_on='htf_close_time',
            direction='backward'
        )
        
        # 4. 清除無歷史對應的 NaN 數據
        df_mtf = df_mtf.dropna()
        
        # 5. 加入 MTF 獨家 Alpha 特徵
        df_mtf = self.add_mtf_alpha_features(df_mtf)
        
        # 最終確保無空值
        df_mtf = df_mtf.dropna()
        
        return df_mtf

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