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
    
    def calculate_non_repainting_nw(self, df: pd.DataFrame, src_col: str = 'close', 
                                    h: float = 8.0, mult: float = 3.0, window: int = 50) -> pd.DataFrame:
        logger.info("Calculating Non-repainting Nadaraya-Watson Envelope...")
        result = df.copy()
        src = result[src_col].values
        n = len(src)
        
        out = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        
        def gauss(x, h):
            return np.exp(-(x**2) / (h * h * 2))
        
        distances = np.arange(window)[::-1]
        weights = gauss(distances, h)
        sum_weights = np.sum(weights)
        
        for i in range(window, n):
            window_src = src[i-window:i]
            
            nw_estimate = np.sum(window_src * weights) / sum_weights
            out[i] = nw_estimate
            
            window_mae = np.sum(np.abs(window_src - nw_estimate)) / window
            mae[i] = window_mae * mult
            
        result['nw_middle'] = out
        result['nw_upper'] = out + mae
        result['nw_lower'] = out - mae
        result['nw_width_pct'] = (result['nw_upper'] - result['nw_lower']) / (result['nw_middle'] + 1e-8)
        
        logger.info("NW Envelope calculated successfully")
        return result
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        result = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']
        
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(window=period).mean()
        
        result['adx'] = adx
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di
        
        return result

    def add_bounce_confluence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        logger.info("Adding bounce confluence features...")
        
        if 'bb_lower' in result.columns:
            result['bb_pierce_lower'] = (result['bb_lower'] - result['low']) / (result['close'] + 1e-8)
            result['bb_pierce_upper'] = (result['high'] - result['bb_upper']) / (result['close'] + 1e-8)
        
        if 'nw_lower' in result.columns:
            result['nw_pierce_lower'] = (result['nw_lower'] - result['low']) / (result['close'] + 1e-8)
            result['nw_pierce_upper'] = (result['high'] - result['nw_upper']) / (result['close'] + 1e-8)
        
        body_size = abs(result['close'] - result['open']) + 1e-8
        lower_wick = result[['open', 'close']].min(axis=1) - result['low']
        upper_wick = result['high'] - result[['open', 'close']].max(axis=1)
        
        result['lower_wick_size'] = lower_wick / body_size
        result['upper_wick_size'] = upper_wick / body_size
        
        if 'volume_ratio' in result.columns:
            result['vwwa_buy_signal'] = result['lower_wick_size'] * result['volume_ratio']
            result['vwwa_sell_signal'] = result['upper_wick_size'] * result['volume_ratio']
        
        if 'net_volume' in result.columns and 'bb_pierce_lower' in result.columns:
            result['sweep_divergence_buy'] = np.where(
                result['bb_pierce_lower'] > 0, 
                result['net_volume'], 
                0
            )
            result['sweep_divergence_sell'] = np.where(
                result['bb_pierce_upper'] > 0,
                -result['net_volume'],
                0
            )
        
        if 'adx' in result.columns:
            result['trend_crush_risk_15m'] = result['adx'] / 100.0
        
        if 'ema_21_dist_1h' in result.columns:
            result['trend_crush_risk_1h'] = abs(result['ema_21_dist_1h'])
        
        if 'bb_width_pct' in result.columns:
            bb_width_ma = result['bb_width_pct'].rolling(window=50).mean()
            result['bb_squeeze_ratio'] = result['bb_width_pct'] / (bb_width_ma + 1e-8)
        
        if 'bb_middle' in result.columns:
            result['price_to_bb_mid_dist'] = (result['close'] - result['bb_middle']) / (result['bb_middle'] + 1e-8)
        
        if 'nw_middle' in result.columns:
            result['price_to_nw_mid_dist'] = (result['close'] - result['nw_middle']) / (result['nw_middle'] + 1e-8)
        
        logger.info("Bounce confluence features added successfully")
        return result
    
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
        result = df.copy()
        for period in periods:
            result[f'return_{period}'] = df['close'].pct_change(period)
        return result
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        result['volume_ratio'] = df['volume'] / (result['volume_ma_20'] + 1e-8)
        
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
        ema_50 = close.ewm(span=50, adjust=False).mean()
        
        result['ema_9_dist'] = (close - ema_9) / (ema_9 + 1e-8)
        result['ema_21_dist'] = (close - ema_21) / (ema_21 + 1e-8)
        result['ema_50_dist'] = (close - ema_50) / (ema_50 + 1e-8)
        result['ema_9_21_ratio'] = ema_9 / (ema_21 + 1e-8)
        result['ema_21_50_ratio'] = ema_21 / (ema_50 + 1e-8)
        result['ema_cross'] = (ema_9 > ema_21).astype(int)
        
        return result
    
    def calculate_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        result['high_low_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
        result['close_open_ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
        result['body_size'] = abs(df['close'] - df['open']) / (df['close'] + 1e-8)
        
        return result
    
    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        taker_buy_col = None
        if 'taker_buy_base_asset_volume' in df.columns:
            taker_buy_col = 'taker_buy_base_asset_volume'
        elif 'taker_buy_volume' in df.columns:
            taker_buy_col = 'taker_buy_volume'
        
        if taker_buy_col is not None and 'volume' in df.columns:
            logger.info("Adding institutional microstructure features...")
            
            taker_sell_volume = df['volume'] - df[taker_buy_col]
            result['net_volume'] = df[taker_buy_col] - taker_sell_volume
            result['cvd_10'] = result['net_volume'].rolling(window=10).sum()
            result['cvd_20'] = result['net_volume'].rolling(window=20).sum()
            
            total_vol_10 = df['volume'].rolling(window=10).sum() + 1e-8
            result['cvd_norm_10'] = result['cvd_10'] / total_vol_10
            
            result['price_pct_10'] = df['close'].pct_change(10)
            result['divergence_score_10'] = result['cvd_norm_10'] - result['price_pct_10']
            
            body_size = abs(df['close'] - df['open']) + 1e-8
            upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
            lower_wick = df[['open', 'close']].min(axis=1) - df['low']
            
            result['upper_wick_ratio'] = upper_wick / body_size
            result['lower_wick_ratio'] = lower_wick / body_size
            
            result['order_flow_imbalance'] = (
                (df[taker_buy_col] - taker_sell_volume) / 
                (df[taker_buy_col] + taker_sell_volume + 1e-8)
            )
            
            logger.info("Added 8 core microstructure features")
        else:
            logger.warning("Taker volume data not available, skipping microstructure features")
        
        return result
    
    def calculate_liquidity_sweep_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        if 'open_interest' in df.columns:
            result['oi_change_pct'] = df['open_interest'].pct_change()
            result['oi_change_4h'] = df['open_interest'].pct_change(4)
            result['oi_normalized'] = (
                (df['open_interest'] - df['open_interest'].rolling(50).mean()) / 
                (df['open_interest'].rolling(50).std() + 1e-8)
            )
        
        if 'funding_rate' in df.columns:
            result['funding_rate_ma_3'] = df['funding_rate'].rolling(3).mean()
        
        return result
    
    def add_mtf_alpha_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        if 'vsr' in result.columns and 'vsr_1h' in result.columns:
            result['mvr'] = result['vsr'] / (result['vsr_1h'] + 1e-8)
        
        if 'cvd_norm_10' in result.columns and 'cvd_norm_10_1h' in result.columns:
            result['cvd_fractal'] = result['cvd_norm_10'] - result['cvd_norm_10_1h']
        
        min_open_close = result[['open', 'close']].min(axis=1)
        max_open_close = result[['open', 'close']].max(axis=1)
        high_low_range = result['high'] - result['low'] + 1e-8
        
        lower_wick_ratio = (min_open_close - result['low']) / high_low_range
        upper_wick_ratio = (result['high'] - max_open_close) / high_low_range
        
        if 'volume_ratio' in result.columns:
            result['vwwa_buy'] = lower_wick_ratio * result['volume_ratio']
            result['vwwa_sell'] = upper_wick_ratio * result['volume_ratio']
            
        if 'ema_cross_1h' in result.columns:
            trend_state = result['ema_cross_1h']
            trend_age = trend_state.groupby((trend_state != trend_state.shift()).cumsum()).cumcount() + 1
            result['htf_trend_age_norm'] = trend_age / 24.0
            
        return result

    def merge_and_build_mtf_features(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
        logger.info("Merging 15m and 1h features for MTF Confluence System...")
        
        df_1h_copy = df_1h.copy()
        df_15m_copy = df_15m.copy()
        
        essential_15m_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'atr']
        missing_essential = [col for col in essential_15m_cols if col not in df_15m_copy.columns]
        if missing_essential:
            logger.error(f"Missing essential columns in df_15m: {missing_essential}")
            raise ValueError(f"df_15m missing required columns: {missing_essential}")
        
        df_1h_copy['htf_close_time'] = df_1h_copy['open_time'] + pd.Timedelta(hours=1)
        df_1h_copy = df_1h_copy.drop(columns=['open_time'])
        
        cols_to_exclude = ['close_time', 'htf_close_time']
        cols_to_keep = [col for col in df_1h_copy.columns if col not in cols_to_exclude]
        rename_dict = {col: f"{col}_1h" for col in cols_to_keep}
        df_1h_renamed = df_1h_copy.rename(columns=rename_dict)
        df_1h_renamed['htf_close_time'] = df_1h_copy['htf_close_time']
        
        df_15m_copy = df_15m_copy.sort_values('open_time').reset_index(drop=True)
        df_1h_renamed = df_1h_renamed.sort_values('htf_close_time').reset_index(drop=True)
        
        df_mtf = pd.merge_asof(
            df_15m_copy,
            df_1h_renamed,
            left_on='open_time',
            right_on='htf_close_time',
            direction='backward'
        )
        
        if 'open_time' not in df_mtf.columns:
            logger.error("CRITICAL: open_time missing after merge_asof!")
            raise ValueError("open_time lost during merge_asof operation")
        
        logger.info(f"After merge_asof: {df_mtf.shape}")
        
        # 只 dropna essential columns，保留其他特徵的 NaN
        essential_check_cols = essential_15m_cols + ['close_1h', 'atr_1h']
        existing_check_cols = [col for col in essential_check_cols if col in df_mtf.columns]
        
        logger.info(f"Before selective dropna: {df_mtf.shape}")
        df_mtf = df_mtf.dropna(subset=existing_check_cols)
        logger.info(f"After selective dropna: {df_mtf.shape}")
        
        if len(df_mtf) == 0:
            logger.error("MTF merge resulted in empty DataFrame after dropna!")
            logger.error(f"Essential check columns: {existing_check_cols}")
            raise ValueError("MTF merge produced empty result")
        
        # 添加 MTF alpha 特徵
        df_mtf = self.add_mtf_alpha_features(df_mtf)
        
        # 最後再做一次檢查
        for col in essential_15m_cols:
            if col not in df_mtf.columns:
                logger.error(f"Essential column {col} missing in final df_mtf")
                raise ValueError(f"Essential column {col} lost during MTF processing")
        
        logger.info(f"MTF merge complete. Final shape: {df_mtf.shape}")
        
        return df_mtf

    def build_features(self, df: pd.DataFrame, 
                      use_fractional_diff: bool = False,
                      include_liquidity_features: bool = False,
                      include_microstructure: bool = True,
                      include_nw_envelope: bool = False,
                      include_adx: bool = False,
                      include_bounce_features: bool = False) -> pd.DataFrame:
        logger.info(f"Building features for {len(df)} rows")
        result = df.copy()
        
        # ATR
        result['atr'] = self.calculate_atr(result)
        result['atr_pct'] = result['atr'] / (result['close'] + 1e-8)
        
        # Bollinger Bands
        result = self.calculate_bollinger_bands(result)
        result['vsr'] = self.calculate_vsr(result)
        
        # Nadaraya-Watson Envelope
        if include_nw_envelope:
            result = self.calculate_non_repainting_nw(result)
        
        # ADX
        if include_adx:
            result = self.calculate_adx(result)
        
        # RSI
        result['rsi'] = self.calculate_rsi(result)
        result['rsi_normalized'] = (result['rsi'] - 50) / 50
        
        # MACD
        result = self.calculate_macd(result)
        
        # Returns
        result = self.calculate_returns(result)
        
        # Volume
        result = self.calculate_volume_features(result)
        
        # EMA
        result = self.calculate_ema_features(result)
        
        # Price Action
        result = self.calculate_price_action(result)
        
        # Microstructure
        if include_microstructure:
            result = self.add_microstructure_features(result)
        
        # Bounce confluence (需要在 MTF 合併後才能計算 HTF 特徵)
        if include_bounce_features:
            result = self.add_bounce_confluence_features(result)
        
        # OI/Funding
        if include_liquidity_features:
            logger.info("Adding OI/Funding liquidity features...")
            result = self.calculate_liquidity_sweep_features(result)
        
        # Fractional Diff
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
        
        return result