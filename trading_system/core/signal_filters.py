import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SignalFilter:
    def __init__(self):
        pass
    
    def filter_by_probability(self, signals: pd.DataFrame, min_probability: float = 0.6) -> pd.DataFrame:
        if len(signals) == 0:
            return signals
        
        filtered = signals[signals['win_probability'] >= min_probability].copy()
        retention = 100 * len(filtered) / len(signals) if len(signals) > 0 else 0
        logger.info(f"Probability filter: {len(signals)} -> {len(filtered)} signals ({retention:.1f}%) (min_prob={min_probability})")
        return filtered
    
    def filter_by_volatility(self, signals: pd.DataFrame, min_vsr: float = 0.3, max_vsr: float = 2.0) -> pd.DataFrame:
        if len(signals) == 0 or 'vsr' not in signals.columns:
            if 'vsr' not in signals.columns:
                logger.warning("VSR column not found, skipping volatility filter")
            return signals
        
        valid_vsr = signals['vsr'].notna()
        filtered = signals[valid_vsr & (signals['vsr'] >= min_vsr) & (signals['vsr'] <= max_vsr)].copy()
        retention = 100 * len(filtered) / len(signals) if len(signals) > 0 else 0
        logger.info(f"Volatility filter: {len(signals)} -> {len(filtered)} signals ({retention:.1f}%) (VSR {min_vsr}-{max_vsr})")
        return filtered
    
    def filter_by_trend(self, signals: pd.DataFrame, use_ema_filter: bool = True) -> pd.DataFrame:
        if not use_ema_filter or len(signals) == 0:
            return signals
        
        if 'ema_9' not in signals.columns or 'ema_21' not in signals.columns:
            logger.warning("EMA columns not found, skipping trend filter")
            return signals
        
        filtered = signals[signals['ema_9'] > signals['ema_21']].copy()
        retention = 100 * len(filtered) / len(signals) if len(signals) > 0 else 0
        logger.info(f"Trend filter: {len(signals)} -> {len(filtered)} signals ({retention:.1f}%) (EMA 9 > 21)")
        return filtered
    
    def filter_by_rsi(self, signals: pd.DataFrame, min_rsi: float = 20, max_rsi: float = 80, enabled: bool = True) -> pd.DataFrame:
        if not enabled or len(signals) == 0:
            return signals
        
        if 'rsi' not in signals.columns:
            logger.warning("RSI column not found, skipping RSI filter")
            return signals
        
        valid_rsi = signals['rsi'].notna()
        filtered = signals[valid_rsi & (signals['rsi'] >= min_rsi) & (signals['rsi'] <= max_rsi)].copy()
        retention = 100 * len(filtered) / len(signals) if len(signals) > 0 else 0
        logger.info(f"RSI filter: {len(signals)} -> {len(filtered)} signals ({retention:.1f}%) (RSI {min_rsi}-{max_rsi})")
        
        if retention < 10 and enabled:
            logger.warning(f"RSI filter removed {100-retention:.1f}% of signals. Consider widening RSI range.")
        
        return filtered
    
    def filter_by_volume(self, signals: pd.DataFrame, min_volume_ratio: float = 1.0, enabled: bool = True) -> pd.DataFrame:
        if not enabled or len(signals) == 0:
            return signals
        
        if 'volume_ratio' not in signals.columns:
            logger.warning("Volume ratio column not found, skipping volume filter")
            return signals
        
        valid_volume = signals['volume_ratio'].notna()
        filtered = signals[valid_volume & (signals['volume_ratio'] >= min_volume_ratio)].copy()
        retention = 100 * len(filtered) / len(signals) if len(signals) > 0 else 0
        logger.info(f"Volume filter: {len(signals)} -> {len(filtered)} signals ({retention:.1f}%) (vol_ratio >= {min_volume_ratio})")
        return filtered
    
    def filter_by_macd(self, signals: pd.DataFrame, require_positive: bool = False, enabled: bool = True) -> pd.DataFrame:
        if not enabled or len(signals) == 0:
            return signals
        
        if 'macd' not in signals.columns or 'macd_signal' not in signals.columns:
            logger.warning("MACD columns not found, skipping MACD filter")
            return signals
        
        if require_positive:
            filtered = signals[(signals['macd'] > signals['macd_signal']) & (signals['macd'] > 0)].copy()
        else:
            filtered = signals[signals['macd'] > signals['macd_signal']].copy()
        
        retention = 100 * len(filtered) / len(signals) if len(signals) > 0 else 0
        logger.info(f"MACD filter: {len(signals)} -> {len(filtered)} signals ({retention:.1f}%)")
        return filtered
    
    def apply_all_filters(self, 
                          signals: pd.DataFrame,
                          min_probability: float = 0.60,
                          min_vsr: Optional[float] = None,
                          max_vsr: Optional[float] = None,
                          use_trend_filter: bool = False,
                          use_rsi_filter: bool = False,
                          min_rsi: float = 20,
                          max_rsi: float = 80,
                          use_volume_filter: bool = False,
                          min_volume_ratio: float = 1.0,
                          use_macd_filter: bool = False,
                          macd_require_positive: bool = False) -> pd.DataFrame:
        
        logger.info(f"Applying signal filters to {len(signals)} signals")
        original_count = len(signals)
        
        if original_count == 0:
            logger.warning("No signals to filter")
            return signals
        
        filtered = signals.copy()
        
        filtered = self.filter_by_probability(filtered, min_probability)
        if len(filtered) == 0:
            logger.warning("All signals filtered out by probability threshold. Lower min_probability.")
            return filtered
        
        if min_vsr is not None and max_vsr is not None:
            filtered = self.filter_by_volatility(filtered, min_vsr, max_vsr)
            if len(filtered) == 0:
                logger.warning("All signals filtered out by volatility filter. Widen VSR range.")
                return filtered
        
        if use_trend_filter:
            before_count = len(filtered)
            filtered = self.filter_by_trend(filtered)
            if len(filtered) == 0:
                logger.warning("All signals filtered out by trend filter. Disable or check market trend.")
                return filtered
        
        if use_rsi_filter:
            before_count = len(filtered)
            filtered = self.filter_by_rsi(filtered, min_rsi, max_rsi, enabled=True)
            if len(filtered) == 0:
                logger.warning(f"All signals filtered out by RSI filter ({min_rsi}-{max_rsi}). Widen range to 20-80.")
                return filtered
        
        if use_volume_filter:
            filtered = self.filter_by_volume(filtered, min_volume_ratio, enabled=True)
            if len(filtered) == 0:
                logger.warning("All signals filtered out by volume filter. Lower min_volume_ratio.")
                return filtered
        
        if use_macd_filter:
            filtered = self.filter_by_macd(filtered, require_positive=macd_require_positive, enabled=True)
            if len(filtered) == 0:
                logger.warning("All signals filtered out by MACD filter. Disable or use less strict settings.")
                return filtered
        
        retention_rate = 100 * len(filtered) / original_count
        logger.info(f"Final filtered signals: {len(filtered)} ({retention_rate:.1f}% of original)")
        
        if retention_rate < 5:
            logger.warning(f"Very low signal retention rate ({retention_rate:.1f}%). Consider loosening filters.")
        elif retention_rate < 10:
            logger.info(f"Low signal retention rate ({retention_rate:.1f}%). Filters are very strict.")
        
        return filtered