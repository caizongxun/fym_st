import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SignalFilter:
    def __init__(self):
        pass
    
    def filter_by_probability(self, signals: pd.DataFrame, min_probability: float = 0.6) -> pd.DataFrame:
        filtered = signals[signals['win_probability'] >= min_probability].copy()
        logger.info(f"Probability filter: {len(signals)} -> {len(filtered)} signals (min_prob={min_probability})")
        return filtered
    
    def filter_by_volatility(self, signals: pd.DataFrame, min_vsr: float = 0.3, max_vsr: float = 1.5) -> pd.DataFrame:
        if 'vsr' not in signals.columns:
            logger.warning("VSR column not found, skipping volatility filter")
            return signals
        
        filtered = signals[(signals['vsr'] >= min_vsr) & (signals['vsr'] <= max_vsr)].copy()
        logger.info(f"Volatility filter: {len(signals)} -> {len(filtered)} signals (VSR {min_vsr}-{max_vsr})")
        return filtered
    
    def filter_by_trend(self, signals: pd.DataFrame, use_ema_filter: bool = True) -> pd.DataFrame:
        if not use_ema_filter:
            return signals
        
        if 'ema_9' not in signals.columns or 'ema_21' not in signals.columns:
            logger.warning("EMA columns not found, skipping trend filter")
            return signals
        
        filtered = signals[signals['ema_9'] > signals['ema_21']].copy()
        logger.info(f"Trend filter: {len(signals)} -> {len(filtered)} signals (EMA 9 > 21)")
        return filtered
    
    def filter_by_rsi(self, signals: pd.DataFrame, min_rsi: float = 30, max_rsi: float = 70) -> pd.DataFrame:
        if 'rsi' not in signals.columns:
            logger.warning("RSI column not found, skipping RSI filter")
            return signals
        
        filtered = signals[(signals['rsi'] >= min_rsi) & (signals['rsi'] <= max_rsi)].copy()
        logger.info(f"RSI filter: {len(signals)} -> {len(filtered)} signals (RSI {min_rsi}-{max_rsi})")
        return filtered
    
    def filter_by_volume(self, signals: pd.DataFrame, min_volume_ratio: float = 1.2) -> pd.DataFrame:
        if 'volume_ratio' not in signals.columns:
            logger.warning("Volume ratio column not found, skipping volume filter")
            return signals
        
        filtered = signals[signals['volume_ratio'] >= min_volume_ratio].copy()
        logger.info(f"Volume filter: {len(signals)} -> {len(filtered)} signals (vol_ratio >= {min_volume_ratio})")
        return filtered
    
    def filter_by_macd(self, signals: pd.DataFrame, require_positive: bool = True) -> pd.DataFrame:
        if 'macd' not in signals.columns or 'macd_signal' not in signals.columns:
            logger.warning("MACD columns not found, skipping MACD filter")
            return signals
        
        if require_positive:
            filtered = signals[(signals['macd'] > signals['macd_signal']) & (signals['macd'] > 0)].copy()
        else:
            filtered = signals[signals['macd'] > signals['macd_signal']].copy()
        
        logger.info(f"MACD filter: {len(signals)} -> {len(filtered)} signals")
        return filtered
    
    def apply_all_filters(self, 
                          signals: pd.DataFrame,
                          min_probability: float = 0.65,
                          min_vsr: float = 0.4,
                          max_vsr: float = 1.3,
                          use_trend_filter: bool = True,
                          min_rsi: float = 35,
                          max_rsi: float = 65,
                          min_volume_ratio: float = 1.3,
                          use_macd_filter: bool = True) -> pd.DataFrame:
        
        logger.info(f"Applying signal filters to {len(signals)} signals")
        
        filtered = signals.copy()
        
        filtered = self.filter_by_probability(filtered, min_probability)
        
        filtered = self.filter_by_volatility(filtered, min_vsr, max_vsr)
        
        if use_trend_filter:
            filtered = self.filter_by_trend(filtered)
        
        filtered = self.filter_by_rsi(filtered, min_rsi, max_rsi)
        
        filtered = self.filter_by_volume(filtered, min_volume_ratio)
        
        if use_macd_filter:
            filtered = self.filter_by_macd(filtered)
        
        logger.info(f"Final filtered signals: {len(filtered)} ({100*len(filtered)/len(signals):.1f}% of original)")
        
        return filtered