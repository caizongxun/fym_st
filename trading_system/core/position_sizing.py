import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class KellyCriterion:
    def __init__(self, tp_multiplier: float = 2.5, sl_multiplier: float = 1.5, kelly_fraction: float = 0.5):
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.kelly_fraction = kelly_fraction
        self.odds_ratio = tp_multiplier / sl_multiplier
    
    def calculate_position_size(self, win_probability: float, max_size: float = 1.0) -> float:
        if win_probability <= 0 or win_probability >= 1:
            return 0.0
        
        lose_probability = 1 - win_probability
        
        kelly_fraction_full = (self.odds_ratio * win_probability - lose_probability) / self.odds_ratio
        
        if kelly_fraction_full <= 0:
            return 0.0
        
        position_size = kelly_fraction_full * self.kelly_fraction
        
        position_size = min(position_size, max_size)
        
        return position_size
    
    def calculate_batch_positions(self, probabilities: np.ndarray) -> np.ndarray:
        return np.array([self.calculate_position_size(p) for p in probabilities])

class RiskManager:
    def __init__(self, max_position_size: float = 0.1, max_total_exposure: float = 0.3):
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
    
    def filter_signals(self, signals: pd.DataFrame, position_size_column: str = 'position_size') -> pd.DataFrame:
        result = signals.copy()
        
        result = result[result[position_size_column] > 0]
        
        result[position_size_column] = result[position_size_column].clip(upper=self.max_position_size)
        
        logger.info(f"Filtered {len(result)} valid signals with position sizing")
        return result