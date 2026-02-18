"""
Ensemble RL-Transformer 交易系統

三層架構:
1. 多時間框架特徵提取 (Transformer)
2. Ensemble 模型預測 (Transformer + LSTM + XGBoost + Attention-GRU)
3. 強化學習決策 (PPO/SAC)
"""

from .tab_data_analysis import render_data_analysis_tab
from .tab_feature_engineering import render_feature_engineering_tab
from .tab_transformer_training import render_transformer_training_tab
from .tab_ensemble_training import render_ensemble_training_tab
from .tab_rl_training import render_rl_training_tab
from .tab_backtest import render_backtest_tab
from .tab_live_trading import render_live_trading_tab

__all__ = [
    'render_data_analysis_tab',
    'render_feature_engineering_tab',
    'render_transformer_training_tab',
    'render_ensemble_training_tab',
    'render_rl_training_tab',
    'render_backtest_tab',
    'render_live_trading_tab'
]