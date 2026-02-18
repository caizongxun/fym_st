"""
BB 反轉精準捕捉系統

雙模型架構:
1. BB 反轉機率預測 (上軌/下軌分開訓練)
2. 趨勢強度判斷 (過濾強勢中的誤判)

決策邏輯:
- 反轉機率 > 70% + 趨勢強度 < 30% = 進場
"""

from .tab_bb_visualization import render_bb_visualization_tab
from .tab_reversal_training import render_reversal_training_tab
from .tab_trend_filter import render_trend_filter_tab
from .tab_backtest import render_backtest_tab
from .tab_live_monitor import render_live_monitor_tab

__all__ = [
    'render_bb_visualization_tab',
    'render_reversal_training_tab',
    'render_trend_filter_tab',
    'render_backtest_tab',
    'render_live_monitor_tab'
]