"""
Tabs 模組

每個 Tab 都是獨立的模組，便於維護和更新
"""

from .tab_bb_visualization import render_bb_visualization_tab
from .tab_bb_training import render_bb_training_tab
from .tab_bb_backtest import render_bb_backtest_tab
from .tab_scalping_training import render_scalping_training_tab
from .tab_scalping_backtest import render_scalping_backtest_tab

__all__ = [
    'render_bb_visualization_tab',
    'render_bb_training_tab',
    'render_bb_backtest_tab',
    'render_scalping_training_tab',
    'render_scalping_backtest_tab'
]