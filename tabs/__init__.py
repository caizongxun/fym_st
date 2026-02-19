"""Streamlit Tabs Module - Export all tab render functions"""

from tabs.tab_bb_visualization import render_bb_visualization_tab
from tabs.tab_reversal_training import render_reversal_training_tab
from tabs.tab_trend_filter import render_trend_filter_tab
from tabs.tab_backtest import render_backtest_tab
from tabs.tab_live_monitor import render_live_monitor_tab
from tabs.tab_range_bound_backtest import render_range_bound_backtest_tab
from tabs.tab_ml_strategy_d import render_ml_strategy_d_tab

__all__ = [
    'render_bb_visualization_tab',
    'render_reversal_training_tab',
    'render_trend_filter_tab',
    'render_backtest_tab',
    'render_live_monitor_tab',
    'render_range_bound_backtest_tab',
    'render_ml_strategy_d_tab',
]
