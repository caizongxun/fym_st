import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from itertools import product

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    def __init__(self, backtester, predictor, signal_filter):
        self.backtester = backtester
        self.predictor = predictor
        self.signal_filter = signal_filter
    
    def optimize_probability_threshold(self, 
                                       predictions: pd.DataFrame,
                                       thresholds: List[float] = [0.55, 0.60, 0.65, 0.70, 0.75],
                                       tp_multiplier: float = 2.5,
                                       sl_multiplier: float = 1.5) -> Dict:
        
        logger.info(f"Optimizing probability threshold across {len(thresholds)} values")
        
        results = []
        
        for threshold in thresholds:
            filtered_signals = predictions[
                (predictions['signal'] == 1) & 
                (predictions['win_probability'] >= threshold)
            ].copy()
            
            if len(filtered_signals) < 10:
                logger.warning(f"Threshold {threshold}: Only {len(filtered_signals)} signals, skipping")
                continue
            
            backtest_result = self.backtester.run_backtest(
                filtered_signals,
                tp_multiplier=tp_multiplier,
                sl_multiplier=sl_multiplier
            )
            
            stats = backtest_result['statistics']
            
            results.append({
                'threshold': threshold,
                'num_signals': len(filtered_signals),
                'total_return': stats['total_return'],
                'win_rate': stats['win_rate'],
                'profit_factor': stats['profit_factor'],
                'max_drawdown': stats['max_drawdown'],
                'sharpe_ratio': stats['sharpe_ratio'],
                'total_trades': stats['total_trades']
            })
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            logger.warning("No valid optimization results")
            return {'results': results_df, 'best_threshold': None}
        
        best_idx = results_df['total_return'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        
        logger.info(f"Best probability threshold: {best_threshold} (return: {results_df.loc[best_idx, 'total_return']*100:.2f}%)")
        
        return {
            'results': results_df,
            'best_threshold': best_threshold,
            'best_stats': results_df.loc[best_idx].to_dict()
        }
    
    def optimize_tp_sl_ratio(self,
                            predictions: pd.DataFrame,
                            tp_multipliers: List[float] = [2.0, 2.5, 3.0, 3.5],
                            sl_multipliers: List[float] = [1.0, 1.25, 1.5, 1.75],
                            min_probability: float = 0.65) -> Dict:
        
        logger.info(f"Optimizing TP/SL ratios: {len(tp_multipliers)} x {len(sl_multipliers)} combinations")
        
        filtered_signals = predictions[
            (predictions['signal'] == 1) & 
            (predictions['win_probability'] >= min_probability)
        ].copy()
        
        if len(filtered_signals) < 10:
            logger.warning(f"Insufficient signals after filtering: {len(filtered_signals)}")
            return {'results': pd.DataFrame(), 'best_tp': None, 'best_sl': None}
        
        results = []
        
        for tp, sl in product(tp_multipliers, sl_multipliers):
            if tp / sl < 1.5:
                continue
            
            backtest_result = self.backtester.run_backtest(
                filtered_signals,
                tp_multiplier=tp,
                sl_multiplier=sl
            )
            
            stats = backtest_result['statistics']
            
            results.append({
                'tp_multiplier': tp,
                'sl_multiplier': sl,
                'risk_reward_ratio': tp / sl,
                'total_return': stats['total_return'],
                'win_rate': stats['win_rate'],
                'profit_factor': stats['profit_factor'],
                'max_drawdown': stats['max_drawdown'],
                'sharpe_ratio': stats['sharpe_ratio']
            })
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            return {'results': results_df, 'best_tp': None, 'best_sl': None}
        
        best_idx = results_df['total_return'].idxmax()
        best_tp = results_df.loc[best_idx, 'tp_multiplier']
        best_sl = results_df.loc[best_idx, 'sl_multiplier']
        
        logger.info(f"Best TP/SL: {best_tp}/{best_sl} (return: {results_df.loc[best_idx, 'total_return']*100:.2f}%)")
        
        return {
            'results': results_df,
            'best_tp': best_tp,
            'best_sl': best_sl,
            'best_stats': results_df.loc[best_idx].to_dict()
        }
    
    def optimize_filters(self,
                        predictions: pd.DataFrame,
                        tp_multiplier: float = 2.5,
                        sl_multiplier: float = 1.5) -> Dict:
        
        logger.info("Optimizing signal filters")
        
        filter_configs = [
            {
                'name': 'No filters (baseline)',
                'min_probability': 0.5,
                'use_filters': False
            },
            {
                'name': 'Probability 0.65 only',
                'min_probability': 0.65,
                'use_filters': False
            },
            {
                'name': 'Probability 0.70 only',
                'min_probability': 0.70,
                'use_filters': False
            },
            {
                'name': 'Light (prob + volume)',
                'min_probability': 0.60,
                'min_vsr': 0.3,
                'max_vsr': 2.0,
                'use_trend_filter': False,
                'use_rsi_filter': False,
                'min_rsi': 20,
                'max_rsi': 80,
                'use_volume_filter': True,
                'min_volume_ratio': 1.1,
                'use_macd_filter': False,
                'macd_require_positive': False,
                'use_filters': True
            },
            {
                'name': 'Moderate (prob + trend + vol)',
                'min_probability': 0.65,
                'min_vsr': 0.4,
                'max_vsr': 1.5,
                'use_trend_filter': True,
                'use_rsi_filter': False,
                'min_rsi': 20,
                'max_rsi': 80,
                'use_volume_filter': True,
                'min_volume_ratio': 1.2,
                'use_macd_filter': False,
                'macd_require_positive': False,
                'use_filters': True
            },
            {
                'name': 'Strict (all filters)',
                'min_probability': 0.70,
                'min_vsr': 0.5,
                'max_vsr': 1.2,
                'use_trend_filter': True,
                'use_rsi_filter': True,
                'min_rsi': 25,
                'max_rsi': 75,
                'use_volume_filter': True,
                'min_volume_ratio': 1.5,
                'use_macd_filter': True,
                'macd_require_positive': False,
                'use_filters': True
            }
        ]
        
        results = []
        
        for config in filter_configs:
            signals = predictions[predictions['signal'] == 1].copy()
            
            if len(signals) == 0:
                logger.warning(f"Config '{config['name']}': No signals to filter")
                continue
            
            if config['use_filters']:
                filter_params = {k: v for k, v in config.items() if k not in ['name', 'use_filters']}
                signals = self.signal_filter.apply_all_filters(signals, **filter_params)
            else:
                signals = signals[signals['win_probability'] >= config['min_probability']].copy()
            
            if len(signals) < 10:
                logger.warning(f"Config '{config['name']}': Only {len(signals)} signals after filtering, skipping")
                continue
            
            backtest_result = self.backtester.run_backtest(
                signals,
                tp_multiplier=tp_multiplier,
                sl_multiplier=sl_multiplier
            )
            
            stats = backtest_result['statistics']
            
            results.append({
                'config_name': config['name'],
                'num_signals': len(signals),
                'total_return': stats['total_return'],
                'win_rate': stats['win_rate'],
                'profit_factor': stats['profit_factor'],
                'max_drawdown': stats['max_drawdown'],
                'sharpe_ratio': stats['sharpe_ratio'],
                'avg_win': stats['avg_win'],
                'avg_loss': stats['avg_loss']
            })
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            return {'results': results_df, 'best_config': None}
        
        best_idx = results_df['total_return'].idxmax()
        best_config = results_df.loc[best_idx, 'config_name']
        
        logger.info(f"Best filter config: {best_config} (return: {results_df.loc[best_idx, 'total_return']*100:.2f}%)")
        
        return {
            'results': results_df,
            'best_config': best_config,
            'best_stats': results_df.loc[best_idx].to_dict()
        }