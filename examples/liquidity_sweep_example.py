#!/usr/bin/env python3
"""
流動性掃蕩系統完整示範

展示如何:
1. 獲取帶有 OI 的數據
2. 偵測流動性掃蕩事件
3. 訓練模型
4. 回測驗證
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../trading_system'))

from core import (
    CryptoDataLoader, FeatureEngineer,
    TripleBarrierLabeling, ModelTrainer,
    Backtester, LiquiditySweepDetector
)
import pandas as pd

def example_1_detect_sweeps():
    """示範 1: 偵測流動性掃蕩"""
    print("="*80)
    print("示範 1: 偵測流動性掃蕩")
    print("="*80)
    
    loader = CryptoDataLoader()
    df = loader.fetch_latest_klines(
        'BTCUSDT', '1h', days=30,
        include_oi=True, include_funding=True
    )
    
    print(f"\n載入 {len(df)} 筆數據")
    
    detector = LiquiditySweepDetector(
        lookback_period=50,
        wick_multiplier=2.0,
        oi_std_threshold=2.0
    )
    
    df_sweep = detector.detect_liquidity_sweep(df, direction='lower')
    df_sweep = detector.calculate_sweep_features(df_sweep)
    
    signals = df_sweep[df_sweep['sweep_lower_signal']]
    print(f"\n偵測到 {len(signals)} 個流動性掃蕩事件\n")
    
    if len(signals) > 0:
        print("事件詳情:")
        for idx, row in signals.iterrows():
            print(f"  {row['open_time']}: ${row['close']:.2f}, 下影線={row['lower_wick_ratio']:.2f}x")
    
    return df_sweep

def example_2_train_with_sweeps(df_sweep):
    """示範 2: 使用流動性特徵訓練模型"""
    print("\n" + "="*80)
    print("示範 2: 訓練模型 (含流動性特徵)")
    print("="*80)
    
    # 建立特徵
    fe = FeatureEngineer()
    df_features = fe.build_features(df_sweep, include_liquidity_features=True)
    
    print(f"\n特徵數量: {len([c for c in df_features.columns if c not in ['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume']])}")
    
    # 標籤
    labeling = TripleBarrierLabeling(tp=3.0, sl=1.0, timeout=24)
    df_labeled = labeling.label(df_features)
    
    # 增強掃蕩事件權重
    df_labeled['sample_weight'] = 1.0
    if 'sweep_lower_signal' in df_labeled.columns:
        df_labeled.loc[df_labeled['sweep_lower_signal'], 'sample_weight'] = 3.0
        sweep_count = df_labeled['sweep_lower_signal'].sum()
        print(f"流動性掃蕩事件: {sweep_count} (權重 3x)")
    
    # 選擇特徵
    features = [
        'atr_pct', 'rsi_normalized', 'bb_position', 'bb_width_pct', 'vsr',
        'ema_9_dist', 'ema_21_dist', 'ema_9_21_ratio',
        'lower_wick_ratio', 'upper_wick_ratio',
        'oi_change_pct', 'oi_change_4h', 'oi_normalized',
        'cvd_slope_5', 'cvd_normalized',
        'dist_to_support_pct',
        'volume_ratio', 'volatility_20', 'momentum_10'
    ]
    
    # 過濾缺失特徵
    available_features = [f for f in features if f in df_labeled.columns]
    print(f"可用特徵: {len(available_features)}/{len(features)}")
    
    if len(available_features) < 10:
        print("警告: 特徵數量不足,跳過訓練")
        return None
    
    # 訓練
    trainer = ModelTrainer()
    
    try:
        trainer.train(
            df_labeled,
            features=available_features,
            label='label',
            sample_weight='sample_weight'
        )
        
        print(f"\n模型訓練完成")
        print(f"AUC: {trainer.model_metrics.get('auc', 0):.3f}")
        print(f"Precision: {trainer.model_metrics.get('precision', 0):.3f}")
        
        return trainer
        
    except Exception as e:
        print(f"\n訓練失敗: {str(e)}")
        return None

def example_3_backtest_with_sweeps():
    """示範 3: 回測 (只在流動性掃蕩事件時進場)"""
    print("\n" + "="*80)
    print("示範 3: 回測 (流動性掃蕩過濾)")
    print("="*80)
    
    # 載入數據
    loader = CryptoDataLoader()
    df = loader.fetch_latest_klines(
        'BTCUSDT', '1h', days=90,
        include_oi=True, include_funding=True
    )
    
    print(f"\n載入 {len(df)} 筆數據")
    
    # 建立特徵
    fe = FeatureEngineer()
    df_features = fe.build_features(df, include_liquidity_features=True)
    
    # 偵測流動性掃蕩
    detector = LiquiditySweepDetector()
    df_sweep = detector.detect_liquidity_sweep(df_features, direction='lower')
    df_sweep = detector.calculate_sweep_features(df_sweep)
    
    sweep_events = df_sweep['sweep_lower_signal'].sum()
    print(f"流動性掃蕩事件: {sweep_events}")
    
    if sweep_events == 0:
        print("無流動性掃蕩事件,調低參數或增加天數")
        return
    
    # 模擬模型預測 (實際使用時應載入真實模型)
    # 這裡為了示範,假設流動性掃蕩事件的機率為 0.7
    df_sweep['win_probability'] = 0.3  # 基礎機率
    df_sweep.loc[df_sweep['sweep_lower_signal'], 'win_probability'] = 0.7  # 掃蕩事件高機率
    
    # 篩選信號
    signals = df_sweep[
        (df_sweep['win_probability'] > 0.65) &
        (df_sweep['sweep_lower_signal'] == True)
    ]
    
    print(f"篩選後信號: {len(signals)}")
    
    if len(signals) == 0:
        print("無符合條件的信號")
        return
    
    # 回測
    backtester = Backtester(
        initial_capital=10000,
        taker_fee=0.0006,
        maker_fee=0.0002,
        slippage=0.0005,
        risk_per_trade=0.01,
        leverage=10
    )
    
    results = backtester.run_backtest(
        signals,
        tp_multiplier=3.5,
        sl_multiplier=1.5,
        direction=1
    )
    
    stats = results['statistics']
    
    print("\n回測結果:")
    print(f"  交易次數: {stats['total_trades']}")
    print(f"  勝率: {stats['win_rate']*100:.1f}%")
    print(f"  總報酬: {stats['total_return']*100:.1f}%")
    print(f"  盈虧比: {stats['profit_factor']:.2f}")
    print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {stats['max_drawdown']*100:.1f}%")

def main():
    print("""
    流動性掃蕩系統示範
    ==========================================
    
    本示範展示如何使用流動性掃蕩系統:
    1. 偵測流動性掃蕩事件
    2. 使用流動性特徵訓練模型
    3. 回測驗證策略
    
    """)
    
    # 示範 1: 偵測流動性掃蕩
    df_sweep = example_1_detect_sweeps()
    
    # 示範 2: 訓練模型 (可選)
    # trainer = example_2_train_with_sweeps(df_sweep)
    
    # 示範 3: 回測
    example_3_backtest_with_sweeps()
    
    print("\n" + "="*80)
    print("示範完成")
    print("="*80)
    print("\n更多詳情請參考:")
    print("  - LIQUIDITY_SWEEP_INTEGRATION.md")
    print("  - docs/LIQUIDITY_SWEEP_THEORY.md")
    print("  - test_liquidity_sweep.py\n")

if __name__ == "__main__":
    main()