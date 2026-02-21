import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer
from label_generation import LabelGenerator


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='15min')
    
    close = 50000 + np.cumsum(np.random.randn(n_samples) * 100)
    high = close + np.random.uniform(50, 200, n_samples)
    low = close - np.random.uniform(50, 200, n_samples)
    open_price = close + np.random.uniform(-100, 100, n_samples)
    volume = np.random.uniform(1000, 10000, n_samples)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


def main():
    print("=" * 60)
    print("第一步: 特徵工程")
    print("=" * 60)
    
    df = generate_sample_data(1000)
    print(f"生成範例數據: {len(df)} 筆")
    
    fe = FeatureEngineer(
        bb_period=20,
        bb_std=2,
        lookback=100,
        pivot_left=3,
        pivot_right=3
    )
    
    df_features = fe.process_features(df)
    print(f"特徵計算完成: {len(df_features)} 筆 (NaN已移除)")
    print(f"特徵欄位: {fe.get_feature_columns()}")
    
    print("\n" + "=" * 60)
    print("第二步: 標籤生成")
    print("=" * 60)
    
    lg = LabelGenerator(
        atr_period=14,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        lookahead_bars=16,
        lower_tolerance=1.001,
        upper_tolerance=0.999
    )
    
    df_labeled = lg.generate_labels(df_features)
    print(f"標籤生成完成: {len(df_labeled)} 筆")
    
    stats = lg.get_label_statistics(df_labeled)
    print("\n標籤統計資訊:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("準備訓練數據")
    print("=" * 60)
    
    df_train_long = lg.prepare_training_data(df_labeled, direction='long')
    print(f"做多樣本: {len(df_train_long)} 筆")
    
    df_train_short = lg.prepare_training_data(df_labeled, direction='short')
    print(f"做空樣本: {len(df_train_short)} 筆")
    
    print("\n" + "=" * 60)
    print("範例數據預覽")
    print("=" * 60)
    print(df_train_long[['close', 'lower', 'upper', 'atr', 'long_sl', 'long_tp', 'target']].head(10))
    
    print("\n模塊化系統建置完成")


if __name__ == "__main__":
    main()
