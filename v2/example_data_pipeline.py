from pipeline import TradingPipeline
from data_loader import CryptoDataLoader


def example_single_symbol():
    print("=" * 60)
    print("Example 1: Process single symbol")
    print("=" * 60)
    
    pipeline = TradingPipeline(
        bb_period=20,
        atr_period=14,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        lookahead_bars=16
    )
    
    df_train, feature_cols = pipeline.prepare_training_data(
        symbol='BTCUSDT',
        timeframe='15m',
        direction='long',
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    print("\nFirst 5 samples:")
    print(df_train[['timestamp', 'close', 'lower', 'atr', 'long_sl', 'long_tp', 'target']].head())
    
    return df_train, feature_cols


def example_batch_processing():
    print("\n" + "=" * 60)
    print("Example 2: Batch process multiple symbols")
    print("=" * 60)
    
    pipeline = TradingPipeline(
        bb_period=20,
        atr_period=14,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        lookahead_bars=16
    )
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    df_combined = pipeline.batch_process(
        symbols=symbols,
        timeframe='15m',
        direction='long',
        start_date='2024-01-01',
        end_date='2024-06-30'
    )
    
    print("\nCombined data summary:")
    print(df_combined.groupby('symbol').agg({
        'target': ['count', 'sum', 'mean']
    }))
    
    return df_combined


def example_dataset_info():
    print("\n" + "=" * 60)
    print("Example 3: Dataset information")
    print("=" * 60)
    
    loader = CryptoDataLoader()
    info = loader.get_dataset_info()
    
    print(f"\nDataset: {info['repo_id']}")
    print(f"Total symbols: {info['total_symbols']}")
    print(f"Available timeframes: {info['timeframes']}")
    print(f"Total files: {info['total_files']}")
    print(f"\nAvailable symbols:")
    for i, symbol in enumerate(info['symbols'], 1):
        print(f"  {i:2d}. {symbol}")


if __name__ == "__main__":
    example_dataset_info()
    
    df_train, feature_cols = example_single_symbol()
    
    df_combined = example_batch_processing()
    
    print("\n" + "=" * 60)
    print("All examples completed")
    print("=" * 60)
