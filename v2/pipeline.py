import pandas as pd
from typing import Optional, Tuple
from data_loader import CryptoDataLoader
from feature_engineering import FeatureEngineer
from label_generation import LabelGenerator


class TradingPipeline:
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: int = 2,
        lookback: int = 100,
        pivot_left: int = 3,
        pivot_right: int = 3,
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.0,
        lookahead_bars: int = 16
    ):
        self.data_loader = CryptoDataLoader()
        self.feature_engineer = FeatureEngineer(
            bb_period=bb_period,
            bb_std=bb_std,
            lookback=lookback,
            pivot_left=pivot_left,
            pivot_right=pivot_right
        )
        self.label_generator = LabelGenerator(
            atr_period=atr_period,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
            lookahead_bars=lookahead_bars
        )
    
    def process_single_symbol(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        print(f"\nProcessing {symbol} {timeframe}...")
        
        print("Step 1: Loading data...")
        df_raw = self.data_loader.load_klines(symbol, timeframe)
        print(f"  Loaded {len(df_raw)} rows")
        
        df_prepared = self.data_loader.prepare_dataframe(df_raw)
        
        if start_date or end_date:
            df_prepared = self.data_loader.filter_date_range(df_prepared, start_date, end_date)
            print(f"  Filtered to {len(df_prepared)} rows")
        
        print("Step 2: Feature engineering...")
        df_features = self.feature_engineer.process_features(df_prepared)
        print(f"  Features calculated: {len(df_features)} rows")
        
        print("Step 3: Label generation...")
        df_labeled = self.label_generator.generate_labels(df_features)
        print(f"  Labels generated: {len(df_labeled)} rows")
        
        stats = self.label_generator.get_label_statistics(df_labeled)
        print("\n  Label Statistics:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
        
        return df_labeled
    
    def prepare_training_data(
        self,
        symbol: str,
        timeframe: str,
        direction: str = 'long',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, list]:
        df_labeled = self.process_single_symbol(symbol, timeframe, start_date, end_date)
        
        df_train = self.label_generator.prepare_training_data(df_labeled, direction=direction)
        
        feature_cols = self.feature_engineer.get_feature_columns()
        
        print(f"\nTraining data prepared:")
        print(f"  Total samples: {len(df_train)}")
        print(f"  Feature columns: {len(feature_cols)}")
        print(f"  Target column: target")
        
        return df_train, feature_cols
    
    def batch_process(
        self,
        symbols: list,
        timeframe: str,
        direction: str = 'long',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        all_data = []
        
        for symbol in symbols:
            try:
                df_train, _ = self.prepare_training_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction,
                    start_date=start_date,
                    end_date=end_date
                )
                df_train['symbol'] = symbol
                all_data.append(df_train)
                print(f"\nSuccessfully processed {symbol}: {len(df_train)} samples\n{'-'*60}")
            except Exception as e:
                print(f"\nFailed to process {symbol}: {str(e)}\n{'-'*60}")
        
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            print(f"\n{'='*60}")
            print(f"Batch processing completed")
            print(f"Total symbols: {len(all_data)}")
            print(f"Total samples: {len(df_combined)}")
            print(f"{'='*60}")
            return df_combined
        else:
            raise ValueError("No data was successfully processed")
