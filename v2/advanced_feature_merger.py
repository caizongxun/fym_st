import pandas as pd
import os
from typing import List, Optional
import glob


class AdvancedFeatureMerger:
    def __init__(self, advanced_data_dir: str = 'v2/advanced_data'):
        self.advanced_data_dir = advanced_data_dir
    
    def load_order_flow_features(self, symbol: str) -> pd.DataFrame:
        filepath = os.path.join(self.advanced_data_dir, f"{symbol}_order_flow.parquet")
        
        if not os.path.exists(filepath):
            print(f"Order flow data not found for {symbol}")
            return pd.DataFrame()
        
        df = pd.read_parquet(filepath)
        
        feature_cols = [
            'delta_volume', 'buy_pressure', 'sell_pressure',
            'trade_intensity', 'avg_trade_size', 'trade_size_volatility',
            'large_trade_count', 'large_trade_ratio'
        ]
        
        rename_dict = {col: f'of_{col}' for col in feature_cols if col in df.columns}
        df = df.rename(columns=rename_dict)
        
        return df
    
    def load_funding_rate_features(self, symbol: str) -> pd.DataFrame:
        filepath = os.path.join(self.advanced_data_dir, f"{symbol}_funding_rate.parquet")
        
        if not os.path.exists(filepath):
            print(f"Funding rate data not found for {symbol}")
            return pd.DataFrame()
        
        df = pd.read_parquet(filepath)
        
        feature_cols = [
            'fundingRate', 'funding_rate_ma8', 'funding_rate_ma24',
            'funding_rate_std', 'funding_rate_extreme'
        ]
        
        rename_dict = {col: f'fr_{col}' if not col.startswith('funding') else col 
                      for col in feature_cols if col in df.columns}
        df = df.rename(columns=rename_dict)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').resample('15T').ffill().reset_index()
        
        return df
    
    def load_open_interest_features(self, symbol: str) -> pd.DataFrame:
        filepath = os.path.join(self.advanced_data_dir, f"{symbol}_open_interest.parquet")
        
        if not os.path.exists(filepath):
            print(f"Open interest data not found for {symbol}")
            return pd.DataFrame()
        
        df = pd.read_parquet(filepath)
        
        feature_cols = [
            'sumOpenInterest', 'sumOpenInterestValue',
            'oi_change', 'oi_change_rate', 'oi_ma7', 'oi_ma30'
        ]
        
        rename_dict = {col: f'oi_{col}' if not col.startswith('oi_') else col 
                      for col in feature_cols if col in df.columns}
        df = df.rename(columns=rename_dict)
        
        return df
    
    def load_long_short_ratio_features(self, symbol: str) -> pd.DataFrame:
        filepath = os.path.join(self.advanced_data_dir, f"{symbol}_long_short_ratio.parquet")
        
        if not os.path.exists(filepath):
            print(f"Long/short ratio data not found for {symbol}")
            return pd.DataFrame()
        
        df = pd.read_parquet(filepath)
        
        feature_cols = [
            'longShortRatio', 'longAccount', 'shortAccount',
            'ls_ratio_ma7', 'ls_ratio_extreme'
        ]
        
        rename_dict = {col: f'ls_{col}' if not col.startswith('ls_') else col 
                      for col in feature_cols if col in df.columns}
        df = df.rename(columns=rename_dict)
        
        return df
    
    def merge_all_features(
        self,
        base_df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        if 'timestamp' not in base_df.columns:
            raise ValueError("base_df must contain 'timestamp' column")
        
        df = base_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"\nMerging advanced features for {symbol}...")
        print(f"Base dataframe: {len(df)} records")
        
        order_flow_df = self.load_order_flow_features(symbol)
        if not order_flow_df.empty:
            order_flow_df['timestamp'] = pd.to_datetime(order_flow_df['timestamp'])
            df = pd.merge(df, order_flow_df, on='timestamp', how='left', suffixes=('', '_of'))
            print(f"  + Order flow features: {len([c for c in df.columns if c.startswith('of_')])} features")
        
        funding_df = self.load_funding_rate_features(symbol)
        if not funding_df.empty:
            funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'])
            df = pd.merge(df, funding_df, on='timestamp', how='left', suffixes=('', '_fr'))
            print(f"  + Funding rate features: {len([c for c in df.columns if 'funding' in c])} features")
        
        oi_df = self.load_open_interest_features(symbol)
        if not oi_df.empty:
            oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'])
            df = pd.merge(df, oi_df, on='timestamp', how='left', suffixes=('', '_oi'))
            print(f"  + Open interest features: {len([c for c in df.columns if c.startswith('oi_')])} features")
        
        ls_df = self.load_long_short_ratio_features(symbol)
        if not ls_df.empty:
            ls_df['timestamp'] = pd.to_datetime(ls_df['timestamp'])
            df = pd.merge(df, ls_df, on='timestamp', how='left', suffixes=('', '_ls'))
            print(f"  + Long/short ratio features: {len([c for c in df.columns if c.startswith('ls_')])} features")
        
        advanced_feature_cols = [
            c for c in df.columns 
            if c.startswith(('of_', 'fr_', 'oi_', 'ls_', 'funding_rate'))
        ]
        
        print(f"\nTotal advanced features added: {len(advanced_feature_cols)}")
        print(f"Final dataframe: {len(df)} records, {len(df.columns)} columns")
        
        return df
    
    def get_advanced_feature_columns(self, df: pd.DataFrame) -> List[str]:
        return [
            c for c in df.columns 
            if c.startswith(('of_', 'fr_', 'oi_', 'ls_', 'funding_rate'))
        ]


if __name__ == '__main__':
    from data_loader import CryptoDataLoader
    
    loader = CryptoDataLoader()
    merger = AdvancedFeatureMerger()
    
    df_base = loader.load_klines('BTCUSDT', '15m')
    df_base = loader.prepare_dataframe(df_base)
    
    df_merged = merger.merge_all_features(df_base, 'BTCUSDT')
    
    advanced_features = merger.get_advanced_feature_columns(df_merged)
    print(f"\nAdvanced features:")
    for feat in advanced_features:
        print(f"  - {feat}")
