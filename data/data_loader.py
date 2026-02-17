import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from binance.client import Client
from datetime import datetime, timedelta
import time
from typing import Optional, List
import os

class DataLoader:
    def __init__(self, hf_repo_id: str, binance_api_key: str = None, binance_api_secret: str = None):
        self.hf_repo_id = hf_repo_id
        self.binance_client = None
        if binance_api_key and binance_api_secret:
            self.binance_client = Client(binance_api_key, binance_api_secret)
    
    def load_from_huggingface(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load historical data from HuggingFace dataset
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Candle timeframe ('15m', '1h', '1d')
        
        Returns:
            DataFrame with OHLCV data
        """
        base = symbol.replace('USDT', '')
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        try:
            local_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=path_in_repo,
                repo_type="dataset"
            )
            df = pd.read_parquet(local_path)
            
            # Convert time columns to datetime
            df['open_time'] = pd.to_datetime(df['open_time'])
            df['close_time'] = pd.to_datetime(df['close_time'])
            
            # Ensure sorted by time
            df = df.sort_values('open_time').reset_index(drop=True)
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Error loading {symbol} {timeframe} from HuggingFace: {str(e)}")
            return pd.DataFrame()
    
    def load_from_binance(self, symbol: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """
        Load real-time data from Binance API
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.binance_client:
            raise ValueError("Binance client not initialized. Provide API credentials.")
        
        try:
            # Calculate start time
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Convert timeframe to Binance format
            interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '3m': Client.KLINE_INTERVAL_3MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '30m': Client.KLINE_INTERVAL_30MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '2h': Client.KLINE_INTERVAL_2HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }
            
            interval = interval_map.get(timeframe)
            if not interval:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Fetch klines
            klines = self.binance_client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time.strftime('%Y-%m-%d'),
                end_str=end_time.strftime('%Y-%m-%d')
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                          'quote_asset_volume', 'taker_buy_base_asset_volume',
                          'taker_buy_quote_asset_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['number_of_trades'] = df['number_of_trades'].astype(int)
            
            return df
            
        except Exception as e:
            print(f"Error loading {symbol} {timeframe} from Binance: {str(e)}")
            return pd.DataFrame()
    
    def get_completed_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the last incomplete candle to ensure signal stability
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            DataFrame without the last (potentially incomplete) candle
        """
        if len(df) == 0:
            return df
        
        # Check if last candle is complete
        current_time = datetime.now()
        last_close_time = df['close_time'].iloc[-1]
        
        # If last candle close time is in the future, it's incomplete
        if pd.notna(last_close_time) and last_close_time > current_time:
            return df.iloc[:-1].copy()
        
        return df.copy()
    
    def load_multi_timeframe(self, symbol: str, timeframes: List[str], 
                            source: str = 'huggingface', days: int = 30) -> dict:
        """
        Load multiple timeframes for a symbol
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes to load
            source: 'huggingface' or 'binance'
            days: Days to fetch (for Binance)
        
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        result = {}
        
        for tf in timeframes:
            if source == 'huggingface':
                df = self.load_from_huggingface(symbol, tf)
            elif source == 'binance':
                df = self.load_from_binance(symbol, tf, days)
            else:
                raise ValueError(f"Unknown source: {source}")
            
            if not df.empty:
                result[tf] = self.get_completed_candles(df)
        
        return result