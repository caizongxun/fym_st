import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
from typing import Optional
import os

class BinanceDataLoader:
    """
    Load historical data from Binance API
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Binance client
        API keys are optional - can use public endpoints for historical data
        """
        if api_key and api_secret:
            self.client = Client(api_key, api_secret)
        else:
            # Use public client (no authentication needed for historical data)
            self.client = Client("", "")
    
    def load_historical_data(self, 
                            symbol: str, 
                            interval: str, 
                            start_date: datetime, 
                            end_date: datetime) -> pd.DataFrame:
        """
        Load historical kline data from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Start datetime
            end_date: End datetime
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Loading {symbol} {interval} data from {start_date} to {end_date}")
        
        # Convert datetime to milliseconds
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        # Fetch klines
        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_ms,
            end_str=end_ms
        )
        
        if not klines:
            raise ValueError(f"No data returned for {symbol} {interval}")
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by time
        df = df.sort_values('open_time').reset_index(drop=True)
        
        print(f"Loaded {len(df)} candles")
        
        return df
    
    def load_latest_candles(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        Load the latest N candles
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            limit: Number of candles (max 1000)
        
        Returns:
            DataFrame with OHLCV data
        """
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price
        """
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])