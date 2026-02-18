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
        try:
            from binance.client import Client
            if api_key and api_secret:
                self.client = Client(api_key, api_secret)
            else:
                # Use public client (no authentication needed for historical data)
                self.client = Client("", "")
        except ImportError:
            print("Warning: python-binance not installed. Live data features will be disabled.")
            self.client = None
    
    def load_historical_data(self, 
                            symbol: str, 
                            interval: str, 
                            start_date: datetime, 
                            end_date: datetime) -> pd.DataFrame:
        """
        Load historical kline data from Binance
        """
        if self.client is None:
            raise ImportError("python-binance library is required")
            
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
        
        return self._process_klines(klines)
    
    def load_latest_candles(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        Load the latest N candles
        """
        if self.client is None:
            raise ImportError("python-binance library is required")
            
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        return self._process_klines(klines)
    
    def load_klines(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        兼容性接口: 加載K線數據
        默認加載最近1000根，如果需要更多請使用 load_historical_data
        """
        # 為了回測，我們嘗試加載更多數據 (例如最近30天)
        # 或者直接調用 load_latest_candles
        try:
            # 嘗試獲取最近30天數據
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            return self.load_historical_data(symbol, interval, start_date, end_date)
        except Exception as e:
            print(f"Warning: Failed to load 30 days history, falling back to 1000 candles. Error: {e}")
            return self.load_latest_candles(symbol, interval, limit=1000)
    
    def _process_klines(self, klines) -> pd.DataFrame:
        """內部方法: 處理K線數據格式"""
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 確保按時間排序
        df = df.sort_values('open_time').reset_index(drop=True)
        
        return df

    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price
        """
        if self.client is None:
            return 0.0
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])