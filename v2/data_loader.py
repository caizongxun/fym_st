import pandas as pd
from huggingface_hub import hf_hub_download
from typing import List, Optional
import os


class CryptoDataLoader:
    REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    
    AVAILABLE_SYMBOLS = [
        'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
        'AVAXUSDT', 'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT',
        'BTCUSDT', 'COMPUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT',
        'ENJUSDT', 'ENSUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT',
        'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT', 'LINKUSDT',
        'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
        'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT',
        'UNIUSDT', 'XRPUSDT', 'ZRXUSDT'
    ]
    
    AVAILABLE_TIMEFRAMES = ['15m', '1h', '1d']
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
    
    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        if symbol not in self.AVAILABLE_SYMBOLS:
            raise ValueError(f"Symbol {symbol} not available. Choose from: {self.AVAILABLE_SYMBOLS}")
        
        if timeframe not in self.AVAILABLE_TIMEFRAMES:
            raise ValueError(f"Timeframe {timeframe} not available. Choose from: {self.AVAILABLE_TIMEFRAMES}")
        
        base = symbol.replace("USDT", "")
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        local_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=path_in_repo,
            repo_type="dataset",
            cache_dir=self.cache_dir
        )
        
        df = pd.read_parquet(local_path)
        
        return df
    
    def load_multiple_symbols(self, symbols: List[str], timeframe: str) -> dict:
        data = {}
        for symbol in symbols:
            try:
                df = self.load_klines(symbol, timeframe)
                data[symbol] = df
                print(f"Loaded {symbol} {timeframe}: {len(df)} rows")
            except Exception as e:
                print(f"Failed to load {symbol} {timeframe}: {str(e)}")
        return data
    
    def load_all_timeframes(self, symbol: str) -> dict:
        data = {}
        for timeframe in self.AVAILABLE_TIMEFRAMES:
            try:
                df = self.load_klines(symbol, timeframe)
                data[timeframe] = df
                print(f"Loaded {symbol} {timeframe}: {len(df)} rows")
            except Exception as e:
                print(f"Failed to load {symbol} {timeframe}: {str(e)}")
        return data
    
    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df = df.rename(columns={
            'open_time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        return df
    
    def get_dataset_info(self) -> dict:
        return {
            'repo_id': self.REPO_ID,
            'total_symbols': len(self.AVAILABLE_SYMBOLS),
            'symbols': self.AVAILABLE_SYMBOLS,
            'timeframes': self.AVAILABLE_TIMEFRAMES,
            'total_files': len(self.AVAILABLE_SYMBOLS) * len(self.AVAILABLE_TIMEFRAMES)
        }
    
    def filter_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        df = df.copy()
        
        time_col = 'timestamp' if 'timestamp' in df.columns else 'open_time'
        
        if start_date:
            df = df[df[time_col] >= pd.to_datetime(start_date)]
        
        if end_date:
            df = df[df[time_col] <= pd.to_datetime(end_date)]
        
        return df.reset_index(drop=True)
