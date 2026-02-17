from huggingface_hub import hf_hub_download
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

class HuggingFaceKlineLoader:
    """
    HuggingFace加密貨幣K線資料載入器
    
    資料集: zongowo111/v2-crypto-ohlcv-data
    支持38個交易對,3個時間週期(15m, 1h, 1d)
    """
    
    REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    
    # 支持的所有38個交易對
    SUPPORTED_SYMBOLS = [
        'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
        'AVAXUSDT', 'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT',
        'BTCUSDT', 'COMPUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT',
        'ENJUSDT', 'ENSUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT',
        'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT', 'LINKUSDT',
        'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
        'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT',
        'UNIUSDT', 'XRPUSDT', 'ZRXUSDT'
    ]
    
    TIMEFRAMES = ['15m', '1h', '1d']
    
    # 欄位映射: HuggingFace -> Binance格式
    COLUMN_MAPPING = {
        'open_time': 'open_time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'close_time': 'close_time',
        'quote_asset_volume': 'quote_volume',
        'number_of_trades': 'trades',
        'taker_buy_base_asset_volume': 'taker_buy_base',
        'taker_buy_quote_asset_volume': 'taker_buy_quote',
        'ignore': 'ignore'
    }
    
    def __init__(self):
        pass
    
    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        載入K線資料
        
        Args:
            symbol: 交易對,例如 'BTCUSDT'
            timeframe: 時間週期 '15m', '1h', '1d'
        
        Returns:
            DataFrame with Binance-compatible columns
        """
        if symbol not in self.SUPPORTED_SYMBOLS:
            raise ValueError(f"Symbol {symbol} not supported. Supported: {len(self.SUPPORTED_SYMBOLS)} symbols")
        
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Timeframe {timeframe} not supported. Supported: {self.TIMEFRAMES}")
        
        # 構造路徑
        base = symbol.replace("USDT", "")
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        # 下載資料
        local_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=path_in_repo,
            repo_type="dataset"
        )
        
        df = pd.read_parquet(local_path)
        
        # 重命名欄位以符合Binance格式
        df = df.rename(columns=self.COLUMN_MAPPING)
        
        # 確俟time欄是datetime
        df['open_time'] = pd.to_datetime(df['open_time'])
        df['close_time'] = pd.to_datetime(df['close_time'])
        
        # 設定index
        df = df.set_index('open_time')
        df = df.sort_index()
        
        return df
    
    def load_historical_data(self, 
                            symbol: str, 
                            interval: str,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        載入指定時間範圍的K線資料(相容BinanceDataLoader)
        
        Args:
            symbol: 交易對
            interval: 時間週期 '15m', '1h', '1d'
            start_time: 開始時間(UTC)
            end_time: 結束時間(UTC)
        
        Returns:
            Filtered DataFrame
        """
        df = self.load_klines(symbol, interval)
        
        # 過濾時間範圍
        if start_time is not None:
            df = df[df.index >= start_time]
        
        if end_time is not None:
            df = df[df.index <= end_time]
        
        return df
    
    @classmethod
    def get_supported_symbols(cls):
        """獲取支持的交易對列表"""
        return cls.SUPPORTED_SYMBOLS.copy()
    
    @classmethod
    def get_symbol_groups(cls):
        """
        獲取分類後的幣種組
        """
        groups = {
            '主流幣': [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT'
            ],
            'Layer1公鏈': [
                'AVAXUSDT', 'DOTUSDT', 'ATOMUSDT', 'NEARUSDT', 'ALGOUSDT'
            ],
            'Layer2擴展': [
                'ARBUSDT', 'OPUSDT', 'MATICUSDT', 'IMXUSDT'
            ],
            'DeFi生態': [
                'UNIUSDT', 'LINKUSDT', 'AAVEUSDT', 'CRVUSDT', 'COMPUSDT', 
                'MKRUSDT', 'SNXUSDT', 'BALUSDT', 'GRTUSDT'
            ],
            'NFT/元宇宙': [
                'SANDUSDT', 'MANAUSDT', 'ENJUSDT', 'GALAUSDT', 'ENSUSDT', 'SPELLUSDT'
            ],
            '傳統幣/其他': [
                'LTCUSDT', 'BCHUSDT', 'ETCUSDT', 'FILUSDT', 'BATUSDT', 'KAVAUSDT', 'ZRXUSDT'
            ]
        }
        return groups
    
    @classmethod
    def get_top_symbols(cls, n: int = 10):
        """
        獲取前n個市值最大的幣種(按市值排序)
        """
        top_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT',
            'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'NEARUSDT'
        ]
        return top_symbols[:min(n, len(top_symbols))]
    
    @classmethod
    def get_all_symbols_by_category(cls):
        """
        獲取所有38個幣種的分類列表
        """
        groups = cls.get_symbol_groups()
        result = []
        for category, symbols in groups.items():
            result.append(f"\n{category} ({len(symbols)}個):")
            result.extend([f"  - {s}" for s in symbols])
        return "\n".join(result)


if __name__ == '__main__':
    loader = HuggingFaceKlineLoader()
    
    print("=" * 70)
    print("HuggingFace 加密貨幣資料集")
    print("=" * 70)
    
    print(f"\n支持的交易對: {len(loader.SUPPORTED_SYMBOLS)} 個")
    print(f"時間週期: {', '.join(loader.TIMEFRAMES)}")
    
    print("\n" + loader.get_all_symbols_by_category())
    
    print("\n" + "=" * 70)
    print("熱門Top 10")
    print("=" * 70)
    top10 = loader.get_top_symbols(10)
    for i, symbol in enumerate(top10, 1):
        print(f"  {i:2d}. {symbol}")
    
    print("\n" + "=" * 70)
    print("分類統計")
    print("=" * 70)
    for category, symbols in loader.get_symbol_groups().items():
        print(f"{category}: {len(symbols)} 個")