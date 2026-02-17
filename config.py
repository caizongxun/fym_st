import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # HuggingFace Dataset
    HF_DATASET_ID = "zongowo111/v2-crypto-ohlcv-data"
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)
    
    # Binance API
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    
    # Trading Symbols
    SUPPORTED_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
        'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
        'LINKUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'ETCUSDT',
        'XLMUSDT', 'FILUSDT', 'MANAUSDT', 'SANDUSDT', 'AAVEUSDT',
        'ALGOUSDT', 'ARBUSDT', 'BALUSDT', 'BATUSDT', 'BCHUSDT',
        'COMPUSDT', 'CRVUSDT', 'ENJUSDT', 'ENSUSDT', 'GALAUSDT',
        'GRTUSDT', 'IMXUSDT', 'KAVAUSDT', 'MKRUSDT', 'NEARUSDT',
        'OPUSDT', 'SNXUSDT', 'SPELLUSDT', 'ZRXUSDT'
    ]
    
    # Timeframes
    TRADING_TIMEFRAME = '15m'
    TREND_TIMEFRAME = '1h'
    
    # Training Parameters
    TRAIN_SIZE = 3000  # Number of candles for training
    OOS_SIZE = 1500    # Out-of-sample validation size
    LOOKBACK = 50      # Number of historical candles for features
    
    # Model Parameters
    TREND_HORIZON = 10     # Candles ahead for trend prediction
    VOLATILITY_HORIZON = 5 # Candles ahead for volatility
    REVERSAL_HORIZON = 10  # Candles ahead for reversal detection
    
    # Trading Rules
    MIN_REVERSAL_PROB = 0.75  # Minimum probability to enter trade
    MIN_TREND_SCORE = 60      # Minimum trend strength for directional trades
    VOLUME_MULTIPLIER = 1.3   # Volume must be > MA * this value
    
    # Backtesting Defaults
    DEFAULT_CAPITAL = 10.0    # USDT
    DEFAULT_LEVERAGE = 10
    DEFAULT_TP_ATR = 3.0      # Take profit at 3x ATR
    DEFAULT_SL_ATR = 2.0      # Stop loss at 2x ATR
    DEFAULT_POSITION_SIZE = 0.95  # Use 95% of available capital per trade
    DEFAULT_MAX_POSITIONS = 1     # Maximum concurrent positions
    
    # Binance Fees (Contract)
    MAKER_FEE = 0.0002  # 0.02%
    TAKER_FEE = 0.0006  # 0.06%
    
    # Paths
    MODEL_DIR = 'models/saved'
    LOG_DIR = 'logs'
    
    @staticmethod
    def get_base_symbol(symbol: str) -> str:
        """Extract base symbol (e.g., BTCUSDT -> BTC)"""
        return symbol.replace('USDT', '')
    
    @staticmethod
    def get_hf_path(symbol: str, timeframe: str) -> str:
        """Generate HuggingFace dataset path"""
        base = Config.get_base_symbol(symbol)
        return f"klines/{symbol}/{base}_{timeframe}.parquet"