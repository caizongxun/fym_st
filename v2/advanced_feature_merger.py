import pandas as pd
import os
from typing import List, Optional


# ============================================================
# Binance API 歷史數據保留期限制 (硬性，無法繞過)
# ------------------------------------------------------------
# funding_rate      完整歷史 2019～今  → 訓練 ✅
# open_interest     僅最近 30 天    → 僅推論 ⚠️
# long_short_ratio  僅最近 30 天    → 僅推論 ⚠️
# taker_buy_sell    僅最近 30 天    → 僅推論 ⚠️
#
# 訓練時禁用 30 天限制數據的原因:
#   1. 樣本數過少 → 嚴重過擬合
#   2. 2019 前全為 NaN → LightGBM 用「有沒有缺失」做時間切分,引發時間偏差
# ============================================================


class AdvancedFeatureMerger:
    def __init__(self, advanced_data_dir: str = 'v2/advanced_data'):
        self.advanced_data_dir = advanced_data_dir

    # ----------------------------------------------------------
    # 資金費率 (完整歷史) — 訓練安全
    # ----------------------------------------------------------
    def load_funding_rate_features(self, symbol: str) -> pd.DataFrame:
        """
        資金費率: 8 小時間隔，具備 2019 至今完整歷史
        衍生特徵:
          fr_roc_8h    = 单期變化 (8h)
          fr_roc_24h   = 三期變化 (24h)
          fr_cumsum_7d = 7 日累計資金費率 (棕桿擠壓指標)
          fr_positive  = 費率為正 (多方主導)
        """
        filepath = os.path.join(self.advanced_data_dir, f"{symbol}_funding_rate.parquet")
        if not os.path.exists(filepath):
            return pd.DataFrame()

        df = pd.read_parquet(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 衍生特徵
        df['fr_roc_8h'] = df['fundingRate'].diff(1)
        df['fr_roc_24h'] = df['fundingRate'].diff(3)
        df['fr_cumsum_7d'] = df['fundingRate'].rolling(21).sum()
        df['fr_positive'] = (df['fundingRate'] > 0).astype(int)
        df['fr_extreme_pos'] = (df['fundingRate'] > 0.001).astype(int)
        df['fr_extreme_neg'] = (df['fundingRate'] < -0.001).astype(int)

        keep = [
            'timestamp', 'fundingRate',
            'funding_rate_ma8', 'funding_rate_ma24',
            'funding_rate_std', 'funding_rate_extreme',
            'fr_roc_8h', 'fr_roc_24h', 'fr_cumsum_7d',
            'fr_positive', 'fr_extreme_pos', 'fr_extreme_neg'
        ]
        keep = [c for c in keep if c in df.columns]
        return df[keep].fillna(0)

    # ----------------------------------------------------------
    # 未平倉量 — 優先讀完整歷史,否則 30 天
    # ----------------------------------------------------------
    def load_open_interest_features(self, symbol: str) -> pd.DataFrame:
        """
        優先使用 data.binance.vision 下載的完整歷史
        ({symbol}_open_interest_full.parquet)
        如果無完整歷史則讀取 API 30 天版本 (僅適用推論)
        """
        full_path = os.path.join(self.advanced_data_dir, f"{symbol}_open_interest_full.parquet")
        normal_path = os.path.join(self.advanced_data_dir, f"{symbol}_open_interest.parquet")
        path = full_path if os.path.exists(full_path) else normal_path
        if not os.path.exists(path):
            return pd.DataFrame()

        df = pd.read_parquet(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        keep = [c for c in ['timestamp', 'sumOpenInterest', 'sumOpenInterestValue',
                             'oi_change', 'oi_change_rate', 'oi_ma7', 'oi_ma30']
                if c in df.columns]
        return df[keep].fillna(0)

    def load_long_short_ratio_features(self, symbol: str) -> pd.DataFrame:
        filepath = os.path.join(self.advanced_data_dir, f"{symbol}_long_short_ratio.parquet")
        if not os.path.exists(filepath):
            return pd.DataFrame()
        df = pd.read_parquet(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        keep = [c for c in ['timestamp', 'longShortRatio', 'longAccount', 'shortAccount',
                             'ls_ratio_ma7', 'ls_ratio_extreme'] if c in df.columns]
        return df[keep].fillna(0)

    def load_taker_buy_sell_features(self, symbol: str) -> pd.DataFrame:
        filepath = os.path.join(self.advanced_data_dir, f"{symbol}_taker_buy_sell.parquet")
        if not os.path.exists(filepath):
            return pd.DataFrame()
        df = pd.read_parquet(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        keep = [c for c in ['timestamp', 'buySellRatio', 'buyVol', 'sellVol',
                             'taker_buy_sell_delta', 'taker_imbalance'] if c in df.columns]
        return df[keep].fillna(0)

    def load_order_flow_features(self, symbol: str) -> pd.DataFrame:
        filepath = os.path.join(self.advanced_data_dir, f"{symbol}_order_flow.parquet")
        if not os.path.exists(filepath):
            return pd.DataFrame()
        df = pd.read_parquet(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    # ----------------------------------------------------------
    # 內部合併工具
    # ----------------------------------------------------------
    def _merge_asof(self, base_df: pd.DataFrame, feature_df: pd.DataFrame, label: str = '') -> pd.DataFrame:
        """
        merge_asof: 對每個 15m K 線，取最近一筆特徵資料
        這樣無論資金費率是 8h 間隔還是其他間隔都能正確對齊
        """
        if feature_df.empty:
            return base_df
        merged = pd.merge_asof(
            base_df.sort_values('timestamp'),
            feature_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        new_cols = [c for c in feature_df.columns if c != 'timestamp' and c not in base_df.columns]
        if label:
            print(f"  + {label}: {len(new_cols)} 個特徵")
        return merged

    # ----------------------------------------------------------
    # 訓練用合併 (僅含完整歷史特徵)
    # ----------------------------------------------------------
    def merge_for_training(self, base_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        訓練集使用:
          資金費率 (2019～今, 完整歷史) ✅

        不含:
          open_interest, long_short_ratio, taker_buy_sell
          (上述三者僅 30 天，训練會引發時間偏差)
        """
        df = base_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"\n合併訓練特徵 ({symbol}) — 僅完整歷史...")
        print(f"  Base: {len(df):,} 筆")

        fr_df = self.load_funding_rate_features(symbol)
        df = self._merge_asof(df, fr_df, '資金費率 (2019～今)')

        print(f"  最終: {len(df):,} 筆, {len(df.columns)} 欄位")
        return df

    # ----------------------------------------------------------
    # 推論用合併 (加入 30 天短期數據)
    # ----------------------------------------------------------
    def merge_for_inference(self, base_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        實盤推論額外加入:
          未平倉量, 多空比, 主動買賣比, CVD
          (這些特徵僅在訓練集最近30天有數據,只用於最後一筆的實時推論)
        """
        df = self.merge_for_training(base_df, symbol)

        print(f"\n合併推論額外特徵 ({symbol}) — 最近 30 天...")

        oi_df = self.load_open_interest_features(symbol)
        df = self._merge_asof(df, oi_df, '未平倉量')

        ls_df = self.load_long_short_ratio_features(symbol)
        df = self._merge_asof(df, ls_df, '多空比')

        taker_df = self.load_taker_buy_sell_features(symbol)
        df = self._merge_asof(df, taker_df, '主動買賣比')

        of_df = self.load_order_flow_features(symbol)
        df = self._merge_asof(df, of_df, 'CVD Order Flow')

        print(f"  推論最終: {len(df):,} 筆, {len(df.columns)} 欄位")
        return df

    # ----------------------------------------------------------
    # 相容舊版
    # ----------------------------------------------------------
    def merge_all_features(self, base_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """相容舊版 API — 內部呼叫 merge_for_training"""
        return self.merge_for_training(base_df, symbol)

    # ----------------------------------------------------------
    # 特徵欄位列表
    # ----------------------------------------------------------
    def get_training_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """*訓練安全* 特徵: 僅資金費率 (2019～今完整歷史)"""
        prefixes = ('fundingRate', 'funding_rate', 'fr_')
        return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]

    def get_inference_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """*推論用* 特徵: 資金費率 + OI + 多空比 + Taker + CVD"""
        prefixes = (
            'fundingRate', 'funding_rate', 'fr_',
            'sumOpenInterest', 'oi_',
            'longShortRatio', 'ls_',
            'buySellRatio', 'taker_',
            'cvd', 'delta_volume', 'buy_pressure', 'sell_pressure'
        )
        return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]

    def get_advanced_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """相容舊版 — 回傳訓練安全特徵"""
        return self.get_training_feature_columns(df)


if __name__ == '__main__':
    from data_loader import CryptoDataLoader

    loader = CryptoDataLoader()
    merger = AdvancedFeatureMerger()

    df_base = loader.load_klines('BTCUSDT', '15m')
    df_base = loader.prepare_dataframe(df_base)

    # 訓練用 (僅資金費率)
    df_train = merger.merge_for_training(df_base, 'BTCUSDT')
    train_feats = merger.get_training_feature_columns(df_train)
    print(f"\n訓練安全特徵 ({len(train_feats)} 個):")
    for f in train_feats:
        print(f"  - {f}")

    # 推論用 (加入 30 天數據)
    df_infer = merger.merge_for_inference(df_base, 'BTCUSDT')
    infer_feats = merger.get_inference_feature_columns(df_infer)
    print(f"\n推論額外特徵 ({len(infer_feats)} 個):")
    for f in infer_feats:
        print(f"  - {f}")
