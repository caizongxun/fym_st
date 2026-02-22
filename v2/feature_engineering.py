import pandas as pd
import numpy as np
from typing import Tuple, Optional


class FeatureEngineer:
    """
    特徵工程: 將 OHLCV 轉換為模型訓練特徵

    特徵組合:
    1. Bollinger Bands (帶寬、擠壓、發散)
    2. Z-Score (價格偏離程度)
    3. Pivot Points + SMC (機構引跟訂單流)
    4. Volume Profile (成交量分布: POC, VAH, VAL, VWAP)
    5. Fair Value Gap (公平價値空缺: FVG)
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: int = 2,
        lookback: int = 100,
        pivot_left: int = 3,
        pivot_right: int = 3,
        vp_lookback: int = 96,   # 96 x 15m = 1 天
        vp_bins: int = 20
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.lookback = lookback
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.vp_lookback = vp_lookback
        self.vp_bins = vp_bins

    # ----------------------------------------------------------
    # 1. Bollinger Bands
    # ----------------------------------------------------------
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['basis'] = df['close'].rolling(window=self.bb_period).mean()
        df['dev'] = df['close'].rolling(window=self.bb_period).std()
        df['upper'] = df['basis'] + (self.bb_std * df['dev'])
        df['lower'] = df['basis'] - (self.bb_std * df['dev'])
        df['bandwidth'] = (df['upper'] - df['lower']) / df['basis']

        def percentile_rank(x):
            if len(x) == 0:
                return np.nan
            return (x.iloc[-1] <= x).sum() / len(x) * 100

        df['bandwidth_percentile'] = df['bandwidth'].rolling(window=self.lookback).apply(
            percentile_rank, raw=False
        )
        df['is_squeeze'] = (df['bandwidth_percentile'] < 20).astype(int)
        df['is_expansion'] = (df['bandwidth_percentile'] > 80).astype(int)
        return df

    # ----------------------------------------------------------
    # 2. Z-Score
    # ----------------------------------------------------------
    def calculate_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'basis' not in df.columns or 'dev' not in df.columns:
            raise ValueError("Must calculate Bollinger Bands first")
        df['z_score'] = (df['close'] - df['basis']) / df['dev']
        return df

    # ----------------------------------------------------------
    # 3. Pivot Points
    # ----------------------------------------------------------
    def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        pivot_highs = np.full(n, np.nan)
        pivot_lows = np.full(n, np.nan)

        for i in range(self.pivot_left, n - self.pivot_right):
            left_high = highs[i - self.pivot_left:i]
            right_high = highs[i + 1:i + 1 + self.pivot_right]
            if highs[i] > left_high.max() and highs[i] > right_high.max():
                pivot_highs[i] = highs[i]
            left_low = lows[i - self.pivot_left:i]
            right_low = lows[i + 1:i + 1 + self.pivot_right]
            if lows[i] < left_low.min() and lows[i] < right_low.min():
                pivot_lows[i] = lows[i]

        df['pivot_high'] = pd.Series(pivot_highs, index=df.index).shift(self.pivot_right).ffill()
        df['pivot_low'] = pd.Series(pivot_lows, index=df.index).shift(self.pivot_right).ffill()
        df.rename(columns={'pivot_high': 'last_ph', 'pivot_low': 'last_pl'}, inplace=True)
        return df

    # ----------------------------------------------------------
    # 4. SMC Features
    # ----------------------------------------------------------
    def calculate_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'last_ph' not in df.columns or 'last_pl' not in df.columns:
            raise ValueError("Must calculate pivot points first")
        df['bear_sweep'] = ((df['high'] > df['last_ph']) & (df['close'] < df['last_ph'])).astype(int)
        df['bull_sweep'] = ((df['low'] < df['last_pl']) & (df['close'] > df['last_pl'])).astype(int)
        df['bull_bos'] = (df['close'] > df['last_ph']).astype(int)
        df['bear_bos'] = (df['close'] < df['last_pl']).astype(int)
        return df

    # ----------------------------------------------------------
    # 5. Volume Profile (POC + Value Area + VWAP)
    # ----------------------------------------------------------
    def calculate_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Point of Control (POC): lookback 期間成交量最大的價格區間
        Value Area High/Low (VAH/VAL): 包含 70% 成交量的價格範圍
        VWAP: 成交量加權平均價 (POC 的快速近似)

        計算复雜度: O(n x lookback) — 關鍵使用 numpy 向量化 bincount
        """
        df = df.copy()
        n = len(df)
        lb = self.vp_lookback
        nb = self.vp_bins

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        poc = np.full(n, np.nan)
        vah = np.full(n, np.nan)
        val = np.full(n, np.nan)

        for i in range(lb, n):
            cl = closes[i - lb:i]
            hi = highs[i - lb:i].max()
            lo = lows[i - lb:i].min()
            vo = volumes[i - lb:i]

            if hi <= lo or vo.sum() == 0:
                continue

            edges = np.linspace(lo, hi, nb + 1)
            mids = (edges[:-1] + edges[1:]) / 2

            # 以 close price 分配成交量到 bin (O(lookback) 向量化)
            idx = np.searchsorted(edges[1:], cl)
            idx = np.clip(idx, 0, nb - 1)
            vol_bins = np.bincount(idx, weights=vo, minlength=nb).astype(float)

            poc_i = int(np.argmax(vol_bins))
            poc[i] = mids[poc_i]

            # Value Area 擴展 (70% 成交量展開)
            total = vol_bins.sum()
            target = total * 0.70
            lo_i = hi_i = poc_i
            acc = vol_bins[poc_i]

            while acc < target:
                can_up = hi_i + 1 < nb
                can_dn = lo_i - 1 >= 0
                if not can_up and not can_dn:
                    break
                up_vol = vol_bins[hi_i + 1] if can_up else -1
                dn_vol = vol_bins[lo_i - 1] if can_dn else -1
                if up_vol >= dn_vol:
                    hi_i += 1
                    acc += vol_bins[hi_i]
                else:
                    lo_i -= 1
                    acc += vol_bins[lo_i]

            vah[i] = mids[hi_i]
            val[i] = mids[lo_i]

        df['poc'] = poc
        df['vah'] = vah
        df['val'] = val

        # 價格距離特徵 (%)
        df['poc_distance'] = (df['close'] - df['poc']) / df['poc']
        df['vah_distance'] = (df['close'] - df['vah']) / df['vah']
        df['val_distance'] = (df['close'] - df['val']) / df['val']

        # BB 下軌是否贼近 POC (支撑共振訊號)
        if 'lower' in df.columns:
            df['lower_near_poc'] = (
                (df['lower'] - df['poc']).abs() / df['poc'] < 0.005
            ).astype(int)
        else:
            df['lower_near_poc'] = 0

        # 價格是否在 Value Area 內/外
        df['in_value_area'] = (
            (df['close'] >= df['val']) & (df['close'] <= df['vah'])
        ).astype(int)
        df['below_value_area'] = (df['close'] < df['val']).astype(int)

        # VWAP 1天 + 1週 (向量化，快速)
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_x_vol = tp * df['volume']
        df['vwap_1d'] = tp_x_vol.rolling(lb).sum() / df['volume'].rolling(lb).sum()
        df['vwap_1w'] = tp_x_vol.rolling(lb * 7).sum() / df['volume'].rolling(lb * 7).sum()
        df['vwap_distance'] = (df['close'] - df['vwap_1d']) / df['vwap_1d']

        return df

    # ----------------------------------------------------------
    # 6. Fair Value Gap (FVG)
    # ----------------------------------------------------------
    def calculate_fvg_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fair Value Gap: 3根 K 線的價格空缺 (流動性磁鐵)

        Bullish FVG: candle[i-2].high < candle[i].low  (gap 向上)
        Bearish FVG: candle[i-2].low  > candle[i].high (gap 向下)

        全部向量化，無循環
        """
        df = df.copy()

        high_2 = df['high'].shift(2)
        low_2 = df['low'].shift(2)

        # Bullish FVG
        bull_gap = df['low'] - high_2
        df['bullish_fvg'] = (bull_gap > 0).astype(int)
        df['bullish_fvg_pct'] = bull_gap.clip(lower=0) / df['close']

        # Bearish FVG
        bear_gap = low_2 - df['high']
        df['bearish_fvg'] = (bear_gap > 0).astype(int)
        df['bearish_fvg_pct'] = bear_gap.clip(lower=0) / df['close']

        # 凈動純度
        df['fvg_net'] = df['bullish_fvg'] - df['bearish_fvg']

        # 最近一次 FVG 中心價 (引力目標)
        bull_mid = ((df['low'] + high_2) / 2).where(df['bullish_fvg'] == 1)
        bear_mid = ((df['high'] + low_2) / 2).where(df['bearish_fvg'] == 1)
        df['last_bull_fvg_mid'] = bull_mid.ffill()
        df['last_bear_fvg_mid'] = bear_mid.ffill()

        # 距最近 FVG 中心的距離 (%)
        df['dist_to_bull_fvg'] = (df['close'] - df['last_bull_fvg_mid']) / df['close']
        df['dist_to_bear_fvg'] = (df['close'] - df['last_bear_fvg_mid']) / df['close']

        return df

    # ----------------------------------------------------------
    # 主流程
    # ----------------------------------------------------------
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain: {required_cols}")

        df = self.calculate_bollinger_bands(df)
        df = self.calculate_zscore(df)
        df = self.calculate_pivot_points(df)
        df = self.calculate_smc_features(df)
        df = self.calculate_volume_profile(df)
        df = self.calculate_fvg_features(df)
        df = df.dropna()
        return df

    def get_feature_columns(self) -> list:
        return [
            # Bollinger Bands
            'basis', 'dev', 'upper', 'lower',
            'bandwidth', 'bandwidth_percentile',
            'is_squeeze', 'is_expansion',
            # Z-Score
            'z_score',
            # Pivot & SMC
            'last_ph', 'last_pl',
            'bear_sweep', 'bull_sweep', 'bull_bos', 'bear_bos',
            # Volume Profile
            'poc', 'vah', 'val',
            'poc_distance', 'vah_distance', 'val_distance',
            'lower_near_poc', 'in_value_area', 'below_value_area',
            'vwap_1d', 'vwap_distance',
            # Fair Value Gap
            'bullish_fvg', 'bearish_fvg',
            'bullish_fvg_pct', 'bearish_fvg_pct',
            'fvg_net', 'dist_to_bull_fvg', 'dist_to_bear_fvg',
        ]
