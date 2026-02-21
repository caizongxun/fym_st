import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EventFilter:
    """
    事件驅動抽樣: 過濾無效的盤整期,只保留10-15%有交易機會的K線
    關鍵: 必須同時滿足多個條件 (AND 邏輯)
    """
    
    def __init__(self, 
                 use_strict_mode: bool = True,
                 min_volume_ratio: float = 1.5,
                 min_vsr: float = 1.0,
                 bb_squeeze_threshold: float = 0.5,
                 lookback_period: int = 20):
        """
        參數:
            use_strict_mode: 嚴格模式 (必須同時滿足多個條件)
            min_volume_ratio: 最小成交量比率 (1.5 = 150%)
            min_vsr: 最小波動率比率
            bb_squeeze_threshold: 布林帶壓縮門檻 (0.5 = 低於中位數)
            lookback_period: 回期期數
        """
        self.use_strict_mode = use_strict_mode
        self.min_volume_ratio = min_volume_ratio
        self.min_vsr = min_vsr
        self.bb_squeeze_threshold = bb_squeeze_threshold
        self.lookback_period = lookback_period
    
    def filter_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        嚴格事件過濾: 同時滿足三個條件
        1. 成交量爆增 (volume_ratio > 1.5)
        2. 價格突破 (創遞期高/低點)
        3. 波動率突破 (從壓縮區爆發)
        """
        logger.info(f"應用嚴格事件過濾器,原始數據: {len(df)} 筆")
        
        result = df.copy()
        
        # 條件 1: 成交量爆增
        if 'volume_ratio' not in result.columns:
            if 'volume' in result.columns:
                result['volume_ma_20'] = result['volume'].rolling(window=self.lookback_period).mean()
                result['volume_ratio'] = result['volume'] / result['volume_ma_20']
            else:
                logger.error("缺少 volume 欄位")
                return df
        
        volume_surge = result['volume_ratio'] > self.min_volume_ratio
        logger.info(f"成交量爆增 (>{self.min_volume_ratio}): {volume_surge.sum()} 事件 ({100*volume_surge.sum()/len(df):.1f}%)")
        
        # 條件 2: 價格突破 (創新高或新低)
        if 'close' in result.columns:
            rolling_high = result['close'].rolling(window=self.lookback_period).max()
            rolling_low = result['close'].rolling(window=self.lookback_period).min()
            
            # 突破新高 或 跌破新低
            breakout_high = result['close'] >= rolling_high.shift(1)
            breakout_low = result['close'] <= rolling_low.shift(1)
            price_breakout = breakout_high | breakout_low
        else:
            price_breakout = pd.Series([True] * len(result), index=result.index)
        
        logger.info(f"價格突破 ({self.lookback_period}期): {price_breakout.sum()} 事件 ({100*price_breakout.sum()/len(df):.1f}%)")
        
        # 條件 3: 波動率突破 (從壓縮區爆發)
        if 'bb_width_pct' in result.columns:
            bb_width_median = result['bb_width_pct'].rolling(window=50).median()
            # 前一根處於壓縮狀態 (低於中位數)
            was_squeezed = result['bb_width_pct'].shift(1) < (bb_width_median.shift(1) * self.bb_squeeze_threshold)
            # 當前波動率擴大
            is_expanding = result['bb_width_pct'] > result['bb_width_pct'].shift(1)
            volatility_breakout = was_squeezed & is_expanding
        elif 'vsr' in result.columns:
            volatility_breakout = result['vsr'] > self.min_vsr
        else:
            volatility_breakout = pd.Series([True] * len(result), index=result.index)
        
        logger.info(f"波動率突破: {volatility_breakout.sum()} 事件 ({100*volatility_breakout.sum()/len(df):.1f}%)")
        
        # 嚴格模式: 必須同時滿足所有條件 (AND)
        if self.use_strict_mode:
            combined_condition = volume_surge & price_breakout & volatility_breakout
            logger.info("使用嚴格模式 (AND 邏輯)")
        else:
            # 寬鬆模式: 滿足任意兩個 (OR with minimum 2)
            condition_count = volume_surge.astype(int) + price_breakout.astype(int) + volatility_breakout.astype(int)
            combined_condition = condition_count >= 2
            logger.info("使用寬鬆模式 (滿足任意2個條件)")
        
        filtered_df = result[combined_condition].copy()
        
        # 如果過濾太嚴格 (少於5%),放寬到只要求成交量+任意一個
        min_samples = int(len(df) * 0.05)
        if len(filtered_df) < min_samples:
            logger.warning(f"過濾太嚴 ({len(filtered_df)} < {min_samples}),放寬條件")
            combined_condition = volume_surge & (price_breakout | volatility_breakout)
            filtered_df = result[combined_condition].copy()
            
            # 最後保障
            if len(filtered_df) < min_samples:
                logger.warning(f"仍然太少,使用成交量排序取前{min_samples}筆")
                filtered_df = result.nlargest(min_samples, 'volume_ratio')
        
        # 如果過濾太寬鬆 (超過25%),再加入趨勢篩選
        max_samples = int(len(df) * 0.25)
        if len(filtered_df) > max_samples:
            logger.info(f"篩選過多 ({len(filtered_df)} > {max_samples}),加入趨勢篩選")
            # 只保留趨勢明確的 (多頭排列)
            if 'ema_9_21_ratio' in filtered_df.columns and 'ema_21_50_ratio' in filtered_df.columns:
                trend_aligned = (filtered_df['ema_9_21_ratio'] > 1.0) & (filtered_df['ema_21_50_ratio'] > 1.0)
                filtered_with_trend = filtered_df[trend_aligned]
                if len(filtered_with_trend) >= min_samples:
                    filtered_df = filtered_with_trend
                    logger.info(f"趨勢篩選後: {len(filtered_df)} 筆")
            
            # 如果還是太多,按波動率排序
            if len(filtered_df) > max_samples:
                if 'vsr' in filtered_df.columns:
                    filtered_df = filtered_df.nlargest(max_samples, 'vsr')
                else:
                    filtered_df = filtered_df.nlargest(max_samples, 'volume_ratio')
                logger.info(f"按波動率排序後: {len(filtered_df)} 筆")
        
        filtered_ratio = len(filtered_df) / len(df)
        logger.info(f"事件過濾完成: 保留 {len(filtered_df)}/{len(df)} 筆 ({100*filtered_ratio:.1f}%)")
        
        if filtered_ratio > 0.30:
            logger.warning(f"警告: 保留比例 {filtered_ratio*100:.1f}% > 30%,過濾器可能過於寬鬆")
        
        return filtered_df


class BBNW_BounceFilter:
    """
    BB + NW 雙通道觸碰過濾器
    專門為 15m 波段反轉交易設計
    
    只保留觸碰或刺穿 BB/NW 上下軌的極端事件，
    過濾掉在通道內部遊走的無效 K 線。
    
    這是整個波段反轉系統的第一關 (觸發層)
    """
    
    def __init__(self, 
                 use_bb: bool = True, 
                 use_nw: bool = True,
                 min_pierce_pct: float = 0.001,
                 require_volume_surge: bool = False,
                 min_volume_ratio: float = 1.2):
        """
        Args:
            use_bb: 是否啟用 BB 通道觸發
            use_nw: 是否啟用 NW 包絡線觸發
            min_pierce_pct: 最小刺穿比例 (預設 0.1%)
            require_volume_surge: 是否要求同時爆量
            min_volume_ratio: 最小成交量比率
        """
        self.use_bb = use_bb
        self.use_nw = use_nw
        self.min_pierce_pct = min_pierce_pct
        self.require_volume_surge = require_volume_surge
        self.min_volume_ratio = min_volume_ratio
        
    def filter_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        篩選觸碰 BB/NW 軌道的極端事件
        
        Returns:
            過濾後的 DataFrame，包含以下新增欄位:
            - is_long_setup: 是否為做多機會 (觸碰下軌)
            - is_short_setup: 是否為做空機會 (觸碰上軌)
            - touch_type: 觸碰類型 (BB_LOWER, BB_UPPER, NW_LOWER, NW_UPPER)
        """
        logger.info(f"BB/NW 觸碰過濾器啟動, 原始數據: {len(df)} 筆")
        result = df.copy()
        
        # 初始化觸發條件
        trigger_condition = pd.Series([False] * len(result), index=result.index)
        touch_types = pd.Series([None] * len(result), index=result.index, dtype=object)
        
        # --- BB 通道觸發 ---
        if self.use_bb and 'bb_lower' in result.columns and 'bb_upper' in result.columns:
            # 觸碰或刺穿下軌 (做多機會)
            bb_touch_lower = result['low'] <= result['bb_lower'] * (1 + self.min_pierce_pct)
            bb_pierce_lower = result['low'] < result['bb_lower']
            
            # 觸碰或突破上軌 (做空機會)
            bb_touch_upper = result['high'] >= result['bb_upper'] * (1 - self.min_pierce_pct)
            bb_pierce_upper = result['high'] > result['bb_upper']
            
            # 標記觸碸類型
            touch_types = touch_types.where(~bb_touch_lower, 'BB_LOWER')
            touch_types = touch_types.where(~bb_touch_upper, 'BB_UPPER')
            
            # 加入觸發條件
            trigger_condition = trigger_condition | bb_touch_lower | bb_touch_upper
            
            logger.info(f"BB 下軌觸碸: {bb_touch_lower.sum()} | BB 上軌觸碸: {bb_touch_upper.sum()}")
        
        # --- NW 包絡線觸發 ---
        if self.use_nw and 'nw_lower' in result.columns and 'nw_upper' in result.columns:
            nw_touch_lower = result['low'] <= result['nw_lower'] * (1 + self.min_pierce_pct)
            nw_touch_upper = result['high'] >= result['nw_upper'] * (1 - self.min_pierce_pct)
            
            # 標記觸碸類型 (如果未被 BB 標記過)
            touch_types = touch_types.where(~(nw_touch_lower & (touch_types == None)), 'NW_LOWER')
            touch_types = touch_types.where(~(nw_touch_upper & (touch_types == None)), 'NW_UPPER')
            
            trigger_condition = trigger_condition | nw_touch_lower | nw_touch_upper
            
            logger.info(f"NW 下軌觸碸: {nw_touch_lower.sum()} | NW 上軌觸碸: {nw_touch_upper.sum()}")
        
        # --- 成交量篩選 (可選) ---
        if self.require_volume_surge:
            if 'volume_ratio' in result.columns:
                volume_surge = result['volume_ratio'] >= self.min_volume_ratio
                trigger_condition = trigger_condition & volume_surge
                logger.info(f"加入成交量篩選 (>={self.min_volume_ratio}): {volume_surge.sum()} 事件")
            else:
                logger.warning("無 volume_ratio 欄位，跳過成交量篩選")
        
        # 篩選結果
        filtered_df = result[trigger_condition].copy()
        
        # 標記做多/做空機會
        filtered_df['is_long_setup'] = (
            (filtered_df['low'] <= filtered_df.get('bb_lower', np.inf)) | 
            (filtered_df['low'] <= filtered_df.get('nw_lower', np.inf))
        )
        
        filtered_df['is_short_setup'] = (
            (filtered_df['high'] >= filtered_df.get('bb_upper', -np.inf)) | 
            (filtered_df['high'] >= filtered_df.get('nw_upper', -np.inf))
        )
        
        filtered_df['touch_type'] = touch_types[trigger_condition]
        
        # 統計輸出
        long_setups = filtered_df['is_long_setup'].sum()
        short_setups = filtered_df['is_short_setup'].sum()
        filtered_ratio = len(filtered_df) / len(df) * 100
        
        logger.info(f"過濾完成: 保留 {len(filtered_df)}/{len(df)} 筆 ({filtered_ratio:.1f}%)")
        logger.info(f"做多機會: {long_setups} | 做空機會: {short_setups}")
        
        # 警告檢查
        if filtered_ratio < 2.0:
            logger.warning(f"保留比例 {filtered_ratio:.1f}% < 2%，過濾過嚴，建議放寬 min_pierce_pct")
        elif filtered_ratio > 20.0:
            logger.warning(f"保留比例 {filtered_ratio:.1f}% > 20%，過濾過寬，建議縮小 min_pierce_pct 或加入成交量篩選")
        
        return filtered_df