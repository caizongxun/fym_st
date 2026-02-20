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