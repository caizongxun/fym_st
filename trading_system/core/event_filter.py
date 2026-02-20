import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EventFilter:
    """
    事件驅動抽樣: 過濾無效的盤整期,只保留有交易機會的K線
    這是解決 AUC 接近 0.5 的關鍵
    """
    
    def __init__(self, 
                 use_volatility_breakout: bool = True,
                 use_volume_surge: bool = True,
                 use_trend_alignment: bool = True,
                 use_macd_cross: bool = False,
                 use_bb_squeeze: bool = True,
                 min_events_ratio: float = 0.05):
        """
        參數:
            use_volatility_breakout: 波動率突破 (VSR > 1.2)
            use_volume_surge: 成交量爆增 (volume_ratio > 1.5)
            use_trend_alignment: 趨勢對齊 (EMA 多頭排列)
            use_macd_cross: MACD 交叉
            use_bb_squeeze: 布林帶壓縮後的突破
            min_events_ratio: 最少保留樣本比例 (0.05 = 5%)
        """
        self.use_volatility_breakout = use_volatility_breakout
        self.use_volume_surge = use_volume_surge
        self.use_trend_alignment = use_trend_alignment
        self.use_macd_cross = use_macd_cross
        self.use_bb_squeeze = use_bb_squeeze
        self.min_events_ratio = min_events_ratio
    
    def filter_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        應用多個事件過濾器,只保留符合條件的K線
        """
        logger.info(f"應用事件過濾器,原始數據: {len(df)} 筆")
        
        conditions = []
        condition_names = []
        
        # 1. 波動率突破: VSR > 1.2 (盤整壓縮後的爆發)
        if self.use_volatility_breakout and 'vsr' in df.columns:
            vsr_condition = df['vsr'] > 1.2
            conditions.append(vsr_condition)
            condition_names.append('VSR突破')
            logger.info(f"VSR突破: {vsr_condition.sum()} 事件 ({100*vsr_condition.sum()/len(df):.1f}%)")
        
        # 2. 成交量爆增: volume_ratio > 1.5
        if self.use_volume_surge and 'volume_ratio' in df.columns:
            volume_condition = df['volume_ratio'] > 1.5
            conditions.append(volume_condition)
            condition_names.append('成交量爆增')
            logger.info(f"成交量爆增: {volume_condition.sum()} 事件 ({100*volume_condition.sum()/len(df):.1f}%)")
        
        # 3. 趨勢對齊: EMA9 > EMA21 > EMA50 (明確多頭)
        if self.use_trend_alignment:
            if 'ema_9_21_ratio' in df.columns and 'ema_21_50_ratio' in df.columns:
                trend_condition = (df['ema_9_21_ratio'] > 1.0) & (df['ema_21_50_ratio'] > 1.0)
                conditions.append(trend_condition)
                condition_names.append('趨勢對齊')
                logger.info(f"趨勢對齊: {trend_condition.sum()} 事件 ({100*trend_condition.sum()/len(df):.1f}%)")
        
        # 4. MACD 交叉: macd_hist 由負轉正
        if self.use_macd_cross and 'macd_hist' in df.columns:
            macd_cross = (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
            conditions.append(macd_cross)
            condition_names.append('MACD交叉')
            logger.info(f"MACD交叉: {macd_cross.sum()} 事件 ({100*macd_cross.sum()/len(df):.1f}%)")
        
        # 5. 布林帶壓縮突破: bb_width_pct 先低於 0.02 再突破
        if self.use_bb_squeeze and 'bb_width_pct' in df.columns:
            bb_squeeze = df['bb_width_pct'].rolling(window=20).min() < 0.02
            bb_breakout = df['bb_width_pct'] > df['bb_width_pct'].shift(1)
            bb_condition = bb_squeeze & bb_breakout
            conditions.append(bb_condition)
            condition_names.append('BB壓縮突破')
            logger.info(f"BB壓縮突破: {bb_condition.sum()} 事件 ({100*bb_condition.sum()/len(df):.1f}%)")
        
        if len(conditions) == 0:
            logger.warning("未啟用任何事件過濾器,返回全部數據")
            return df
        
        # 組合條件: 只要滿足任意一個條件 (OR 邏輯)
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition | condition
        
        filtered_df = df[combined_condition].copy()
        
        # 確保保留至少 min_events_ratio 的數據
        min_events = int(len(df) * self.min_events_ratio)
        if len(filtered_df) < min_events:
            logger.warning(f"過濾後樣本太少 ({len(filtered_df)}),放寬條件以保留至少 {min_events} 筆")
            # 使用更寬鬆的條件
            fallback_condition = (df['volume_ratio'] > 1.2) | (df['vsr'] > 1.0) if 'vsr' in df.columns else (df['volume_ratio'] > 1.2)
            filtered_df = df[fallback_condition].copy()
            
            if len(filtered_df) < min_events:
                # 最後保障: 根據成交量排序,保留前 N%
                df_sorted = df.nlargest(min_events, 'volume_ratio') if 'volume_ratio' in df.columns else df
                filtered_df = df_sorted.copy()
        
        filtered_ratio = len(filtered_df) / len(df)
        logger.info(f"事件過濾完成: 保留 {len(filtered_df)}/{len(df)} 筆 ({100*filtered_ratio:.1f}%)")
        logger.info(f"啟用的過濾器: {', '.join(condition_names)}")
        
        return filtered_df
    
    def apply_aggressive_filter(self, df: pd.DataFrame, target_ratio: float = 0.10) -> pd.DataFrame:
        """
        更激進的過濾: 只保留10%最有機會的K線
        同時滿足多個條件 (AND 邏輯)
        """
        logger.info(f"應用激進過濾器,目標保留 {target_ratio*100:.0f}%")
        
        # 必須同時滿足:
        # 1. 成交量 > 1.3倍
        # 2. 波動率 > 1.0倍
        # 3. RSI 不在極端區 (30-70)
        
        conditions = pd.Series([True] * len(df), index=df.index)
        
        if 'volume_ratio' in df.columns:
            conditions &= df['volume_ratio'] > 1.3
        
        if 'vsr' in df.columns:
            conditions &= df['vsr'] > 1.0
        
        if 'rsi' in df.columns:
            conditions &= (df['rsi'] > 30) & (df['rsi'] < 70)
        
        filtered_df = df[conditions].copy()
        
        # 如果還是太多,按波動率排序取前 N%
        if len(filtered_df) > len(df) * target_ratio * 2:
            target_count = int(len(df) * target_ratio)
            if 'vsr' in filtered_df.columns:
                filtered_df = filtered_df.nlargest(target_count, 'vsr')
            else:
                filtered_df = filtered_df.nlargest(target_count, 'volume_ratio')
        
        logger.info(f"激進過濾完成: 保留 {len(filtered_df)}/{len(df)} 筆 ({100*len(filtered_df)/len(df):.1f}%)")
        
        return filtered_df