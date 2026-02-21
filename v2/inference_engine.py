import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import joblib


class InferenceEngine:
    def __init__(
        self,
        bounce_model_path: str,
        filter_model_path: str,
        bounce_threshold: float = 0.65,
        filter_threshold: float = 0.40
    ):
        self.bounce_model_data = joblib.load(bounce_model_path)
        self.filter_model_data = joblib.load(filter_model_path)
        
        self.bounce_model = self.bounce_model_data['model']
        self.filter_model = self.filter_model_data['model']
        
        self.bounce_features = self.bounce_model_data['feature_names']
        self.filter_features = self.filter_model_data['feature_names']
        
        self.bounce_threshold = bounce_threshold
        self.filter_threshold = filter_threshold
        
        print("Inference Engine Initialized")
        print(f"Bounce model: {self.bounce_model_data['model_type']}")
        print(f"Filter model: {self.filter_model_data['model_type']}")
        print(f"Bounce threshold: {self.bounce_threshold}")
        print(f"Filter threshold: {self.filter_threshold}")
    
    def predict_single(self, features: pd.Series) -> Dict:
        bounce_input = features[self.bounce_features].values.reshape(1, -1)
        filter_input = features[self.filter_features].values.reshape(1, -1)
        
        p_bounce = self.bounce_model.predict_proba(bounce_input)[0, 1]
        p_filter = self.filter_model.predict_proba(filter_input)[0, 1]
        
        signal = self._apply_confluence_veto(p_bounce, p_filter)
        
        result = {
            'p_bounce': p_bounce,
            'p_filter': p_filter,
            'signal': signal,
            'bounce_pass': p_bounce > self.bounce_threshold,
            'filter_pass': p_filter < self.filter_threshold,
            'reason': self._get_signal_reason(p_bounce, p_filter)
        }
        
        return result
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        bounce_input = df[self.bounce_features]
        filter_input = df[self.filter_features]
        
        df['p_bounce'] = self.bounce_model.predict_proba(bounce_input)[:, 1]
        df['p_filter'] = self.filter_model.predict_proba(filter_input)[:, 1]
        
        df['bounce_pass'] = df['p_bounce'] > self.bounce_threshold
        df['filter_pass'] = df['p_filter'] < self.filter_threshold
        
        df['signal'] = df.apply(
            lambda row: self._apply_confluence_veto(row['p_bounce'], row['p_filter']),
            axis=1
        )
        
        df['reason'] = df.apply(
            lambda row: self._get_signal_reason(row['p_bounce'], row['p_filter']),
            axis=1
        )
        
        return df
    
    def _apply_confluence_veto(self, p_bounce: float, p_filter: float) -> int:
        if p_bounce > self.bounce_threshold and p_filter < self.filter_threshold:
            return 1
        else:
            return 0
    
    def _get_signal_reason(self, p_bounce: float, p_filter: float) -> str:
        if p_bounce > self.bounce_threshold and p_filter < self.filter_threshold:
            return 'ENTRY_APPROVED'
        elif p_bounce <= self.bounce_threshold:
            return 'BOUNCE_WEAK'
        elif p_filter >= self.filter_threshold:
            return 'TREND_VETO'
        else:
            return 'UNKNOWN'
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        if 'signal' not in df.columns:
            df = self.predict_batch(df)
        
        stats = {
            'total_samples': len(df),
            'entry_approved': int(df['signal'].sum()),
            'entry_rate': df['signal'].mean() * 100,
            'avg_p_bounce': df['p_bounce'].mean(),
            'avg_p_filter': df['p_filter'].mean(),
            'bounce_pass_count': int(df['bounce_pass'].sum()),
            'filter_pass_count': int(df['filter_pass'].sum()),
            'reason_counts': df['reason'].value_counts().to_dict()
        }
        
        if 'target' in df.columns:
            approved_samples = df[df['signal'] == 1]
            if len(approved_samples) > 0:
                stats['approved_success_rate'] = approved_samples['target'].mean() * 100
            else:
                stats['approved_success_rate'] = 0.0
        
        return stats
