import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import joblib
import os
from typing import Dict, Tuple

class DualModelTrainer:
    """
    雙模型訓練器
    
    模型A: 方向預測 (RandomForestClassifier)
    - 預測下一根K棒是漲還是跌
    - 輸出: 0=跌, 1=漲
    
    模型B: 價格範圍預測 (2個RandomForestRegressor)
    - 預測下一根K棒的最高價和最低價
    - 輸出: high_pct%, low_pct% (相對於當前close)
    """
    
    def __init__(self, 
                 model_dir: str = 'models/saved',
                 direction_params: dict = None,
                 price_params: dict = None):
        
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # 預設參數
        default_direction_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        
        default_price_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.direction_params = direction_params or default_direction_params
        self.price_params = price_params or default_price_params
        
        # 初始化模型
        self.direction_model = None
        self.high_model = None
        self.low_model = None
        
        self.metrics = {}
    
    def train_direction_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """訓練方向預測模型"""
        print("\n[1/3] 訓練方向預測模型...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        self.direction_model = RandomForestClassifier(**self.direction_params)
        self.direction_model.fit(X_train, y_train)
        
        # 評估
        y_pred = self.direction_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 獲取機率預測
        y_proba = self.direction_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'up_ratio': y_test.mean(),
            'avg_confidence': y_proba.mean()
        }
        
        print(f"  測試集準確率: {accuracy:.4f}")
        print(f"  樣本分布 - 漮:{y_test.sum()}, 跌:{len(y_test)-y_test.sum()}")
        
        return metrics
    
    def train_price_models(self, X: pd.DataFrame, y_high: pd.Series, y_low: pd.Series) -> Dict:
        """訓練價格範圏預測模型"""
        print("\n[2/3] 訓練最高價預測模型...")
        
        X_train, X_test, y_high_train, y_high_test, y_low_train, y_low_test = train_test_split(
            X, y_high, y_low, test_size=0.2, random_state=42, shuffle=False
        )
        
        # 訓練最高價模型
        self.high_model = RandomForestRegressor(**self.price_params)
        self.high_model.fit(X_train, y_high_train)
        
        y_high_pred = self.high_model.predict(X_test)
        high_mae = mean_absolute_error(y_high_test, y_high_pred)
        high_r2 = r2_score(y_high_test, y_high_pred)
        
        print(f"  MAE: {high_mae:.4f}%, R2: {high_r2:.4f}")
        
        # 訓練最低價模型
        print("\n[3/3] 訓練最低價預測模型...")
        self.low_model = RandomForestRegressor(**self.price_params)
        self.low_model.fit(X_train, y_low_train)
        
        y_low_pred = self.low_model.predict(X_test)
        low_mae = mean_absolute_error(y_low_test, y_low_pred)
        low_r2 = r2_score(y_low_test, y_low_pred)
        
        print(f"  MAE: {low_mae:.4f}%, R2: {low_r2:.4f}")
        
        metrics = {
            'high_mae': high_mae,
            'high_r2': high_r2,
            'low_mae': low_mae,
            'low_r2': low_r2,
            'avg_high': y_high_test.mean(),
            'avg_low': y_low_test.mean()
        }
        
        return metrics
    
    def train_all_models(self, X: pd.DataFrame, y_dict: Dict[str, pd.Series]) -> Dict:
        """
        訓練所有模型
        
        Args:
            X: 特徵矩陣
            y_dict: Label字典 {'direction', 'high_pct', 'low_pct'}
        
        Returns:
            綜合指標
        """
        print("="*60)
        print("雙模型訓練開始")
        print(f"總樣本數: {len(X)}")
        print(f"特徵維度: {X.shape[1]}")
        print("="*60)
        
        # 訓練方向模型
        direction_metrics = self.train_direction_model(X, y_dict['direction'])
        
        # 訓練價格模型
        price_metrics = self.train_price_models(X, y_dict['high_pct'], y_dict['low_pct'])
        
        # 合併指標
        self.metrics = {**direction_metrics, **price_metrics}
        
        print("\n" + "="*60)
        print("訓練完成!")
        print("="*60)
        
        return self.metrics
    
    def save_models(self, prefix: str = ''):
        """保存模型"""
        if prefix:
            prefix = f"{prefix}_"
        
        direction_path = os.path.join(self.model_dir, f"{prefix}dual_direction.pkl")
        high_path = os.path.join(self.model_dir, f"{prefix}dual_high.pkl")
        low_path = os.path.join(self.model_dir, f"{prefix}dual_low.pkl")
        
        joblib.dump(self.direction_model, direction_path)
        joblib.dump(self.high_model, high_path)
        joblib.dump(self.low_model, low_path)
        
        print(f"\n模型已保存:")
        print(f"  - {direction_path}")
        print(f"  - {high_path}")
        print(f"  - {low_path}")
    
    def load_models(self, prefix: str = '') -> bool:
        """載入模型"""
        if prefix:
            prefix = f"{prefix}_"
        
        direction_path = os.path.join(self.model_dir, f"{prefix}dual_direction.pkl")
        high_path = os.path.join(self.model_dir, f"{prefix}dual_high.pkl")
        low_path = os.path.join(self.model_dir, f"{prefix}dual_low.pkl")
        
        try:
            self.direction_model = joblib.load(direction_path)
            self.high_model = joblib.load(high_path)
            self.low_model = joblib.load(low_path)
            print(f"模型載入成功: {prefix}")
            return True
        except FileNotFoundError:
            print(f"模型檔案不存在: {prefix}")
            return False
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        預測
        
        Returns:
            {
                'direction': array of 0/1,
                'direction_proba': array of probabilities,
                'high_pct': array of high predictions,
                'low_pct': array of low predictions
            }
        """
        if self.direction_model is None or self.high_model is None or self.low_model is None:
            raise ValueError("模型尚未訓練或載入!")
        
        direction_pred = self.direction_model.predict(X)
        direction_proba = self.direction_model.predict_proba(X)[:, 1]
        
        high_pred = self.high_model.predict(X)
        low_pred = self.low_model.predict(X)
        
        return {
            'direction': direction_pred,
            'direction_proba': direction_proba,
            'high_pct': high_pred,
            'low_pct': low_pred
        }
    
    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        """獲取特徵重要性"""
        if self.direction_model is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'direction_importance': self.direction_model.feature_importances_,
            'high_importance': self.high_model.feature_importances_,
            'low_importance': self.low_model.feature_importances_
        })
        
        importance_df['avg_importance'] = importance_df[[
            'direction_importance', 'high_importance', 'low_importance'
        ]].mean(axis=1)
        
        return importance_df.nlargest(top_n, 'avg_importance')


if __name__ == '__main__':
    from utils.dual_model_features import DualModelFeatureExtractor
    
    print("雙模型訓練器測試")
    print("="*60)
    
    # 生成測試數據
    dates = pd.date_range('2024-01-01', periods=2000, freq='15min')
    np.random.seed(42)
    
    base_price = 50000
    prices = base_price + np.random.randn(2000).cumsum() * 100
    
    df = pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': prices + np.random.rand(2000) * 50,
        'low': prices - np.random.rand(2000) * 50,
        'close': prices + np.random.randn(2000) * 20,
        'volume': np.random.randint(1000, 5000, 2000)
    })
    
    # 處理數據
    extractor = DualModelFeatureExtractor()
    df_processed = extractor.process(df, create_labels=True)
    
    X, y_dict = extractor.get_training_data(df_processed)
    
    # 訓練模型
    trainer = DualModelTrainer()
    metrics = trainer.train_all_models(X, y_dict)
    
    # 顯示特徵重要性
    print("\nTop 10 重要特徵:")
    importance = trainer.get_feature_importance(extractor.get_feature_columns(), top_n=10)
    print(importance[['feature', 'avg_importance']].to_string(index=False))
    
    # 保存模型
    trainer.save_models(prefix='test')
    
    # 預測測試
    print("\n預測測試:")
    predictions = trainer.predict(X.head(5))
    print(f"方向預測: {predictions['direction']}")
    print(f"方向機率: {predictions['direction_proba']}")
    print(f"最高價%: {predictions['high_pct']}")
    print(f"最低價%: {predictions['low_pct']}")