import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from typing import Tuple, Optional

from utils.scalping_feature_extractor import ScalpingFeatureExtractor
from utils.scalping_label_generator import ScalpingLabelGenerator

class ScalpingModelTrainer:
    """
    剝頭皮模型訓練器
    
    使用 LightGBM 進行三分類預測:
    - 0: SHORT
    - 1: LONG
    - 2: NEUTRAL (訓練時過濾掉)
    """
    
    def __init__(self, model_dir: str = 'models/saved'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.feature_extractor = ScalpingFeatureExtractor()
        self.label_generator = ScalpingLabelGenerator()
        
        # LightGBM 參數 (二分類)
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'min_data_in_leaf': 100,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1
        }
    
    def prepare_data(self, df: pd.DataFrame, 
                    target_pct: float = 0.003,
                    lookforward: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        準備訓練數據
        """
        print(f"載入 {len(df)} 根K棒")
        
        # 1. 提取特徵
        print("提取特徵...")
        df_features = self.feature_extractor.extract_features(df)
        
        # 2. 生成標籤
        print("生成標籤...")
        self.label_generator.target_pct = target_pct
        self.label_generator.lookforward = lookforward
        df_labeled = self.label_generator.generate_labels(df_features)
        
        # 3. 過濾有效樣本
        df_train = self.label_generator.filter_training_data(df_labeled)
        
        # 統計
        stats = self.label_generator.get_label_distribution(df_labeled)
        print(f"總樣本: {stats['total_samples']}")
        print(f"做多樣本: {stats['long_signals']} ({stats['long_pct']:.1f}%)")
        print(f"做空樣本: {stats['short_signals']} ({stats['short_pct']:.1f}%)")
        print(f"觀望樣本: {stats['neutral_signals']}")
        print(f"可交易樣本比例: {stats['tradeable_pct']:.1f}%")
        
        return df_train, df_labeled
    
    def train(self, df: pd.DataFrame, 
             target_pct: float = 0.003,
             lookforward: int = 5,
             test_size: float = 0.2) -> dict:
        """
        訓練模型
        """
        # 準備數據
        df_train, df_all = self.prepare_data(df, target_pct, lookforward)
        
        if len(df_train) < 100:
            raise ValueError(f"可訓練樣本太少: {len(df_train)}")
        
        # 提取X, y
        feature_cols = self.feature_extractor.get_feature_columns()
        X = df_train[feature_cols].fillna(0)
        y = df_train['label']
        
        # 分割訓練/驗證集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"
訓練集: {len(X_train)} | 驗證集: {len(X_test)}")
        
        # 訓練 LightGBM
        print("
開始訓練 LightGBM...")
        self.model = lgb.LGBMClassifier(**self.lgb_params, n_estimators=200)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # 評估
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"
訓練集準確率: {train_acc:.2%}")
        print(f"驗證集準確率: {test_acc:.2%}")
        print(f"
分類報告:\n{classification_report(y_test, y_pred_test, target_names=['SHORT', 'LONG'])}")
        
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': self.get_feature_importance(feature_cols)
        }
        
        return metrics
    
    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        """
        獲取特徵重要性
        """
        if self.model is None:
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False).head(top_n)
        
        return importance
    
    def save_model(self, symbol: str, prefix: str = ''):
        """
        保存模型
        """
        if self.model is None:
            print("沒有訓練好的模型")
            return
        
        filename = f"{symbol}_{prefix}_scalping_lgb.pkl" if prefix else f"{symbol}_scalping_lgb.pkl"
        filepath = os.path.join(self.model_dir, filename)
        
        # 保存模型和配置
        model_package = {
            'model': self.model,
            'feature_columns': self.feature_extractor.get_feature_columns(),
            'target_pct': self.label_generator.target_pct,
            'lookforward': self.label_generator.lookforward
        }
        
        joblib.dump(model_package, filepath)
        print(f"模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        """
        載入模型
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        model_package = joblib.load(filepath)
        self.model = model_package['model']
        
        print(f"模型已載入: {filepath}")
        return model_package

if __name__ == '__main__':
    print("Scalping Model Trainer")
    print("請在App中使用")