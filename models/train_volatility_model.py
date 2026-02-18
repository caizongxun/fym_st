import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os

from utils.volatility_feature_extractor import VolatilityFeatureExtractor
from utils.volatility_label_generator import VolatilityLabelGenerator

class VolatilityModelTrainer:
    """
    波動率預測模型訓練器
    """
    
    def __init__(self, model_dir: str = 'models/saved', model_type: str = 'lightgbm'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model_type = model_type
        self.model = None
        self.feature_extractor = VolatilityFeatureExtractor()
        self.label_generator = VolatilityLabelGenerator()
        self.actual_feature_columns = []
        
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
            'min_data_in_leaf': 100
        }
        
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 100,
            'verbosity': 0
        }
    
    def prepare_data(self, df: pd.DataFrame,
                    vol_threshold: float = 0.005,
                    lookforward: int = 3):
        
        print(f"載入 {len(df)} 根K棒")
        
        # 1. 提取特徵
        print("提取特徵...")
        df_features = self.feature_extractor.extract_features(df)
        
        # 2. 生成標籤
        print("生成標籤...")
        self.label_generator.vol_threshold = vol_threshold
        self.label_generator.lookforward = lookforward
        df_labeled = self.label_generator.generate_labels(df_features)
        
        # 3. 統計
        stats = self.label_generator.get_label_distribution(df_labeled)
        print(f"總樣本: {stats['total_samples']}")
        print(f"高波動樣本: {stats['high_vol_signals']} ({stats['high_vol_pct']:.1f}%)")
        print(f"低波動樣本: {stats['low_vol_signals']}")
        print(f"平均波動率: {stats['avg_volatility']:.4f}")
        print(f"平均達峰時間: {stats['avg_time_to_peak']:.1f} 根K線")
        
        return df_labeled
    
    def train_with_oos(self, df_full: pd.DataFrame,
                      vol_threshold: float = 0.005,
                      lookforward: int = 3,
                      oos_days: int = 30,
                      test_size: float = 0.2) -> dict:
        
        # 分割OOS
        oos_candles = oos_days * 96
        df_oos = df_full.tail(oos_candles).copy()
        df_train_full = df_full.iloc[:-oos_candles].copy()
        
        print(f"\n====== OOS切割 ======")
        print(f"訓練集: {len(df_train_full)} 根")
        print(f"OOS測試集: {len(df_oos)} 根 ({oos_days}天)")
        
        # 準備訓練數據
        df_train = self.prepare_data(df_train_full, vol_threshold, lookforward)
        
        # 準備OOS數據
        print(f"\n====== 處理OOS數據 ======")
        df_oos_processed = self.prepare_data(df_oos, vol_threshold, lookforward)
        
        # 動態檢測可用特徵
        all_possible_features = self.feature_extractor.get_all_possible_features()
        available_features = [f for f in all_possible_features if f in df_train.columns]
        self.actual_feature_columns = available_features
        
        print(f"\n可用特徵數: {len(self.actual_feature_columns)}")
        
        # 提取X, y
        X_train_full = df_train[self.actual_feature_columns].fillna(0)
        y_train_full = df_train['label']
        
        X_oos = df_oos_processed[self.actual_feature_columns].fillna(0)
        y_oos = df_oos_processed['label']
        
        # 分割訓練/驗證集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=test_size, random_state=42, stratify=y_train_full
        )
        
        print(f"\n訓練集: {len(X_train)} | 驗證集: {len(X_val)} | OOS: {len(X_oos)}")
        
        # 訓練模型
        print(f"\n====== 訓練 {self.model_type.upper()} ======")
        self.model = self._train_single_model(X_train, y_train, X_val, y_val)
        metrics = self._evaluate_with_oos(X_train, y_train, X_val, y_val, X_oos, y_oos)
        
        return metrics
    
    def _train_single_model(self, X_train, y_train, X_val, y_val):
        if self.model_type == 'lightgbm':
            model = lgb.LGBMClassifier(**self.lgb_params, n_estimators=200)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        elif self.model_type == 'xgboost':
            model = xgb.XGBClassifier(**self.xgb_params, n_estimators=200, early_stopping_rounds=50)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            raise ValueError(f"不支持的模型: {self.model_type}")
        
        return model
    
    def _evaluate_with_oos(self, X_train, y_train, X_val, y_val, X_oos, y_oos):
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        y_pred_oos = self.model.predict(X_oos)
        
        # 機率預測
        y_prob_train = self.model.predict_proba(X_train)[:, 1]
        y_prob_val = self.model.predict_proba(X_val)[:, 1]
        y_prob_oos = self.model.predict_proba(X_oos)[:, 1]
        
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        oos_acc = accuracy_score(y_oos, y_pred_oos)
        
        train_auc = roc_auc_score(y_train, y_prob_train)
        val_auc = roc_auc_score(y_val, y_prob_val)
        oos_auc = roc_auc_score(y_oos, y_prob_oos)
        
        print(f"\n====== 結果 ======")
        print(f"訓練集 - 準確率: {train_acc:.2%} | AUC: {train_auc:.4f}")
        print(f"驗證集 - 準確率: {val_acc:.2%} | AUC: {val_auc:.4f}")
        print(f"OOS測試 - 準確率: {oos_acc:.2%} | AUC: {oos_auc:.4f}")
        print(f"\nOOS 分類報告:\n{classification_report(y_oos, y_pred_oos, target_names=['LOW_VOL', 'HIGH_VOL'])}")
        
        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'oos_accuracy': oos_acc,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'oos_auc': oos_auc,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'oos_samples': len(X_oos),
            'feature_importance': self.get_feature_importance(self.actual_feature_columns)
        }
        
        return metrics
    
    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False).head(top_n)
        
        return importance
    
    def save_model(self, symbol: str, prefix: str = ''):
        if self.model is None:
            print("沒有訓練好的模型")
            return
        
        filename = f"{symbol}_{prefix}_volatility_{self.model_type}.pkl" if prefix else f"{symbol}_volatility_{self.model_type}.pkl"
        filepath = os.path.join(self.model_dir, filename)
        
        model_package = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.actual_feature_columns,
            'vol_threshold': self.label_generator.vol_threshold,
            'lookforward': self.label_generator.lookforward
        }
        
        joblib.dump(model_package, filepath)
        print(f"模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        model_package = joblib.load(filepath)
        self.model = model_package['model']
        self.model_type = model_package.get('model_type', 'lightgbm')
        
        print(f"模型已載入: {filepath}")
        return model_package

if __name__ == '__main__':
    print("Volatility Model Trainer")
    print("請在App中使用")