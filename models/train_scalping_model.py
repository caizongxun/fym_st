import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from typing import Tuple, Optional

from utils.scalping_feature_extractor import ScalpingFeatureExtractor
from utils.scalping_label_generator import ScalpingLabelGenerator

class ScalpingModelTrainer:
    """
    剝頭皮模型訓練器 (OOS驗證 + 多模型支持)
    
    支持模型:
    - LightGBM (lightgbm)
    - XGBoost (xgboost)
    - CatBoost (catboost)
    - Ensemble (組合模型)
    """
    
    def __init__(self, model_dir: str = 'models/saved', model_type: str = 'lightgbm'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model_type = model_type
        self.model = None
        self.models = {}  # 用於Ensemble
        self.feature_extractor = ScalpingFeatureExtractor()
        self.label_generator = ScalpingLabelGenerator()
        self.actual_feature_columns = []
        
        # 模型參數
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
        
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 100,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': 0
        }
        
        self.catboost_params = {
            'iterations': 200,
            'learning_rate': 0.05,
            'depth': 7,
            'l2_leaf_reg': 3,
            'verbose': False,
            'random_seed': 42
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
    
    def train_with_oos(self, df_full: pd.DataFrame,
                      target_pct: float = 0.003,
                      lookforward: int = 5,
                      oos_days: int = 30,
                      test_size: float = 0.2) -> dict:
        """
        訓練模型 (含OOS驗證)
        """
        # 分割OOS
        oos_candles = oos_days * 96
        df_oos = df_full.tail(oos_candles).copy()
        df_train_full = df_full.iloc[:-oos_candles].copy()
        
        print(f"\n====== OOS切割 ======")
        print(f"訓練集: {len(df_train_full)} 根")
        print(f"OOS測試集: {len(df_oos)} 根 ({oos_days}天)")
        
        # 準備訓練數據
        df_train, _ = self.prepare_data(df_train_full, target_pct, lookforward)
        
        if len(df_train) < 100:
            raise ValueError(f"可訓練樣本太少: {len(df_train)}")
        
        # 準備OOS數據
        print(f"\n====== 處理OOS數據 ======")
        df_oos_processed, _ = self.prepare_data(df_oos, target_pct, lookforward)
        
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
        
        # 分割訓練/驗證集 (在訓練集內部)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=test_size, random_state=42, stratify=y_train_full
        )
        
        print(f"\n訓練集: {len(X_train)} | 驗證集: {len(X_val)} | OOS: {len(X_oos)}")
        
        # 訓練模型
        print(f"\n====== 訓練 {self.model_type.upper()} ======")
        
        if self.model_type == 'ensemble':
            metrics = self._train_ensemble(X_train, y_train, X_val, y_val, X_oos, y_oos)
        else:
            self.model = self._train_single_model(X_train, y_train, X_val, y_val)
            metrics = self._evaluate_with_oos(X_train, y_train, X_val, y_val, X_oos, y_oos)
        
        return metrics
    
    def _train_single_model(self, X_train, y_train, X_val, y_val):
        """
        訓練單一模型
        """
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
        
        elif self.model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost 未安裝: pip install catboost")
            model = cb.CatBoostClassifier(**self.catboost_params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50
            )
        
        else:
            raise ValueError(f"不支持的模型: {self.model_type}")
        
        return model
    
    def _train_ensemble(self, X_train, y_train, X_val, y_val, X_oos, y_oos):
        """
        訓練Ensemble模型
        """
        print("訓練 LightGBM...")
        lgb_model = lgb.LGBMClassifier(**self.lgb_params, n_estimators=200)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        print("訓練 XGBoost...")
        xgb_model = xgb.XGBClassifier(**self.xgb_params, n_estimators=200, early_stopping_rounds=50)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        self.models = {'lgb': lgb_model, 'xgb': xgb_model}
        
        # 評估Ensemble
        y_pred_train = self._predict_ensemble(X_train)
        y_pred_val = self._predict_ensemble(X_val)
        y_pred_oos = self._predict_ensemble(X_oos)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'val_accuracy': accuracy_score(y_val, y_pred_val),
            'oos_accuracy': accuracy_score(y_oos, y_pred_oos),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'oos_samples': len(X_oos),
            'feature_importance': None
        }
        
        return metrics
    
    def _predict_ensemble(self, X):
        """
        Ensemble預測 (投票)
        """
        predictions = []
        for model in self.models.values():
            predictions.append(model.predict(X))
        
        # 簡單投票
        predictions = np.array(predictions)
        final_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions)
        return final_pred
    
    def _evaluate_with_oos(self, X_train, y_train, X_val, y_val, X_oos, y_oos):
        """
        評估模型 (含OOS)
        """
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        y_pred_oos = self.model.predict(X_oos)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        oos_acc = accuracy_score(y_oos, y_pred_oos)
        
        print(f"\n====== 結果 ======")
        print(f"訓練集準確率: {train_acc:.2%}")
        print(f"驗證集準確率: {val_acc:.2%}")
        print(f"OOS準確率: {oos_acc:.2%}")
        print(f"\nOOS 分類報告:\n{classification_report(y_oos, y_pred_oos, target_names=['SHORT', 'LONG'])}")
        
        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'oos_accuracy': oos_acc,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'oos_samples': len(X_oos),
            'feature_importance': self.get_feature_importance(self.actual_feature_columns)
        }
        
        return metrics
    
    def train(self, df: pd.DataFrame, 
             target_pct: float = 0.003,
             lookforward: int = 5,
             test_size: float = 0.2) -> dict:
        """
        普通訓練 (無OOS)
        """
        df_train, df_all = self.prepare_data(df, target_pct, lookforward)
        
        if len(df_train) < 100:
            raise ValueError(f"可訓練樣本太少: {len(df_train)}")
        
        all_possible_features = self.feature_extractor.get_all_possible_features()
        available_features = [f for f in all_possible_features if f in df_train.columns]
        self.actual_feature_columns = available_features
        
        print(f"可用特徵數: {len(self.actual_feature_columns)}")
        
        X = df_train[self.actual_feature_columns].fillna(0)
        y = df_train['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n訓練集: {len(X_train)} | 驗證集: {len(X_test)}")
        
        self.model = self._train_single_model(X_train, y_train, X_test, y_test)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n訓練集準確率: {train_acc:.2%}")
        print(f"驗證集準確率: {test_acc:.2%}")
        print(f"\n分類報告:\n{classification_report(y_test, y_pred_test, target_names=['SHORT', 'LONG'])}")
        
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
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
        if self.model is None and not self.models:
            print("沒有訓練好的模型")
            return
        
        filename = f"{symbol}_{prefix}_scalping_{self.model_type}.pkl" if prefix else f"{symbol}_scalping_{self.model_type}.pkl"
        filepath = os.path.join(self.model_dir, filename)
        
        model_package = {
            'model': self.model,
            'models': self.models,
            'model_type': self.model_type,
            'feature_columns': self.actual_feature_columns,
            'target_pct': self.label_generator.target_pct,
            'lookforward': self.label_generator.lookforward
        }
        
        joblib.dump(model_package, filepath)
        print(f"模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        model_package = joblib.load(filepath)
        self.model = model_package['model']
        self.models = model_package.get('models', {})
        self.model_type = model_package.get('model_type', 'lightgbm')
        
        print(f"模型已載入: {filepath}")
        return model_package

if __name__ == '__main__':
    print("Scalping Model Trainer with OOS")
    print("請在App中使用")