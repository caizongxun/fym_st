import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import pickle
import os

class DualModelTrainerLGB:
    """
    雙模型訓練器 - LightGBM版本
    
    模型A: 預測漨跌方向 (LGBM Classifier)
    模型B: 預測最高/最低價 (LGBM Regressor)
    
    LightGBM優勢:
    - 訓練速度快 3-10倍
    - 更高的準確率
    - 更好的特徵重要性分析
    - 內建類別權重平衡
    """
    
    def __init__(self, 
                 model_dir: str = 'models/saved',
                 direction_params: dict = None,
                 price_params: dict = None):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # LightGBM預設參數 - 分類模型
        self.direction_params = direction_params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 10,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'is_unbalance': True  # 自動平衡類別
        }
        
        # LightGBM預設參數 - 回歸模型
        self.price_params = price_params or {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 10,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        self.direction_model = None
        self.high_model = None
        self.low_model = None
    
    def train_direction_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        print("\n===== 訓練模型A: 漨跌方向預測 (LightGBM) =====")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"訓練集: {len(X_train)} | 測試集: {len(X_test)}")
        print(f"訓練集分佈: {y_train.value_counts().to_dict()}")
        
        self.direction_model = lgb.LGBMClassifier(**self.direction_params)
        
        self.direction_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        y_pred = self.direction_model.predict(X_test)
        y_pred_proba = self.direction_model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n分類報告:")
        print(classification_report(y_test, y_pred, target_names=['下跌', '上漨']))
        
        return {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def train_price_models(self, X: pd.DataFrame, y_high: pd.Series, y_low: pd.Series) -> Dict:
        print("\n===== 訓練模型B: 最高/最低價預測 (LightGBM) =====")
        
        X_train, X_test, yh_train, yh_test, yl_train, yl_test = train_test_split(
            X, y_high, y_low, test_size=0.2, random_state=42
        )
        
        # 訓練最高價模型
        print("\n訓練最高價預測模型...")
        self.high_model = lgb.LGBMRegressor(**self.price_params)
        self.high_model.fit(
            X_train, yh_train,
            eval_set=[(X_test, yh_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        yh_pred = self.high_model.predict(X_test)
        high_mae = mean_absolute_error(yh_test, yh_pred)
        
        # 訓練最低價模型
        print("訓練最低價預測模型...")
        self.low_model = lgb.LGBMRegressor(**self.price_params)
        self.low_model.fit(
            X_train, yl_train,
            eval_set=[(X_test, yl_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        yl_pred = self.low_model.predict(X_test)
        low_mae = mean_absolute_error(yl_test, yl_pred)
        
        print(f"\n最高價 MAE: {high_mae:.4f}%")
        print(f"最低價 MAE: {low_mae:.4f}%")
        
        return {
            'high_mae': high_mae,
            'low_mae': low_mae,
            'yh_test': yh_test,
            'yh_pred': yh_pred,
            'yl_test': yl_test,
            'yl_pred': yl_pred
        }
    
    def train_all_models(self, X: pd.DataFrame, y_dict: Dict[str, pd.Series]) -> Dict:
        print("\n" + "="*60)
        print("    LightGBM 雙模型訓練")
        print("="*60)
        print(f"特徵數量: {X.shape[1]}")
        print(f"樣本數量: {X.shape[0]}")
        
        # 檢查無限值
        if np.isinf(X).any().any():
            print("⚠️  警告: X中有無限值")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if X.isna().any().any():
            print("⚠️  警告: X中有NaN")
            X = X.fillna(0)
        
        # 訓練模型A
        direction_metrics = self.train_direction_model(X, y_dict['direction'])
        
        # 訓練模型B
        price_metrics = self.train_price_models(X, y_dict['high_pct'], y_dict['low_pct'])
        
        combined_metrics = {
            'accuracy': direction_metrics['accuracy'],
            'high_mae': price_metrics['high_mae'],
            'low_mae': price_metrics['low_mae']
        }
        
        print("\n" + "="*60)
        print("    訓練完成")
        print("="*60)
        print(f"方向準確率: {direction_metrics['accuracy']:.4f}")
        print(f"最高價MAE: {price_metrics['high_mae']:.4f}%")
        print(f"最低價MAE: {price_metrics['low_mae']:.4f}%")
        
        return combined_metrics
    
    def save_models(self, prefix: str = ''):
        if prefix:
            prefix = f"{prefix}_"
        
        direction_path = os.path.join(self.model_dir, f"{prefix}dual_direction_lgb.pkl")
        high_path = os.path.join(self.model_dir, f"{prefix}dual_high_lgb.pkl")
        low_path = os.path.join(self.model_dir, f"{prefix}dual_low_lgb.pkl")
        
        with open(direction_path, 'wb') as f:
            pickle.dump(self.direction_model, f)
        
        with open(high_path, 'wb') as f:
            pickle.dump(self.high_model, f)
        
        with open(low_path, 'wb') as f:
            pickle.dump(self.low_model, f)
        
        print(f"\n模型已保存:")
        print(f"  - {direction_path}")
        print(f"  - {high_path}")
        print(f"  - {low_path}")
    
    def load_models(self, prefix: str = ''):
        if prefix:
            prefix = f"{prefix}_"
        
        direction_path = os.path.join(self.model_dir, f"{prefix}dual_direction_lgb.pkl")
        high_path = os.path.join(self.model_dir, f"{prefix}dual_high_lgb.pkl")
        low_path = os.path.join(self.model_dir, f"{prefix}dual_low_lgb.pkl")
        
        with open(direction_path, 'rb') as f:
            self.direction_model = pickle.load(f)
        
        with open(high_path, 'rb') as f:
            self.high_model = pickle.load(f)
        
        with open(low_path, 'rb') as f:
            self.low_model = pickle.load(f)
        
        print(f"\n模型已載入:")
        print(f"  - {direction_path}")
        print(f"  - {high_path}")
        print(f"  - {low_path}")
    
    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        if self.direction_model is None:
            raise ValueError("模型尚未訓練")
        
        direction_importance = self.direction_model.feature_importances_
        high_importance = self.high_model.feature_importances_
        low_importance = self.low_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'direction_importance': direction_importance,
            'high_importance': high_importance,
            'low_importance': low_importance
        })
        
        importance_df['avg_importance'] = importance_df[[
            'direction_importance', 'high_importance', 'low_importance'
        ]].mean(axis=1)
        
        importance_df = importance_df.sort_values('avg_importance', ascending=False)
        
        return importance_df.head(top_n)


if __name__ == '__main__':
    print("雙模型訓練器 - LightGBM版")
    print("="*60)
    
    # 測試數據
    np.random.seed(42)
    n_samples = 5000
    n_features = 50
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y_dict = {
        'direction': pd.Series(np.random.randint(0, 2, n_samples)),
        'high_pct': pd.Series(np.random.uniform(0, 2, n_samples)),
        'low_pct': pd.Series(np.random.uniform(-2, 0, n_samples)),
        'next_high': pd.Series(np.random.uniform(50000, 51000, n_samples)),
        'next_low': pd.Series(np.random.uniform(49000, 50000, n_samples))
    }
    
    trainer = DualModelTrainerLGB()
    metrics = trainer.train_all_models(X, y_dict)
    
    importance = trainer.get_feature_importance(X.columns.tolist(), top_n=10)
    print("\nTop 10 重要特徵:")
    print(importance[['feature', 'avg_importance']])