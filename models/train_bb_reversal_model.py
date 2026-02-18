import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

class BBReversalModelTrainer:
    """
    BB反轉點模型訓練器
    
    基於有效BB反轉點訓練二元分類模型
    預測: 1 = 做多 (下軌反轉), 0 = 做空 (上軌反轉)
    """
    
    def __init__(self, 
                 model_dir: str = 'models/saved',
                 lgb_params: dict = None):
        
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # LightGBM參數
        self.lgb_params = lgb_params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'min_child_samples': 30,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'is_unbalance': True
        }
        
        self.model = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        訓練模型
        """
        print("\n" + "="*60)
        print("    BB反轉點模型訓練 (LightGBM)")
        print("="*60)
        print(f"特徵數量: {X.shape[1]}")
        print(f"總樣本: {X.shape[0]}")
        print(f"做多樣本: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
        print(f"做空樣本: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
        
        if len(X) < 50:
            raise ValueError(f"樣本數量太少: {len(X)}, 需要至少50個")
        
        # 分割訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n訓練集: {len(X_train)} | 測試集: {len(X_test)}")
        
        # 訓練LightGBM
        self.model = lgb.LGBMClassifier(**self.lgb_params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # 預測
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # 評估
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n分類報告:")
        print(classification_report(y_test, y_pred, target_names=['做空', '做多']))
        
        print(f"\n混淆矩陣:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        metrics = {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def save_model(self, prefix: str = ''):
        """
        保存模型
        """
        if self.model is None:
            raise ValueError("模型尚未訓練")
        
        if prefix:
            prefix = f"{prefix}_"
        
        model_path = os.path.join(self.model_dir, f"{prefix}bb_reversal_lgb.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\n模型已保存: {model_path}")
    
    def load_model(self, prefix: str = ''):
        """
        載入模型
        """
        if prefix:
            prefix = f"{prefix}_"
        
        model_path = os.path.join(self.model_dir, f"{prefix}bb_reversal_lgb.pkl")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"\n模型已載入: {model_path}")
    
    def get_feature_importance(self, feature_names: list, top_n: int = 15) -> pd.DataFrame:
        """
        獲取特徵重要性
        """
        if self.model is None:
            raise ValueError("模型尚未訓練")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


if __name__ == '__main__':
    print("BB反轉點模型訓練器")
    print("="*60)
    print("請在App中使用")