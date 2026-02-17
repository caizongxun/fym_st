import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bb_bounce_features import BBBounceFeatureExtractor

class BBBounceModelTrainer:
    """
    BB反彈預測模型訓練器
    
    訓練兩個模型:
    1. upper_bounce_model: 預測觸碰上軌後是否反彈(做空機會)
    2. lower_bounce_model: 預測觸碰下軌後是否反彈(做多機會)
    """
    
    def __init__(self, model_dir: str = 'models/saved'):
        self.model_dir = model_dir
        self.feature_extractor = BBBounceFeatureExtractor()
        self.upper_model = None
        self.lower_model = None
        
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_training_data(self, df: pd.DataFrame, touch_type: str):
        """
        準備訓練數據
        
        Args:
            df: 處理過的DataFrame(含特徵和標籤)
            touch_type: 'upper' 或 'lower'
        """
        # 獲取特徵列
        feature_cols = self.feature_extractor.get_feature_columns()
        
        if touch_type == 'upper':
            # 只使用觸碰上軌的樣本
            mask = df['touch_upper'] == 1
            label_col = 'upper_bounce_label'
        else:
            # 只使用觸碰下軌的樣本
            mask = df['touch_lower'] == 1
            label_col = 'lower_bounce_label'
        
        # 篩選數據
        df_filtered = df[mask].copy()
        
        # 移除NaN
        df_filtered = df_filtered.dropna(subset=feature_cols + [label_col])
        
        print(f"\n{touch_type.upper()} 軌道訓練數據:")
        print(f"  總樣本數: {len(df_filtered)}")
        print(f"  正樣本(反彈): {df_filtered[label_col].sum()} ({df_filtered[label_col].mean()*100:.1f}%)")
        print(f"  負樣本(未反彈): {(df_filtered[label_col]==0).sum()} ({(1-df_filtered[label_col].mean())*100:.1f}%)")
        
        # 檢查趨勢狀態分佈
        print(f"\n  趨勢狀態分佈:")
        print(df_filtered['trend_state'].value_counts())
        
        X = df_filtered[feature_cols]
        y = df_filtered[label_col]
        
        return X, y, df_filtered
    
    def train_model(self, X, y, model_name: str, use_class_weights: bool = True):
        """
        訓練XGBoost模型
        """
        # 分割訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n訓練 {model_name}...")
        print(f"  訓練集: {len(X_train)} 樣本")
        print(f"  測試集: {len(X_test)} 樣本")
        
        # 計算類別權重(處理不平衡)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if use_class_weights else 1
        
        # XGBoost參數
        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': scale_pos_weight,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist'
        }
        
        # 訓練模型
        model = xgb.XGBClassifier(**params)
        
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # 預測
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 評估
        print(f"\n{model_name} 測試集結果:")
        print(classification_report(y_test, y_pred, target_names=['No Bounce', 'Bounce']))
        
        print(f"\nAUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        print("\n混淆矩陣:")
        print(confusion_matrix(y_test, y_pred))
        
        # 特徵重要性
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 15 重要特徵:")
        print(feature_importance.head(15).to_string(index=False))
        
        return model, feature_importance
    
    def train_both_models(self, df: pd.DataFrame):
        """
        訓練上軌和下軌兩個模型
        """
        print("="*70)
        print("BB反彈預測模型訓練")
        print("="*70)
        
        # 1. 訓練上軌反彈模型(做空機會)
        print("\n" + "="*70)
        print("上軌反彈模型(做空信號)")
        print("="*70)
        
        X_upper, y_upper, df_upper = self.prepare_training_data(df, 'upper')
        
        if len(X_upper) < 50:
            print("\n警告: 上軌樣本數過少,跳過訓練")
        else:
            self.upper_model, upper_importance = self.train_model(
                X_upper, y_upper, 'Upper Bounce Model'
            )
        
        # 2. 訓練下軌反彈模型(做多機會)
        print("\n" + "="*70)
        print("下軌反彈模型(做多信號)")
        print("="*70)
        
        X_lower, y_lower, df_lower = self.prepare_training_data(df, 'lower')
        
        if len(X_lower) < 50:
            print("\n警告: 下軌樣本數過少,跳過訓練")
        else:
            self.lower_model, lower_importance = self.train_model(
                X_lower, y_lower, 'Lower Bounce Model'
            )
    
    def save_models(self, prefix: str = ''):
        """
        保存模型
        
        Args:
            prefix: 檔案名前綴(支持多幣種,例如'BTCUSDT')
        """
        if self.upper_model is not None:
            if prefix:
                upper_path = os.path.join(self.model_dir, f'{prefix}_bb_upper_bounce_model.pkl')
            else:
                upper_path = os.path.join(self.model_dir, 'bb_upper_bounce_model.pkl')
            joblib.dump(self.upper_model, upper_path)
            print(f"\n上軌模型已保存: {upper_path}")
        
        if self.lower_model is not None:
            if prefix:
                lower_path = os.path.join(self.model_dir, f'{prefix}_bb_lower_bounce_model.pkl')
            else:
                lower_path = os.path.join(self.model_dir, 'bb_lower_bounce_model.pkl')
            joblib.dump(self.lower_model, lower_path)
            print(f"下軌模型已保存: {lower_path}")
    
    def load_models(self, prefix: str = ''):
        """
        載入模型
        
        Args:
            prefix: 檔案名前綴(支持多幣種,例如'BTCUSDT')
        """
        if prefix:
            upper_path = os.path.join(self.model_dir, f'{prefix}_bb_upper_bounce_model.pkl')
            lower_path = os.path.join(self.model_dir, f'{prefix}_bb_lower_bounce_model.pkl')
        else:
            upper_path = os.path.join(self.model_dir, 'bb_upper_bounce_model.pkl')
            lower_path = os.path.join(self.model_dir, 'bb_lower_bounce_model.pkl')
        
        if os.path.exists(upper_path):
            self.upper_model = joblib.load(upper_path)
            print(f"已載入上軌模型: {upper_path}")
        
        if os.path.exists(lower_path):
            self.lower_model = joblib.load(lower_path)
            print(f"已載入下軌模型: {lower_path}")


if __name__ == '__main__':
    print("示例: BB反彈模型訓練\n")
    print("使用方法:")
    print("""
    # 1. 載入數據
    df = pd.read_csv('your_data.csv')
    
    # 2. 處理特徵
    from utils.bb_bounce_features import BBBounceFeatureExtractor
    extractor = BBBounceFeatureExtractor()
    df_processed = extractor.process(df, create_labels=True)
    
    # 3. 訓練模型
    trainer = BBBounceModelTrainer()
    trainer.train_both_models(df_processed)
    trainer.save_models(prefix='BTCUSDT')  # 多幣種支持
    
    # 4. 預測
    trainer.load_models(prefix='BTCUSDT')
    
    # 獲取觸碰上軌的樣本
    upper_samples = df_processed[df_processed['touch_upper'] == 1]
    X_upper = upper_samples[extractor.get_feature_columns()]
    upper_bounce_prob = trainer.upper_model.predict_proba(X_upper)[:, 1]
    
    # 獲取觸碰下軌的樣本
    lower_samples = df_processed[df_processed['touch_lower'] == 1]
    X_lower = lower_samples[extractor.get_feature_columns()]
    lower_bounce_prob = trainer.lower_model.predict_proba(X_lower)[:, 1]
    """)