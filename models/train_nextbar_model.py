import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

from utils.nextbar_feature_extractor import NextBarFeatureExtractor
from utils.nextbar_label_generator import NextBarLabelGenerator

class NextBarModelTrainer:
    """
    下一根K棒高低點預測模型訓練器
    
    使用回歸模型預測下一根K棒的:
    - high_pct: 最高價相對當前close的百分比
    - low_pct: 最低價相對當前close的百分比
    """
    
    def __init__(self, model_dir: str = 'models/saved', model_type: str = 'xgboost'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model_type = model_type
        self.model_high = None  # 預測 high_pct
        self.model_low = None   # 預測 low_pct
        self.feature_extractor = NextBarFeatureExtractor()
        self.label_generator = NextBarLabelGenerator()
        self.actual_feature_columns = []
        
        # XGBoost 參數 (回歸)
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 50,
            'verbosity': 0
        }
        
        # LightGBM 參數 (回歸)
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 6,
            'min_data_in_leaf': 50
        }
    
    def prepare_data(self, df: pd.DataFrame, max_range_pct: float = 0.015):
        print(f"載入 {len(df)} 根K棒")
        
        # 1. 提取特徵
        print("提取特徵...")
        df_features = self.feature_extractor.extract_features(df)
        
        # 2. 生成標籤
        print("生成標籤...")
        df_labeled = self.label_generator.generate_labels(df_features)
        
        # 3. 統計
        stats = self.label_generator.get_label_statistics(df_labeled)
        print(f"\n標籤統計:")
        print(f"總樣本: {stats['total_samples']}")
        print(f"平均high_pct: {stats['avg_high_pct']:.4f} ({stats['avg_high_pct']*100:.2f}%)")
        print(f"平均low_pct: {stats['avg_low_pct']:.4f} ({stats['avg_low_pct']*100:.2f}%)")
        print(f"平均區間: {stats['avg_range_pct']:.4f} ({stats['avg_range_pct']*100:.2f}%)")
        print(f"標準差high: {stats['std_high_pct']:.4f}")
        print(f"標準差low: {stats['std_low_pct']:.4f}")
        
        # 4. 過濾異常數據
        print(f"\n過濾異常波動 (range > {max_range_pct:.2%})...")
        df_filtered = self.label_generator.filter_training_data(df_labeled, max_range_pct)
        print(f"過濾後: {len(df_filtered)} 樣本")
        
        return df_filtered
    
    def train_with_oos(self, df_full: pd.DataFrame,
                      max_range_pct: float = 0.015,
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
        df_train = self.prepare_data(df_train_full, max_range_pct)
        
        # 準備OOS數據
        print(f"\n====== 處理OOS數據 ======")
        df_oos_processed = self.prepare_data(df_oos, max_range_pct)
        
        # 動態檢測可用特徵
        all_possible_features = self.feature_extractor.get_all_possible_features()
        available_features = [f for f in all_possible_features if f in df_train.columns]
        self.actual_feature_columns = available_features
        
        print(f"\n可用特徵數: {len(self.actual_feature_columns)}")
        
        # 提取X, y
        X_train_full = df_train[self.actual_feature_columns].fillna(0)
        y_high_train_full = df_train['next_high_pct']
        y_low_train_full = df_train['next_low_pct']
        
        X_oos = df_oos_processed[self.actual_feature_columns].fillna(0)
        y_high_oos = df_oos_processed['next_high_pct']
        y_low_oos = df_oos_processed['next_low_pct']
        
        # 分割訓練/驗證集
        X_train, X_val, y_high_train, y_high_val = train_test_split(
            X_train_full, y_high_train_full, test_size=test_size, random_state=42
        )
        _, _, y_low_train, y_low_val = train_test_split(
            X_train_full, y_low_train_full, test_size=test_size, random_state=42
        )
        
        print(f"\n訓練集: {len(X_train)} | 驗證集: {len(X_val)} | OOS: {len(X_oos)}")
        
        # 訓練模型
        print(f"\n====== 訓練 {self.model_type.upper()} ======")
        self.model_high, self.model_low = self._train_models(
            X_train, y_high_train, y_low_train,
            X_val, y_high_val, y_low_val
        )
        
        # 評估
        metrics = self._evaluate_with_oos(
            X_train, y_high_train, y_low_train,
            X_val, y_high_val, y_low_val,
            X_oos, y_high_oos, y_low_oos
        )
        
        return metrics
    
    def _train_models(self, X_train, y_high_train, y_low_train,
                     X_val, y_high_val, y_low_val):
        
        print("\n訓練 HIGH 預測模型...")
        if self.model_type == 'xgboost':
            model_high = xgb.XGBRegressor(**self.xgb_params, n_estimators=200, early_stopping_rounds=30)
            model_high.fit(
                X_train, y_high_train,
                eval_set=[(X_val, y_high_val)],
                verbose=False
            )
        else:  # lightgbm
            model_high = lgb.LGBMRegressor(**self.lgb_params, n_estimators=200)
            model_high.fit(
                X_train, y_high_train,
                eval_set=[(X_val, y_high_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
        
        print("訓練 LOW 預測模型...")
        if self.model_type == 'xgboost':
            model_low = xgb.XGBRegressor(**self.xgb_params, n_estimators=200, early_stopping_rounds=30)
            model_low.fit(
                X_train, y_low_train,
                eval_set=[(X_val, y_low_val)],
                verbose=False
            )
        else:  # lightgbm
            model_low = lgb.LGBMRegressor(**self.lgb_params, n_estimators=200)
            model_low.fit(
                X_train, y_low_train,
                eval_set=[(X_val, y_low_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
        
        return model_high, model_low
    
    def _evaluate_with_oos(self, X_train, y_high_train, y_low_train,
                          X_val, y_high_val, y_low_val,
                          X_oos, y_high_oos, y_low_oos):
        
        # 預測
        y_high_pred_train = self.model_high.predict(X_train)
        y_high_pred_val = self.model_high.predict(X_val)
        y_high_pred_oos = self.model_high.predict(X_oos)
        
        y_low_pred_train = self.model_low.predict(X_train)
        y_low_pred_val = self.model_low.predict(X_val)
        y_low_pred_oos = self.model_low.predict(X_oos)
        
        # 評估指標
        metrics = {
            # HIGH 指標
            'high_train_mae': mean_absolute_error(y_high_train, y_high_pred_train),
            'high_val_mae': mean_absolute_error(y_high_val, y_high_pred_val),
            'high_oos_mae': mean_absolute_error(y_high_oos, y_high_pred_oos),
            'high_oos_rmse': np.sqrt(mean_squared_error(y_high_oos, y_high_pred_oos)),
            
            # LOW 指標
            'low_train_mae': mean_absolute_error(y_low_train, y_low_pred_train),
            'low_val_mae': mean_absolute_error(y_low_val, y_low_pred_val),
            'low_oos_mae': mean_absolute_error(y_low_oos, y_low_pred_oos),
            'low_oos_rmse': np.sqrt(mean_squared_error(y_low_oos, y_low_pred_oos)),
            
            # 區間指標
            'range_oos_mae': mean_absolute_error(
                y_high_oos - y_low_oos,
                y_high_pred_oos - y_low_pred_oos
            ),
            
            # 樣本數
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'oos_samples': len(X_oos),
            
            # 特徵重要性
            'feature_importance_high': self.get_feature_importance(self.model_high, self.actual_feature_columns),
            'feature_importance_low': self.get_feature_importance(self.model_low, self.actual_feature_columns)
        }
        
        print(f"\n====== 結果 ======")
        print(f"\nHIGH 預測:")
        print(f"  訓練MAE: {metrics['high_train_mae']:.6f} ({metrics['high_train_mae']*100:.3f}%)")
        print(f"  驗證MAE: {metrics['high_val_mae']:.6f} ({metrics['high_val_mae']*100:.3f}%)")
        print(f"  OOS MAE: {metrics['high_oos_mae']:.6f} ({metrics['high_oos_mae']*100:.3f}%)")
        print(f"  OOS RMSE: {metrics['high_oos_rmse']:.6f} ({metrics['high_oos_rmse']*100:.3f}%)")
        
        print(f"\nLOW 預測:")
        print(f"  訓練MAE: {metrics['low_train_mae']:.6f} ({metrics['low_train_mae']*100:.3f}%)")
        print(f"  驗證MAE: {metrics['low_val_mae']:.6f} ({metrics['low_val_mae']*100:.3f}%)")
        print(f"  OOS MAE: {metrics['low_oos_mae']:.6f} ({metrics['low_oos_mae']*100:.3f}%)")
        print(f"  OOS RMSE: {metrics['low_oos_rmse']:.6f} ({metrics['low_oos_rmse']*100:.3f}%)")
        
        print(f"\n區間預測:")
        print(f"  OOS MAE: {metrics['range_oos_mae']:.6f} ({metrics['range_oos_mae']*100:.3f}%)")
        
        return metrics
    
    def get_feature_importance(self, model, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        if model is None:
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False).head(top_n)
        
        return importance
    
    def save_model(self, symbol: str, prefix: str = ''):
        if self.model_high is None or self.model_low is None:
            print("沒有訓練好的模型")
            return
        
        filename = f"{symbol}_{prefix}_nextbar_{self.model_type}.pkl" if prefix else f"{symbol}_nextbar_{self.model_type}.pkl"
        filepath = os.path.join(self.model_dir, filename)
        
        model_package = {
            'model_high': self.model_high,
            'model_low': self.model_low,
            'model_type': self.model_type,
            'feature_columns': self.actual_feature_columns
        }
        
        joblib.dump(model_package, filepath)
        print(f"\n模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        model_package = joblib.load(filepath)
        self.model_high = model_package['model_high']
        self.model_low = model_package['model_low']
        self.model_type = model_package.get('model_type', 'xgboost')
        self.actual_feature_columns = model_package['feature_columns']
        
        print(f"模型已載入: {filepath}")
        return model_package

if __name__ == '__main__':
    print("Next Bar Model Trainer")
    print("請在App中使用")