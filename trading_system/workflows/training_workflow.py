"""Training workflow orchestration"""
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from core import (
    CryptoDataLoader, FeatureEngineer, 
    TripleBarrierLabeling, ModelTrainer
)
from core.event_filter import BBNW_BounceFilter
from config import (
    FeatureConfig, FilterConfig, LabelConfig, 
    ModelConfig, DataConfig
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration"""
    symbol: str
    use_2024_only: bool = True
    nw_h: float = FeatureConfig.NW_H
    nw_mult: float = FeatureConfig.NW_MULT
    use_bb_trigger: bool = True
    use_nw_trigger: bool = True
    tp_multiplier: float = LabelConfig.DEFAULT_TP_MULTIPLIER
    sl_multiplier: float = LabelConfig.DEFAULT_SL_MULTIPLIER
    max_hold_bars: int = LabelConfig.DEFAULT_MAX_HOLD_BARS
    model_type: str = ModelConfig.DEFAULT_MODEL
    cv_folds: int = ModelConfig.CV_FOLDS
    early_stopping_rounds: int = ModelConfig.EARLY_STOPPING_ROUNDS

@dataclass
class TrainingResult:
    """Training result"""
    success: bool
    message: str
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    feature_importance: Optional[pd.DataFrame] = None
    df_filtered: Optional[pd.DataFrame] = None
    df_labeled: Optional[pd.DataFrame] = None

class TrainingWorkflow:
    """Orchestrates the complete training workflow"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.loader = CryptoDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.bounce_filter = None
        self.labeler = None
        self.trainer = ModelTrainer()
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Step 1: Load data"""
        logger.info(f"Loading data for {self.config.symbol}...")
        
        df_15m = self.loader.load_klines(self.config.symbol, '15m')
        df_1h = self.loader.load_klines(self.config.symbol, '1h')
        
        if self.config.use_2024_only:
            df_15m = df_15m[df_15m['open_time'] >= DataConfig.OOS_START_DATE].copy()
            df_1h = df_1h[df_1h['open_time'] >= DataConfig.OOS_START_DATE].copy()
        
        logger.info(f"15m data: {len(df_15m)} rows ({df_15m['open_time'].min()} ~ {df_15m['open_time'].max()})")
        logger.info(f"1h data: {len(df_1h)} rows ({df_1h['open_time'].min()} ~ {df_1h['open_time'].max()})")
        
        return df_15m, df_1h
    
    def build_features(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Build features"""
        logger.info("Building features...")
        
        # 15m features
        df_15m_features = self.feature_engineer.build_features(
            df_15m,
            include_microstructure=True,
            include_nw_envelope=True,
            include_adx=True,
            include_bounce_features=False
        )
        logger.info(f"15m features complete: {df_15m_features.shape}")
        
        # 1h features
        df_1h_features = self.feature_engineer.build_features(
            df_1h,
            include_microstructure=True,
            include_nw_envelope=True,
            include_adx=True,
            include_bounce_features=False
        )
        logger.info(f"1h features complete: {df_1h_features.shape}")
        
        # MTF merge
        df_mtf = self.feature_engineer.merge_and_build_mtf_features(
            df_15m_features, df_1h_features
        )
        logger.info(f"MTF merge complete: {df_mtf.shape}")
        
        # Add bounce confluence features
        df_mtf = self.feature_engineer.add_bounce_confluence_features(df_mtf)
        logger.info(f"Bounce features added: {df_mtf.shape}")
        
        return df_mtf
    
    def filter_events(self, df_mtf: pd.DataFrame) -> pd.DataFrame:
        """Step 3: Filter events"""
        logger.info("Filtering events with BB/NW bounce filter...")
        
        self.bounce_filter = BBNW_BounceFilter(
            use_bb=self.config.use_bb_trigger,
            use_nw=self.config.use_nw_trigger,
            min_pierce_pct=FilterConfig.MIN_PIERCE_PCT,
            require_volume_surge=FilterConfig.REQUIRE_VOLUME_SURGE
        )
        
        df_filtered = self.bounce_filter.filter_events(df_mtf)
        
        logger.info(f"Filtered: {len(df_filtered)}/{len(df_mtf)} events ({100*len(df_filtered)/len(df_mtf) if len(df_mtf) > 0 else 0:.1f}%)")
        
        return df_filtered
    
    def create_labels(self, df_filtered: pd.DataFrame) -> pd.DataFrame:
        """Step 4: Create labels"""
        logger.info("Creating Triple Barrier labels...")
        
        self.labeler = TripleBarrierLabeling(
            tp_multiplier=self.config.tp_multiplier,
            sl_multiplier=self.config.sl_multiplier,
            max_hold_bars=self.config.max_hold_bars
        )
        
        df_labeled = self.labeler.create_labels(df_filtered)
        
        positive_ratio = df_labeled['label'].mean()
        logger.info(f"Labeled: {len(df_labeled)} samples, positive ratio: {positive_ratio:.2%}")
        
        return df_labeled
    
    def train_model(self, df_labeled: pd.DataFrame) -> Dict[str, Any]:
        """Step 5: Train model"""
        logger.info("Training model...")
        
        metrics = self.trainer.train(
            df_labeled,
            model_type=self.config.model_type,
            cv_folds=self.config.cv_folds,
            early_stopping_rounds=self.config.early_stopping_rounds
        )
        
        logger.info(f"Training complete. CV AUC: {metrics.get('cv_auc_mean', 0):.3f}")
        
        return metrics
    
    def save_model(self) -> str:
        """Step 6: Save model"""
        model_filename = (
            f"{self.config.symbol}_15m_"
            f"BB{int(self.config.use_bb_trigger)}_"
            f"NW{int(self.config.use_nw_trigger)}_"
            f"TP{self.config.tp_multiplier:.1f}_"
            f"SL{self.config.sl_multiplier:.1f}.pkl"
        )
        
        model_path = f"models/{model_filename}"
        self.trainer.save_model(model_path)
        
        logger.info(f"Model saved: {model_path}")
        
        return model_path
    
    def run(self) -> TrainingResult:
        """Run complete training workflow"""
        try:
            # Step 1: Load data
            df_15m, df_1h = self.load_data()
            
            # Step 2: Build features
            df_mtf = self.build_features(df_15m, df_1h)
            
            if len(df_mtf) == 0:
                return TrainingResult(
                    success=False,
                    message="MTF merge resulted in empty DataFrame"
                )
            
            # Step 3: Filter events
            df_filtered = self.filter_events(df_mtf)
            
            if len(df_filtered) == 0:
                return TrainingResult(
                    success=False,
                    message="Event filter resulted in no samples"
                )
            
            # Step 4: Create labels
            df_labeled = self.create_labels(df_filtered)
            
            if len(df_labeled) == 0:
                return TrainingResult(
                    success=False,
                    message="Labeling resulted in no samples"
                )
            
            # Step 5: Train model
            metrics = self.train_model(df_labeled)
            
            # Step 6: Save model
            model_path = self.save_model()
            
            # Get feature importance
            feature_importance = None
            if hasattr(self.trainer, 'feature_importance_'):
                feature_importance = pd.DataFrame({
                    'feature': self.trainer.feature_names,
                    'importance': self.trainer.feature_importance_
                }).sort_values('importance', ascending=False)
            
            return TrainingResult(
                success=True,
                message="Training completed successfully",
                model_path=model_path,
                metrics=metrics,
                feature_importance=feature_importance,
                df_filtered=df_filtered,
                df_labeled=df_labeled
            )
        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return TrainingResult(
                success=False,
                message=f"Training failed: {str(e)}"
            )