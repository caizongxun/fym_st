from pipeline import TradingPipeline
from model_trainer import ModelTrainer, TrendFilterTrainer
from inference_engine import InferenceEngine
import os


def train_bounce_model():
    print("=" * 60)
    print("Step 1: Train Bounce Prediction Model (Model A)")
    print("=" * 60)
    
    pipeline = TradingPipeline(
        bb_period=20,
        atr_period=14,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        lookahead_bars=16
    )
    
    df_train, _ = pipeline.prepare_training_data(
        symbol='BTCUSDT',
        timeframe='15m',
        direction='long',
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    trainer = ModelTrainer(
        model_type='bounce',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7
    )
    
    results = trainer.train(df_train, train_ratio=0.8)
    
    os.makedirs('v2/models', exist_ok=True)
    trainer.save_model('v2/models/bb_bounce_model.pkl')
    
    return results


def train_filter_model():
    print("\n" + "=" * 60)
    print("Step 2: Train Trend Filter Model (Model B)")
    print("=" * 60)
    
    pipeline = TradingPipeline(
        bb_period=20,
        atr_period=14,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        lookahead_bars=16
    )
    
    df_train, _ = pipeline.prepare_training_data(
        symbol='BTCUSDT',
        timeframe='15m',
        direction='long',
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    trainer = TrendFilterTrainer(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7
    )
    
    results = trainer.train(df_train, train_ratio=0.8)
    
    trainer.save_model('v2/models/trend_filter_model.pkl')
    
    return results


def test_inference_engine():
    print("\n" + "=" * 60)
    print("Step 3: Test Inference Engine with Confluence-Veto Logic")
    print("=" * 60)
    
    engine = InferenceEngine(
        bounce_model_path='v2/models/bb_bounce_model.pkl',
        filter_model_path='v2/models/trend_filter_model.pkl',
        bounce_threshold=0.65,
        filter_threshold=0.40
    )
    
    pipeline = TradingPipeline()
    df_test, _ = pipeline.prepare_training_data(
        symbol='ETHUSDT',
        timeframe='15m',
        direction='long',
        start_date='2024-01-01',
        end_date='2024-03-31'
    )
    
    df_predictions = engine.predict_batch(df_test)
    
    stats = engine.get_statistics(df_predictions)
    
    print("\nInference Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nSample Predictions:")
    sample_cols = ['timestamp', 'close', 'p_bounce', 'p_filter', 'signal', 'reason', 'target']
    print(df_predictions[sample_cols].head(20))
    
    return df_predictions, stats


if __name__ == "__main__":
    bounce_results = train_bounce_model()
    
    filter_results = train_filter_model()
    
    df_predictions, stats = test_inference_engine()
    
    print("\n" + "=" * 60)
    print("Training and Inference Complete")
    print("=" * 60)
    print(f"Bounce Model Test AUC: {bounce_results['test_auc']:.4f}")
    print(f"Filter Model Test AUC: {filter_results['test_auc']:.4f}")
    print(f"Entry Approval Rate: {stats['entry_rate']:.2f}%")
    if 'approved_success_rate' in stats:
        print(f"Approved Entry Success Rate: {stats['approved_success_rate']:.2f}%")
