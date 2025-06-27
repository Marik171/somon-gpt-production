#!/usr/bin/env python3
"""
XGBoost Model Training Pipeline for Tajikistan Real Estate Price Prediction

This script trains an XGBoost regression model using preprocessed data.
It includes hyperparameter tuning, early stopping, and comprehensive evaluation.

Author: Real Estate ML Pipeline
Date: June 18, 2025
"""

import numpy as np
import pandas as pd
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

# XGBoost and sklearn imports
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class XGBoostTrainer:
    """
    XGBoost model trainer with comprehensive evaluation and model management
    """
    
    def __init__(self, data_dir: str, model_dir: str, random_state: int = 42):
        """
        Initialize the XGBoost trainer
        
        Args:
            data_dir: Directory containing preprocessed data
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.random_state = random_state
        
        # Create model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and data containers
        self.model = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.encoding_info = None
        self.preprocessing_metadata = None
        
        logger.info(f"Initialized XGBoost Trainer")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Model directory: {self.model_dir}")
    
    def load_preprocessed_data(self) -> None:
        """
        Load preprocessed data and artifacts
        """
        logger.info("Loading preprocessed data...")
        
        # Check if all required files exist
        required_files = ['X_train.npy', 'X_valid.npy', 'y_train.npy', 'y_valid.npy']
        for file_name in required_files:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load numpy arrays
        self.X_train = np.load(self.data_dir / 'X_train.npy')
        self.X_valid = np.load(self.data_dir / 'X_valid.npy')
        self.y_train = np.load(self.data_dir / 'y_train.npy')
        self.y_valid = np.load(self.data_dir / 'y_valid.npy')
        
        # Load preprocessing artifacts
        try:
            self.encoding_info = joblib.load(self.data_dir / 'encoding_info.pkl')
            self.preprocessing_metadata = joblib.load(self.data_dir / 'preprocessing_metadata.pkl')
        except FileNotFoundError as e:
            logger.warning(f"Could not load preprocessing metadata: {e}")
            self.encoding_info = {'feature_names': [f'feature_{i}' for i in range(self.X_train.shape[1])]}
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Training set: {self.X_train.shape}")
        logger.info(f"  Validation set: {self.X_valid.shape}")
        logger.info(f"  Features: {self.X_train.shape[1]}")
        logger.info(f"  Target range: ${self.y_train.min():,.0f} - ${self.y_train.max():,.0f}")
    
    def create_xgboost_model(self, custom_params: Dict[str, Any] = None) -> xgb.XGBRegressor:
        """
        Create XGBoost model with optimized hyperparameters
        
        Args:
            custom_params: Custom hyperparameters to override defaults
            
        Returns:
            Configured XGBoost model
        """
        # Default hyperparameters optimized for real estate data
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'tree_method': 'hist',
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_jobs': -1
        }
        
        # Override with custom parameters if provided
        if custom_params:
            default_params.update(custom_params)
        
        logger.info("XGBoost Model Configuration:")
        for param, value in default_params.items():
            logger.info(f"  {param}: {value}")
        
        return xgb.XGBRegressor(**default_params)
    
    def train_model(self, early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """
        Train the XGBoost model with early stopping
        
        Args:
            early_stopping_rounds: Number of rounds for early stopping
            
        Returns:
            Training history and metrics
        """
        logger.info("Starting XGBoost model training...")
        
        # Create model
        self.model = self.create_xgboost_model()
        
        # Create DMatrix for better performance and early stopping
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_valid, label=self.y_valid)
        
        # Set up evaluation
        evallist = [(dtrain, 'train'), (dval, 'validation')]
        
        # Train model with early stopping
        start_time = datetime.now()
        
        # Use XGBoost's native training for better control
        params = self.model.get_params()
        params['eval_metric'] = 'rmse'
        
        # Convert some parameters for xgb.train
        num_boost_round = params.pop('n_estimators', 500)
        
        evals_result = {}
        
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evallist,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=50
        )
        
        # Update our model with the trained booster
        self.model._Booster = bst
        
        training_time = datetime.now() - start_time
        
        logger.info(f"Training completed in {training_time}")
        logger.info(f"Best iteration: {bst.best_iteration}")
        logger.info(f"Best score: {bst.best_score:.4f}")
        
        return {
            'training_time': training_time.total_seconds(),
            'best_iteration': bst.best_iteration,
            'best_score': bst.best_score,
            'evals_result': evals_result
        }
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_valid_pred = self.model.predict(self.X_valid)
        
        # Calculate metrics
        metrics = {}
        
        # Training metrics
        metrics['train_rmse'] = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        metrics['train_mae'] = mean_absolute_error(self.y_train, y_train_pred)
        metrics['train_r2'] = r2_score(self.y_train, y_train_pred)
        
        # Validation metrics
        metrics['valid_rmse'] = np.sqrt(mean_squared_error(self.y_valid, y_valid_pred))
        metrics['valid_mae'] = mean_absolute_error(self.y_valid, y_valid_pred)
        metrics['valid_r2'] = r2_score(self.y_valid, y_valid_pred)
        
        # Calculate percentage errors
        metrics['train_mape'] = np.mean(np.abs((self.y_train - y_train_pred) / self.y_train)) * 100
        metrics['valid_mape'] = np.mean(np.abs((self.y_valid - y_valid_pred) / self.y_valid)) * 100
        
        # Overfitting detection
        metrics['overfitting_ratio'] = metrics['valid_rmse'] / metrics['train_rmse']
        
        logger.info("Model Evaluation Results:")
        logger.info(f"  Training RMSE: ${metrics['train_rmse']:,.0f}")
        logger.info(f"  Validation RMSE: ${metrics['valid_rmse']:,.0f}")
        logger.info(f"  Training MAE: ${metrics['train_mae']:,.0f}")
        logger.info(f"  Validation MAE: ${metrics['valid_mae']:,.0f}")
        logger.info(f"  Training R²: {metrics['train_r2']:.4f}")
        logger.info(f"  Validation R²: {metrics['valid_r2']:.4f}")
        logger.info(f"  Training MAPE: {metrics['train_mape']:.2f}%")
        logger.info(f"  Validation MAPE: {metrics['valid_mape']:.2f}%")
        logger.info(f"  Overfitting Ratio: {metrics['overfitting_ratio']:.3f}")
        
        return metrics
    
    def analyze_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Analyze and log feature importance
        
        Args:
            top_n: Number of top features to display
            
        Returns:
            Feature importance dictionary
        """
        logger.info("Analyzing feature importance...")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Get feature names
        feature_names = self.encoding_info.get('feature_names', [f'feature_{i}' for i in range(len(importance))])
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importance))
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Top {top_n} Most Important Features:")
        for i, (feature, imp) in enumerate(sorted_importance[:top_n]):
            logger.info(f"  {i+1:2d}. {feature}: {imp:.4f}")
        
        return importance_dict
    
    def save_model_and_artifacts(self, metrics: Dict[str, float], 
                                training_history: Dict[str, Any],
                                feature_importance: Dict[str, float]) -> None:
        """
        Save the trained model and all associated artifacts
        
        Args:
            metrics: Evaluation metrics
            training_history: Training history
            feature_importance: Feature importance scores
        """
        logger.info("Saving model and artifacts...")
        
        # Save the XGBoost model
        model_path = self.model_dir / 'xgboost_price_model.pkl'
        joblib.dump(self.model, model_path)
        
        # Save model metadata
        model_metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'XGBoost Regressor',
            'framework_version': xgb.__version__,
            'target_variable': 'price',
            'n_features': self.X_train.shape[1],
            'n_train_samples': len(self.X_train),
            'n_valid_samples': len(self.X_valid),
            'random_state': self.random_state,
            'feature_names': self.encoding_info.get('feature_names', []),
            'preprocessing_metadata': self.preprocessing_metadata
        }
        
        # Save metrics
        metrics_path = self.model_dir / 'model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save training history
        training_history_path = self.model_dir / 'training_history.json'
        # Convert numpy types to Python types for JSON serialization
        training_history_json = {}
        for key, value in training_history.items():
            if isinstance(value, np.ndarray):
                training_history_json[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                training_history_json[key] = value.item()
            else:
                training_history_json[key] = value
        
        with open(training_history_path, 'w') as f:
            json.dump(training_history_json, f, indent=2)
        
        # Save feature importance
        importance_path = self.model_dir / 'feature_importance.json'
        # Convert numpy float32 to Python float for JSON serialization
        feature_importance_json = {k: float(v) for k, v in feature_importance.items()}
        with open(importance_path, 'w') as f:
            json.dump(feature_importance_json, f, indent=2)
        
        # Save model metadata
        metadata_path = self.model_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        logger.info(f"Model and artifacts saved to {self.model_dir}")
        
        # Log file sizes
        for file_path in self.model_dir.glob('*'):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  {file_path.name}: {size_mb:.2f} MB")
    
    def run_training_pipeline(self, early_stopping_rounds: int = 50) -> None:
        """
        Run the complete training pipeline
        
        Args:
            early_stopping_rounds: Number of rounds for early stopping
        """
        logger.info("=" * 60)
        logger.info("STARTING XGBOOST TRAINING PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load preprocessed data
            self.load_preprocessed_data()
            
            # Step 2: Train model
            training_history = self.train_model(early_stopping_rounds)
            
            # Step 3: Evaluate model
            metrics = self.evaluate_model()
            
            # Step 4: Analyze feature importance
            feature_importance = self.analyze_feature_importance()
            
            # Step 5: Save model and artifacts
            self.save_model_and_artifacts(metrics, training_history, feature_importance)
            
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            
            # Final summary
            logger.info("TRAINING SUMMARY:")
            logger.info(f"  • Model: XGBoost Regressor")
            logger.info(f"  • Training samples: {len(self.X_train):,}")
            logger.info(f"  • Validation samples: {len(self.X_valid):,}")
            logger.info(f"  • Features: {self.X_train.shape[1]}")
            logger.info(f"  • Validation RMSE: ${metrics.get('valid_rmse', 0):,.0f}")
            logger.info(f"  • Validation R²: {metrics.get('valid_r2', 0):.4f}")
            logger.info(f"  • Validation MAPE: {metrics.get('valid_mape', 0):.2f}%")
            logger.info(f"  • Model saved to: {self.model_dir / 'xgboost_price_model.pkl'}")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    """
    Main function with CLI interface
    """
    parser = argparse.ArgumentParser(
        description="XGBoost Model Training Pipeline for Real Estate Price Prediction"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/ml_model_preprocessed',
        help='Directory containing preprocessed data'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=50,
        help='Number of rounds for early stopping'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Initialize and run trainer
    trainer = XGBoostTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        random_state=args.random_state
    )
    
    trainer.run_training_pipeline(early_stopping_rounds=args.early_stopping)


if __name__ == "__main__":
    main()
