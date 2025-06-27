#!/usr/bin/env python3
"""
Real Estate Price Prediction Inference Script

This script loads the trained XGBoost model and provides price predictions
for new property listings with confidence intervals and explanations.

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
from typing import Dict, Any, List, Tuple

import joblib
import xgboost as xgb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealEstatePricePredictor:
    """
    Real estate price prediction system using trained XGBoost model
    """
    
    def __init__(self, model_dir: str = "models", preprocessing_dir: str = "data/ml_model_preprocessed"):
        """
        Initialize the price predictor
        
        Args:
            model_dir: Directory containing trained model
            preprocessing_dir: Directory containing preprocessing artifacts
        """
        self.model_dir = Path(model_dir)
        self.preprocessing_dir = Path(preprocessing_dir)
        
        # Initialize containers
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.onehot_encoder = None
        self.encoding_info = None
        self.model_metadata = None
        
        # Load all artifacts
        self.load_artifacts()
        
        logger.info("Real Estate Price Predictor initialized successfully")
    
    def load_artifacts(self) -> None:
        """
        Load trained model and preprocessing artifacts
        """
        logger.info("Loading model and preprocessing artifacts...")
        
        # Load trained model
        model_path = self.model_dir / 'xgboost_price_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Load preprocessing artifacts
        self.scaler = joblib.load(self.preprocessing_dir / 'scaler.pkl')
        self.label_encoders = joblib.load(self.preprocessing_dir / 'label_encoders.pkl')
        self.onehot_encoder = joblib.load(self.preprocessing_dir / 'onehot_encoder.pkl')
        self.encoding_info = joblib.load(self.preprocessing_dir / 'encoding_info.pkl')
        
        # Load model metadata
        try:
            with open(self.model_dir / 'model_metadata.json', 'r') as f:
                self.model_metadata = json.load(f)
        except FileNotFoundError:
            logger.warning("Model metadata not found")
            self.model_metadata = {}
        
        logger.info(f"Model loaded: {self.model_metadata.get('model_type', 'XGBoost')}")
        logger.info(f"Features: {self.encoding_info.get('n_features', 'Unknown')}")
    
    def preprocess_input(self, property_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess input property data for prediction
        
        Args:
            property_data: Dictionary containing property features
            
        Returns:
            Preprocessed feature array
        """
        # Create DataFrame from input
        df = pd.DataFrame([property_data])
        
        # Feature engineering (same as training)
        df['publication_date'] = pd.to_datetime(df.get('publication_date', datetime.now()))
        df['year'] = df['publication_date'].dt.year
        df['month'] = df['publication_date'].dt.month
        df['day_of_week'] = df['publication_date'].dt.dayofweek
        
        # Create price per m2 (placeholder for consistency)
        df['price_per_m2'] = 0  # Will be ignored in prediction
        
        # Create categorical features
        df['floor_category'] = pd.cut(
            df['floor'], 
            bins=[0, 1, 4, 7, float('inf')], 
            labels=['ground', 'low', 'middle', 'high']
        )
        
        df['area_category'] = pd.cut(
            df['area_m2'],
            bins=[0, 60, 80, 120, float('inf')],
            labels=['compact', 'standard', 'spacious', 'premium']
        )
        
        df['photo_category'] = pd.cut(
            df['photo_count'],
            bins=[0, 5, 10, float('inf')],
            labels=['low', 'medium', 'high']
        )
        
        # Handle missing values
        numerical_features = ['area_m2', 'floor', 'photo_count']
        for col in numerical_features:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Encode features following the same process as training
        feature_columns = []
        
        # Add numerical and temporal features
        temporal_features = ['year', 'month', 'day_of_week']
        for col in numerical_features + temporal_features:
            if col in df.columns:
                feature_columns.append(df[col].values.reshape(-1, 1))
        
        # Label encode high cardinality features
        high_cardinality_cats = self.encoding_info.get('high_cardinality_cats', [])
        for col in high_cardinality_cats:
            if col in df.columns:
                # Handle unknown categories
                known_categories = self.label_encoders[col].classes_
                df[col] = df[col].astype(str)
                
                # Replace unknown categories with the most frequent one
                unknown_mask = ~df[col].isin(known_categories)
                if unknown_mask.any():
                    most_frequent = known_categories[0]  # Use first category as fallback
                    df.loc[unknown_mask, col] = most_frequent
                    logger.warning(f"Unknown category in {col}, using fallback: {most_frequent}")
                
                encoded_values = self.label_encoders[col].transform(df[col])
                feature_columns.append(encoded_values.reshape(-1, 1))
        
        # One-hot encode low cardinality features
        low_cardinality_cats = self.encoding_info.get('low_cardinality_cats', [])
        if low_cardinality_cats:
            onehot_data = df[low_cardinality_cats].astype(str)
            onehot_encoded = self.onehot_encoder.transform(onehot_data)
            feature_columns.append(onehot_encoded)
        
        # Combine all features
        if feature_columns:
            X = np.hstack(feature_columns)
        else:
            raise ValueError("No valid features found for encoding")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict_price(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict price for a single property
        
        Args:
            property_data: Dictionary containing property features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess input
            X_processed = self.preprocess_input(property_data)
            
            # Make prediction
            predicted_price = self.model.predict(X_processed)[0]
            
            # Calculate confidence interval (approximate)
            # Using training RMSE as uncertainty estimate
            training_rmse = 91647  # From training results
            confidence_interval = {
                'lower': max(0, predicted_price - 1.96 * training_rmse),
                'upper': predicted_price + 1.96 * training_rmse
            }
            
            # Get feature importance for this prediction
            feature_importance = self.get_prediction_explanation(X_processed)
            
            result = {
                'predicted_price': float(predicted_price),
                'confidence_interval': confidence_interval,
                'feature_importance': feature_importance,
                'input_features': property_data,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_prediction_explanation(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get feature importance for prediction explanation
        
        Args:
            X: Preprocessed feature array
            
        Returns:
            Dictionary of feature importance scores
        """
        # Get global feature importance
        feature_importance = self.model.feature_importances_
        feature_names = self.encoding_info.get('feature_names', [f'feature_{i}' for i in range(len(feature_importance))])
        
        # Create importance dictionary
        importance_dict = {}
        for i, name in enumerate(feature_names):
            if i < len(feature_importance):
                importance_dict[name] = float(feature_importance[i])
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def predict_batch(self, properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict prices for multiple properties
        
        Args:
            properties: List of property dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, property_data in enumerate(properties):
            try:
                result = self.predict_price(property_data)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict for property {i}: {str(e)}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'input_features': property_data
                })
        
        return results
    
    def validate_input(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean input property data
        
        Args:
            property_data: Raw property data
            
        Returns:
            Validated and cleaned property data
        """
        cleaned_data = property_data.copy()
        
        # Required fields with defaults
        defaults = {
            'area_m2': 75.0,
            'floor': 3.0,
            'photo_count': 5,
            'district': '18 мкр',
            'build_type': 'Новостройка',
            'renovation': 'Без ремонта (коробка)',
            'bathroom': 'Раздельный',
            'heating': 'Нет',
            'tech_passport': 'Неизвестно',
            'publication_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Apply defaults for missing fields
        for field, default_value in defaults.items():
            if field not in cleaned_data or cleaned_data[field] is None:
                cleaned_data[field] = default_value
                logger.warning(f"Missing field '{field}', using default: {default_value}")
        
        # Validate numeric fields
        numeric_fields = ['area_m2', 'floor', 'photo_count']
        for field in numeric_fields:
            try:
                cleaned_data[field] = float(cleaned_data[field])
                if cleaned_data[field] <= 0:
                    cleaned_data[field] = defaults[field]
                    logger.warning(f"Invalid {field} value, using default: {defaults[field]}")
            except (ValueError, TypeError):
                cleaned_data[field] = defaults[field]
                logger.warning(f"Invalid {field} format, using default: {defaults[field]}")
        
        return cleaned_data

def main():
    """
    Main function with CLI interface for price prediction
    """
    parser = argparse.ArgumentParser(
        description="Real Estate Price Prediction CLI"
    )
    
    parser.add_argument(
        '--area',
        type=float,
        default=75.0,
        help='Property area in square meters'
    )
    
    parser.add_argument(
        '--floor',
        type=int,
        default=3,
        help='Floor number'
    )
    
    parser.add_argument(
        '--district',
        type=str,
        default='18 мкр',
        help='District name'
    )
    
    parser.add_argument(
        '--build-type',
        type=str,
        default='Новостройка',
        choices=['Новостройка', 'Вторичный рынок'],
        help='Building type'
    )
    
    parser.add_argument(
        '--renovation',
        type=str,
        default='Без ремонта (коробка)',
        choices=['Без ремонта (коробка)', 'С ремонтом', 'Новый ремонт'],
        help='Renovation status'
    )
    
    parser.add_argument(
        '--bathroom',
        type=str,
        default='Раздельный',
        choices=['Раздельный', 'Совмещенный', 'Неизвестно'],
        help='Bathroom type'
    )
    
    parser.add_argument(
        '--heating',
        type=str,
        default='Нет',
        choices=['Есть', 'Нет', 'Неизвестно'],
        help='Heating availability'
    )
    
    parser.add_argument(
        '--tech-passport',
        type=str,
        default='Неизвестно',
        choices=['Есть', 'Нет', 'Неизвестно'],
        help='Technical passport availability'
    )
    
    parser.add_argument(
        '--photo-count',
        type=int,
        default=5,
        help='Number of photos'
    )
    
    args = parser.parse_args()
    
    # Create property data from arguments
    property_data = {
        'area_m2': args.area,
        'floor': args.floor,
        'district': args.district,
        'build_type': args.build_type,
        'renovation': args.renovation,
        'bathroom': args.bathroom,
        'heating': args.heating,
        'tech_passport': args.tech_passport,
        'photo_count': args.photo_count,
        'publication_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    # Initialize predictor
    try:
        predictor = RealEstatePricePredictor()
        
        # Make prediction
        result = predictor.predict_price(property_data)
        
        # Display results
        print("=" * 60)
        print("🏠 REAL ESTATE PRICE PREDICTION")
        print("=" * 60)
        
        print(f"📍 Property Details:")
        print(f"   • Area: {property_data['area_m2']} m²")
        print(f"   • Floor: {property_data['floor']}")
        print(f"   • District: {property_data['district']}")
        print(f"   • Build Type: {property_data['build_type']}")
        print(f"   • Renovation: {property_data['renovation']}")
        print(f"   • Tech Passport: {property_data['tech_passport']}")
        
        print(f"\n💰 Price Prediction:")
        print(f"   • Predicted Price: ${result['predicted_price']:,.0f}")
        print(f"   • Confidence Interval: ${result['confidence_interval']['lower']:,.0f} - ${result['confidence_interval']['upper']:,.0f}")
        print(f"   • Price per m²: ${result['predicted_price'] / property_data['area_m2']:,.0f}")
        
        print(f"\n🔍 Top Contributing Factors:")
        top_features = list(result['feature_importance'].items())[:5]
        for i, (feature, importance) in enumerate(top_features):
            print(f"   {i+1}. {feature}: {importance:.3f}")
        
        print(f"\n⏰ Prediction made at: {result['prediction_timestamp']}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
