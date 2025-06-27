#!/usr/bin/env python3
"""
ML Data Preprocessing Pipeline for Tajikistan Real Estate Price Prediction

This script preprocesses the cleaned real estate data for machine learning modeling.
It handles feature engineering, encoding, scaling, and data splitting for XGBoost training.

Author: Real Estate ML Pipeline
Date: June 18, 2025
"""

import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealEstateMLPreprocessor:
    """
    Comprehensive ML preprocessing pipeline for real estate data
    """
    
    def __init__(self, data_path: str, output_dir: str, random_state: int = 42):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the cleaned CSV file
            output_dir: Directory to save preprocessed data and artifacts
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize transformers
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = None
        self.column_transformer = None
        
        # Feature definitions
        self.numerical_features = ['area_m2', 'floor', 'photo_count']
        self.categorical_features = ['district', 'build_type', 'renovation', 'bathroom', 'heating', 'tech_passport']
        self.temporal_features = ['year', 'month', 'day_of_week']
        
        logger.info(f"Initialized ML Preprocessor with output dir: {self.output_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the cleaned dataset
        
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Log basic statistics
        logger.info(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer features for ML modeling
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        df_ml = df.copy()
        
        # Extract temporal features from publication_date
        df_ml['publication_date'] = pd.to_datetime(df_ml['publication_date'])
        df_ml['year'] = df_ml['publication_date'].dt.year
        df_ml['month'] = df_ml['publication_date'].dt.month
        df_ml['day_of_week'] = df_ml['publication_date'].dt.dayofweek
        
        # Create price per square meter feature (useful for model understanding)
        df_ml['price_per_m2'] = df_ml['price'] / df_ml['area_m2']
        
        # Floor category (ground, low, middle, high)
        df_ml['floor_category'] = pd.cut(
            df_ml['floor'], 
            bins=[0, 1, 4, 7, float('inf')], 
            labels=['ground', 'low', 'middle', 'high']
        )
        
        # Area category (compact, standard, spacious, premium)
        df_ml['area_category'] = pd.cut(
            df_ml['area_m2'],
            bins=[0, 60, 80, 120, float('inf')],
            labels=['compact', 'standard', 'spacious', 'premium']
        )
        
        # Photo count category (low, medium, high)
        df_ml['photo_category'] = pd.cut(
            df_ml['photo_count'],
            bins=[0, 5, 10, float('inf')],
            labels=['low', 'medium', 'high']
        )
        
        # Add these new categorical features to our list
        self.categorical_features.extend(['floor_category', 'area_category', 'photo_category'])
        
        logger.info(f"Feature engineering complete. New features: {len(df_ml.columns) - len(df.columns)}")
        
        return df_ml
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values appropriately for each feature type
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        
        df_clean = df.copy()
        
        # Handle numerical features - fill with median
        for col in self.numerical_features:
            if col in df_clean.columns and df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_val}")
        
        # Handle categorical features - fill with mode or 'Unknown'
        for col in self.categorical_features:
            if col in df_clean.columns and df_clean[col].isnull().any():
                mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled {col} missing values with mode: {mode_val}")
        
        # Handle temporal features - use current values as fallback
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        if 'year' in df_clean.columns:
            df_clean['year'].fillna(current_year, inplace=True)
        if 'month' in df_clean.columns:
            df_clean['month'].fillna(current_month, inplace=True)
        if 'day_of_week' in df_clean.columns:
            df_clean['day_of_week'].fillna(1, inplace=True)  # Monday as default
        
        logger.info(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
        
        return df_clean
    
    def encode_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Encode categorical features using appropriate encoding strategies
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (encoded_features_array, encoding_info)
        """
        logger.info("Encoding categorical features...")
        
        # Separate features for different encoding strategies
        low_cardinality_cats = []
        high_cardinality_cats = []
        
        for col in self.categorical_features:
            if col in df.columns:
                unique_count = df[col].nunique()
                if unique_count <= 10:  # Use one-hot encoding for low cardinality
                    low_cardinality_cats.append(col)
                else:  # Use label encoding for high cardinality
                    high_cardinality_cats.append(col)
                
                logger.info(f"{col}: {unique_count} unique values ({'one-hot' if unique_count <= 10 else 'label'} encoding)")
        
        # Prepare feature columns
        feature_columns = []
        all_features = self.numerical_features + self.temporal_features
        
        # Add numerical and temporal features (already numeric)
        for col in all_features:
            if col in df.columns:
                feature_columns.append(df[col].values.reshape(-1, 1))
        
        # Label encode high cardinality features
        for col in high_cardinality_cats:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                encoded_values = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                encoded_values = self.label_encoders[col].transform(df[col].astype(str))
            
            feature_columns.append(encoded_values.reshape(-1, 1))
        
        # One-hot encode low cardinality features
        if low_cardinality_cats:
            onehot_data = df[low_cardinality_cats].astype(str)
            
            if self.onehot_encoder is None:
                self.onehot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                onehot_encoded = self.onehot_encoder.fit_transform(onehot_data)
            else:
                onehot_encoded = self.onehot_encoder.transform(onehot_data)
            
            feature_columns.append(onehot_encoded)
        
        # Combine all features
        if feature_columns:
            X = np.hstack(feature_columns)
        else:
            raise ValueError("No valid features found for encoding")
        
        # Create feature names for reference
        feature_names = []
        
        # Add numerical and temporal feature names
        for col in all_features:
            if col in df.columns:
                feature_names.append(col)
        
        # Add label encoded feature names
        for col in high_cardinality_cats:
            feature_names.append(f"{col}_encoded")
        
        # Add one-hot encoded feature names
        if low_cardinality_cats and self.onehot_encoder is not None:
            onehot_names = self.onehot_encoder.get_feature_names_out(low_cardinality_cats)
            feature_names.extend(onehot_names)
        
        encoding_info = {
            'feature_names': feature_names,
            'low_cardinality_cats': low_cardinality_cats,
            'high_cardinality_cats': high_cardinality_cats,
            'n_features': X.shape[1]
        }
        
        logger.info(f"Encoding complete. Final feature count: {X.shape[1]}")
        
        return X, encoding_info
    
    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """
        Scale numerical features using StandardScaler
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        logger.info("Scaling features...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Features scaled. Shape: {X_scaled.shape}")
        logger.info(f"Feature means: min={X_scaled.mean(axis=0).min():.4f}, max={X_scaled.mean(axis=0).max():.4f}")
        
        return X_scaled
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training and validation sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for validation
            
        Returns:
            Tuple of (X_train, X_valid, y_train, y_valid)
        """
        logger.info(f"Splitting data into train/validation sets (test_size={test_size})")
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=None
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_valid.shape[0]} samples")
        logger.info(f"Training target range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
        logger.info(f"Validation target range: ${y_valid.min():,.0f} - ${y_valid.max():,.0f}")
        
        return X_train, X_valid, y_train, y_valid
    
    def save_artifacts(self, X_train: np.ndarray, X_valid: np.ndarray, 
                      y_train: np.ndarray, y_valid: np.ndarray, 
                      encoding_info: Dict[str, Any]) -> None:
        """
        Save all preprocessing artifacts for later use
        
        Args:
            X_train: Training features
            X_valid: Validation features
            y_train: Training targets
            y_valid: Validation targets
            encoding_info: Information about encoding process
        """
        logger.info("Saving preprocessing artifacts...")
        
        # Save numpy arrays
        np.save(self.output_dir / 'X_train.npy', X_train)
        np.save(self.output_dir / 'X_valid.npy', X_valid)
        np.save(self.output_dir / 'y_train.npy', y_train)
        np.save(self.output_dir / 'y_valid.npy', y_valid)
        
        # Save transformers
        joblib.dump(self.scaler, self.output_dir / 'scaler.pkl')
        joblib.dump(self.label_encoders, self.output_dir / 'label_encoders.pkl')
        
        if self.onehot_encoder is not None:
            joblib.dump(self.onehot_encoder, self.output_dir / 'onehot_encoder.pkl')
        
        # Save encoding information
        joblib.dump(encoding_info, self.output_dir / 'encoding_info.pkl')
        
        # Save preprocessing metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'n_samples_train': len(X_train),
            'n_samples_valid': len(X_valid),
            'n_features': X_train.shape[1],
            'feature_names': encoding_info['feature_names'],
            'random_state': self.random_state
        }
        
        joblib.dump(metadata, self.output_dir / 'preprocessing_metadata.pkl')
        
        logger.info(f"All artifacts saved to {self.output_dir}")
        
        # Log file sizes
        for file_path in self.output_dir.glob('*'):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  {file_path.name}: {size_mb:.2f} MB")
    
    def run_preprocessing_pipeline(self) -> None:
        """
        Run the complete preprocessing pipeline
        """
        logger.info("=" * 60)
        logger.info("STARTING REAL ESTATE ML PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load data
            df = self.load_data()
            
            # Step 2: Feature engineering
            df_engineered = self.feature_engineering(df)
            
            # Step 3: Handle missing values
            df_clean = self.handle_missing_values(df_engineered)
            
            # Step 4: Prepare target variable
            y = df_clean['price'].values
            logger.info(f"Target variable (price) range: ${y.min():,.0f} - ${y.max():,.0f}")
            
            # Step 5: Encode features
            X, encoding_info = self.encode_features(df_clean)
            
            # Step 6: Scale features
            X_scaled = self.scale_features(X)
            
            # Step 7: Split data
            X_train, X_valid, y_train, y_valid = self.split_data(X_scaled, y)
            
            # Step 8: Save artifacts
            self.save_artifacts(X_train, X_valid, y_train, y_valid, encoding_info)
            
            logger.info("=" * 60)
            logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            
            # Summary statistics
            logger.info("FINAL SUMMARY:")
            logger.info(f"  • Total samples: {len(df_clean)}")
            logger.info(f"  • Training samples: {len(X_train)}")
            logger.info(f"  • Validation samples: {len(X_valid)}")
            logger.info(f"  • Features: {X_train.shape[1]}")
            logger.info(f"  • Target range: ${y.min():,.0f} - ${y.max():,.0f}")
            logger.info(f"  • Artifacts saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise


def main():
    """
    Main function with CLI interface
    """
    parser = argparse.ArgumentParser(
        description="ML Preprocessing Pipeline for Real Estate Price Prediction"
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/preprocessed/cleaned_listings_v2.csv',
        help='Path to the cleaned CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/ml_model_preprocessed',
        help='Directory to save preprocessed data and artifacts'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Initialize and run preprocessor
    preprocessor = RealEstateMLPreprocessor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        random_state=args.random_state
    )
    
    preprocessor.run_preprocessing_pipeline()


if __name__ == "__main__":
    main()
