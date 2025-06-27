#!/usr/bin/env python3
"""
Robust Bargain-Finding Algorithm for Real Estate Analysis

This module provides an improved, mathematically sound approach to identifying
bargain properties based on composite scoring without relying on price predictions.

Key Improvements:
1. Fixed mathematical errors in component calculations
2. Robust outlier handling and data validation
3. Adaptive thresholds based on market distribution
4. Comprehensive scoring with proper normalization
5. Transparent and explainable scoring logic

Author: Real Estate ML Pipeline
Date: June 27, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List
from scipy import stats
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustBargainFinder:
    """
    Robust bargain-finding system with improved mathematical foundations
    and comprehensive validation.
    """
    
    def __init__(self, outlier_threshold: float = 3.0, min_sample_size: int = 10):
        """
        Initialize the bargain finder.
        
        Args:
            outlier_threshold: Z-score threshold for outlier detection
            min_sample_size: Minimum sample size for district calculations
        """
        self.outlier_threshold = outlier_threshold
        self.min_sample_size = min_sample_size
        self.district_stats = {}
        self.market_stats = {}
        
    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input data and handle missing values/outliers.
        
        Args:
            df: Input DataFrame with property data
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("🔍 Validating and cleaning input data...")
        
        df_clean = df.copy()
        
        # Required columns for bargain analysis
        required_cols = ['price', 'area_m2', 'district', 'price_per_m2']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with critical missing values
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['price', 'area_m2', 'district'])
        
        if len(df_clean) < initial_count:
            logger.warning(f"Removed {initial_count - len(df_clean)} rows with missing critical data")
        
        # Calculate price_per_m2 if missing
        if 'price_per_m2' not in df_clean.columns or df_clean['price_per_m2'].isna().any():
            df_clean['price_per_m2'] = df_clean['price'] / df_clean['area_m2']
        
        # Handle outliers using IQR method (more robust than z-score)
        for col in ['price', 'area_m2', 'price_per_m2']:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.warning(f"Found {outlier_count} outliers in {col}, capping values")
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        
        # Validate data ranges
        df_clean = df_clean[
            (df_clean['price'] > 0) & 
            (df_clean['area_m2'] > 0) & 
            (df_clean['price_per_m2'] > 0)
        ]
        
        logger.info(f"✅ Data validation complete: {len(df_clean)} valid properties")
        return df_clean
    
    def calculate_district_market_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate robust district-level market statistics.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary of district statistics
        """
        logger.info("📊 Calculating district market statistics...")
        
        district_stats = {}
        
        for district in df['district'].unique():
            district_data = df[df['district'] == district]
            
            if len(district_data) < self.min_sample_size:
                logger.warning(f"Insufficient data for {district}: {len(district_data)} properties")
                continue
            
            # Use robust statistics (median, IQR) instead of mean/std
            stats_dict = {
                'count': len(district_data),
                'price_median': district_data['price'].median(),
                'price_q25': district_data['price'].quantile(0.25),
                'price_q75': district_data['price'].quantile(0.75),
                'price_per_m2_median': district_data['price_per_m2'].median(),
                'price_per_m2_q25': district_data['price_per_m2'].quantile(0.25),
                'price_per_m2_q75': district_data['price_per_m2'].quantile(0.75),
                'area_median': district_data['area_m2'].median(),
                'area_q25': district_data['area_m2'].quantile(0.25),
                'area_q75': district_data['area_m2'].quantile(0.75),
            }
            
            # Calculate price spread (measure of market volatility)
            stats_dict['price_spread'] = (stats_dict['price_q75'] - stats_dict['price_q25']) / stats_dict['price_median']
            stats_dict['price_per_m2_spread'] = (stats_dict['price_per_m2_q75'] - stats_dict['price_per_m2_q25']) / stats_dict['price_per_m2_median']
            
            district_stats[district] = stats_dict
        
        self.district_stats = district_stats
        logger.info(f"✅ Calculated statistics for {len(district_stats)} districts")
        return district_stats
    
    def calculate_market_position_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate market position score (0-1, higher = better value).
        
        Args:
            df: DataFrame with property data
            
        Returns:
            Series with market position scores
        """
        # Use percentile rank (inverted so lower price = higher score)
        price_percentile = df['price_per_m2'].rank(pct=True)
        return 1 - price_percentile
    
    def calculate_price_advantage_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate price advantage score compared to district median.
        
        Args:
            df: DataFrame with property data
            
        Returns:
            Series with price advantage scores
        """
        def get_price_advantage(row):
            district = row['district']
            if district not in self.district_stats:
                return 0.5  # Neutral score for unknown districts
            
            district_median = self.district_stats[district]['price_per_m2_median']
            property_price = row['price_per_m2']
            
            # Calculate advantage as percentage below district median
            advantage = (district_median - property_price) / district_median
            
            # Convert to 0-1 score (capped at reasonable bounds)
            # Properties 30% below median get score of 1.0
            # Properties at median get score of 0.5
            # Properties 30% above median get score of 0.0
            score = 0.5 + (advantage / 0.6)  # 0.6 = 2 * 0.3 for symmetry
            return np.clip(score, 0, 1)
        
        return df.apply(get_price_advantage, axis=1)
    
    def calculate_quality_features_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate normalized quality features score.
        
        Args:
            df: DataFrame with property data
            
        Returns:
            Series with quality scores
        """
        quality_components = []
        
        # Renovation score (if available)
        if 'renovation_score' in df.columns:
            # Normalize to 0-1 range
            renovation_norm = df['renovation_score'] / df['renovation_score'].max() if df['renovation_score'].max() > 0 else 0
            quality_components.append(renovation_norm * 0.4)  # 40% weight
        elif 'renovation' in df.columns:
            # Create renovation score from categorical data
            renovation_map = {
                'Новый ремонт': 1.0,
                'С ремонтом': 0.7,
                'Без ремонта (коробка)': 0.3
            }
            renovation_score = df['renovation'].map(renovation_map).fillna(0.5)
            quality_components.append(renovation_score * 0.4)
        
        # Heating score
        if 'heating_score' in df.columns:
            quality_components.append(df['heating_score'] * 0.2)  # 20% weight
        elif 'heating' in df.columns:
            heating_score = (df['heating'] == 'Есть').astype(float)
            quality_components.append(heating_score * 0.2)
        
        # Bathroom score
        if 'bathroom_score' in df.columns:
            quality_components.append(df['bathroom_score'] * 0.2)  # 20% weight
        elif 'bathroom' in df.columns:
            bathroom_score = (df['bathroom'] == 'Раздельный').astype(float)
            quality_components.append(bathroom_score * 0.2)
        
        # Floor preference score
        if 'floor_preference_score' in df.columns:
            quality_components.append(df['floor_preference_score'] * 0.2)  # 20% weight
        elif 'floor' in df.columns:
            # Create floor preference: middle floors (2-7) are preferred
            floor_score = np.where(
                (df['floor'] >= 2) & (df['floor'] <= 7), 1.0,
                np.where(df['floor'] == 1, 0.6, 0.8)  # Ground floor less preferred
            )
            quality_components.append(pd.Series(floor_score, index=df.index) * 0.2)
        
        # Combine components
        if quality_components:
            total_quality = sum(quality_components)
            # Normalize to ensure 0-1 range
            return np.clip(total_quality, 0, 1)
        else:
            logger.warning("No quality features found, using neutral score")
            return pd.Series(0.5, index=df.index)
    
    def calculate_size_appropriateness_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate size appropriateness score (fixed mathematical issues).
        
        Args:
            df: DataFrame with property data
            
        Returns:
            Series with size appropriateness scores
        """
        def get_size_score(row):
            district = row['district']
            if district not in self.district_stats:
                return 0.5  # Neutral score for unknown districts
            
            district_median = self.district_stats[district]['area_median']
            property_area = row['area_m2']
            
            # Calculate relative difference from district median
            relative_diff = abs(property_area - district_median) / district_median
            
            # Score decreases as property deviates from district norm
            # Properties within 20% of median get score > 0.8
            # Properties within 50% of median get score > 0.5
            score = max(0, 1 - (relative_diff / 0.5))  # Linear decay, bottoms out at 50% difference
            return min(score, 1.0)
        
        return df.apply(get_size_score, axis=1)
    
    def calculate_documentation_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate documentation/legal score.
        
        Args:
            df: DataFrame with property data
            
        Returns:
            Series with documentation scores
        """
        if 'tech_passport_score' in df.columns:
            return np.clip(df['tech_passport_score'], 0, 1)
        elif 'tech_passport' in df.columns:
            return (df['tech_passport'] == 'Есть').astype(float)
        else:
            logger.warning("No tech passport data found, using neutral score")
            return pd.Series(0.5, index=df.index)
    
    def calculate_adaptive_thresholds(self, scores: pd.Series) -> Dict[str, float]:
        """
        Calculate adaptive thresholds based on score distribution.
        
        Args:
            scores: Series of bargain scores
            
        Returns:
            Dictionary with category thresholds
        """
        # Use percentiles to create more balanced categories
        thresholds = {
            'exceptional_opportunity': scores.quantile(0.95),  # Top 5%
            'excellent_bargain': scores.quantile(0.85),        # Top 15%
            'good_bargain': scores.quantile(0.70),             # Top 30%
            'fair_value': scores.quantile(0.50),               # Top 50%
            'market_price': scores.quantile(0.25),             # Bottom 75%
            # Below 25th percentile = overpriced
        }
        
        # Ensure minimum separation between thresholds
        min_separation = 0.05
        for i, (category, threshold) in enumerate(list(thresholds.items())[1:], 1):
            prev_threshold = list(thresholds.values())[i-1]
            if threshold > prev_threshold - min_separation:
                thresholds[category] = prev_threshold - min_separation
        
        return thresholds
    
    def calculate_robust_bargain_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate robust bargain scores with improved methodology.
        
        Args:
            df: Input DataFrame with property data
            
        Returns:
            DataFrame with bargain scores and categories
        """
        logger.info("🎯 Calculating robust bargain scores...")
        
        # Validate and clean data
        df_clean = self.validate_and_clean_data(df)
        
        # Calculate district statistics
        self.calculate_district_market_stats(df_clean)
        
        # Calculate component scores
        components = {
            'price_advantage': self.calculate_price_advantage_score(df_clean),
            'quality_features': self.calculate_quality_features_score(df_clean),
            'market_position': self.calculate_market_position_score(df_clean),
            'size_appropriateness': self.calculate_size_appropriateness_score(df_clean),
            'documentation': self.calculate_documentation_score(df_clean)
        }
        
        # Improved weights based on analysis
        weights = {
            'price_advantage': 0.35,      # 35% - Most important for bargains
            'quality_features': 0.30,     # 30% - Property condition
            'market_position': 0.20,      # 20% - Overall market positioning
            'size_appropriateness': 0.10, # 10% - Size fit for district
            'documentation': 0.05         # 5% - Legal/documentation bonus
        }
        
        # Validate components are in 0-1 range
        for name, component in components.items():
            if component.min() < 0 or component.max() > 1:
                logger.warning(f"Component {name} outside 0-1 range: {component.min():.3f} - {component.max():.3f}")
                components[name] = np.clip(component, 0, 1)
        
        # Calculate weighted bargain score
        bargain_score = sum(components[comp] * weights[comp] for comp in components.keys())
        df_clean['bargain_score'] = np.clip(bargain_score, 0, 1)
        
        # Add component scores for transparency
        for name, component in components.items():
            df_clean[f'component_{name}'] = component
        
        # Calculate adaptive thresholds
        thresholds = self.calculate_adaptive_thresholds(df_clean['bargain_score'])
        
        # Categorize bargains using adaptive thresholds
        def categorize_bargain_adaptive(score):
            if score >= thresholds['exceptional_opportunity']:
                return 'exceptional_opportunity'
            elif score >= thresholds['excellent_bargain']:
                return 'excellent_bargain'
            elif score >= thresholds['good_bargain']:
                return 'good_bargain'
            elif score >= thresholds['fair_value']:
                return 'fair_value'
            elif score >= thresholds['market_price']:
                return 'market_price'
            else:
                return 'overpriced'
        
        df_clean['bargain_category'] = df_clean['bargain_score'].apply(categorize_bargain_adaptive)
        
        # Log results
        self._log_bargain_analysis(df_clean, thresholds)
        
        # Return results aligned with original DataFrame
        result_df = df.copy()
        for col in ['bargain_score', 'bargain_category'] + [f'component_{name}' for name in components.keys()]:
            result_df[col] = df_clean[col].reindex(df.index).fillna(0.5 if 'score' in col else 'market_price')
        
        return result_df
    
    def _log_bargain_analysis(self, df: pd.DataFrame, thresholds: Dict[str, float]) -> None:
        """Log detailed bargain analysis results."""
        logger.info("📈 Bargain Analysis Results:")
        
        # Distribution
        bargain_dist = df['bargain_category'].value_counts()
        total = len(df)
        
        logger.info("Category Distribution:")
        for category, count in bargain_dist.items():
            percentage = (count / total) * 100
            logger.info(f"  {category}: {count} properties ({percentage:.1f}%)")
        
        # Thresholds
        logger.info("Adaptive Thresholds:")
        for category, threshold in thresholds.items():
            logger.info(f"  {category}: {threshold:.3f}")
        
        # Top bargains
        top_bargains = df.nlargest(5, 'bargain_score')
        logger.info("Top 5 Bargain Properties:")
        for _, row in top_bargains.iterrows():
            logger.info(f"  Score: {row['bargain_score']:.3f} | Category: {row['bargain_category']} | "
                       f"District: {row.get('district', 'N/A')} | Price: {row['price']:,.0f} TJS")
        
        # Component analysis
        logger.info("Average Component Scores:")
        for component in ['price_advantage', 'quality_features', 'market_position', 'size_appropriateness', 'documentation']:
            col_name = f'component_{component}'
            if col_name in df.columns:
                avg_score = df[col_name].mean()
                logger.info(f"  {component}: {avg_score:.3f}")

def improve_existing_bargain_scores(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Improve bargain scores for existing dataset.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (optional)
        
    Returns:
        DataFrame with improved bargain scores
    """
    logger.info(f"🚀 Improving bargain scores for {input_csv}")
    
    # Load data
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} properties")
    
    # Initialize bargain finder
    finder = RobustBargainFinder()
    
    # Calculate improved scores
    df_improved = finder.calculate_robust_bargain_score(df)
    
    # Save results
    if output_csv:
        df_improved.to_csv(output_csv, index=False)
        logger.info(f"✅ Saved improved results to {output_csv}")
    
    return df_improved

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust Bargain Finder for Real Estate")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("-o", "--output", help="Path to output CSV file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    output_path = args.output or args.input_csv.replace('.csv', '_improved_bargains.csv')
    improve_existing_bargain_scores(args.input_csv, output_path) 