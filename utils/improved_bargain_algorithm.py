#!/usr/bin/env python3
"""
Improved Bargain-Finding Algorithm for Real Estate Analysis

This module addresses the critical issues in the current bargain scoring system:
1. Fixed mathematical errors in component calculations
2. Better threshold calibration for realistic distributions
3. Robust handling of districts with insufficient data
4. Improved component normalization and scoring logic

Author: Real Estate ML Pipeline
Date: June 27, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple

# Set up logging
logger = logging.getLogger(__name__)

def standardize_renovation_category(build_state):
    """
    Standardize renovation categories from somon.tj data.
    
    Args:
        build_state: The build_state field value
        
    Returns:
        Standardized category string
    """
    if pd.isna(build_state):
        return 'unknown'
    
    value = str(build_state).strip()
    
    # Handle empty strings
    if not value:
        return 'unknown'
    
    if value == "Без ремонта (коробка)":
        return 'shell'
    elif value == "С ремонтом":
        return 'standard_renovation'  
    elif value == "Новый ремонт":
        return 'new_renovation'
    else:
        return 'other'

def calculate_category_aware_thresholds(df: pd.DataFrame, score_column: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate bargain thresholds within each renovation category.
    
    Args:
        df: DataFrame with bargain scores and renovation categories
        score_column: Name of the score column to use
        
    Returns:
        Dictionary of thresholds for each renovation category
    """
    logger.info("🏗️ Calculating category-specific bargain thresholds...")
    
    renovation_categories = df['renovation_category'].unique()
    category_thresholds = {}
    min_sample_size = 5
    
    for category in renovation_categories:
        category_data = df[df['renovation_category'] == category]
        
        if len(category_data) < min_sample_size:
            logger.warning(f"Insufficient data for {category}: {len(category_data)} properties")
            # Use global thresholds as fallback
            scores = df[score_column]
            category_thresholds[category] = {
                'exceptional_opportunity': scores.quantile(0.90),
                'excellent_bargain': scores.quantile(0.75),
                'good_bargain': scores.quantile(0.50),
                'fair_value': scores.quantile(0.25)
            }
            continue
        
        # Calculate category-specific thresholds
        scores = category_data[score_column]
        thresholds = {
            'exceptional_opportunity': scores.quantile(0.90),  # Top 10% within category
            'excellent_bargain': scores.quantile(0.75),        # Top 25% within category  
            'good_bargain': scores.quantile(0.50),             # Top 50% within category
            'fair_value': scores.quantile(0.25),               # Top 75% within category
        }
        
        # Ensure minimum separation
        min_separation = 0.03
        for i, (cat_name, threshold) in enumerate(list(thresholds.items())[1:], 1):
            prev_threshold = list(thresholds.values())[i-1]
            if threshold > prev_threshold - min_separation:
                thresholds[cat_name] = prev_threshold - min_separation
        
        category_thresholds[category] = thresholds
        
        logger.info(f"✅ {category}: {len(category_data)} properties, "
                   f"excellent threshold: {thresholds['excellent_bargain']:.3f}")
    
    return category_thresholds

def calculate_improved_bargain_score(df: pd.DataFrame, use_category_aware: bool = True) -> pd.DataFrame:
    """
    Calculate improved bargain scores with optional category-aware classification.
    
    Args:
        df: DataFrame with property data including existing features
        use_category_aware: Whether to use category-specific thresholds
        
    Returns:
        DataFrame with improved bargain scores and categories
    """
    logger.info("🎯 Calculating improved bargain scores...")
    if use_category_aware:
        logger.info("🏗️ Using category-aware bargain classification")
    
    df_result = df.copy()
    
    # Add renovation categories
    renovation_col = None
    possible_cols = ['build_state', 'renovation', 'condition', 'build_condition']
    
    for col in possible_cols:
        if col in df_result.columns:
            renovation_col = col
            break
    
    if renovation_col:
        df_result['renovation_category'] = df_result[renovation_col].apply(standardize_renovation_category)
        logger.info(f"📊 Renovation categories found: {df_result['renovation_category'].value_counts().to_dict()}")
    else:
        logger.warning("No renovation category column found, using 'unknown'")
        df_result['renovation_category'] = 'unknown'
    
    # 1. IMPROVED PRICE ADVANTAGE CALCULATION
    # Fix the mathematical issues in price advantage scoring
    def calculate_robust_price_advantage(df):
        """Calculate price advantage with proper bounds and outlier handling."""
        price_ratio = df['price_per_m2_vs_district_avg'].copy()
        
        # Handle extreme values (cap at reasonable bounds)
        price_ratio = np.clip(price_ratio, 0.3, 2.5)  # Properties 70% below to 150% above district avg
        
        # Convert to advantage score: lower ratio = higher advantage
        # Ratio of 0.7 (30% below district) = score of 1.0
        # Ratio of 1.0 (at district level) = score of 0.5  
        # Ratio of 1.3 (30% above district) = score of 0.0
        advantage_score = np.clip(1.5 - price_ratio, 0, 1)
        
        return advantage_score
    
    df_result['improved_price_advantage'] = calculate_robust_price_advantage(df_result)
    
    # 2. IMPROVED QUALITY FEATURES CALCULATION
    # Fix normalization issues and provide fallbacks
    def calculate_robust_quality_score(df):
        """Calculate quality score with proper normalization."""
        quality_components = []
        
        # Renovation score (normalize to 0-1)
        if 'renovation_score' in df.columns:
            renovation_norm = (df['renovation_score'] - 1) / 2  # Convert 1-3 scale to 0-1
            quality_components.append(renovation_norm * 0.4)
        
        # Heating score (already 0-1)
        if 'heating_score' in df.columns:
            quality_components.append(df['heating_score'] * 0.25)
        
        # Bathroom score (normalize to 0-1)
        if 'bathroom_score' in df.columns:
            bathroom_norm = df['bathroom_score'] / 2  # Convert 0-2 scale to 0-1
            quality_components.append(bathroom_norm * 0.25)
        
        # Floor preference score (already normalized)
        if 'floor_preference_score' in df.columns:
            quality_components.append(df['floor_preference_score'] * 0.1)
        
        # Combine and ensure 0-1 range
        if quality_components:
            total_quality = sum(quality_components)
            return np.clip(total_quality, 0, 1)
        else:
            return pd.Series(0.5, index=df.index)  # Neutral score if no data
    
    df_result['improved_quality_score'] = calculate_robust_quality_score(df_result)
    
    # 3. IMPROVED MARKET POSITION CALCULATION
    # Use robust percentile ranking
    def calculate_robust_market_position(df):
        """Calculate market position using robust percentile ranking."""
        if 'price_per_m2_market_percentile' in df.columns:
            # Invert percentile so lower prices get higher scores
            market_score = 1 - df['price_per_m2_market_percentile']
            return np.clip(market_score, 0, 1)
        else:
            # Fallback: calculate percentiles directly
            price_percentile = df['price_per_m2'].rank(pct=True)
            return 1 - price_percentile
    
    df_result['improved_market_position'] = calculate_robust_market_position(df_result)
    
    # 4. IMPROVED SIZE APPROPRIATENESS CALCULATION
    # Fix the mathematical error that can produce negative values
    def calculate_robust_size_score(df):
        """Calculate size appropriateness with proper bounds."""
        if 'area_m2_mean' in df.columns:
            # Calculate relative difference from district mean
            relative_diff = np.abs(df['area_m2'] - df['area_m2_mean']) / df['area_m2_mean']
            
            # Convert to score: smaller difference = higher score
            # Properties within 20% of mean get score > 0.8
            # Properties within 50% of mean get score > 0.5
            size_score = np.maximum(0, 1 - (relative_diff / 0.6))  # Gentle decay
            return np.clip(size_score, 0, 1)
        else:
            return pd.Series(0.5, index=df.index)  # Neutral score if no district data
    
    df_result['improved_size_score'] = calculate_robust_size_score(df_result)
    
    # 5. DOCUMENTATION SCORE
    def calculate_documentation_score(df):
        """Calculate documentation score."""
        if 'tech_passport_score' in df.columns:
            return np.clip(df['tech_passport_score'], 0, 1)
        else:
            return pd.Series(0.5, index=df.index)  # Neutral score
    
    df_result['improved_documentation_score'] = calculate_documentation_score(df_result)
    
    # 6. CALCULATE IMPROVED COMPOSITE SCORE
    # Use refined weights based on analysis
    weights = {
        'price_advantage': 0.40,      # 40% - Most important for bargains
        'quality_features': 0.25,     # 25% - Property condition  
        'market_position': 0.20,      # 20% - Overall market position
        'size_appropriateness': 0.10, # 10% - Size fit for district
        'documentation': 0.05         # 5% - Legal documentation
    }
    
    # Calculate weighted composite score
    improved_score = (
        df_result['improved_price_advantage'] * weights['price_advantage'] +
        df_result['improved_quality_score'] * weights['quality_features'] +
        df_result['improved_market_position'] * weights['market_position'] +
        df_result['improved_size_score'] * weights['size_appropriateness'] +
        df_result['improved_documentation_score'] * weights['documentation']
    )
    
    df_result['improved_bargain_score'] = np.clip(improved_score, 0, 1)
    
    # 7. CALCULATE THRESHOLDS (GLOBAL OR CATEGORY-AWARE)
    if use_category_aware and len(df_result['renovation_category'].unique()) > 1:
        # Calculate category-specific thresholds
        category_thresholds = calculate_category_aware_thresholds(df_result, 'improved_bargain_score')
        
        # 8. CATEGORIZE WITH CATEGORY-AWARE THRESHOLDS
        def categorize_category_aware_bargain(row):
            score = row['improved_bargain_score']
            renovation_cat = row['renovation_category']
            
            if renovation_cat not in category_thresholds:
                # Fallback to global categorization
                return categorize_global_bargain(score, df_result['improved_bargain_score'])
            
            thresholds = category_thresholds[renovation_cat]
            
            if score >= thresholds['exceptional_opportunity']:
                return 'exceptional_opportunity'
            elif score >= thresholds['excellent_bargain']:
                return 'excellent_bargain'
            elif score >= thresholds['good_bargain']:
                return 'good_bargain'
            elif score >= thresholds['fair_value']:
                return 'fair_value'
            elif score >= 0.4:  # Fixed threshold for market price
                return 'market_price'
            else:
                return 'overpriced'
        
        df_result['improved_bargain_category'] = df_result.apply(categorize_category_aware_bargain, axis=1)
        
        # Keep global classification for comparison
        score_percentiles = df_result['improved_bargain_score'].quantile([0.90, 0.75, 0.50, 0.25])
        global_thresholds = {
            'exceptional_opportunity': score_percentiles[0.90],
            'excellent_bargain': score_percentiles[0.75],
            'good_bargain': score_percentiles[0.50],
            'fair_value': score_percentiles[0.25]
        }
        
        def categorize_global_bargain_row(score):
            return categorize_global_bargain(score, df_result['improved_bargain_score'])
        
        df_result['global_bargain_category'] = df_result['improved_bargain_score'].apply(categorize_global_bargain_row)
        
    else:
        # Use global thresholds (original behavior)
        score_percentiles = df_result['improved_bargain_score'].quantile([0.90, 0.75, 0.50, 0.25])
        
        adaptive_thresholds = {
            'exceptional_opportunity': score_percentiles[0.90],  # Top 10%
            'excellent_bargain': score_percentiles[0.75],        # Top 25%
            'good_bargain': score_percentiles[0.50],             # Top 50%
            'fair_value': score_percentiles[0.25],               # Top 75%
        }
        
        # Ensure minimum separation between thresholds
        min_gap = 0.03
        if adaptive_thresholds['excellent_bargain'] > adaptive_thresholds['exceptional_opportunity'] - min_gap:
            adaptive_thresholds['excellent_bargain'] = adaptive_thresholds['exceptional_opportunity'] - min_gap
        if adaptive_thresholds['good_bargain'] > adaptive_thresholds['excellent_bargain'] - min_gap:
            adaptive_thresholds['good_bargain'] = adaptive_thresholds['excellent_bargain'] - min_gap
        if adaptive_thresholds['fair_value'] > adaptive_thresholds['good_bargain'] - min_gap:
            adaptive_thresholds['fair_value'] = adaptive_thresholds['good_bargain'] - min_gap
        
        # 8. CATEGORIZE WITH GLOBAL THRESHOLDS
        def categorize_improved_bargain(score):
            if score >= adaptive_thresholds['exceptional_opportunity']:
                return 'exceptional_opportunity'
            elif score >= adaptive_thresholds['excellent_bargain']:
                return 'excellent_bargain'
            elif score >= adaptive_thresholds['good_bargain']:
                return 'good_bargain'
            elif score >= adaptive_thresholds['fair_value']:
                return 'fair_value'
            elif score >= 0.4:  # Fixed threshold for market price
                return 'market_price'
            else:
                return 'overpriced'
        
        df_result['improved_bargain_category'] = df_result['improved_bargain_score'].apply(categorize_improved_bargain)
        df_result['global_bargain_category'] = df_result['improved_bargain_category']  # Same as category-aware when global

    # 9. LOG ANALYSIS RESULTS
    logger.info("📈 Improved Bargain Analysis Results:")
    
    # Show category-aware vs global comparison if using category-aware
    if use_category_aware and 'global_bargain_category' in df_result.columns:
        logger.info("\n🏗️ Category-Aware vs Global Comparison:")
        
        # Overall distribution
        logger.info("\nCategory-Aware Distribution:")
        category_dist = df_result['improved_bargain_category'].value_counts()
        total = len(df_result)
        for category, count in category_dist.items():
            percentage = (count / total) * 100
            logger.info(f"  {category}: {count} properties ({percentage:.1f}%)")
        
        logger.info("\nGlobal Distribution:")
        global_dist = df_result['global_bargain_category'].value_counts()
        for category, count in global_dist.items():
            percentage = (count / total) * 100
            logger.info(f"  {category}: {count} properties ({percentage:.1f}%)")
        
        # Distribution by renovation category
        logger.info("\n🏗️ Excellent Deals by Renovation Category:")
        excellent_by_renovation = df_result[df_result['improved_bargain_category'] == 'excellent_bargain']['renovation_category'].value_counts()
        for renovation_cat, count in excellent_by_renovation.items():
            total_in_cat = (df_result['renovation_category'] == renovation_cat).sum()
            percentage = (count / total_in_cat) * 100 if total_in_cat > 0 else 0
            logger.info(f"  {renovation_cat}: {count}/{total_in_cat} ({percentage:.1f}%)")
        
        # Show impact of category-aware classification
        excellent_global = (df_result['global_bargain_category'] == 'excellent_bargain').sum()
        excellent_category = (df_result['improved_bargain_category'] == 'excellent_bargain').sum()
        logger.info(f"\n📊 Excellent deals - Global: {excellent_global}, Category-aware: {excellent_category}")
    
    return df_result

def categorize_global_bargain(score: float, all_scores: pd.Series) -> str:
    """Helper function for global bargain categorization."""
    percentiles = all_scores.quantile([0.90, 0.75, 0.50, 0.25])
    
    if score >= percentiles[0.90]:
        return 'exceptional_opportunity'
    elif score >= percentiles[0.75]:
        return 'excellent_bargain'
    elif score >= percentiles[0.50]:
        return 'good_bargain'
    elif score >= percentiles[0.25]:
        return 'fair_value'
    elif score >= 0.4:
        return 'market_price'
    else:
        return 'overpriced'

def apply_improved_bargain_algorithm(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Apply improved bargain algorithm to existing dataset.
    
    Args:
        input_csv: Path to input CSV with existing features
        output_csv: Path to save improved results (optional)
        
    Returns:
        DataFrame with improved bargain scores
    """
    logger.info(f"🚀 Applying improved bargain algorithm to {input_csv}")
    
    # Load existing data
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} properties with {len(df.columns)} features")
    
    # Apply improvements
    df_improved = calculate_improved_bargain_score(df)
    
    # Save results if requested
    if output_csv:
        df_improved.to_csv(output_csv, index=False)
        logger.info(f"✅ Saved improved results to {output_csv}")
    
    return df_improved

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved Bargain Finding Algorithm")
    parser.add_argument("input_csv", help="Path to input CSV file with existing features")
    parser.add_argument("-o", "--output", help="Path to output CSV file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    output_path = args.output or args.input_csv.replace('.csv', '_improved_bargains.csv')
    apply_improved_bargain_algorithm(args.input_csv, output_path) 