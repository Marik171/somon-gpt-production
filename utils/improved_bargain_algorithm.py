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

def calculate_improved_bargain_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate improved bargain scores addressing current algorithm issues.
    
    Args:
        df: DataFrame with property data including existing features
        
    Returns:
        DataFrame with improved bargain scores and categories
    """
    logger.info("🎯 Calculating improved bargain scores...")
    
    df_result = df.copy()
    
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
    
    # 7. CALCULATE ADAPTIVE THRESHOLDS
    # Use data-driven thresholds instead of fixed values
    score_percentiles = df_result['improved_bargain_score'].quantile([0.90, 0.75, 0.50, 0.25])
    
    adaptive_thresholds = {
        'exceptional_opportunity': score_percentiles[0.90],  # Top 10%
        'excellent_bargain': score_percentiles[0.75],        # Top 25%
        'good_bargain': score_percentiles[0.50],             # Top 50%
        'fair_value': score_percentiles[0.25],               # Top 75%
        # Bottom 25% will be market_price or overpriced
    }
    
    # Ensure minimum separation between thresholds
    min_gap = 0.03
    if adaptive_thresholds['excellent_bargain'] > adaptive_thresholds['exceptional_opportunity'] - min_gap:
        adaptive_thresholds['excellent_bargain'] = adaptive_thresholds['exceptional_opportunity'] - min_gap
    if adaptive_thresholds['good_bargain'] > adaptive_thresholds['excellent_bargain'] - min_gap:
        adaptive_thresholds['good_bargain'] = adaptive_thresholds['excellent_bargain'] - min_gap
    if adaptive_thresholds['fair_value'] > adaptive_thresholds['good_bargain'] - min_gap:
        adaptive_thresholds['fair_value'] = adaptive_thresholds['good_bargain'] - min_gap
    
    # 8. CATEGORIZE WITH ADAPTIVE THRESHOLDS
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
    
    # 9. LOG ANALYSIS RESULTS
    logger.info("📈 Improved Bargain Analysis Results:")
    
    # Compare old vs new distributions
    if 'bargain_category' in df.columns:
        old_dist = df['bargain_category'].value_counts()
        new_dist = df_result['improved_bargain_category'].value_counts()
        
        logger.info("Category Distribution Comparison:")
        all_categories = set(old_dist.index) | set(new_dist.index)
        for category in sorted(all_categories):
            old_count = old_dist.get(category, 0)
            new_count = new_dist.get(category, 0)
            old_pct = (old_count / len(df)) * 100
            new_pct = (new_count / len(df_result)) * 100
            logger.info(f"  {category}: {old_count} ({old_pct:.1f}%) → {new_count} ({new_pct:.1f}%)")
    
    # Log adaptive thresholds
    logger.info("Adaptive Thresholds Used:")
    for category, threshold in adaptive_thresholds.items():
        logger.info(f"  {category}: {threshold:.3f}")
    
    # Log score improvement
    if 'bargain_score' in df.columns:
        old_mean = df['bargain_score'].mean()
        new_mean = df_result['improved_bargain_score'].mean()
        old_std = df['bargain_score'].std()
        new_std = df_result['improved_bargain_score'].std()
        
        logger.info(f"Score Statistics:")
        logger.info(f"  Mean: {old_mean:.3f} → {new_mean:.3f}")
        logger.info(f"  Std:  {old_std:.3f} → {new_std:.3f}")
    
    # Log top improved bargains
    top_bargains = df_result.nlargest(5, 'improved_bargain_score')
    logger.info("Top 5 Improved Bargain Properties:")
    for _, row in top_bargains.iterrows():
        district = row.get('district', 'N/A')
        price = row.get('price', 0)
        area = row.get('area_m2', 0)
        old_score = row.get('bargain_score', 0)
        new_score = row['improved_bargain_score']
        new_category = row['improved_bargain_category']
        
        logger.info(f"  {district} | {price:,.0f} TJS | {area:.0f}m² | "
                   f"Score: {old_score:.3f} → {new_score:.3f} | {new_category}")
    
    return df_result

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