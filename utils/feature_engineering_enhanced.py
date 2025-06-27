#!/usr/bin/env python3
"""
Feature Engineering for Bargain Finder Real Estate System

This script creates engineered features to help identify undervalued real estate listings
by calculating price ratios, market indicators, and composite bargain scores.

Author: Real Estate Analytics Assistant
Date: June 2025
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')

# Fix sklearn threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import MiniBatchKMeans  # Use MiniBatchKMeans instead of KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available, will use simple clustering")

try:
    import joblib
    import yaml
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logging.warning('joblib or yaml not available, will use basic rental estimates')

# Global rental model variables
RENTAL_MODEL = None
RENTAL_MODEL_METADATA = None
RENTAL_DISTRICT_STATS = None


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration for the feature engineering script."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('feature_engineering.log'),
            logging.StreamHandler()
        ]
    )


def load_cleaned_data(csv_path: str) -> pd.DataFrame:
    """Load the cleaned dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Successfully loaded cleaned dataset with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logging.error(f"Error loading cleaned CSV file: {e}")
        raise


def calculate_price_per_sqm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price per square meter for each property."""
    logging.info("Calculating price per square meter")
    
    # Calculate price per m²
    df['price_per_m2'] = df['price'] / df['area_m2']
    
    # Log statistics
    price_per_m2_stats = df['price_per_m2'].describe()
    logging.info(f"Price per m² statistics: min={price_per_m2_stats['min']:.0f}, "
                f"max={price_per_m2_stats['max']:.0f}, mean={price_per_m2_stats['mean']:.0f} TJS/m²")
    
    return df


def calculate_district_market_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market indicators by district (average prices, price ranges, etc.)."""
    logging.info("Calculating district market indicators")
    
    # Calculate district-level statistics
    district_stats = df.groupby('district').agg({
        'price': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'price_per_m2': ['mean', 'median', 'std'],
        'area_m2': ['mean', 'median']
    }).round(0)
    
    # Flatten column names
    district_stats.columns = ['_'.join(col).strip() for col in district_stats.columns]
    district_stats = district_stats.reset_index()
    
    # Filter districts with at least 2 properties for reliable statistics
    district_stats = district_stats[district_stats['price_count'] >= 2]
    
    logging.info(f"Calculated market indicators for {len(district_stats)} districts with ≥2 properties")
    
    # Merge back to main dataframe
    df = df.merge(district_stats, on='district', how='left', suffixes=('', '_district'))
    
    return df


def calculate_price_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price ratios relative to district averages."""
    logging.info("Calculating price ratios relative to district market")
    
    # Price ratio vs district average
    df['price_vs_district_avg'] = df['price'] / df['price_mean']
    df['price_per_m2_vs_district_avg'] = df['price_per_m2'] / df['price_per_m2_mean']
    
    # Price percentile within district
    df['price_percentile_in_district'] = df.groupby('district')['price'].rank(pct=True)
    df['price_per_m2_percentile_in_district'] = df.groupby('district')['price_per_m2'].rank(pct=True)
    
    # Log some statistics
    avg_price_ratio = df['price_vs_district_avg'].mean()
    avg_price_per_m2_ratio = df['price_per_m2_vs_district_avg'].mean()
    
    logging.info(f"Average price vs district average: {avg_price_ratio:.2f}")
    logging.info(f"Average price/m² vs district average: {avg_price_per_m2_ratio:.2f}")
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features with proper mappings based on actual data values.
    """
    df = df.copy()
    logging.info("Encoding categorical features")
    
    # Renovation quality mapping (based on actual data values)
    renovation_mapping = {
        'Без ремонта (коробка)': 1,  # Lowest quality
        'С ремонтом': 2,             # Standard renovation
        'Новый ремонт': 3,           # Highest quality
        'Неизвестно': 0              # Handle unknown values
    }
    
    # Heating system mapping (based on actual data values)
    heating_mapping = {
        'Нет': 0,          # No heating
        'Есть': 1,         # Has heating
        'Неизвестно': 0    # Handle unknown values as no heating
    }
    
    # Bathroom mapping (based on actual data values)
    bathroom_mapping = {
        'Совмещенный': 1,   # Combined bathroom
        'Раздельный': 2,    # Separate bathroom (better)
        'Неизвестно': 0     # Handle unknown values
    }
    
    # Tech passport mapping (based on actual data values)
    tech_passport_mapping = {
        'Нет': 0,          # No tech passport
        'Есть': 1,         # Has tech passport
        'Неизвестно': 0    # Handle unknown values as no passport
    }
    
    # Apply mappings with proper handling of missing values
    df['renovation_score'] = df['renovation'].map(renovation_mapping).fillna(0)
    df['heating_score'] = df['heating'].map(heating_mapping).fillna(0)
    df['bathroom_score'] = df['bathroom'].map(bathroom_mapping).fillna(0)
    df['tech_passport_score'] = df['tech_passport'].map(tech_passport_mapping).fillna(0)
    
    # Note: build_type is constant (all "Вторичный рынок"), so no score needed
    # Note: built_status is constant (all "Построено"), so no score needed
    
    logging.info("Categorical features encoded successfully")
    return df


def calculate_property_age_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate property age proxy based on renovation status."""
    logging.info("Calculating property age proxy")
    
    # Create age proxy based primarily on renovation status
    # Since all properties have the same build_type, we use renovation as the main indicator
    df['age_proxy_score'] = df['renovation_score']
    
    # Normalize to 0-1 scale where 1 = newest/best condition
    if df['age_proxy_score'].max() > df['age_proxy_score'].min():
        df['age_proxy_normalized'] = (df['age_proxy_score'] - df['age_proxy_score'].min()) / \
                                    (df['age_proxy_score'].max() - df['age_proxy_score'].min())
    else:
        df['age_proxy_normalized'] = 0.5  # Default if all values are the same
    
    return df


def calculate_size_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamically categorize apartments by area size based on actual data distribution.
    Auto-detects the dominant room type and creates quartile-based categories from real data.
    """
    logging.info("Dynamically categorizing apartments by area size based on actual data distribution")
    
    # Extract room information from URLs
    def extract_room_count(url):
        if pd.isna(url):
            return None
        # Look for patterns like '1-komn', '2-komn', '3-komn', etc.
        import re
        match = re.search(r'(\d+)-komn', url)
        if match:
            return int(match.group(1))
        return None
    
    df['room_count'] = df['url'].apply(extract_room_count)
    
    # Log room distribution
    room_dist = df['room_count'].value_counts().sort_index()
    logging.info(f"Room distribution: {dict(room_dist)}")
    
    # Detect dominant room type (the one with most properties)
    if not df['room_count'].isna().all():
        dominant_room_type = df['room_count'].mode().iloc[0] if len(df['room_count'].mode()) > 0 else 2
        dominant_data = df[df['room_count'] == dominant_room_type]
        
        logging.info(f"Detected dominant room type: {dominant_room_type}-room apartments ({len(dominant_data)} properties)")
        
        # Calculate area distribution statistics for the dominant room type
        area_data = dominant_data['area_m2'].dropna()
        
        if len(area_data) >= 4:  # Need at least 4 properties for quartiles
            # Calculate quartiles from actual data
            q25 = area_data.quantile(0.25)
            q50 = area_data.quantile(0.50)  # median
            q75 = area_data.quantile(0.75)
            
            area_stats = area_data.describe()
            logging.info(f"Area distribution for {dominant_room_type}-room apartments:")
            logging.info(f"  Range: {area_stats['min']:.1f} - {area_stats['max']:.1f}m²")
            logging.info(f"  Mean: {area_stats['mean']:.1f}m², Median: {q50:.1f}m²")
            logging.info(f"  Quartiles: Q25={q25:.1f}m², Q50={q50:.1f}m², Q75={q75:.1f}m²")
            
            # Create dynamic category names based on room type
            if dominant_room_type == 1:
                category_names = {
                    'compact': 'compact_studio',
                    'standard': 'standard_studio', 
                    'spacious': 'spacious_studio',
                    'premium': 'premium_studio'
                }
            else:
                category_names = {
                    'compact': f'compact_{int(dominant_room_type)}room',
                    'standard': f'standard_{int(dominant_room_type)}room',
                    'spacious': f'spacious_{int(dominant_room_type)}room', 
                    'premium': f'premium_{int(dominant_room_type)}room'
                }
            
            logging.info(f"Dynamic categories based on actual data quartiles:")
            logging.info(f"  {category_names['compact']}: < {q25:.0f}m² (bottom 25%)")
            logging.info(f"  {category_names['standard']}: {q25:.0f} - {q50:.0f}m² (25th-50th percentile)")
            logging.info(f"  {category_names['spacious']}: {q50:.0f} - {q75:.0f}m² (50th-75th percentile)")
            logging.info(f"  {category_names['premium']}: > {q75:.0f}m² (top 25%)")
            
            # Define categorization function using actual data quartiles
            def categorize_by_actual_distribution(row):
                room_count = row['room_count']
                area = row['area_m2']
                
                if pd.isna(area):
                    return 'unknown_size'
                
                # Use quartile-based categorization for the dominant room type
                if room_count == dominant_room_type:
                    if area < q25:
                        return category_names['compact']
                    elif area < q50:
                        return category_names['standard']
                    elif area < q75:
                        return category_names['spacious']
                    else:
                        return category_names['premium']
                
                # For non-dominant room types, use relative positioning
                elif not pd.isna(room_count):
                    # Get the room-specific area if available
                    room_specific_data = df[df['room_count'] == room_count]['area_m2'].dropna()
                    
                    if len(room_specific_data) >= 2:
                        # Use room-specific median if we have enough data
                        room_median = room_specific_data.median()
                        if room_count == 1:
                            if area < room_median:
                                return 'compact_studio'
                            else:
                                return 'standard_studio'
                        else:
                            if area < room_median:
                                return f'compact_{int(room_count)}room'
                            else:
                                return f'standard_{int(room_count)}room'
                    else:
                        # Fallback to general categorization for rare room types
                        if room_count == 1:
                            return 'standard_studio' if area >= 35 else 'compact_studio'
                        else:
                            return f'standard_{int(room_count)}room'
                
                # Fallback for unknown room counts
                if area < q25:
                    return 'compact_unknown'
                elif area < q50:
                    return 'standard_unknown'
                elif area < q75:
                    return 'spacious_unknown'
                else:
                    return 'premium_unknown'
            
            # Apply the dynamic categorization
            df['size_category'] = df.apply(categorize_by_actual_distribution, axis=1)
            
            # Create preference scores based on normal distribution preference
            # Standard and spacious are typically most preferred (close to median)
            preference_mapping = {
                category_names['compact']: 0.7,   # Smaller but functional
                category_names['standard']: 1.0,  # Most preferred (close to median)
                category_names['spacious']: 0.9,  # Good size, slightly above median
                category_names['premium']: 0.8    # Large but might be expensive
            }
            
            # Add preferences for other room types
            for room_type in [1, 2, 3, 4, 5, 6]:
                if room_type == 1:
                    preference_mapping.update({
                        'compact_studio': 0.7, 'standard_studio': 1.0,
                        'spacious_studio': 0.9, 'premium_studio': 0.8
                    })
                else:
                    preference_mapping.update({
                        f'compact_{room_type}room': 0.7,
                        f'standard_{room_type}room': 1.0,
                        f'spacious_{room_type}room': 0.9,
                        f'premium_{room_type}room': 0.8
                    })
            
            # Add fallback preferences
            preference_mapping.update({
                'compact_unknown': 0.7, 'standard_unknown': 1.0,
                'spacious_unknown': 0.9, 'premium_unknown': 0.8,
                'unknown_size': 0.5
            })
            
        else:
            logging.warning(f"Not enough data for {dominant_room_type}-room apartments, using fallback categorization")
            # Fallback to simple area-based categorization
            median_area = df['area_m2'].median()
            
            def simple_categorize(row):
                area = row['area_m2']
                room_count = row['room_count']
                
                if pd.isna(area):
                    return 'unknown_size'
                
                if not pd.isna(room_count):
                    if room_count == 1:
                        return 'standard_studio' if area >= 35 else 'compact_studio'
                    else:
                        suffix = f'{int(room_count)}room'
                        return f'standard_{suffix}' if area >= median_area else f'compact_{suffix}'
                else:
                    return 'standard_unknown' if area >= median_area else 'compact_unknown'
            
            df['size_category'] = df.apply(simple_categorize, axis=1)
            
            # Simple preference mapping
            preference_mapping = {
                'compact_studio': 0.7, 'standard_studio': 1.0,
                'compact_2room': 0.7, 'standard_2room': 1.0,
                'compact_3room': 0.7, 'standard_3room': 1.0,
                'compact_unknown': 0.7, 'standard_unknown': 1.0,
                'unknown_size': 0.5
            }
    
    else:
        logging.warning("No room information found in URLs, using area-only categorization")
        # Pure area-based categorization when no room info is available
        q25, q50, q75 = df['area_m2'].quantile([0.25, 0.5, 0.75])
        
        def area_only_categorize(area):
            if pd.isna(area):
                return 'unknown_size'
            elif area < q25:
                return 'compact_unit'
            elif area < q50:
                return 'standard_unit'
            elif area < q75:
                return 'spacious_unit'
            else:
                return 'premium_unit'
        
        df['size_category'] = df['area_m2'].apply(area_only_categorize)
        
        preference_mapping = {
            'compact_unit': 0.7, 'standard_unit': 1.0,
            'spacious_unit': 0.9, 'premium_unit': 0.8,
            'unknown_size': 0.5
        }
    
    # Apply size preference scores
    df['size_preference_score'] = df['size_category'].map(preference_mapping).fillna(0.5)
    
    # Log final category distribution
    size_dist = df['size_category'].value_counts()
    logging.info(f"Final dynamic size category distribution: {dict(size_dist)}")
    
    # Log statistics by detected categories
    unique_categories = df['size_category'].unique()
    for category in sorted(unique_categories):
        if category != 'unknown_size':
            cat_data = df[df['size_category'] == category]
            if len(cat_data) > 0:
                area_stats = cat_data['area_m2'].describe()
                logging.info(f"  {category} ({len(cat_data)} properties): "
                           f"area range {area_stats['min']:.0f}-{area_stats['max']:.0f}m², "
                           f"mean {area_stats['mean']:.1f}m²")
    
    # Log preference score distribution
    pref_stats = df['size_preference_score'].describe()
    logging.info(f"Size preference scores: min={pref_stats['min']:.2f}, mean={pref_stats['mean']:.2f}, max={pref_stats['max']:.2f}")
    
    return df


def calculate_floor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate floor-related features."""
    logging.info("Calculating floor features")
    
    # Floor category (ground, middle, top)
    def categorize_floor(floor):
        if floor == 1:
            return 'ground'
        elif floor <= 3:
            return 'low'
        elif floor <= 6:
            return 'middle'
        else:
            return 'high'
    
    df['floor_category'] = df['floor'].apply(categorize_floor)
    
    # Floor preference score (middle floors often preferred)
    floor_preference = {
        'ground': 0.6,  # Less preferred (noise, security)
        'low': 0.8,     # Good
        'middle': 1.0,  # Most preferred
        'high': 0.7     # Good views but elevator dependency
    }
    df['floor_preference_score'] = df['floor_category'].map(floor_preference)
    
    return df


def calculate_market_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market position features for bargain identification."""
    logging.info("Calculating market position features")
    
    # Overall market percentiles
    df['price_market_percentile'] = df['price'].rank(pct=True)
    df['price_per_m2_market_percentile'] = df['price_per_m2'].rank(pct=True)
    
    # Value score: good features vs low price
    # Higher score = better value (good features, relatively low price)
    # Updated to remove build_type_score since all properties have same build type
    feature_score = (df['renovation_score'] + df['heating_score'] + 
                    df['bathroom_score'] + df['floor_preference_score'] * 3 +
                    df['size_preference_score'] * 2) / 5
    
    # Normalize feature score
    if feature_score.max() > feature_score.min():
        feature_score_norm = (feature_score - feature_score.min()) / (feature_score.max() - feature_score.min())
    else:
        feature_score_norm = pd.Series([0.5] * len(feature_score), index=feature_score.index)
    
    # Value score: high features, low price percentile = good value
    df['value_score'] = feature_score_norm * (1 - df['price_per_m2_market_percentile'])
    
    return df


def add_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features using publication date."""
    logging.info("Adding time-based features")
    
    # Handle missing publication dates
    if df['publication_date'].isna().all():
        logging.warning("No publication dates available, using current date as default")
        df['publication_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Convert publication_date to datetime
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    current_date = datetime.now()
    
    # Days since publication (use 0 for missing dates)
    df['days_on_market'] = (current_date - df['publication_date']).dt.days.fillna(0)
    
    # Urgency score (longer on market might indicate flexibility)
    df['urgency_score'] = np.where(df['days_on_market'] > 30, 0.8,
                          np.where(df['days_on_market'] > 14, 0.6,
                          np.where(df['days_on_market'] > 7, 0.4, 0.2)))
    
    # Day of week and month features (handle NaT values)
    df['publication_weekday'] = df['publication_date'].dt.day_name().fillna('Unknown')
    df['publication_month'] = df['publication_date'].dt.month.fillna(0)
    
    return df


def create_district_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Create district clusters based on market characteristics using enhanced KMeans clustering."""
    logging.info("Creating district clusters with enhanced feature set")
    
    # Prepare district-level data for clustering
    district_features = df.groupby('district').agg({
        'price_per_m2_mean': 'first',
        'price_mean': 'first', 
        'price_std': 'first',
        'area_m2_mean': 'first',
        'price_count': 'first'
    }).reset_index()
    
    # Calculate quality and infrastructure scores for each district
    district_quality_stats = df.groupby('district').agg({
        'renovation_score': 'mean',
        'heating_score': 'mean', 
        'tech_passport_score': 'mean',
        'photo_count': 'mean'
    }).round(2)
    
    # Rename columns to match feature names
    district_quality_stats.columns = [f'{col}_mean' for col in district_quality_stats.columns]
    district_quality_stats = district_quality_stats.reset_index()
    
    # Merge quality stats with district features
    district_features = district_features.merge(district_quality_stats, on='district', how='left')
    
    # Filter districts with enough data
    district_features = district_features[district_features['price_count'] >= 2]
    
    if len(district_features) >= 3:  # Need at least 3 districts for clustering
        try:
            # Enhanced clustering with more significant features for robust market segmentation
            clustering_feature_names = [
                'price_per_m2_mean',        # Core pricing metric
                'price_std',                # Price volatility in district
                'area_m2_mean',             # Average property size
                'price_count',              # Market activity/liquidity
                'renovation_score_mean',    # Quality indicator
                'heating_score_mean',       # Infrastructure quality
                'tech_passport_score_mean', # Legal/documentation quality
                'photo_count_mean'          # Marketing effort indicator
            ]
            
            # Select features for clustering (only those that exist and have valid data)
            available_features = []
            for feature in clustering_feature_names:
                if feature in district_features.columns:
                    # Check if feature has valid data (not all NaN)
                    if not district_features[feature].isna().all():
                        available_features.append(feature)
            
            if len(available_features) < 3:
                # Fall back to basic features if enhanced features aren't available
                available_features = ['price_per_m2_mean', 'price_mean', 'area_m2_mean']
                available_features = [f for f in available_features if f in district_features.columns]
            
            features_for_clustering = district_features[available_features].fillna(0)
            
            logging.info(f"Enhanced clustering using {len(available_features)} features: {available_features}")
            
            # Standardize features manually 
            features_scaled = (features_for_clustering - features_for_clustering.mean()) / (features_for_clustering.std() + 1e-8)
            features_scaled = features_scaled.fillna(0)  # Handle any remaining NaN
            
            # Custom KMeans implementation
            n_clusters = min(4, len(district_features))
            cluster_labels = custom_kmeans(features_scaled.values, n_clusters, random_state=42)
            
            district_features['market_cluster'] = cluster_labels
            
            # Sort clusters by average price per m2 for consistent labeling
            cluster_prices = []
            for cluster_id in range(n_clusters):
                cluster_mask = district_features['market_cluster'] == cluster_id
                if cluster_mask.any():
                    avg_price_per_m2 = district_features[cluster_mask]['price_per_m2_mean'].mean()
                    cluster_prices.append((cluster_id, avg_price_per_m2))
            
            # Sort by price and assign labels
            cluster_prices.sort(key=lambda x: x[1])
            cluster_labels_names = ['budget_market', 'mid_market', 'premium_market', 'luxury_market']
            
            cluster_mapping = {}
            for i, (cluster_id, _) in enumerate(cluster_prices):
                if i < len(cluster_labels_names):
                    cluster_mapping[cluster_id] = cluster_labels_names[i]
                else:
                    cluster_mapping[cluster_id] = 'premium_market'  # fallback
            
            district_features['market_segment'] = district_features['market_cluster'].map(cluster_mapping)
            
            # Log enhanced cluster information
            logging.info("Enhanced KMeans clustering successful!")
            for i, (cluster_id, price) in enumerate(cluster_prices):
                segment_name = cluster_mapping[cluster_id]
                districts_in_cluster = district_features[district_features['market_cluster'] == cluster_id]['district'].tolist()
                
                # Calculate cluster characteristics
                cluster_data = district_features[district_features['market_cluster'] == cluster_id]
                avg_quality = cluster_data['renovation_score_mean'].mean() if 'renovation_score_mean' in cluster_data.columns else 0
                avg_photos = cluster_data['photo_count_mean'].mean() if 'photo_count_mean' in cluster_data.columns else 0
                market_size = cluster_data['price_count'].sum()
                
                logging.info(f"  {segment_name}: {price:.0f} TJS/m², Quality: {avg_quality:.1f}, Photos: {avg_photos:.1f}, "
                           f"Market size: {market_size} properties")
                logging.info(f"    Districts: {districts_in_cluster}")
            
            # Merge back to main dataframe
            df = df.merge(district_features[['district', 'market_segment']], on='district', how='left')
            
            # Log cluster distribution
            cluster_dist = df['market_segment'].value_counts()
            total_properties = len(df)
            logging.info("Enhanced market segment distribution:")
            for segment, count in cluster_dist.items():
                percentage = count / total_properties * 100
                logging.info(f"  {segment}: {count} properties ({percentage:.1f}%)")
            
        except Exception as e:
            logging.warning(f"Enhanced KMeans clustering failed ({e}), falling back to quartile-based clustering")
            # Fallback to quartile-based clustering
            df = _fallback_quartile_clustering(df, district_features)
    else:
        logging.warning("Not enough districts for clustering, using quartile-based clustering")
        # Fallback to quartile-based clustering
        df = _fallback_quartile_clustering(df, district_features)
    
    return df


def custom_kmeans(X, n_clusters, random_state=42, max_iters=100):
    """Custom KMeans implementation using only numpy to avoid sklearn threading issues."""
    np.random.seed(random_state)
    
    # Initialize centroids randomly
    n_samples, n_features = X.shape
    centroids = np.random.randn(n_clusters, n_features)
    
    for _ in range(max_iters):
        # Assign points to closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k] 
                                 for k in range(n_clusters)])
        
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
    
    return labels


def _fallback_quartile_clustering(df: pd.DataFrame, district_features: pd.DataFrame) -> pd.DataFrame:
    """Fallback clustering method using price quartiles."""
    if len(district_features) >= 3:
        # Use price per m2 percentiles for market segmentation
        price_per_m2_values = district_features['price_per_m2_mean'].values
        
        # Calculate quartiles for market segmentation
        q25 = np.percentile(price_per_m2_values, 25)
        q50 = np.percentile(price_per_m2_values, 50)  
        q75 = np.percentile(price_per_m2_values, 75)
        
        def assign_market_segment(price_per_m2):
            if price_per_m2 <= q25:
                return 'budget_market'
            elif price_per_m2 <= q50:
                return 'mid_market'
            elif price_per_m2 <= q75:
                return 'premium_market'
            else:
                return 'luxury_market'
        
        district_features['market_segment'] = district_features['price_per_m2_mean'].apply(assign_market_segment)
        
        # Log market segment thresholds
        logging.info(f"Quartile-based market segmentation thresholds (TJS/m²):")
        logging.info(f"  Budget market: ≤{q25:.0f}")
        logging.info(f"  Mid market: {q25:.0f} - {q50:.0f}")
        logging.info(f"  Premium market: {q50:.0f} - {q75:.0f}")
        logging.info(f"  Luxury market: >{q75:.0f}")
        
        # Merge back to main dataframe
        df = df.merge(district_features[['district', 'market_segment']], on='district', how='left')
        
        # Log cluster distribution
        cluster_dist = df['market_segment'].value_counts()
        logging.info(f"Market segment distribution: {dict(cluster_dist)}")
        
        # Log districts by segment
        for segment in ['budget_market', 'mid_market', 'premium_market', 'luxury_market']:
            districts_in_segment = district_features[district_features['market_segment'] == segment]['district'].tolist()
            if districts_in_segment:
                logging.info(f"  {segment}: {districts_in_segment}")
    else:
        logging.warning("Not enough districts for any clustering, assigning default market segment")
        df['market_segment'] = 'general_market'
    
    return df


def calculate_bargain_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive bargain score using improved robust algorithm."""
    logging.info("💰 Calculating improved bargain scores...")
    
    # Import the improved algorithm
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    
    try:
        from improved_bargain_algorithm import calculate_improved_bargain_score
        
        # Use the improved algorithm
        df_with_improved = calculate_improved_bargain_score(df)
        
        # Copy improved scores to main columns (for backward compatibility)
        df['bargain_score'] = df_with_improved['improved_bargain_score']
        df['bargain_category'] = df_with_improved['improved_bargain_category']
        
        # Also keep the component scores for analysis
        component_cols = [col for col in df_with_improved.columns if col.startswith('improved_')]
        for col in component_cols:
            df[col] = df_with_improved[col]
        
        logging.info("✅ Applied improved bargain scoring algorithm")
        
    except ImportError as e:
        logging.warning(f"Could not import improved algorithm: {e}")
        logging.info("Falling back to legacy bargain scoring...")
        
        # Fallback to legacy algorithm (improved version of original)
        # Components of bargain score (all should be between 0-1)
        components = {
            'price_advantage': np.clip(1.5 - df['price_per_m2_vs_district_avg'], 0, 1),  # Fixed mathematical issue
            'quality_features': np.clip((df['renovation_score'] / 3 + df['heating_score'] + 
                               df['bathroom_score'] / 2 + df['floor_preference_score']) / 4, 0, 1),
            'market_position': np.clip(1 - df['price_per_m2_market_percentile'], 0, 1),
            'tech_passport_bonus': np.clip(df['tech_passport_score'] * 0.1, 0, 1),
            'size_appropriateness': np.clip(np.maximum(0, 1 - np.abs(df['area_m2'] - df['area_m2_mean']) / (df['area_m2_mean'] * 0.6)), 0, 1)  # Fixed negative values
        }
        
        # Improved weights
        weights = {
            'price_advantage': 0.4,      # 40% - most important
            'quality_features': 0.25,    # 25% - property quality
            'market_position': 0.2,      # 20% - market positioning
            'tech_passport_bonus': 0.05, # 5% - documentation
            'size_appropriateness': 0.1  # 10% - size fit
        }
        
        # Calculate weighted bargain score
        bargain_score = sum(components[comp] * weights[comp] for comp in components.keys())
        df['bargain_score'] = np.clip(bargain_score, 0, 1)  # Ensure 0-1 range
        
        # Use adaptive thresholds
        score_percentiles = df['bargain_score'].quantile([0.90, 0.75, 0.50, 0.25])
        
        def categorize_bargain_adaptive(score):
            if score >= score_percentiles[0.90]:
                return 'exceptional_opportunity'
            elif score >= score_percentiles[0.75]:
                return 'excellent_bargain'
            elif score >= score_percentiles[0.50]:
                return 'good_bargain'
            elif score >= score_percentiles[0.25]:
                return 'fair_value'
            elif score >= 0.4:
                return 'market_price'
            else:
                return 'overpriced'
        
        df['bargain_category'] = df['bargain_score'].apply(categorize_bargain_adaptive)
    
    # Log bargain distribution
    bargain_dist = df['bargain_category'].value_counts()
    logging.info(f"Bargain distribution: {dict(bargain_dist)}")
    
    # Log top bargains
    top_bargains = df.nlargest(5, 'bargain_score')[['ad_number', 'district', 'price', 
                                                   'area_m2', 'price_per_m2', 'bargain_score', 'bargain_category']]
    logging.info("Top 5 bargain properties:")
    for _, row in top_bargains.iterrows():
        logging.info(f"  Ad {row['ad_number']}: {row['district']}, {row['area_m2']}m², "
                    f"{row['price']:,.0f} TJS ({row['price_per_m2']:.0f} TJS/m²), "
                    f"Score: {row['bargain_score']:.3f} ({row['bargain_category']})")
    
    return df


def save_engineered_features(df: pd.DataFrame, output_dir: str, base_filename: str = "listings_with_features") -> Tuple[str, Optional[str]]:
    """Save the dataset with engineered features."""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    parquet_path = os.path.join(output_dir, f"{base_filename}.parquet")
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved engineered dataset to CSV: {csv_path}")
    
    # Save Parquet
    try:
        df.to_parquet(parquet_path, index=False)
        logging.info(f"Saved engineered dataset to Parquet: {parquet_path}")
        return csv_path, parquet_path
    except ImportError:
        logging.warning("PyArrow not available. Skipping Parquet export.")
        return csv_path, None


def generate_feature_summary(df: pd.DataFrame) -> None:
    """Generate summary of engineered features."""
    logging.info("=== FEATURE ENGINEERING SUMMARY ===")
    
    # Basic statistics
    logging.info(f"Total properties: {len(df)}")
    logging.info(f"Total features: {len(df.columns)}")
    
    # Key feature statistics
    key_features = ['price_per_m2', 'price_vs_district_avg', 'bargain_score', 'value_score']
    for feature in key_features:
        if feature in df.columns:
            stats = df[feature].describe()
            logging.info(f"{feature}: min={stats['min']:.3f}, mean={stats['mean']:.3f}, max={stats['max']:.3f}")
    
    # Bargain categories
    if 'bargain_category' in df.columns:
        bargain_summary = df['bargain_category'].value_counts()
        logging.info("Bargain categories:")
        for category, count in bargain_summary.items():
            logging.info(f"  {category}: {count} properties ({count/len(df)*100:.1f}%)")
    
    # Market segments
    if 'market_segment' in df.columns:
        segment_summary = df['market_segment'].value_counts()
        logging.info("Market segments:")
        for segment, count in segment_summary.items():
            logging.info(f"  {segment}: {count} properties ({count/len(df)*100:.1f}%)")


def load_rental_market_data():
    """Load rental market data to create proper rental-based features."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        somon_project_root = os.path.dirname(current_dir)
        parent_root = os.path.dirname(somon_project_root)
        rental_data_path = os.path.join(parent_root, "rental_prediction/data/raw/rental_data_khujand_all_20250624_173607.csv")
        
        if os.path.exists(rental_data_path):
            rental_df = pd.read_csv(rental_data_path)
            logging.info(f"✅ Loaded rental market data: {len(rental_df)} records")
            return rental_df
        else:
            logging.warning("⚠️ Rental market data not found")
            return None
    except Exception as e:
        logging.warning(f"⚠️ Failed to load rental market data: {e}")
        return None

def load_rental_prediction_model():
    """Load the rental prediction model and metadata."""
    global RENTAL_MODEL, RENTAL_MODEL_METADATA, RENTAL_DISTRICT_STATS
    
    try:
        # Try to load from rental_prediction directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        somon_project_root = os.path.dirname(current_dir)
        parent_root = os.path.dirname(somon_project_root)
        rental_model_path = os.path.join(parent_root, "rental_prediction/models/xgboost_model.joblib")
        rental_metadata_path = os.path.join(parent_root, "rental_prediction/models/xgboost_metadata.yaml")
        
        if os.path.exists(rental_model_path) and os.path.exists(rental_metadata_path):
            RENTAL_MODEL = joblib.load(rental_model_path)
            try:
                with open(rental_metadata_path, 'r') as f:
                    RENTAL_MODEL_METADATA = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                # Handle numpy objects in YAML by loading only essential parts
                logging.warning("YAML contains numpy objects, loading feature names only")
                with open(rental_metadata_path, 'r') as f:
                    content = f.read()
                    # Extract feature names manually
                    import re
                    feature_match = re.search(r'feature_names:\s*\n((?:- .*\n)*)', content)
                    if feature_match:
                        feature_lines = feature_match.group(1).strip().split('\n')
                        feature_names = [line.strip('- ').strip() for line in feature_lines]
                        RENTAL_MODEL_METADATA = {'feature_names': feature_names}
                    else:
                        RENTAL_MODEL_METADATA = None
            
            # Load actual district statistics from training data
            RENTAL_DISTRICT_STATS = load_training_district_statistics()
            
            logging.info("✅ Rental prediction model loaded successfully")
            logging.info("🔧 Will properly align features with rental model expectations")
            if RENTAL_DISTRICT_STATS:
                logging.info(f"✅ Loaded actual district statistics for {len(RENTAL_DISTRICT_STATS)} districts")
            else:
                logging.warning("⚠️ Could not load district statistics, using fallback values")
            return True
        else:
            logging.warning("⚠️ Rental prediction model not found - using market estimates")
            return False
    except Exception as e:
        logging.warning(f"⚠️ Failed to load rental prediction model: {e}")
        return False

def load_training_district_statistics():
    """Load actual district statistics from the rental training data."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        somon_project_root = os.path.dirname(current_dir)
        parent_root = os.path.dirname(somon_project_root)
        training_data_path = os.path.join(parent_root, "rental_prediction/data/features/engineered_features.csv")
        
        if not os.path.exists(training_data_path):
            logging.warning(f"Training data not found at: {training_data_path}")
            return {}
        
        import pandas as pd
        training_data = pd.read_csv(training_data_path)
        logging.info(f"Loaded training data for district statistics: {len(training_data)} records")
        
        district_stats = {}
        for district in training_data['district'].unique():
            district_data = training_data[training_data['district'] == district]
            
            if len(district_data) >= 5:  # Only include districts with sufficient data
                stats = {
                    'avg_price': float(district_data['price'].mean()),
                    'median_price': float(district_data['price'].median()),
                    'price_per_m2': float(district_data['price_per_m2'].mean()),
                    'listing_count': int(len(district_data)),
                    'price_std': float(district_data['price'].std()),
                    'avg_area': float(district_data['area_m2'].mean())
                }
                district_stats[district] = stats
        
        logging.info(f"Extracted statistics for {len(district_stats)} districts from training data")
        return district_stats
        
    except Exception as e:
        logging.error(f"Error loading district statistics: {e}")
        return {}

def get_district_statistics(district, fallback_value=None):
    """Get district statistics for a given district with fallback."""
    global RENTAL_DISTRICT_STATS
    
    if not RENTAL_DISTRICT_STATS:
        RENTAL_DISTRICT_STATS = load_training_district_statistics()
    
    if district in RENTAL_DISTRICT_STATS:
        return RENTAL_DISTRICT_STATS[district]
    
    # Fallback: use average across all districts
    if RENTAL_DISTRICT_STATS:
        all_districts = list(RENTAL_DISTRICT_STATS.values())
        fallback_stats = {
            'avg_price': np.mean([d['avg_price'] for d in all_districts]),
            'median_price': np.mean([d['median_price'] for d in all_districts]),
            'price_per_m2': np.mean([d['price_per_m2'] for d in all_districts]),
            'listing_count': int(np.mean([d['listing_count'] for d in all_districts])),
            'price_std': np.mean([d['price_std'] for d in all_districts]),
            'avg_area': np.mean([d['avg_area'] for d in all_districts])
        }
        
        logging.debug(f"District '{district}' not found, using fallback statistics")
        return fallback_stats
    
    # Last resort: use the old hardcoded values (but warn)
    logging.warning(f"No district statistics available, using hardcoded fallback for '{district}'")
    return fallback_value or {
        'avg_price': 1800,
        'median_price': 1600,
        'price_per_m2': 25,
        'listing_count': 45,
        'price_std': 400,
        'avg_area': 68
    }
    
    # Original code commented out:
    # try:
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     somon_project_root = os.path.dirname(current_dir)
    #     parent_root = os.path.dirname(somon_project_root)
    #     rental_model_path = os.path.join(parent_root, "rental_prediction/models/xgboost_model.joblib")
    #     rental_metadata_path = os.path.join(parent_root, "rental_prediction/models/xgboost_metadata.yaml")
    #     
    #     if os.path.exists(rental_model_path) and os.path.exists(rental_metadata_path):
    #         RENTAL_MODEL = joblib.load(rental_model_path)
    #         with open(rental_metadata_path, 'r') as f:
    #             RENTAL_MODEL_METADATA = yaml.safe_load(f)
    #         logging.info("✅ Rental prediction model loaded successfully")
    #         return True
    #     else:
    #         logging.warning("⚠️ Rental prediction model not found - using market estimates")
    #         return False
    # except Exception as e:
    #     logging.warning(f"⚠️ Failed to load rental prediction model: {e}")
    #     return False

def apply_renovation_premium(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rental premium for properties requiring renovation once they're renovated."""
    logging.info("✨ Calculating rental premium for renovated properties")
    
    def get_renovation_premium(renovation_status, district, rooms):
        """Calculate rental premium multiplier based on renovation investment."""
        renovation_lower = str(renovation_status).lower()
        district_lower = str(district).lower()
        
        # Base premium rates for different renovation scenarios
        if 'без ремонта' in renovation_lower or 'коробка' in renovation_lower:
            # High renovation investment = high premium potential
            base_premium = 0.20  # 20% premium for full renovation
            
            # District-based adjustments
            if any(premium in district_lower for premium in ['универмаг', 'центр', 'шелкокомбинат']):
                # Premium districts: tenants pay more for quality
                district_multiplier = 1.25  # +25% to premium
            elif any(budget in district_lower for budget in ['29 мкр', '30 мкр', '31 мкр', '32 мкр']):
                # Budget districts: lower premium expectations
                district_multiplier = 0.75  # -25% to premium
            else:
                district_multiplier = 1.0
            
            # Room-based adjustments (larger apartments = higher premium sensitivity)
            room_multiplier = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2}.get(int(rooms), 1.0)
            
            final_premium = base_premium * district_multiplier * room_multiplier
            return 1 + final_premium  # Convert to multiplier
            
        elif 'с ремонтом' in renovation_lower:
            # Minor renovation = modest premium
            return 1.05  # 5% premium
            
        elif 'новый ремонт' in renovation_lower:
            # Already has new renovation = small existing premium
            return 1.03  # 3% premium (already built into market rent)
        else:
            # Unknown condition = no premium
            return 1.0
    
    # Calculate premium multipliers
    df['renovation_premium_multiplier'] = df.apply(
        lambda row: get_renovation_premium(row['renovation'], row['district'], row['rooms']), 
        axis=1
    )
    
    # Store original rent for comparison
    df['base_monthly_rent'] = df['estimated_monthly_rent'].copy()
    
    # Apply premium to rental income
    df['estimated_monthly_rent'] = (df['base_monthly_rent'] * df['renovation_premium_multiplier']).round(0)
    
    # Calculate premium amounts
    df['monthly_rent_premium'] = df['estimated_monthly_rent'] - df['base_monthly_rent']
    df['annual_rent_premium'] = df['monthly_rent_premium'] * 12
    
    # Log premium analysis
    premium_properties = df[df['renovation_premium_multiplier'] > 1.0]
    if len(premium_properties) > 0:
        logging.info(f"🎯 Renovation premium analysis:")
        logging.info(f"   Properties with rental premium: {len(premium_properties)}/{len(df)}")
        logging.info(f"   Average premium multiplier: {premium_properties['renovation_premium_multiplier'].mean():.2f}x")
        logging.info(f"   Average monthly premium: {premium_properties['monthly_rent_premium'].mean():.0f} TJS")
        logging.info(f"   Average annual premium: {premium_properties['annual_rent_premium'].mean():.0f} TJS")
        logging.info(f"   Premium range: {premium_properties['renovation_premium_multiplier'].min():.2f}x - {premium_properties['renovation_premium_multiplier'].max():.2f}x")
    
    return df

def calculate_renovation_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate renovation costs based on property characteristics."""
    logging.info("🔨 Calculating renovation costs and investment adjustments")
    
    def get_base_renovation_cost(rooms):
        """Base renovation cost by room count in TJS."""
        base_costs = {
            1: 20000,  # 1-room: 15,000 - 25,000 TJS (using mid-point)
            2: 32500,  # 2-room: 25,000 - 40,000 TJS
            3: 45000,  # 3-room: 35,000 - 55,000 TJS
            4: 57500,  # 4-room: 45,000 - 70,000 TJS
            5: 70000   # 5+ rooms: 55,000 - 85,000 TJS
        }
        return base_costs.get(int(rooms), 57500)  # Default to 4-room cost
    
    def get_area_adjustment(area_m2):
        """Area-based cost adjustment factor."""
        if area_m2 < 80:
            return 0.85  # Small apartments: -15% cost
        elif area_m2 > 120:
            return 1.20  # Large apartments: +20% cost
        else:
            return 1.0   # Standard apartments: base cost
    
    def get_district_adjustment(district):
        """District-based cost adjustment factor."""
        district_lower = str(district).lower()
        
        # Premium districts (higher standards expected)
        if any(premium in district_lower for premium in ['универмаг', 'центр', 'шелкокомбинат']):
            return 1.25  # +25% cost
        
        # Budget districts (lower standards acceptable)
        elif any(budget in district_lower for budget in ['29 мкр', '30 мкр', '31 мкр', '32 мкр', '33 мкр', '34 мкр']):
            return 0.85  # -15% cost
        
        # Standard districts (base cost)
        else:
            return 1.0
    
    def get_condition_factor(renovation_status):
        """Renovation condition factor."""
        renovation_lower = str(renovation_status).lower()
        
        if 'без ремонта' in renovation_lower or 'коробка' in renovation_lower:
            return 1.0    # Full renovation required (100% of base cost)
        elif 'с ремонтом' in renovation_lower:
            return 0.10   # Minor touch-ups (10% of base cost)
        elif 'новый ремонт' in renovation_lower:
            return 0.0    # No renovation needed (0% cost)
        else:
            return 0.05   # Unknown condition, assume minor work (5% cost)
    
    # Calculate renovation costs for each property
    df['base_renovation_cost'] = df['rooms'].apply(get_base_renovation_cost)
    df['area_adjustment_factor'] = df['area_m2'].apply(get_area_adjustment)
    df['district_adjustment_factor'] = df['district'].apply(get_district_adjustment)
    df['condition_factor'] = df['renovation'].apply(get_condition_factor)
    
    # Final renovation cost calculation
    df['estimated_renovation_cost'] = (
        df['base_renovation_cost'] * 
        df['area_adjustment_factor'] * 
        df['district_adjustment_factor'] * 
        df['condition_factor']
    ).round(0)
    
    # Add renovation buffer (15% for cost overruns)
    df['renovation_cost_with_buffer'] = (df['estimated_renovation_cost'] * 1.15).round(0)
    
    # Total investment required (purchase + renovation)
    df['total_investment_required'] = df['price'] + df['renovation_cost_with_buffer']
    
    # Log renovation cost statistics
    renovation_needed = df[df['estimated_renovation_cost'] > 1000]  # Properties needing significant renovation
    if len(renovation_needed) > 0:
        logging.info(f"📊 Renovation analysis:")
        logging.info(f"   Properties needing renovation: {len(renovation_needed)}/{len(df)}")
        logging.info(f"   Average renovation cost: {renovation_needed['estimated_renovation_cost'].mean():.0f} TJS")
        logging.info(f"   Renovation cost range: {renovation_needed['estimated_renovation_cost'].min():.0f} - {renovation_needed['estimated_renovation_cost'].max():.0f} TJS")
        logging.info(f"   Average total investment: {renovation_needed['total_investment_required'].mean():.0f} TJS")
    
    return df

def calculate_renovation_risk_assessment(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive risk assessment for renovation investments."""
    logging.info("⚠️ Calculating renovation risk assessment matrix")
    
    def get_renovation_complexity_score(renovation_status, area_m2):
        """Calculate renovation complexity (0-1, higher = more complex)."""
        renovation_lower = str(renovation_status).lower()
        
        if 'без ремонта' in renovation_lower or 'коробка' in renovation_lower:
            # Full renovation needed
            base_complexity = 0.8
            
            # Area complexity adjustment
            if area_m2 > 120:
                area_factor = 1.2  # Large apartments are harder to renovate
            elif area_m2 < 60:
                area_factor = 1.1  # Small apartments have tight spaces
            else:
                area_factor = 1.0
                
            return min(base_complexity * area_factor, 1.0)
            
        elif 'с ремонтом' in renovation_lower:
            return 0.3  # Minor renovation, low complexity
        elif 'новый ремонт' in renovation_lower:
            return 0.1  # Minimal work needed
        else:
            return 0.2  # Unknown condition, assume minor work
    
    def get_financial_risk_score(renovation_cost, price, monthly_rent):
        """Calculate financial risk (0-1, higher = more risky)."""
        renovation_ratio = renovation_cost / price if price > 0 else 0
        
        # High renovation cost relative to purchase price = higher risk
        if renovation_ratio > 0.15:  # >15% of property value
            cost_risk = 0.8
        elif renovation_ratio > 0.10:  # 10-15%
            cost_risk = 0.6
        elif renovation_ratio > 0.05:  # 5-10%
            cost_risk = 0.4
        else:
            cost_risk = 0.2  # <5%
        
        # Cash flow risk: low rent relative to investment
        total_investment = price + renovation_cost
        if monthly_rent > 0:
            months_to_recover_renovation = renovation_cost / (monthly_rent * 0.75)  # 75% net
            if months_to_recover_renovation > 36:  # >3 years to recover renovation
                cashflow_risk = 0.8
            elif months_to_recover_renovation > 24:  # 2-3 years
                cashflow_risk = 0.6
            elif months_to_recover_renovation > 12:  # 1-2 years
                cashflow_risk = 0.4
            else:
                cashflow_risk = 0.2  # <1 year
        else:
            cashflow_risk = 1.0  # No rental income = maximum risk
        
        return (cost_risk + cashflow_risk) / 2
    
    def get_market_risk_score(district, market_segment, price_vs_district_avg):
        """Calculate market risk (0-1, higher = more risky)."""
        district_lower = str(district).lower()
        
        # District stability risk
        if any(stable in district_lower for stable in ['универмаг', 'центр', 'шелкокомбинат']):
            district_risk = 0.2  # Stable premium districts
        elif any(mid in district_lower for mid in ['18 мкр', '19 мкр', '13 мкр', '20 мкр']):
            district_risk = 0.3  # Established districts
        elif any(emerging in district_lower for emerging in ['12 мкр', '28 мкр', '8 мкр']):
            district_risk = 0.4  # Emerging areas
        else:
            district_risk = 0.6  # Less established districts
        
        # Market positioning risk
        if market_segment == 'luxury_market':
            segment_risk = 0.7  # Luxury is more volatile
        elif market_segment == 'premium_market':
            segment_risk = 0.4
        elif market_segment == 'mid_market':
            segment_risk = 0.3
        else:  # budget_market
            segment_risk = 0.5  # Budget can be unstable
        
        # Price positioning risk
        if price_vs_district_avg > 1.5:
            price_risk = 0.8  # Significantly above market
        elif price_vs_district_avg > 1.2:
            price_risk = 0.6
        elif price_vs_district_avg < 0.7:
            price_risk = 0.7  # Suspiciously cheap
        else:
            price_risk = 0.3  # Fair market pricing
        
        return (district_risk + segment_risk + price_risk) / 3
    
    def get_execution_risk_score(renovation_cost, district):
        """Calculate renovation execution risk (0-1, higher = more risky)."""
        district_lower = str(district).lower()
        
        # Access to quality contractors by district
        if any(premium in district_lower for premium in ['универмаг', 'центр', 'шелкокомбинат']):
            contractor_risk = 0.2  # Good access to quality contractors
        elif any(mid in district_lower for mid in ['18 мкр', '19 мкр', '13 мкр']):
            contractor_risk = 0.3
        else:
            contractor_risk = 0.5  # Limited contractor options
        
        # Project size complexity
        if renovation_cost > 50000:
            size_risk = 0.7  # Large projects have more things that can go wrong
        elif renovation_cost > 30000:
            size_risk = 0.5
        elif renovation_cost > 10000:
            size_risk = 0.3
        else:
            size_risk = 0.2  # Minor renovations are easier to manage
        
        return (contractor_risk + size_risk) / 2
    
    # Calculate individual risk scores
    df['renovation_complexity_risk'] = df.apply(
        lambda row: get_renovation_complexity_score(row['renovation'], row['area_m2']), axis=1
    )
    
    df['financial_risk'] = df.apply(
        lambda row: get_financial_risk_score(
            row['renovation_cost_with_buffer'], row['price'], row['estimated_monthly_rent']
        ), axis=1
    )
    
    df['market_risk'] = df.apply(
        lambda row: get_market_risk_score(
            row['district'], row['market_segment'], row['price_vs_district_avg']
        ), axis=1
    )
    
    df['execution_risk'] = df.apply(
        lambda row: get_execution_risk_score(row['renovation_cost_with_buffer'], row['district']), axis=1
    )
    
    # Calculate overall risk score (weighted average)
    risk_weights = {
        'renovation_complexity_risk': 0.25,
        'financial_risk': 0.35,  # Most important
        'market_risk': 0.25,
        'execution_risk': 0.15
    }
    
    df['overall_risk_score'] = (
        df['renovation_complexity_risk'] * risk_weights['renovation_complexity_risk'] +
        df['financial_risk'] * risk_weights['financial_risk'] +
        df['market_risk'] * risk_weights['market_risk'] +
        df['execution_risk'] * risk_weights['execution_risk']
    ).round(2)
    
    # Risk categorization
    def categorize_risk(risk_score):
        if risk_score <= 0.3:
            return 'low_risk'
        elif risk_score <= 0.5:
            return 'moderate_risk'
        elif risk_score <= 0.7:
            return 'high_risk'
        else:
            return 'very_high_risk'
    
    df['risk_category'] = df['overall_risk_score'].apply(categorize_risk)
    
    # Investment recommendation based on risk vs return
    def get_investment_recommendation(row):
        risk = row['overall_risk_score']
        net_yield = row.get('net_rental_yield', 0)  # This will be calculated later
        renovation_needed = row['estimated_renovation_cost'] > 1000
        
        # Pre-calculate expected yield for recommendation
        if renovation_needed:
            if risk <= 0.3 and net_yield >= 6:
                return 'strong_buy'
            elif risk <= 0.4 and net_yield >= 5:
                return 'buy'
            elif risk <= 0.6 and net_yield >= 4:
                return 'consider'
            else:
                return 'avoid'
        else:
            # Properties not needing renovation have different criteria
            if risk <= 0.4 and net_yield >= 5:
                return 'strong_buy'
            elif risk <= 0.5 and net_yield >= 4:
                return 'buy'
            elif risk <= 0.6 and net_yield >= 3:
                return 'consider'
            else:
                return 'avoid'
    
    # Temporary recommendation (will be updated after yield calculation)
    df['preliminary_investment_recommendation'] = df.apply(
        lambda row: get_investment_recommendation(row), axis=1
    )
    
    # Log risk analysis
    risk_dist = df['risk_category'].value_counts()
    logging.info(f"🎯 Risk assessment summary:")
    logging.info(f"   Risk distribution: {dict(risk_dist)}")
    logging.info(f"   Average overall risk: {df['overall_risk_score'].mean():.2f}")
    logging.info(f"   High-risk properties: {len(df[df['overall_risk_score'] > 0.6])}/{len(df)}")
    
    high_risk_props = df[df['overall_risk_score'] > 0.7]
    if len(high_risk_props) > 0:
        logging.info(f"   Very high risk properties: {len(high_risk_props)}")
        logging.info(f"   Avg renovation cost for high-risk: {high_risk_props['renovation_cost_with_buffer'].mean():.0f} TJS")
    
    return df

def calculate_rental_yield_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rental yield and investment features."""
    logging.info("🏠 Calculating rental yield and investment features")
    
    # Extract room count from URL if not available
    def extract_room_count(url):
        if pd.isna(url):
            return 2  # Default
        
        # Look for patterns like '1-komn', '2-komn', '3-komn', etc.
        import re
        match = re.search(r'(\d+)-komn', str(url))
        if match:
            return int(match.group(1))
        
        # Fallback to older patterns for compatibility
        if '1-komnatnyie' in str(url):
            return 1
        elif '2-komnatnyie' in str(url):
            return 2
        elif '3-komnatnyie' in str(url):
            return 3
        elif '4-komnatnyie' in str(url):
            return 4
        elif '5-komnatnyie' in str(url):
            return 5
        else:
            return 2  # Default
    
    # Add rooms feature if not present, prioritize existing room_count
    if 'rooms' not in df.columns:
        if 'room_count' in df.columns:
            # Use room_count from size categorization if available
            df['rooms'] = df['room_count'].fillna(df['url'].apply(extract_room_count))
        else:
            df['rooms'] = df['url'].apply(extract_room_count)
    
    if RENTAL_MODEL is None or not JOBLIB_AVAILABLE:
        logging.warning("Using market-based rental estimates")
        # Market-based estimation for Tajikistan rental market (updated with realistic rates)
        def estimate_monthly_rent(row):
            # Base rates per m² per month by room count (based on market research)
            base_rates = {
                1: 50,  # 1-room: ~50 TJS/m²/month
                2: 45,  # 2-room: ~45 TJS/m²/month  
                3: 40,  # 3-room: ~40 TJS/m²/month
                4: 35,  # 4-room: ~35 TJS/m²/month
                5: 30   # 5+ rooms: ~30 TJS/m²/month
            }
            
            rooms = int(row.get('rooms', 4))
            area = row['area_m2']
            base_rate = base_rates.get(rooms, 35)
            
            # District premium/discount factors
            district = row.get('district', '')
            district_factor = 1.0
            
            # Premium districts
            if any(premium in district.lower() for premium in ['центр', 'универмаг', 'шелкокомбинат']):
                district_factor = 1.2
            # Popular microdistricts
            elif any(mkr in district for mkr in ['18 мкр', '19 мкр', '13 мкр', '16 мкр']):
                district_factor = 1.1
            # Less popular areas
            elif any(less_popular in district for less_popular in ['29 мкр', '30 мкр', '31 мкр', '32 мкр', '33 мкр', '34 мкр']):
                district_factor = 0.9
            
            # Quality adjustments
            renovation = row.get('renovation', '')
            renovation_factor = 1.0
            if 'новый ремонт' in str(renovation).lower():
                renovation_factor = 1.15
            elif 'с ремонтом' in str(renovation).lower():
                renovation_factor = 1.05
            elif 'без ремонта' in str(renovation).lower():
                renovation_factor = 0.85
            
            # Calculate final rent
            monthly_rent = area * base_rate * district_factor * renovation_factor
            
            # Apply realistic bounds based on market observations
            min_rent = {1: 1500, 2: 2000, 3: 2500, 4: 3000, 5: 3500}.get(rooms, 3000)
            max_rent = {1: 4000, 2: 5000, 3: 6000, 4: 7000, 5: 8000}.get(rooms, 7000)
            
            return max(min_rent, min(monthly_rent, max_rent))
        
        df['estimated_monthly_rent'] = df.apply(estimate_monthly_rent, axis=1)
        df['rental_prediction_confidence'] = 0.75  # High confidence for market-based estimates
    else:
        # Use rental prediction model with properly aligned features
        try:
            logging.info("🤖 Preparing features for rental prediction model with proper alignment")
            
            # Load rental market data for feature alignment
            rental_market_data = load_rental_market_data()
            
            if rental_market_data is not None:
                # Normalize district names in rental data using the same mapping
                rental_market_data['district_normalized'] = rental_market_data['district'].apply(
                    lambda x: x.replace('мкр', ' мкр').replace('  ', ' ').strip()
                )
                
                # Clean rental data first
                rental_clean = rental_market_data[
                    (rental_market_data['area_m2'] > 20) &  # Minimum 20m² (realistic apartment size)
                    (rental_market_data['area_m2'] < 200) &  # Maximum 200m² (filter outliers)
                    (rental_market_data['price'] > 500) &   # Minimum 500 TJS rent
                    (rental_market_data['price'] < 20000)   # Maximum 20,000 TJS rent
                ]
                
                # Create rental-based district statistics (from cleaned rental market data)
                rental_district_stats = rental_clean.groupby('district_normalized').agg({
                    'price': ['mean', 'median', 'std', 'count'],
                    'area_m2': ['mean']
                }).round(2)
                
                # Flatten column names
                rental_district_stats.columns = ['_'.join(col).strip() for col in rental_district_stats.columns]
                rental_district_stats = rental_district_stats.rename(columns={
                    'price_mean': 'rental_district_avg_price',
                    'price_median': 'rental_district_median_price', 
                    'price_std': 'rental_district_price_std',
                    'price_count': 'rental_district_listing_count',
                    'area_m2_mean': 'rental_district_avg_area'
                })
                
                # Calculate rental price per m2 for districts
                rental_clean['rental_price_per_m2'] = rental_clean['price'] / rental_clean['area_m2']
                
                # Remove extreme price_per_m2 outliers (likely data entry errors)
                price_per_m2_q99 = rental_clean['rental_price_per_m2'].quantile(0.99)
                price_per_m2_q01 = rental_clean['rental_price_per_m2'].quantile(0.01)
                rental_clean = rental_clean[
                    (rental_clean['rental_price_per_m2'] >= price_per_m2_q01) &
                    (rental_clean['rental_price_per_m2'] <= price_per_m2_q99)
                ]
                
                rental_district_price_per_m2 = rental_clean.groupby('district_normalized')['rental_price_per_m2'].mean()
                
                logging.info(f"   Cleaned rental data: {len(rental_clean)}/{len(rental_market_data)} records used")
                logging.info(f"   Rental price_per_m2 range: {rental_clean['rental_price_per_m2'].min():.1f} - {rental_clean['rental_price_per_m2'].max():.1f} TJS/m²")
                
                logging.info(f"✅ Created rental-based district statistics from {len(rental_clean)}/{len(rental_market_data)} rental records")
            else:
                logging.warning("⚠️ No rental market data available, using fallback estimates")
                rental_district_stats = None
                rental_district_price_per_m2 = None
            
            # Prepare features exactly as expected by the rental model
            model_features = pd.DataFrame()
            
            # Basic features (these should match the rental data scale)
            model_features['rooms'] = df['rooms']
            model_features['area_m2'] = df['area_m2']
            model_features['floor'] = df['floor']
            
            # CRITICAL: Use rental-based price_per_m2, not sales-based
            if rental_district_price_per_m2 is not None:
                # Map sales properties to rental price_per_m2 based on district
                model_features['price_per_m2'] = df['district'].map(rental_district_price_per_m2).fillna(50)  # 50 TJS/m2 fallback
            else:
                # Fallback: estimate rental price_per_m2 from market knowledge
                model_features['price_per_m2'] = df['rooms'].map({1: 55, 2: 50, 3: 45, 4: 40, 5: 35}).fillna(40)
            
            # Derived numeric features
            model_features['area_per_room'] = df['area_m2'] / df['rooms']
            model_features['floor_normalized'] = df['floor'] / df.groupby('district')['floor'].transform('max')
            
            # CRITICAL: Use actual training district statistics, not hardcoded estimates
            if rental_district_stats is not None:
                model_features['district_encoded'] = df['district'].map(rental_district_stats['rental_district_avg_price']).fillna(3000)
                model_features['district_avg_price'] = df['district'].map(rental_district_stats['rental_district_avg_price']).fillna(3000)
                model_features['district_median_price'] = df['district'].map(rental_district_stats['rental_district_median_price']).fillna(3000)
                model_features['district_price_per_m2'] = df['district'].map(rental_district_price_per_m2).fillna(50)
                model_features['district_listing_count'] = df['district'].map(rental_district_stats['rental_district_listing_count']).fillna(10)
                model_features['district_price_std'] = df['district'].map(rental_district_stats['rental_district_price_std']).fillna(500)
                model_features['district_avg_area'] = df['district'].map(rental_district_stats['rental_district_avg_area']).fillna(60)
            else:
                # Use ACTUAL training district statistics instead of hardcoded estimates
                logging.info("🔧 Using actual training district statistics for model features")
                
                def get_district_features(district):
                    """Get district features using actual training statistics."""
                    stats = get_district_statistics(district)
                    return pd.Series({
                        'district_encoded': stats['avg_price'],
                        'district_avg_price': stats['avg_price'],
                        'district_median_price': stats['median_price'],
                        'district_price_per_m2': stats['price_per_m2'],
                        'district_listing_count': stats['listing_count'],
                        'district_price_std': stats['price_std'],
                        'district_avg_area': stats['avg_area']
                    })
                
                # Apply district statistics to each property
                district_features_df = df['district'].apply(get_district_features)
                
                for col in ['district_encoded', 'district_avg_price', 'district_median_price', 
                           'district_price_per_m2', 'district_listing_count', 'district_price_std', 'district_avg_area']:
                    model_features[col] = district_features_df[col]
            
            # Interaction features (using rental-based values)
            model_features['area_price_per_room'] = model_features['price_per_m2'] * model_features['area_per_room']
            model_features['district_price_ratio'] = model_features['district_encoded'] / model_features['district_avg_price']  # Should be ~1.0
            model_features['area_to_district_avg'] = model_features['area_m2'] / model_features['district_avg_area']
            
            # Time-based features
            if 'publication_date' in df.columns:
                pub_date = pd.to_datetime(df['publication_date'])
                month = pub_date.dt.month
                model_features['month_sin'] = np.sin(2 * np.pi * month / 12)
                model_features['month_cos'] = np.cos(2 * np.pi * month / 12)
                model_features['is_weekend'] = pub_date.dt.dayofweek.isin([5, 6]).astype(int)
                
                # Day of week features
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                for day in day_names:
                    model_features[f'day_{day}'] = (pub_date.dt.day_name() == day).astype(int)
                
                # Season features
                season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                             3: 'spring', 4: 'spring', 5: 'spring',
                             6: 'summer', 7: 'summer', 8: 'summer',
                             9: 'fall', 10: 'fall', 11: 'fall'}
                for season in ['winter', 'spring', 'summer', 'fall']:
                    model_features[f'season_{season}'] = month.map(lambda x: season_map.get(x, 'winter') == season).astype(int)
            else:
                # Default time features
                model_features['month_sin'] = 0
                model_features['month_cos'] = 1
                model_features['is_weekend'] = 0
                for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                    model_features[f'day_{day}'] = 0
                for season in ['winter', 'spring', 'summer', 'fall']:
                    model_features[f'season_{season}'] = 0
                model_features['day_Monday'] = 1
                model_features['season_spring'] = 1
            
            # Categorical features (match rental model training)
            renovation_map = {'Без ремонта (коробка)': 0, 'С ремонтом': 1, 'Новый ремонт': 2}
            bathroom_map = {'Совмещенный': 0, 'Раздельный': 1}
            heating_map = {'Нет': 0, 'Есть': 1}
            
            model_features['renovation'] = df['renovation'].map(renovation_map).fillna(1)
            model_features['bathroom'] = df['bathroom'].map(bathroom_map).fillna(0)
            model_features['heating'] = df['heating'].map(heating_map).fillna(0)
            model_features['district'] = df['district']
            
            # Handle missing values and infinity
            model_features = model_features.fillna(0)
            
            # Replace infinity values with reasonable defaults
            for col in model_features.select_dtypes(include=[np.number]).columns:
                inf_mask = np.isinf(model_features[col])
                if inf_mask.any():
                    logging.warning(f"Found {inf_mask.sum()} infinity values in {col}, replacing with median")
                    median_val = model_features[col][~inf_mask].median() if (~inf_mask).any() else 0
                    model_features.loc[inf_mask, col] = median_val
            
            # Ensure correct feature order
            required_features = RENTAL_MODEL_METADATA['feature_names'] if RENTAL_MODEL_METADATA else []
            if required_features:
                missing_features = [f for f in required_features if f not in model_features.columns]
                if missing_features:
                    logging.warning(f"Missing features for rental model: {missing_features}")
                    for feature in missing_features:
                        model_features[feature] = 0
                
                model_features = model_features[required_features]
            
            # Make predictions
            predictions = RENTAL_MODEL.predict(model_features)
            df['estimated_monthly_rent'] = np.maximum(predictions, 1000)  # Minimum 1000 TJS/month
            df['rental_prediction_confidence'] = 0.85
            
            logging.info(f"✅ Generated ML rental predictions with proper feature alignment")
            logging.info(f"   Prediction range: {df['estimated_monthly_rent'].min():.0f} - {df['estimated_monthly_rent'].max():.0f} TJS/month")
            logging.info(f"   Average predicted rent: {df['estimated_monthly_rent'].mean():.0f} TJS/month")
            
            # Apply renovation premium after ML predictions
            df = apply_renovation_premium(df)
            
        except Exception as e:
            logging.warning(f"Rental prediction failed: {e}, using market estimates")
            def estimate_monthly_rent(row):
                # Use the same realistic market-based estimation as above
                base_rates = {
                    1: 50, 2: 45, 3: 40, 4: 35, 5: 30
                }
                
                rooms = int(row.get('rooms', 4))
                area = row['area_m2']
                base_rate = base_rates.get(rooms, 35)
                
                district = row.get('district', '')
                district_factor = 1.0
                
                if any(premium in district.lower() for premium in ['центр', 'универмаг', 'шелкокомбинат']):
                    district_factor = 1.2
                elif any(mkr in district for mkr in ['18 мкр', '19 мкр', '13 мкр', '16 мкр']):
                    district_factor = 1.1
                elif any(less_popular in district for less_popular in ['29 мкр', '30 мкр', '31 мкр', '32 мкр', '33 мкр', '34 мкр']):
                    district_factor = 0.9
                
                renovation = row.get('renovation', '')
                renovation_factor = 1.0
                if 'новый ремонт' in str(renovation).lower():
                    renovation_factor = 1.15
                elif 'с ремонтом' in str(renovation).lower():
                    renovation_factor = 1.05
                elif 'без ремонта' in str(renovation).lower():
                    renovation_factor = 0.85
                
                monthly_rent = area * base_rate * district_factor * renovation_factor
                
                min_rent = {1: 1500, 2: 2000, 3: 2500, 4: 3000, 5: 3500}.get(rooms, 3000)
                max_rent = {1: 4000, 2: 5000, 3: 6000, 4: 7000, 5: 8000}.get(rooms, 7000)
                
                return max(min_rent, min(monthly_rent, max_rent))
            
            df['estimated_monthly_rent'] = df.apply(estimate_monthly_rent, axis=1)
            df['rental_prediction_confidence'] = 0.75
            
            # Apply renovation premium after base rental calculation
            df = apply_renovation_premium(df)
    
    # Calculate renovation costs before investment metrics
    df = calculate_renovation_costs(df)
    
    # Add risk assessment before investment metrics
    df = calculate_renovation_risk_assessment(df)
    
    # Calculate comprehensive investment metrics
    df['annual_rental_income'] = df['estimated_monthly_rent'] * 12
    
    # Calculate yields based on TOTAL INVESTMENT (purchase + renovation)
    df['gross_rental_yield'] = (df['annual_rental_income'] / df['total_investment_required'] * 100).round(2)
    
    # Estimate annual expenses (maintenance, taxes, management - typically 25% of rent in Tajikistan)
    df['estimated_annual_expenses'] = df['annual_rental_income'] * 0.25
    df['net_annual_income'] = df['annual_rental_income'] - df['estimated_annual_expenses']
    
    # ROI based on total investment including renovation costs
    df['net_rental_yield'] = (df['net_annual_income'] / df['total_investment_required'] * 100).round(2)
    df['roi_percentage'] = df['net_rental_yield']  # Same as net rental yield
    
    # Payback period (years to recover TOTAL investment through rental income)
    df['payback_period_years'] = np.where(
        df['net_annual_income'] > 0,
        df['total_investment_required'] / df['net_annual_income'],
        99  # Very high number for properties with no/negative cash flow
    ).round(1)
    
    # Cap unrealistic payback periods
    df['payback_period_years'] = np.clip(df['payback_period_years'], 0, 50)
    
    # Monthly cash flow after expenses
    df['monthly_cash_flow'] = (df['net_annual_income'] / 12).round(0)
    
    # Add renovation-specific metrics
    df['renovation_payback_years'] = np.where(
        df['net_annual_income'] > 0,
        df['renovation_cost_with_buffer'] / df['net_annual_income'],
        99  # Very high number for properties with no/negative cash flow
    ).round(1)
    
    df['renovation_impact_on_yield'] = ((df['annual_rental_income'] / df['price']) - 
                                       (df['annual_rental_income'] / df['total_investment_required'])).round(2)
    
    # Investment efficiency metrics
    df['cost_per_rental_dollar'] = (df['total_investment_required'] / df['annual_rental_income']).round(2)
    df['renovation_percentage_of_price'] = (df['renovation_cost_with_buffer'] / df['price'] * 100).round(1)
    
    # Investment score (0-1, higher is better)
    df['investment_score'] = np.clip(df['net_rental_yield'] / 10, 0, 1.2)
    
    # Risk-adjusted investment score
    df['risk_adjusted_investment_score'] = df['investment_score'] * df['rental_prediction_confidence']
    
    # Investment quality categories
    def categorize_investment(yield_pct):
        if yield_pct >= 10:
            return 'excellent_investment'
        elif yield_pct >= 7:
            return 'good_investment' 
        elif yield_pct >= 5:
            return 'fair_investment'
        elif yield_pct >= 3:
            return 'poor_investment'
        else:
            return 'avoid_investment'
    
    df['investment_category'] = df['net_rental_yield'].apply(categorize_investment)
    
    # Cash flow categories
    def categorize_cash_flow(monthly_flow):
        if monthly_flow >= 500:
            return 'excellent_cash_flow'
        elif monthly_flow >= 300:
            return 'good_cash_flow'
        elif monthly_flow >= 100:
            return 'moderate_cash_flow'
        elif monthly_flow >= 0:
            return 'break_even'
        else:
            return 'negative_cash_flow'
    
    df['cash_flow_category'] = df['monthly_cash_flow'].apply(categorize_cash_flow)
    
    # Calculate renovation costs and premiums
    logging.info("🔨 Calculating renovation costs and investment adjustments")
    df = calculate_renovation_costs(df)
    df = apply_renovation_premium(df)
    df = calculate_renovation_risk_assessment(df)
    
    # Update investment recommendations with final yield data
    df = finalize_investment_recommendations(df)
    
    # Log investment statistics
    logging.info(f"💰 Investment metrics calculated:")
    logging.info(f"  Average gross rental yield: {df['gross_rental_yield'].mean():.2f}%")
    logging.info(f"  Average net rental yield: {df['net_rental_yield'].mean():.2f}%")
    logging.info(f"  Average ROI: {df['roi_percentage'].mean():.2f}%")
    logging.info(f"  Average payback period: {df['payback_period_years'].mean():.1f} years")
    logging.info(f"  Average monthly cash flow: {df['monthly_cash_flow'].mean():.0f} TJS")
    
    return df

def finalize_investment_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Final investment recommendation engine combining all renovation metrics."""
    logging.info("🎯 Finalizing comprehensive investment recommendations")
    
    def get_final_investment_recommendation(row):
        """Comprehensive investment recommendation based on all factors."""
        risk = row['overall_risk_score']
        net_yield = row['net_rental_yield']
        renovation_cost = row['estimated_renovation_cost']
        renovation_ratio = row['renovation_percentage_of_price'] / 100
        payback_years = row['payback_period_years']
        monthly_premium = row['monthly_rent_premium']
        district = row['district'].lower()
        
        # Renovation investment logic
        if renovation_cost > 1000:  # Properties needing significant renovation
            
            # Premium districts with high renovation potential
            if any(premium in district for premium in ['универмаг', 'центр', 'шелкокомбинат', '18 мкр', '19 мкр']):
                if risk <= 0.4 and net_yield >= 6.0 and monthly_premium >= 500:
                    return 'strong_buy_renovation'
                elif risk <= 0.5 and net_yield >= 5.0 and monthly_premium >= 400:
                    return 'buy_renovation'
                elif risk <= 0.6 and net_yield >= 4.0:
                    return 'consider_renovation'
                else:
                    return 'avoid_high_risk'
            
            # Mid-tier districts
            elif any(mid in district for mid in ['13 мкр', '20 мкр', '12 мкр', '8 мкр']):
                if risk <= 0.5 and net_yield >= 5.5 and renovation_ratio <= 0.12:
                    return 'buy_renovation'
                elif risk <= 0.6 and net_yield >= 4.5 and renovation_ratio <= 0.15:
                    return 'consider_renovation'
                else:
                    return 'avoid_moderate_risk'
            
            # Budget districts
            else:
                if risk <= 0.5 and net_yield >= 6.0 and renovation_ratio <= 0.10:
                    return 'consider_budget_renovation'
                elif payback_years <= 15 and monthly_premium >= 300:
                    return 'speculative_buy'
                else:
                    return 'avoid_budget_risk'
        
        else:  # Properties not needing major renovation
            if risk <= 0.3 and net_yield >= 5.0:
                return 'strong_buy_turnkey'
            elif risk <= 0.4 and net_yield >= 4.0:
                return 'buy_turnkey'
            elif risk <= 0.5 and net_yield >= 3.5:
                return 'consider_turnkey'
            else:
                return 'avoid_low_yield'
    
    def get_investment_priority_score(row):
        """Calculate investment priority (0-100, higher = better opportunity)."""
        net_yield = row['net_rental_yield']
        risk = row['overall_risk_score']
        monthly_premium = row['monthly_rent_premium']
        renovation_ratio = row['renovation_percentage_of_price'] / 100
        payback_years = row['payback_period_years']
        
        # Base score from yield (0-40 points)
        yield_score = min(net_yield * 6, 40)
        
        # Risk penalty (0-25 point deduction)
        risk_penalty = risk * 25
        
        # Renovation efficiency bonus (0-20 points)
        if monthly_premium > 0:
            premium_bonus = min(monthly_premium / 50, 20)  # Up to 20 points for 1000+ TJS premium
        else:
            premium_bonus = 0
        
        # Payback efficiency (0-15 points)
        if payback_years <= 10:
            payback_bonus = 15
        elif payback_years <= 15:
            payback_bonus = 10
        elif payback_years <= 20:
            payback_bonus = 5
        else:
            payback_bonus = 0
        
        # Renovation cost efficiency penalty
        if renovation_ratio > 0.15:
            renovation_penalty = 10  # High renovation cost relative to price
        elif renovation_ratio > 0.10:
            renovation_penalty = 5
        else:
            renovation_penalty = 0
        
        total_score = yield_score - risk_penalty + premium_bonus + payback_bonus - renovation_penalty
        return max(0, min(100, total_score))
    
    def categorize_investment_priority(score):
        """Categorize investment priority based on score."""
        if score >= 80:
            return 'top_priority'
        elif score >= 65:
            return 'high_priority'
        elif score >= 50:
            return 'medium_priority'
        elif score >= 35:
            return 'low_priority'
        else:
            return 'not_recommended'
    
    # Calculate final recommendations
    df['final_investment_recommendation'] = df.apply(get_final_investment_recommendation, axis=1)
    df['investment_priority_score'] = df.apply(get_investment_priority_score, axis=1).round(1)
    df['investment_priority_category'] = df['investment_priority_score'].apply(categorize_investment_priority)
    
    # Calculate renovation ROI (for properties needing renovation)
    df['renovation_roi_annual'] = np.where(
        df['estimated_renovation_cost'] > 1000,
        (df['annual_rent_premium'] / df['renovation_cost_with_buffer'] * 100).round(1),
        0
    )
    
    # Investment summary flags
    df['is_premium_district'] = df['district'].str.lower().str.contains('универмаг|центр|шелкокомбинат|18 мкр|19 мкр', na=False)
    df['has_high_renovation_roi'] = df['renovation_roi_annual'] >= 15
    df['is_fast_payback'] = df['payback_period_years'] <= 18
    df['has_significant_premium'] = df['monthly_rent_premium'] >= 400
    
    # Log final recommendations
    rec_dist = df['final_investment_recommendation'].value_counts()
    priority_dist = df['investment_priority_category'].value_counts()
    
    logging.info(f"🎯 Final investment recommendations:")
    logging.info(f"   Recommendation distribution: {dict(rec_dist)}")
    logging.info(f"   Priority distribution: {dict(priority_dist)}")
    
    top_opportunities = df[df['investment_priority_score'] >= 70]
    if len(top_opportunities) > 0:
        logging.info(f"   Top opportunities: {len(top_opportunities)} properties")
        logging.info(f"   Avg yield for top opportunities: {top_opportunities['net_rental_yield'].mean():.1f}%")
        logging.info(f"   Avg risk for top opportunities: {top_opportunities['overall_risk_score'].mean():.2f}")
    
    strong_buys = df[df['final_investment_recommendation'].str.contains('strong_buy', na=False)]
    if len(strong_buys) > 0:
        logging.info(f"   Strong buy recommendations: {len(strong_buys)} properties")
        logging.info(f"   Avg renovation cost for strong buys: {strong_buys['estimated_renovation_cost'].mean():.0f} TJS")
    
    return df

def calculate_enhanced_bargain_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate enhanced bargain score based on price, quality, and market factors (NO rental criteria)."""
    logging.info("🎯 Calculating enhanced bargain score based on price and quality factors only")
    
    # Enhanced components (all normalized to 0-1) - REMOVED rental/investment criteria
    components = {
        'price_advantage': 1 - np.clip(df['price_per_m2_vs_district_avg'], 0, 2) / 2,
        'quality_features': (df['renovation_score'] / 3 + df['heating_score'] + 
                           df['bathroom_score'] / 2 + df['floor_preference_score']) / 4,
        'market_position': 1 - df['price_per_m2_market_percentile'],
        'size_preference': df['size_preference_score'],  # NEW: Size preference factor
        'risk_factors': (df['tech_passport_score'] + 
                        (1 - np.clip(abs(df['area_m2'] - df['area_m2_mean']) / df['area_m2_mean'], 0, 1))) / 2
    }
    
    # Enhanced weights (NO rental/investment weighting)
    weights = {
        'price_advantage': 0.40,        # 40% - Most important: price vs market
        'quality_features': 0.30,       # 30% - Property condition & features
        'market_position': 0.20,        # 20% - Market positioning  
        'size_preference': 0.05,        # 5% - Size desirability
        'risk_factors': 0.05           # 5% - Documentation & size fit
    }
    
    # Calculate enhanced weighted bargain score (NO rental dependency)
    enhanced_bargain_score = sum(components[comp] * weights[comp] for comp in components.keys())
    df['enhanced_bargain_score'] = np.clip(enhanced_bargain_score, 0, 1)
    
    # Enhanced bargain categories (STRICT criteria, NO rental dependency)
    def categorize_enhanced_bargain(row):
        score = row['enhanced_bargain_score']
        price_advantage = components['price_advantage'].iloc[row.name] if hasattr(components['price_advantage'], 'iloc') else (1 - np.clip(row['price_per_m2_vs_district_avg'], 0, 2) / 2)
        quality_score = components['quality_features'].iloc[row.name] if hasattr(components['quality_features'], 'iloc') else ((row['renovation_score'] / 3 + row['heating_score'] + row['bathroom_score'] / 2 + row['floor_preference_score']) / 4)
        
        # Pure bargain categorization based on price and quality only
        if score >= 0.80 and price_advantage >= 0.7 and quality_score >= 0.6:
            return 'exceptional_opportunity'
        elif score >= 0.70 and price_advantage >= 0.6:
            return 'excellent_bargain'
        elif score >= 0.60 and (price_advantage >= 0.5 or quality_score >= 0.6):
            return 'good_bargain'
        elif score >= 0.50:
            return 'fair_value'
        elif score >= 0.40:
            return 'market_price'
        else:
            return 'overpriced'
    
    df['enhanced_bargain_category'] = df.apply(categorize_enhanced_bargain, axis=1)
    
    # Keep original bargain score for compatibility
    df['bargain_score'] = df['enhanced_bargain_score']
    df['bargain_category'] = df['enhanced_bargain_category']
    
    # Log enhanced bargain distribution
    enhanced_bargain_dist = df['enhanced_bargain_category'].value_counts()
    logging.info(f"Enhanced bargain distribution (NO rental criteria): {dict(enhanced_bargain_dist)}")
    
    return df


def main(input_csv_path: str, output_dir: str = ".", log_level: str = "INFO") -> None:
    """Main feature engineering pipeline for bargain finder system."""
    # Setup logging
    setup_logging(log_level)
    
    logging.info("=== STARTING FEATURE ENGINEERING FOR BARGAIN FINDER ===")
    logging.info(f"Input file: {input_csv_path}")
    logging.info(f"Output directory: {output_dir}")
    
    try:
        # Load cleaned data
        df = load_cleaned_data(input_csv_path)
        
        # Feature engineering pipeline
        df = calculate_price_per_sqm(df)
        df = calculate_district_market_indicators(df)
        df = calculate_price_ratios(df)
        df = encode_categorical_features(df)
        df = calculate_property_age_proxy(df)
        df = calculate_size_category(df)
        df = calculate_floor_features(df)
        df = add_time_based_features(df)
        df = calculate_market_position_features(df)
        df = create_district_clusters(df)
                
        # NEW: Load rental model and calculate investment features
        load_rental_prediction_model()
        df = calculate_rental_yield_features(df)
        df = calculate_enhanced_bargain_score(df)  # Enhanced version with investment potential
          # This should be last as it uses other features
        
        # Generate feature summary
        generate_feature_summary(df)
        
        # Save engineered dataset
        csv_path, parquet_path = save_engineered_features(df, output_dir)
        
        logging.info("=== FEATURE ENGINEERING COMPLETED SUCCESSFULLY ===")
        logging.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        logging.info(f"Output files: {csv_path}" + (f", {parquet_path}" if parquet_path else ""))
        
        # Show new feature columns
        original_columns = ['url', 'price', 'ad_number', 'publication_date', 'area_m2', 'floor', 
                          'build_type', 'renovation', 'bathroom', 'district', 'heating', 
                          'built_status', 'tech_passport', 'photo_count']
        new_features = [col for col in df.columns if col not in original_columns]
        logging.info(f"New engineered features ({len(new_features)}): {new_features}")
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Engineering for Bargain Finder Real Estate System")
    parser.add_argument("input_csv", help="Path to cleaned listings CSV file")
    parser.add_argument("-o", "--output", default=".", help="Output directory (default: current directory)")
    parser.add_argument("-l", "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    main(args.input_csv, args.output, args.log_level)
