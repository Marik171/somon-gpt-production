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
    """Calculate composite bargain score to identify undervalued properties."""
    logging.info("Calculating composite bargain score")
    
    # Components of bargain score (all should be between 0-1)
    components = {
        'price_advantage': 1 - np.clip(df['price_per_m2_vs_district_avg'], 0, 2) / 2,  # Lower price vs district = higher score
        'quality_features': (df['renovation_score'] / 3 + df['heating_score'] + 
                           df['bathroom_score'] / 2 + df['floor_preference_score']) / 4,  # Updated without build_type_score
        'market_position': 1 - df['price_per_m2_market_percentile'],  # Lower market position = higher score
        'tech_passport_bonus': df['tech_passport_score'] * 0.1,  # Small bonus for having tech passport
        'size_appropriateness': 1 - abs(df['area_m2'] - df['area_m2_mean']) / df['area_m2_mean']  # Close to district average size
    }
    
    # Weights for different components
    weights = {
        'price_advantage': 0.4,      # 40% - most important
        'quality_features': 0.3,     # 30% - property quality
        'market_position': 0.2,      # 20% - market positioning
        'tech_passport_bonus': 0.05, # 5% - documentation
        'size_appropriateness': 0.05 # 5% - size fit
    }
    
    # Calculate weighted bargain score
    bargain_score = sum(components[comp] * weights[comp] for comp in components.keys())
    df['bargain_score'] = np.clip(bargain_score, 0, 1)  # Ensure 0-1 range
    
    # Create bargain categories
    def categorize_bargain(score):
        if score >= 0.75:
            return 'excellent_bargain'
        elif score >= 0.65:
            return 'good_bargain'
        elif score >= 0.55:
            return 'fair_value'
        elif score >= 0.45:
            return 'market_price'
        else:
            return 'overpriced'
    
    df['bargain_category'] = df['bargain_score'].apply(categorize_bargain)
    
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
        df = calculate_bargain_score(df)  # This should be last as it uses other features
        
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
