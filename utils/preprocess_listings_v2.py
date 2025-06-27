#!/usr/bin/env python3
"""
Real Estate Listings Data Preprocessing Script - Version 2

This script cleans and normalizes real estate listings data extracted from somon.tj,
with modifications to remove unwanted columns and normalize district names.

Author: Data Preprocessing Assistant
Date: June 2025
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import re
import os
import yaml
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration for the preprocessing script."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing.log'),
            logging.StreamHandler()
        ]
    )


def load_data(file_path: str) -> pd.DataFrame:
    """Load raw scraped data with robust error handling for malformed CSV rows."""
    try:
        # First try standard pandas reading with bad line skipping
        df = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"✅ Successfully loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        print(f"⚠️ Standard CSV reading failed: {e}")
        
        # Fallback: manual CSV cleaning
        try:
            import csv
            import io
            
            print("🔧 Attempting manual CSV repair...")
            
            # Read and clean the CSV manually
            cleaned_rows = []
            expected_columns = 15
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Ensure header has correct number of columns
                if len(header) != expected_columns:
                    print(f"⚠️ Header has {len(header)} columns, expected {expected_columns}")
                    # Truncate or pad header if needed
                    header = header[:expected_columns] + [''] * max(0, expected_columns - len(header))
                
                cleaned_rows.append(header)
                
                for line_num, row in enumerate(reader, 2):
                    if len(row) == expected_columns:
                        cleaned_rows.append(row)
                    elif len(row) > expected_columns:
                        # Truncate extra columns (likely caused by commas in data)
                        cleaned_rows.append(row[:expected_columns])
                    else:
                        # Pad missing columns
                        padded_row = row + [''] * (expected_columns - len(row))
                        cleaned_rows.append(padded_row)
            
            # Create DataFrame from cleaned data
            df = pd.DataFrame(cleaned_rows[1:], columns=cleaned_rows[0])
            print(f"✅ Manual repair successful: {len(df)} records recovered")
            return df
            
        except Exception as manual_error:
            print(f"❌ Manual CSV repair failed: {manual_error}")
            raise Exception(f"Could not load CSV file: {file_path}. Original error: {e}")


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the price column."""
    initial_count = len(df)
    
    # Convert to numeric, coercing errors to NaN
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Remove negative prices
    df.loc[df['price'] < 0, 'price'] = np.nan
    
    # Log outliers and missing values
    missing_count = df['price'].isna().sum()
    if missing_count > 0:
        logging.warning(f"Found {missing_count} missing/invalid prices")
    
    logging.info(f"Price column cleaned. Valid prices: {df['price'].notna().sum()}/{initial_count}")
    return df


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize publication dates to ISO 8601 format (YYYY-MM-DD)."""
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    def parse_russian_date(date_str: str) -> Optional[str]:
        """Parse Russian date strings and convert to ISO format."""
        if pd.isna(date_str) or not isinstance(date_str, str):
            return None
            
        date_str = date_str.strip()
        
        # Handle "Сегодня" (Today)
        if "Сегодня" in date_str:
            return today.isoformat()
        
        # Handle "Вчера" (Yesterday)
        if "Вчера" in date_str:
            return yesterday.isoformat()
        
        # Handle DD.MM.YYYY format
        date_pattern = r'(\d{1,2})\.(\d{1,2})\.(\d{4})'
        match = re.search(date_pattern, date_str)
        if match:
            day, month, year = match.groups()
            try:
                parsed_date = datetime(int(year), int(month), int(day)).date()
                return parsed_date.isoformat()
            except ValueError:
                logging.warning(f"Invalid date format: {date_str}")
                return None
        
        logging.warning(f"Unrecognized date format: {date_str}")
        return None
    
    initial_count = len(df)
    df['publication_date'] = df['publication_date'].apply(parse_russian_date)
    
    missing_count = df['publication_date'].isna().sum()
    valid_count = df['publication_date'].notna().sum()
    
    logging.info(f"Date normalization completed. Valid dates: {valid_count}/{initial_count}")
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize numeric columns: area_m2, floor, photo_count."""
    initial_count = len(df)
    
    # Clean area_m2
    df['area_m2'] = pd.to_numeric(df['area_m2'], errors='coerce')
    df.loc[df['area_m2'] <= 0, 'area_m2'] = np.nan
    
    # Clean floor
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
    df.loc[df['floor'] <= 0, 'floor'] = np.nan
    
    # Clean photo_count and num_images (if they exist)
    if 'photo_count' in df.columns:
        df['photo_count'] = pd.to_numeric(df['photo_count'], errors='coerce')
        df.loc[df['photo_count'] < 0, 'photo_count'] = 0
    
    if 'num_images' in df.columns:
        df['num_images'] = pd.to_numeric(df['num_images'], errors='coerce')
        df.loc[df['num_images'] < 0, 'num_images'] = 0
    
    area_missing = df['area_m2'].isna().sum()
    floor_missing = df['floor'].isna().sum()
    
    logging.info(f"Numeric columns cleaned:")
    logging.info(f"  - area_m2: {df['area_m2'].notna().sum()}/{initial_count} valid, {area_missing} missing")
    logging.info(f"  - floor: {df['floor'].notna().sum()}/{initial_count} valid, {floor_missing} missing")
    
    return df


def normalize_image_urls(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize image URLs to ensure consistent format."""
    if 'image_urls' not in df.columns:
        logging.info("No image_urls column found, skipping image URL normalization")
        return df
    
    initial_count = len(df)
    
    def clean_image_urls(url_string):
        """Clean and validate image URL strings."""
        if pd.isna(url_string) or url_string == '' or str(url_string).strip() == '':
            return None
        
        url_str = str(url_string).strip()
        
        # Split by semicolon and clean each URL
        urls = [url.strip() for url in url_str.split(';') if url.strip()]
        
        # Filter valid URLs (basic validation)
        valid_urls = []
        for url in urls:
            if url.startswith('http') and 'somon.tj' in url:
                valid_urls.append(url)
        
        # Return semicolon-separated string or None
        return ';'.join(valid_urls) if valid_urls else None
    
    df['image_urls'] = df['image_urls'].apply(clean_image_urls)
    
    # Count valid image URLs
    valid_image_count = df['image_urls'].notna().sum()
    logging.info(f"Image URL normalization completed: {valid_image_count}/{initial_count} properties have valid image URLs")
    
    return df


def normalize_city_information(df: pd.DataFrame) -> pd.DataFrame:
    """Keep city information as-is from the scraped data."""
    initial_count = len(df)
    
    if 'city' in df.columns:
        city_distribution = df['city'].value_counts()
        logging.info(f"City information preserved from scraped data:")
        for city, count in city_distribution.items():
            logging.info(f"  - {city}: {count} properties")
    else:
        logging.info("No city column found in data")
    
    return df


def load_district_mapping():
    """Load district mapping from YAML configuration file."""
    try:
        # Try to find the district mapping file in multiple locations
        current_dir = Path(__file__).parent
        possible_paths = [
            current_dir.parent / "rental_prediction" / "configs" / "district_mapping.yaml",
            Path(__file__).parent.parent.parent / "rental_prediction" / "configs" / "district_mapping.yaml",
            current_dir / "district_mapping.yaml"
        ]
        
        for yaml_path in possible_paths:
            if yaml_path.exists():
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    mapping = yaml.safe_load(f)
                logging.info(f"Loaded district mapping from: {yaml_path}")
                return mapping
        
        logging.warning("District mapping YAML file not found, using fallback patterns")
        return None
    except Exception as e:
        logging.warning(f"Error loading district mapping: {e}, using fallback patterns")
        return None


def normalize_district_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced district normalization using YAML-based mapping patterns.
    Integrates the comprehensive district mapping from rental prediction system.
    """
    logging.info("Starting enhanced district normalization with YAML mapping")
    
    # Load district mapping configuration
    district_mapping = load_district_mapping()
    
    # Clean district names first
    df['district'] = df['district'].astype(str).str.strip()
    df['district'] = df['district'].str.replace(r'\s+', ' ', regex=True)
    
    original_count = len(df['district'].unique())
    logging.info(f"Original district count: {original_count}")
    
    def standardize_district_with_yaml(district_name):
        """Apply YAML-based standardization to district names."""
        if pd.isna(district_name) or district_name == 'nan':
            return 'Неизвестно'
        
        district = str(district_name).strip().lower()
        
        if district_mapping:
            # 1. Microdistricts pattern matching
            if 'microdistricts' in district_mapping:
                micro_config = district_mapping['microdistricts']
                patterns = micro_config.get('patterns', [])
                valid_range = micro_config.get('valid_range', [1, 50])
                output_format = micro_config.get('output_format', "{number} мкр")
                
                for pattern in patterns:
                    match = re.search(pattern, district, re.IGNORECASE)
                    if match:
                        try:
                            number = int(match.group(1))
                            if valid_range[0] <= number <= valid_range[1]:
                                return output_format.format(number=number)
                        except (ValueError, IndexError):
                            continue
            
            # 2. Landmarks mapping
            if 'landmarks' in district_mapping:
                landmarks = district_mapping['landmarks']
                
                # Silk factory
                if 'silk_factory' in landmarks:
                    silk_config = landmarks['silk_factory']
                    for pattern in silk_config.get('patterns', []):
                        if re.search(pattern, district, re.IGNORECASE):
                            return silk_config.get('normalized', 'Шелкокомбинат')
                
                # Shopping areas
                if 'shopping' in landmarks:
                    for shop_key, shop_config in landmarks['shopping'].items():
                        if isinstance(shop_config, dict) and 'patterns' in shop_config:
                            for pattern in shop_config['patterns']:
                                if re.search(pattern, district, re.IGNORECASE):
                                    return shop_config.get('normalized', shop_key.title())
            
            # 3. Residential areas
            if 'residential' in district_mapping:
                residential = district_mapping['residential']
                for res_key, res_config in residential.items():
                    if isinstance(res_config, dict) and 'patterns' in res_config:
                        for pattern in res_config['patterns']:
                            if re.search(pattern, district, re.IGNORECASE):
                                return res_config.get('normalized', res_key.title())
            
            # 4. Districts
            if 'districts' in district_mapping:
                districts = district_mapping['districts']
                for dist_key, dist_config in districts.items():
                    if isinstance(dist_config, dict) and 'patterns' in dist_config:
                        for pattern in dist_config['patterns']:
                            if re.search(pattern, district, re.IGNORECASE):
                                return dist_config.get('normalized', dist_key.title())
                
                # Handle other patterns
                if 'other' in districts and 'normalized_patterns' in districts['other']:
                    for item in districts['other']['normalized_patterns']:
                        pattern = item.get('pattern', '')
                        normalized = item.get('normalized', '')
                        if pattern and re.search(pattern, district, re.IGNORECASE):
                            return normalized
            
            # 5. Facilities
            if 'facilities' in district_mapping:
                facilities = district_mapping['facilities']
                for facility_type, facility_data in facilities.items():
                    for facility_key, facility_config in facility_data.items():
                        if isinstance(facility_config, dict) and 'patterns' in facility_config:
                            for pattern in facility_config['patterns']:
                                if re.search(pattern, district, re.IGNORECASE):
                                    return facility_config.get('normalized', facility_key.replace('_', ' ').title())
        
        # Fallback patterns if YAML mapping is not available or no match found
        return standardize_district_fallback(district_name)
    
    def standardize_district_fallback(district_name):
        """Fallback standardization patterns if YAML mapping fails."""
        district = str(district_name).strip()
        
        # Basic microdistrict pattern
        number_pattern = r'(\d{1,2})'
        match = re.search(number_pattern, district)
        if match:
            number = int(match.group(1))
            if 7 <= number <= 50:
                return f"{number} мкр"
        
        # Basic landmark patterns
        basic_mappings = {
            r'.*ш[её]лк.*': 'Шелкокомбинат',
            r'.*универмаг.*': 'Универмаг',
            r'.*панчшанбе.*': 'Панчшанбе',
            r'.*гулбахор.*': 'Гулбахор',
            r'.*центр.*': 'Центр',
            r'.*ватан.*': 'Ватан',
            r'.*кооператор.*': 'Кооператор',
            r'.*бахор.*': 'кв. Бахор',
        }
        
        for pattern, standard_name in basic_mappings.items():
            if re.search(pattern, district, re.IGNORECASE):
                return standard_name
        
        return district
    
    # Apply standardization
    df['district'] = df['district'].apply(standardize_district_with_yaml)
    
    new_count = len(df['district'].unique())
    reduction = original_count - new_count
    
    logging.info(f"Enhanced district normalization completed: {original_count} -> {new_count} unique districts")
    logging.info(f"Reduced district count by: {reduction} ({reduction/original_count*100:.1f}%)")
    
    # Log the final district distribution
    district_counts = df['district'].value_counts()
    logging.info(f"Top 10 districts after normalization:")
    for district, count in district_counts.head(10).items():
        logging.info(f"  {district}: {count}")
    
    return df


def remove_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are not needed for final analysis."""
    columns_to_remove = [
        # 'image_urls',  # Keep this column for frontend display
        'photo_count_mismatch',
        'has_images',
        'has_tech_passport',
        'is_renovated',
        'has_heating',
        'bathroom_separate',
        'is_new_construction',
        'is_secondary_market',
        'image_urls_list'  # Also remove this if it exists
    ]
    
    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    if existing_columns_to_remove:
        df = df.drop(columns=existing_columns_to_remove)
        logging.info(f"Removed columns: {existing_columns_to_remove}")
    else:
        logging.info("No unwanted columns found to remove")
        
    return df


def smart_floor_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligent floor imputation using district-based median with fallbacks.
    This preserves all properties while maintaining realistic floor values for price comparison.
    """
    if 'floor' not in df.columns:
        logging.warning("Floor column not found, skipping floor imputation")
        return df
    
    initial_missing = df['floor'].isna().sum()
    if initial_missing == 0:
        logging.info("No missing floor values found")
        return df
    
    logging.info(f"Starting smart floor imputation for {initial_missing} missing values")
    
    # Step 1: Fill with district median floor
    logging.info("Step 1: Filling missing floors with district median")
    df['floor'] = df.groupby('district')['floor'].transform(lambda x: x.fillna(x.median()))
    
    # Step 2: If district has no floor data, use overall median
    remaining_missing = df['floor'].isna().sum()
    if remaining_missing > 0:
        overall_median = df['floor'].median()
        logging.info(f"Step 2: Filling {remaining_missing} remaining missing floors with overall median: {overall_median}")
        df['floor'].fillna(overall_median, inplace=True)
    
    # Step 3: Ensure floors are positive integers
    df['floor'] = df['floor'].round().astype(int)
    df.loc[df['floor'] <= 0, 'floor'] = 1  # Ensure minimum floor is 1
    
    final_missing = df['floor'].isna().sum()
    filled_count = initial_missing - final_missing
    
    logging.info(f"Smart floor imputation completed:")
    logging.info(f"  - Filled {filled_count} missing floor values")
    logging.info(f"  - Remaining missing: {final_missing}")
    logging.info(f"  - Floor range after imputation: {df['floor'].min()} - {df['floor'].max()}")
    
    # Log district-wise floor statistics
    district_floor_stats = df.groupby('district')['floor'].agg(['count', 'median', 'min', 'max']).round(1)
    logging.info("Floor statistics by district (top 10):")
    for district, stats in district_floor_stats.head(10).iterrows():
        logging.info(f"  {district}: count={stats['count']}, median={stats['median']}, range={stats['min']}-{stats['max']}")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using smart imputation instead of dropping rows."""
    initial_count = len(df)
    
    # Essential columns that require valid data (excluding floor, which we'll impute)
    essential_columns = ['price', 'area_m2', 'district']
    
    # Log missing values for all important columns
    important_columns = essential_columns + ['floor']
    for col in important_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logging.warning(f"Column '{col}' has {missing_count} missing values")
    
    # Apply smart floor imputation BEFORE dropping any rows
    df = smart_floor_imputation(df)
    
    # Now drop rows only if they're missing truly essential data (not floor)
    df_clean = df.dropna(subset=[col for col in essential_columns if col in df.columns])
    dropped_count = initial_count - len(df_clean)
    
    if dropped_count > 0:
        logging.warning(f"Dropped {dropped_count} rows due to missing essential data (price, area, district)")
    else:
        logging.info("No rows dropped - all essential data present after imputation")
    
    # Fill missing non-essential categorical fields
    categorical_fills = {
        'heating': 'Неизвестно',
        'build_type': 'Неизвестно',
        'renovation': 'Неизвестно',
        'bathroom': 'Неизвестно',
        'built_status': 'Неизвестно',
        'tech_passport': 'Неизвестно'
    }
    
    for col, fill_value in categorical_fills.items():
        if col in df_clean.columns:
            missing_before = df_clean[col].isna().sum()
            if missing_before > 0:
                df_clean[col] = df_clean[col].fillna(fill_value)
                logging.info(f"Filled {missing_before} missing values in '{col}' with '{fill_value}'")
    
    logging.info(f"Missing value handling completed. Retained {len(df_clean)}/{initial_count} rows")
    
    return df_clean


def detect_area_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and remove anomalies in the area_m2 column.
    Uses statistical methods to identify unrealistic area values.
    """
    logging.info("Starting area_m2 anomaly detection")
    initial_count = len(df)
    
    if 'area_m2' not in df.columns:
        logging.warning("area_m2 column not found, skipping anomaly detection")
        return df
    
    # Remove rows with missing area data
    valid_area_mask = df['area_m2'].notna()
    df_with_area = df[valid_area_mask]
    
    if len(df_with_area) == 0:
        logging.warning("No valid area data found")
        return df
    
    # Log initial area statistics
    area_stats = df_with_area['area_m2'].describe()
    logging.info(f"Initial area statistics: min={area_stats['min']:.1f}, max={area_stats['max']:.1f}, "
                f"mean={area_stats['mean']:.1f}, std={area_stats['std']:.1f}")
    
    # Define realistic area bounds for residential properties
    # Based on typical apartment sizes in Tajikistan
    MIN_REALISTIC_AREA = 15.0  # Minimum realistic apartment size in m²
    MAX_REALISTIC_AREA = 200.0  # Maximum realistic apartment size in m²
    
    # Apply hard limits first
    area_too_small = df_with_area['area_m2'] < MIN_REALISTIC_AREA
    area_too_large = df_with_area['area_m2'] > MAX_REALISTIC_AREA
    
    anomalies_small = df_with_area[area_too_small]
    anomalies_large = df_with_area[area_too_large]
    
    # Log anomalies found
    if len(anomalies_small) > 0:
        logging.warning(f"Found {len(anomalies_small)} properties with unrealistically small area (< {MIN_REALISTIC_AREA} m²):")
        for _, row in anomalies_small.iterrows():
            logging.warning(f"  - Ad {row['ad_number']}: {row['area_m2']} m² in {row.get('district', 'Unknown')} "
                          f"for {row.get('price', 'Unknown')} TJS")
    
    if len(anomalies_large) > 0:
        logging.warning(f"Found {len(anomalies_large)} properties with unrealistically large area (> {MAX_REALISTIC_AREA} m²):")
        for _, row in anomalies_large.iterrows():
            logging.warning(f"  - Ad {row['ad_number']}: {row['area_m2']} m² in {row.get('district', 'Unknown')} "
                          f"for {row.get('price', 'Unknown')} TJS")
    
    # Remove anomalies
    valid_area_range = (df_with_area['area_m2'] >= MIN_REALISTIC_AREA) & (df_with_area['area_m2'] <= MAX_REALISTIC_AREA)
    df_clean = df_with_area[valid_area_range]
    
    # Also include rows that had missing area data (if we want to keep them)
    df_missing_area = df[~valid_area_mask]
    if len(df_missing_area) > 0:
        logging.info(f"Keeping {len(df_missing_area)} rows with missing area data")
        df_clean = pd.concat([df_clean, df_missing_area], ignore_index=True)
    
    removed_count = initial_count - len(df_clean)
    
    if removed_count > 0:
        logging.info(f"Removed {removed_count} area anomalies ({removed_count/initial_count*100:.1f}% of data)")
        
        # Log final area statistics
        if len(df_clean[df_clean['area_m2'].notna()]) > 0:
            final_stats = df_clean['area_m2'].describe()
            logging.info(f"Final area statistics: min={final_stats['min']:.1f}, max={final_stats['max']:.1f}, "
                        f"mean={final_stats['mean']:.1f}, std={final_stats['std']:.1f}")
    else:
        logging.info("No area anomalies detected")
    
    return df_clean


def validate_data_quality(df: pd.DataFrame) -> None:
    """Perform data quality checks and log potential issues."""
    logging.info("=== DATA QUALITY VALIDATION ===")
    
    # Check for duplicates
    if 'ad_number' in df.columns:
        duplicates = df.duplicated(subset=['ad_number']).sum()
        if duplicates > 0:
            logging.warning(f"Found {duplicates} duplicate ad_numbers")
    
    # Check data ranges
    if len(df) > 0:
        price_stats = df['price'].describe()
        area_stats = df['area_m2'].describe()
        
        logging.info(f"Price range: {price_stats['min']:,.0f} - {price_stats['max']:,.0f} TJS")
        logging.info(f"Area range: {area_stats['min']:.1f} - {area_stats['max']:.1f} m²")
        logging.info(f"Floor range: {df['floor'].min()} - {df['floor'].max()}")


def save_cleaned_data(df: pd.DataFrame, output_dir: str, base_filename: str = "cleaned_listings_v2") -> Tuple[str, Optional[str]]:
    """Save the cleaned dataset in both CSV and Parquet formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    parquet_path = os.path.join(output_dir, f"{base_filename}.parquet")
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved cleaned data to CSV: {csv_path}")
    
    # Save Parquet (more efficient for large datasets)
    try:
        df.to_parquet(parquet_path, index=False)
        logging.info(f"Saved cleaned data to Parquet: {parquet_path}")
        return csv_path, parquet_path
    except ImportError:
        logging.warning("PyArrow not available. Skipping Parquet export.")
        return csv_path, None


def main(input_csv_path: str, output_dir: str = ".", log_level: str = "INFO") -> None:
    """Main preprocessing pipeline for real estate listings data."""
    # Setup logging
    setup_logging(log_level)
    
    logging.info("=== STARTING REAL ESTATE DATA PREPROCESSING V2 ===")
    logging.info(f"Input file: {input_csv_path}")
    logging.info(f"Output directory: {output_dir}")
    
    try:
        # Load data
        df = load_data(input_csv_path)
        
        # Data cleaning pipeline
        df = clean_price_column(df)
        df = normalize_dates(df)
        df = clean_numeric_columns(df)
        df = normalize_image_urls(df)
        df = normalize_district_column(df)
        df = normalize_city_information(df)
        df = handle_missing_values(df)
        df = remove_unwanted_columns(df)
        df = detect_area_anomalies(df)  # Detect and remove area anomalies
        
        # Validate data quality
        validate_data_quality(df)
        
        # Save cleaned data
        csv_path, parquet_path = save_cleaned_data(df, output_dir)
        
        logging.info("=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
        logging.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        logging.info(f"Output files: {csv_path}" + (f", {parquet_path}" if parquet_path else ""))
        
        # Show final column list
        logging.info(f"Final columns: {list(df.columns)}")
        
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess real estate listings data - Version 2")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("-o", "--output", default=".", help="Output directory (default: current directory)")
    parser.add_argument("-l", "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    main(args.input_csv, args.output, args.log_level)
