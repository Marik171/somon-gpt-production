#!/usr/bin/env python3
"""
Simplified Rental Price Prediction for SomonGPT
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RentalPricePredictor:
    """Simplified rental price prediction system"""
    
    def __init__(self, rental_model_dir: Optional[str] = None):
        if rental_model_dir is None:
            current_dir = Path(__file__).parent.parent.parent
            self.model_dir = current_dir / "rental_prediction" / "models"
        else:
            self.model_dir = Path(rental_model_dir)
        
        self.model = None
        self.district_stats = self._load_district_statistics()
        self.load_model()
    
    def _load_district_statistics(self) -> Dict[str, Dict[str, float]]:
        """Load actual district statistics from training data."""
        try:
            current_dir = Path(__file__).parent.parent.parent
            training_data_path = current_dir / "rental_prediction" / "data" / "features" / "engineered_features.csv"
            
            if not training_data_path.exists():
                logger.warning(f"Training data not found at: {training_data_path}, using fallback statistics")
                return {}
            
            import pandas as pd
            training_data = pd.read_csv(training_data_path)
            logger.info(f"Loaded training data for district statistics: {len(training_data)} records")
            
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
            
            logger.info(f"Loaded statistics for {len(district_stats)} districts")
            return district_stats
            
        except Exception as e:
            logger.error(f"Error loading district statistics: {e}")
            return {}

    def _get_district_statistics(self, district: str) -> Dict[str, float]:
        """Get district statistics for a given district with fallback."""
        if district in self.district_stats:
            return self.district_stats[district]
        
        # Fallback: use average across all districts
        if self.district_stats:
            all_districts = list(self.district_stats.values())
            fallback_stats = {
                'avg_price': np.mean([d['avg_price'] for d in all_districts]),
                'median_price': np.mean([d['median_price'] for d in all_districts]),
                'price_per_m2': np.mean([d['price_per_m2'] for d in all_districts]),
                'listing_count': int(np.mean([d['listing_count'] for d in all_districts])),
                'price_std': np.mean([d['price_std'] for d in all_districts]),
                'avg_area': np.mean([d['avg_area'] for d in all_districts])
            }
            
            logger.warning(f"District '{district}' not found, using fallback statistics")
            return fallback_stats
        
        # Last resort: use the old hardcoded values (but warn)
        logger.warning(f"No district statistics available, using hardcoded fallback for '{district}'")
        return {
            'avg_price': 1800,
            'median_price': 1600,
            'price_per_m2': 25,
            'listing_count': 45,
            'price_std': 400,
            'avg_area': 68
        }

    def load_model(self):
        """Load the XGBoost rental model"""
        try:
            model_path = self.model_dir / "xgboost_model.joblib"
            logger.info(f"Loading rental model from: {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Rental model not found: {model_path}")
            
            self.model = joblib.load(model_path)
            logger.info("Rental model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading rental model: {e}")
            raise
    
    def validate_input(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic input validation and cleaning"""
        validated = {}
        
        # Required fields with defaults
        validated['rooms'] = int(property_data.get('rooms', 3))
        validated['area_m2'] = float(property_data.get('area_m2', 75))
        validated['floor'] = int(property_data.get('floor', 3))
        
        # String fields
        validated['district'] = str(property_data.get('district', 'Худжанд'))  # Default to Khujand since model is trained on Khujand data
        validated['renovation'] = str(property_data.get('renovation', 'С ремонтом'))
        validated['bathroom'] = str(property_data.get('bathroom', 'Раздельный'))
        validated['heating'] = str(property_data.get('heating', 'Есть'))
        
        return validated
    
    def prepare_features(self, validated_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare all features that the model expects"""
        # Basic features
        rooms = validated_data['rooms']
        area_m2 = validated_data['area_m2']
        floor = validated_data['floor']
        district = validated_data['district']
        renovation = validated_data['renovation']
        bathroom = validated_data['bathroom']
        heating = validated_data['heating']
        
        # Derived features
        area_per_room = area_m2 / max(rooms, 1)
        floor_normalized = min(floor / 10.0, 1.0)
        
        # District encoding (hash-based like in original)
        district_encoded = hash(district) % 100
        
        # GET CORRECT DISTRICT STATISTICS from training data
        district_info = self._get_district_statistics(district)
        district_avg_price = district_info['avg_price']
        district_median_price = district_info['median_price']
        district_price_per_m2 = district_info['price_per_m2']
        district_listing_count = district_info['listing_count']
        district_price_std = district_info['price_std']
        district_avg_area = district_info['avg_area']
        
        # Estimate rental price_per_m2 based on property characteristics
        # This is a rough estimation based on market knowledge
        base_rental_per_m2 = district_price_per_m2
        
        # Identify luxury districts
        luxury_districts = ['универмаг', 'центр', 'downtown', 'center']
        is_luxury_district = any(keyword in district.lower() for keyword in luxury_districts)
        
        # CORRECT FIX: Apply luxury district base premium FIRST
        if is_luxury_district:
            base_rental_per_m2 *= 1.25  # 25% base premium for luxury districts
        
        # Apply renovation adjustments (same for all districts)
        if renovation == 'Новый ремонт':
            base_rental_per_m2 *= 1.3   # 30% premium for new renovation
        elif renovation == 'С ремонтом':
            base_rental_per_m2 *= 1.1   # 10% premium for good renovation
        # No adjustment for 'Без ремонта (коробка)'
        
        # Adjust based on floor (middle floors are preferred)
        if 3 <= floor <= 7:
            base_rental_per_m2 *= 1.05  # 5% premium for good floors
        elif floor == 1:
            base_rental_per_m2 *= 0.95  # 5% discount for ground floor
        
        # Adjust based on bathroom
        if bathroom == 'Раздельный':
            base_rental_per_m2 *= 1.02  # Small premium for separate bathroom
        
        # Adjust based on heating
        if heating == 'Есть':
            base_rental_per_m2 *= 1.05  # Premium for heating
        
        price_per_m2 = base_rental_per_m2
        
        # District-based calculations 
        area_price_per_room = price_per_m2 * area_per_room
        
        # CORRECTED: Use estimated property price with premiums for ratio calculation
        # Training data shows high-price properties have ratio > 1.0, luxury districts need ratio 1.2-1.6
        estimated_property_price = price_per_m2 * area_m2  # With renovation and other premiums
        district_price_ratio = estimated_property_price / district_avg_price
        
        # Apply luxury district calibration to match training patterns
        if is_luxury_district:
            # Luxury districts need higher ratios to match training patterns for high-price properties
            if district_price_ratio < 1.4:
                district_price_ratio = min(district_price_ratio * 1.4, 1.7)  # Boost to high-price range
        
        area_to_district_avg = area_m2 / district_avg_area
        
        # Time features
        now = datetime.now()
        month_sin = np.sin(2 * np.pi * now.month / 12)
        month_cos = np.cos(2 * np.pi * now.month / 12)
        is_weekend = 1 if now.weekday() >= 5 else 0
        
        # Day of week features
        day_features = {}
        for i, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']):
            day_features[f'day_{day}'] = 1 if now.weekday() == i else 0
        
        # Season features
        season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                     3: 'spring', 4: 'spring', 5: 'spring',
                     6: 'summer', 7: 'summer', 8: 'summer',
                     9: 'fall', 10: 'fall', 11: 'fall'}
        current_season = season_map[now.month]
        
        season_features = {}
        for season in ['winter', 'spring', 'summer', 'fall']:
            season_features[f'season_{season}'] = 1 if current_season == season else 0
        
        # Create the complete feature dictionary in the exact order expected by the model
        data = {
            'rooms': rooms,
            'area_m2': area_m2,
            'floor': floor,
            'price_per_m2': price_per_m2,
            'area_per_room': area_per_room,
            'floor_normalized': floor_normalized,
            'district_encoded': district_encoded,
            'district_avg_price': district_avg_price,
            'district_median_price': district_median_price,
            'district_price_per_m2': district_price_per_m2,
            'district_listing_count': district_listing_count,
            'district_price_std': district_price_std,
            'district_avg_area': district_avg_area,
            'area_price_per_room': area_price_per_room,
            'district_price_ratio': district_price_ratio,
            'area_to_district_avg': area_to_district_avg,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'is_weekend': is_weekend,
            'renovation': renovation,
            'bathroom': bathroom,
            'heating': heating,
            'district': district,
            **day_features,
            **season_features
        }
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Ensure the order matches exactly what the model expects
        expected_features = [
            'rooms', 'area_m2', 'floor', 'price_per_m2', 'area_per_room', 'floor_normalized',
            'district_encoded', 'district_avg_price', 'district_median_price', 'district_price_per_m2',
            'district_listing_count', 'district_price_std', 'district_avg_area', 'area_price_per_room',
            'district_price_ratio', 'area_to_district_avg', 'month_sin', 'month_cos', 'is_weekend',
            'renovation', 'bathroom', 'heating', 'district',
            'day_Monday', 'day_Tuesday', 'day_Wednesday', 'day_Thursday', 'day_Friday', 'day_Saturday', 'day_Sunday',
            'season_winter', 'season_spring', 'season_summer', 'season_fall'
        ]
        
        return df[expected_features]
    
    def predict_rental_price(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make rental price prediction"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Validate input
            validated_data = self.validate_input(property_data)
            logger.info(f"Validated input data: {validated_data}")
            
            # Prepare features
            features_df = self.prepare_features(validated_data)
            logger.info(f"Feature columns: {list(features_df.columns)}")
            logger.info(f"Feature values: {features_df.iloc[0].to_dict()}")
            
            # Make prediction
            predicted_rental = self.model.predict(features_df)[0]
            logger.info(f"Raw model prediction: {predicted_rental}")
            
            # Calculate derived metrics
            annual_rental_income = predicted_rental * 12
            
            # Calculate rental yield (estimate property purchase price for Khujand market)
            # Average property price in Khujand is around 1000-1500 USD per m² = 1200-1800 TJS per m²
            estimated_property_price_per_m2 = 1400  # TJS per m² (purchase price, not rental)
            total_property_value = validated_data['area_m2'] * estimated_property_price_per_m2
            gross_rental_yield = (annual_rental_income / total_property_value) * 100 if total_property_value > 0 else 0
            
            # Simple confidence interval (±15%)
            confidence_lower = predicted_rental * 0.85
            confidence_upper = predicted_rental * 1.15
            
            return {
                'predicted_rental': float(predicted_rental),
                'confidence_interval': {
                    'lower': float(confidence_lower),
                    'upper': float(confidence_upper)
                },
                'annual_rental_income': float(annual_rental_income),
                'gross_rental_yield': float(gross_rental_yield)
            }
            
        except Exception as e:
            logger.error(f"Error during rental prediction: {e}")
            raise

if __name__ == "__main__":
    # Test the predictor
    predictor = RentalPricePredictor()
    
    test_property = {
        'rooms': 3,
        'area_m2': 75,
        'floor': 5,
        'district': 'Душанбе',
        'renovation': 'С ремонтом',
        'bathroom': 'Раздельный',
        'heating': 'Есть',
        'price_per_m2': 1500
    }
    
    result = predictor.predict_rental_price(test_property)
    print(f"Predicted rental: {result['predicted_rental']:.2f} TJS/month")
    print(f"Annual income: {result['annual_rental_income']:.2f} TJS")
    print(f"Rental yield: {result['gross_rental_yield']:.2f}%")
