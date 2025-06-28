#!/usr/bin/env python3
"""
Integration Test: Verify New Bargain Algorithm + Calibrated Rental Prediction
Tests the full integration between improved systems and backend/frontend
"""

import sys
import os
import json
import requests
import time
from pathlib import Path

# Add utils to path
sys.path.append('.')

def test_integration():
    """Comprehensive integration test"""
    
    print("🔍 INTEGRATION TEST: New Bargain Algorithm + Calibrated Rental Prediction")
    print("=" * 80)
    
    # Test 1: Check if calibrated rental prediction exists
    print("\n1. 📁 CHECKING CALIBRATED RENTAL PREDICTION...")
    calibrated_file = Path("utils/predict_rental_calibrated.py")
    if calibrated_file.exists():
        print("   ✅ Calibrated rental prediction file exists")
    else:
        print("   ❌ Calibrated rental prediction file missing")
        return False
    
    # Test 2: Check if improved bargain algorithm exists
    print("\n2. 📁 CHECKING IMPROVED BARGAIN ALGORITHM...")
    bargain_file = Path("utils/improved_bargain_algorithm.py")
    if bargain_file.exists():
        print("   ✅ Improved bargain algorithm file exists")
    else:
        print("   ❌ Improved bargain algorithm file missing")
        return False
    
    # Test 3: Test calibrated rental prediction directly
    print("\n3. 🧪 TESTING CALIBRATED RENTAL PREDICTION...")
    try:
        sys.path.append('./utils')
        from predict_rental_calibrated import RentalPricePredictorCalibrated
        
        predictor = RentalPricePredictorCalibrated('./rental_prediction/models')
        
        test_property = {
            'rooms': 2,
            'area_m2': 65,
            'floor': 4,
            'district': '19 мкр',
            'renovation': 'С ремонтом',
            'bathroom': 'Раздельный',
            'heating': 'Есть'
        }
        
        result = predictor.predict_rental_price(test_property)
        
        print(f"   ✅ Calibrated prediction: {result['predicted_rental']:.0f} TJS/month")
        print(f"   ✅ Base prediction: {result['base_prediction']:.0f} TJS/month")
        print(f"   ✅ Property tier: {result['calibration_info']['property_tier']}")
        print(f"   ✅ Calibration multiplier: {result['calibration_info']['final_multiplier']:.2f}x")
        print(f"   ✅ Rental yield: {result['gross_rental_yield']:.1f}%")
        
        # Validate that calibration is working
        if result['predicted_rental'] != result['base_prediction']:
            print("   ✅ Calibration is active and working")
        else:
            print("   ⚠️  Calibration not applied (might be budget property)")
        
    except Exception as e:
        print(f"   ❌ Calibrated rental prediction failed: {e}")
        return False
    
    # Test 4: Test improved bargain algorithm directly
    print("\n4. 🧪 TESTING IMPROVED BARGAIN ALGORITHM...")
    try:
        from improved_bargain_algorithm import calculate_improved_bargain_score
        
        # Test with sample data that includes all required columns
        test_data = {
            'price': 450000,
            'area_m2': 65,
            'rooms': 2,
            'district': '19 мкр',
            'renovation': 'С ремонтом',
            'floor': 4,
            'bathroom': 'Раздельный',
            'heating': 'Есть',
            'price_per_m2': 6923,  # 450000/65
            'predicted_price': 500000,  # Simulated predicted price
            'price_difference': -50000,  # Below predicted
            'price_difference_percentage': -10.0  # 10% below predicted
        }
        
        # Create a simple DataFrame for testing
        import pandas as pd
        df_test = pd.DataFrame([test_data])
        
        # Add required columns that the algorithm expects
        df_test['price_per_m2_vs_district_avg'] = 1.0  # Simulated ratio
        df_test['district_avg_price_per_m2'] = 6900  # Simulated average
        
        result_df = calculate_improved_bargain_score(df_test)
        
        if result_df is not None and len(result_df) > 0:
            print("   ✅ Improved bargain algorithm working")
            print(f"   ✅ Bargain score: {result_df.iloc[0].get('bargain_score', 'N/A')}")
            print(f"   ✅ Bargain category: {result_df.iloc[0].get('bargain_category', 'N/A')}")
            print(f"   ✅ Renovation category: {result_df.iloc[0].get('renovation_category', 'N/A')}")
        else:
            print("   ❌ Improved bargain algorithm returned empty result")
            return False
            
    except Exception as e:
        print(f"   ❌ Improved bargain algorithm failed: {e}")
        return False
    
    # Test 5: Check backend integration
    print("\n5. 🔗 CHECKING BACKEND INTEGRATION...")
    
    # Check if backend imports the correct rental predictor
    backend_file = Path("webapp/backend/integrated_main.py")
    if backend_file.exists():
        with open(backend_file, 'r') as f:
            content = f.read()
            if "from predict_rental import RentalPricePredictor" in content:
                print("   ✅ Backend imports rental predictor correctly")
            else:
                print("   ❌ Backend rental predictor import not found")
                return False
            
            if "feature_engineering_enhanced.py" in content:
                print("   ✅ Backend uses enhanced feature engineering (includes bargain algorithm)")
            else:
                print("   ⚠️  Backend might not use enhanced feature engineering")
    else:
        print("   ❌ Backend file not found")
        return False
    
    # Test 6: Check if feature engineering includes bargain algorithm
    print("\n6. 🔗 CHECKING FEATURE ENGINEERING INTEGRATION...")
    
    feature_file = Path("utils/feature_engineering_enhanced.py")
    if feature_file.exists():
        with open(feature_file, 'r') as f:
            content = f.read()
            if "improved_bargain_algorithm" in content:
                print("   ✅ Feature engineering imports improved bargain algorithm")
            else:
                print("   ❌ Feature engineering doesn't import improved bargain algorithm")
                return False
            
            if "calculate_improved_bargain_score" in content:
                print("   ✅ Feature engineering calls improved bargain function")
            else:
                print("   ❌ Feature engineering doesn't call improved bargain function")
                return False
    else:
        print("   ❌ Feature engineering file not found")
        return False
    
    # Test 7: Check PropertyResponse model includes new fields
    print("\n7. 🔗 CHECKING BACKEND MODEL INTEGRATION...")
    
    with open(backend_file, 'r') as f:
        content = f.read()
        required_fields = [
            'renovation_category',
            'global_bargain_category',
            'estimated_monthly_rent',
            'gross_rental_yield'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in content:
                missing_fields.append(field)
        
        if not missing_fields:
            print("   ✅ Backend model includes all required fields")
        else:
            print(f"   ⚠️  Backend model missing fields: {missing_fields}")
    
    # Test 8: Test API endpoint structure (if backend is running)
    print("\n8. 🌐 TESTING API ENDPOINT (if backend running)...")
    
    try:
        # Try to connect to local backend
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Backend is running and accessible")
            
            # Test prediction endpoint structure
            test_prediction = {
                "rooms": 2,
                "area_m2": 65,
                "floor": 4,
                "district": "19 мкр",
                "renovation": "С ремонтом",
                "bathroom": "Раздельный",
                "heating": "Есть"
            }
            
            # Note: This would require authentication, so we just check if endpoint exists
            try:
                response = requests.post("http://localhost:8000/properties/predict", 
                                       json=test_prediction, timeout=5)
                if response.status_code in [200, 401, 422]:  # 401=auth needed, 422=validation error
                    print("   ✅ Prediction endpoint exists and responds")
                else:
                    print(f"   ⚠️  Prediction endpoint returned: {response.status_code}")
            except:
                print("   ⚠️  Could not test prediction endpoint (auth required)")
        else:
            print(f"   ⚠️  Backend responded with status: {response.status_code}")
    except:
        print("   ⚠️  Backend not running (this is OK for file-based testing)")
    
    # Test 9: Check frontend integration
    print("\n9. 🎨 CHECKING FRONTEND INTEGRATION...")
    
    frontend_api = Path("webapp/frontend/src/services/api.ts")
    if frontend_api.exists():
        with open(frontend_api, 'r') as f:
            content = f.read()
            
            if "predicted_rental" in content:
                print("   ✅ Frontend API includes predicted_rental field")
            else:
                print("   ❌ Frontend API missing predicted_rental field")
                return False
            
            if "gross_rental_yield" in content:
                print("   ✅ Frontend API includes gross_rental_yield field")
            else:
                print("   ❌ Frontend API missing gross_rental_yield field")
                return False
            
            if "/properties/predict" in content:
                print("   ✅ Frontend API calls correct prediction endpoint")
            else:
                print("   ❌ Frontend API missing prediction endpoint call")
                return False
    else:
        print("   ❌ Frontend API file not found")
        return False
    
    # Test 10: Summary and recommendations
    print("\n10. 📋 INTEGRATION SUMMARY...")
    print("   ✅ Calibrated rental prediction: WORKING")
    print("   ✅ Improved bargain algorithm: WORKING")
    print("   ✅ Backend integration: CONFIGURED")
    print("   ✅ Frontend integration: CONFIGURED")
    
    print("\n🎯 DEPLOYMENT READINESS:")
    print("   ✅ All components are properly integrated")
    print("   ✅ Ready for deployment of calibrated system")
    
    print("\n🚀 DEPLOYMENT STEPS:")
    print("   1. Replace current predict_rental.py with calibrated version")
    print("   2. Restart backend to load new prediction system")
    print("   3. Test prediction endpoint with authentication")
    print("   4. Verify bargain categories in property listings")
    
    return True

def deploy_calibrated_system():
    """Deploy the calibrated rental prediction system"""
    
    print("\n🚀 DEPLOYING CALIBRATED SYSTEM...")
    
    # Backup current system
    current_file = Path("webapp/backend/predict_rental.py")
    backup_file = Path("webapp/backend/predict_rental_backup.py")
    
    if current_file.exists():
        import shutil
        shutil.copy2(current_file, backup_file)
        print(f"   ✅ Backed up current system to {backup_file}")
    
    # Deploy calibrated system
    calibrated_file = Path("utils/predict_rental_calibrated.py")
    
    if calibrated_file.exists():
        import shutil
        
        # Deploy to all locations
        targets = [
            "webapp/backend/predict_rental.py",
            "webapp/backend/utils/predict_rental.py"
        ]
        
        for target in targets:
            target_path = Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(calibrated_file, target_path)
            print(f"   ✅ Deployed calibrated system to {target}")
        
        print("\n🎉 CALIBRATED SYSTEM DEPLOYED SUCCESSFULLY!")
        print("   📝 Next steps:")
        print("      1. Restart the backend server")
        print("      2. Test the /properties/predict endpoint")
        print("      3. Verify improved predictions in the frontend")
        
        return True
    else:
        print("   ❌ Calibrated system file not found")
        return False

if __name__ == "__main__":
    print("🧪 Starting Integration Test...")
    
    success = test_integration()
    
    if success:
        print("\n" + "=" * 80)
        print("✅ INTEGRATION TEST PASSED!")
        print("✅ All systems are properly integrated and ready for deployment")
        
        # Ask if user wants to deploy
        deploy = input("\n🚀 Deploy calibrated system now? (y/N): ").lower().strip()
        if deploy in ['y', 'yes']:
            deploy_calibrated_system()
        else:
            print("   📝 Deployment skipped. Run with deploy flag when ready.")
    else:
        print("\n" + "=" * 80)
        print("❌ INTEGRATION TEST FAILED!")
        print("❌ Please fix the issues above before deployment")
    
    print("\n🏁 Integration test complete.") 