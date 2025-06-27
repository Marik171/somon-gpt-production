import React, { useState } from 'react';
import { propertyService, PredictionRequest, PredictionResponse } from '../../services/property';
import { useAuth } from '../../services/auth';

const PricePredictionForm: React.FC = () => {
  const [formData, setFormData] = useState<PredictionRequest>({
    area_m2: 75,
    floor: 3,
    district: '18 мкр',
    build_type: 'Новостройка',
    renovation: 'Без ремонта (коробка)',
    bathroom: 'Раздельный',
    heating: 'Нет',
    tech_passport: 'Неизвестно',
    photo_count: 5
  });
  
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { isAuthenticated } = useAuth();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'area_m2' || name === 'floor' || name === 'photo_count' 
        ? parseFloat(value) || 0 
        : value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!isAuthenticated) {
      setError('Please login to use price prediction');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const result = await propertyService.predictPrice(formData);
      setPrediction(result);
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0
    }).format(price);
  };

  if (!isAuthenticated) {
    return (
      <div className="max-w-2xl mx-auto p-6 text-center">
        <h1 className="text-3xl font-bold text-gray-800 mb-4">Price Prediction</h1>
        <p className="text-gray-600">Please login to access the price prediction tool.</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-4 sm:p-6">
      <h1 className="text-2xl sm:text-3xl font-bold text-gray-800 mb-4 sm:mb-6">Property Price Prediction</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-8">
        {/* Form */}
        <div className="bg-white p-4 sm:p-6 rounded-lg shadow">
          <h2 className="text-lg sm:text-xl font-semibold mb-4">Property Details</h2>
          
          <form onSubmit={handleSubmit} className="space-y-3 sm:space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Area (m²) *
              </label>
              <input
                type="number"
                name="area_m2"
                value={formData.area_m2}
                onChange={handleChange}
                min="20"
                max="500"
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Floor
              </label>
              <input
                type="number"
                name="floor"
                value={formData.floor || ''}
                onChange={handleChange}
                min="1"
                max="50"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                District
              </label>
              <select
                name="district"
                value={formData.district || ''}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="18 мкр">18 мкр</option>
                <option value="центр">центр</option>
                <option value="Давлатчо">Давлатчо</option>
                <option value="Истиклол">Истиклол</option>
                <option value="Сино">Сино</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Building Type
              </label>
              <select
                name="build_type"
                value={formData.build_type || ''}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="Новостройка">Новостройка</option>
                <option value="Вторичный рынок">Вторичный рынок</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Renovation
              </label>
              <select
                name="renovation"
                value={formData.renovation || ''}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="Без ремонта (коробка)">Без ремонта (коробка)</option>
                <option value="С ремонтом">С ремонтом</option>
                <option value="Новый ремонт">Новый ремонт</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Bathroom
              </label>
              <select
                name="bathroom"
                value={formData.bathroom || ''}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="Раздельный">Раздельный</option>
                <option value="Совмещенный">Совмещенный</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Heating
              </label>
              <select
                name="heating"
                value={formData.heating || ''}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="Нет">Нет</option>
                <option value="Есть">Есть</option>
                <option value="Неизвестно">Неизвестно</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Technical Passport
              </label>
              <select
                name="tech_passport"
                value={formData.tech_passport || ''}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="Неизвестно">Неизвестно</option>
                <option value="Есть">Есть</option>
                <option value="Нет">Нет</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Number of Photos
              </label>
              <input
                type="number"
                name="photo_count"
                value={formData.photo_count || ''}
                onChange={handleChange}
                min="0"
                max="50"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 text-white py-3 sm:py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-base font-medium min-h-[44px]"
            >
              {loading ? 'Predicting...' : 'Predict Price'}
            </button>
          </form>

          {error && (
            <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
              {error}
            </div>
          )}
        </div>

        {/* Results */}
        <div className="bg-white p-4 sm:p-6 rounded-lg shadow">
          <h2 className="text-lg sm:text-xl font-semibold mb-4">Prediction Results</h2>
          
          {prediction ? (
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="text-lg font-bold text-blue-800 mb-2">
                  Predicted Price: {formatPrice(prediction.predicted_price)}
                </h3>
                <p className="text-blue-600">
                  Price per m²: {formatPrice(prediction.price_per_sqm)}
                </p>
                <p className="text-sm text-blue-600 mt-2">
                  Confidence Range:
                  <br />
                  {formatPrice(prediction.confidence_interval.lower)} - {formatPrice(prediction.confidence_interval.upper)}
                </p>
              </div>

              {prediction.is_bargain && (
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-green-800 font-medium">
                    🎉 This could be a bargain! 
                    {prediction.bargain_category && ` (${prediction.bargain_category})`}
                  </p>
                </div>
              )}

              <div className="text-xs text-gray-500">
                Model Version: {prediction.model_version}
              </div>
            </div>
          ) : (
            <p className="text-gray-500">Fill out the form and click "Predict Price" to see results.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default PricePredictionForm;
