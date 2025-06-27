import axios from 'axios';
import API_BASE_URL from '../config';

// Use the centralized config for API base URL

// Create axios instance with default configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // Increased to 30 seconds for general requests
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging and authentication
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    
    // Add authentication token if available
    const token = localStorage.getItem('real_estate_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('Response error:', error);
    if (error.response?.status === 503) {
      // Service unavailable
      console.error('Backend service is unavailable');
    }
    return Promise.reject(error);
  }
);

// Types
export interface Property {
  id?: number;
  title?: string;
  url?: string;
  price: number;
  price_per_sqm?: number;
  rooms?: number;
  area?: number;
  floor?: number;
  total_floors?: number;
  city?: string;
  district?: string;
  address?: string;
  build_state?: string;
  property_type?: string;
  renovation?: string;
  image_urls?: string[];
  predicted_price?: number;
  price_difference?: number;
  price_difference_percentage?: number;
  investment_score?: number;
  bargain_category?: string;
  // Investment metrics from rental prediction model
  estimated_monthly_rent?: number;
  annual_rental_income?: number;
  gross_rental_yield?: number;
  net_rental_yield?: number;
  roi_percentage?: number;
  payback_period_years?: number;
  monthly_cash_flow?: number;
  investment_category?: string;
  cash_flow_category?: string;
  rental_prediction_confidence?: number;
  // NEW: Renovation cost analysis
  estimated_renovation_cost?: number;
  renovation_cost_with_buffer?: number;
  total_investment_required?: number;
  renovation_percentage_of_price?: number;
  // NEW: Rental premium for renovations
  monthly_rent_premium?: number;
  annual_rent_premium?: number;
  renovation_premium_multiplier?: number;
  renovation_roi_annual?: number;
  // NEW: Risk assessment
  overall_risk_score?: number;
  risk_category?: string;
  renovation_complexity_risk?: number;
  financial_risk?: number;
  market_risk?: number;
  execution_risk?: number;
  // NEW: Final recommendations
  final_investment_recommendation?: string;
  investment_priority_score?: number;
  investment_priority_category?: string;
  // NEW: Investment flags
  is_premium_district?: boolean;
  has_high_renovation_roi?: boolean;
  is_fast_payback?: boolean;
  has_significant_premium?: boolean;
}

export interface ChartData {
  labels: string[];
  values: number[];
  counts?: number[];
}

export interface ActivityData {
  x: string[];
  y: number[];
}

export interface PropertyFilters {
  min_price?: number;
  max_price?: number;
  min_area?: number;
  max_area?: number;
  min_rooms?: number;
  max_rooms?: number;
  cities?: string;
  districts?: string;
  build_states?: string;
  renovations?: string;
  min_floor?: number;
  max_floor?: number;
  bargain_categories?: string;
  limit?: number;
  offset?: number;
}

export interface PredictionRequest {
  rooms?: number;
  area_m2: number;
  floor?: number;
  district?: string;
  renovation?: string;
  bathroom?: string;
  heating?: string;
}

export interface PredictionResponse {
  predicted_rental: number;
  confidence_interval_lower: number;
  confidence_interval_upper: number;
  annual_rental_income: number;
  gross_rental_yield: number;
  features_used: Record<string, any>;
  model_info?: {
    model_type: string;
    accuracy: number;
    mape: number;
  };
}

export interface MarketStats {
  total_listings: number;
  avg_price: number;
  median_price: number;
  min_price: number;
  max_price: number;
  avg_price_per_sqm?: number;
  total_bargains: number;
  excellent_bargains: number;
  good_bargains: number;
  price_distribution: Record<string, number>;
  room_distribution: Record<string, number>;
  city_distribution: Record<string, number>;
}

export interface FilterRanges {
  price_min: number;
  price_max: number;
  area_min: number;
  area_max: number;
  floor_min: number;
  floor_max: number;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  database_connected: boolean;
  total_properties: number;
  avg_price?: number;
  investment_opportunities?: number;
  model_accuracy?: number;
  last_updated: string;
}

export interface DistrictInvestmentScores {
  labels: string[];
  values: number[];
  counts: number[];
}

// API Service class
class ApiService {
  // Health check
  async getHealthStatus(): Promise<HealthStatus> {
    const response = await api.get('/');
    return response.data;
  }

  // Property search
  async searchProperties(filters: PropertyFilters = {}): Promise<Property[]> {
    const params = new URLSearchParams();
    
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        params.append(key, value.toString());
      }
    });

    const response = await api.get(`/properties?${params.toString()}`);
    return response.data;
  }

  // Get bargain properties
  async getBargainProperties(category: string = 'all', limit: number = 20): Promise<Property[]> {
    const response = await api.get('/bargains', {
      params: { category, limit }
    });
    return response.data;
  }

  // Get market statistics
  async getMarketStats(city?: string): Promise<MarketStats> {
    const params = city ? { city } : {};
    const response = await api.get('/market-stats', { params });
    return response.data;
  }

  // Predict rental price
  async predictPrice(features: PredictionRequest): Promise<PredictionResponse> {
    const response = await api.post('/properties/predict', features);
    return response.data;
  }

  // Get available districts
  async getDistricts(city?: string): Promise<string[]> {
    const params = city ? { city } : {};
    const response = await api.get('/districts', { params });
    return response.data.districts || [];
  }

  // Get available cities
  async getCities(): Promise<string[]> {
    const response = await api.get('/cities');
    return response.data.cities || [];
  }

  // Get model information
  async getModelInfo(): Promise<any> {
    const response = await api.get('/model-info');
    return response.data;
  }

  // Get dynamic filter ranges
  async getFilterRanges(): Promise<FilterRanges> {
    const response = await api.get('/filter-ranges');
    return response.data;
  }

  // Get district investment scores
  async getDistrictInvestmentScores(): Promise<DistrictInvestmentScores> {
    const response = await api.get('/district-investment-scores');
    return response.data;
  }

  async getRenovationImpactData(): Promise<ChartData> {
    const response = await api.get('/chart-data/renovation-impact');
    return response.data;
  }

  async getBuildTypeData(): Promise<ChartData> {
    const response = await api.get('/chart-data/build-type-analysis');
    return response.data;
  }

  async getMarketSegmentsData(): Promise<ChartData> {
    const response = await api.get('/chart-data/market-segments');
    return response.data;
  }

  async getSizeAnalysisData(): Promise<ChartData> {
    const response = await api.get('/chart-data/size-analysis');
    return response.data;
  }

  async getBargainDistributionData(): Promise<ChartData> {
    const response = await api.get('/chart-data/bargain-distribution');
    return response.data;
  }

  async getActivityByDayData(): Promise<ActivityData> {
    const response = await api.get('/chart-data/activity-by-day');
    return response.data;
  }

  // Data Collection and Pipeline Methods
  async runScraping(params: {
    rooms: string;
    city: string;
    build_state: string;
    property_type?: string;
  }): Promise<any> {
    const response = await api.post('/data/scrape', params, {
      timeout: 300000 // 5 minutes timeout for scraping operations
    });
    return response.data;
  }

  async runPreprocessing(): Promise<any> {
    const response = await api.post('/data/preprocess', {}, {
      timeout: 120000 // 2 minutes timeout for preprocessing
    });
    return response.data;
  }

  async runFeatureEngineering(): Promise<any> {
    const response = await api.post('/data/feature-engineering', {}, {
      timeout: 120000 // 2 minutes timeout for feature engineering
    });
    return response.data;
  }

  async importToDatabase(): Promise<any> {
    const response = await api.post('/data/import-to-database', {}, {
      timeout: 120000 // 2 minutes timeout for database import
    });
    return response.data;
  }

  async runFullPipeline(params: {
    rooms: string;
    city: string;
    build_state: string;
    property_type?: string;
  }): Promise<any> {
    const response = await api.post('/data/run-full-pipeline', params, {
      timeout: 600000 // 10 minutes timeout for full pipeline
    });
    return response.data;
  }

  async testPipeline(): Promise<any> {
    const response = await api.post('/data/test-pipeline');
    return response.data;
  }

  async getPipelineStatus(): Promise<any> {
    const response = await api.get('/data/pipeline-status');
    return response.data;
  }

  // Investment Analytics Endpoints
  async getRoiDistribution(): Promise<ChartData> {
    const response = await api.get('/analytics/roi-distribution');
    return response.data;
  }

  async getPaybackAnalysis(): Promise<ChartData> {
    const response = await api.get('/analytics/payback-analysis');
    return response.data;
  }

  async getRentalYieldByDistrict(): Promise<ChartData & { avg_rent: number[], avg_price: number[] }> {
    const response = await api.get('/analytics/rental-yield-by-district');
    return response.data;
  }

  async getInvestmentVsPrice(): Promise<{
    price: number[];
    roi: number[];
    yield: number[];
    payback: number[];
    district: string[];
    bargain_category: string[];
    investment_category: string[];
  }> {
    const response = await api.get('/analytics/investment-vs-price');
    return response.data;
  }

  async getCashFlowDistribution(): Promise<ChartData> {
    const response = await api.get('/analytics/cash-flow-distribution');
    return response.data;
  }

}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;
