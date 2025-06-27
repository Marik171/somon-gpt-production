import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Types matching backend
export interface PredictionRequest {
  area_m2: number;
  floor?: number;
  district?: string;
  build_type?: string;
  renovation?: string;
  bathroom?: string;
  heating?: string;
  tech_passport?: string;
  photo_count?: number;
}

export interface PredictionResponse {
  predicted_price: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  model_version: string;
  is_bargain: boolean;
  bargain_category?: string;
  features_used: Record<string, any>;
  price_per_sqm: number;
}

export interface Property {
  id: number;
  title?: string;
  url: string;
  price: number;
  area?: number;
  rooms?: number;
  floor?: number;
  total_floors?: number;
  city?: string;
  district?: string;
  address?: string;
  build_state?: string;
  property_type?: string;
  predicted_price?: number;
  price_difference?: number;
  price_difference_percentage?: number;
  bargain_score?: number;
  bargain_category?: string;
  is_favorite?: boolean;
  view_count?: number;
}

class PropertyService {
  async getProperties(params?: {
    page?: number;
    size?: number;
    min_price?: number;
    max_price?: number;
    city?: string;
    bargain_only?: boolean;
  }): Promise<Property[]> {
    try {
      const response = await axios.get<Property[]>(`${API_BASE_URL}/properties`, {
        params
      });
      return response.data;
    } catch (error: any) {
      console.error('Error fetching properties:', error);
      throw new Error(error.response?.data?.detail || 'Failed to fetch properties');
    }
  }

  async predictPrice(request: PredictionRequest): Promise<PredictionResponse> {
    try {
      const response = await axios.post<PredictionResponse>(`${API_BASE_URL}/predict`, request);
      return response.data;
    } catch (error: any) {
      console.error('Error predicting price:', error);
      throw new Error(error.response?.data?.detail || 'Failed to predict price');
    }
  }
}

// Export singleton instance
export const propertyService = new PropertyService();
