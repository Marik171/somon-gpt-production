import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Chip,
  Grid
} from '@mui/material';
import { useAuth } from '../../services/auth';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface SearchHistory {
  id: number;
  search_type: string;
  search_params: any;
  results_count: number;
  created_at: string;
}

interface PredictionHistory {
  id: number;
  property_features: any;
  predicted_price: number;
  confidence_lower: number;
  confidence_upper: number;
  created_at: string;
}

const UserHistory: React.FC = () => {
  const [searchHistory, setSearchHistory] = useState<SearchHistory[]>([]);
  const [predictionHistory, setPredictionHistory] = useState<PredictionHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState<'searches' | 'predictions'>('searches');
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    if (isAuthenticated) {
      fetchHistory();
    }
  }, [isAuthenticated]);

  const fetchHistory = async () => {
    try {
      console.log('🔍 UserHistory: Fetching user history...');
      setLoading(true);
      const [searchRes, predictionRes] = await Promise.all([
        fetch(`${API_BASE_URL}/user/searches`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('real_estate_token')}`,
            'Content-Type': 'application/json',
          },
        }),
        fetch(`${API_BASE_URL}/user/predictions`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('real_estate_token')}`,
            'Content-Type': 'application/json',
          },
        })
      ]);
      
      if (!searchRes.ok || !predictionRes.ok) {
        throw new Error(`HTTP error! Search: ${searchRes.status}, Predictions: ${predictionRes.status}`);
      }
      
      const searchData = await searchRes.json();
      const predictionData = await predictionRes.json();
      
      console.log('✅ UserHistory: Search history received:', searchData.length);
      console.log('✅ UserHistory: Prediction history received:', predictionData.length);
      
      setSearchHistory(searchData);
      setPredictionHistory(predictionData);
    } catch (error: any) {
      console.error('❌ UserHistory: Error fetching history:', error);
      setError('Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
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
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <Typography variant="body1" color="text.secondary">
          Please login to view your history.
        </Typography>
      </Box>
    );
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <CircularProgress size={40} />
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: '4xl', mx: 'auto', p: 3 }}>      
      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}

      {/* Tab Navigation */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
          <Tab 
            label={`Search History (${searchHistory.length})`} 
            value="searches" 
          />
          <Tab 
            label={`Prediction History (${predictionHistory.length})`} 
            value="predictions" 
          />
        </Tabs>
      </Box>

      {/* Search History Tab */}
      {activeTab === 'searches' && (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {searchHistory.length === 0 ? (
            <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', py: 8 }}>
              No search history found.
            </Typography>
          ) : (
            searchHistory.map((search) => (
              <Card key={search.id} variant="outlined">
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="h6" gutterBottom>
                        {search.search_type} Search
                      </Typography>
                      <Grid container spacing={1} sx={{ mb: 2 }}>
                        {Object.entries(search.search_params).map(([key, value]) => (
                          <Grid item key={key}>
                            <Chip 
                              label={`${key}: ${String(value)}`} 
                              variant="outlined" 
                              size="small" 
                            />
                          </Grid>
                        ))}
                      </Grid>
                      <Typography variant="body2" color="text.secondary">
                        Found {search.results_count} results
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      {formatDate(search.created_at)}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            ))
          )}
        </Box>
      )}

      {/* Prediction History Tab */}
      {activeTab === 'predictions' && (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {predictionHistory.length === 0 ? (
            <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', py: 8 }}>
              No prediction history found.
            </Typography>
          ) : (
            predictionHistory.map((prediction) => (
              <Card key={prediction.id} variant="outlined">
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="h6" gutterBottom>
                        Price Prediction
                      </Typography>
                      <Grid container spacing={1} sx={{ mb: 2 }}>
                        {Object.entries(prediction.property_features).map(([key, value]) => (
                          <Grid item key={key}>
                            <Chip 
                              label={`${key}: ${String(value)}`} 
                              variant="outlined" 
                              size="small" 
                            />
                          </Grid>
                        ))}
                      </Grid>
                      <Card sx={{ bgcolor: 'primary.50', p: 2 }}>
                        <Typography variant="h6" color="primary.main" gutterBottom>
                          Predicted Price: {formatPrice(prediction.predicted_price)}
                        </Typography>
                        <Typography variant="body2" color="primary.dark">
                          Range: {formatPrice(prediction.confidence_lower)} - {formatPrice(prediction.confidence_upper)}
                        </Typography>
                      </Card>
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      {formatDate(prediction.created_at)}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            ))
          )}
        </Box>
      )}
    </Box>
  );
};

export default UserHistory;
