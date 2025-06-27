import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Grid,
  Box,
  Card,
  CardContent,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Alert,
  CircularProgress,
  Chip,
  Divider,
  Paper,
} from '@mui/material';
import {
  PsychologyAlt,
  Calculate,
  TrendingUp,
  Info,
  CheckCircle,
  Home,
  MonetizationOn,
} from '@mui/icons-material';
import { apiService, PredictionRequest, PredictionResponse } from '../services/api';

const RentalPredictor: React.FC = () => {
  const [formData, setFormData] = useState<PredictionRequest>({
    rooms: 3,
    area_m2: 75,
    floor: 5,
    district: '',
    renovation: 'С ремонтом',
    bathroom: 'Раздельный',
    heating: 'Есть',
  });
  
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [districts, setDistricts] = useState<string[]>([]);
  const [formValid, setFormValid] = useState(false);

  // Load districts
  useEffect(() => {
    const loadDistricts = async () => {
      try {
        const data = await apiService.getDistricts();
        setDistricts(data);
        if (data.length > 0 && !formData.district) {
          setFormData(prev => ({ ...prev, district: data[0] }));
        }
      } catch (err) {
        console.error('Error loading districts:', err);
      }
    };
    
    loadDistricts();
  }, []);

    // Validate form
  useEffect(() => {
    const isValid = 
      formData.area_m2 > 0 &&
      formData.rooms !== undefined && formData.rooms > 0 &&
      formData.floor !== undefined && formData.floor > 0 &&
      formData.district !== '' &&
      formData.renovation !== '' &&
      formData.bathroom !== '' &&
      formData.heating !== '';
    
    setFormValid(isValid);
  }, [formData]);

  const handleInputChange = (field: keyof PredictionRequest, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handlePredict = async () => {
    if (!formValid) return;
    
    setLoading(true);
    setError(null);
    setPrediction(null);
    
    try {
      const result = await apiService.predictPrice(formData);
      setPrediction(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to predict rental price. Please try again.');
      console.error('Rental prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'TJS',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price);
  };

  const renovationOptions = [
    'Новый ремонт',
    'С ремонтом',
    'Без ремонта (коробка)',
  ];

  const bathroomOptions = [
    'Раздельный',
    'Совмещенный',
  ];

  const heatingOptions = [
    'Есть',
    'Нет',
  ];

  const getConfidenceLevel = (prediction: PredictionResponse) => {
    const range = prediction.confidence_interval_upper - prediction.confidence_interval_lower;
    const percentage = (range / prediction.predicted_rental) * 100;
    
    if (percentage < 20) return { level: 'High', color: 'success' };
    if (percentage < 40) return { level: 'Medium', color: 'warning' };
    return { level: 'Low', color: 'error' };
  };

  return (
    <Container maxWidth="lg" sx={{ py: { xs: 2, sm: 4 } }}>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: { xs: 3, sm: 4 } }}>
        <Typography
          variant="h2"
          sx={{
            fontWeight: 700,
            color: 'primary.main',
            mb: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 2,
            fontSize: { xs: '1.75rem', sm: '2.5rem', md: '3rem' },
            flexDirection: { xs: 'column', sm: 'row' },
          }}
        >
          <Home sx={{ fontSize: { xs: 40, sm: 48 } }} />
          AI Rental Predictor
        </Typography>
        <Typography 
          variant="h6" 
          color="text.secondary" 
          sx={{ 
            maxWidth: 800, 
            mx: 'auto',
            fontSize: { xs: '1rem', sm: '1.125rem', md: '1.25rem' },
            px: { xs: 2, sm: 0 },
          }}
        >
          Get accurate rental price predictions using machine learning trained on Khujand rental market data
        </Typography>
      </Box>

      <Grid container spacing={{ xs: 2, sm: 4 }}>
        {/* Input Form */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent sx={{ p: { xs: 3, sm: 4 } }}>
              <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                Property Details
              </Typography>
              
              <Grid container spacing={{ xs: 2, sm: 3 }}>
                {/* Rooms */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Number of Rooms"
                    type="number"
                    value={formData.rooms}
                    onChange={(e) => handleInputChange('rooms', Number(e.target.value))}
                    inputProps={{ min: 1, max: 10 }}
                    helperText="Number of rooms in the property"
                  />
                </Grid>
                
                {/* Area */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Area (m²)"
                    type="number"
                    value={formData.area_m2}
                    onChange={(e) => handleInputChange('area_m2', Number(e.target.value))}
                    inputProps={{ min: 1, max: 500 }}
                    helperText="Property area in square meters"
                  />
                </Grid>
                
                {/* Floor */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Floor"
                    type="number"
                    value={formData.floor}
                    onChange={(e) => handleInputChange('floor', Number(e.target.value))}
                    inputProps={{ min: 1, max: 50 }}
                    helperText="Floor number"
                  />
                </Grid>
                

                
                {/* District */}
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>District</InputLabel>
                    <Select
                      value={formData.district}
                      label="District"
                      onChange={(e) => handleInputChange('district', e.target.value)}
                    >
                      {districts.map((district) => (
                        <MenuItem key={district} value={district}>
                          {district}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                
                {/* Renovation */}
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Renovation Status</InputLabel>
                    <Select
                      value={formData.renovation}
                      label="Renovation Status"
                      onChange={(e) => handleInputChange('renovation', e.target.value)}
                    >
                      {renovationOptions.map((renovation) => (
                        <MenuItem key={renovation} value={renovation}>
                          {renovation}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                
                {/* Bathroom */}
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Bathroom Type</InputLabel>
                    <Select
                      value={formData.bathroom}
                      label="Bathroom Type"
                      onChange={(e) => handleInputChange('bathroom', e.target.value)}
                    >
                      {bathroomOptions.map((bathroom) => (
                        <MenuItem key={bathroom} value={bathroom}>
                          {bathroom}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                {/* Heating */}
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Heating</InputLabel>
                    <Select
                      value={formData.heating}
                      label="Heating"
                      onChange={(e) => handleInputChange('heating', e.target.value)}
                    >
                      {heatingOptions.map((heating) => (
                        <MenuItem key={heating} value={heating}>
                          {heating}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
              
              {/* Predict Button */}
              <Button
                variant="contained"
                size="large"
                fullWidth
                startIcon={loading ? <CircularProgress size={20} /> : <Calculate />}
                onClick={handlePredict}
                disabled={!formValid || loading}
                sx={{ 
                  mt: 4, 
                  py: 1.5,
                  fontSize: { xs: '0.9rem', sm: '1rem' },
                }}
              >
                {loading ? 'Predicting...' : 'Predict Rental Price'}
              </Button>
              
              {/* Form Validation */}
              {!formValid && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Info />
                    Please fill in all required fields to get a rental price prediction.
                  </Box>
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Results */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: 'fit-content' }}>
            <CardContent sx={{ p: { xs: 3, sm: 4 } }}>
              <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                Rental Prediction
              </Typography>
              
              {/* Error State */}
              {error && (
                <Alert severity="error" sx={{ mb: 3 }}>
                  {error}
                </Alert>
              )}
              
              {/* Loading State */}
              {loading && (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <CircularProgress size={60} />
                  <Typography variant="body2" sx={{ mt: 2 }}>
                    Analyzing rental market data...
                  </Typography>
                </Box>
              )}
              
              {/* Prediction Results */}
              {prediction && !loading && (
                <Box>
                  {/* Main Prediction */}
                  <Paper sx={{ p: 3, mb: 3, bgcolor: 'primary.50', textAlign: 'center' }}>
                    <Typography variant="h3" sx={{ fontWeight: 700, color: 'primary.main', mb: 1 }}>
                      {formatPrice(prediction.predicted_rental)}
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                      Monthly Rental Price
                    </Typography>
                  </Paper>

                  {/* Investment Metrics */}
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={6}>
                      <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.50' }}>
                        <MonetizationOn sx={{ color: 'success.main', mb: 1 }} />
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {formatPrice(prediction.annual_rental_income)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Annual Income
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={6}>
                      <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'info.50' }}>
                        <TrendingUp sx={{ color: 'info.main', mb: 1 }} />
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {prediction.gross_rental_yield.toFixed(1)}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Rental Yield
                        </Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                  
                  {/* Confidence Range */}
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
                      Confidence Range
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="body2">
                        Low: {formatPrice(prediction.confidence_interval_lower)}
                      </Typography>
                      <Chip
                        label={`${getConfidenceLevel(prediction).level} Confidence`}
                        color={getConfidenceLevel(prediction).color as any}
                        size="small"
                      />
                      <Typography variant="body2">
                        High: {formatPrice(prediction.confidence_interval_upper)}
                      </Typography>
                    </Box>
                    <Box sx={{ 
                      height: 8, 
                      bgcolor: 'grey.200', 
                      borderRadius: 1, 
                      position: 'relative',
                      overflow: 'hidden'
                    }}>
                      <Box sx={{ 
                        height: '100%', 
                        bgcolor: 'primary.main', 
                        width: '100%',
                        borderRadius: 1
                      }} />
                      <Box sx={{
                        position: 'absolute',
                        top: 0,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        height: '100%',
                        width: 2,
                        bgcolor: 'error.main'
                      }} />
                    </Box>
                  </Box>
                  

                </Box>
              )}
              
              {/* Default State */}
              {!prediction && !loading && !error && (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Home sx={{ fontSize: 64, color: 'grey.300', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary">
                    Enter property details to get AI-powered rental prediction
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Our rental model analyzes market data to provide accurate rental estimates
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Additional Information */}
      <Card sx={{ mt: 4, bgcolor: 'grey.50' }}>
        <CardContent sx={{ p: { xs: 3, sm: 4 } }}>
          <Typography 
            variant="h5" 
            sx={{ 
              mb: 3, 
              fontWeight: 600,
              fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2rem' },
            }}
          >
            How It Works
          </Typography>
          <Grid container spacing={{ xs: 2, sm: 3 }}>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <PsychologyAlt sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                <Typography variant="h6" sx={{ mb: 1 }}>
                  XGBoost Algorithm
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Advanced machine learning model trained on rental market data with 35+ features
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <TrendingUp sx={{ fontSize: 48, color: 'secondary.main', mb: 2 }} />
                <Typography variant="h6" sx={{ mb: 1 }}>
                  Market Analysis
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Considers location, property characteristics, and current rental market trends
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <CheckCircle sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />
                <Typography variant="h6" sx={{ mb: 1 }}>
                  Reliable Predictions
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Comprehensive validation with real rental market data for trusted results
                </Typography>
              </Box>
            </Grid>
          </Grid>
          
                      <Alert severity="info" sx={{ mt: 3 }}>
              <Typography variant="body2">
                <strong>Disclaimer:</strong> This rental prediction model is specifically trained on Khujand market data. 
                Predictions are most accurate for properties in Khujand and surrounding areas. 
                Actual rental prices may vary due to market conditions, property condition, and seasonal factors. 
                Always consult with local real estate professionals for investment decisions.
              </Typography>
            </Alert>
        </CardContent>
      </Card>
    </Container>
  );
};

export default RentalPredictor;
