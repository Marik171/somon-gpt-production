import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Grid,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Chip,
  Button,
} from '@mui/material';
import {
  TrendingUp,
  Star,
  Refresh,
  FilterList,
} from '@mui/icons-material';
import PropertyCard from '../components/PropertyCard';
import { apiService, Property } from '../services/api';

const BargainFinder: React.FC = () => {
  const [properties, setProperties] = useState<Property[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [category, setCategory] = useState<string>('all');
  const [limit, setLimit] = useState<number>(20);
  const [favorites, setFavorites] = useState<Set<number>>(new Set());

  const fetchBargains = async () => {
    console.log('🔍 BargainFinder: fetchBargains called with category:', category, 'limit:', limit);
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.getBargainProperties(category, limit);
      console.log('✅ BargainFinder: Received bargain properties:', data?.length || 0);
      console.log('First bargain sample:', data[0]);
      setProperties(data);
    } catch (err) {
      console.error('❌ BargainFinder: Error fetching bargains:', err);
      setError('Failed to load bargain properties. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBargains();
  }, [category, limit]);

  const handleFavorite = (property: Property) => {
    if (!property.id) return;
    
    const newFavorites = new Set(favorites);
    if (favorites.has(property.id)) {
      newFavorites.delete(property.id);
    } else {
      newFavorites.add(property.id);
    }
    setFavorites(newFavorites);
    
    // In a real app, you would save this to localStorage or backend
    localStorage.setItem('favorites', JSON.stringify(Array.from(newFavorites)));
  };

  // Load favorites from localStorage on mount
  useEffect(() => {
    const savedFavorites = localStorage.getItem('favorites');
    if (savedFavorites) {
      setFavorites(new Set(JSON.parse(savedFavorites)));
    }
  }, []);

  const getCategoryStats = () => {
    const excellent = properties.filter(p => 
      p.bargain_category === 'exceptional_opportunity' || 
      p.bargain_category === 'excellent_bargain'
    ).length;
    const good = properties.filter(p => p.bargain_category === 'good_bargain').length;
    const total = properties.length;
    
    return { excellent, good, total };
  };

  const categoryOptions = [
    { value: 'all', label: 'All Bargains' },
    { value: 'excellent', label: 'Excellent Deals' },
    { value: 'good', label: 'Good Investments' },
    { value: 'fair', label: 'Fair Value' },
  ];

  const limitOptions = [
    { value: 10, label: '10 properties' },
    { value: 20, label: '20 properties' },
    { value: 50, label: '50 properties' },
  ];

  const stats = getCategoryStats();

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: 4 }}>
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
          }}
        >
          <TrendingUp sx={{ fontSize: 48 }} />
          Investment Bargain Finder
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 800, mx: 'auto' }}>
          Discover undervalued properties with high investment potential using AI-powered market analysis
        </Typography>
      </Box>

      {/* Controls */}
      <Card sx={{ mb: 4, p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
          <FilterList sx={{ color: 'primary.main' }} />
          <Typography variant="h6">Filter Options</Typography>
        </Box>
        
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Category</InputLabel>
              <Select
                value={category}
                label="Category"
                onChange={(e) => setCategory(e.target.value)}
              >
                {categoryOptions.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Show</InputLabel>
              <Select
                value={limit}
                label="Show"
                onChange={(e) => setLimit(Number(e.target.value))}
              >
                {limitOptions.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Button
              variant="outlined"
              fullWidth
              startIcon={<Refresh />}
              onClick={fetchBargains}
              disabled={loading}
            >
              Refresh Results
            </Button>
          </Grid>
        </Grid>
      </Card>

      {/* Statistics */}
      {!loading && !error && (
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={3}>
            <Card sx={{ textAlign: 'center', p: 2, bgcolor: 'success.50' }}>
              <Typography variant="h3" sx={{ fontWeight: 700, color: 'success.main' }}>
                {stats.excellent}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Excellent Deals
              </Typography>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card sx={{ textAlign: 'center', p: 2, bgcolor: 'primary.50' }}>
              <Typography variant="h3" sx={{ fontWeight: 700, color: 'primary.main' }}>
                {stats.good}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Good Investments
              </Typography>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card sx={{ textAlign: 'center', p: 2, bgcolor: 'info.50' }}>
              <Typography variant="h3" sx={{ fontWeight: 700, color: 'info.main' }}>
                {stats.total}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Bargains
              </Typography>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card sx={{ textAlign: 'center', p: 2, bgcolor: 'warning.50' }}>
              <Typography variant="h3" sx={{ fontWeight: 700, color: 'warning.main' }}>
                {favorites.size}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Your Favorites
              </Typography>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Loading State */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress size={60} />
        </Box>
      )}

      {/* Error State */}
      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}

      {/* Results */}
      {!loading && !error && (
        <>
          {properties.length === 0 ? (
            <Card sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="h6" color="text.secondary">
                No bargain properties found with the selected filters.
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Try adjusting your filter criteria or check back later for new opportunities.
              </Typography>
            </Card>
          ) : (
            <>
              {/* Results Header */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h5" sx={{ fontWeight: 600 }}>
                  {properties.length} Investment Opportunities Found
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Chip
                    icon={<Star />}
                    label={`${favorites.size} Favorites`}
                    color="primary"
                    variant="outlined"
                  />
                </Box>
              </Box>

              {/* Property Grid */}
              <Grid container spacing={3}>
                {properties.map((property, index) => (
                  <Grid item xs={12} sm={6} lg={4} key={property.id || index}>
                    <PropertyCard
                      property={property}
                      showInvestmentMetrics={true}
                      onFavorite={handleFavorite}
                      isFavorited={property.id ? favorites.has(property.id) : false}
                    />
                  </Grid>
                ))}
              </Grid>
            </>
          )}
        </>
      )}

      {/* Investment Tips */}
      {!loading && properties.length > 0 && (
        <Card sx={{ mt: 6, p: 4, bgcolor: 'primary.50' }}>
          <Typography variant="h5" sx={{ mb: 3, fontWeight: 600, color: 'primary.main' }}>
            💡 Investment Tips
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                Excellent Deals
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Properties significantly undervalued by our AI model. These represent the best investment opportunities with highest potential returns.
              </Typography>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                Due Diligence
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Always visit properties in person, verify legal documentation, and consider factors like neighborhood development and transportation.
              </Typography>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                Market Timing
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Our predictions are based on current market data. Consider economic factors and future development plans for the area.
              </Typography>
            </Grid>
          </Grid>
        </Card>
      )}
    </Container>
  );
};

export default BargainFinder;
