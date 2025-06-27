import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Grid, 
  CircularProgress, 
  Alert,
  Button
} from '@mui/material';
import { Favorite, FavoriteBorder, Refresh } from '@mui/icons-material';
import { useAuth } from '../services/auth';
import ProtectedRoute from '../components/auth/ProtectedRoute';
import PropertyCard from '../components/PropertyCard';
import { Property } from '../services/api';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const FavoritesPage: React.FC = () => {
  const { user, isAuthenticated } = useAuth();
  const [favorites, setFavorites] = useState<Property[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchFavorites = async () => {
    if (!isAuthenticated) return;
    
    try {
      setLoading(true);
      setError('');
      
      // Get favorite property IDs from localStorage
      const savedFavorites = localStorage.getItem('favorites');
      const favoriteIds: number[] = savedFavorites ? JSON.parse(savedFavorites) : [];
      
      if (favoriteIds.length === 0) {
        setFavorites([]);
        setLoading(false);
        return;
      }
      
      // Fetch property details for favorite IDs
      const favoriteProperties: Property[] = [];
      
      for (const propertyId of favoriteIds) {
        try {
          const response = await fetch(`${API_BASE_URL}/properties/${propertyId}`, {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('real_estate_token')}`,
              'Content-Type': 'application/json',
            },
          });
          
          if (response.ok) {
            const property = await response.json();
            favoriteProperties.push(property);
          }
        } catch (error) {
          console.warn(`Failed to fetch property ${propertyId}:`, error);
        }
      }
      
      setFavorites(favoriteProperties);
      console.log('✅ Favorites loaded from localStorage:', favoriteProperties.length);
    } catch (error: any) {
      console.error('❌ Error fetching favorites:', error);
      setError('Failed to load favorite properties');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFavorites();
  }, [isAuthenticated]);

  // Listen for localStorage changes to update favorites in real-time
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'favorites') {
        fetchFavorites();
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const handleToggleFavorite = async (propertyId: number) => {
    if (!isAuthenticated) return;
    
    try {
      // Remove from localStorage
      const savedFavorites = localStorage.getItem('favorites');
      const favoriteIds: number[] = savedFavorites ? JSON.parse(savedFavorites) : [];
      const updatedFavorites = favoriteIds.filter(id => id !== propertyId);
      localStorage.setItem('favorites', JSON.stringify(updatedFavorites));
      
      // Remove from local state
      setFavorites(prev => prev.filter(prop => prop.id !== propertyId));
      
      console.log(`✅ Removed property ${propertyId} from favorites`);
    } catch (error) {
      console.error('Error removing favorite:', error);
    }
  };

  if (!isAuthenticated) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <FavoriteBorder sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h5" gutterBottom>
            Please Login to View Favorites
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Sign in to save and view your favorite properties.
          </Typography>
        </Box>
      </Container>
    );
  }

  return (
    <ProtectedRoute>
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
              <Favorite sx={{ color: 'error.main', fontSize: 40 }} />
              Your Favorite Properties
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Properties you've saved for {user?.email}
            </Typography>
          </Box>
          
          <Button 
            variant="outlined" 
            startIcon={<Refresh />}
            onClick={fetchFavorites}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 4 }}>
            {error}
          </Alert>
        )}

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
            <CircularProgress size={40} />
          </Box>
        ) : favorites.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <FavoriteBorder sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h5" gutterBottom>
              No Favorite Properties Yet
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              Start exploring properties and click the heart icon to save your favorites here.
            </Typography>
            <Button 
              variant="contained" 
              href="/search"
              sx={{ mt: 2 }}
            >
              Browse Properties
            </Button>
          </Box>
        ) : (
          <>
            <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="h6" color="text.secondary">
                {favorites.length} Favorite {favorites.length === 1 ? 'Property' : 'Properties'}
              </Typography>
            </Box>
            
            <Grid container spacing={3}>
              {favorites.map((property) => (
                <Grid item xs={12} sm={6} md={4} key={property.id}>
                  <PropertyCard 
                    property={property}
                    isFavorited={true} // Always true since this is the favorites page
                    onFavorite={(prop) => handleToggleFavorite(prop.id!)}
                  />
                </Grid>
              ))}
            </Grid>
          </>
        )}
      </Container>
    </ProtectedRoute>
  );
};

export default FavoritesPage;
