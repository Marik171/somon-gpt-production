import React, { useState, useCallback, useMemo } from 'react';
import {
  Card,
  CardContent,
  CardMedia,
  Typography,
  Box,
  Chip,
  Button,
  Grid,
  IconButton,
  Tooltip,
  Skeleton,
  Badge,
} from '@mui/material';
import {
  LocationOn,
  Home,
  Layers,
  TrendingUp,
  TrendingDown,
  OpenInNew,
  Favorite,
  FavoriteBorder,
  ArrowBackIos,
  ArrowForwardIos,
  PhotoLibrary,
  AccessTime,
  AccountBalance,
} from '@mui/icons-material';
import { Property } from '../services/api';

interface PropertyCardProps {
  property: Property;
  showInvestmentMetrics?: boolean;
  onFavorite?: (property: Property) => void;
  isFavorited?: boolean;
}

const PropertyCard: React.FC<PropertyCardProps> = ({
  property,
  showInvestmentMetrics = true,
  onFavorite,
  isFavorited = false,
}) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  // Get all available images
  const images = useMemo(() => {
    if (property.image_urls && property.image_urls.length > 0) {
      return property.image_urls;
    }
    return ['https://via.placeholder.com/400x250/e0e0e0/9e9e9e?text=No+Image'];
  }, [property.image_urls]);

  const currentImage = images[currentImageIndex];
  const hasMultipleImages = images.length > 1;

  // Image carousel navigation
  const nextImage = useCallback(() => {
    setCurrentImageIndex((prev) => (prev + 1) % images.length);
    setImageLoaded(false); // Reset loading state for new image
  }, [images.length]);

  const prevImage = useCallback(() => {
    setCurrentImageIndex((prev) => (prev - 1 + images.length) % images.length);
    setImageLoaded(false); // Reset loading state for new image
  }, [images.length]);

  const goToImage = useCallback((index: number) => {
    setCurrentImageIndex(index);
    setImageLoaded(false); // Reset loading state for new image
  }, []);

  const formatPrice = useCallback((price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'TJS',
      currencyDisplay: 'code',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price).replace('TJS', 'TJS');
  }, []);

  const getBargainColor = useCallback((category?: string) => {
    switch (category) {
      case 'exceptional_opportunity':
        return 'success';
      case 'excellent_bargain':
        return 'success';
      case 'good_bargain':
        return 'primary';
      case 'fair_value':
        return 'warning';
      case 'market_price':
        return 'info';
      case 'overpriced':
        return 'error';
      default:
        return 'default';
    }
  }, []);

  const getBargainLabel = useCallback((category?: string) => {
    switch (category) {
      case 'exceptional_opportunity':
        return 'Exceptional Deal!';
      case 'excellent_bargain':
        return 'Excellent Deal!';
      case 'good_bargain':
        return 'Good Investment';
      case 'fair_value':
        return 'Fair Value';
      case 'market_price':
        return 'Market Price';
      case 'overpriced':
        return 'Overpriced';
      default:
        return 'Not Rated';
    }
  }, []);

  const handleImageLoad = useCallback(() => {
    setImageLoaded(true);
  }, []);

  const handleImageError = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    setImageError(true);
    setImageLoaded(true);
    e.currentTarget.src = 'https://via.placeholder.com/400x250/e0e0e0/9e9e9e?text=No+Image';
  }, []);

  return (
    <Card
      className="property-card"
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        borderRadius: 3,
        overflow: 'hidden',
      }}
    >
      {/* Image Carousel Section */}
      <Box sx={{ position: 'relative', height: 250, overflow: 'hidden' }}>
        {!imageLoaded && !imageError && (
          <Skeleton
            variant="rectangular"
            width="100%"
            height={250}
            sx={{ position: 'absolute', top: 0, left: 0, zIndex: 1 }}
          />
        )}
        
        <CardMedia
          component="img"
          height="250"
          image={currentImage}
          alt={property.title || 'Property'}
          onLoad={handleImageLoad}
          onError={handleImageError}
          sx={{ 
            objectFit: 'cover',
            transition: 'opacity 0.3s ease-in-out',
            opacity: imageLoaded ? 1 : 0,
            width: '100%',
            height: '100%',
          }}
        />

        {/* Image Navigation - Only show if multiple images */}
        {hasMultipleImages && imageLoaded && (
          <>
            {/* Previous Button */}
            <IconButton
              onClick={prevImage}
              sx={{
                position: 'absolute',
                left: 8,
                top: '50%',
                transform: 'translateY(-50%)',
                bgcolor: 'rgba(0, 0, 0, 0.5)',
                color: 'white',
                '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.7)' },
                zIndex: 2,
                width: 32,
                height: 32,
              }}
            >
              <ArrowBackIos sx={{ fontSize: 16, ml: 0.5 }} />
            </IconButton>

            {/* Next Button */}
            <IconButton
              onClick={nextImage}
              sx={{
                position: 'absolute',
                right: 8,
                top: '50%',
                transform: 'translateY(-50%)',
                bgcolor: 'rgba(0, 0, 0, 0.5)',
                color: 'white',
                '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.7)' },
                zIndex: 2,
                width: 32,
                height: 32,
              }}
            >
              <ArrowForwardIos sx={{ fontSize: 16 }} />
            </IconButton>

            {/* Image Indicators */}
            <Box
              sx={{
                position: 'absolute',
                bottom: 8,
                left: '50%',
                transform: 'translateX(-50%)',
                display: 'flex',
                gap: 0.5,
                zIndex: 2,
              }}
            >
              {images.map((_, index) => (
                <Box
                  key={index}
                  onClick={() => goToImage(index)}
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    bgcolor: index === currentImageIndex ? 'white' : 'rgba(255, 255, 255, 0.5)',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    '&:hover': {
                      bgcolor: 'white',
                      transform: 'scale(1.2)',
                    },
                  }}
                />
              ))}
            </Box>

            {/* Image Counter */}
            <Chip
              icon={<PhotoLibrary sx={{ fontSize: 16 }} />}
              label={`${currentImageIndex + 1}/${images.length}`}
              size="small"
              sx={{
                position: 'absolute',
                bottom: 8,
                right: 8,
                bgcolor: 'rgba(0, 0, 0, 0.7)',
                color: 'white',
                fontSize: '0.7rem',
                height: 24,
                zIndex: 2,
              }}
            />
          </>
        )}
        
        {/* Bargain Badge */}
        {showInvestmentMetrics && property.bargain_category && (
          <Chip
            label={getBargainLabel(property.bargain_category)}
            color={getBargainColor(property.bargain_category) as any}
            size="small"
            sx={{
              position: 'absolute',
              top: 12,
              left: 12,
              fontWeight: 600,
              textTransform: 'uppercase',
              fontSize: '0.75rem',
            }}
          />
        )}

        {/* Investment Score */}
        {showInvestmentMetrics && property.investment_score && (
          <Box
            sx={{
              position: 'absolute',
              top: 12,
              right: 12,
              bgcolor: 'rgba(0, 0, 0, 0.8)',
              color: 'white',
              px: 1.5,
              py: 0.5,
              borderRadius: 2,
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
            }}
          >
            <TrendingUp sx={{ fontSize: 16 }} />
            <Typography variant="caption" sx={{ fontWeight: 600 }}>
              {property.investment_score.toFixed(1)}
            </Typography>
          </Box>
        )}

        {/* Favorite Button */}
        {onFavorite && (
          <IconButton
            onClick={() => onFavorite(property)}
            sx={{
              position: 'absolute',
              bottom: 12,
              right: 12,
              bgcolor: 'rgba(255, 255, 255, 0.9)',
              '&:hover': { bgcolor: 'white' },
            }}
          >
            {isFavorited ? (
              <Favorite sx={{ color: 'red' }} />
            ) : (
              <FavoriteBorder />
            )}
          </IconButton>
        )}
      </Box>

      <CardContent sx={{ flexGrow: 1, p: 3 }}>
        {/* Price */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h5" sx={{ fontWeight: 700, color: 'primary.main' }}>
            {formatPrice(property.price)}
          </Typography>
          {property.price_per_sqm && (
            <Typography variant="body2" color="text.secondary">
              {formatPrice(property.price_per_sqm)}/m²
            </Typography>
          )}
        </Box>

        {/* Title */}
        {property.title && (
          <Typography
            variant="h6"
            sx={{
              mb: 2,
              fontWeight: 500,
              lineHeight: 1.3,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
            }}
          >
            {property.title}
          </Typography>
        )}

        {/* Property Details */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {property.rooms && (
            <Grid item xs={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Home sx={{ fontSize: 18, color: 'text.secondary' }} />
                <Typography variant="body2" color="text.secondary">
                  {property.rooms} rooms
                </Typography>
              </Box>
            </Grid>
          )}
          {property.area && (
            <Grid item xs={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Layers sx={{ fontSize: 18, color: 'text.secondary' }} />
                <Typography variant="body2" color="text.secondary">
                  {property.area}m²
                </Typography>
              </Box>
            </Grid>
          )}
          {property.floor && (
            <Grid item xs={6}>
              <Typography variant="body2" color="text.secondary">
                Floor {property.floor}
                {property.total_floors && `/${property.total_floors}`}
              </Typography>
            </Grid>
          )}
          {property.build_state && (
            <Grid item xs={6}>
              <Typography variant="body2" color="text.secondary">
                {property.build_state}
              </Typography>
            </Grid>
          )}
        </Grid>

        {/* Location */}
        {(property.city || property.district) && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 2 }}>
            <LocationOn sx={{ fontSize: 18, color: 'text.secondary' }} />
            <Typography variant="body2" color="text.secondary">
              {[property.district, property.city].filter(Boolean).join(', ')}
            </Typography>
          </Box>
        )}

                {/* Renovation Status Badge */}
        {property.renovation && (
          <Box sx={{ mb: 2 }}>
            <Chip
              label={property.renovation}
              size="small"
              color={
                property.renovation?.includes('Новый ремонт') ? 'success' :
                property.renovation?.includes('С ремонтом') ? 'primary' :
                property.renovation?.includes('Без ремонта') ? 'warning' :
                'default'
              }
              sx={{ fontWeight: 500 }}
            />
                         {property.estimated_renovation_cost && property.estimated_renovation_cost > 0 && (
               <Chip
                 label={`+${formatPrice(property.estimated_renovation_cost)} renovation`}
                 size="small"
                 variant="outlined"
                 color="warning"
                 sx={{ ml: 1, fontWeight: 500 }}
               />
             )}
          </Box>
        )}

        {/* Enhanced Investment Metrics */}
        {showInvestmentMetrics && (
          <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600, color: 'primary.main' }}>
              💰 Investment Analysis
            </Typography>
            
            {/* ROI and Rental Yield */}
            <Grid container spacing={1} sx={{ mb: 1.5 }}>
              {property.roi_percentage && (
                <Grid item xs={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <TrendingUp sx={{ fontSize: 16, color: 'success.main' }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1 }}>
                        ROI
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600, color: 'success.main' }}>
                        {property.roi_percentage.toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
              )}
              
              {property.payback_period_years && property.payback_period_years < 50 && (
                <Grid item xs={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <AccessTime sx={{ fontSize: 16, color: 'info.main' }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1 }}>
                        Payback
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600, color: 'info.main' }}>
                        {property.payback_period_years.toFixed(1)}y
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
              )}
            </Grid>

            {/* Monthly Rent and Cash Flow */}
            <Grid container spacing={1} sx={{ mb: 1.5 }}>
              {property.estimated_monthly_rent && (
                <Grid item xs={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Home sx={{ fontSize: 16, color: 'primary.main' }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1 }}>
                        Monthly Rent
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {formatPrice(property.estimated_monthly_rent)}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
              )}
              
              {property.monthly_cash_flow !== undefined && (
                <Grid item xs={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <AccountBalance sx={{ fontSize: 16, color: property.monthly_cash_flow >= 0 ? 'success.main' : 'error.main' }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1 }}>
                        Cash Flow
                      </Typography>
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          fontWeight: 600, 
                          color: property.monthly_cash_flow >= 0 ? 'success.main' : 'error.main' 
                        }}
                      >
                        {formatPrice(property.monthly_cash_flow)}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
              )}
            </Grid>

            {/* Renovation Investment Section */}
            {property.estimated_renovation_cost && property.estimated_renovation_cost > 0 && (
              <Box sx={{ mt: 2, p: 2, bgcolor: 'orange.50', borderRadius: 1, border: '1px solid', borderColor: 'orange.200' }}>
                <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600, color: 'orange.800' }}>
                  🔨 Renovation Investment
                </Typography>
                
                <Grid container spacing={1} sx={{ mb: 1 }}>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                      Total Investment
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {formatPrice(property.total_investment_required || (property.price + property.estimated_renovation_cost))}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                      Renovation Cost
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'orange.800' }}>
                      {formatPrice(property.estimated_renovation_cost)}
                    </Typography>
                  </Grid>
                </Grid>

                {property.monthly_rent_premium && property.monthly_rent_premium > 0 && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
                    <TrendingUp sx={{ fontSize: 16, color: 'success.main' }} />
                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'success.main' }}>
                      +{formatPrice(property.monthly_rent_premium)}/month premium
                    </Typography>
                  </Box>
                )}

                {property.renovation_roi_annual && property.renovation_roi_annual > 0 && (
                  <Typography variant="caption" color="success.main" sx={{ fontWeight: 600 }}>
                    {property.renovation_roi_annual.toFixed(1)}% annual ROI on renovation
                  </Typography>
                )}
              </Box>
            )}





            {/* Legacy metrics (if new ones not available) */}
            {!property.roi_percentage && property.predicted_price && (
              <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                Predicted: {formatPrice(property.predicted_price)}
              </Typography>
            )}
            {!property.roi_percentage && property.price_difference_percentage && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                {property.price_difference_percentage > 0 ? (
                  <TrendingDown sx={{ fontSize: 16, color: 'success.main' }} />
                ) : (
                  <TrendingUp sx={{ fontSize: 16, color: 'error.main' }} />
                )}
                <Typography
                  variant="body2"
                  sx={{
                    color: property.price_difference_percentage > 0 ? 'success.main' : 'error.main',
                    fontWeight: 600,
                  }}
                >
                  {Math.abs(property.price_difference_percentage).toFixed(1)}%{' '}
                  {property.price_difference_percentage > 0 ? 'below market' : 'above market'}
                </Typography>
              </Box>
            )}
          </Box>
        )}

        {/* Action Button */}
        <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
          {property.url && (
            <Button
              variant="contained"
              fullWidth
              endIcon={<OpenInNew />}
              onClick={() => window.open(property.url, '_blank')}
              sx={{ borderRadius: 2 }}
            >
              View Details
            </Button>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default PropertyCard;
