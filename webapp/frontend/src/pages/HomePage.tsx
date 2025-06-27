import React, { useEffect, useState } from 'react';
import {
  Container,
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  LinearProgress,
  Alert,
  useTheme,
} from '@mui/material';
import {
  TrendingUp,
  Search,
  Dashboard,
  PsychologyAlt,
  CloudDownload,
  Home as HomeIcon,
  AttachMoney,
  Assessment,
  SmartToy,
} from '@mui/icons-material';
import { Link } from 'react-router-dom';
import { apiService, HealthStatus } from '../services/api';
import { useAuth } from '../services/auth';

const HomePage: React.FC = () => {
  const theme = useTheme();
  const { user } = useAuth();
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealthStatus = async () => {
      try {
        const status = await apiService.getHealthStatus();
        setHealthStatus(status);
      } catch (err) {
        setError('Failed to connect to backend services');
        console.error('Health check failed:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchHealthStatus();
  }, []);

  const features = [
    {
      title: 'Data Collection',
      description: 'Start by collecting property data based on your criteria. This is the first step to enable all other features.',
      icon: <CloudDownload sx={{ fontSize: { xs: 32, sm: 36 }, color: 'success.main' }} />,
      link: '/collect',
      color: 'success',
      stats: healthStatus?.total_properties ? 'Data ready' : 'No data yet - Start here!',
      primary: !healthStatus?.total_properties,
    },
    {
      title: 'Bargain Finder',
      description: 'Discover undervalued properties with high investment potential using AI-powered analysis.',
      icon: <TrendingUp sx={{ fontSize: { xs: 32, sm: 36 }, color: 'primary.main' }} />,
      link: '/bargains',
      color: 'primary',
      stats: healthStatus ? `${healthStatus.total_properties} properties analyzed` : 'Loading...',
      disabled: !healthStatus?.total_properties,
    },
    {
      title: 'Property Search',
      description: 'Advanced search with smart filters to find properties matching your exact criteria.',
      icon: <Search sx={{ fontSize: { xs: 32, sm: 36 }, color: 'secondary.main' }} />,
      link: '/search',
      color: 'secondary',
      stats: 'Smart filtering available',
      disabled: !healthStatus?.total_properties,
    },
    {
      title: 'Market Dashboard',
      description: 'Interactive analytics and market trends with real-time data visualization.',
      icon: <Dashboard sx={{ fontSize: { xs: 32, sm: 36 }, color: 'info.main' }} />,
      link: '/dashboard',
      color: 'info',
      stats: 'Live market data',
      disabled: !healthStatus?.total_properties,
    },
    {
      title: 'AI Price Predictor',
      description: 'Get accurate property valuations using machine learning trained on market data.',
      icon: <PsychologyAlt sx={{ fontSize: { xs: 32, sm: 36 }, color: 'warning.main' }} />,
      link: '/predict',
      color: 'warning',
      stats: user?.has_collected_data && healthStatus?.model_loaded ? 'AI model ready' : 'Collect data first',
      disabled: !user?.has_collected_data,
    },
  ];

  const keyMetrics = [
    {
      label: 'Total Properties',
      value: healthStatus?.total_properties?.toLocaleString() || '0',
      icon: <HomeIcon />,
      showForNewUsers: false, // Hide for new users
    },
    {
      label: 'Avg. Price',
      value: healthStatus?.avg_price 
        ? `${Math.round(healthStatus.avg_price).toLocaleString()} TJS`
        : 'Loading...',
      icon: <AttachMoney />,
      showForNewUsers: false, // Hide for new users
    },
    {
      label: 'Investment Opportunities',
      value: healthStatus?.investment_opportunities?.toString() || 'Loading...',
      icon: <TrendingUp />,
      showForNewUsers: false, // Hide for new users
    },
  ];

  // Filter metrics to show based on user data collection status
  const displayedMetrics = keyMetrics.filter(metric => 
    user?.has_collected_data || metric.showForNewUsers
  );

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Hero Section */}
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography
          variant="h1"
          sx={{
            fontSize: { xs: '2.5rem', md: '3.5rem' },
            fontWeight: 700,
            color: 'primary.main',
            mb: 2,
          }}
        >
          {user?.has_collected_data 
            ? 'Real Estate Intelligence Platform'
            : 'Welcome to Real Estate Analytics'
          }
        </Typography>
        <Typography
          variant="h5"
          sx={{
            color: 'text.secondary',
            mb: 4,
            maxWidth: 800,
            mx: 'auto',
            lineHeight: 1.6,
          }}
        >
          {user?.has_collected_data 
            ? 'AI-powered property analysis and investment intelligence for the Tajikistan real estate market'
            : 'Get started with AI-powered property analysis and investment intelligence'
          }
        </Typography>

        {/* Welcome message for new users */}
        {!user?.has_collected_data && (
          <Alert severity="info" sx={{ mb: 4, maxWidth: 600, mx: 'auto' }}>
            <Typography variant="body1" sx={{ fontWeight: 500 }}>
              Welcome! Start by collecting property data to unlock all features and personalized insights.
            </Typography>
          </Alert>
        )}

        {/* System Status */}
        {loading && (
          <Box sx={{ mb: 4 }}>
            <LinearProgress />
            <Typography variant="body2" sx={{ mt: 1 }}>
              Connecting to services...
            </Typography>
          </Box>
        )}

        {error && (
          <Alert severity="warning" sx={{ mb: 4, maxWidth: 600, mx: 'auto' }}>
            {error} - Some features may be limited
          </Alert>
        )}

      </Box>

      {/* Key Metrics - Only show for users who have collected data */}
      {user?.has_collected_data && displayedMetrics.length > 0 && (
        <Grid container spacing={{ xs: 2, md: 3 }} sx={{ mb: 6, justifyContent: 'center' }}>
          {displayedMetrics.map((metric, index) => (
            <Grid item xs={6} md={4} key={index}>
              <Card
                sx={{
                  textAlign: 'center',
                  p: { xs: 1.5, sm: 2 },
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: 'white',
                  minHeight: { xs: 120, sm: 140 },
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                }}
              >
                <Box sx={{ mb: { xs: 0.5, sm: 1 }, fontSize: { xs: 20, sm: 24 } }}>{metric.icon}</Box>
                <Typography 
                  variant="h4" 
                  sx={{ 
                    fontWeight: 700, 
                    mb: 1,
                    fontSize: { xs: '1.25rem', sm: '1.5rem', md: '2rem' },
                  }}
                >
                  {metric.value}
                </Typography>
                <Typography 
                  variant="body2" 
                  sx={{ 
                    opacity: 0.9,
                    fontSize: { xs: '0.75rem', sm: '0.875rem' },
                  }}
                >
                  {metric.label}
                </Typography>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Main Features */}
      <Grid container spacing={{ xs: 2, sm: 3 }} sx={{ mb: 6 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} sm={6} lg={4} xl={6} key={index}>
            <Card
              className="property-card"
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
                overflow: 'hidden',
                minHeight: { xs: 220, sm: 280 },
              }}
            >
              <CardContent sx={{ flexGrow: 1, p: { xs: 2.5, sm: 3 } }}>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'flex-start', 
                  mb: 2, 
                  gap: { xs: 1, sm: 1.5 },
                  flexWrap: 'wrap',
                }}>
                  <Box sx={{ flexShrink: 0 }}>
                    {feature.icon}
                  </Box>
                  <Typography 
                    variant="h6" 
                    sx={{ 
                      fontWeight: 600,
                      fontSize: { xs: '1rem', sm: '1.1rem', md: '1.25rem' },
                      lineHeight: 1.3,
                      flexGrow: 1,
                      minWidth: 0,
                      mt: { xs: 0.5, sm: 0 },
                    }}
                  >
                    {feature.title}
                  </Typography>
                  {feature.primary && (
                    <Chip 
                      label="START HERE" 
                      color="success" 
                      size="small" 
                      sx={{ 
                        flexShrink: 0,
                        fontSize: { xs: '0.65rem', sm: '0.75rem' },
                        height: { xs: 20, sm: 24 },
                      }}
                    />
                  )}
                </Box>
                <Typography 
                  variant="body1" 
                  sx={{ 
                    mb: 2, 
                    color: 'text.secondary',
                    fontSize: { xs: '0.9rem', sm: '1rem' },
                    lineHeight: 1.5,
                  }}
                >
                  {feature.description}
                </Typography>
                <Typography 
                  variant="body2" 
                  sx={{ 
                    mb: 3, 
                    color: feature.disabled ? 'text.disabled' : 'text.secondary',
                    fontSize: '0.85rem',
                  }}
                >
                  {feature.stats}
                </Typography>
                <Button
                  component={Link}
                  to={feature.link}
                  variant={feature.primary ? "contained" : "outlined"}
                  color={feature.color as any}
                  size="large"
                  fullWidth
                  disabled={feature.disabled}
                  sx={{ 
                    borderRadius: 2,
                    fontSize: { xs: '0.85rem', sm: '0.95rem' },
                    py: 1.5,
                  }}
                >
                  {feature.disabled ? 'Collect Data First' : `Open ${feature.title}`}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default HomePage;
