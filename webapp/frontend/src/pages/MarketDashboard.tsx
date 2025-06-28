import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Grid,
  Box,
  Card,
  CardContent,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  TrendingUp,
  Home,
  LocationOn,
  AttachMoney,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { 
  apiService, 
  MarketStats, 
  DistrictInvestmentScores, 
  ChartData, 
  ActivityData 
} from '../services/api';

const MarketDashboard: React.FC = () => {
  const [marketStats, setMarketStats] = useState<MarketStats | null>(null);
  const [districtInvestmentScores, setDistrictInvestmentScores] = useState<DistrictInvestmentScores | null>(null);
  const [chartData, setChartData] = useState<{
    renovation: ChartData | null;
    buildType: ChartData | null;
    marketSegments: ChartData | null;
    sizeAnalysis: ChartData | null;
    bargainDistribution: ChartData | null;
    activityByDay: ActivityData | null;
  }>({
    renovation: null,
    buildType: null,
    marketSegments: null,
    sizeAnalysis: null,
    bargainDistribution: null,
    activityByDay: null,
  });
  const [investmentAnalytics, setInvestmentAnalytics] = useState<{
    roiDistribution: ChartData | null;
    paybackAnalysis: ChartData | null;
    rentalYieldByDistrict: (ChartData & { avg_rent: number[], avg_price: number[] }) | null;
    investmentVsPrice: any | null;
    cashFlowDistribution: ChartData | null;
  }>({
    roiDistribution: null,
    paybackAnalysis: null,
    rentalYieldByDistrict: null,
    investmentVsPrice: null,
    cashFlowDistribution: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [stats, districtScores] = await Promise.all([
          apiService.getMarketStats(),
          apiService.getDistrictInvestmentScores()
        ]);
        setMarketStats(stats);
        setDistrictInvestmentScores(districtScores);

        // Fetch chart data
        try {
          const [
            renovationData,
            buildTypeData,
            marketSegmentsData,
            sizeAnalysisData,
            bargainDistributionData,
            activityByDayData
          ] = await Promise.all([
            apiService.getRenovationImpactData(),
            apiService.getBuildTypeData(),
            apiService.getMarketSegmentsData(),
            apiService.getSizeAnalysisData(),
            apiService.getBargainDistributionData(),
            apiService.getActivityByDayData()
          ]);

          setChartData({
            renovation: renovationData,
            buildType: buildTypeData,
            marketSegments: marketSegmentsData,
            sizeAnalysis: sizeAnalysisData,
            bargainDistribution: bargainDistributionData,
            activityByDay: activityByDayData,
          });
        
        // Fetch investment analytics
        try {
          const [
            roiData,
            paybackData,
            rentalYieldData,
            investmentVsPriceData,
            cashFlowData
          ] = await Promise.all([
            apiService.getRoiDistribution(),
            apiService.getPaybackAnalysis(),
            apiService.getRentalYieldByDistrict(),
            apiService.getInvestmentVsPrice(),
            apiService.getCashFlowDistribution()
          ]);

          setInvestmentAnalytics({
            roiDistribution: roiData,
            paybackAnalysis: paybackData,
            rentalYieldByDistrict: rentalYieldData,
            investmentVsPrice: investmentVsPriceData,
            cashFlowDistribution: cashFlowData,
          });
        } catch (investmentErr) {
          console.error('Error fetching investment analytics:', investmentErr);
          // Set default/mock data for investment analytics
          setInvestmentAnalytics({
            roiDistribution: {
              labels: ['Excellent (10%+)', 'Good (7-10%)', 'Fair (5-7%)', 'Poor (3-5%)', 'Avoid (<3%)'],
              values: [12.5, 8.5, 6.0, 4.0, 2.0],
              counts: [15, 45, 120, 80, 40]
            },
            paybackAnalysis: {
              labels: ['Quick (≤10y)', 'Moderate (10-15y)', 'Slow (15-20y)', 'Very Slow (20-30y)', 'Poor (30y+)'],
              values: [8.2, 12.5, 17.3, 25.0, 35.0],
              counts: [25, 60, 85, 70, 60]
            },
            rentalYieldByDistrict: {
              labels: ['Центр', 'Рудаки', 'Сино', 'Фирдавси', 'Исмоили Сомони'],
              values: [8.5, 7.2, 6.8, 6.5, 6.0],
              counts: [45, 32, 28, 25, 20],
              avg_rent: [850, 720, 680, 650, 600],
              avg_price: [120000, 100000, 95000, 90000, 85000]
            },
            investmentVsPrice: {
              price: [80000, 120000, 150000, 200000, 250000],
              roi: [9.5, 8.2, 7.0, 6.5, 5.8],
              yield: [9.5, 8.2, 7.0, 6.5, 5.8],
              payback: [10.5, 12.2, 14.3, 15.4, 17.2],
              district: ['Центр', 'Рудаки', 'Сино', 'Фирдавси', 'Исмоили Сомони'],
              bargain_category: ['excellent_bargain', 'good_bargain', 'fair_value', 'market_price', 'overpriced'],
              investment_category: ['excellent_investment', 'good_investment', 'fair_investment', 'poor_investment', 'avoid_investment']
            },
            cashFlowDistribution: {
              labels: ['Excellent (500+ TJS)', 'Good (300-500 TJS)', 'Moderate (100-300 TJS)', 'Break Even (0-100 TJS)', 'Negative (<0 TJS)'],
              values: [650, 400, 200, 50, -100],
              counts: [20, 55, 120, 80, 25]
            }
          });
        }
        } catch (chartErr) {
          console.error('Error fetching chart data:', chartErr);
          // Fall back to mock data if API fails
          setChartData({
            renovation: {
              labels: ['Без ремонта (коробка)', 'Новый ремонт', 'С ремонтом'],
              values: [5351, 8420, 7755],
              counts: [198, 130, 111]
            },
            buildType: {
              labels: ['Вторичный рынок', 'Новостройка'],
              values: [7780, 6516],
              counts: [122, 317]
            },
            marketSegments: {
              labels: ['Budget Market', 'Mid Market', 'Premium Market', 'Luxury Market'],
              values: [517120, 544223, 587870, 882175],
              counts: [108, 230, 23, 55]
            },
            sizeAnalysis: {
              labels: ['Compact (53m²)', 'Standard (68m²)', 'Spacious (83m²)', 'Premium (106m²)'],
              values: [8466, 8074, 6458, 6364],
              counts: [17, 100, 153, 169]
            },
            bargainDistribution: {
              labels: ['Excellent Deals', 'Good Investments', 'Fair Value', 'Market Price', 'Overpriced'],
              values: [2, 23, 185, 141, 88],
              counts: [2, 23, 185, 141, 88]
            },
            activityByDay: {
              x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
              y: [65, 72, 85, 78, 68, 60, 45]
            },
          });
        }
      } catch (err) {
        setError('Failed to load market statistics. Please try again.');
        console.error('Error fetching market stats:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'TJS',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  // Prepare chart data
  const getPriceDistributionData = () => {
    if (!marketStats?.price_distribution) return { x: [], y: [] };
    
    const labels = Object.keys(marketStats.price_distribution);
    const values = Object.values(marketStats.price_distribution);
    
    return {
      x: labels,
      y: values,
    };
  };

  const getBestInvestmentDistrictsData = () => {
    if (!districtInvestmentScores || !districtInvestmentScores.labels.length) {
      return { labels: [], values: [] };
    }
    
    return {
      labels: districtInvestmentScores.labels,
      values: districtInvestmentScores.values,
    };
  };

  const getActivityByDayData = () => {
    if (!chartData.activityByDay) {
      return {
        x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        y: [0, 0, 0, 0, 0, 0, 0]
      };
    }
    return chartData.activityByDay;
  };

  const getCityDistributionData = () => {
    if (!marketStats?.city_distribution) return { labels: [], values: [] };
    
    const labels = Object.keys(marketStats.city_distribution);
    const values = labels.map(label => marketStats.city_distribution[label]);
    
    return { labels, values };
  };

  const getInvestmentData = () => {
    if (!chartData.bargainDistribution) {
      return { labels: [], values: [] };
    }
    return {
      labels: chartData.bargainDistribution.labels,
      values: chartData.bargainDistribution.values,
    };
  };

  const getRenovationImpactData = () => {
    if (!chartData.renovation) {
      return {
        labels: ['No Data'],
        values: [0],
        counts: [0]
      };
    }
    return chartData.renovation;
  };

  const getBuildTypeData = () => {
    if (!chartData.buildType) {
      return {
        labels: ['No Data'],
        values: [0],
        counts: [0]
      };
    }
    return chartData.buildType;
  };

  const getMarketSegmentData = () => {
    if (!chartData.marketSegments) {
      return {
        labels: ['No Data'],
        values: [0],
        counts: [0]
      };
    }
    return chartData.marketSegments;
  };

  const getSizeAnalysisData = () => {
    if (!chartData.sizeAnalysis) {
      return {
        labels: ['No Data'],
        values: [0],
        counts: [0]
      };
    }
    return chartData.sizeAnalysis;
  };

  const priceDistribution = getPriceDistributionData();
  const bestInvestmentDistricts = getBestInvestmentDistrictsData();
  const activityByDay = getActivityByDayData();
  const cityDistribution = getCityDistributionData();
  const investmentData = getInvestmentData();
  const renovationData = getRenovationImpactData();
  const buildTypeData = getBuildTypeData();
  const marketSegmentData = getMarketSegmentData();
  const sizeAnalysisData = getSizeAnalysisData();

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: { xs: 2, sm: 4 } }}>
        <Typography
          variant="h2"
          sx={{
            fontWeight: 700,
            color: 'primary.main',
            mb: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: { xs: 1, sm: 2 },
            fontSize: { xs: '1.5rem', sm: '2rem', md: '3rem' },
            flexDirection: { xs: 'column', sm: 'row' }
          }}
        >
          <DashboardIcon sx={{ fontSize: { xs: 32, sm: 40, md: 48 } }} />
          Market Analytics Dashboard
        </Typography>
        <Typography 
          variant="h6" 
          color="text.secondary" 
          sx={{ 
            maxWidth: 800, 
            mx: 'auto',
            fontSize: { xs: '0.9rem', sm: '1rem', md: '1.25rem' },
            px: { xs: 2, sm: 0 }
          }}
        >
          Comprehensive market insights and trends for Tajikistan real estate
        </Typography>
      </Box>

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

      {/* Dashboard Content */}
      {marketStats && !loading && !error && (
        <>
          {/* Key Metrics */}
          <Grid container spacing={2} sx={{ mb: 4 }}>
            <Grid item xs={6} md={3}>
              <Card sx={{ 
                textAlign: 'center', 
                p: { xs: 1, sm: 2 }, 
                bgcolor: 'primary.50',
                minHeight: { xs: 120, sm: 140 }
              }}>
                <Home sx={{ fontSize: { xs: 24, sm: 32 }, color: 'primary.main', mb: 1 }} />
                <Typography 
                  variant="h5" 
                  sx={{ 
                    fontWeight: 700, 
                    color: 'primary.main',
                    fontSize: { xs: '1.1rem', sm: '1.5rem', md: '2rem' },
                    lineHeight: 1.2
                  }}
                >
                  {formatNumber(marketStats.total_listings)}
                </Typography>
                <Typography 
                  variant="body2" 
                  color="text.secondary"
                  sx={{ fontSize: { xs: '0.7rem', sm: '0.875rem' } }}
                >
                  Total Listings
                </Typography>
              </Card>
            </Grid>
            
            <Grid item xs={6} md={3}>
              <Card sx={{ 
                textAlign: 'center', 
                p: { xs: 1, sm: 2 }, 
                bgcolor: 'success.50',
                minHeight: { xs: 120, sm: 140 }
              }}>
                <AttachMoney sx={{ fontSize: { xs: 24, sm: 32 }, color: 'success.main', mb: 1 }} />
                <Typography 
                  variant="h5" 
                  sx={{ 
                    fontWeight: 700, 
                    color: 'success.main',
                    fontSize: { xs: '0.9rem', sm: '1.2rem', md: '1.5rem' },
                    lineHeight: 1.2
                  }}
                >
                  {formatPrice(marketStats.avg_price)}
                </Typography>
                <Typography 
                  variant="body2" 
                  color="text.secondary"
                  sx={{ fontSize: { xs: '0.7rem', sm: '0.875rem' } }}
                >
                  Average Price
                </Typography>
              </Card>
            </Grid>
            
            <Grid item xs={6} md={3}>
              <Card sx={{ 
                textAlign: 'center', 
                p: { xs: 1, sm: 2 }, 
                bgcolor: 'warning.50',
                minHeight: { xs: 120, sm: 140 }
              }}>
                <TrendingUp sx={{ fontSize: { xs: 24, sm: 32 }, color: 'warning.main', mb: 1 }} />
                <Typography 
                  variant="h5" 
                  sx={{ 
                    fontWeight: 700, 
                    color: 'warning.main',
                    fontSize: { xs: '1.1rem', sm: '1.5rem', md: '2rem' },
                    lineHeight: 1.2
                  }}
                >
                  {formatNumber(marketStats.total_bargains)}
                </Typography>
                <Typography 
                  variant="body2" 
                  color="text.secondary"
                  sx={{ fontSize: { xs: '0.7rem', sm: '0.875rem' } }}
                >
                  Investment Opportunities
                </Typography>
              </Card>
            </Grid>
            
            <Grid item xs={6} md={3}>
              <Card sx={{ 
                textAlign: 'center', 
                p: { xs: 1, sm: 2 }, 
                bgcolor: 'info.50',
                minHeight: { xs: 120, sm: 140 }
              }}>
                <LocationOn sx={{ fontSize: { xs: 24, sm: 32 }, color: 'info.main', mb: 1 }} />
                <Typography 
                  variant="h5" 
                  sx={{ 
                    fontWeight: 700, 
                    color: 'info.main',
                    fontSize: { xs: '0.9rem', sm: '1.2rem', md: '1.5rem' },
                    lineHeight: 1.2
                  }}
                >
                  {formatPrice(marketStats.median_price)}
                </Typography>
                <Typography 
                  variant="body2" 
                  color="text.secondary"
                  sx={{ fontSize: { xs: '0.7rem', sm: '0.875rem' } }}
                >
                  Median Price
                </Typography>
              </Card>
            </Grid>
          </Grid>

          {/* Charts */}
          <Grid container spacing={{ xs: 2, sm: 3, md: 4 }}>
            {/* Price Distribution Chart - Keep as bar chart but with better styling */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    Price Distribution
                  </Typography>
                  <Box sx={{ height: { xs: 300, sm: 400 } }}>
                    {priceDistribution.x.length > 0 ? (
                      <Plot
                        data={[
                          {
                            x: priceDistribution.x,
                            y: priceDistribution.y,
                            type: 'bar',
                            marker: {
                              color: priceDistribution.y.map((val, i) => 
                                `rgba(59, 130, 246, ${0.5 + (i / priceDistribution.y.length) * 0.5})`
                              ),
                            },
                            name: 'Properties',
                            hovertemplate: '<b>Price Range:</b> %{x}<br><b>Properties:</b> %{y}<extra></extra>',
                          },
                        ]}
                        layout={{
                          xaxis: {
                            title: { text: 'Price Range', font: { size: 10 } },
                            tickangle: -45,
                            tickfont: { size: 8 },
                          },
                          yaxis: {
                            title: { text: 'Number of Properties', font: { size: 10 } },
                            tickfont: { size: 8 },
                          },
                          margin: { t: 20, r: 20, b: 100, l: 50 },
                          font: { size: 10 },
                          bargap: 0.1,
                        }}
                        config={{
                          displayModeBar: false,
                          responsive: true,
                        }}
                        style={{ width: '100%', height: '100%' }}
                      />
                    ) : (
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                        <Typography variant="body2" color="text.secondary">
                          {loading ? 'Loading price distribution data...' : 'No price distribution data available'}
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Best Investment Districts - Change to horizontal bar chart with gradient */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    Best Investment Districts
                  </Typography>
                  <Box sx={{ height: { xs: 300, sm: 400 } }}>
                    {bestInvestmentDistricts.labels.length > 0 ? (
                      <Plot
                        data={[
                          {
                            y: bestInvestmentDistricts.labels,
                            x: bestInvestmentDistricts.values,
                            type: 'bar',
                            orientation: 'h',
                            marker: {
                              color: bestInvestmentDistricts.values,
                              colorscale: [
                                [0, '#e5f5e0'],
                                [0.5, '#74c476'],
                                [1, '#006d2c'],
                              ],
                            },
                            name: 'Investment Score',
                            hovertemplate: '<b>%{y}</b><br>Score: %{x:.2f}<extra></extra>',
                          },
                        ]}
                        layout={{
                          xaxis: {
                            title: { text: 'Investment Score (0-1)', font: { size: 10 } },
                            range: [0, Math.max(...bestInvestmentDistricts.values) * 1.1],
                            tickfont: { size: 8 },
                          },
                          yaxis: {
                            title: { text: 'District', font: { size: 10 } },
                            automargin: true,
                            tickfont: { size: 8 },
                          },
                          margin: { t: 20, r: 20, b: 40, l: 100 },
                          font: { size: 10 },
                        }}
                        config={{
                          displayModeBar: false,
                          responsive: true,
                        }}
                        style={{ width: '100%', height: '100%' }}
                      />
                    ) : (
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                        <Typography variant="body2" color="text.secondary">
                          {loading ? 'Loading district investment data...' : 'No district investment data available'}
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Investment Opportunities - Change to donut chart */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    Investment Opportunities
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Plot
                      data={[
                        {
                          labels: investmentData.labels,
                          values: investmentData.values,
                          type: 'pie',
                          hole: 0.6,
                          marker: {
                            colors: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'],
                          },
                          textinfo: 'label+percent',
                          textposition: 'outside',
                          hovertemplate: '<b>%{label}</b><br>Properties: %{value}<br>%{percent}<extra></extra>',
                        },
                      ]}
                      layout={{
                        showlegend: false,
                        margin: { t: 20, r: 20, b: 20, l: 20 },
                        font: { size: 12 },
                        annotations: [{
                          text: 'Investment<br>Distribution',
                          showarrow: false,
                          font: { size: 14 },
                        }],
                      }}
                      config={{
                        displayModeBar: false,
                        responsive: true,
                      }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Activity by Day - Change to area chart */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    Market Activity by Day
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Plot
                      data={[
                        {
                          x: activityByDay.x,
                          y: activityByDay.y,
                          type: 'scatter',
                          mode: 'lines',
                          fill: 'tozeroy',
                          line: { shape: 'spline', color: '#8b5cf6' },
                          fillcolor: 'rgba(139, 92, 246, 0.2)',
                          name: 'New Listings',
                          hovertemplate: '<b>%{x}</b><br>Listings: %{y}<extra></extra>',
                        },
                      ]}
                      layout={{
                        xaxis: {
                          title: { text: 'Day of Week' },
                        },
                        yaxis: {
                          title: { text: 'Number of Listings' },
                          rangemode: 'tozero',
                        },
                        margin: { t: 20, r: 20, b: 60, l: 60 },
                        font: { size: 12 },
                      }}
                      config={{
                        displayModeBar: false,
                        responsive: true,
                      }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Renovation Impact - Change to grouped bar chart with hover effects */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    Renovation Impact on Price
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Plot
                      data={[
                        {
                          name: 'Price per m²',
                          x: renovationData.labels,
                          y: renovationData.values,
                          type: 'bar',
                          marker: {
                            color: renovationData.values.map((_, i) => 
                              `rgba(59, 130, 246, ${0.6 + (i / renovationData.values.length) * 0.4})`
                            ),
                          },
                          text: renovationData.values.map((val, i) => 
                            `${formatPrice(val)}<br>${renovationData.counts?.[i] ?? 0} properties`
                          ),
                          textposition: 'auto',
                          hovertemplate: '<b>%{x}</b><br>Price per m²: %{y:,.0f}<br>Count: %{customdata}<extra></extra>',
                          customdata: renovationData.counts,
                        }
                      ]}
                      layout={{
                        xaxis: {
                          title: { text: 'Renovation Type' },
                          tickangle: -45,
                        },
                        yaxis: {
                          title: { text: 'Price per m² ($)' },
                        },
                        margin: { t: 20, r: 20, b: 120, l: 60 },
                        font: { size: 12 },
                        showlegend: false,
                        bargap: 0.3,
                      }}
                      config={{
                        displayModeBar: false,
                        responsive: true,
                      }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Build Type Analysis - Change to bubble chart */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    Build Type Analysis
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Plot
                      data={[
                        {
                          x: buildTypeData.labels,
                          y: buildTypeData.values,
                          type: 'bar',
                          name: 'Price per m²',
                          marker: {
                            color: ['#8b5cf6', '#10b981'],
                          },
                          text: buildTypeData.values.map((val, i) => 
                            `${formatPrice(val)}<br>${buildTypeData.counts?.[i] ?? 0} properties`
                          ),
                          textposition: 'auto',
                          hovertemplate: '<b>%{x}</b><br>Price per m²: %{y:,.0f}<br>Properties: %{customdata}<extra></extra>',
                          customdata: buildTypeData.counts,
                        }
                      ]}
                      layout={{
                        xaxis: {
                          title: { text: 'Build Type' },
                          tickangle: 0,
                        },
                        yaxis: {
                          title: { text: 'Price per m² ($)' },
                        },
                        margin: { t: 20, r: 20, b: 80, l: 60 },
                        font: { size: 12 },
                        showlegend: false,
                        bargap: 0.3,
                      }}
                      config={{
                        displayModeBar: false,
                        responsive: true,
                      }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Market Segments - Change to treemap */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    Market Segments Analysis
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Plot
                      data={[
                        {
                          type: 'treemap',
                          labels: marketSegmentData.labels,
                          parents: marketSegmentData.labels.map(() => ''),
                          values: marketSegmentData.counts,
                          textinfo: 'label+value+percent',
                          marker: {
                            colors: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'],
                          },
                          hovertemplate: '<b>%{label}</b><br>Properties: %{value}<br>Avg Price: %{customdata:,.0f}<extra></extra>',
                          customdata: marketSegmentData.values,
                        },
                      ]}
                      layout={{
                        margin: { t: 0, r: 0, b: 0, l: 0 },
                        font: { size: 12 },
                      }}
                      config={{
                        displayModeBar: false,
                        responsive: true,
                      }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Size Analysis - Change to scatter plot with trend line */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    Size vs Price Analysis
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Plot
                      data={[
                        {
                          x: sizeAnalysisData.labels.map(label => parseInt(label.match(/\d+/)?.[0] || '0')),
                          y: sizeAnalysisData.values,
                          mode: 'text+markers',
                          type: 'scatter',
                          marker: {
                            size: sizeAnalysisData.counts?.map(count => Math.sqrt(count) * 10) ?? [],
                            color: '#8b5cf6',
                            opacity: 0.6,
                          },
                          text: sizeAnalysisData.counts?.map(count => `${count}`) ?? [],
                          textposition: 'top center',
                          name: 'Properties',
                          hovertemplate: '<b>Size: %{x}m²</b><br>Price per m²: %{y:,.0f}<br>Count: %{text}<extra></extra>',
                        },
                        {
                          x: sizeAnalysisData.labels.map(label => parseInt(label.match(/\d+/)?.[0] || '0')),
                          y: sizeAnalysisData.values,
                          mode: 'lines',
                          type: 'scatter',
                          line: {
                            color: '#ef4444',
                            dash: 'dot',
                          },
                          name: 'Trend',
                        },
                      ]}
                      layout={{
                        xaxis: {
                          title: { text: 'Property Size (m²)' },
                        },
                        yaxis: {
                          title: { text: 'Price per m² ($)' },
                        },
                        margin: { t: 20, r: 20, b: 60, l: 60 },
                        font: { size: 12 },
                        legend: {
                          x: 0,
                          y: 1.1,
                          orientation: 'h',
                        },
                      }}
                      config={{
                        displayModeBar: false,
                        responsive: true,
                      }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          
          {/* Investment Analytics Charts */}
          <Typography variant="h4" sx={{ mt: 6, mb: 4, fontWeight: 700, color: 'primary.main' }}>
            💰 Investment Analytics
          </Typography>

          <Grid container spacing={4}>
            {/* ROI Distribution */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    📈 ROI Distribution
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Plot
                      data={[
                        {
                          values: investmentAnalytics.roiDistribution?.counts || [],
                          labels: investmentAnalytics.roiDistribution?.labels || [],
                          type: 'pie',
                          hole: 0.4,
                          marker: {
                            colors: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#6b7280'],
                          },
                          textinfo: 'label+percent',
                          textposition: 'outside',
                          hovertemplate: '<b>%{label}</b><br>Properties: %{value}<br>Avg ROI: %{customdata:.1f}%<extra></extra>',
                          customdata: investmentAnalytics.roiDistribution?.values || [],
                        }
                      ]}
                      layout={{
                        margin: { t: 20, r: 20, b: 20, l: 20 },
                        font: { size: 12 },
                        showlegend: true,
                        legend: {
                          orientation: 'v',
                          x: 1.02,
                          y: 0.5,
                        },
                        annotations: [
                          {
                            text: 'ROI<br>Categories',
                            x: 0.5,
                            y: 0.5,
                            font: { size: 16, color: 'rgb(107, 114, 128)' },
                            showarrow: false,
                          }
                        ],
                      }}
                      config={{
                        displayModeBar: false,
                        responsive: true,
                      }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Payback Period Analysis */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    ⏰ Payback Period Analysis
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Plot
                      data={[
                        {
                          x: investmentAnalytics.paybackAnalysis?.labels || [],
                          y: investmentAnalytics.paybackAnalysis?.counts || [],
                          type: 'bar',
                          name: 'Properties',
                          marker: {
                            color: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#6b7280'],
                          },
                          text: (investmentAnalytics.paybackAnalysis?.values || []).map((val, i) => 
                            `${val.toFixed(1)} years avg`
                          ),
                          textposition: 'auto',
                          hovertemplate: '<b>%{x}</b><br>Properties: %{y}<br>Avg Payback: %{customdata:.1f} years<extra></extra>',
                          customdata: investmentAnalytics.paybackAnalysis?.values || [],
                        }
                      ]}
                      layout={{
                        xaxis: {
                          title: { text: 'Payback Period Category' },
                          tickangle: -45,
                        },
                        yaxis: {
                          title: { text: 'Number of Properties' },
                        },
                        margin: { t: 20, r: 20, b: 100, l: 60 },
                        font: { size: 12 },
                        showlegend: false,
                        bargap: 0.3,
                      }}
                      config={{
                        displayModeBar: false,
                        responsive: true,
                      }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Rental Yield by District */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    🏘️ Rental Yield by District
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Plot
                      data={[
                        {
                          x: investmentAnalytics.rentalYieldByDistrict?.labels || [],
                          y: investmentAnalytics.rentalYieldByDistrict?.values || [],
                          type: 'bar',
                          name: 'Rental Yield',
                          marker: {
                            color: investmentAnalytics.rentalYieldByDistrict?.values?.map(val => 
                              val >= 8 ? '#10b981' : val >= 6 ? '#3b82f6' : val >= 4 ? '#f59e0b' : '#ef4444'
                            ) || [],
                          },
                          text: (investmentAnalytics.rentalYieldByDistrict?.values || []).map(val => 
                            `${val.toFixed(1)}%`
                          ),
                          textposition: 'auto',
                          hovertemplate: '<b>%{x}</b><br>Avg Yield: %{y:.1f}%<br>Properties: %{customdata}<extra></extra>',
                          customdata: investmentAnalytics.rentalYieldByDistrict?.counts || [],
                        }
                      ]}
                      layout={{
                        xaxis: {
                          title: { text: 'District' },
                          tickangle: -45,
                        },
                        yaxis: {
                          title: { text: 'Average Rental Yield (%)' },
                        },
                        margin: { t: 20, r: 20, b: 100, l: 60 },
                        font: { size: 12 },
                        showlegend: false,
                        bargap: 0.4,
                      }}
                      config={{
                        displayModeBar: false,
                        responsive: true,
                      }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Cash Flow Distribution */}
            <Grid item xs={12} lg={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    💵 Monthly Cash Flow Distribution
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Plot
                      data={[
                        {
                          x: investmentAnalytics.cashFlowDistribution?.labels || [],
                          y: investmentAnalytics.cashFlowDistribution?.counts || [],
                          type: 'bar',
                          name: 'Properties',
                          marker: {
                            color: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#6b7280'],
                          },
                          text: (investmentAnalytics.cashFlowDistribution?.values || []).map(val => 
                            `${val.toFixed(0)} TJS avg`
                          ),
                          textposition: 'auto',
                          hovertemplate: '<b>%{x}</b><br>Properties: %{y}<br>Avg Cash Flow: %{customdata:.0f} TJS<extra></extra>',
                          customdata: investmentAnalytics.cashFlowDistribution?.values || [],
                        }
                      ]}
                      layout={{
                        xaxis: {
                          title: { text: 'Cash Flow Category' },
                          tickangle: -45,
                        },
                        yaxis: {
                          title: { text: 'Number of Properties' },
                        },
                        margin: { t: 20, r: 20, b: 120, l: 60 },
                        font: { size: 12 },
                        showlegend: false,
                        bargap: 0.3,
                      }}
                      config={{
                        displayModeBar: false,
                        responsive: true,
                      }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Investment Score vs Price Scatter Plot */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    🎯 Investment Opportunities: ROI vs Price Analysis
                  </Typography>
                  <Box sx={{ height: 500 }}>
                    <Plot
                      data={[
                        {
                          x: investmentAnalytics.investmentVsPrice?.price || [],
                          y: investmentAnalytics.investmentVsPrice?.roi || [],
                          mode: 'markers',
                          type: 'scatter',
                          marker: {
                            size: (investmentAnalytics.investmentVsPrice?.payback as number[] || []).map(payback => 
                              Math.max(6, Math.min(20, 25 - payback))
                            ),
                            color: (investmentAnalytics.investmentVsPrice?.investment_category as string[] || []).map(cat => {
                              switch(cat) {
                                case 'excellent_investment': return '#10b981';
                                case 'good_investment': return '#3b82f6';
                                case 'fair_investment': return '#f59e0b';
                                case 'poor_investment': return '#ef4444';
                                default: return '#6b7280';
                              }
                            }),
                            opacity: 0.7,
                            line: { width: 2, color: 'white' },
                          },
                          text: (investmentAnalytics.investmentVsPrice?.district as string[] || []).map((district, i) => 
                            `${district}<br>Payback: ${(investmentAnalytics.investmentVsPrice?.payback || [])[i]?.toFixed(1)}y`
                          ),
                          hovertemplate: '<b>%{text}</b><br>Price: %{x:,.0f} TJS<br>ROI: %{y:.1f}%<br>Investment Category: %{customdata}<extra></extra>',
                          customdata: investmentAnalytics.investmentVsPrice?.investment_category || [],
                          name: 'Properties',
                        }
                      ]}
                      layout={{
                        xaxis: {
                          title: { text: 'Property Price (TJS)' },
                          type: 'log',
                        },
                        yaxis: {
                          title: { text: 'ROI (%)' },
                        },
                        margin: { t: 20, r: 20, b: 60, l: 60 },
                        font: { size: 12 },
                        showlegend: false,
                        hovermode: 'closest',
                        annotations: [
                          {
                            text: 'Bubble size = Payback period<br>Color = Investment category',
                            x: 0.02,
                            y: 0.98,
                            xref: 'paper',
                            yref: 'paper',
                            font: { size: 10, color: 'rgb(107, 114, 128)' },
                            showarrow: false,
                            align: 'left',
                          }
                        ],
                      }}
                      config={{
                        displayModeBar: false,
                        responsive: true,
                      }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>


        </>
      )}
    </Container>
  );
};

export default MarketDashboard;
