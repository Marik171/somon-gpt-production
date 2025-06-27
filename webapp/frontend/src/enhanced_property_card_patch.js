// Enhanced PropertyCard investment metrics section
const enhancedInvestmentMetrics = `
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
              
              {property.payback_period_years && (
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
              
              {property.monthly_cash_flow && (
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

            {/* Investment Category Badge */}
            {property.investment_category && (
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                <Chip
                  label={property.investment_category.replace('_', ' ').toUpperCase()}
                  size="small"
                  color={
                    property.investment_category === 'excellent_investment' ? 'success' :
                    property.investment_category === 'good_investment' ? 'primary' :
                    property.investment_category === 'fair_investment' ? 'warning' :
                    'default'
                  }
                  sx={{ 
                    fontWeight: 600, 
                    fontSize: '0.7rem',
                    textTransform: 'capitalize'
                  }}
                />
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
        )}`;

console.log('Enhanced investment metrics section created');
