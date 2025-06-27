import React from 'react';
import { Box, Container, Typography, Link, Divider } from '@mui/material';

const Footer: React.FC = () => {
  return (
    <Box
      component="footer"
      sx={{
        bgcolor: 'grey.900',
        color: 'white',
        py: 4,
        mt: 8,
      }}
    >
      <Container maxWidth="lg">
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            justifyContent: 'space-between',
            alignItems: { xs: 'center', md: 'flex-start' },
            gap: 4,
          }}
        >
          {/* Brand Section */}
          <Box sx={{ textAlign: { xs: 'center', md: 'left' } }}>
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
              Real Estate Intelligence Platform
            </Typography>
            <Typography variant="body2" sx={{ color: 'grey.400', maxWidth: 300 }}>
              AI-powered property analysis and investment intelligence for the Tajikistan real estate market.
            </Typography>
          </Box>

          {/* Features Section */}
          <Box sx={{ textAlign: { xs: 'center', md: 'left' } }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
              Features
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Link href="/search" color="grey.400" underline="hover">
                Property Search
              </Link>
              <Link href="/bargains" color="grey.400" underline="hover">
                Investment Bargains
              </Link>
              <Link href="/predict" color="grey.400" underline="hover">
                AI Price Prediction
              </Link>
              <Link href="/dashboard" color="grey.400" underline="hover">
                Market Analytics
              </Link>
            </Box>
          </Box>
        </Box>

        <Divider sx={{ my: 3, bgcolor: 'grey.700' }} />

        {/* Bottom Section */}
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            justifyContent: 'center',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <Typography variant="body2" color="grey.400">
            © 2025 Real Estate Intelligence Platform. Built for Tajikistan market analysis.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;
