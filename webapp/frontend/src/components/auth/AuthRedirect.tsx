import React, { useEffect } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../../services/auth';
import { Box, CircularProgress, Typography } from '@mui/material';

const AuthRedirect: React.FC = () => {
  const { isAuthenticated, isLoading, user } = useAuth();

  // Show loading while checking authentication
  if (isLoading) {
    return (
      <Box 
        sx={{ 
          display: 'flex', 
          flexDirection: 'column',
          alignItems: 'center', 
          justifyContent: 'center', 
          minHeight: '100vh',
          gap: 2
        }}
      >
        <CircularProgress size={60} />
        <Typography variant="body1" color="text.secondary">
          Checking authentication...
        </Typography>
      </Box>
    );
  }

  // If authenticated, redirect to home
  if (isAuthenticated && user) {
    return <Navigate to="/home" replace />;
  }

  // If not authenticated, redirect to login
  return <Navigate to="/login" replace />;
};

export default AuthRedirect; 