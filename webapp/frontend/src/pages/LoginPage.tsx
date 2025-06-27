import React from 'react';
import { Container, Paper, Box, Typography } from '@mui/material';
import { LoginForm } from '../components/auth';

const LoginPage: React.FC = () => {
  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <Typography component="h1" variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
            Login to Your Account
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 4, textAlign: 'center' }}>
            Access your property search history and personalized recommendations
          </Typography>
          <LoginForm />
        </Box>
      </Paper>
    </Container>
  );
};

export default LoginPage;
