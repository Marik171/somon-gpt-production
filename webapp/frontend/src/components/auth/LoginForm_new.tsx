import React, { useState } from 'react';
import { useAuth } from '../../services/auth';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  TextField,
  Button,
  Typography,
  Alert,
  Paper,
  CircularProgress,
} from '@mui/material';

interface LoginFormProps {
  onSuccess?: () => void;
}

const LoginForm: React.FC<LoginFormProps> = ({ onSuccess }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { login, isLoading } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      await login(email, password);
      if (onSuccess) {
        onSuccess();
      } else {
        navigate('/home'); // Redirect to home page instead of dashboard
      }
    } catch (error: any) {
      setError(error.message || 'Login failed');
    }
  };

  return (
    <Paper
      elevation={3}
      sx={{
        maxWidth: 400,
        mx: 'auto',
        mt: 4,
        p: 4,
        borderRadius: 2,
      }}
    >
      <Typography
        variant="h4"
        component="h2"
        sx={{
          textAlign: 'center',
          mb: 3,
          fontWeight: 600,
          color: 'primary.main',
        }}
      >
        Login
      </Typography>
      
      <Box component="form" onSubmit={handleSubmit} sx={{ width: '100%' }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <TextField
          fullWidth
          type="email"
          label="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          sx={{ mb: 2 }}
          placeholder="Enter your email"
        />
        
        <TextField
          fullWidth
          type="password"
          label="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          sx={{ mb: 3 }}
          placeholder="Enter your password"
        />
        
        <Button
          type="submit"
          fullWidth
          variant="contained"
          size="large"
          disabled={isLoading || !email || !password}
          sx={{
            py: 1.5,
            borderRadius: 2,
            textTransform: 'none',
            fontSize: '1rem',
            fontWeight: 500,
          }}
        >
          {isLoading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={20} color="inherit" />
              Logging in...
            </Box>
          ) : (
            'Login'
          )}
        </Button>
      </Box>
      
      <Typography
        variant="body2"
        sx={{
          mt: 3,
          textAlign: 'center',
          color: 'text.secondary',
        }}
      >
        Don't have an account?{' '}
        <Button
          variant="text"
          onClick={() => navigate('/register')}
          sx={{
            textTransform: 'none',
            fontWeight: 500,
            p: 0,
            minWidth: 'auto',
          }}
        >
          Register here
        </Button>
      </Typography>
    </Paper>
  );
};

export default LoginForm;
