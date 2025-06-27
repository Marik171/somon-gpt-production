import React from 'react';
import { 
  Container, 
  Paper, 
  Box, 
  Typography, 
  Card, 
  CardContent,
  Chip,
  Divider
} from '@mui/material';
import { 
  Email as EmailIcon,
  Person as PersonIcon,
  CalendarToday as CalendarIcon,
  Notifications as NotificationIcon
} from '@mui/icons-material';
import { useAuth } from '../services/auth';
import { ProtectedRoute } from '../components/auth';

const ProfilePage: React.FC = () => {
  const { user } = useAuth();

  if (!user) return null;

  return (
    <ProtectedRoute>
      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 600, mb: 4 }}>
          Profile Settings
        </Typography>

        <Card elevation={2}>
          <CardContent sx={{ p: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Box
                sx={{
                  width: 64,
                  height: 64,
                  borderRadius: '50%',
                  bgcolor: 'primary.main',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mr: 3,
                }}
              >
                <Typography variant="h4" color="white" sx={{ fontWeight: 'bold' }}>
                  {user.full_name?.charAt(0) || user.email?.charAt(0) || 'U'}
                </Typography>
              </Box>
              <Box>
                <Typography variant="h5" sx={{ fontWeight: 600 }}>
                  {user.full_name || 'User'}
                </Typography>
                <Chip 
                  label={user.is_active ? 'Active' : 'Inactive'} 
                  color={user.is_active ? 'success' : 'default'}
                  size="small"
                />
              </Box>
            </Box>

            <Divider sx={{ my: 3 }} />

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <EmailIcon color="primary" />
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">
                    Email Address
                  </Typography>
                  <Typography variant="body1">{user.email}</Typography>
                </Box>
              </Box>

              {user.username && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <PersonIcon color="primary" />
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary">
                      Username
                    </Typography>
                    <Typography variant="body1">{user.username}</Typography>
                  </Box>
                </Box>
              )}

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <CalendarIcon color="primary" />
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">
                    Member Since
                  </Typography>
                  <Typography variant="body1">
                    {new Date(user.created_at).toLocaleDateString()}
                  </Typography>
                </Box>
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <NotificationIcon color="primary" />
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">
                    Notification Preferences
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                    <Chip 
                      label="Email Notifications" 
                      variant={user.email_notifications ? 'filled' : 'outlined'}
                      color={user.email_notifications ? 'primary' : 'default'}
                      size="small"
                    />
                    <Chip 
                      label="Push Notifications" 
                      variant={user.push_notifications ? 'filled' : 'outlined'}
                      color={user.push_notifications ? 'primary' : 'default'}
                      size="small"
                    />
                  </Box>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    Frequency: {user.notification_frequency}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Container>
    </ProtectedRoute>
  );
};

export default ProfilePage;
