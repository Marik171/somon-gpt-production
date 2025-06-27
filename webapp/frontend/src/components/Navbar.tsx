import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
  useTheme,
  useMediaQuery,
  IconButton,
  Menu,
  MenuItem,
  Avatar,
  Divider,
} from '@mui/material';
import {
  Home as HomeIcon,
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  Dashboard as DashboardIcon,
  PsychologyAlt as PredictIcon,
  CloudDownload as CollectIcon,
  Menu as MenuIcon,
  Login as LoginIcon,
  PersonAdd as RegisterIcon,
  AccountCircle as AccountIcon,
  Favorite as FavoriteIcon,
  Logout as LogoutIcon,
} from '@mui/icons-material';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../services/auth';

const Navbar: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const location = useLocation();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [userMenuAnchor, setUserMenuAnchor] = React.useState<null | HTMLElement>(null);
  const { user, isAuthenticated, logout } = useAuth();

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const handleLogout = () => {
    logout();
    handleUserMenuClose();
  };

  const navItems = [
    { label: 'Home', path: '/home', icon: <HomeIcon /> },
    { label: 'Data Collection', path: '/collect', icon: <CollectIcon /> },
    { label: 'Search Properties', path: '/search', icon: <SearchIcon /> },
    { label: 'Bargain Finder', path: '/bargains', icon: <TrendingUpIcon /> },
    { label: 'Market Dashboard', path: '/dashboard', icon: <DashboardIcon /> },
    { label: 'Price Predictor', path: '/predict', icon: <PredictIcon /> },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <AppBar position="sticky" elevation={2} sx={{ bgcolor: 'white', color: 'text.primary' }}>
      <Container maxWidth="xl">
        <Toolbar disableGutters>          {/* Logo */}
          <Box sx={{ display: 'flex', alignItems: 'center', mr: 4 }}>
            <TrendingUpIcon sx={{ color: 'primary.main', mr: 1, fontSize: 28 }} />
            {isAuthenticated ? (
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 700,
                  color: 'primary.main',
                  fontSize: { xs: '1.1rem', md: '1.25rem' },
                  cursor: 'default',
                }}
              >
                Real Estate Intelligence
              </Typography>
            ) : (
              <Typography
                variant="h6"
                component={Link}
                to="/"
                sx={{
                  fontWeight: 700,
                  color: 'primary.main',
                  textDecoration: 'none',
                  fontSize: { xs: '1.1rem', md: '1.25rem' },
                }}
              >
                Real Estate Intelligence
              </Typography>
            )}
          </Box>

          {/* Desktop Navigation - Only show for authenticated users */}
          {!isMobile && isAuthenticated && (
            <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center' }}>
              {navItems.map((item) => (
                <Button
                  key={item.path}
                  component={Link}
                  to={item.path}
                  startIcon={item.icon}
                  sx={{
                    mx: 1,
                    color: isActive(item.path) ? 'primary.main' : 'text.primary',
                    fontWeight: isActive(item.path) ? 600 : 400,
                    backgroundColor: isActive(item.path) ? 'primary.50' : 'transparent',
                    '&:hover': {
                      backgroundColor: 'primary.50',
                      color: 'primary.main',
                    },
                    borderRadius: 2,
                    px: 2,
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </Box>
          )}

          {/* Add flex-grow for unauthenticated users to push login buttons to the right */}
          {!isAuthenticated && <Box sx={{ flexGrow: 1 }} />}

          {/* Mobile Navigation - Only show for authenticated users */}
          {isMobile && isAuthenticated && (
            <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'flex-end' }}>
              <IconButton
                size="large"
                aria-label="menu"
                aria-controls="menu-appbar"
                aria-haspopup="true"
                onClick={handleMenuOpen}
                color="inherit"
              >
                <MenuIcon />
              </IconButton>
              <Menu
                id="menu-appbar"
                anchorEl={anchorEl}
                anchorOrigin={{
                  vertical: 'bottom',
                  horizontal: 'right',
                }}
                keepMounted
                transformOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
              >
                {navItems.map((item) => (
                  <MenuItem
                    key={item.path}
                    component={Link}
                    to={item.path}
                    onClick={handleMenuClose}
                    sx={{
                      color: isActive(item.path) ? 'primary.main' : 'text.primary',
                      fontWeight: isActive(item.path) ? 600 : 400,
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {item.icon}
                      {item.label}
                    </Box>
                  </MenuItem>
                ))}
              </Menu>
            </Box>
          )}

          {/* Add flex-grow for unauthenticated mobile users */}
          {isMobile && !isAuthenticated && <Box sx={{ flexGrow: 1 }} />}

          {/* User Actions */}
          <Box sx={{ ml: 2 }}>
            {isAuthenticated ? (
              <>
                <IconButton
                  size="large"
                  edge="end"
                  aria-label="account of current user"
                  aria-controls="user-menu"
                  aria-haspopup="true"
                  onClick={handleUserMenuOpen}
                  color="inherit"
                >
                  <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                    {user?.full_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
                  </Avatar>
                </IconButton>
                <Menu
                  id="user-menu"
                  anchorEl={userMenuAnchor}
                  anchorOrigin={{
                    vertical: 'bottom',
                    horizontal: 'right',
                  }}
                  keepMounted
                  transformOrigin={{
                    vertical: 'top',
                    horizontal: 'right',
                  }}
                  open={Boolean(userMenuAnchor)}
                  onClose={handleUserMenuClose}
                >
                  <MenuItem disabled>
                    <Box>
                      <Typography variant="subtitle2">{user?.full_name || 'User'}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {user?.email}
                      </Typography>
                    </Box>
                  </MenuItem>
                  <Divider />
                  <MenuItem component={Link} to="/history" onClick={handleUserMenuClose}>
                    <FavoriteIcon sx={{ mr: 1 }} />
                    Favorites
                  </MenuItem>
                  <MenuItem component={Link} to="/profile" onClick={handleUserMenuClose}>
                    <AccountIcon sx={{ mr: 1 }} />
                    Profile
                  </MenuItem>
                  <Divider />
                  <MenuItem onClick={handleLogout}>
                    <LogoutIcon sx={{ mr: 1 }} />
                    Logout
                  </MenuItem>
                </Menu>
              </>
            ) : (
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  component={Link}
                  to="/login"
                  variant="text"
                  startIcon={<LoginIcon />}
                  sx={{ color: 'text.primary' }}
                >
                  Login
                </Button>
                <Button
                  component={Link}
                  to="/register"
                  variant="contained"
                  startIcon={<RegisterIcon />}
                >
                  Register
                </Button>
              </Box>
            )}
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Navbar;
