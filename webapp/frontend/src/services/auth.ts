import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Types
export interface User {
  id: number;
  email: string;
  username?: string;
  full_name?: string;
  is_active: boolean;
  has_collected_data: boolean;  // Track if user has collected data
  created_at: string;
  email_notifications: boolean;
  push_notifications: boolean;
  notification_frequency: string;
}

export interface LoginRequest {
  username: string; // Actually email
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  full_name?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface ApiResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
}

class AuthService {
  private tokenKey = 'real_estate_token';
  private userKey = 'real_estate_user';
  private isRedirecting = false; // Prevent multiple redirects

  constructor() {
    // Set up axios interceptor for adding auth token
    axios.interceptors.request.use(
      (config) => {
        const token = this.getToken();
        if (token && config.url?.includes(API_BASE_URL)) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for handling auth errors
    axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401 && !this.isRedirecting) {
          this.isRedirecting = true;
          this.logout();
          
          // Use React Router navigation instead of window.location.href
          // Only redirect if not already on login page
          if (window.location.pathname !== '/login') {
            setTimeout(() => {
              window.location.href = '/login';
              this.isRedirecting = false;
            }, 100);
          } else {
            this.isRedirecting = false;
          }
        }
        return Promise.reject(error);
      }
    );
  }

  async login(email: string, password: string): Promise<AuthResponse> {
    try {
      const params = new URLSearchParams();
      params.append('username', email); // Backend expects username field
      params.append('password', password);

      const response = await axios.post<AuthResponse>(`${API_BASE_URL}/auth/token`, params, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      const { access_token, user } = response.data;
      
      // Store token and user info
      localStorage.setItem(this.tokenKey, access_token);
      localStorage.setItem(this.userKey, JSON.stringify(user));

      return response.data;
    } catch (error: any) {
      console.error('Login error:', error);
      throw new Error(error.response?.data?.detail || 'Login failed');
    }
  }

  async register(userData: RegisterRequest): Promise<ApiResponse> {
    try {
      const response = await axios.post<ApiResponse>(`${API_BASE_URL}/auth/register`, userData);
      return response.data;
    } catch (error: any) {
      console.error('Registration error:', error);
      throw new Error(error.response?.data?.detail || 'Registration failed');
    }
  }

  async getCurrentUser(): Promise<User> {
    try {
      const response = await axios.get<User>(`${API_BASE_URL}/auth/me`);
      
      // Update stored user info
      localStorage.setItem(this.userKey, JSON.stringify(response.data));
      
      return response.data;
    } catch (error: any) {
      console.error('Get current user error:', error);
      throw new Error(error.response?.data?.detail || 'Failed to get user info');
    }
  }

  logout(): void {
    try {
      localStorage.removeItem(this.tokenKey);
      localStorage.removeItem(this.userKey);
    } catch (error) {
      console.error('Error clearing localStorage during logout:', error);
    }
  }

  getToken(): string | null {
    try {
      return localStorage.getItem(this.tokenKey);
    } catch (error) {
      console.error('Error accessing localStorage for token:', error);
      return null;
    }
  }

  getUser(): User | null {
    try {
      const userStr = localStorage.getItem(this.userKey);
      if (userStr) {
        try {
          return JSON.parse(userStr);
        } catch (error) {
          console.error('Error parsing user data:', error);
          // Clear corrupted data
          this.logout();
          return null;
        }
      }
      return null;
    } catch (error) {
      console.error('Error accessing localStorage for user:', error);
      return null;
    }
  }

  isAuthenticated(): boolean {
    return !!this.getToken() && !this.isTokenExpired();
  }

  isTokenExpired(): boolean {
    const token = this.getToken();
    if (!token) return true;

    try {
      // Decode JWT token (basic check)
      const payload = JSON.parse(atob(token.split('.')[1]));
      const currentTime = Date.now() / 1000;
      return payload.exp < currentTime;
    } catch (error) {
      return true;
    }
  }

  async refreshUserIfNeeded(): Promise<void> {
    if (this.isAuthenticated() && !this.isTokenExpired()) {
      try {
        await this.getCurrentUser();
      } catch (error) {
        console.error('Failed to refresh user:', error);
        this.logout();
      }
    }
  }
}

// Export singleton instance
export const authService = new AuthService();

// React hook for authentication
import { useState, useEffect, useCallback } from 'react';

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Stable check auth function to prevent infinite loops
  const checkAuth = useCallback(async () => {
    try {
      setIsLoading(true);
      
      const token = authService.getToken();
      const storedUser = authService.getUser();
      
      if (token && !authService.isTokenExpired() && storedUser) {
        // Try to verify token with backend
        try {
          const currentUser = await authService.getCurrentUser();
          setUser(currentUser);
          setIsAuthenticated(true);
        } catch (error) {
          // Token is invalid, clear auth state
          authService.logout();
          setUser(null);
          setIsAuthenticated(false);
        }
      } else {
        // No valid token, clear auth state
        setUser(null);
        setIsAuthenticated(false);
      }
    } catch (error) {
      console.error('Auth check error:', error);
      setUser(null);
      setIsAuthenticated(false);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    checkAuth();

    // Listen for storage changes (e.g., login in another tab)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'real_estate_token' || e.key === 'real_estate_user') {
        checkAuth();
      }
    };

    window.addEventListener('storage', handleStorageChange);
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [checkAuth]);

  const login = async (email: string, password: string) => {
    setIsLoading(true);
    try {
      const response = await authService.login(email, password);
      setUser(response.user);
      setIsAuthenticated(true);
      return response;
    } catch (error) {
      setUser(null);
      setIsAuthenticated(false);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (userData: RegisterRequest) => {
    setIsLoading(true);
    try {
      const response = await authService.register(userData);
      return response;
    } catch (error) {
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    authService.logout();
    setUser(null);
    setIsAuthenticated(false);
  };

  return {
    user,
    isAuthenticated,
    isLoading,
    login,
    register,
    logout,
  };
}
