// Production API configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? process.env.REACT_APP_API_URL || 'https://web-production-2ed0f.up.railway.app'
  : 'http://localhost:8000';

// Debug: Log the API URL being used
console.log('API_BASE_URL:', API_BASE_URL);
console.log('NODE_ENV:', process.env.NODE_ENV);
console.log('REACT_APP_API_URL:', process.env.REACT_APP_API_URL);

export default API_BASE_URL;
