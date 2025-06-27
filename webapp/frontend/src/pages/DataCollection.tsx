import React, { useState } from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  CircularProgress,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Stepper,
  Step,
  StepLabel,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Search as SearchIcon,
  Download as DownloadIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { apiService } from '../services/api';

interface ScrapingRequest {
  rooms: string;
  city: string;
  build_state: string;
  property_type?: string;
}

interface ScrapingResponse {
  status: string;
  message: string;
  task_id?: string;
  total_scraped?: number;
  processing_status: string;
  files_created?: string[];
  data?: any; // Full pipeline response data
}

const DataCollection: React.FC = () => {
  const [formData, setFormData] = useState<ScrapingRequest>({
    rooms: '3-komnatnyie',
    city: 'hudzhand',
    build_state: 'sostoyanie---10',
    property_type: ''
  });

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ScrapingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const roomOptions = [
    { value: '1-komnatnyie', label: '1-bedroom' },
    { value: '2-komnatnyie', label: '2-bedroom' },
    { value: '3-komnatnyie', label: '3-bedroom' },
    { value: '4-komnatnyie', label: '4-bedroom' },
    { value: '5-komnatnyie', label: '5-bedroom' }
  ];

  const cityOptions = [
    { value: 'hudzhand', label: 'Khujand' },
    { value: 'dushanbe', label: 'Dushanbe' },
    { value: 'kulob', label: 'Kulob' },
    { value: 'qurghonteppa', label: 'Qurghonteppa' },
    { value: 'istiklol', label: 'Istiklol' }
  ];

  const buildStateOptions = [
    { value: 'sostoyanie---10', label: 'Completed' },
    { value: 'sostoyanie---11', label: 'Under Construction' }
  ];

  const propertyTypeOptions = [
    { value: '', label: 'All Types' },
    { value: 'type---1', label: 'Secondary Market' },
    { value: 'type---2', label: 'New Construction' }
  ];

  const getStepStatus = (step: string) => {
    if (!result) return 'pending';
    
    // Handle full pipeline response structure
    if (result.data && typeof result.data === 'object') {
      const pipelineData = result.data as any;
      
      switch (step) {
        case 'scraping':
          if (pipelineData.scraping) {
            return pipelineData.scraping.success ? 'completed' : 'error';
          }
          return isLoading ? 'active' : 'pending';
        
        case 'preprocessing':
          if (pipelineData.preprocessing) {
            return pipelineData.preprocessing.success ? 'completed' : 'error';
          }
          return pipelineData.scraping?.success ? 'active' : 'pending';
        
        case 'feature_engineering':
          if (pipelineData.feature_engineering) {
            return pipelineData.feature_engineering.success ? 'completed' : 'error';
          }
          return pipelineData.preprocessing?.success ? 'active' : 'pending';
        
        case 'database_import':
          if (pipelineData.database_import) {
            return pipelineData.database_import.success ? 'completed' : 'error';
          }
          return pipelineData.feature_engineering?.success ? 'active' : 'pending';
        
        default:
          return 'pending';
      }
    }
    
    // Fallback to original logic for backward compatibility
    switch (step) {
      case 'scraping':
        return result.processing_status === 'failed' ? 'error' : 
               result.processing_status === 'scraping_complete' || 
               result.processing_status === 'preprocessing_complete' ||
               result.processing_status === 'feature_engineering_complete' ||
               result.processing_status === 'complete' ? 'completed' : 'active';
      
      case 'preprocessing':
        return result.processing_status === 'failed' ? 'error' :
               result.processing_status === 'preprocessing_complete' ||
               result.processing_status === 'feature_engineering_complete' ||
               result.processing_status === 'complete' ? 'completed' :
               result.processing_status === 'scraping_complete' ? 'active' : 'pending';
      
      case 'feature_engineering':
        return result.processing_status === 'failed' ? 'error' :
               result.processing_status === 'complete' ? 'completed' :
               result.processing_status === 'feature_engineering_complete' ? 'completed' :
               result.processing_status === 'preprocessing_complete' ? 'active' : 'pending';
      
      default:
        return 'pending';
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log('🔍 DataCollection: Starting data collection with params:', formData);
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data: ScrapingResponse = await apiService.runFullPipeline(formData);
      console.log('✅ DataCollection: Full pipeline response received:', data);
      setResult(data);
      
      if (data.status === 'failed') {
        setError(data.message);
      }
    } catch (err: any) {
      console.error('❌ DataCollection: Error during scraping:', err);
      setError(err.response?.data?.detail || err.message || 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (field: keyof ScrapingRequest, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Real Estate Data Collection
        </Typography>
        
        <Typography variant="body1" color="text.secondary" paragraph>
          Start by defining your search criteria, then the system will automatically collect real estate data, 
          process it, and prepare it for analysis.
        </Typography>

        <Paper sx={{ p: 3, mb: 3 }}>
          <form onSubmit={handleSubmit}>
            <Typography variant="h6" gutterBottom>
              Search Criteria
            </Typography>
            
            <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' } }}>
              <FormControl fullWidth>
                <InputLabel>Number of Rooms</InputLabel>
                <Select
                  value={formData.rooms}
                  label="Number of Rooms"
                  onChange={(e) => handleInputChange('rooms', e.target.value)}
                >
                  {roomOptions.map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl fullWidth>
                <InputLabel>City</InputLabel>
                <Select
                  value={formData.city}
                  label="City"
                  onChange={(e) => handleInputChange('city', e.target.value)}
                >
                  {cityOptions.map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl fullWidth>
                <InputLabel>Construction Status</InputLabel>
                <Select
                  value={formData.build_state}
                  label="Construction Status"
                  onChange={(e) => handleInputChange('build_state', e.target.value)}
                >
                  {buildStateOptions.map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl fullWidth>
                <InputLabel>Property Type</InputLabel>
                <Select
                  value={formData.property_type || ''}
                  label="Property Type"
                  onChange={(e) => handleInputChange('property_type', e.target.value)}
                >
                  {propertyTypeOptions.map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            <Box sx={{ mt: 3 }}>
              <Button
                type="submit"
                variant="contained"
                size="large"
                disabled={isLoading}
                startIcon={isLoading ? <CircularProgress size={20} /> : <SearchIcon />}
                fullWidth
              >
                {isLoading ? 'Collecting Data...' : 'Start Data Collection'}
              </Button>
            </Box>
          </form>
        </Paper>

        {/* Progress Steps */}
        {(isLoading || result) && (
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Data Processing Pipeline
            </Typography>
            
            <Stepper orientation="vertical">
              <Step 
                active={getStepStatus('scraping') === 'active'} 
                completed={getStepStatus('scraping') === 'completed'}
              >
                <StepLabel 
                  error={getStepStatus('scraping') === 'error'}
                  icon={getStepStatus('scraping') === 'completed' ? <CheckCircleIcon /> : undefined}
                >
                  Data Collection from Website
                </StepLabel>
              </Step>
              
              <Step 
                active={getStepStatus('preprocessing') === 'active'} 
                completed={getStepStatus('preprocessing') === 'completed'}
              >
                <StepLabel 
                  error={getStepStatus('preprocessing') === 'error'}
                  icon={getStepStatus('preprocessing') === 'completed' ? <CheckCircleIcon /> : undefined}
                >
                  Data Preprocessing
                </StepLabel>
              </Step>
              
              <Step 
                active={getStepStatus('feature_engineering') === 'active'} 
                completed={getStepStatus('feature_engineering') === 'completed'}
              >
                <StepLabel 
                  error={getStepStatus('feature_engineering') === 'error'}
                  icon={getStepStatus('feature_engineering') === 'completed' ? <CheckCircleIcon /> : undefined}
                >
                  Feature Engineering
                </StepLabel>
              </Step>
              
              <Step 
                active={getStepStatus('database_import') === 'active'} 
                completed={getStepStatus('database_import') === 'completed'}
              >
                <StepLabel 
                  error={getStepStatus('database_import') === 'error'}
                  icon={getStepStatus('database_import') === 'completed' ? <CheckCircleIcon /> : undefined}
                >
                  Database Import
                </StepLabel>
              </Step>
            </Stepper>
            
            {isLoading && (
              <Box sx={{ mt: 2 }}>
                <LinearProgress />
              </Box>
            )}
          </Paper>
        )}

        {/* Results */}
        {result && (
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Results
            </Typography>
            
            {result.status === 'success' ? (
              <Alert severity="success" sx={{ mb: 2 }}>
                {result.message}
              </Alert>
            ) : (
              <Alert severity="error" sx={{ mb: 2 }}>
                {result.message}
              </Alert>
            )}

            {result.total_scraped !== undefined && result.total_scraped !== null && (
              <Box sx={{ mb: 2 }}>
                <Chip 
                  icon={<InfoIcon />}
                  label={`Properties Collected: ${result.total_scraped}`}
                  color="primary"
                  variant="outlined"
                />
              </Box>
            )}

            {result.files_created && result.files_created.length > 0 && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Generated Files ({result.files_created.length})</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <List dense>
                    {result.files_created?.map((file, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <DownloadIcon />
                        </ListItemIcon>
                        <ListItemText 
                          primary={file}
                          secondary={`File ${index + 1} of ${result.files_created?.length || 0}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>
            )}
          </Paper>
        )}

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* Info Panel */}
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Process Information
          </Typography>
          
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>What happens during data collection?</Typography>
            </AccordionSummary>
            <AccordionDetails>
                              <Typography component="div" variant="body2">
                  <ol>
                    <li><strong>Data Collection</strong> - The system automatically collects real estate listings from Somon.tj based on your specified criteria</li>
                    <li><strong>Data Preprocessing</strong> - Data cleaning, normalization, and duplicate removal</li>
                    <li><strong>Feature Engineering</strong> - Creation of additional characteristics for analysis and prediction</li>
                    <li><strong>Ready for Analysis</strong> - After completion, data will be available for search and analysis</li>
                  </ol>
                </Typography>
            </AccordionDetails>
          </Accordion>
          
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>What data is collected?</Typography>
            </AccordionSummary>
            <AccordionDetails>
                              <Typography variant="body2">
                  • Price and price per square meter<br/>
                  • Number of rooms and total area<br/>
                  • Floor and total number of floors<br/>
                  • District and address<br/>
                  • Construction type and condition<br/>
                  • Property photos<br/>
                  • Additional characteristics
                </Typography>
            </AccordionDetails>
          </Accordion>
        </Paper>
      </Box>
    </Container>
  );
};

export default DataCollection;
