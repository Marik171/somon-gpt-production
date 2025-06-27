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
    { value: '1-komnatnyie', label: '1-комнатные' },
    { value: '2-komnatnyie', label: '2-комнатные' },
    { value: '3-komnatnyie', label: '3-комнатные' },
    { value: '4-komnatnyie', label: '4-комнатные' },
    { value: '5-komnatnyie', label: '5-комнатные' }
  ];

  const cityOptions = [
    { value: 'hudzhand', label: 'Худжанд' },
    { value: 'dushanbe', label: 'Душанбе' },
    { value: 'kulob', label: 'Кулоб' },
    { value: 'qurghonteppa', label: 'Курган-Тюбе' },
    { value: 'istiklol', label: 'Истиклол' }
  ];

  const buildStateOptions = [
    { value: 'sostoyanie---10', label: 'Построено' },
    { value: 'sostoyanie---11', label: 'На стадии строительства' }
  ];

  const propertyTypeOptions = [
    { value: '', label: 'Все типы' },
    { value: 'type---1', label: 'Вторичный рынок' },
    { value: 'type---2', label: 'Новостройки' }
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
          Сбор данных недвижимости
        </Typography>
        
        <Typography variant="body1" color="text.secondary" paragraph>
          Начните с определения критериев поиска, затем система автоматически соберет данные о недвижимости, 
          обработает их и подготовит для анализа.
        </Typography>

        <Paper sx={{ p: 3, mb: 3 }}>
          <form onSubmit={handleSubmit}>
            <Typography variant="h6" gutterBottom>
              Критерии поиска
            </Typography>
            
            <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' } }}>
              <FormControl fullWidth>
                <InputLabel>Количество комнат</InputLabel>
                <Select
                  value={formData.rooms}
                  label="Количество комнат"
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
                <InputLabel>Город</InputLabel>
                <Select
                  value={formData.city}
                  label="Город"
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
                <InputLabel>Состояние строительства</InputLabel>
                <Select
                  value={formData.build_state}
                  label="Состояние строительства"
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
                <InputLabel>Тип недвижимости</InputLabel>
                <Select
                  value={formData.property_type || ''}
                  label="Тип недвижимости"
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
                {isLoading ? 'Сбор данных...' : 'Начать сбор данных'}
              </Button>
            </Box>
          </form>
        </Paper>

        {/* Progress Steps */}
        {(isLoading || result) && (
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Процесс обработки данных
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
                  Сбор данных с сайта
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
                  Предварительная обработка
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
                  Инженерия признаков
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
                  Импорт в базу данных
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
              Результаты
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
                  label={`Собрано объектов: ${result.total_scraped}`}
                  color="primary"
                  variant="outlined"
                />
              </Box>
            )}

            {result.files_created && result.files_created.length > 0 && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Созданные файлы ({result.files_created.length})</Typography>
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
                          secondary={`Файл ${index + 1} из ${result.files_created?.length || 0}`}
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
            Информация о процессе
          </Typography>
          
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>Что происходит при сборе данных?</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography component="div" variant="body2">
                <ol>
                  <li><strong>Сбор данных</strong> - Система автоматически соберет объявления о недвижимости с сайта Somon.tj по заданным критериям</li>
                  <li><strong>Предварительная обработка</strong> - Очистка и нормализация данных, удаление дубликатов</li>
                  <li><strong>Инженерия признаков</strong> - Создание дополнительных характеристик для анализа и прогнозирования</li>
                  <li><strong>Готовность к анализу</strong> - После завершения данные будут доступны для поиска и анализа</li>
                </ol>
              </Typography>
            </AccordionDetails>
          </Accordion>
          
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>Какие данные собираются?</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2">
                • Цена и цена за квадратный метр<br/>
                • Количество комнат и общая площадь<br/>
                • Этаж и общее количество этажей<br/>
                • Район и адрес<br/>
                • Тип строительства и состояние<br/>
                • Фотографии объекта<br/>
                • Дополнительные характеристики
              </Typography>
            </AccordionDetails>
          </Accordion>
        </Paper>
      </Box>
    </Container>
  );
};

export default DataCollection;
