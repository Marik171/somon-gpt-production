import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Container,
  Typography,
  Grid,
  Box,
  Card,
  CardContent,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  CircularProgress,
  Alert,
  Pagination,
  FormControlLabel,
  Switch,
} from '@mui/material';
import {
  Search,
  FilterList,
  ExpandMore,
  Clear,
  Tune,
} from '@mui/icons-material';
import PropertyCard from '../components/PropertyCard';
import { apiService, Property, PropertyFilters, FilterRanges } from '../services/api';

const PropertySearch: React.FC = () => {
  const [properties, setProperties] = useState<Property[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [districts, setDistricts] = useState<string[]>([]);
  const [favorites, setFavorites] = useState<Set<number>>(new Set());
  
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const itemsPerPage = 24; // Increased from 12 to 24 for better UX
  
  // Dynamic filter ranges state
  const [filterRanges, setFilterRanges] = useState<FilterRanges>({
    price_min: 0,
    price_max: 2000000,
    area_min: 0,
    area_max: 500,
    floor_min: 0,
    floor_max: 50,
  });
  
  // Filter state
  const [filters, setFilters] = useState<PropertyFilters>({
    limit: itemsPerPage,
    offset: 0,
  });
  
  // Form state (will be updated with dynamic ranges)
  const [priceRange, setPriceRange] = useState<number[]>([0, 2000000]);
  const [areaRange, setAreaRange] = useState<number[]>([0, 500]);
  const [floorRange, setFloorRange] = useState<number[]>([0, 50]);
  const [selectedDistricts, setSelectedDistricts] = useState<string[]>([]);
  const [selectedBuildStates, setSelectedBuildStates] = useState<string[]>([]);
  const [selectedRenovations, setSelectedRenovations] = useState<string[]>([]);
  const [selectedBargainCategories, setSelectedBargainCategories] = useState<string[]>([]);
  const [excludeBasement, setExcludeBasement] = useState<boolean>(true);

  // Load initial data
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        console.log('🔍 Loading initial data for PropertySearch...');
        const [districtsData, filterRangesData] = await Promise.all([
          apiService.getDistricts(),
          apiService.getFilterRanges(),
        ]);
        console.log('✅ Districts loaded:', districtsData);
        console.log('✅ Filter ranges loaded:', filterRangesData);
        setDistricts(districtsData);
        setFilterRanges(filterRangesData);
        
        // Update initial filter ranges based on actual data
        setPriceRange([filterRangesData.price_min, filterRangesData.price_max]);
        setAreaRange([filterRangesData.area_min, filterRangesData.area_max]);
        setFloorRange([filterRangesData.floor_min, filterRangesData.floor_max]);
        console.log('✅ Initial filter ranges set');
      } catch (err) {
        console.error('❌ Error loading initial data:', err);
      }
    };
    
    loadInitialData();
    
    // Load favorites from localStorage
    const savedFavorites = localStorage.getItem('favorites');
    if (savedFavorites) {
      setFavorites(new Set(JSON.parse(savedFavorites)));
    }
  }, []);

  // Search properties
  const searchProperties = useCallback(async (resetPage = false) => {
    console.log('🔍 searchProperties called with resetPage:', resetPage);
    setLoading(true);
    setError(null);
    
    const currentPage = resetPage ? 1 : page;
    const offset = (currentPage - 1) * itemsPerPage;
    
    const searchFilters: PropertyFilters = {
      min_price: priceRange[0] > filterRanges.price_min ? priceRange[0] : undefined,
      max_price: priceRange[1] < filterRanges.price_max ? priceRange[1] : undefined,
      min_area: areaRange[0] > filterRanges.area_min ? areaRange[0] : undefined,
      max_area: areaRange[1] < filterRanges.area_max ? areaRange[1] : undefined,
      min_floor: floorRange[0] > filterRanges.floor_min ? floorRange[0] : undefined,
      max_floor: floorRange[1] < filterRanges.floor_max ? floorRange[1] : undefined,
      districts: selectedDistricts.length > 0 ? selectedDistricts.join(',') : undefined,
      build_states: selectedBuildStates.length > 0 ? selectedBuildStates.join(',') : undefined,
      renovations: selectedRenovations.length > 0 ? selectedRenovations.join(',') : undefined,
      bargain_categories: selectedBargainCategories.length > 0 ? selectedBargainCategories.join(',') : undefined,
      limit: itemsPerPage,
      offset,
    };
    
    try {
      console.log('Making API call with filters:', searchFilters);
      const data = await apiService.searchProperties(searchFilters);
      console.log('Received properties:', data?.length || 0);
      console.log('First property sample:', data[0]);
      
      // Filter out basement properties if excludeBasement is true
      const filteredData = excludeBasement 
        ? data.filter(property => property.floor !== 0)
        : data;
      
      console.log(`🏠 PropertySearch: Filtered ${data.length} → ${filteredData.length} properties (basement excluded: ${excludeBasement})`);
      setProperties(filteredData);
      
      // Calculate total pages based on whether we got a full page or not
      // If we got less than itemsPerPage, we're on the last page
      if (data.length < itemsPerPage) {
        setTotalPages(currentPage);
      } else {
        // If we got a full page, there might be more pages
        // For better UX, we'll show at least one more page option
        setTotalPages(currentPage + 1);
      }
      
      if (resetPage) {
        setPage(1);
      }
    } catch (err) {
      console.error('❌ Error searching properties:', err);
      setError('Failed to search properties. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [
    page, 
    itemsPerPage,
    priceRange,
    areaRange,
    floorRange,
    selectedDistricts,
    selectedBuildStates,
    selectedRenovations,
    selectedBargainCategories,
    excludeBasement
  ]);

  // Initial search - only after ranges are loaded
  useEffect(() => {
    console.log('🔍 Initial search useEffect triggered. filterRanges.price_max:', filterRanges.price_max);
    if (filterRanges.price_max > 200000) { // Only search after ranges are loaded
      console.log('✅ Filter ranges loaded, calling searchProperties()');
      searchProperties();
    } else {
      console.log('⏳ Waiting for filter ranges to load...');
    }
  }, [page, filterRanges.price_max]);

  // Auto-search when filters change (with debouncing to avoid too many requests)
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      // Only search if we have loaded the filter ranges to avoid premature searches
      if (filterRanges.price_max > 200000) { // Only search after ranges are loaded
        searchProperties(true); // Reset to page 1 when filters change
      }
    }, 300); // Reduced debounce time

    return () => clearTimeout(timeoutId);
  }, [
    priceRange,
    areaRange,
    floorRange,
    selectedDistricts,
    selectedBuildStates,
    selectedRenovations,
    selectedBargainCategories,
    excludeBasement,
    filterRanges.price_max, // Add this dependency to trigger search after ranges load
  ]);

  const clearFilters = useCallback(() => {
    setPriceRange([filterRanges.price_min, filterRanges.price_max]);
    setAreaRange([filterRanges.area_min, filterRanges.area_max]);
    setFloorRange([filterRanges.floor_min, filterRanges.floor_max]);
    setSelectedDistricts([]);
    setSelectedBuildStates([]);
    setSelectedRenovations([]);
    setSelectedBargainCategories([]);
    setExcludeBasement(true); // Reset to default (exclude basements)
  }, [filterRanges]);

  const handleFavorite = useCallback((property: Property) => {
    if (!property.id) return;
    
    const newFavorites = new Set(favorites);
    if (favorites.has(property.id)) {
      newFavorites.delete(property.id);
    } else {
      newFavorites.add(property.id);
    }
    setFavorites(newFavorites);
    localStorage.setItem('favorites', JSON.stringify(Array.from(newFavorites)));
  }, [favorites]);

  const getActiveFiltersCount = useMemo(() => {
    let count = 0;
    if (priceRange[0] > filterRanges.price_min || priceRange[1] < filterRanges.price_max) count++;
    if (areaRange[0] > filterRanges.area_min || areaRange[1] < filterRanges.area_max) count++;
    if (floorRange[0] > filterRanges.floor_min || floorRange[1] < filterRanges.floor_max) count++;
    if (selectedDistricts.length > 0) count++;
    if (selectedBuildStates.length > 0) count++;
    if (selectedRenovations.length > 0) count++;
    if (selectedBargainCategories.length > 0) count++;
    if (excludeBasement) count++; // Count basement exclusion as an active filter
    return count;
  }, [
    priceRange,
    areaRange,
    floorRange,
    selectedDistricts,
    selectedBuildStates,
    selectedRenovations,
    selectedBargainCategories,
    excludeBasement,
    filterRanges,
  ]);

  const buildStateOptions = [
    'Новостройка',
    'Вторичный рынок',
  ];

  const renovationOptions = [
    'Без ремонта (коробка)',
    'Новый ремонт',
    'С ремонтом',
  ];

  const bargainCategoryOptions = [
    { value: 'exceptional_opportunity', label: 'Exceptional Opportunity' },
    { value: 'excellent_bargain', label: 'Excellent Bargain' },
    { value: 'good_bargain', label: 'Good Bargain' },
    { value: 'fair_value', label: 'Fair Value' },
    { value: 'market_price', label: 'Market Price' },
    { value: 'overpriced', label: 'Overpriced' },
  ];

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Typography
          variant="h2"
          sx={{
            fontWeight: 700,
            color: 'primary.main',
            mb: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 2,
          }}
        >
          <Search sx={{ fontSize: 48 }} />
          Property Search
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 800, mx: 'auto' }}>
          Find your perfect property with advanced filtering and AI-powered insights
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Filters Sidebar */}
        <Grid item xs={12} lg={3}>
          <Card sx={{ position: 'sticky', top: 100 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Tune sx={{ color: 'primary.main' }} />
                  <Typography variant="h6">Filters</Typography>
                  {getActiveFiltersCount > 0 && (
                    <Chip size="small" label={getActiveFiltersCount} color="primary" />
                  )}
                </Box>
                <Button
                  size="small"
                  startIcon={<Clear />}
                  onClick={clearFilters}
                >
                  Clear
                </Button>
              </Box>

              {/* Price Range */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle2">Price Range</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ px: 1 }}>
                    <Slider
                      value={priceRange}
                      onChange={(_, newValue) => setPriceRange(newValue as number[])}
                      valueLabelDisplay="auto"
                      min={filterRanges.price_min}
                      max={filterRanges.price_max}
                      step={5000}
                      valueLabelFormat={(value) => `${value.toLocaleString()} TJS`}
                    />
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                      <Typography variant="caption">{priceRange[0].toLocaleString()} TJS</Typography>
                      <Typography variant="caption">{priceRange[1].toLocaleString()} TJS</Typography>
                    </Box>
                  </Box>
                </AccordionDetails>
              </Accordion>

              {/* Area Range */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle2">Area (m²)</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ px: 1 }}>
                    <Slider
                      value={areaRange}
                      onChange={(_, newValue) => setAreaRange(newValue as number[])}
                      valueLabelDisplay="auto"
                      min={filterRanges.area_min}
                      max={filterRanges.area_max}
                      step={10}
                      valueLabelFormat={(value) => `${value}m²`}
                    />
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                      <Typography variant="caption">{areaRange[0]}m²</Typography>
                      <Typography variant="caption">{areaRange[1]}m²</Typography>
                    </Box>
                  </Box>
                </AccordionDetails>
              </Accordion>

              {/* Floor */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle2">Floor</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ px: 1 }}>
                    <Slider
                      value={floorRange}
                      onChange={(_, newValue) => setFloorRange(newValue as number[])}
                      valueLabelDisplay="auto"
                      min={filterRanges.floor_min}
                      max={filterRanges.floor_max}
                      step={1}
                      valueLabelFormat={(value) => value === 0 ? 'Basement' : `Floor ${value}`}
                    />
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1, mb: 2 }}>
                      <Typography variant="caption">
                        {floorRange[0] === 0 ? 'Basement' : `Floor ${floorRange[0]}`}
                      </Typography>
                      <Typography variant="caption">
                        {floorRange[1] === 0 ? 'Basement' : `Floor ${floorRange[1]}`}
                      </Typography>
                    </Box>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={excludeBasement}
                          onChange={(e) => setExcludeBasement(e.target.checked)}
                          color="primary"
                        />
                      }
                      label="Exclude Basement Properties"
                      sx={{ 
                        '& .MuiFormControlLabel-label': { 
                          fontSize: '0.875rem',
                          fontWeight: 500,
                        }
                      }}
                    />
                  </Box>
                </AccordionDetails>
              </Accordion>

              {/* Districts */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle2">Districts ({districts.length} available)</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <FormControl fullWidth>
                    <InputLabel>Select Districts</InputLabel>
                    <Select
                      multiple
                      value={selectedDistricts}
                      onChange={(e) => setSelectedDistricts(e.target.value as string[])}
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <Chip key={value} label={value} size="small" />
                          ))}
                        </Box>
                      )}
                    >
                      {districts.map((district) => (
                        <MenuItem key={district} value={district}>
                          {district}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </AccordionDetails>
              </Accordion>

              {/* Build State */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle2">Build State</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <FormControl fullWidth>
                    <InputLabel>Select Build States</InputLabel>
                    <Select
                      multiple
                      value={selectedBuildStates}
                      onChange={(e) => setSelectedBuildStates(e.target.value as string[])}
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <Chip key={value} label={value} size="small" />
                          ))}
                        </Box>
                      )}
                    >
                      {buildStateOptions.map((state) => (
                        <MenuItem key={state} value={state}>
                          {state}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </AccordionDetails>
              </Accordion>

              {/* Renovation */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle2">Renovation</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <FormControl fullWidth>
                    <InputLabel>Select Renovation States</InputLabel>
                    <Select
                      multiple
                      value={selectedRenovations}
                      onChange={(e) => setSelectedRenovations(e.target.value as string[])}
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <Chip key={value} label={value} size="small" />
                          ))}
                        </Box>
                      )}
                    >
                      {renovationOptions.map((renovation) => (
                        <MenuItem key={renovation} value={renovation}>
                          {renovation}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </AccordionDetails>
              </Accordion>

              {/* Investment Quality */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle2">Investment Quality</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <FormControl fullWidth>
                    <InputLabel>Select Categories</InputLabel>
                    <Select
                      multiple
                      value={selectedBargainCategories}
                      onChange={(e) => setSelectedBargainCategories(e.target.value as string[])}
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <Chip 
                              key={value} 
                              label={bargainCategoryOptions.find(o => o.value === value)?.label} 
                              size="small" 
                            />
                          ))}
                        </Box>
                      )}
                    >
                      {bargainCategoryOptions.map((option) => (
                        <MenuItem key={option.value} value={option.value}>
                          {option.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </AccordionDetails>
              </Accordion>
            </CardContent>
          </Card>
        </Grid>

        {/* Results */}
        <Grid item xs={12} lg={9}>
          {/* Results Header */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {loading ? 'Searching...' : `${properties.length} Properties Found`}
            </Typography>
            <Chip
              icon={<FilterList />}
              label={`${getActiveFiltersCount} Active Filters`}
              color={getActiveFiltersCount > 0 ? 'primary' : 'default'}
              variant="outlined"
            />
          </Box>

          {/* Loading State */}
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
              <CircularProgress size={60} />
            </Box>
          )}

          {/* Error State */}
          {error && (
            <Alert severity="error" sx={{ mb: 4 }}>
              {error}
            </Alert>
          )}

          {/* Results Grid */}
          {!loading && !error && (
            <>
              {properties.length === 0 ? (
                <Card sx={{ p: 4, textAlign: 'center' }}>
                  <Typography variant="h6" color="text.secondary">
                    No properties found matching your criteria.
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Try adjusting your filters to see more results.
                  </Typography>
                </Card>
              ) : (
                <>
                  <Grid container spacing={3}>
                    {properties.map((property, index) => (
                      <Grid item xs={12} sm={6} xl={4} key={property.url || `property-${index}`}>
                        <PropertyCard
                          property={property}
                          showInvestmentMetrics={true}
                          onFavorite={handleFavorite}
                          isFavorited={property.id ? favorites.has(property.id) : false}
                        />
                      </Grid>
                    ))}
                  </Grid>

                  {/* Pagination */}
                  {totalPages > 1 && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
                      <Pagination
                        count={totalPages}
                        page={page}
                        onChange={(_, newPage) => setPage(newPage)}
                        color="primary"
                        size="large"
                      />
                    </Box>
                  )}
                </>
              )}
            </>
          )}
        </Grid>
      </Grid>
    </Container>
  );
};

export default PropertySearch;
