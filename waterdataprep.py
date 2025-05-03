import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import re
from clusterer import HierarchicalClusterer


class WaterDataPrep:
    """
    A class to prepare water quality data by adding measurement types, adding additional features such as Season, Standardize columns, etc.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 water_body_source_col: str = 'water_body_source',
                 lat_col: str = 'latitude(sample)',
                 lon_col: str = 'longitude(sample)',
                 site_id_col: str = 'site_id',
                 date_col: str = 'measured_on',
                 transparency_cols = {'tube':'transparenciesTubeImageDisappearanceCm',
                                      'disk':'transparenciesTransparencyDiskImageDisappearanceM',
                                      'sensor':'transparenciesSensorTurbidityNtu'}):
        """
        Initialize the data preparation class.
        
        Parameters:
        - data: DataFrame containing water quality measurements
        - water_body_source_col: Column name for water body source
        - site_id_col: Column name for site identifiers
        """
        self.df = data.copy()
        self.water_body_source_col = water_body_source_col
        self.site_id_col = site_id_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.date_col = date_col
        # Define transparency column names
        for k,v in transparency_cols.items():
            if 'TUBE' in v.upper():
                self.tube_col = v
            elif 'DISK' in v.upper():
                self.disk_col = v
            elif 'SENSOR' in v.upper():
                self.sensor_col = v
        # self.disk_col = 'transparencies:transparency disk image disappearance (m)'
        # self.tube_col = 'transparencies:tube image disappearance (cm)'
        # self.sensor_col = 'transparencies:sensor turbidity ntu'
        
        # Store results
        self.measurement_dfs = None
        self.water_source_dfs = None
    
    def determine_primary_measurement(self) -> pd.DataFrame:
        """
        Adds a 'primary_measurement_type' column based on available non-null transparency data.
        Priority: disk > tube > sensor.
        
        Returns:
        - DataFrame with added 'primary_measurement_type' column
        """
        print("Step 1: Determining primary measurement type for each row...")
        conditions = []
        choices = []
        
        # Check if columns exist and convert to numeric
        if self.disk_col in self.df.columns:
            self.df[self.disk_col] = pd.to_numeric(self.df[self.disk_col], errors='coerce')
            conditions.append(self.df[self.disk_col].notna())
            choices.append('disk')
        else:
            print(f"  - Debug: Disk column '{self.disk_col}' not found.")
        
        if self.tube_col in self.df.columns:
            self.df[self.tube_col] = pd.to_numeric(self.df[self.tube_col], errors='coerce')
            conditions.append(self.df[self.tube_col].notna())
            choices.append('tube')
        else:
            print(f"  - Debug: Tube column '{self.tube_col}' not found.")
        
        if self.sensor_col in self.df.columns:
            self.df[self.sensor_col] = pd.to_numeric(self.df[self.sensor_col], errors='coerce')
            conditions.append(self.df[self.sensor_col].notna())
            choices.append('sensor')
        else:
            print(f"  - Debug: Sensor column '{self.sensor_col}' not found.")
        
        # Assign primary measurement type
        if not conditions:
            print("  - ERROR: No valid transparency columns found. Assigning 'none' to all rows.")
            self.df['primary_measurement_type'] = 'none'
        else:
            self.df['primary_measurement_type'] = np.select(conditions, choices, default='none')
        
        # Display counts
        counts = self.df['primary_measurement_type'].value_counts()
        print("  - Primary measurement type counts:")
        for m_type, count in counts.items():
            print(f"    - {m_type}: {count}")
        
        return self.df
    
    def add_measurement_flags(self) -> pd.DataFrame:
        """
        Adds boolean flags for each measurement type availability.
        
        Returns:
        - DataFrame with added boolean columns for each measurement type
        """
        print("Adding measurement availability flags...")
        
        # Create boolean flags for each measurement type
        if self.disk_col in self.df.columns:
            self.df['has_disk_measurement'] = self.df[self.disk_col].notna()
        else:
            self.df['has_disk_measurement'] = False
        
        if self.tube_col in self.df.columns:
            self.df['has_tube_measurement'] = self.df[self.tube_col].notna()
        else:
            self.df['has_tube_measurement'] = False
        
        if self.sensor_col in self.df.columns:
            self.df['has_sensor_measurement'] = self.df[self.sensor_col].notna()
        else:
            self.df['has_sensor_measurement'] = False
        
        # Count available measurements per row
        self.df['num_measurement_types'] = (
            self.df['has_disk_measurement'].astype(int) +
            self.df['has_tube_measurement'].astype(int) +
            self.df['has_sensor_measurement'].astype(int)
        )
        
        print(f"  - Added measurement flags and count")
        return self.df
    
    def standardize_water_source_names(self) -> pd.DataFrame:
        """
        Standardizes water source names and creates a safe key column.
        
        Returns:
        - DataFrame with standardized water source column
        """
        print("Standardizing water source names...")
        
        if self.water_body_source_col not in self.df.columns:
            print(f"  - WARNING: Water body source column '{self.water_body_source_col}' not found.")
            self.df['water_source_standardized'] = 'unknown'
            self.df['water_source_key'] = 'unknown'
            return self.df
        
        # Fill NaN values
        self.df[self.water_body_source_col] = self.df[self.water_body_source_col].fillna('Unknown')
        print('Unknowns filled')
        # Create standardized version (title case, trimmed)
        self.df['water_source_standardized'] = self.df[self.water_body_source_col].str.strip().str.title()
        
        # Create safe key version (lowercase, alphanumeric with underscores)
        self.df['water_source_key'] = self.df['water_source_standardized'].apply(
            lambda x: re.sub(r'\W+', '_', str(x).lower())
        )
        
        # Display unique standardized values
        unique_sources = self.df['water_source_standardized'].value_counts()
        print(f"  - Found {len(unique_sources)} unique water sources:")
        for source, count in unique_sources.items():
            print(f"    - {source}: {count}")
        
        return self.df
    
    def add_coordinate_validity_flags(self) -> pd.DataFrame:
        """
        Adds flags indicating whether coordinates are valid.
        
        Returns:
        - DataFrame with coordinate validity flags
        """
        print("Adding coordinate validity flags...")
        
        lat_col = self.lat_col
        lon_col = self.lon_col
        
        if lat_col in self.df.columns and lon_col in self.df.columns:
            # Check if coordinates are numeric and within valid ranges
            self.df['has_valid_coordinates'] = (
                pd.to_numeric(self.df[lat_col], errors='coerce').notna() &
                pd.to_numeric(self.df[lon_col], errors='coerce').notna() &
                (pd.to_numeric(self.df[lat_col], errors='coerce').between(-90, 90)) &
                (pd.to_numeric(self.df[lon_col], errors='coerce').between(-180, 180))
            )
            
            valid_count = self.df['has_valid_coordinates'].sum()
            total_count = len(self.df)
            print(f"  - {valid_count}/{total_count} rows have valid coordinates")
        else:
            print(f"  - WARNING: Coordinate columns not found")
            self.df['has_valid_coordinates'] = False
        
        return self.df
    
    def add_date_features(self) -> pd.DataFrame:
        """
        Extracts date features if date columns exist.
        
        Returns:
        - DataFrame with added date features
        """
        print("Adding date features...")
        
        # Common date column names to check
        date_columns = ['date', 'sample_date', 'sampling_date', 'datetime', 'timestamp']
        
        date_col = None
        for col in date_columns:
            if col in self.df.columns:
                date_col = col
                break
        
        if date_col:
            try:
                # Convert to datetime
                self.df['datetime'] = pd.to_datetime(self.df[date_col], errors='coerce')
                
                # Extract features
                self.df['year'] = self.df['datetime'].dt.year
                self.df['month'] = self.df['datetime'].dt.month
                self.df['day'] = self.df['datetime'].dt.day
                self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
                self.df['day_name'] = self.df['datetime'].dt.day_name()
                self.df['month_name'] = self.df['datetime'].dt.month_name()
                self.df['quarter'] = self.df['datetime'].dt.quarter
                self.df['is_weekend'] = self.df['datetime'].dt.dayofweek.isin([5, 6])
                
                print(f"  - Extracted date features from '{date_col}'")
            except Exception as e:
                print(f"  - Error extracting date features: {e}")
        else:
            print("  - No date column found")
        
        return self.df
    
    def create_composite_keys(self) -> pd.DataFrame:
        """
        Creates composite keys for easier grouping and analysis.
        
        Returns:
        - DataFrame with composite key columns
        """
        print("Creating composite keys...")
        
        # Create measurement + water source key
        if 'primary_measurement_type' in self.df.columns and 'water_source_key' in self.df.columns:
            self.df['measurement_source_key'] = (
                self.df['primary_measurement_type'] + '_' + self.df['water_source_key']
            )
        
        # Create site + measurement type key if site_id exists
        if self.site_id_col in self.df.columns and 'primary_measurement_type' in self.df.columns:
            self.df['site_measurement_key'] = (
                self.df[self.site_id_col].astype(str) + '_' + self.df['primary_measurement_type']
            )
        
        print("  - Created composite keys")
        return self.df
    
    # def prepare_data(self) -> pd.DataFrame:
    #     """
    #     Run all data preparation steps.
        
    #     Returns:
    #     - Fully prepared DataFrame
    #     """
    #     print("Starting water data preparation pipeline...")
    #     print("=" * 50)
        
    #     # Run all preparation steps
    #     self.determine_primary_measurement()
    #     self.add_measurement_flags()
    #     self.standardize_water_source_names()
    #     self.add_coordinate_validity_flags()
    #     self.add_date_features()
    #     self.add_seasonal_features()
    #     self.create_composite_keys()
        
    #     print("=" * 50)
    #     print("Data preparation complete!")
    #     print(f"Final DataFrame shape: {self.df.shape}")
        
    #     return self.df
    
    def get_data_quality_report(self) -> pd.DataFrame:
        """
        Generate a data quality report.
        
        Returns:
        - DataFrame with data quality metrics
        """
        report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values_total': self.df.isnull().sum().sum(),
            'missing_values_percentage': (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        }
        
        # Add specific checks if columns exist
        if 'has_valid_coordinates' in self.df.columns:
            report['valid_coordinates_count'] = self.df['has_valid_coordinates'].sum()
            report['valid_coordinates_percentage'] = (self.df['has_valid_coordinates'].sum() / len(self.df)) * 100
        
        if 'primary_measurement_type' in self.df.columns:
            for m_type in self.df['primary_measurement_type'].unique():
                report[f'measurement_{m_type}_count'] = (self.df['primary_measurement_type'] == m_type).sum()
        
        if 'water_source_standardized' in self.df.columns:
            report['unique_water_sources'] = self.df['water_source_standardized'].nunique()
        
        report_df = pd.DataFrame([report]).T
        report_df.columns = ['Value']
        
        return report_df
    
    ######################################################## Helper function ######################################################################
    def split_by_attribute(self, attribute_col: str) -> Dict[str, pd.DataFrame]:
        """
        Splits the DataFrame into a dictionary of DataFrames based on unique values in a specified column.
        
        Parameters:
        - attribute_col: Column name to split the data on
        
        Returns:
        - Dictionary where keys are unique values from the attribute column and values are DataFrames
        """
        print(f"Splitting data by '{attribute_col}'...")
        
        if attribute_col not in self.df.columns:
            print(f"  - ERROR: Column '{attribute_col}' not found in DataFrame")
            return {}
        
        # Get unique values
        unique_values = self.df[attribute_col].unique()
        print(f"  - Found {len(unique_values)} unique values in '{attribute_col}'")
        
        # Create dictionary of DataFrames
        result_dict = {}
        for value in unique_values:
            df_subset = self.df[self.df[attribute_col] == value].copy()
            result_dict[str(value)] = df_subset
            print(f"    - {value}: {len(df_subset)} rows")
        
        return result_dict

    def create_measurement_dfs(self) -> Dict[str, pd.DataFrame]:
        """
        Creates a dictionary of DataFrames split by primary measurement type.
        
        Returns:
        - Dictionary where keys are measurement types and values are DataFrames
        """
        if 'primary_measurement_type' not in self.df.columns:
            print("  - ERROR: 'primary_measurement_type' column not found. Run prepare_data() first.")
            return {}
        
        self.measurement_dfs = self.split_by_attribute('primary_measurement_type')
        return self

    def create_water_source_dfs(self) -> Dict[str, pd.DataFrame]:
        """
        Creates a dictionary of DataFrames split by water source.
        
        Returns:
        - Dictionary where keys are water sources and values are DataFrames
        """
        if 'water_source_standardized' not in self.df.columns:
            print("  - ERROR: 'water_source_standardized' column not found. Run prepare_data() first.")
            return {}
        
        self.water_source_dfs = self.split_by_attribute('water_source_standardized')
        return self
    
    
    def prepare_data(self, drop_invalid_dates=False) -> pd.DataFrame:
        """
        Run all data preparation steps.
        
        Parameters:
        - drop_invalid_dates: If True, drops rows with invalid dates
        
        Returns:
        - Fully prepared DataFrame
        """
        print("Starting water data preparation pipeline...")
        print("=" * 50)
        
        # Clean dates first if requested
        if drop_invalid_dates:
            print("Cleaning dates and dropping invalid rows...")
            # Convert dates
            self.df['date_clean'] = pd.to_datetime(self.df[self.date_col], errors='coerce')
            # Drop rows with invalid dates
            before_count = len(self.df)
            self.df = self.df[self.df['date_clean'].notna()].copy()
            after_count = len(self.df)
            print(f"  - Dropped {before_count - after_count} rows with invalid dates")
            # Replace original date column with clean dates
            self.df[self.date_col] = self.df['date_clean']
            self.df.drop('date_clean', axis=1, inplace=True)
        
        # Run all preparation steps
        self.determine_primary_measurement()
        self.add_measurement_flags()
        self.standardize_water_source_names()
        self.add_coordinate_validity_flags()
        self.add_date_features()
        self.add_seasonal_features()
        self.create_composite_keys()
        
        print("=" * 50)
        print("Data preparation complete!")
        print(f"Final DataFrame shape: {self.df.shape}")
    
        return self.df

    def get_season(self, row) -> str:
        """
        Determines the season based on the measurement date and latitude.
        Takes into account hemisphere differences (flips seasons for Southern hemisphere).
        
        Parameters:
        - row: DataFrame row containing 'measured_on' and latitude data
        
        Returns:
        - Season name as string
        """
        # Get date and latitude from row
        date = pd.to_datetime(row[self.date_col])
        try:
            lat = pd.to_numeric(row[self.lat_col], errors='coerce')
            if pd.isna(lat):
                # If latitude is invalid, default to Northern hemisphere
                lat = 0
        except:
            # If conversion fails, default to Northern hemisphere
            lat = 0
        
        # Normalize year for consistent comparison
        year = 2000
        date = date.replace(year=year)
        
        # Define seasonal boundaries for Northern hemisphere
        seasons_north = {
            'Spring': (pd.Timestamp(f'{year}-03-21'), pd.Timestamp(f'{year}-06-20')),
            'Summer': (pd.Timestamp(f'{year}-06-21'), pd.Timestamp(f'{year}-09-22')),
            'Autumn': (pd.Timestamp(f'{year}-09-23'), pd.Timestamp(f'{year}-12-20')),
        }
        
        # Handle Winter separately (crosses year boundary)
        if (date >= pd.Timestamp(f'{year}-12-21')) or (date <= pd.Timestamp(f'{year}-03-20')):
            season = 'Winter'
        else:
            # Default to None to catch unexpected cases
            season = None
            for season_name, (start, end) in seasons_north.items():
                if start <= date <= end:
                    season = season_name
                    break
        
        if season is None:
            # Fallback (this shouldn't happen, but safe to include)
            season = 'Unknown'
        
        # Flip seasons for Southern Hemisphere
        if lat < 0:
            season_map = {
                'Spring': 'Autumn',
                'Summer': 'Winter',
                'Autumn': 'Spring',
                'Winter': 'Summer'
            }
            season = season_map.get(season, 'Unknown')
        
        return season

    def add_seasonal_features(self) -> pd.DataFrame:
        """
        Adds seasonal information to the dataframe based on measurement date and location.
        
        Returns:
        - DataFrame with added 'season' column
        """
        print("Adding seasonal features...")
        
        # Check if required columns exist
        if self.date_col not in self.df.columns:
            print(f"  - ERROR: '{self.date_col}' column not found.")
            return self.df
        
        if self.lat_col not in self.df.columns:
            print(f"  - ERROR: Latitude column '{self.lat_col}' not found.")
            return self.df
        
        # Apply get_season to each row
        try:
            self.df['season'] = self.df.apply(self.get_season, axis=1)
            
            # Display season counts
            season_counts = self.df['season'].value_counts()
            print("  - Season counts:")
            for season, count in season_counts.items():
                print(f"    - {season}: {count}")
                
        except Exception as e:
            print(f"  - Error adding seasonal features: {e}")
        
        return self.df

# Example usage:
# df = pd.read_csv('your_water_quality_data.csv')
# prep = WaterDataPrep(df)
# prepared_df = prep.prepare_data()
# quality_report = prep.get_data_quality_report()
# print(quality_report)