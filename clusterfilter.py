import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List
from scipy import stats

class ClusterFilter:
    """
    A class for filtering clusters based on variability metrics.
    
    IMPORTANT: This class expects cluster IDs to contain the measurement type as a substring.
    For example, if measurement_cols = {'tube': 'column_name'}, then cluster IDs 
    containing 'tube' (like 'tube_River_Stream_0') will be mapped to that measurement column.
    
    The measurement type must be present in the cluster_id for proper column mapping.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 cluster_col: str,
                 measurement_cols: Dict[str, str],
                 site_col: Optional[str] = None,
                 date_col: Optional[str] = None):
        """
        Initialize the ClusterFilter.
        
        Args:
            data: Input dataframe
            cluster_col: Column that identifies clusters (e.g., 'cluster_type_id')
            measurement_cols: Dictionary mapping measurement types to column names.
                             The keys (measurement types) MUST appear in the cluster IDs.
                             e.g., {'tube': 'transparencies:tube image disappearance (cm)',
                                    'disk': 'transparencies:transparency disk image disappearance (m)',
                                    'sensor': 'transparencies:sensor turbidity ntu'}
            site_col: Column that identifies sites within clusters (optional)
            date_col: Column containing measurement dates (optional)
            
        Note:
            The cluster IDs in cluster_col MUST contain the measurement type as defined
            in measurement_cols keys. For example, if you have 'tube' as a key in 
            measurement_cols, cluster IDs like 'tube_River_Stream_0' will be properly
            mapped to the tube measurement column.
        """
        self.data = data.copy()
        self.cluster_col = cluster_col
        self.measurement_cols = measurement_cols  # This is the dictionary mapping types to columns
        self.site_col = site_col
        self.date_col = date_col
        
        # Validate that required columns exist
        self._validate_columns()
        
        # Convert measurement columns to numeric
        for col in self.measurement_cols.values():
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Warn about cluster ID requirements
        print("Note: Cluster IDs must contain measurement type strings for proper mapping.")
        print(f"Expected measurement types in cluster IDs: {list(self.measurement_cols.keys())}")
    
    def _validate_columns(self):
        """Validate that specified columns exist in the dataframe."""
        # Check cluster column
        if self.cluster_col not in self.data.columns:
            raise ValueError(f"Cluster column '{self.cluster_col}' not found in dataframe")
        
        # Check measurement columns
        for m_type, col in self.measurement_cols.items():
            if col not in self.data.columns:
                print(f"Warning: Measurement column '{col}' for type '{m_type}' not found in dataframe")
        
        # Check optional columns
        if self.site_col and self.site_col not in self.data.columns:
            print(f"Warning: Site column '{self.site_col}' not found. Proceeding without site information.")
            self.site_col = None
            
        if self.date_col and self.date_col not in self.data.columns:
            print(f"Warning: Date column '{self.date_col}' not found. Proceeding without temporal information.")
            self.date_col = None
    
    def _extract_measurement_type_from_cluster_id(self, cluster_id: str) -> Optional[str]:
        """
        Extract measurement type from cluster ID by checking which measurement type
        string is contained in the cluster ID.
        
        Args:
            cluster_id: The cluster identifier (e.g., 'tube_River_Stream_0')
            
        Returns:
            Measurement type if found in cluster_id, None otherwise
            
        Note:
            This method looks for measurement type keys within the cluster_id string.
            The first matching measurement type found will be returned.
        """
        cluster_id_lower = cluster_id.lower()
        
        # Check each measurement type to see if it's in the cluster ID
        for m_type in self.measurement_cols.keys():
            if m_type.lower() in cluster_id_lower:
                return m_type
        
        return None
    
    def calculate_cv(self, values: pd.Series) -> float:
        """Calculate standard coefficient of variation."""
        if len(values) < 2:
            return np.nan
        
        mean = values.mean()
        std = values.std()
        
        if mean == 0 or np.isnan(mean):
            return np.nan
        
        return std / abs(mean)

    def calculate_robust_cv(self, values: pd.Series) -> float:
        """Calculate robust coefficient of variation using IQR/median."""
        if len(values) < 4:  # Need at least 4 points for quartiles
            return np.nan
        
        median = values.median()
        q75, q25 = values.quantile([0.75, 0.25])
        iqr = q75 - q25
        
        if median == 0 or np.isnan(median):
            return np.nan
        
        return iqr / abs(median)

    def analyze_clusters(self) -> pd.DataFrame:
        """
        Analyze each cluster and calculate variability metrics.
        
        Returns:
            DataFrame with cluster-level statistics
            
        Note:
            This method extracts the measurement type from each cluster_id and uses
            it to determine which measurement column to analyze. If no measurement
            type can be extracted from a cluster_id, that cluster will be skipped.
        """
        results = []
        
        for cluster_id, cluster_data in self.data.groupby(self.cluster_col):
            # Extract measurement type from cluster_id
            measurement_type = self._extract_measurement_type_from_cluster_id(cluster_id)
            
            if measurement_type is None:
                print(f"Warning: Could not determine measurement type for cluster '{cluster_id}'")
                print(f"  Cluster ID should contain one of: {list(self.measurement_cols.keys())}")
                continue
            
            # Get the appropriate measurement column from the dictionary
            measurement_col = self.measurement_cols.get(measurement_type)
            
            if measurement_col is None or measurement_col not in cluster_data.columns:
                print(f"Warning: No measurement column found for cluster '{cluster_id}' with type '{measurement_type}'")
                continue
            
            # Get non-null measurement values
            values = cluster_data[measurement_col].dropna()
            
            # Calculate basic statistics
            cluster_stats = {
                'cluster_id': cluster_id,
                'measurement_type': measurement_type,
                'measurement_column': measurement_col,
                'n_obs': len(values),
                'n_missing': cluster_data[measurement_col].isna().sum(),
                'mean': values.mean() if len(values) > 0 else np.nan,
                'median': values.median() if len(values) > 0 else np.nan,
                'std': values.std() if len(values) > 1 else np.nan,
                'cv': self.calculate_cv(values),
                'robust_cv': self.calculate_robust_cv(values),
                'iqr': values.quantile(0.75) - values.quantile(0.25) if len(values) >= 4 else np.nan,
                'min': values.min() if len(values) > 0 else np.nan,
                'max': values.max() if len(values) > 0 else np.nan,
            }
            
            # Add site information if available
            if self.site_col:
                cluster_stats['n_sites'] = cluster_data[self.site_col].nunique()
            
            # Add temporal information if available
            if self.date_col:
                dates = pd.to_datetime(cluster_data[self.date_col], errors='coerce')
                valid_dates = dates.dropna()
                if len(valid_dates) > 0:
                    cluster_stats['date_range_days'] = (valid_dates.max() - valid_dates.min()).days
                    cluster_stats['n_unique_dates'] = valid_dates.nunique()
                else:
                    cluster_stats['date_range_days'] = np.nan
                    cluster_stats['n_unique_dates'] = 0
            
            results.append(cluster_stats)
        
        return pd.DataFrame(results)
    
######################################################## Function used to filter based on results from IQR and CV ######################################################################

def filter_clusters_with_rules(cluster_stats, 
                             min_obs=5, 
                             min_robust_cv=0.05, 
                             max_robust_cv=1.7):
    """
    Filter clusters based on simple, sensible rules.
    
    Parameters:
    - cluster_stats: DataFrame with cluster statistics
    - min_obs: Minimum number of observations required
    - min_robust_cv: Minimum robust CV (to exclude zero-variance clusters)
    - max_robust_cv: Maximum robust CV (to exclude extreme variability)
    
    Returns:
    - valid_clusters: Clusters that meet the criteria
    - excluded_clusters: Clusters that don't meet the criteria
    """
    # Apply filtering rules
    valid_mask = (
        (cluster_stats['n_obs'] >= min_obs) &
        (cluster_stats['robust_cv'] >= min_robust_cv) &
        (cluster_stats['robust_cv'] <= max_robust_cv) &
        (cluster_stats['robust_cv'].notna())  # Ensure robust_cv is not NaN
    )
    
    valid_clusters = cluster_stats[valid_mask].copy()
    excluded_clusters = cluster_stats[~valid_mask].copy()
    
    # Add exclusion reasons
    excluded_clusters['exclusion_reason'] = ''
    excluded_clusters.loc[excluded_clusters['n_obs'] < min_obs, 'exclusion_reason'] += 'too_few_obs; '
    excluded_clusters.loc[excluded_clusters['robust_cv'] < min_robust_cv, 'exclusion_reason'] += 'too_low_variability; '
    excluded_clusters.loc[excluded_clusters['robust_cv'] > max_robust_cv, 'exclusion_reason'] += 'too_high_variability; '
    excluded_clusters.loc[excluded_clusters['robust_cv'].isna(), 'exclusion_reason'] += 'missing_robust_cv; '
    
    # Print summary
    print(f"Total clusters: {len(cluster_stats)}")
    print(f"Valid clusters: {len(valid_clusters)}")
    print(f"Excluded clusters: {len(excluded_clusters)}")
    
    # Break down by measurement type
    print("\nValid clusters by measurement type:")
    for mtype in valid_clusters['measurement_type'].unique():
        count = len(valid_clusters[valid_clusters['measurement_type'] == mtype])
        print(f"  {mtype}: {count}")
    
    print("\nExclusion reasons:")
    for reason in ['too_few_obs', 'too_low_variability', 'too_high_variability', 'missing_robust_cv']:
        count = excluded_clusters['exclusion_reason'].str.contains(reason).sum()
        print(f"  {reason}: {count}")
    
    return valid_clusters, excluded_clusters