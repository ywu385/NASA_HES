import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple
from sklearn.cluster import DBSCAN
from haversine import haversine, Unit


class HierarchicalClusterer:
    """
    Performs hierarchical clustering: measurement -> water body -> location
    Now handles site deduplication and accepts different clustering radii for different water body types.
    """
    
    def __init__(self,
                 measurement_col: str = 'primary_measurement_type',
                 water_body_col: str = 'water_source_standardized',
                 site_id_col: str = 'site_id',
                 lat_col: str = 'latitude(sample)',
                 lon_col: str = 'longitude(sample)',
                 default_eps: float = 1.0):
        """
        Initialize the hierarchical clusterer.
        
        Parameters:
        - default_eps: Default clustering radius if not specified in eps_dict
        """
        self.measurement_col = measurement_col
        self.water_body_col = water_body_col
        self.site_id_col = site_id_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.default_eps = default_eps
    
    @staticmethod
    def _haversine_distance(coord1, coord2):
        """Calculate haversine distance between two coordinates."""
        return haversine((coord1[0], coord1[1]), (coord2[0], coord2[1]), unit=Unit.KILOMETERS)
    
    def _create_unique_sites(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a dataframe of unique sites based on site_id, latitude, and longitude.
        
        Returns:
        - DataFrame with one row per unique site
        """
        # Drop duplicates based on site_id, latitude, and longitude
        unique_sites = df[[self.site_id_col, self.lat_col, self.lon_col, 
                          self.measurement_col, self.water_body_col]].drop_duplicates(
            subset=[self.site_id_col, self.lat_col, self.lon_col]
        ).copy()
        
        print(f"Reduced {len(df)} observations to {len(unique_sites)} unique sites")
        return unique_sites
    
    def _merge_clusters_back(self, original_df: pd.DataFrame, clustered_sites: pd.DataFrame) -> pd.DataFrame:
        """
        Merge cluster assignments back to the original dataframe.
        
        Parameters:
        - original_df: Original dataframe with all observations
        - clustered_sites: Dataframe with unique sites and their cluster assignments
        
        Returns:
        - Original dataframe with cluster assignments
        """
        # Create a mapping dataframe with only the columns needed for merging
        cluster_mapping = clustered_sites[[self.site_id_col, self.lat_col, self.lon_col, 
                                         'cluster_id', 'cluster_type_id']].copy()
        
        # Merge back to original dataframe
        result_df = original_df.merge(
            cluster_mapping,
            on=[self.site_id_col, self.lat_col, self.lon_col],
            how='left'
        )
        
        # Fill any missing values (shouldn't happen if data is clean)
        result_df['cluster_id'] = result_df['cluster_id'].fillna(-1).astype(int)
        result_df['cluster_type_id'] = result_df['cluster_type_id'].fillna('noise')
        
        return result_df
    
    def split_by_measurement(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the data by measurement type.
        
        Returns:
        - Dictionary with measurement types as keys and DataFrames as values
        """
        if self.measurement_col not in df.columns:
            raise ValueError(f"Column '{self.measurement_col}' not found in DataFrame")
        
        measurement_dfs = {}
        for measurement_type in df[self.measurement_col].unique():
            measurement_dfs[measurement_type] = df[
                df[self.measurement_col] == measurement_type
            ].copy()
        
        print(f"Split data into {len(measurement_dfs)} measurement types")
        return measurement_dfs
    
    def split_by_water_body(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split a DataFrame by water body type.
        
        Parameters:
        - df: DataFrame to split
        
        Returns:
        - Dictionary with water body types as keys and DataFrames as values
        """
        if self.water_body_col not in df.columns:
            raise ValueError(f"Column '{self.water_body_col}' not found in DataFrame")
        
        water_body_dfs = {}
        for water_body in df[self.water_body_col].unique():
            water_body_dfs[water_body] = df[
                df[self.water_body_col] == water_body
            ].copy()
        
        return water_body_dfs
    
    def cluster_by_location(self, 
                           df: pd.DataFrame, 
                           measurement: str, 
                           water_body: str,
                           eps_km: float,
                           cluster_id_start: int,
                           output_dir: str = None) -> Tuple[pd.DataFrame, int]:
        """
        Cluster a DataFrame by location and assign cluster IDs.
        
        Parameters:
        - df: DataFrame to cluster
        - measurement: Measurement type label
        - water_body: Water body type label
        - eps_km: Clustering radius
        - cluster_id_start: Starting cluster ID for this measurement type
        - output_dir: Optional directory to save mapping files
        
        Returns:
        - Tuple of (clustered DataFrame, next available cluster ID)
        """
        df = df.copy()
        df['cluster_id'] = -1
        df['cluster_type_id'] = 'noise'
        
        # Validate coordinates
        valid_coords_mask = (
            df[self.lat_col].notna() & 
            df[self.lon_col].notna()
        )
        
        if valid_coords_mask.sum() < 1:
            print(f"  - No valid coordinates for {measurement}/{water_body}")
            return df, cluster_id_start
        
        valid_df = df[valid_coords_mask].copy()
        coords = valid_df[[self.lat_col, self.lon_col]].values
        
        # Perform DBSCAN
        dbscan = DBSCAN(eps=eps_km, min_samples=1, metric=self._haversine_distance)
        
        try:
            labels = dbscan.fit_predict(coords)
            
            # Assign cluster IDs starting from cluster_id_start
            current_id = cluster_id_start
            id_mapping = {}
            
            for unique_label in set(labels):
                if unique_label != -1:  # Not noise
                    id_mapping[unique_label] = current_id
                    current_id += 1
            
            # Assign cluster_id and cluster_type_id
            for idx, label in enumerate(labels):
                if label == -1:
                    valid_df.loc[valid_df.index[idx], 'cluster_id'] = -1
                    valid_df.loc[valid_df.index[idx], 'cluster_type_id'] = f"{measurement}_{water_body}_noise"
                else:
                    cluster_id = id_mapping[label]
                    valid_df.loc[valid_df.index[idx], 'cluster_id'] = cluster_id
                    valid_df.loc[valid_df.index[idx], 'cluster_type_id'] = f"{measurement}_{water_body}_{cluster_id}"
            
            # Update original dataframe
            df.loc[valid_df.index, 'cluster_id'] = valid_df['cluster_id']
            df.loc[valid_df.index, 'cluster_type_id'] = valid_df['cluster_type_id']
            
            # Save mapping file if output_dir provided
            if output_dir:
                self._save_mapping(df, measurement, water_body, output_dir)
            
            n_clusters = len(id_mapping)
            n_noise = (df['cluster_id'] == -1).sum()
            print(f"  - {measurement}/{water_body}: {n_clusters} clusters, {n_noise} noise points")
            
            return df, current_id
            
        except Exception as e:
            print(f"  - Error clustering {measurement}/{water_body}: {e}")
            return df, cluster_id_start
    
    def _save_mapping(self, df: pd.DataFrame, measurement: str, water_body: str, output_dir: str):
        """Save cluster mapping to file."""
        os.makedirs(os.path.join(output_dir, measurement), exist_ok=True)
        
        mapping_df = df[[self.site_id_col, 'cluster_id', 'cluster_type_id']].copy()
        mapping_df = mapping_df[mapping_df['cluster_id'] != -1]  # Exclude noise
        
        filename = os.path.join(output_dir, measurement, f"site_cluster_map_{water_body}.csv")
        mapping_df.to_csv(filename, index=False)
        print(f"    - Saved mapping to: {filename}")
    
    def cluster(self, df: pd.DataFrame, eps_km: float = None, eps_dict: Dict[str, float] = None, output_dir: str = None) -> pd.DataFrame:
        """
        Main clustering method that orchestrates the entire process.
        
        Parameters:
        - df: DataFrame to cluster
        - eps_km: Single clustering radius (deprecated, use eps_dict instead)
        - eps_dict: Dictionary mapping water body types to clustering radii
        - output_dir: Optional directory to save output files
        
        Returns:
        - Original DataFrame with cluster_id and cluster_type_id columns added
        """
        # Handle backwards compatibility
        if eps_km is not None and eps_dict is None:
            print(f"Warning: eps_km parameter is deprecated. Using it as default for all water bodies.")
            eps_dict = {}
            default_radius = eps_km
        else:
            default_radius = self.default_eps
            
        if eps_dict is None:
            eps_dict = {}
        
        print(f"Starting hierarchical clustering with custom radii per water body type")
        print(f"Default radius: {default_radius}km")
        if eps_dict:
            print("Custom radii:")
            for water_body, radius in eps_dict.items():
                print(f"  - {water_body}: {radius}km")
        
        # Step 1: Create unique sites dataframe
        unique_sites = self._create_unique_sites(df)
        
        # Initialize result dataframe for unique sites
        result_sites = unique_sites.copy()
        result_sites['cluster_id'] = -1
        result_sites['cluster_type_id'] = 'noise'
        
        # Split by measurement
        measurement_dfs = self.split_by_measurement(unique_sites)
        
        # Process each measurement type
        for measurement, measurement_df in measurement_dfs.items():
            print(f"\nProcessing measurement type: {measurement}")
            cluster_id_start = 0
            
            # Split by water body
            water_body_dfs = self.split_by_water_body(measurement_df)
            
            # Process each water body
            for water_body, water_body_df in water_body_dfs.items():
                # Get the appropriate eps value for this water body type
                current_eps = eps_dict.get(water_body, default_radius)
                print(f"  Processing water body: {water_body} (eps={current_eps}km)")
                
                # Cluster by location
                clustered_df, next_cluster_id = self.cluster_by_location(
                    df=water_body_df,
                    measurement=measurement,
                    water_body=water_body,
                    eps_km=current_eps,  # Use the specific eps for this water body
                    cluster_id_start=cluster_id_start,
                    output_dir=output_dir
                )
                
                # Update result dataframe
                result_sites.loc[clustered_df.index, 'cluster_id'] = clustered_df['cluster_id']
                result_sites.loc[clustered_df.index, 'cluster_type_id'] = clustered_df['cluster_type_id']
                
                # Update starting ID for next water body
                cluster_id_start = next_cluster_id
        
        # Step 2: Merge clusters back to original dataframe
        result_df = self._merge_clusters_back(df, result_sites)
        
        # Print summary
        self._print_summary(result_df)
        
        return result_df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print clustering summary statistics."""
        print("\nClustering complete!")
        
        # Get unique sites for accurate counts
        unique_sites = df.drop_duplicates(subset=[self.site_id_col, self.lat_col, self.lon_col])
        
        print(f"Total clusters: {unique_sites[unique_sites['cluster_id'] >= 0]['cluster_id'].nunique()}")
        print(f"Total noise points: {(unique_sites['cluster_id'] == -1).sum()}")
        
        # Group by summary - now counting unique sites, not all observations
        cluster_summary = unique_sites.groupby([self.measurement_col, self.water_body_col]).agg({
            'cluster_id': lambda x: len(set(x) - {-1}),  # Count unique cluster IDs (excluding -1)
            self.site_id_col: 'count'  # Count unique sites
        }).reset_index()
        cluster_summary.columns = ['measurement', 'water_body', 'n_clusters', 'n_sites']
        
        print("\nSummary by measurement and water body:")
        print(cluster_summary)
    
    def get_site_level_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics at the site level (not observation level).
        
        Parameters:
        - df: DataFrame with cluster assignments
        
        Returns:
        - DataFrame with site-level statistics
        """
        # Get unique sites first
        unique_sites = df.drop_duplicates(subset=[self.site_id_col, self.lat_col, self.lon_col])
        
        stats = unique_sites[unique_sites['cluster_id'] >= 0].groupby('cluster_type_id').agg({
            self.site_id_col: 'count',
            self.lat_col: ['mean', 'std'],
            self.lon_col: ['mean', 'std']
        }).reset_index()
        
        stats.columns = ['cluster_type_id', 'site_count', 'lat_mean', 'lat_std', 'lon_mean', 'lon_std']
        return stats