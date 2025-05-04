from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize


class DistributionTest(ABC):
    """Abstract base class for distribution tests."""
    
    @abstractmethod
    def fit(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit the distribution and return results."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this distribution test."""
        pass
    
    def _calculate_aic_bic(self, values: np.ndarray, log_likelihood: float, n_params: int) -> Dict[str, float]:
        """Calculate AIC and BIC for model comparison."""
        n = len(values)
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n) - 2 * log_likelihood
        return {'aic': aic, 'bic': bic}
    
    def _kolmogorov_smirnov_test(self, values: np.ndarray, cdf_func, *params) -> Dict[str, float]:
        """Perform Kolmogorov-Smirnov test."""
        ks_stat, p_value = stats.kstest(values, cdf_func, args=params)
        return {'ks_statistic': ks_stat, 'ks_pvalue': p_value}


class BetaTest(DistributionTest):
    """Beta distribution test for [0,1] bounded data."""
    
    def get_name(self) -> str:
        return "beta"
    
    def fit(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit beta distribution to the data."""
        # Handle edge cases (0 and 1 values)
        values = self._transform_for_beta(values)
        
        # Fit beta distribution
        try:
            params = stats.beta.fit(values, floc=0, fscale=1)
            alpha, beta_param, loc, scale = params
            
            # Calculate log-likelihood
            log_likelihood = np.sum(stats.beta.logpdf(values, alpha, beta_param, loc=loc, scale=scale))
            
            # Calculate AIC/BIC
            metrics = self._calculate_aic_bic(values, log_likelihood, 2)  # 2 parameters for beta
            
            # Perform KS test
            ks_results = self._kolmogorov_smirnov_test(values, 'beta', alpha, beta_param, loc, scale)
            
            return {
                'alpha': alpha,
                'beta': beta_param,
                'loc': loc,
                'scale': scale,
                'log_likelihood': log_likelihood,
                **metrics,
                **ks_results
            }
        except Exception as e:
            return {'error': f'Beta fitting failed: {str(e)}'}
    
    def _transform_for_beta(self, values: np.ndarray) -> np.ndarray:
        """Transform data to avoid exact 0 and 1 values."""
        n = len(values)
        # Common transformation: (x * (n-1) + 0.5) / n
        return (values * (n - 1) + 0.5) / n


class NormalTest(DistributionTest):
    """Normal distribution test."""
    
    def get_name(self) -> str:
        return "normal"
    
    def fit(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit normal distribution to the data."""
        try:
            params = stats.norm.fit(values)
            mu, sigma = params
            
            # Calculate log-likelihood
            log_likelihood = np.sum(stats.norm.logpdf(values, mu, sigma))
            
            # Calculate AIC/BIC
            metrics = self._calculate_aic_bic(values, log_likelihood, 2)  # 2 parameters
            
            # Perform KS test
            ks_results = self._kolmogorov_smirnov_test(values, 'norm', mu, sigma)
            
            return {
                'mu': mu,
                'sigma': sigma,
                'log_likelihood': log_likelihood,
                **metrics,
                **ks_results
            }
        except Exception as e:
            return {'error': f'Normal fitting failed: {str(e)}'}


class LogitNormalTest(DistributionTest):
    """Logit-normal distribution test for [0,1] bounded data."""
    
    def get_name(self) -> str:
        return "logit_normal"
    
    def fit(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit logit-normal distribution to the data."""
        # Transform for (0,1) interval
        values = self._transform_for_logit(values)
        
        try:
            # Transform to logit scale
            logit_values = np.log(values / (1 - values))
            
            # Fit normal distribution to logit-transformed values
            mu, sigma = stats.norm.fit(logit_values)
            
            # Calculate log-likelihood on original scale
            log_likelihood = self._logit_normal_logpdf(values, mu, sigma)
            
            # Calculate AIC/BIC
            metrics = self._calculate_aic_bic(values, log_likelihood, 2)
            
            # KS test with custom CDF
            ks_stat = self._custom_ks_test(values, mu, sigma)
            
            return {
                'mu': mu,
                'sigma': sigma,
                'log_likelihood': log_likelihood,
                'ks_statistic': ks_stat,
                'ks_pvalue': None,  # p-value calculation for custom distributions is complex
                **metrics
            }
        except Exception as e:
            return {'error': f'Logit-normal fitting failed: {str(e)}'}
    
    def _transform_for_logit(self, values: np.ndarray) -> np.ndarray:
        """Transform data to avoid exact 0 and 1 values."""
        n = len(values)
        return (values * (n - 1) + 0.5) / n
    
    def _logit_normal_logpdf(self, values: np.ndarray, mu: float, sigma: float) -> float:
        """Calculate log PDF for logit-normal distribution."""
        logit_values = np.log(values / (1 - values))
        log_pdf = stats.norm.logpdf(logit_values, mu, sigma)
        # Add the Jacobian term
        log_pdf += np.log(1 / (values * (1 - values)))
        return np.sum(log_pdf)
    
    def _custom_ks_test(self, values: np.ndarray, mu: float, sigma: float) -> float:
        """Custom KS test for logit-normal distribution."""
        n = len(values)
        sorted_values = np.sort(values)
        empirical_cdf = np.arange(1, n + 1) / n
        
        # Calculate theoretical CDF
        logit_values = np.log(sorted_values / (1 - sorted_values))
        theoretical_cdf = stats.norm.cdf(logit_values, mu, sigma)
        
        # Calculate KS statistic
        ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
        return ks_stat


class UniformTest(DistributionTest):
    """Uniform distribution test."""
    
    def get_name(self) -> str:
        return "uniform"
    
    def fit(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit uniform distribution to the data."""
        try:
            params = stats.uniform.fit(values)
            loc, scale = params
            
            # Calculate log-likelihood
            log_likelihood = np.sum(stats.uniform.logpdf(values, loc, scale))
            
            # Calculate AIC/BIC
            metrics = self._calculate_aic_bic(values, log_likelihood, 2)
            
            # Perform KS test
            ks_results = self._kolmogorov_smirnov_test(values, 'uniform', loc, scale)
            
            return {
                'loc': loc,
                'scale': scale,
                'log_likelihood': log_likelihood,
                **metrics,
                **ks_results
            }
        except Exception as e:
            return {'error': f'Uniform fitting failed: {str(e)}'}


class DistributionTestPipeline:
    """
    A flexible pipeline for testing multiple distributions on water quality data.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 cluster_col: str,
                 measurement_cols: Dict[str, str]):
        """
        Initialize the distribution testing pipeline.
        
        Args:
            data: DataFrame with cluster assignments and measurement values
            cluster_col: Column containing cluster identifiers
            measurement_cols: Dictionary mapping measurement types to value columns
        """
        self.data = data.copy()
        self.cluster_col = cluster_col
        self.measurement_cols = measurement_cols
        self.tests: List[DistributionTest] = []
        
        # Validate columns
        self._validate_columns()
    
    def add_test(self, test: DistributionTest) -> 'DistributionTestPipeline':
        """Add a distribution test to the pipeline."""
        self.tests.append(test)
        return self
    
    def add_tests(self, tests: List[DistributionTest]) -> 'DistributionTestPipeline':
        """Add multiple distribution tests to the pipeline."""
        self.tests.extend(tests)
        return self
    
    def _validate_columns(self):
        """Validate that specified columns exist."""
        if self.cluster_col not in self.data.columns:
            raise ValueError(f"Cluster column '{self.cluster_col}' not found")
            
        for m_type, col in self.measurement_cols.items():
            if col not in self.data.columns:
                print(f"Warning: Column '{col}' for measurement type '{m_type}' not found")
    
    def _extract_measurement_type(self, cluster_id: str) -> Optional[str]:
        """Extract measurement type from cluster ID."""
        cluster_id_lower = cluster_id.lower()
        for m_type in self.measurement_cols.keys():
            if m_type.lower() in cluster_id_lower:
                return m_type
        return None
    
    def run_single_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """Run all tests on a single cluster."""
        # Extract measurement type
        measurement_type = self._extract_measurement_type(cluster_id)
        if measurement_type is None:
            return {'error': f'Could not determine measurement type for cluster {cluster_id}'}
        
        # Get the appropriate value column
        value_col = self.measurement_cols.get(measurement_type)
        if value_col is None:
            return {'error': f'No value column found for measurement type {measurement_type}'}
        
        # Get data for this cluster
        cluster_data = self.data[self.data[self.cluster_col] == cluster_id]
        values = cluster_data[value_col].dropna().values
        
        if len(values) < 5:  # Minimum sample size
            return {'error': f'Insufficient data points ({len(values)})'}
        
        # Run all tests
        results = {
            'cluster_id': cluster_id,
            'measurement_type': measurement_type,
            'value_column': value_col,
            'n_samples': len(values),
            'test_results': {}
        }
        
        for test in self.tests:
            try:
                results['test_results'][test.get_name()] = test.fit(values)
            except Exception as e:
                results['test_results'][test.get_name()] = {'error': str(e)}
        
        # Find best distribution based on AIC
        results['best_distribution'] = self._find_best_distribution(results['test_results'])
        
        return results
    
    def _find_best_distribution(self, test_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Find the best fitting distribution based on AIC."""
        best_dist = None
        best_aic = float('inf')
        
        for dist_name, dist_results in test_results.items():
            if 'error' not in dist_results and 'aic' in dist_results:
                if dist_results['aic'] < best_aic:
                    best_aic = dist_results['aic']
                    best_dist = dist_name
        
        if best_dist is None:
            return {'name': 'none', 'aic': None}
        
        return {
            'name': best_dist,
            'aic': best_aic,
            'pvalue': test_results[best_dist].get('ks_pvalue', None)
        }
    
    def run_all_clusters(self, parallel: bool = False) -> pd.DataFrame:
        """Run all tests on all clusters."""
        results = []
        cluster_ids = self.data[self.cluster_col].unique()
        
        print(f"Testing {len(cluster_ids)} clusters...")
        
        for i, cluster_id in enumerate(cluster_ids):
            if i % 100 == 0:
                print(f"Processing cluster {i+1}/{len(cluster_ids)}")
            
            cluster_results = self.run_single_cluster(cluster_id)
            results.append(cluster_results)
        
        # Convert to DataFrame
        return self._results_to_dataframe(results)
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert list of results to a structured DataFrame."""
        rows = []
        
        for result in results:
            if 'error' in result:
                rows.append({
                    'cluster_id': result.get('cluster_id', 'unknown'),
                    'error': result['error']
                })
                continue
            
            row = {
                'cluster_id': result['cluster_id'],
                'measurement_type': result['measurement_type'],
                'value_column': result['value_column'],
                'n_samples': result['n_samples'],
                'best_distribution': result['best_distribution']['name'],
                'best_aic': result['best_distribution']['aic']
            }
            
            # Add test results
            for test_name, test_result in result['test_results'].items():
                if 'error' in test_result:
                    row[f'{test_name}_error'] = test_result['error']
                else:
                    # Flatten test results
                    for key, value in test_result.items():
                        row[f'{test_name}_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a summary report from the test results."""
        report = {
            'total_clusters': len(results_df),
            'clusters_with_errors': len(results_df[results_df.get('error', '').notna()]),
            'distribution_counts': results_df['best_distribution'].value_counts().to_dict(),
            'measurement_type_summaries': {}
        }
        
        # Summarize by measurement type
        for m_type in results_df['measurement_type'].unique():
            if pd.notna(m_type):
                m_data = results_df[results_df['measurement_type'] == m_type]
                report['measurement_type_summaries'][m_type] = {
                    'n_clusters': len(m_data),
                    'distribution_counts': m_data['best_distribution'].value_counts().to_dict(),
                    'avg_n_samples': m_data['n_samples'].mean()
                }
        
        return report

class InflatedBetaTest(DistributionTest):
    """One-inflated beta distribution test for [0,1] bounded data with excess 1s."""
    
    def get_name(self) -> str:
        return "inflated_beta"
    
    def fit(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit one-inflated beta distribution to the data."""
        try:
            # Separate inflated (=1) and non-inflated values
            is_one = values == 1.0
            p_inflate = np.mean(is_one)
            
            # If no inflation or all inflated, return simple result
            if p_inflate == 0 or p_inflate == 1:
                return {
                    'error': f'No mixture needed: p_inflate={p_inflate}',
                    'p_inflate': p_inflate
                }
            
            # Fit beta to non-inflated values
            non_inflated = values[~is_one]
            # Transform to avoid boundary issues
            non_inflated = self._transform_for_beta(non_inflated)
            
            # Fit beta distribution
            params = stats.beta.fit(non_inflated, floc=0, fscale=1)
            alpha, beta_param, loc, scale = params
            
            # Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood(values, p_inflate, alpha, beta_param)
            
            # Number of parameters: p_inflate, alpha, beta
            n_params = 3
            
            # Calculate AIC/BIC
            metrics = self._calculate_aic_bic(values, log_likelihood, n_params)
            
            return {
                'p_inflate': p_inflate,
                'alpha': alpha,
                'beta': beta_param,
                'log_likelihood': log_likelihood,
                **metrics,
                'n_ones': np.sum(is_one),
                'n_total': len(values)
            }
            
        except Exception as e:
            return {'error': f'Inflated beta fitting failed: {str(e)}'}
    
    def _transform_for_beta(self, values: np.ndarray) -> np.ndarray:
        """Transform data to avoid exact 0 and 1 values."""
        n = len(values)
        # Only transform values that are exactly 1 (move them slightly away)
        transformed = values.copy()
        transformed[transformed == 1] = 1 - 1/(2*n)
        transformed[transformed == 0] = 1/(2*n)
        return transformed
    
    def _calculate_log_likelihood(self, values: np.ndarray, p_inflate: float, 
                                alpha: float, beta_param: float) -> float:
        """Calculate log-likelihood for inflated beta distribution."""
        is_one = values == 1.0
        
        # Log likelihood for inflated values
        ll_inflated = np.sum(np.log(p_inflate) * is_one)
        
        # Log likelihood for non-inflated values
        if np.sum(~is_one) > 0:
            non_inflated = values[~is_one]
            # Transform to avoid boundary issues for beta pdf
            non_inflated_transformed = self._transform_for_beta(non_inflated)
            ll_non_inflated = np.sum(np.log(1 - p_inflate) + 
                                   stats.beta.logpdf(non_inflated_transformed, alpha, beta_param))
        else:
            ll_non_inflated = 0
            
        return ll_inflated + ll_non_inflated

class InflatedGammaTest(DistributionTest):
    """Inflated gamma distribution test for positive data with an excess at a boundary value."""
    
    def get_name(self) -> str:
        return "inflated_gamma"
    
    def fit(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit inflated gamma distribution to the data."""
        try:
            # Find the most common maximum value (potential inflation point)
            value_counts = pd.Series(values).value_counts()
            max_value = values.max()
            
            # Check if max value is inflated
            max_count = value_counts.get(max_value, 0)
            total_count = len(values)
            
            # If max value appears more than expected (e.g., >10% of data), treat as inflated
            if max_count / total_count > 0.1:
                inflate_value = max_value
            else:
                # Find the most frequent value
                inflate_value = value_counts.idxmax()
            
            # Separate inflated and non-inflated values
            is_inflated = values == inflate_value
            p_inflate = np.mean(is_inflated)
            
            # If no inflation or all inflated, return simple result
            if p_inflate == 0 or p_inflate == 1:
                return {
                    'error': f'No mixture needed: p_inflate={p_inflate}',
                    'p_inflate': p_inflate,
                    'inflate_value': inflate_value
                }
            
            # Fit gamma to non-inflated values
            non_inflated = values[~is_inflated]
            
            # Fit gamma distribution
            params = stats.gamma.fit(non_inflated, floc=0)
            shape, loc, scale = params
            
            # Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood(values, p_inflate, inflate_value, 
                                                          shape, loc, scale)
            
            # Number of parameters: p_inflate, shape, scale (loc fixed at 0)
            n_params = 3
            
            # Calculate AIC/BIC
            metrics = self._calculate_aic_bic(values, log_likelihood, n_params)
            
            return {
                'p_inflate': p_inflate,
                'inflate_value': inflate_value,
                'shape': shape,
                'loc': loc,
                'scale': scale,
                'log_likelihood': log_likelihood,
                **metrics,
                'n_inflated': np.sum(is_inflated),
                'n_total': len(values)
            }
            
        except Exception as e:
            return {'error': f'Inflated gamma fitting failed: {str(e)}'}
    
    def _calculate_log_likelihood(self, values: np.ndarray, p_inflate: float, 
                                inflate_value: float, shape: float, loc: float, 
                                scale: float) -> float:
        """Calculate log-likelihood for inflated gamma distribution."""
        is_inflated = values == inflate_value
        
        # Log likelihood for inflated values
        ll_inflated = np.sum(np.log(p_inflate) * is_inflated)
        
        # Log likelihood for non-inflated values
        if np.sum(~is_inflated) > 0:
            non_inflated = values[~is_inflated]
            ll_non_inflated = np.sum(np.log(1 - p_inflate) + 
                                   stats.gamma.logpdf(non_inflated, shape, loc=loc, scale=scale))
        else:
            ll_non_inflated = 0
            
        return ll_inflated + ll_non_inflated

# Also add standard Gamma test since we'll need it
class GammaTest(DistributionTest):
    """Gamma distribution test for positive continuous data."""
    
    def get_name(self) -> str:
        return "gamma"
    
    def fit(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit gamma distribution to the data."""
        try:
            # Fit gamma distribution
            params = stats.gamma.fit(values, floc=0)  # Fix location at 0
            shape, loc, scale = params
            
            # Calculate log-likelihood
            log_likelihood = np.sum(stats.gamma.logpdf(values, shape, loc=loc, scale=scale))
            
            # Calculate AIC/BIC (shape and scale parameters)
            metrics = self._calculate_aic_bic(values, log_likelihood, 2)
            
            # Perform KS test
            ks_results = self._kolmogorov_smirnov_test(values, 'gamma', shape, loc, scale)
            
            return {
                'shape': shape,
                'loc': loc,
                'scale': scale,
                'log_likelihood': log_likelihood,
                **metrics,
                **ks_results
            }
        except Exception as e:
            return {'error': f'Gamma fitting failed: {str(e)}'}

# Example usage function
def run_distribution_testing(data: pd.DataFrame, 
                           cluster_col: str, 
                           measurement_cols: Dict[str, str]) -> pd.DataFrame:
    """
    Convenience function to run distribution testing with default tests.
    
    Args:
        data: DataFrame with cluster assignments and measurement values
        cluster_col: Column containing cluster identifiers
        measurement_cols: Dictionary mapping measurement types to value columns
        
    Returns:
        DataFrame with test results
    """
    # Initialize pipeline
    pipeline = DistributionTestPipeline(data, cluster_col, measurement_cols)
    
    # Add standard tests
    pipeline.add_tests([
        BetaTest(),
        NormalTest(),
        LogitNormalTest(),
        UniformTest()
    ])
    
    # Run tests
    results = pipeline.run_all_clusters()
    
    # Generate summary
    summary = pipeline.generate_summary_report(results)
    print("\nDistribution Testing Summary:")
    print(f"Total clusters tested: {summary['total_clusters']}")
    print(f"Clusters with errors: {summary['clusters_with_errors']}")
    print("\nBest fitting distributions:")
    for dist, count in summary['distribution_counts'].items():
        print(f"  {dist}: {count}")
    
    return results