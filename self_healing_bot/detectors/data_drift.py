"""Data drift detection and monitoring with advanced statistical methods."""

from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
import hashlib
from collections import defaultdict
import warnings

from .base import BaseDetector
from ..core.context import Context

logger = logging.getLogger(__name__)


class DataDriftDetector(BaseDetector):
    """Detect data drift in ML pipelines with comprehensive statistical analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Drift detection thresholds
        self.drift_threshold = self.config.get("drift_threshold", 0.1)
        self.psi_threshold = self.config.get("psi_threshold", 0.1)
        self.ks_threshold = self.config.get("ks_threshold", 0.05)
        self.chi2_threshold = self.config.get("chi2_threshold", 0.05)
        self.js_threshold = self.config.get("js_threshold", 0.1)
        
        # Statistical test configuration
        self.statistical_tests = self.config.get("statistical_tests", ["ks", "psi", "chi2", "js"])
        self.min_sample_size = self.config.get("min_sample_size", 100)
        self.confidence_level = self.config.get("confidence_level", 0.95)
        
        # Historical tracking
        self.drift_history = defaultdict(list)
        self.baseline_data_cache = {}
        self.window_hours = self.config.get("window_hours", 168)  # 1 week
        
        # Feature importance for prioritized monitoring
        self.feature_weights = self.config.get("feature_weights", {})
        self.critical_features = set(self.config.get("critical_features", []))
        
        # Data quality thresholds
        self.missing_threshold = self.config.get("missing_threshold", 0.1)
        self.outlier_threshold = self.config.get("outlier_threshold", 3.0)  # z-score
        
        # Alert configuration
        self.alert_cooldown = self.config.get("alert_cooldown_hours", 6)
        self.last_alerts = {}
    
    def get_supported_events(self) -> List[str]:
        return ["push", "schedule", "workflow_run", "data_pipeline"]
    
    async def detect(self, context: Context) -> List[Dict[str, Any]]:
        """Detect data drift using comprehensive statistical methods."""
        issues = []
        
        try:
            # Get training and production data
            training_data = await self._get_training_data(context)
            production_data = await self._get_production_data(context)
            
            if not training_data or not production_data:
                logger.warning("Unable to fetch data for drift detection")
                return issues
            
            # Validate data quality first
            data_quality_issues = await self._check_data_quality(production_data, context)
            issues.extend(data_quality_issues)
            
            # Perform comprehensive drift detection
            drift_results = await self._detect_comprehensive_drift(training_data, production_data)
            
            # Generate drift issues
            for feature_name, drift_info in drift_results.items():
                if drift_info["drift_detected"]:
                    # Check alert cooldown
                    if not self._should_alert(feature_name):
                        continue
                    
                    severity = self._get_drift_severity(feature_name, drift_info)
                    
                    issues.append(self.create_issue(
                        issue_type="data_drift",
                        severity=severity,
                        message=f"Data drift detected in feature '{feature_name}' using {drift_info['best_test']} (score: {drift_info['drift_score']:.4f})",
                        data={
                            "feature_name": feature_name,
                            "drift_score": drift_info["drift_score"],
                            "test_method": drift_info["best_test"],
                            "p_value": drift_info.get("p_value"),
                            "confidence_interval": drift_info.get("confidence_interval"),
                            "effect_size": drift_info.get("effect_size"),
                            "threshold": self._get_threshold_for_test(drift_info["best_test"]),
                            "statistics": drift_info.get("all_test_results", {}),
                            "distribution_comparison": drift_info.get("distribution_stats", {}),
                            "recommendation": self._get_drift_recommendation(feature_name, drift_info),
                            "historical_trend": self._get_historical_trend(feature_name),
                            "feature_importance": self.feature_weights.get(feature_name, 0.0)
                        }
                    ))
                    
                    # Update alert history
                    self.last_alerts[feature_name] = datetime.utcnow()
                    
                    # Track in history
                    self._track_drift_history(feature_name, drift_info)
            
            # Check for systematic drift across multiple features
            systematic_drift_issues = await self._detect_systematic_drift(drift_results)
            issues.extend(systematic_drift_issues)
            
        except Exception as e:
            logger.exception(f"Error in data drift detection: {e}")
            issues.append(self.create_issue(
                issue_type="drift_detection_error",
                severity="medium",
                message=f"Data drift detection failed: {str(e)}",
                data={"error_details": str(e)}
            ))
        
        return issues
    
    async def _get_training_data(self, context: Context) -> Optional[Dict[str, np.ndarray]]:
        """Get training/reference data for drift comparison."""
        cache_key = f"training_data_{context.repo_full_name}"
        
        if cache_key in self.baseline_data_cache:
            cached_data, timestamp = self.baseline_data_cache[cache_key]
            # Refresh cache if older than 24 hours
            if datetime.utcnow() - timestamp < timedelta(hours=24):
                return cached_data
        
        # Mock implementation - in production, this would fetch from data storage
        np.random.seed(42)
        training_data = {
            "numerical_feature_1": np.random.normal(0, 1, 2000),
            "numerical_feature_2": np.random.exponential(2, 2000),
            "numerical_feature_3": np.random.uniform(0, 10, 2000),
            "categorical_feature_1": np.random.choice(['A', 'B', 'C'], 2000, p=[0.5, 0.3, 0.2]),
            "categorical_feature_2": np.random.choice(['X', 'Y', 'Z'], 2000, p=[0.4, 0.4, 0.2])
        }
        
        # Cache the data
        self.baseline_data_cache[cache_key] = (training_data, datetime.utcnow())
        return training_data
    
    async def _get_production_data(self, context: Context) -> Optional[Dict[str, np.ndarray]]:
        """Get recent production data for drift comparison."""
        # Mock implementation with varying degrees of drift
        np.random.seed(123 + int(datetime.utcnow().hour))  # Time-based seed for variation
        
        # Simulate different drift scenarios
        drift_factor = np.random.uniform(0.8, 1.5)  # Random drift intensity
        
        production_data = {
            # Mild drift in numerical features
            "numerical_feature_1": np.random.normal(0.1 * drift_factor, 1.1, 800),
            "numerical_feature_2": np.random.exponential(2.2 * drift_factor, 800),
            
            # Significant drift in feature 3
            "numerical_feature_3": np.random.uniform(2 * drift_factor, 12 * drift_factor, 800),
            
            # Category distribution shift
            "categorical_feature_1": np.random.choice(['A', 'B', 'C'], 800, 
                                                    p=[0.3, 0.4, 0.3] if drift_factor > 1.2 
                                                    else [0.5, 0.3, 0.2]),
            "categorical_feature_2": np.random.choice(['X', 'Y', 'Z'], 800, p=[0.6, 0.3, 0.1])
        }
        
        return production_data
    
    async def _check_data_quality(self, data: Dict[str, np.ndarray], context: Context) -> List[Dict[str, Any]]:
        """Check data quality issues that might affect drift detection."""
        issues = []
        
        for feature_name, values in data.items():
            # Check for missing values (represented as NaN)
            if hasattr(values, 'dtype') and np.issubdtype(values.dtype, np.floating):
                missing_ratio = np.isnan(values).mean()
                if missing_ratio > self.missing_threshold:
                    issues.append(self.create_issue(
                        issue_type="data_quality_missing",
                        severity="medium",
                        message=f"High missing value ratio in feature '{feature_name}': {missing_ratio:.2%}",
                        data={
                            "feature_name": feature_name,
                            "missing_ratio": missing_ratio,
                            "threshold": self.missing_threshold,
                            "recommendation": "Investigate data collection issues and consider imputation strategies"
                        }
                    ))
            
            # Check for outliers in numerical features
            if hasattr(values, 'dtype') and np.issubdtype(values.dtype, np.number):
                clean_values = values[~np.isnan(values)] if np.any(np.isnan(values)) else values
                if len(clean_values) > 0:
                    z_scores = np.abs(stats.zscore(clean_values))
                    outlier_ratio = (z_scores > self.outlier_threshold).mean()
                    
                    if outlier_ratio > 0.05:  # More than 5% outliers
                        issues.append(self.create_issue(
                            issue_type="data_quality_outliers",
                            severity="low",
                            message=f"High outlier ratio in feature '{feature_name}': {outlier_ratio:.2%}",
                            data={
                                "feature_name": feature_name,
                                "outlier_ratio": outlier_ratio,
                                "outlier_threshold": self.outlier_threshold,
                                "recommendation": "Review data preprocessing and outlier handling"
                            }
                        ))
            
            # Check for constant values
            unique_values = len(np.unique(values))
            if unique_values == 1:
                issues.append(self.create_issue(
                    issue_type="data_quality_constant",
                    severity="high",
                    message=f"Feature '{feature_name}' has constant values",
                    data={
                        "feature_name": feature_name,
                        "unique_values": unique_values,
                        "recommendation": "Remove constant features or check data collection pipeline"
                    }
                ))
        
        return issues
    
    async def _detect_comprehensive_drift(self, training_data: Dict[str, np.ndarray], 
                                        production_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Perform comprehensive drift detection using multiple statistical tests."""
        drift_results = {}
        
        common_features = set(training_data.keys()) & set(production_data.keys())
        
        for feature_name in common_features:
            train_values = training_data[feature_name]
            prod_values = production_data[feature_name]
            
            # Skip if insufficient data
            if len(train_values) < self.min_sample_size or len(prod_values) < self.min_sample_size:
                continue
            
            # Determine if feature is numerical or categorical
            is_numerical = self._is_numerical_feature(train_values)
            
            if is_numerical:
                drift_info = await self._detect_numerical_drift(train_values, prod_values, feature_name)
            else:
                drift_info = await self._detect_categorical_drift(train_values, prod_values, feature_name)
            
            drift_results[feature_name] = drift_info
        
        return drift_results
    
    def _is_numerical_feature(self, values: np.ndarray) -> bool:
        """Determine if a feature is numerical or categorical."""
        return np.issubdtype(values.dtype, np.number) and len(np.unique(values)) > 10
    
    async def _detect_numerical_drift(self, train_data: np.ndarray, prod_data: np.ndarray, 
                                    feature_name: str) -> Dict[str, Any]:
        """Detect drift in numerical features using multiple statistical tests."""
        results = {
            "drift_detected": False,
            "drift_score": 0.0,
            "best_test": "none",
            "all_test_results": {},
            "distribution_stats": {},
            "effect_size": 0.0
        }
        
        try:
            # Clean data (remove NaN values)
            train_clean = train_data[~np.isnan(train_data)] if np.any(np.isnan(train_data)) else train_data
            prod_clean = prod_data[~np.isnan(prod_data)] if np.any(np.isnan(prod_data)) else prod_data
            
            # Calculate distribution statistics
            results["distribution_stats"] = {
                "train_mean": float(np.mean(train_clean)),
                "prod_mean": float(np.mean(prod_clean)),
                "train_std": float(np.std(train_clean)),
                "prod_std": float(np.std(prod_clean)),
                "train_median": float(np.median(train_clean)),
                "prod_median": float(np.median(prod_clean)),
                "mean_shift": float(np.mean(prod_clean) - np.mean(train_clean)),
                "std_ratio": float(np.std(prod_clean) / np.std(train_clean)) if np.std(train_clean) > 0 else 1.0
            }
            
            # Kolmogorov-Smirnov test
            if "ks" in self.statistical_tests:
                ks_stat, ks_p = stats.ks_2samp(train_clean, prod_clean)
                results["all_test_results"]["ks"] = {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "significant": ks_p < self.ks_threshold
                }
                
                if ks_p < self.ks_threshold and ks_stat > results["drift_score"]:
                    results.update({
                        "drift_detected": True,
                        "drift_score": float(ks_stat),
                        "best_test": "kolmogorov_smirnov",
                        "p_value": float(ks_p)
                    })
            
            # Population Stability Index
            if "psi" in self.statistical_tests:
                psi_score = self._calculate_psi(train_clean, prod_clean)
                results["all_test_results"]["psi"] = {
                    "statistic": float(psi_score),
                    "significant": psi_score > self.psi_threshold
                }
                
                if psi_score > self.psi_threshold and psi_score > results["drift_score"]:
                    results.update({
                        "drift_detected": True,
                        "drift_score": float(psi_score),
                        "best_test": "population_stability_index"
                    })
            
            # Jensen-Shannon divergence
            if "js" in self.statistical_tests:
                js_divergence = self._calculate_js_divergence(train_clean, prod_clean)
                results["all_test_results"]["js"] = {
                    "statistic": float(js_divergence),
                    "significant": js_divergence > self.js_threshold
                }
                
                if js_divergence > self.js_threshold and js_divergence > results["drift_score"]:
                    results.update({
                        "drift_detected": True,
                        "drift_score": float(js_divergence),
                        "best_test": "jensen_shannon_divergence"
                    })
            
            # Mann-Whitney U test for distribution shift
            if "mannwhitney" in self.statistical_tests:
                mw_stat, mw_p = stats.mannwhitneyu(train_clean, prod_clean, alternative='two-sided')
                results["all_test_results"]["mannwhitney"] = {
                    "statistic": float(mw_stat),
                    "p_value": float(mw_p),
                    "significant": mw_p < 0.05
                }
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(train_clean) - 1) * np.var(train_clean, ddof=1) + 
                                (len(prod_clean) - 1) * np.var(prod_clean, ddof=1)) / 
                               (len(train_clean) + len(prod_clean) - 2))
            
            if pooled_std > 0:
                cohens_d = (np.mean(prod_clean) - np.mean(train_clean)) / pooled_std
                results["effect_size"] = float(abs(cohens_d))
        
        except Exception as e:
            logger.exception(f"Error in numerical drift detection for feature {feature_name}: {e}")
        
        return results
    
    async def _detect_categorical_drift(self, train_data: np.ndarray, prod_data: np.ndarray, 
                                      feature_name: str) -> Dict[str, Any]:
        """Detect drift in categorical features using statistical tests."""
        results = {
            "drift_detected": False,
            "drift_score": 0.0,
            "best_test": "none",
            "all_test_results": {},
            "distribution_stats": {}
        }
        
        try:
            # Get unique categories and their frequencies
            all_categories = np.unique(np.concatenate([train_data, prod_data]))
            
            train_counts = np.array([np.sum(train_data == cat) for cat in all_categories])
            prod_counts = np.array([np.sum(prod_data == cat) for cat in all_categories])
            
            train_freq = train_counts / len(train_data)
            prod_freq = prod_counts / len(prod_data)
            
            # Store distribution statistics
            results["distribution_stats"] = {
                "categories": all_categories.tolist(),
                "train_frequencies": train_freq.tolist(),
                "prod_frequencies": prod_freq.tolist(),
                "frequency_shifts": (prod_freq - train_freq).tolist()
            }
            
            # Chi-square test
            if "chi2" in self.statistical_tests:
                # Prepare contingency table
                contingency = np.array([train_counts, prod_counts])
                
                # Avoid zero counts by adding small constant
                contingency = contingency + 1
                
                chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency)
                results["all_test_results"]["chi2"] = {
                    "statistic": float(chi2_stat),
                    "p_value": float(chi2_p),
                    "degrees_of_freedom": int(dof),
                    "significant": chi2_p < self.chi2_threshold
                }
                
                if chi2_p < self.chi2_threshold:
                    results.update({
                        "drift_detected": True,
                        "drift_score": float(chi2_stat),
                        "best_test": "chi_square",
                        "p_value": float(chi2_p)
                    })
            
            # Population Stability Index for categorical data
            if "psi" in self.statistical_tests:
                # Avoid division by zero
                train_freq_safe = np.where(train_freq == 0, 1e-10, train_freq)
                prod_freq_safe = np.where(prod_freq == 0, 1e-10, prod_freq)
                
                psi_score = np.sum((prod_freq_safe - train_freq_safe) * 
                                 np.log(prod_freq_safe / train_freq_safe))
                
                results["all_test_results"]["psi"] = {
                    "statistic": float(psi_score),
                    "significant": psi_score > self.psi_threshold
                }
                
                if psi_score > self.psi_threshold and psi_score > results["drift_score"]:
                    results.update({
                        "drift_detected": True,
                        "drift_score": float(psi_score),
                        "best_test": "population_stability_index"
                    })
            
            # Total Variation Distance
            if "tvd" in self.statistical_tests:
                tvd_score = 0.5 * np.sum(np.abs(prod_freq - train_freq))
                results["all_test_results"]["tvd"] = {
                    "statistic": float(tvd_score),
                    "significant": tvd_score > 0.1  # 10% threshold
                }
        
        except Exception as e:
            logger.exception(f"Error in categorical drift detection for feature {feature_name}: {e}")
        
        return results
    
    async def _detect_systematic_drift(self, drift_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect systematic drift across multiple features."""
        issues = []
        
        # Count features with drift
        drift_features = [name for name, info in drift_results.items() if info["drift_detected"]]
        total_features = len(drift_results)
        
        if total_features == 0:
            return issues
        
        drift_ratio = len(drift_features) / total_features
        
        # Check for widespread drift
        if drift_ratio > 0.5 and len(drift_features) >= 3:
            issues.append(self.create_issue(
                issue_type="systematic_data_drift",
                severity="critical",
                message=f"Systematic drift detected across {len(drift_features)} of {total_features} features ({drift_ratio:.1%})",
                data={
                    "affected_features": drift_features,
                    "drift_ratio": drift_ratio,
                    "total_features": total_features,
                    "recommendation": "Investigate data collection pipeline and consider model retraining"
                }
            ))
        
        # Check for critical feature drift
        critical_drift_features = [name for name in drift_features if name in self.critical_features]
        if critical_drift_features:
            issues.append(self.create_issue(
                issue_type="critical_feature_drift",
                severity="high",
                message=f"Drift detected in critical features: {', '.join(critical_drift_features)}",
                data={
                    "critical_features": critical_drift_features,
                    "recommendation": "Immediate attention required - critical features showing drift"
                }
            ))
        
        return issues
    
    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index with improved binning."""
        try:
            # Handle edge cases
            if len(expected) == 0 or len(actual) == 0:
                return 0.0
            
            # For very small datasets, reduce number of buckets
            if len(expected) < 100:
                buckets = min(buckets, len(expected) // 10, 5)
                buckets = max(buckets, 2)
            
            # Create buckets based on expected distribution quantiles
            breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
            breakpoints = np.unique(breakpoints)  # Remove duplicates
            
            if len(breakpoints) <= 2:
                return 0.0  # Cannot calculate PSI with too few buckets
            
            # Calculate frequencies
            expected_freq = np.histogram(expected, bins=breakpoints)[0]
            actual_freq = np.histogram(actual, bins=breakpoints)[0]
            
            # Normalize to proportions
            expected_prop = expected_freq / len(expected)
            actual_prop = actual_freq / len(actual)
            
            # Avoid division by zero - add small constant
            expected_prop = np.where(expected_prop == 0, 1e-10, expected_prop)
            actual_prop = np.where(actual_prop == 0, 1e-10, actual_prop)
            
            # Calculate PSI
            psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
            
            return max(0.0, psi)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0
    
    def _calculate_js_divergence(self, p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
        """Calculate Jensen-Shannon divergence with improved binning."""
        try:
            # Determine appropriate number of bins
            total_samples = len(p) + len(q)
            bins = min(bins, int(np.sqrt(total_samples)), max(10, total_samples // 20))
            
            # Create common bin edges
            combined_data = np.concatenate([p, q])
            bin_edges = np.linspace(np.min(combined_data), np.max(combined_data), bins + 1)
            
            # Handle edge case where all values are the same
            if np.min(combined_data) == np.max(combined_data):
                return 0.0
            
            # Create histograms
            p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
            q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
            
            # Normalize to probability distributions
            p_hist = p_hist / np.sum(p_hist) if np.sum(p_hist) > 0 else p_hist
            q_hist = q_hist / np.sum(q_hist) if np.sum(q_hist) > 0 else q_hist
            
            # Avoid zeros by adding small constant
            p_hist = np.where(p_hist == 0, 1e-10, p_hist)
            q_hist = np.where(q_hist == 0, 1e-10, q_hist)
            
            # Calculate JS divergence
            m = 0.5 * (p_hist + q_hist)
            js_div = 0.5 * stats.entropy(p_hist, m) + 0.5 * stats.entropy(q_hist, m)
            
            # Handle potential NaN/inf values
            if np.isnan(js_div) or np.isinf(js_div):
                return 0.0
            
            return float(js_div)
            
        except Exception as e:
            logger.warning(f"JS divergence calculation failed: {e}")
            return 0.0
    
    def _get_threshold_for_test(self, test_name: str) -> float:
        """Get threshold value for specific statistical test."""
        threshold_map = {
            "kolmogorov_smirnov": self.ks_threshold,
            "population_stability_index": self.psi_threshold,
            "jensen_shannon_divergence": self.js_threshold,
            "chi_square": self.chi2_threshold
        }
        return threshold_map.get(test_name, self.drift_threshold)
    
    def _get_drift_severity(self, feature_name: str, drift_info: Dict[str, Any]) -> str:
        """Get severity level based on drift characteristics."""
        drift_score = drift_info["drift_score"]
        
        # Adjust severity based on feature importance
        weight = self.feature_weights.get(feature_name, 1.0)
        adjusted_score = drift_score * weight
        
        # Critical features get elevated severity
        if feature_name in self.critical_features:
            if adjusted_score > 0.1:
                return "critical"
            elif adjusted_score > 0.05:
                return "high"
        
        # Standard severity levels
        if adjusted_score > 0.5:
            return "critical"
        elif adjusted_score > 0.25:
            return "high"
        elif adjusted_score > 0.1:
            return "medium"
        else:
            return "low"
    
    def _get_drift_recommendation(self, feature_name: str, drift_info: Dict[str, Any]) -> str:
        """Get specific recommendation based on drift characteristics."""
        drift_score = drift_info["drift_score"]
        test_method = drift_info["best_test"]
        
        base_recommendations = {
            "kolmogorov_smirnov": "Distribution shape has changed - investigate data preprocessing",
            "population_stability_index": "Feature distribution shift detected - check data sources",
            "jensen_shannon_divergence": "Significant distribution divergence - validate data pipeline",
            "chi_square": "Category frequencies have shifted - review data collection"
        }
        
        base_rec = base_recommendations.get(test_method, "Data distribution has changed")
        
        # Add severity-based recommendations
        if drift_score > 0.5:
            return f"{base_rec}. Immediate retraining required - severe drift detected."
        elif drift_score > 0.25:
            return f"{base_rec}. Schedule retraining within 24-48 hours."
        elif drift_score > 0.1:
            return f"{base_rec}. Monitor closely and prepare for retraining."
        else:
            return f"{base_rec}. Continue monitoring."
    
    def _get_historical_trend(self, feature_name: str) -> Dict[str, Any]:
        """Get historical drift trend for a feature."""
        history = self.drift_history.get(feature_name, [])
        
        if len(history) < 2:
            return {"trend": "insufficient_data", "drift_count_7d": 0}
        
        # Count recent drift events
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_drifts = [h for h in history if h["timestamp"] > week_ago]
        
        # Determine trend
        if len(history) >= 3:
            recent_scores = [h["drift_score"] for h in history[-3:]]
            if len(recent_scores) >= 2:
                if recent_scores[-1] > recent_scores[0] * 1.5:
                    trend = "increasing"
                elif recent_scores[-1] < recent_scores[0] * 0.5:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "unknown"
        else:
            trend = "unknown"
        
        return {
            "trend": trend,
            "drift_count_7d": len(recent_drifts),
            "avg_drift_score_7d": np.mean([h["drift_score"] for h in recent_drifts]) if recent_drifts else 0.0,
            "last_drift_time": history[-1]["timestamp"].isoformat() if history else None
        }
    
    def _should_alert(self, feature_name: str) -> bool:
        """Check if enough time has passed since last alert for this feature."""
        last_alert = self.last_alerts.get(feature_name)
        if not last_alert:
            return True
        
        time_since_alert = datetime.utcnow() - last_alert
        return time_since_alert.total_seconds() > (self.alert_cooldown * 3600)
    
    def _track_drift_history(self, feature_name: str, drift_info: Dict[str, Any]) -> None:
        """Track drift detection in history for trend analysis."""
        self.drift_history[feature_name].append({
            "timestamp": datetime.utcnow(),
            "drift_score": drift_info["drift_score"],
            "test_method": drift_info["best_test"],
            "p_value": drift_info.get("p_value")
        })
        
        # Keep only recent history
        cutoff_time = datetime.utcnow() - timedelta(hours=self.window_hours)
        self.drift_history[feature_name] = [
            entry for entry in self.drift_history[feature_name]
            if entry["timestamp"] > cutoff_time
        ]


class DistributionAnalyzer:
    """Helper class for advanced distribution analysis and comparison."""
    
    @staticmethod
    def compare_distributions(baseline: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """Compare two distributions with comprehensive statistics."""
        comparison = {
            "basic_stats": {},
            "shape_analysis": {},
            "tail_analysis": {},
            "stability_metrics": {}
        }
        
        try:
            # Basic statistics
            comparison["basic_stats"] = {
                "baseline_mean": float(np.mean(baseline)),
                "current_mean": float(np.mean(current)),
                "baseline_std": float(np.std(baseline)),
                "current_std": float(np.std(current)),
                "mean_shift_pct": float((np.mean(current) - np.mean(baseline)) / np.mean(baseline) * 100) if np.mean(baseline) != 0 else 0.0,
                "std_ratio": float(np.std(current) / np.std(baseline)) if np.std(baseline) != 0 else 1.0
            }
            
            # Shape analysis
            baseline_skew = stats.skew(baseline)
            current_skew = stats.skew(current)
            baseline_kurt = stats.kurtosis(baseline)
            current_kurt = stats.kurtosis(current)
            
            comparison["shape_analysis"] = {
                "baseline_skewness": float(baseline_skew),
                "current_skewness": float(current_skew),
                "baseline_kurtosis": float(baseline_kurt),
                "current_kurtosis": float(current_kurt),
                "skewness_change": float(current_skew - baseline_skew),
                "kurtosis_change": float(current_kurt - baseline_kurt)
            }
            
            # Tail analysis (95th and 5th percentiles)
            baseline_q95, baseline_q05 = np.percentile(baseline, [95, 5])
            current_q95, current_q05 = np.percentile(current, [95, 5])
            
            comparison["tail_analysis"] = {
                "baseline_q95": float(baseline_q95),
                "current_q95": float(current_q95),
                "baseline_q05": float(baseline_q05),
                "current_q05": float(current_q05),
                "upper_tail_shift": float(current_q95 - baseline_q95),
                "lower_tail_shift": float(current_q05 - baseline_q05)
            }
            
            # Stability metrics
            comparison["stability_metrics"] = {
                "coefficient_of_variation_baseline": float(np.std(baseline) / np.mean(baseline)) if np.mean(baseline) != 0 else 0.0,
                "coefficient_of_variation_current": float(np.std(current) / np.mean(current)) if np.mean(current) != 0 else 0.0,
                "range_baseline": float(np.max(baseline) - np.min(baseline)),
                "range_current": float(np.max(current) - np.min(current)),
                "iqr_baseline": float(np.percentile(baseline, 75) - np.percentile(baseline, 25)),
                "iqr_current": float(np.percentile(current, 75) - np.percentile(current, 25))
            }
            
        except Exception as e:
            logger.warning(f"Distribution comparison failed: {e}")
        
        return comparison
    
    @staticmethod
    def detect_distribution_type(data: np.ndarray) -> str:
        """Detect the likely distribution type of the data."""
        try:
            # Test for common distributions
            distributions = [
                ('normal', stats.norm),
                ('exponential', stats.expon),
                ('uniform', stats.uniform),
                ('lognormal', stats.lognorm),
                ('gamma', stats.gamma)
            ]
            
            best_fit = None
            best_p_value = 0
            
            for dist_name, dist in distributions:
                try:
                    # Fit distribution parameters
                    params = dist.fit(data)
                    
                    # Perform KS test
                    ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
                    
                    if p_value > best_p_value:
                        best_p_value = p_value
                        best_fit = dist_name
                        
                except Exception:
                    continue
            
            return best_fit or "unknown"
            
        except Exception as e:
            logger.warning(f"Distribution type detection failed: {e}")
            return "unknown"
    
    @staticmethod
    def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for the mean."""
        try:
            mean = np.mean(data)
            sem = stats.sem(data)
            h = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
            return float(mean - h), float(mean + h)
        except Exception:
            return 0.0, 0.0