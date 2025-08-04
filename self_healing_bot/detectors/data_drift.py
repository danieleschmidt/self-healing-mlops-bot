"""Data drift detection and monitoring."""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

from .base import BaseDetector
from ..core.context import Context

logger = logging.getLogger(__name__)


class DataDriftDetector(BaseDetector):
    """Detect data drift in ML pipelines."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.drift_threshold = self.config.get("drift_threshold", 0.1)
        self.statistical_tests = self.config.get("statistical_tests", ["ks", "psi"])
    
    def get_supported_events(self) -> List[str]:
        return ["push", "schedule", "workflow_run"]
    
    async def detect(self, context: Context) -> List[Dict[str, Any]]:
        """Detect data drift using statistical methods."""
        issues = []
        
        # Mock data for demonstration - in real implementation, this would
        # fetch actual training and production data
        training_data = self._get_mock_training_data()
        production_data = self._get_mock_production_data()
        
        if training_data is None or production_data is None:
            logger.warning("Unable to fetch data for drift detection")
            return issues
        
        # Perform drift detection
        drift_results = await self._detect_drift(training_data, production_data)
        
        for feature_name, drift_info in drift_results.items():
            if drift_info["drift_detected"]:
                issues.append(self.create_issue(
                    issue_type="data_drift",
                    severity=self._get_drift_severity(drift_info["drift_score"]),
                    message=f"Data drift detected in feature '{feature_name}' (score: {drift_info['drift_score']:.3f})",
                    data={
                        "feature_name": feature_name,
                        "drift_score": drift_info["drift_score"],
                        "test_method": drift_info["test_method"],
                        "p_value": drift_info.get("p_value"),
                        "threshold": self.drift_threshold,
                        "recommendation": self._get_drift_recommendation(drift_info["drift_score"])
                    }
                ))
        
        return issues
    
    async def _detect_drift(self, training_data: Dict[str, np.ndarray], production_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Perform drift detection on all features."""
        drift_results = {}
        
        common_features = set(training_data.keys()) & set(production_data.keys())
        
        for feature_name in common_features:
            train_values = training_data[feature_name]
            prod_values = production_data[feature_name]
            
            # Skip if insufficient data
            if len(train_values) < 10 or len(prod_values) < 10:
                continue
            
            # Perform statistical tests
            drift_info = await self._run_statistical_tests(train_values, prod_values, feature_name)
            drift_results[feature_name] = drift_info
        
        return drift_results
    
    async def _run_statistical_tests(self, train_data: np.ndarray, prod_data: np.ndarray, feature_name: str) -> Dict[str, Any]:
        """Run statistical tests for drift detection."""
        results = {
            "drift_detected": False,
            "drift_score": 0.0,
            "test_method": "none",
            "p_value": None
        }
        
        try:
            # Kolmogorov-Smirnov test
            if "ks" in self.statistical_tests:
                ks_stat, p_value = stats.ks_2samp(train_data, prod_data)
                
                if ks_stat > self.drift_threshold:
                    results.update({
                        "drift_detected": True,
                        "drift_score": ks_stat,
                        "test_method": "kolmogorov_smirnov",
                        "p_value": p_value
                    })
                    return results
            
            # Population Stability Index (PSI)
            if "psi" in self.statistical_tests:
                psi_score = self._calculate_psi(train_data, prod_data)
                
                if psi_score > self.drift_threshold:
                    results.update({
                        "drift_detected": True,
                        "drift_score": psi_score,
                        "test_method": "population_stability_index"
                    })
                    return results
            
            # Jensen-Shannon divergence
            if "js" in self.statistical_tests:
                js_divergence = self._calculate_js_divergence(train_data, prod_data)
                
                if js_divergence > self.drift_threshold:
                    results.update({
                        "drift_detected": True,
                        "drift_score": js_divergence,
                        "test_method": "jensen_shannon_divergence"
                    })
        
        except Exception as e:
            logger.exception(f"Error running statistical tests for feature {feature_name}: {e}")
        
        return results
    
    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index."""
        try:
            # Create buckets based on expected distribution
            breakpoints = np.histogram_bin_edges(expected, bins=buckets)
            
            # Calculate frequencies
            expected_freq = np.histogram(expected, bins=breakpoints)[0]
            actual_freq = np.histogram(actual, bins=breakpoints)[0]
            
            # Normalize to proportions
            expected_prop = expected_freq / len(expected)
            actual_prop = actual_freq / len(actual)
            
            # Avoid division by zero
            expected_prop = np.where(expected_prop == 0, 0.0001, expected_prop)
            actual_prop = np.where(actual_prop == 0, 0.0001, actual_prop)
            
            # Calculate PSI
            psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
            
            return psi
        
        except Exception:
            return 0.0
    
    def _calculate_js_divergence(self, p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
        """Calculate Jensen-Shannon divergence."""
        try:
            # Create histograms
            p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
            q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
            
            # Normalize
            p_hist = p_hist / np.sum(p_hist)
            q_hist = q_hist / np.sum(q_hist)
            
            # Avoid zeros
            p_hist = np.where(p_hist == 0, 1e-10, p_hist)
            q_hist = np.where(q_hist == 0, 1e-10, q_hist)
            
            # Calculate JS divergence
            m = 0.5 * (p_hist + q_hist)
            js_div = 0.5 * stats.entropy(p_hist, m) + 0.5 * stats.entropy(q_hist, m)
            
            return js_div
        
        except Exception:
            return 0.0
    
    def _get_drift_severity(self, drift_score: float) -> str:
        """Get severity level based on drift score."""
        if drift_score > 0.5:
            return "critical"
        elif drift_score > 0.25:
            return "high" 
        elif drift_score > 0.1:
            return "medium"
        else:
            return "low"
    
    def _get_drift_recommendation(self, drift_score: float) -> str:
        """Get recommendation based on drift score."""
        if drift_score > 0.5:
            return "Immediate retraining required - significant distribution shift detected"
        elif drift_score > 0.25:
            return "Schedule retraining within 24 hours - notable drift detected"
        elif drift_score > 0.1:
            return "Monitor closely and consider retraining - mild drift detected"
        else:
            return "Continue monitoring - minimal drift detected"
    
    def _get_mock_training_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Generate mock training data for demonstration."""
        np.random.seed(42)
        return {
            "feature_1": np.random.normal(0, 1, 1000),
            "feature_2": np.random.exponential(2, 1000),
            "feature_3": np.random.uniform(0, 10, 1000)
        }
    
    def _get_mock_production_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Generate mock production data with some drift."""
        np.random.seed(123)
        return {
            "feature_1": np.random.normal(0.2, 1.1, 500),  # Slight drift
            "feature_2": np.random.exponential(2, 500),     # No drift
            "feature_3": np.random.uniform(1, 11, 500)      # Significant drift
        }