"""Model performance degradation detection."""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import statistics

from .base import BaseDetector
from ..core.context import Context

logger = logging.getLogger(__name__)


class ModelDegradationDetector(BaseDetector):
    """Detect model performance degradation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.performance_threshold = self.config.get("performance_threshold", 0.05)  # 5% degradation
        self.window_hours = self.config.get("window_hours", 24)
        self.min_predictions = self.config.get("min_predictions", 100)
    
    def get_supported_events(self) -> List[str]:
        return ["schedule", "workflow_run", "push"]
    
    async def detect(self, context: Context) -> List[Dict[str, Any]]:
        """Detect model performance degradation."""
        issues = []
        
        # Get model performance metrics (mock data for demonstration)
        current_metrics = await self._get_current_metrics(context)
        baseline_metrics = await self._get_baseline_metrics(context)
        
        if not current_metrics or not baseline_metrics:
            logger.warning("Unable to fetch metrics for degradation detection")
            return issues
        
        # Compare metrics
        degradation_results = await self._compare_metrics(current_metrics, baseline_metrics)
        
        for metric_name, degradation_info in degradation_results.items():
            if degradation_info["degradation_detected"]:
                issues.append(self.create_issue(
                    issue_type="model_degradation",
                    severity=self._get_degradation_severity(degradation_info["degradation_percentage"]),
                    message=f"Model degradation detected in {metric_name}: {degradation_info['degradation_percentage']:.1f}% decrease",
                    data={
                        "metric_name": metric_name,
                        "current_value": degradation_info["current_value"],
                        "baseline_value": degradation_info["baseline_value"],
                        "degradation_percentage": degradation_info["degradation_percentage"],
                        "threshold": self.performance_threshold * 100,
                        "recommendation": self._get_degradation_recommendation(degradation_info["degradation_percentage"])
                    }
                ))
        
        return issues
    
    async def _get_current_metrics(self, context: Context) -> Optional[Dict[str, float]]:
        """Get current model performance metrics."""
        # Mock current metrics with some degradation
        return {
            "accuracy": 0.87,      # Down from baseline
            "precision": 0.89,     # Stable
            "recall": 0.82,        # Down from baseline
            "f1_score": 0.85,      # Down from baseline
            "auc_roc": 0.91,       # Stable
            "latency_p95": 250.0,  # Increased from baseline
            "throughput": 950.0    # Down from baseline
        }
    
    async def _get_baseline_metrics(self, context: Context) -> Optional[Dict[str, float]]:
        """Get baseline model performance metrics."""
        # Mock baseline metrics
        return {
            "accuracy": 0.92,      # Baseline
            "precision": 0.90,     # Baseline
            "recall": 0.88,        # Baseline  
            "f1_score": 0.89,      # Baseline
            "auc_roc": 0.92,       # Baseline
            "latency_p95": 180.0,  # Baseline
            "throughput": 1200.0   # Baseline
        }
    
    async def _compare_metrics(self, current: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Compare current metrics with baseline."""
        comparison_results = {}
        
        for metric_name in current.keys():
            if metric_name not in baseline:
                continue
            
            current_value = current[metric_name]
            baseline_value = baseline[metric_name]
            
            # Calculate degradation percentage
            # For latency, higher is worse; for others, lower is worse
            if metric_name.startswith("latency"):
                degradation_pct = ((current_value - baseline_value) / baseline_value) * 100
            else:
                degradation_pct = ((baseline_value - current_value) / baseline_value) * 100
            
            # Check if degradation exceeds threshold
            degradation_detected = degradation_pct > (self.performance_threshold * 100)
            
            comparison_results[metric_name] = {
                "current_value": current_value,
                "baseline_value": baseline_value,
                "degradation_percentage": degradation_pct,
                "degradation_detected": degradation_detected
            }
        
        return comparison_results
    
    def _get_degradation_severity(self, degradation_percentage: float) -> str:
        """Get severity level based on degradation percentage."""
        if degradation_percentage > 20:
            return "critical"
        elif degradation_percentage > 10:
            return "high"
        elif degradation_percentage > 5:
            return "medium"
        else:
            return "low"
    
    def _get_degradation_recommendation(self, degradation_percentage: float) -> str:
        """Get recommendation based on degradation level."""
        if degradation_percentage > 20:
            return "Immediate rollback and investigation required"
        elif degradation_percentage > 10:
            return "Consider rollback and schedule retraining"
        elif degradation_percentage > 5:
            return "Monitor closely and prepare for retraining"
        else:
            return "Continue monitoring current performance"


class PerformanceAnomalyDetector:
    """Helper class for detecting performance anomalies using statistical methods."""
    
    @staticmethod
    def detect_anomalies(values: List[float], threshold_std: float = 2.0) -> List[int]:
        """Detect anomalies using standard deviation method."""
        if len(values) < 3:
            return []
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
            if z_score > threshold_std:
                anomalies.append(i)
        
        return anomalies
    
    @staticmethod
    def calculate_trend(values: List[float]) -> str:
        """Calculate trend direction of performance metrics."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"
    
    @staticmethod
    def detect_seasonal_patterns(values: List[float], period: int = 24) -> Dict[str, Any]:
        """Detect seasonal patterns in performance metrics."""
        if len(values) < period * 2:
            return {"has_pattern": False, "confidence": 0.0}
        
        # Simple autocorrelation at seasonal lag
        n = len(values)
        mean_val = statistics.mean(values)
        
        # Calculate autocorrelation at seasonal lag
        numerator = sum((values[i] - mean_val) * (values[i - period] - mean_val) 
                       for i in range(period, n))
        denominator = sum((values[i] - mean_val) ** 2 for i in range(n))
        
        if denominator == 0:
            return {"has_pattern": False, "confidence": 0.0}
        
        autocorr = numerator / denominator
        
        return {
            "has_pattern": abs(autocorr) > 0.5,
            "confidence": abs(autocorr),
            "pattern_strength": "strong" if abs(autocorr) > 0.7 else "moderate"
        }