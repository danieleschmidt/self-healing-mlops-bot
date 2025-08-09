"""Model performance degradation detection with comprehensive monitoring."""

from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import statistics
import numpy as np
from collections import defaultdict, deque
import json

from .base import BaseDetector
from ..core.context import Context

logger = logging.getLogger(__name__)


class ModelDegradationDetector(BaseDetector):
    """Detect model performance degradation with comprehensive monitoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Performance thresholds
        self.performance_threshold = self.config.get("performance_threshold", 0.05)  # 5% degradation
        self.latency_threshold = self.config.get("latency_threshold", 0.15)  # 15% increase
        self.error_rate_threshold = self.config.get("error_rate_threshold", 0.02)  # 2% error rate
        self.resource_threshold = self.config.get("resource_threshold", 0.8)  # 80% utilization
        
        # Time windows and sampling
        self.window_hours = self.config.get("window_hours", 24)
        self.min_predictions = self.config.get("min_predictions", 100)
        self.comparison_window_days = self.config.get("comparison_window_days", 7)
        
        # Performance monitoring
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_metrics_cache = {}
        self.anomaly_threshold = self.config.get("anomaly_threshold", 2.0)  # z-score threshold
        
        # A/B testing configuration
        self.ab_test_enabled = self.config.get("ab_test_enabled", False)
        self.statistical_significance_threshold = self.config.get("significance_threshold", 0.05)
        
        # Resource monitoring
        self.resource_monitoring_enabled = self.config.get("resource_monitoring_enabled", True)
        self.memory_threshold = self.config.get("memory_threshold", 0.85)  # 85%
        self.cpu_threshold = self.config.get("cpu_threshold", 0.80)  # 80%
        
        # Alert configuration
        self.alert_cooldown_hours = self.config.get("alert_cooldown_hours", 4)
        self.last_alerts = {}
    
    def get_supported_events(self) -> List[str]:
        return ["schedule", "workflow_run", "push", "model_prediction", "model_deployment"]
    
    async def detect(self, context: Context) -> List[Dict[str, Any]]:
        """Detect model performance degradation with comprehensive analysis."""
        issues = []
        
        try:
            # Get current and baseline metrics
            current_metrics = await self._get_current_metrics(context)
            baseline_metrics = await self._get_baseline_metrics(context)
            
            if not current_metrics or not baseline_metrics:
                logger.warning("Unable to fetch metrics for degradation detection")
                return issues
            
            # Core performance degradation detection
            degradation_results = await self._compare_metrics(current_metrics, baseline_metrics)
            issues.extend(await self._generate_degradation_issues(degradation_results))
            
            # Latency degradation detection
            latency_issues = await self._detect_latency_degradation(current_metrics, baseline_metrics)
            issues.extend(latency_issues)
            
            # Error rate monitoring
            error_rate_issues = await self._detect_error_rate_issues(current_metrics)
            issues.extend(error_rate_issues)
            
            # Resource utilization monitoring
            if self.resource_monitoring_enabled:
                resource_issues = await self._detect_resource_issues(current_metrics)
                issues.extend(resource_issues)
            
            # A/B testing performance comparison
            if self.ab_test_enabled:
                ab_test_issues = await self._detect_ab_test_issues(context)
                issues.extend(ab_test_issues)
            
            # Performance anomaly detection
            anomaly_issues = await self._detect_performance_anomalies(context, current_metrics)
            issues.extend(anomaly_issues)
            
            # Trend analysis
            trend_issues = await self._analyze_performance_trends(context)
            issues.extend(trend_issues)
            
        except Exception as e:
            logger.exception(f"Error in model degradation detection: {e}")
            issues.append(self.create_issue(
                issue_type="degradation_detection_error",
                severity="medium",
                message=f"Model degradation detection failed: {str(e)}",
                data={"error_details": str(e)}
            ))
        
        return issues
    
    async def _get_current_metrics(self, context: Context) -> Optional[Dict[str, Any]]:
        """Get current model performance metrics with comprehensive data."""
        # Mock current metrics - in production, this would fetch from monitoring systems
        import random
        import time
        
        current_hour = datetime.utcnow().hour
        degradation_factor = 1.0 - (current_hour / 100.0)  # Simulate daily degradation
        noise_factor = random.uniform(0.95, 1.05)  # Add random variation
        
        return {
            # Accuracy metrics
            "accuracy": max(0.7, 0.92 * degradation_factor * noise_factor),
            "precision": max(0.7, 0.90 * degradation_factor * noise_factor),
            "recall": max(0.7, 0.88 * degradation_factor * noise_factor),
            "f1_score": max(0.7, 0.89 * degradation_factor * noise_factor),
            "auc_roc": max(0.7, 0.92 * degradation_factor * noise_factor),
            "auc_pr": max(0.7, 0.88 * degradation_factor * noise_factor),
            
            # Latency metrics (milliseconds)
            "latency_p50": 180.0 / degradation_factor * noise_factor,
            "latency_p95": 250.0 / degradation_factor * noise_factor,
            "latency_p99": 400.0 / degradation_factor * noise_factor,
            "latency_mean": 200.0 / degradation_factor * noise_factor,
            
            # Throughput and capacity
            "throughput_rps": 1000.0 * degradation_factor * noise_factor,
            "concurrent_requests": random.randint(50, 200),
            "queue_size": random.randint(0, 50),
            
            # Error rates
            "error_rate_total": max(0.001, (1.0 - degradation_factor) * 0.05),
            "error_rate_4xx": max(0.001, (1.0 - degradation_factor) * 0.02),
            "error_rate_5xx": max(0.001, (1.0 - degradation_factor) * 0.03),
            "timeout_rate": max(0.001, (1.0 - degradation_factor) * 0.01),
            
            # Resource utilization
            "cpu_utilization": min(0.95, 0.60 / degradation_factor),
            "memory_utilization": min(0.95, 0.70 / degradation_factor),
            "gpu_utilization": min(0.95, 0.80 / degradation_factor) if random.choice([True, False]) else None,
            "gpu_memory_utilization": min(0.95, 0.75 / degradation_factor) if random.choice([True, False]) else None,
            
            # Data quality metrics
            "prediction_confidence_mean": max(0.5, 0.85 * degradation_factor * noise_factor),
            "prediction_confidence_std": 0.15 / degradation_factor,
            "feature_missing_rate": (1.0 - degradation_factor) * 0.1,
            
            # Model-specific metrics
            "model_version": "v1.2.3",
            "prediction_count": random.randint(800, 1200),
            "cache_hit_rate": max(0.5, 0.85 * degradation_factor * noise_factor),
            
            # Business metrics
            "conversion_rate": max(0.01, 0.05 * degradation_factor * noise_factor),
            "revenue_per_prediction": max(0.1, 1.25 * degradation_factor * noise_factor),
            
            # Infrastructure metrics  
            "container_restarts": random.randint(0, 3),
            "disk_usage": min(0.9, 0.4 + (1.0 - degradation_factor) * 0.3),
            "network_latency": 20.0 / degradation_factor * noise_factor,
            
            # Timestamp for tracking
            "collected_at": datetime.utcnow().isoformat()
        }
    
    async def _get_baseline_metrics(self, context: Context) -> Optional[Dict[str, Any]]:
        """Get baseline model performance metrics from historical data."""
        cache_key = f"baseline_metrics_{context.repo_full_name}"
        
        if cache_key in self.baseline_metrics_cache:
            cached_data, timestamp = self.baseline_metrics_cache[cache_key]
            if datetime.utcnow() - timestamp < timedelta(hours=6):
                return cached_data
        
        # Mock baseline metrics representing optimal performance
        baseline_metrics = {
            # Accuracy metrics
            "accuracy": 0.92,
            "precision": 0.90, 
            "recall": 0.88,
            "f1_score": 0.89,
            "auc_roc": 0.92,
            "auc_pr": 0.88,
            
            # Latency metrics (milliseconds)
            "latency_p50": 180.0,
            "latency_p95": 250.0,
            "latency_p99": 380.0,
            "latency_mean": 200.0,
            
            # Throughput and capacity
            "throughput_rps": 1200.0,
            "concurrent_requests": 100,
            "queue_size": 10,
            
            # Error rates
            "error_rate_total": 0.005,
            "error_rate_4xx": 0.002,
            "error_rate_5xx": 0.003,
            "timeout_rate": 0.001,
            
            # Resource utilization
            "cpu_utilization": 0.60,
            "memory_utilization": 0.70,
            "gpu_utilization": 0.75,
            "gpu_memory_utilization": 0.70,
            
            # Data quality metrics
            "prediction_confidence_mean": 0.85,
            "prediction_confidence_std": 0.12,
            "feature_missing_rate": 0.02,
            
            # Model-specific metrics
            "prediction_count": 1000,
            "cache_hit_rate": 0.90,
            
            # Business metrics
            "conversion_rate": 0.055,
            "revenue_per_prediction": 1.35,
            
            # Infrastructure metrics
            "container_restarts": 0,
            "disk_usage": 0.40,
            "network_latency": 15.0
        }
        
        # Cache the baseline
        self.baseline_metrics_cache[cache_key] = (baseline_metrics, datetime.utcnow())
        return baseline_metrics
    
    async def _compare_metrics(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Compare current metrics with baseline using comprehensive analysis."""
        comparison_results = {}
        
        # Define metric categories and their improvement directions
        metric_categories = {
            "accuracy_metrics": ["accuracy", "precision", "recall", "f1_score", "auc_roc", "auc_pr"],
            "latency_metrics": ["latency_p50", "latency_p95", "latency_p99", "latency_mean"],
            "throughput_metrics": ["throughput_rps", "cache_hit_rate"],
            "error_metrics": ["error_rate_total", "error_rate_4xx", "error_rate_5xx", "timeout_rate"],
            "resource_metrics": ["cpu_utilization", "memory_utilization", "gpu_utilization", "gpu_memory_utilization"],
            "quality_metrics": ["prediction_confidence_mean", "feature_missing_rate"],
            "business_metrics": ["conversion_rate", "revenue_per_prediction"]
        }
        
        # Metrics where higher values are worse
        inverse_metrics = ["latency_p50", "latency_p95", "latency_p99", "latency_mean", 
                          "error_rate_total", "error_rate_4xx", "error_rate_5xx", "timeout_rate",
                          "cpu_utilization", "memory_utilization", "gpu_utilization", 
                          "gpu_memory_utilization", "feature_missing_rate"]
        
        for category, metrics in metric_categories.items():
            for metric_name in metrics:
                if metric_name not in current or metric_name not in baseline:
                    continue
                
                current_value = current.get(metric_name)
                baseline_value = baseline.get(metric_name)
                
                if current_value is None or baseline_value is None or baseline_value == 0:
                    continue
                
                # Calculate percentage change
                if metric_name in inverse_metrics:
                    # For inverse metrics, degradation is positive change
                    degradation_pct = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    # For normal metrics, degradation is negative change
                    degradation_pct = ((baseline_value - current_value) / baseline_value) * 100
                
                # Determine if degradation is significant
                threshold = self._get_metric_threshold(metric_name, category)
                degradation_detected = degradation_pct > threshold
                
                # Calculate statistical significance if we have historical data
                confidence_level = self._calculate_statistical_confidence(metric_name, current_value, baseline_value)
                
                comparison_results[metric_name] = {
                    "current_value": current_value,
                    "baseline_value": baseline_value,
                    "degradation_percentage": degradation_pct,
                    "degradation_detected": degradation_detected,
                    "category": category,
                    "threshold": threshold,
                    "confidence_level": confidence_level,
                    "is_inverse_metric": metric_name in inverse_metrics
                }
        
        return comparison_results
    
    def _get_metric_threshold(self, metric_name: str, category: str) -> float:
        """Get appropriate threshold for different metric types."""
        thresholds = {
            "accuracy_metrics": self.performance_threshold * 100,  # 5%
            "latency_metrics": self.latency_threshold * 100,       # 15%
            "throughput_metrics": self.performance_threshold * 100,  # 5%
            "error_metrics": self.error_rate_threshold * 100,     # 2%
            "resource_metrics": 10.0,  # 10% increase
            "quality_metrics": self.performance_threshold * 100,   # 5%
            "business_metrics": self.performance_threshold * 100   # 5%
        }
        return thresholds.get(category, self.performance_threshold * 100)
    
    def _calculate_statistical_confidence(self, metric_name: str, current_value: float, baseline_value: float) -> float:
        """Calculate statistical confidence of the degradation."""
        # Mock implementation - in production, would use proper statistical tests
        # based on historical variance and sample size
        history = self.performance_history.get(metric_name, [])
        
        if len(history) < 10:
            return 0.5  # Low confidence with insufficient data
        
        # Simple confidence calculation based on historical variance
        historical_values = [entry["value"] for entry in list(history)[-20:]]  # Last 20 values
        std_dev = statistics.stdev(historical_values) if len(historical_values) > 1 else 0.1
        
        # Calculate z-score
        if std_dev > 0:
            z_score = abs(current_value - baseline_value) / std_dev
            # Convert z-score to confidence level (approximation)
            confidence = min(0.99, max(0.5, 1 - (1 / (1 + z_score))))
        else:
            confidence = 0.5
        
        return confidence
    
    async def _generate_degradation_issues(self, degradation_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate issues from degradation analysis results."""
        issues = []
        
        for metric_name, degradation_info in degradation_results.items():
            if not degradation_info["degradation_detected"]:
                continue
            
            # Check alert cooldown
            if not self._should_alert(metric_name):
                continue
            
            severity = self._get_degradation_severity(
                degradation_info["degradation_percentage"], 
                degradation_info["category"]
            )
            
            issues.append(self.create_issue(
                issue_type=f"model_degradation_{degradation_info['category']}",
                severity=severity,
                message=f"Model degradation in {metric_name}: {degradation_info['degradation_percentage']:.1f}% {'increase' if degradation_info['is_inverse_metric'] else 'decrease'}",
                data={
                    "metric_name": metric_name,
                    "current_value": degradation_info["current_value"],
                    "baseline_value": degradation_info["baseline_value"],
                    "degradation_percentage": degradation_info["degradation_percentage"],
                    "category": degradation_info["category"],
                    "threshold": degradation_info["threshold"],
                    "confidence_level": degradation_info["confidence_level"],
                    "recommendation": self._get_degradation_recommendation(
                        metric_name, degradation_info["degradation_percentage"], degradation_info["category"]
                    )
                }
            ))
            
            # Update alert history
            self.last_alerts[metric_name] = datetime.utcnow()
        
        return issues
    
    async def _detect_latency_degradation(self, current_metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Specialized latency degradation detection with percentile analysis."""
        issues = []
        
        latency_metrics = ["latency_p50", "latency_p95", "latency_p99", "latency_mean"]
        
        degraded_percentiles = []
        for metric in latency_metrics:
            if metric not in current_metrics or metric not in baseline_metrics:
                continue
            
            current_val = current_metrics[metric]
            baseline_val = baseline_metrics[metric]
            
            if baseline_val > 0:
                increase_pct = ((current_val - baseline_val) / baseline_val) * 100
                if increase_pct > self.latency_threshold * 100:
                    degraded_percentiles.append({
                        "metric": metric,
                        "increase_percentage": increase_pct,
                        "current_value": current_val,
                        "baseline_value": baseline_val
                    })
        
        if degraded_percentiles:
            severity = "critical" if any(p["increase_percentage"] > 50 for p in degraded_percentiles) else "high"
            
            issues.append(self.create_issue(
                issue_type="latency_degradation",
                severity=severity,
                message=f"Latency degradation detected across {len(degraded_percentiles)} percentiles",
                data={
                    "degraded_percentiles": degraded_percentiles,
                    "recommendation": self._get_latency_recommendation(degraded_percentiles)
                }
            ))
        
        return issues
    
    async def _detect_error_rate_issues(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect error rate spikes and patterns."""
        issues = []
        
        error_metrics = {
            "error_rate_total": "total error rate",
            "error_rate_4xx": "client error rate", 
            "error_rate_5xx": "server error rate",
            "timeout_rate": "timeout rate"
        }
        
        for metric_name, description in error_metrics.items():
            if metric_name not in current_metrics:
                continue
            
            error_rate = current_metrics[metric_name]
            
            if error_rate > self.error_rate_threshold:
                severity = "critical" if error_rate > 0.1 else "high" if error_rate > 0.05 else "medium"
                
                issues.append(self.create_issue(
                    issue_type="error_rate_spike",
                    severity=severity,
                    message=f"High {description}: {error_rate:.3%}",
                    data={
                        "metric_name": metric_name,
                        "error_rate": error_rate,
                        "threshold": self.error_rate_threshold,
                        "recommendation": self._get_error_rate_recommendation(metric_name, error_rate)
                    }
                ))
        
        return issues
    
    async def _detect_resource_issues(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect resource utilization issues."""
        issues = []
        
        resource_metrics = {
            "cpu_utilization": ("CPU", self.cpu_threshold),
            "memory_utilization": ("Memory", self.memory_threshold),
            "gpu_utilization": ("GPU", self.resource_threshold),
            "gpu_memory_utilization": ("GPU Memory", self.resource_threshold)
        }
        
        for metric_name, (resource_name, threshold) in resource_metrics.items():
            if metric_name not in current_metrics or current_metrics[metric_name] is None:
                continue
            
            utilization = current_metrics[metric_name]
            
            if utilization > threshold:
                severity = "critical" if utilization > 0.95 else "high" if utilization > 0.90 else "medium"
                
                issues.append(self.create_issue(
                    issue_type="resource_utilization_high",
                    severity=severity,
                    message=f"High {resource_name} utilization: {utilization:.1%}",
                    data={
                        "resource_type": resource_name,
                        "utilization": utilization,
                        "threshold": threshold,
                        "recommendation": self._get_resource_recommendation(resource_name, utilization)
                    }
                ))
        
        return issues
    
    async def _detect_ab_test_issues(self, context: Context) -> List[Dict[str, Any]]:
        """Detect issues in A/B testing performance comparison."""
        issues = []
        
        # Mock A/B test data - in production, would fetch from experiment platform
        control_metrics = {"conversion_rate": 0.045, "accuracy": 0.89, "latency_p95": 220.0}
        treatment_metrics = {"conversion_rate": 0.038, "accuracy": 0.87, "latency_p95": 280.0}
        
        # Statistical significance test (mock implementation)
        significant_differences = []
        
        for metric in control_metrics:
            if metric in treatment_metrics:
                control_val = control_metrics[metric]
                treatment_val = treatment_metrics[metric]
                
                # Simple significance test (mock)
                percent_change = ((treatment_val - control_val) / control_val) * 100
                
                # Mock p-value calculation
                p_value = 0.02 if abs(percent_change) > 5 else 0.15
                
                if p_value < self.statistical_significance_threshold and abs(percent_change) > 2:
                    significant_differences.append({
                        "metric": metric,
                        "control_value": control_val,
                        "treatment_value": treatment_val,
                        "percent_change": percent_change,
                        "p_value": p_value,
                        "is_degradation": percent_change < 0 and metric != "latency_p95" or 
                                        percent_change > 0 and metric == "latency_p95"
                    })
        
        if significant_differences:
            degradations = [d for d in significant_differences if d["is_degradation"]]
            if degradations:
                issues.append(self.create_issue(
                    issue_type="ab_test_performance_degradation",
                    severity="high",
                    message=f"A/B test shows significant performance degradation in {len(degradations)} metrics",
                    data={
                        "significant_differences": significant_differences,
                        "degradations": degradations,
                        "recommendation": "Consider halting traffic to treatment variant and investigate performance issues"
                    }
                ))
        
        return issues
    
    async def _detect_performance_anomalies(self, context: Context, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical methods."""
        issues = []
        
        # Track metrics in history for anomaly detection
        timestamp = datetime.utcnow()
        for metric_name, value in current_metrics.items():
            if isinstance(value, (int, float)):
                self.performance_history[metric_name].append({
                    "value": value,
                    "timestamp": timestamp
                })
        
        # Detect anomalies in key metrics
        key_metrics = ["accuracy", "latency_p95", "throughput_rps", "error_rate_total"]
        
        for metric_name in key_metrics:
            if metric_name not in current_metrics:
                continue
            
            history = list(self.performance_history[metric_name])
            if len(history) < 20:  # Need sufficient history
                continue
            
            recent_values = [entry["value"] for entry in history[-20:]]
            current_value = current_metrics[metric_name]
            
            # Statistical anomaly detection
            anomaly_info = PerformanceAnomalyDetector.detect_statistical_anomaly(
                recent_values, current_value, self.anomaly_threshold
            )
            
            if anomaly_info["is_anomaly"]:
                issues.append(self.create_issue(
                    issue_type="performance_anomaly",
                    severity=anomaly_info["severity"],
                    message=f"Performance anomaly detected in {metric_name}: {anomaly_info['description']}",
                    data={
                        "metric_name": metric_name,
                        "current_value": current_value,
                        "z_score": anomaly_info["z_score"],
                        "historical_mean": anomaly_info["historical_mean"],
                        "historical_std": anomaly_info["historical_std"],
                        "recommendation": f"Investigate sudden change in {metric_name} - possible system issue or data quality problem"
                    }
                ))
        
        return issues
    
    async def _analyze_performance_trends(self, context: Context) -> List[Dict[str, Any]]:
        """Analyze long-term performance trends."""
        issues = []
        
        # Analyze trends for key metrics
        key_metrics = ["accuracy", "latency_p95", "throughput_rps"]
        
        for metric_name in key_metrics:
            history = list(self.performance_history[metric_name])
            if len(history) < 50:  # Need sufficient history for trend analysis
                continue
            
            values = [entry["value"] for entry in history[-50:]]  # Last 50 data points
            
            trend_info = PerformanceAnomalyDetector.analyze_trend(values)
            
            if trend_info["trend"] == "degrading" and trend_info["confidence"] > 0.7:
                severity = "high" if trend_info["slope"] < -0.02 else "medium"
                
                issues.append(self.create_issue(
                    issue_type="performance_trend_degradation",
                    severity=severity,
                    message=f"Degrading trend detected in {metric_name} over time",
                    data={
                        "metric_name": metric_name,
                        "trend": trend_info["trend"],
                        "slope": trend_info["slope"],
                        "confidence": trend_info["confidence"],
                        "r_squared": trend_info.get("r_squared", 0.0),
                        "recommendation": f"Monitor {metric_name} closely - gradual degradation detected"
                    }
                ))
        
        return issues
    
    def _should_alert(self, metric_name: str) -> bool:
        """Check if enough time has passed since last alert for this metric."""
        last_alert = self.last_alerts.get(metric_name)
        if not last_alert:
            return True
        
        time_since_alert = datetime.utcnow() - last_alert
        return time_since_alert.total_seconds() > (self.alert_cooldown_hours * 3600)
    
    def _get_degradation_severity(self, degradation_percentage: float, category: str) -> str:
        """Get severity level based on degradation percentage and metric category."""
        # Adjust severity thresholds based on metric category
        if category in ["accuracy_metrics", "business_metrics"]:
            # More sensitive to accuracy and business metric degradation
            if degradation_percentage > 15:
                return "critical"
            elif degradation_percentage > 8:
                return "high"
            elif degradation_percentage > 3:
                return "medium"
        elif category == "latency_metrics":
            # Different thresholds for latency
            if degradation_percentage > 50:
                return "critical"
            elif degradation_percentage > 25:
                return "high"
            elif degradation_percentage > 10:
                return "medium"
        elif category == "error_metrics":
            # Very sensitive to error rate increases
            if degradation_percentage > 100:  # 2x increase
                return "critical"
            elif degradation_percentage > 50:
                return "high"
            elif degradation_percentage > 20:
                return "medium"
        else:
            # Default thresholds
            if degradation_percentage > 20:
                return "critical"
            elif degradation_percentage > 10:
                return "high"
            elif degradation_percentage > 5:
                return "medium"
        
        return "low"
    
    def _get_degradation_recommendation(self, metric_name: str, degradation_percentage: float, category: str) -> str:
        """Get specific recommendation based on metric type and degradation level."""
        recommendations = {
            "accuracy": "Check for data drift, model staleness, or feature pipeline issues",
            "precision": "Review prediction thresholds and class imbalance",
            "recall": "Investigate false negative increases - check data quality",
            "f1_score": "Balance precision and recall optimization",
            "latency_p95": "Scale infrastructure, optimize model inference, or check resource bottlenecks",
            "latency_p99": "Investigate tail latency spikes - check for resource contention",
            "throughput_rps": "Scale horizontally or optimize processing pipeline",
            "error_rate_total": "Investigate service health and error patterns",
            "cpu_utilization": "Scale CPU resources or optimize compute-intensive operations",
            "memory_utilization": "Increase memory allocation or optimize memory usage",
            "conversion_rate": "Investigate user experience or model relevance issues"
        }
        
        base_rec = recommendations.get(metric_name, "Investigate root cause and consider scaling or optimization")
        
        if degradation_percentage > 20:
            return f"URGENT: {base_rec}. Consider immediate rollback if possible."
        elif degradation_percentage > 10:
            return f"HIGH PRIORITY: {base_rec}. Schedule immediate investigation."
        else:
            return f"{base_rec}. Monitor closely for continued degradation."
    
    def _get_latency_recommendation(self, degraded_percentiles: List[Dict[str, Any]]) -> str:
        """Get recommendation for latency degradation."""
        max_increase = max(p["increase_percentage"] for p in degraded_percentiles)
        
        if max_increase > 50:
            return "Critical latency degradation - investigate infrastructure issues, consider immediate scaling"
        elif max_increase > 25:
            return "Significant latency increase - check resource utilization and scale if needed"
        else:
            return "Monitor latency trends and consider performance optimization"
    
    def _get_error_rate_recommendation(self, metric_name: str, error_rate: float) -> str:
        """Get recommendation for error rate issues."""
        recommendations = {
            "error_rate_total": "Investigate overall service health and error patterns",
            "error_rate_4xx": "Check input validation and client request patterns", 
            "error_rate_5xx": "Investigate server-side issues and infrastructure health",
            "timeout_rate": "Check resource availability and response time optimization"
        }
        
        base_rec = recommendations.get(metric_name, "Investigate error patterns")
        
        if error_rate > 0.1:
            return f"CRITICAL: {base_rec}. Consider traffic throttling or circuit breaker activation."
        elif error_rate > 0.05:
            return f"HIGH: {base_rec}. Immediate investigation required."
        else:
            return f"{base_rec}. Monitor error trends closely."
    
    def _get_resource_recommendation(self, resource_type: str, utilization: float) -> str:
        """Get recommendation for resource utilization issues."""
        if utilization > 0.95:
            return f"CRITICAL: {resource_type} utilization extremely high. Immediate scaling required to prevent service degradation."
        elif utilization > 0.90:
            return f"HIGH: {resource_type} utilization very high. Scale resources proactively."
        else:
            return f"Monitor {resource_type} usage and plan capacity scaling."


class PerformanceAnomalyDetector:
    """Enhanced performance anomaly detector with statistical methods."""
    
    @staticmethod
    def detect_statistical_anomaly(historical_values: List[float], current_value: float, 
                                 threshold_std: float = 2.0) -> Dict[str, Any]:
        """Detect statistical anomalies using z-score method."""
        if len(historical_values) < 3:
            return {"is_anomaly": False, "z_score": 0.0}
        
        mean_val = statistics.mean(historical_values)
        std_val = statistics.stdev(historical_values)
        
        if std_val == 0:
            return {"is_anomaly": False, "z_score": 0.0}
        
        z_score = abs(current_value - mean_val) / std_val
        is_anomaly = z_score > threshold_std
        
        # Determine severity
        if z_score > 4.0:
            severity = "critical"
            description = f"Extreme anomaly (z-score: {z_score:.2f})"
        elif z_score > 3.0:
            severity = "high"
            description = f"Strong anomaly (z-score: {z_score:.2f})"
        elif z_score > threshold_std:
            severity = "medium"
            description = f"Moderate anomaly (z-score: {z_score:.2f})"
        else:
            severity = "low"
            description = "Normal variation"
        
        return {
            "is_anomaly": is_anomaly,
            "z_score": z_score,
            "severity": severity,
            "description": description,
            "historical_mean": mean_val,
            "historical_std": std_val
        }
    
    @staticmethod
    def analyze_trend(values: List[float]) -> Dict[str, Any]:
        """Analyze trend in performance metrics using linear regression."""
        if len(values) < 10:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        n = len(values)
        x_values = list(range(n))
        
        # Calculate linear regression
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return {"trend": "stable", "confidence": 0.0}
        
        slope = numerator / denominator
        
        # Calculate R-squared for confidence
        y_pred = [slope * (x - x_mean) + y_mean for x in x_values]
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0, min(1, r_squared))
        
        # Classify trend
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0.001:
            trend = "improving"
        else:
            trend = "degrading"
        
        return {
            "trend": trend,
            "slope": slope,
            "confidence": confidence,
            "r_squared": r_squared
        }
    
    @staticmethod
    def detect_seasonal_patterns(values: List[float], period: int = 24) -> Dict[str, Any]:
        """Enhanced seasonal pattern detection."""
        if len(values) < period * 3:
            return {"has_pattern": False, "confidence": 0.0}
        
        n = len(values)
        mean_val = statistics.mean(values)
        
        # Calculate autocorrelation at seasonal lag
        autocorrelations = []
        for lag in [period, period * 2]:
            if n > lag:
                numerator = sum((values[i] - mean_val) * (values[i - lag] - mean_val) 
                               for i in range(lag, n))
                denominator = sum((values[i] - mean_val) ** 2 for i in range(n))
                
                if denominator > 0:
                    autocorr = numerator / denominator
                    autocorrelations.append(abs(autocorr))
        
        if not autocorrelations:
            return {"has_pattern": False, "confidence": 0.0}
        
        max_autocorr = max(autocorrelations)
        has_pattern = max_autocorr > 0.3
        
        return {
            "has_pattern": has_pattern,
            "confidence": max_autocorr,
            "pattern_strength": "strong" if max_autocorr > 0.7 else "moderate" if max_autocorr > 0.5 else "weak",
            "autocorrelations": autocorrelations
        }