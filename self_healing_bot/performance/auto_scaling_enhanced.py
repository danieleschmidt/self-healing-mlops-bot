"""Enhanced auto-scaling with ML predictions, multi-dimensional scaling, and cloud integration."""

import asyncio
import time
import math
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import pickle
import statistics
from pathlib import Path

# Cloud provider clients (would be imported based on availability)
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import compute_v1
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.compute import ComputeManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    KUBERNETES = "kubernetes"
    LOCAL = "local"


class ScalingDimension(Enum):
    """Different dimensions for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    ACTIVE_CONNECTIONS = "active_connections"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass
class MetricSample:
    """A single metric sample with timestamp."""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingDecision:
    """Represents a scaling decision with rationale."""
    timestamp: datetime
    dimension: ScalingDimension
    direction: ScalingDirection
    current_capacity: int
    target_capacity: int
    confidence: float
    reasoning: str
    predicted_metrics: Dict[str, float]
    cost_impact: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "dimension": self.dimension.value,
            "direction": self.direction.value,
            "current_capacity": self.current_capacity,
            "target_capacity": self.target_capacity,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "predicted_metrics": self.predicted_metrics,
            "cost_impact": self.cost_impact
        }


class TimeSeriesPredictor:
    """Simple time series predictor for metric forecasting."""
    
    def __init__(self, history_size: int = 1440):  # 24 hours at 1-minute intervals
        self.history_size = history_size
        self.data: deque = deque(maxlen=history_size)
        self.seasonal_patterns: Dict[int, float] = {}  # hour -> adjustment factor
        self.trend_window = 60  # 1 hour for trend calculation
        
    def add_sample(self, timestamp: float, value: float):
        """Add a new sample to the predictor."""
        self.data.append(MetricSample(timestamp, value))
        
        # Update seasonal patterns
        hour = datetime.fromtimestamp(timestamp).hour
        if hour not in self.seasonal_patterns:
            self.seasonal_patterns[hour] = []
        
        # Keep only recent samples for this hour
        self.seasonal_patterns[hour] = [value] + self.seasonal_patterns.get(hour, [])[:20]
    
    def predict(self, horizon_minutes: int = 30) -> Tuple[float, float]:
        """Predict future value and return (prediction, confidence)."""
        if len(self.data) < 10:
            return 0.0, 0.0
        
        # Get recent values
        recent_values = [sample.value for sample in list(self.data)[-self.trend_window:]]
        
        # Calculate trend (simple linear regression)
        if len(recent_values) >= 2:
            x = np.arange(len(recent_values))
            coeffs = np.polyfit(x, recent_values, 1)
            trend = coeffs[0]  # slope
            base_value = coeffs[1] + coeffs[0] * len(recent_values)
        else:
            trend = 0
            base_value = recent_values[-1]
        
        # Apply seasonal adjustment
        future_time = time.time() + (horizon_minutes * 60)
        future_hour = datetime.fromtimestamp(future_time).hour
        
        seasonal_factor = 1.0
        if future_hour in self.seasonal_patterns and self.seasonal_patterns[future_hour]:
            hour_average = statistics.mean(self.seasonal_patterns[future_hour])
            overall_average = statistics.mean([sample.value for sample in self.data])
            if overall_average > 0:
                seasonal_factor = hour_average / overall_average
        
        # Make prediction
        prediction = (base_value + trend * horizon_minutes) * seasonal_factor
        
        # Calculate confidence based on prediction stability
        recent_predictions = []
        for i in range(max(1, len(recent_values) - 10), len(recent_values)):
            if i >= 2:
                pred_x = np.arange(i)
                pred_coeffs = np.polyfit(pred_x, recent_values[:i], 1)
                pred_value = pred_coeffs[1] + pred_coeffs[0] * i
                recent_predictions.append(pred_value)
        
        if len(recent_predictions) > 1:
            prediction_variance = np.var(recent_predictions)
            confidence = max(0.1, min(0.9, 1.0 / (1.0 + prediction_variance)))
        else:
            confidence = 0.5
        
        return max(0, prediction), confidence
    
    def detect_anomaly(self, current_value: float) -> Tuple[bool, float]:
        """Detect if current value is anomalous."""
        if len(self.data) < 20:
            return False, 0.0
        
        recent_values = [sample.value for sample in list(self.data)[-20:]]
        mean_val = statistics.mean(recent_values)
        std_val = statistics.stdev(recent_values)
        
        if std_val == 0:
            return False, 0.0
        
        z_score = abs(current_value - mean_val) / std_val
        is_anomaly = z_score > 2.5
        
        return is_anomaly, z_score


class MultiDimensionalScaler:
    """Multi-dimensional auto-scaler with ML predictions."""
    
    def __init__(
        self,
        cloud_provider: CloudProvider = CloudProvider.LOCAL,
        enable_ml_predictions: bool = True,
        cost_optimization: bool = True
    ):
        self.cloud_provider = cloud_provider
        self.enable_ml_predictions = enable_ml_predictions
        self.cost_optimization = cost_optimization
        
        # Predictors for each dimension
        self.predictors: Dict[ScalingDimension, TimeSeriesPredictor] = {
            dimension: TimeSeriesPredictor() for dimension in ScalingDimension
        }
        
        # Current metrics
        self.current_metrics: Dict[ScalingDimension, float] = {}
        
        # Scaling configuration
        self.scaling_config: Dict[ScalingDimension, Dict[str, Any]] = {
            ScalingDimension.CPU_UTILIZATION: {
                "scale_up_threshold": 75.0,
                "scale_down_threshold": 25.0,
                "weight": 0.3,
                "critical_threshold": 90.0
            },
            ScalingDimension.MEMORY_UTILIZATION: {
                "scale_up_threshold": 80.0,
                "scale_down_threshold": 30.0,
                "weight": 0.25,
                "critical_threshold": 95.0
            },
            ScalingDimension.QUEUE_DEPTH: {
                "scale_up_threshold": 50.0,
                "scale_down_threshold": 10.0,
                "weight": 0.2,
                "critical_threshold": 100.0
            },
            ScalingDimension.RESPONSE_TIME: {
                "scale_up_threshold": 1000.0,  # ms
                "scale_down_threshold": 200.0,
                "weight": 0.15,
                "critical_threshold": 5000.0
            },
            ScalingDimension.ERROR_RATE: {
                "scale_up_threshold": 5.0,  # percent
                "scale_down_threshold": 1.0,
                "weight": 0.1,
                "critical_threshold": 20.0
            }
        }
        
        # Capacity management
        self.current_capacity = 3
        self.min_capacity = 1
        self.max_capacity = 100
        self.scaling_cooldown = 300  # 5 minutes
        self.last_scaling_time = 0
        
        # Decision history
        self.decision_history: deque = deque(maxlen=1000)
        
        # Cost tracking
        self.cost_per_instance_hour = 0.10  # Default cost
        
        # Cloud provider clients
        self.cloud_client = None
        self._initialize_cloud_client()
    
    def _initialize_cloud_client(self):
        """Initialize cloud provider client."""
        try:
            if self.cloud_provider == CloudProvider.AWS and AWS_AVAILABLE:
                self.cloud_client = boto3.client('ec2')
            elif self.cloud_provider == CloudProvider.GCP and GCP_AVAILABLE:
                self.cloud_client = compute_v1.InstancesClient()
            elif self.cloud_provider == CloudProvider.AZURE and AZURE_AVAILABLE:
                credential = DefaultAzureCredential()
                self.cloud_client = ComputeManagementClient(credential, "subscription-id")
            else:
                logger.info(f"Using local scaling for provider: {self.cloud_provider.value}")
        except Exception as e:
            logger.warning(f"Failed to initialize cloud client: {e}")
    
    def update_metric(self, dimension: ScalingDimension, value: float, metadata: Dict[str, Any] = None):
        """Update a metric value."""
        current_time = time.time()
        self.current_metrics[dimension] = value
        
        if self.enable_ml_predictions:
            self.predictors[dimension].add_sample(current_time, value)
    
    async def evaluate_scaling_decision(self) -> Optional[ScalingDecision]:
        """Evaluate whether scaling is needed and return decision."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_time < self.scaling_cooldown:
            return None
        
        # Calculate weighted scaling score
        scaling_scores = {}
        predicted_metrics = {}
        reasoning_parts = []
        
        for dimension, config in self.scaling_config.items():
            current_value = self.current_metrics.get(dimension, 0)
            
            # Get prediction if ML is enabled
            if self.enable_ml_predictions:
                predicted_value, confidence = self.predictors[dimension].predict(horizon_minutes=15)
                predicted_metrics[dimension.value] = predicted_value
                
                # Use predicted value if confidence is high enough
                if confidence > 0.6:
                    evaluation_value = predicted_value
                    reasoning_parts.append(f"{dimension.value}: predicted {predicted_value:.2f} (confidence {confidence:.2f})")
                else:
                    evaluation_value = current_value
                    reasoning_parts.append(f"{dimension.value}: current {current_value:.2f} (low prediction confidence)")
            else:
                evaluation_value = current_value
                predicted_metrics[dimension.value] = current_value
                reasoning_parts.append(f"{dimension.value}: {current_value:.2f}")
            
            # Calculate scaling score
            weight = config["weight"]
            scale_up_threshold = config["scale_up_threshold"]
            scale_down_threshold = config["scale_down_threshold"]
            critical_threshold = config["critical_threshold"]
            
            if evaluation_value >= critical_threshold:
                # Critical situation - immediate scale up
                scaling_scores[dimension] = 2.0 * weight
            elif evaluation_value >= scale_up_threshold:
                # Scale up needed
                excess = (evaluation_value - scale_up_threshold) / (critical_threshold - scale_up_threshold)
                scaling_scores[dimension] = excess * weight
            elif evaluation_value <= scale_down_threshold:
                # Scale down possible
                deficit = (scale_down_threshold - evaluation_value) / scale_down_threshold
                scaling_scores[dimension] = -deficit * weight
            else:
                # No scaling needed for this dimension
                scaling_scores[dimension] = 0.0
        
        # Calculate overall scaling score
        overall_score = sum(scaling_scores.values())
        
        # Determine scaling decision
        if overall_score > 0.3:
            direction = ScalingDirection.UP
            scale_factor = min(2.0, 1.0 + overall_score)
            target_capacity = min(self.max_capacity, math.ceil(self.current_capacity * scale_factor))
        elif overall_score < -0.2:
            direction = ScalingDirection.DOWN
            scale_factor = max(0.5, 1.0 + overall_score)
            target_capacity = max(self.min_capacity, math.floor(self.current_capacity * scale_factor))
        else:
            return None  # No scaling needed
        
        # Cost consideration
        cost_impact = None
        if self.cost_optimization:
            capacity_change = target_capacity - self.current_capacity
            cost_impact = capacity_change * self.cost_per_instance_hour
            
            # Adjust decision based on cost-benefit
            if direction == ScalingDirection.UP and cost_impact > 10.0:  # Expensive scale up
                if overall_score < 0.7:  # Not critical enough
                    target_capacity = min(target_capacity, self.current_capacity + 1)  # Conservative scaling
                    cost_impact = self.cost_per_instance_hour
        
        # Calculate confidence
        confidence = min(0.9, max(0.1, abs(overall_score)))
        
        # Create decision
        decision = ScalingDecision(
            timestamp=datetime.now(),
            dimension=max(scaling_scores.keys(), key=lambda k: abs(scaling_scores[k])),
            direction=direction,
            current_capacity=self.current_capacity,
            target_capacity=target_capacity,
            confidence=confidence,
            reasoning=f"Score: {overall_score:.3f}, " + ", ".join(reasoning_parts[:3]),
            predicted_metrics=predicted_metrics,
            cost_impact=cost_impact
        )
        
        return decision
    
    async def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision."""
        try:
            success = False
            
            if self.cloud_provider == CloudProvider.AWS and self.cloud_client:
                success = await self._scale_aws(decision.target_capacity)
            elif self.cloud_provider == CloudProvider.GCP and self.cloud_client:
                success = await self._scale_gcp(decision.target_capacity)
            elif self.cloud_provider == CloudProvider.AZURE and self.cloud_client:
                success = await self._scale_azure(decision.target_capacity)
            elif self.cloud_provider == CloudProvider.KUBERNETES:
                success = await self._scale_kubernetes(decision.target_capacity)
            else:
                # Local/mock scaling
                success = await self._scale_local(decision.target_capacity)
            
            if success:
                self.current_capacity = decision.target_capacity
                self.last_scaling_time = time.time()
                self.decision_history.append(decision)
                
                logger.info(f"Scaling executed: {decision.current_capacity} -> {decision.target_capacity} "
                           f"({decision.direction.value}) - {decision.reasoning}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
            return False
    
    async def _scale_aws(self, target_capacity: int) -> bool:
        """Scale on AWS using Auto Scaling Groups."""
        try:
            autoscaling = boto3.client('autoscaling')
            
            # Update Auto Scaling Group capacity
            response = autoscaling.update_auto_scaling_group(
                AutoScalingGroupName='bot-asg',  # This would be configurable
                DesiredCapacity=target_capacity,
                MinSize=self.min_capacity,
                MaxSize=self.max_capacity
            )
            
            return True
        except Exception as e:
            logger.error(f"AWS scaling error: {e}")
            return False
    
    async def _scale_gcp(self, target_capacity: int) -> bool:
        """Scale on GCP using Managed Instance Groups."""
        try:
            # This would implement GCP scaling
            logger.info(f"Would scale GCP instance group to {target_capacity}")
            return True
        except Exception as e:
            logger.error(f"GCP scaling error: {e}")
            return False
    
    async def _scale_azure(self, target_capacity: int) -> bool:
        """Scale on Azure using Virtual Machine Scale Sets."""
        try:
            # This would implement Azure scaling
            logger.info(f"Would scale Azure VMSS to {target_capacity}")
            return True
        except Exception as e:
            logger.error(f"Azure scaling error: {e}")
            return False
    
    async def _scale_kubernetes(self, target_capacity: int) -> bool:
        """Scale on Kubernetes using HPA or direct deployment scaling."""
        try:
            # This would implement Kubernetes scaling
            logger.info(f"Would scale Kubernetes deployment to {target_capacity}")
            return True
        except Exception as e:
            logger.error(f"Kubernetes scaling error: {e}")
            return False
    
    async def _scale_local(self, target_capacity: int) -> bool:
        """Local/mock scaling for development and testing."""
        logger.info(f"Local scaling: {self.current_capacity} -> {target_capacity}")
        await asyncio.sleep(1)  # Simulate scaling delay
        return True
    
    def detect_scaling_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous scaling patterns."""
        anomalies = []
        
        for dimension, predictor in self.predictors.items():
            current_value = self.current_metrics.get(dimension, 0)
            is_anomaly, z_score = predictor.detect_anomaly(current_value)
            
            if is_anomaly:
                anomalies.append({
                    "dimension": dimension.value,
                    "current_value": current_value,
                    "z_score": z_score,
                    "severity": "high" if z_score > 3 else "medium"
                })
        
        return anomalies
    
    def get_scaling_recommendations(self) -> List[Dict[str, str]]:
        """Get scaling optimization recommendations."""
        recommendations = []
        
        # Analyze recent scaling decisions
        if len(self.decision_history) >= 10:
            recent_decisions = list(self.decision_history)[-10:]
            
            # Check for oscillation (frequent up/down scaling)
            directions = [d.direction for d in recent_decisions]
            up_count = sum(1 for d in directions if d == ScalingDirection.UP)
            down_count = sum(1 for d in directions if d == ScalingDirection.DOWN)
            
            if abs(up_count - down_count) <= 2 and up_count + down_count >= 6:
                recommendations.append({
                    "type": "configuration",
                    "title": "Reduce scaling oscillation",
                    "description": "Frequent up/down scaling detected. Consider increasing cooldown period or adjusting thresholds.",
                    "action": "Increase scaling_cooldown or adjust scaling thresholds"
                })
            
            # Check for cost optimization
            if self.cost_optimization:
                total_cost_impact = sum(d.cost_impact or 0 for d in recent_decisions)
                if total_cost_impact > 50.0:  # High cost
                    recommendations.append({
                        "type": "cost",
                        "title": "High scaling costs detected",
                        "description": f"Recent scaling decisions had ${total_cost_impact:.2f} cost impact.",
                        "action": "Review scaling thresholds and consider scheduled scaling"
                    })
            
            # Check for underutilized capacity
            avg_cpu = statistics.mean([
                self.current_metrics.get(ScalingDimension.CPU_UTILIZATION, 0) 
                for _ in range(len(recent_decisions))
            ])
            
            if avg_cpu < 30 and self.current_capacity > self.min_capacity:
                recommendations.append({
                    "type": "optimization",
                    "title": "Underutilized capacity",
                    "description": f"Average CPU utilization is {avg_cpu:.1f}%.",
                    "action": "Consider reducing base capacity or implementing more aggressive scale-down"
                })
        
        return recommendations
    
    def export_model(self, file_path: str):
        """Export ML models for backup or transfer."""
        try:
            model_data = {
                "predictors": {},
                "config": self.scaling_config,
                "history": list(self.decision_history)
            }
            
            for dimension, predictor in self.predictors.items():
                model_data["predictors"][dimension.value] = {
                    "data": list(predictor.data),
                    "seasonal_patterns": predictor.seasonal_patterns
                }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Models exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export models: {e}")
            return False
    
    def import_model(self, file_path: str):
        """Import ML models from backup."""
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore predictors
            for dimension_name, predictor_data in model_data["predictors"].items():
                dimension = ScalingDimension(dimension_name)
                
                # Restore data
                self.predictors[dimension].data.clear()
                for sample_data in predictor_data["data"]:
                    if isinstance(sample_data, dict):
                        sample = MetricSample(**sample_data)
                    else:
                        # Backward compatibility
                        sample = sample_data
                    self.predictors[dimension].data.append(sample)
                
                # Restore seasonal patterns
                self.predictors[dimension].seasonal_patterns = predictor_data["seasonal_patterns"]
            
            # Restore configuration
            if "config" in model_data:
                self.scaling_config.update(model_data["config"])
            
            # Restore decision history
            if "history" in model_data:
                self.decision_history.clear()
                self.decision_history.extend(model_data["history"])
            
            logger.info(f"Models imported from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import models: {e}")
            return False
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        recent_decisions = list(self.decision_history)[-50:]  # Last 50 decisions
        
        stats = {
            "current_status": {
                "capacity": self.current_capacity,
                "min_capacity": self.min_capacity,
                "max_capacity": self.max_capacity,
                "cloud_provider": self.cloud_provider.value,
                "last_scaling_time": self.last_scaling_time
            },
            "current_metrics": {
                dimension.value: value for dimension, value in self.current_metrics.items()
            },
            "predictions": {},
            "decision_history": {
                "total_decisions": len(self.decision_history),
                "recent_decisions": [d.to_dict() for d in recent_decisions],
                "scale_up_count": sum(1 for d in recent_decisions if d.direction == ScalingDirection.UP),
                "scale_down_count": sum(1 for d in recent_decisions if d.direction == ScalingDirection.DOWN),
                "avg_confidence": statistics.mean([d.confidence for d in recent_decisions]) if recent_decisions else 0
            },
            "anomalies": self.detect_scaling_anomalies(),
            "recommendations": self.get_scaling_recommendations(),
            "cost_tracking": {
                "cost_per_instance_hour": self.cost_per_instance_hour,
                "estimated_monthly_cost": self.current_capacity * self.cost_per_instance_hour * 24 * 30
            }
        }
        
        # Add predictions for each dimension
        for dimension in ScalingDimension:
            if self.enable_ml_predictions:
                prediction, confidence = self.predictors[dimension].predict()
                stats["predictions"][dimension.value] = {
                    "predicted_value": prediction,
                    "confidence": confidence,
                    "samples_count": len(self.predictors[dimension].data)
                }
        
        return stats


class EnhancedAutoScaler:
    """Enhanced auto-scaler with multiple scaling strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize scalers for different services
        self.scalers: Dict[str, MultiDimensionalScaler] = {}
        
        # Global scaling settings
        self.enable_predictive_scaling = self.config.get("enable_predictive_scaling", True)
        self.enable_cost_optimization = self.config.get("enable_cost_optimization", True)
        self.scaling_interval = self.config.get("scaling_interval", 60)  # seconds
        
        # Background tasks
        self.scaling_task: Optional[asyncio.Task] = None
        self.monitoring_enabled = True
    
    def add_service(
        self,
        service_name: str,
        cloud_provider: CloudProvider = CloudProvider.LOCAL,
        min_capacity: int = 1,
        max_capacity: int = 20,
        scaling_config: Dict[ScalingDimension, Dict[str, Any]] = None
    ):
        """Add a service to be auto-scaled."""
        scaler = MultiDimensionalScaler(
            cloud_provider=cloud_provider,
            enable_ml_predictions=self.enable_predictive_scaling,
            cost_optimization=self.enable_cost_optimization
        )
        
        scaler.min_capacity = min_capacity
        scaler.max_capacity = max_capacity
        
        if scaling_config:
            scaler.scaling_config.update(scaling_config)
        
        self.scalers[service_name] = scaler
        logger.info(f"Added service '{service_name}' for auto-scaling")
    
    def update_service_metric(
        self,
        service_name: str,
        dimension: ScalingDimension,
        value: float,
        metadata: Dict[str, Any] = None
    ):
        """Update a metric for a specific service."""
        if service_name in self.scalers:
            self.scalers[service_name].update_metric(dimension, value, metadata)
    
    async def start_monitoring(self):
        """Start the auto-scaling monitoring loop."""
        if self.scaling_task and not self.scaling_task.done():
            return
        
        self.monitoring_enabled = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Auto-scaling monitoring started")
    
    async def stop_monitoring(self):
        """Stop the auto-scaling monitoring loop."""
        self.monitoring_enabled = False
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto-scaling monitoring stopped")
    
    async def _scaling_loop(self):
        """Main auto-scaling loop."""
        while self.monitoring_enabled:
            try:
                for service_name, scaler in self.scalers.items():
                    # Evaluate scaling decision
                    decision = await scaler.evaluate_scaling_decision()
                    
                    if decision:
                        logger.info(f"Scaling decision for {service_name}: {decision.to_dict()}")
                        
                        # Execute scaling decision
                        success = await scaler.execute_scaling_decision(decision)
                        
                        if not success:
                            logger.error(f"Failed to execute scaling for {service_name}")
                
                await asyncio.sleep(self.scaling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all services."""
        return {
            "global_config": {
                "enable_predictive_scaling": self.enable_predictive_scaling,
                "enable_cost_optimization": self.enable_cost_optimization,
                "scaling_interval": self.scaling_interval,
                "monitoring_enabled": self.monitoring_enabled
            },
            "services": {
                service_name: scaler.get_comprehensive_stats()
                for service_name, scaler in self.scalers.items()
            }
        }
    
    async def force_scaling(self, service_name: str, target_capacity: int) -> bool:
        """Force scaling for a specific service."""
        if service_name not in self.scalers:
            logger.error(f"Service '{service_name}' not found")
            return False
        
        scaler = self.scalers[service_name]
        
        # Create manual scaling decision
        decision = ScalingDecision(
            timestamp=datetime.now(),
            dimension=ScalingDimension.CPU_UTILIZATION,  # Placeholder
            direction=ScalingDirection.UP if target_capacity > scaler.current_capacity else ScalingDirection.DOWN,
            current_capacity=scaler.current_capacity,
            target_capacity=target_capacity,
            confidence=1.0,
            reasoning="Manual scaling request",
            predicted_metrics={}
        )
        
        return await scaler.execute_scaling_decision(decision)


# Global auto-scaler instance
auto_scaler = EnhancedAutoScaler()