"""Auto-scaling system for dynamic resource management."""

import asyncio
import time
import math
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json

from ..monitoring.logging import get_logger, performance_logger
from ..monitoring.metrics import metrics
from .concurrency import ResourceMonitor

logger = get_logger(__name__)


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: datetime
    action: str  # "scale_up", "scale_down", "no_action"
    metric_name: str
    metric_value: float
    threshold: float
    previous_capacity: int
    new_capacity: int
    reason: str


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling policies."""
    metric_name: str
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    min_capacity: int = 1
    max_capacity: int = 100
    cooldown_period: int = 300  # seconds
    evaluation_periods: int = 2
    datapoints_to_alarm: int = 2


class AutoScaler:
    """Intelligent auto-scaling system for bot components."""
    
    def __init__(self):
        self.policies: Dict[str, ScalingPolicy] = {}
        self.current_capacity: Dict[str, int] = {}
        self.scaling_history: Dict[str, deque] = {}
        self.metrics_history: Dict[str, deque] = {}
        self.last_scaling_action: Dict[str, float] = {}
        self.resource_monitor = ResourceMonitor()
        self.scaling_enabled = True
        self.max_history_size = 100
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default scaling policies."""
        # Worker threads scaling
        self.add_policy(ScalingPolicy(
            metric_name="worker_utilization",
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            scale_up_factor=1.5,
            scale_down_factor=0.7,
            min_capacity=2,
            max_capacity=50,
            cooldown_period=300,
            evaluation_periods=3,
            datapoints_to_alarm=2
        ))
        
        # Event processing queue scaling
        self.add_policy(ScalingPolicy(
            metric_name="queue_length",
            scale_up_threshold=20.0,
            scale_down_threshold=5.0,
            scale_up_factor=2.0,
            scale_down_factor=0.6,
            min_capacity=1,
            max_capacity=20,
            cooldown_period=180,
            evaluation_periods=2,
            datapoints_to_alarm=2
        ))
        
        # Detector concurrency scaling
        self.add_policy(ScalingPolicy(
            metric_name="detector_latency",
            scale_up_threshold=5.0,  # 5 seconds
            scale_down_threshold=1.0,  # 1 second
            scale_up_factor=1.3,
            scale_down_factor=0.8,
            min_capacity=1,
            max_capacity=10,
            cooldown_period=240,
            evaluation_periods=3,
            datapoints_to_alarm=3
        ))
    
    def add_policy(self, policy: ScalingPolicy):
        """Add or update a scaling policy."""
        self.policies[policy.metric_name] = policy
        self.current_capacity[policy.metric_name] = policy.min_capacity
        self.scaling_history[policy.metric_name] = deque(maxlen=self.max_history_size)
        self.metrics_history[policy.metric_name] = deque(maxlen=self.max_history_size)
        self.last_scaling_action[policy.metric_name] = 0
        
        logger.info(f"Added scaling policy for {policy.metric_name}")
    
    async def record_metric(self, metric_name: str, value: float):
        """Record a metric value for scaling decisions."""
        if metric_name not in self.policies:
            return
        
        timestamp = time.time()
        self.metrics_history[metric_name].append((timestamp, value))
        
        # Check if scaling action is needed
        if self.scaling_enabled:
            await self._evaluate_scaling_action(metric_name)
    
    async def _evaluate_scaling_action(self, metric_name: str):
        """Evaluate if scaling action is needed for a metric."""
        policy = self.policies[metric_name]
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_action[metric_name] < policy.cooldown_period:
            return
        
        # Get recent metric values
        recent_metrics = self._get_recent_metrics(metric_name, policy.evaluation_periods)
        if len(recent_metrics) < policy.evaluation_periods:
            return
        
        # Check if threshold is breached consistently
        scale_up_count = sum(1 for value in recent_metrics if value > policy.scale_up_threshold)
        scale_down_count = sum(1 for value in recent_metrics if value < policy.scale_down_threshold)
        
        current_capacity = self.current_capacity[metric_name]
        scaling_action = None
        new_capacity = current_capacity
        
        if scale_up_count >= policy.datapoints_to_alarm:
            # Scale up
            new_capacity = min(
                policy.max_capacity,
                math.ceil(current_capacity * policy.scale_up_factor)
            )
            scaling_action = "scale_up"
            
        elif scale_down_count >= policy.datapoints_to_alarm:
            # Scale down
            new_capacity = max(
                policy.min_capacity,
                math.floor(current_capacity * policy.scale_down_factor)
            )
            scaling_action = "scale_down"
        
        if scaling_action and new_capacity != current_capacity:
            await self._execute_scaling_action(metric_name, scaling_action, new_capacity)
    
    def _get_recent_metrics(self, metric_name: str, count: int) -> List[float]:
        """Get recent metric values."""
        metrics_list = list(self.metrics_history[metric_name])
        recent_metrics = metrics_list[-count:] if len(metrics_list) >= count else metrics_list
        return [value for _, value in recent_metrics]
    
    async def _execute_scaling_action(self, metric_name: str, action: str, new_capacity: int):
        """Execute a scaling action."""
        policy = self.policies[metric_name]
        current_capacity = self.current_capacity[metric_name]
        current_metric_value = self._get_recent_metrics(metric_name, 1)[0] if self._get_recent_metrics(metric_name, 1) else 0
        
        # Determine threshold that triggered the action
        threshold = policy.scale_up_threshold if action == "scale_up" else policy.scale_down_threshold
        
        # Create scaling event
        scaling_event = ScalingEvent(
            timestamp=datetime.utcnow(),
            action=action,
            metric_name=metric_name,
            metric_value=current_metric_value,
            threshold=threshold,
            previous_capacity=current_capacity,
            new_capacity=new_capacity,
            reason=f"{metric_name} {action} triggered: {current_metric_value:.2f} {'>' if action == 'scale_up' else '<'} {threshold}"
        )
        
        # Execute the scaling
        success = await self._apply_scaling(metric_name, new_capacity)
        
        if success:
            self.current_capacity[metric_name] = new_capacity
            self.last_scaling_action[metric_name] = time.time()
            self.scaling_history[metric_name].append(scaling_event)
            
            logger.info(f"Scaling {action} for {metric_name}: {current_capacity} -> {new_capacity}")
            
            # Log to performance metrics
            performance_logger.log_execution_time(
                f"auto_scaling_{action}",
                0,  # Scaling is instantaneous from logging perspective
                True,
                metric_name=metric_name,
                previous_capacity=current_capacity,
                new_capacity=new_capacity
            )
        else:
            logger.error(f"Failed to execute scaling {action} for {metric_name}")
    
    async def _apply_scaling(self, metric_name: str, new_capacity: int) -> bool:
        """Apply scaling to the actual resource."""
        try:
            if metric_name == "worker_utilization":
                # Scale worker threads
                await self._scale_worker_threads(new_capacity)
            elif metric_name == "queue_length":
                # Scale event processing queue
                await self._scale_event_processors(new_capacity)
            elif metric_name == "detector_latency":
                # Scale detector concurrency
                await self._scale_detector_concurrency(new_capacity)
            else:
                logger.warning(f"Unknown metric for scaling: {metric_name}")
                return False
            
            return True
            
        except Exception as e:
            logger.exception(f"Error applying scaling for {metric_name}: {e}")
            return False
    
    async def _scale_worker_threads(self, new_capacity: int):
        """Scale worker thread pool."""
        # This would integrate with the AdaptiveExecutor
        logger.info(f"Would scale worker threads to {new_capacity}")
    
    async def _scale_event_processors(self, new_capacity: int):
        """Scale event processing capacity."""
        # This would integrate with event processing queue
        logger.info(f"Would scale event processors to {new_capacity}")
    
    async def _scale_detector_concurrency(self, new_capacity: int):
        """Scale detector processing concurrency."""
        # This would integrate with detector execution
        logger.info(f"Would scale detector concurrency to {new_capacity}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        status = {
            "scaling_enabled": self.scaling_enabled,
            "policies": {},
            "current_capacity": self.current_capacity.copy(),
            "recent_events": {}
        }
        
        for metric_name, policy in self.policies.items():
            status["policies"][metric_name] = {
                "scale_up_threshold": policy.scale_up_threshold,
                "scale_down_threshold": policy.scale_down_threshold,
                "min_capacity": policy.min_capacity,
                "max_capacity": policy.max_capacity,
                "cooldown_period": policy.cooldown_period
            }
            
            # Get recent scaling events
            recent_events = list(self.scaling_history[metric_name])[-5:]  # Last 5 events
            status["recent_events"][metric_name] = [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "action": event.action,
                    "metric_value": event.metric_value,
                    "threshold": event.threshold,
                    "capacity_change": f"{event.previous_capacity} -> {event.new_capacity}",
                    "reason": event.reason
                }
                for event in recent_events
            ]
        
        return status
    
    def enable_scaling(self):
        """Enable auto-scaling."""
        self.scaling_enabled = True
        logger.info("Auto-scaling enabled")
    
    def disable_scaling(self):
        """Disable auto-scaling."""
        self.scaling_enabled = False
        logger.info("Auto-scaling disabled")
    
    async def force_scale(self, metric_name: str, capacity: int) -> bool:
        """Force scaling to a specific capacity."""
        if metric_name not in self.policies:
            logger.error(f"No policy found for metric: {metric_name}")
            return False
        
        policy = self.policies[metric_name]
        if capacity < policy.min_capacity or capacity > policy.max_capacity:
            logger.error(f"Capacity {capacity} outside allowed range [{policy.min_capacity}, {policy.max_capacity}]")
            return False
        
        current_capacity = self.current_capacity[metric_name]
        action = "scale_up" if capacity > current_capacity else "scale_down"
        
        # Create manual scaling event
        scaling_event = ScalingEvent(
            timestamp=datetime.utcnow(),
            action=f"manual_{action}",
            metric_name=metric_name,
            metric_value=0,  # Manual scaling doesn't have metric trigger
            threshold=0,
            previous_capacity=current_capacity,
            new_capacity=capacity,
            reason=f"Manual scaling to {capacity}"
        )
        
        success = await self._apply_scaling(metric_name, capacity)
        
        if success:
            self.current_capacity[metric_name] = capacity
            self.scaling_history[metric_name].append(scaling_event)
            logger.info(f"Manual scaling for {metric_name}: {current_capacity} -> {capacity}")
        
        return success


class PredictiveScaler:
    """Predictive scaling based on historical patterns."""
    
    def __init__(self, auto_scaler: AutoScaler):
        self.auto_scaler = auto_scaler
        self.prediction_window = 300  # 5 minutes ahead
        self.historical_patterns = {}
        self.pattern_recognition_enabled = True
    
    async def predict_and_scale(self):
        """Predict future resource needs and pre-scale."""
        if not self.pattern_recognition_enabled:
            return
        
        current_time = datetime.utcnow()
        hour_of_day = current_time.hour
        day_of_week = current_time.weekday()
        
        for metric_name in self.auto_scaler.policies.keys():
            predicted_load = await self._predict_load(metric_name, hour_of_day, day_of_week)
            
            if predicted_load:
                await self._proactive_scale(metric_name, predicted_load)
    
    async def _predict_load(self, metric_name: str, hour: int, day: int) -> Optional[float]:
        """Predict load based on historical patterns."""
        # This would implement actual prediction logic
        # For now, return None to indicate no prediction
        return None
    
    async def _proactive_scale(self, metric_name: str, predicted_load: float):
        """Proactively scale based on prediction."""
        policy = self.auto_scaler.policies[metric_name]
        current_capacity = self.auto_scaler.current_capacity[metric_name]
        
        # Simple prediction-based scaling logic
        if predicted_load > policy.scale_up_threshold:
            recommended_capacity = min(
                policy.max_capacity,
                math.ceil(current_capacity * 1.2)  # Conservative proactive scaling
            )
            
            if recommended_capacity > current_capacity:
                logger.info(f"Proactive scaling for {metric_name}: predicted load {predicted_load:.2f}")
                await self.auto_scaler.force_scale(metric_name, recommended_capacity)


# Global auto-scaler instance
auto_scaler = AutoScaler()
predictive_scaler = PredictiveScaler(auto_scaler)