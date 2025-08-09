"""Advanced cache monitoring, analytics, and performance optimization system."""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import logging
import threading
import psutil
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of cache metrics."""
    HIT_RATE = "hit_rate"
    MISS_RATE = "miss_rate"
    EVICTION_RATE = "eviction_rate"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    COMPRESSION_RATIO = "compression_ratio"


@dataclass
class CacheAlert:
    """Cache performance alert."""
    timestamp: datetime
    severity: AlertSeverity
    metric: MetricType
    message: str
    current_value: float
    threshold: float
    level: str  # Cache level (L1, L2, L3)
    suggested_action: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "metric": self.metric.value,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "level": self.level,
            "suggested_action": self.suggested_action
        }


@dataclass
class PerformanceSnapshot:
    """Point-in-time cache performance snapshot."""
    timestamp: datetime
    level: str
    hit_rate: float
    miss_rate: float
    eviction_rate: float
    error_rate: float
    memory_usage_mb: float
    avg_response_time_ms: float
    throughput_ops_per_sec: float
    compression_ratio: float
    active_keys: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "eviction_rate": self.eviction_rate,
            "error_rate": self.error_rate,
            "memory_usage_mb": self.memory_usage_mb,
            "avg_response_time_ms": self.avg_response_time_ms,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "compression_ratio": self.compression_ratio,
            "active_keys": self.active_keys
        }


class CacheAnalytics:
    """Advanced cache analytics and insights."""
    
    def __init__(self):
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours at 1-minute intervals
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.access_patterns: Dict[str, Dict] = defaultdict(dict)
        self.key_popularity: Dict[str, int] = defaultdict(int)
        self.temporal_patterns: Dict[int, Dict] = defaultdict(dict)  # hour -> patterns
        
    def add_performance_snapshot(self, snapshot: PerformanceSnapshot):
        """Add a performance snapshot for analysis."""
        self.performance_history[snapshot.level].append(snapshot)
        
        # Update temporal patterns
        hour = snapshot.timestamp.hour
        if snapshot.level not in self.temporal_patterns[hour]:
            self.temporal_patterns[hour][snapshot.level] = {
                'hit_rates': [],
                'response_times': [],
                'throughput': []
            }
        
        self.temporal_patterns[hour][snapshot.level]['hit_rates'].append(snapshot.hit_rate)
        self.temporal_patterns[hour][snapshot.level]['response_times'].append(snapshot.avg_response_time_ms)
        self.temporal_patterns[hour][snapshot.level]['throughput'].append(snapshot.throughput_ops_per_sec)
    
    def add_response_time(self, level: str, response_time_ms: float):
        """Add response time measurement."""
        self.response_times[level].append(response_time_ms)
    
    def analyze_trends(self, level: str, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends for a cache level."""
        if level not in self.performance_history:
            return {"error": f"No data for level {level}"}
        
        snapshots = list(self.performance_history[level])
        if not snapshots:
            return {"error": f"No snapshots for level {level}"}
        
        # Filter by time window
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate trends
        hit_rates = [s.hit_rate for s in recent_snapshots]
        response_times = [s.avg_response_time_ms for s in recent_snapshots]
        memory_usage = [s.memory_usage_mb for s in recent_snapshots]
        throughput = [s.throughput_ops_per_sec for s in recent_snapshots]
        
        trends = {
            "hit_rate_trend": self._calculate_trend(hit_rates),
            "response_time_trend": self._calculate_trend(response_times),
            "memory_usage_trend": self._calculate_trend(memory_usage),
            "throughput_trend": self._calculate_trend(throughput),
            "current_metrics": {
                "hit_rate": recent_snapshots[-1].hit_rate,
                "avg_response_time_ms": recent_snapshots[-1].avg_response_time_ms,
                "memory_usage_mb": recent_snapshots[-1].memory_usage_mb,
                "throughput_ops_per_sec": recent_snapshots[-1].throughput_ops_per_sec
            },
            "statistics": {
                "hit_rate": self._calculate_statistics(hit_rates),
                "response_time": self._calculate_statistics(response_times),
                "memory_usage": self._calculate_statistics(memory_usage),
                "throughput": self._calculate_statistics(throughput)
            }
        }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend direction and strength."""
        if len(values) < 2:
            return {"direction": 0.0, "strength": 0.0}
        
        # Simple linear regression
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return {"direction": 0.0, "strength": 0.0}
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Calculate R-squared for trend strength
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in values)
        ss_res = sum((values[i] - (slope * x[i] + (sum_y - slope * sum_x) / n)) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "direction": slope,
            "strength": r_squared,
            "interpretation": self._interpret_trend(slope, r_squared)
        }
    
    def _interpret_trend(self, slope: float, r_squared: float) -> str:
        """Interpret trend direction and strength."""
        if r_squared < 0.3:
            return "no clear trend"
        elif abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            if r_squared > 0.7:
                return "strong upward trend"
            else:
                return "moderate upward trend"
        else:
            if r_squared > 0.7:
                return "strong downward trend"
            else:
                return "moderate downward trend"
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures."""
        if not values:
            return {}
        
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "percentile_95": np.percentile(values, 95) if len(values) > 1 else values[0],
            "percentile_99": np.percentile(values, 99) if len(values) > 1 else values[0]
        }
    
    def detect_anomalies(self, level: str, metric: MetricType, window_hours: int = 24) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical methods."""
        if level not in self.performance_history:
            return []
        
        snapshots = list(self.performance_history[level])
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 10:  # Need minimum data points
            return []
        
        # Extract metric values
        values = []
        for snapshot in recent_snapshots:
            if metric == MetricType.HIT_RATE:
                values.append(snapshot.hit_rate)
            elif metric == MetricType.RESPONSE_TIME:
                values.append(snapshot.avg_response_time_ms)
            elif metric == MetricType.MEMORY_USAGE:
                values.append(snapshot.memory_usage_mb)
            elif metric == MetricType.THROUGHPUT:
                values.append(snapshot.throughput_ops_per_sec)
            # Add more metrics as needed
        
        if not values:
            return []
        
        # Statistical anomaly detection using Z-score
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        anomalies = []
        threshold = 2.5  # Z-score threshold
        
        for i, (snapshot, value) in enumerate(zip(recent_snapshots, values)):
            if std_val > 0:
                z_score = abs(value - mean_val) / std_val
                if z_score > threshold:
                    anomalies.append({
                        "timestamp": snapshot.timestamp.isoformat(),
                        "metric": metric.value,
                        "value": value,
                        "z_score": z_score,
                        "expected_range": [mean_val - 2 * std_val, mean_val + 2 * std_val],
                        "severity": "high" if z_score > 3 else "medium"
                    })
        
        return anomalies
    
    def predict_performance(self, level: str, hours_ahead: int = 4) -> Dict[str, Any]:
        """Predict future performance based on historical patterns."""
        if level not in self.performance_history:
            return {"error": f"No data for level {level}"}
        
        snapshots = list(self.performance_history[level])
        if len(snapshots) < 20:  # Need sufficient history
            return {"error": "Insufficient historical data"}
        
        # Simple time series prediction using moving averages and trends
        recent_snapshots = snapshots[-20:]  # Last 20 data points
        
        hit_rates = [s.hit_rate for s in recent_snapshots]
        response_times = [s.avg_response_time_ms for s in recent_snapshots]
        memory_usage = [s.memory_usage_mb for s in recent_snapshots]
        
        predictions = {
            "forecast_horizon_hours": hours_ahead,
            "predicted_hit_rate": self._predict_value(hit_rates),
            "predicted_response_time_ms": self._predict_value(response_times),
            "predicted_memory_usage_mb": self._predict_value(memory_usage),
            "confidence": self._calculate_prediction_confidence(recent_snapshots)
        }
        
        return predictions
    
    def _predict_value(self, values: List[float]) -> Dict[str, float]:
        """Predict future value using exponential smoothing."""
        if not values:
            return {"value": 0.0, "trend": 0.0}
        
        alpha = 0.3  # Smoothing parameter
        
        # Simple exponential smoothing
        smoothed = values[0]
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # Calculate trend
        if len(values) > 1:
            recent_trend = (values[-1] - values[-5]) / 4 if len(values) >= 5 else (values[-1] - values[0]) / len(values)
        else:
            recent_trend = 0
        
        return {
            "value": smoothed + recent_trend,
            "trend": recent_trend
        }
    
    def _calculate_prediction_confidence(self, snapshots: List[PerformanceSnapshot]) -> float:
        """Calculate confidence level for predictions."""
        if len(snapshots) < 5:
            return 0.2
        
        # Calculate coefficient of variation for stability
        hit_rates = [s.hit_rate for s in snapshots]
        mean_hit_rate = statistics.mean(hit_rates)
        std_hit_rate = statistics.stdev(hit_rates) if len(hit_rates) > 1 else 0
        
        if mean_hit_rate == 0:
            return 0.3
        
        cv = std_hit_rate / mean_hit_rate
        confidence = max(0.2, min(0.9, 1.0 - cv))
        
        return confidence


class CacheMonitor:
    """Comprehensive cache monitoring and alerting system."""
    
    def __init__(self, cache_instance):
        self.cache = cache_instance
        self.analytics = CacheAnalytics()
        self.alerts: deque = deque(maxlen=1000)  # Keep last 1000 alerts
        self.alert_handlers: List[Callable] = []
        self.monitoring_enabled = True
        
        # Alert thresholds
        self.alert_thresholds = {
            "hit_rate_low": 0.7,
            "hit_rate_critical": 0.5,
            "response_time_high": 100.0,  # ms
            "response_time_critical": 500.0,  # ms
            "memory_usage_high": 0.8,  # 80% of limit
            "memory_usage_critical": 0.95,  # 95% of limit
            "error_rate_high": 0.05,  # 5%
            "error_rate_critical": 0.10,  # 10%
            "eviction_rate_high": 10.0,  # evictions per minute
            "eviction_rate_critical": 50.0  # evictions per minute
        }
        
        # Background monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.snapshot_interval = 60  # seconds
        self.last_snapshot_time = defaultdict(float)
        
        # Performance metrics tracking
        self.operation_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.throughput_counters: Dict[str, int] = defaultdict(int)
        self.last_throughput_reset = time.time()
    
    async def start_monitoring(self):
        """Start continuous cache monitoring."""
        if self.monitoring_task and not self.monitoring_task.done():
            return
        
        self.monitoring_enabled = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Cache monitoring started")
    
    async def stop_monitoring(self):
        """Stop cache monitoring."""
        self.monitoring_enabled = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Cache monitoring stopped")
    
    def add_alert_handler(self, handler: Callable[[CacheAlert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    def record_operation_timing(self, level: str, operation: str, duration_ms: float):
        """Record operation timing for performance analysis."""
        key = f"{level}_{operation}"
        self.operation_timings[key].append(duration_ms)
        self.throughput_counters[key] += 1
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                await self._collect_performance_snapshots()
                await self._check_alert_conditions()
                await self._analyze_performance()
                
                await asyncio.sleep(self.snapshot_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache monitoring error: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    async def _collect_performance_snapshots(self):
        """Collect performance snapshots from all cache levels."""
        current_time = time.time()
        
        try:
            stats = self.cache.get_comprehensive_stats()
            
            for level_name, level_stats in stats["levels"].items():
                # Calculate throughput
                throughput = self._calculate_throughput(level_name)
                
                # Calculate average response time
                avg_response_time = self._calculate_avg_response_time(level_name)
                
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.now(),
                    level=level_name,
                    hit_rate=level_stats["hit_rate"],
                    miss_rate=1.0 - level_stats["hit_rate"],
                    eviction_rate=self._calculate_eviction_rate(level_name, level_stats),
                    error_rate=level_stats["error_rate"],
                    memory_usage_mb=level_stats["memory_usage"] / (1024 * 1024),
                    avg_response_time_ms=avg_response_time,
                    throughput_ops_per_sec=throughput,
                    compression_ratio=level_stats.get("compression_ratio", 0.0),
                    active_keys=stats.get("l1_cache_info", {}).get("size", 0) if level_name == "l1_memory" else 0
                )
                
                self.analytics.add_performance_snapshot(snapshot)
                self.last_snapshot_time[level_name] = current_time
                
        except Exception as e:
            logger.error(f"Error collecting performance snapshots: {e}")
    
    def _calculate_throughput(self, level: str) -> float:
        """Calculate operations per second for a cache level."""
        current_time = time.time()
        time_delta = current_time - self.last_throughput_reset
        
        if time_delta < 1.0:  # Avoid division by zero
            return 0.0
        
        total_ops = 0
        for key, count in self.throughput_counters.items():
            if key.startswith(level):
                total_ops += count
        
        # Reset counters periodically
        if time_delta >= 60:  # Reset every minute
            self.throughput_counters.clear()
            self.last_throughput_reset = current_time
        
        return total_ops / time_delta
    
    def _calculate_avg_response_time(self, level: str) -> float:
        """Calculate average response time for a cache level."""
        all_times = []
        for key, times in self.operation_timings.items():
            if key.startswith(level):
                all_times.extend(times)
        
        return statistics.mean(all_times) if all_times else 0.0
    
    def _calculate_eviction_rate(self, level: str, level_stats: Dict[str, Any]) -> float:
        """Calculate eviction rate per minute."""
        # This would need to track evictions over time
        # For now, return the total evictions (simplified)
        return level_stats.get("evictions", 0) / 60.0  # Rough estimate
    
    async def _check_alert_conditions(self):
        """Check for alert conditions and trigger alerts."""
        try:
            stats = self.cache.get_comprehensive_stats()
            
            for level_name, level_stats in stats["levels"].items():
                await self._check_level_alerts(level_name, level_stats)
                
        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")
    
    async def _check_level_alerts(self, level: str, stats: Dict[str, Any]):
        """Check alert conditions for a specific cache level."""
        current_time = datetime.now()
        
        # Hit rate alerts
        hit_rate = stats["hit_rate"]
        if hit_rate < self.alert_thresholds["hit_rate_critical"]:
            await self._trigger_alert(
                AlertSeverity.CRITICAL,
                MetricType.HIT_RATE,
                f"Critical hit rate for {level}: {hit_rate:.2%}",
                hit_rate,
                self.alert_thresholds["hit_rate_critical"],
                level,
                "Consider cache warming, increasing cache size, or reviewing access patterns"
            )
        elif hit_rate < self.alert_thresholds["hit_rate_low"]:
            await self._trigger_alert(
                AlertSeverity.HIGH,
                MetricType.HIT_RATE,
                f"Low hit rate for {level}: {hit_rate:.2%}",
                hit_rate,
                self.alert_thresholds["hit_rate_low"],
                level,
                "Review cache configuration and access patterns"
            )
        
        # Error rate alerts
        error_rate = stats["error_rate"]
        if error_rate > self.alert_thresholds["error_rate_critical"]:
            await self._trigger_alert(
                AlertSeverity.CRITICAL,
                MetricType.ERROR_RATE,
                f"Critical error rate for {level}: {error_rate:.2%}",
                error_rate,
                self.alert_thresholds["error_rate_critical"],
                level,
                "Check cache backend connectivity and resource availability"
            )
        elif error_rate > self.alert_thresholds["error_rate_high"]:
            await self._trigger_alert(
                AlertSeverity.HIGH,
                MetricType.ERROR_RATE,
                f"High error rate for {level}: {error_rate:.2%}",
                error_rate,
                self.alert_thresholds["error_rate_high"],
                level,
                "Monitor cache backend health"
            )
        
        # Memory usage alerts (for L1 cache)
        if level == "l1_memory" and "l1_cache_info" in self.cache.get_comprehensive_stats():
            l1_info = self.cache.get_comprehensive_stats()["l1_cache_info"]
            memory_utilization = l1_info.get("memory_utilization", 0)
            
            if memory_utilization > self.alert_thresholds["memory_usage_critical"]:
                await self._trigger_alert(
                    AlertSeverity.CRITICAL,
                    MetricType.MEMORY_USAGE,
                    f"Critical memory usage for {level}: {memory_utilization:.1%}",
                    memory_utilization,
                    self.alert_thresholds["memory_usage_critical"],
                    level,
                    "Increase memory limit or optimize eviction policy"
                )
            elif memory_utilization > self.alert_thresholds["memory_usage_high"]:
                await self._trigger_alert(
                    AlertSeverity.MEDIUM,
                    MetricType.MEMORY_USAGE,
                    f"High memory usage for {level}: {memory_utilization:.1%}",
                    memory_utilization,
                    self.alert_thresholds["memory_usage_high"],
                    level,
                    "Monitor memory growth and consider optimization"
                )
    
    async def _trigger_alert(
        self,
        severity: AlertSeverity,
        metric: MetricType,
        message: str,
        current_value: float,
        threshold: float,
        level: str,
        suggested_action: str = ""
    ):
        """Trigger a cache alert."""
        alert = CacheAlert(
            timestamp=datetime.now(),
            severity=severity,
            metric=metric,
            message=message,
            current_value=current_value,
            threshold=threshold,
            level=level,
            suggested_action=suggested_action
        )
        
        self.alerts.append(alert)
        logger.warning(f"Cache alert ({severity.value}): {message}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert) if asyncio.iscoroutinefunction(handler) else handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    async def _analyze_performance(self):
        """Perform periodic performance analysis."""
        try:
            for level in ["l1_memory", "l2_redis", "l3_persistent", "global"]:
                # Detect anomalies
                anomalies = self.analytics.detect_anomalies(level, MetricType.HIT_RATE)
                for anomaly in anomalies:
                    if anomaly["severity"] == "high":
                        await self._trigger_alert(
                            AlertSeverity.HIGH,
                            MetricType.HIT_RATE,
                            f"Performance anomaly detected in {level}: hit rate {anomaly['value']:.2%}",
                            anomaly["value"],
                            anomaly["expected_range"][0],
                            level,
                            "Investigate recent changes or system load"
                        )
                
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        try:
            current_stats = self.cache.get_comprehensive_stats()
            recent_alerts = list(self.alerts)[-10:]  # Last 10 alerts
            
            dashboard = {
                "current_status": current_stats,
                "recent_alerts": [alert.to_dict() for alert in recent_alerts],
                "performance_trends": {},
                "predictions": {},
                "recommendations": self._generate_recommendations()
            }
            
            # Add trends for each level
            for level in ["l1_memory", "l2_redis", "l3_persistent", "global"]:
                try:
                    trends = self.analytics.analyze_trends(level, hours=24)
                    if "error" not in trends:
                        dashboard["performance_trends"][level] = trends
                except Exception as e:
                    logger.debug(f"Could not analyze trends for {level}: {e}")
                
                try:
                    predictions = self.analytics.predict_performance(level, hours_ahead=4)
                    if "error" not in predictions:
                        dashboard["predictions"][level] = predictions
                except Exception as e:
                    logger.debug(f"Could not generate predictions for {level}: {e}")
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating monitoring dashboard: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        try:
            stats = self.cache.get_comprehensive_stats()
            
            # Analyze L1 cache
            l1_info = stats.get("l1_cache_info", {})
            l1_utilization = l1_info.get("memory_utilization", 0)
            l1_hit_rate = stats["levels"].get("l1_memory", {}).get("hit_rate", 0)
            
            if l1_hit_rate < 0.8 and l1_utilization < 0.5:
                recommendations.append({
                    "type": "optimization",
                    "priority": "medium",
                    "title": "Increase L1 cache size",
                    "description": f"L1 hit rate is {l1_hit_rate:.1%} but memory utilization is only {l1_utilization:.1%}. Consider increasing L1 cache size.",
                    "action": "Increase max_l1_size parameter"
                })
            
            if l1_utilization > 0.9:
                recommendations.append({
                    "type": "performance",
                    "priority": "high",
                    "title": "L1 cache memory pressure",
                    "description": f"L1 cache memory utilization is {l1_utilization:.1%}. This may cause frequent evictions.",
                    "action": "Increase max_l1_memory_mb or optimize eviction policy"
                })
            
            # Analyze error rates
            for level, level_stats in stats["levels"].items():
                error_rate = level_stats.get("error_rate", 0)
                if error_rate > 0.02:  # > 2%
                    recommendations.append({
                        "type": "reliability",
                        "priority": "high",
                        "title": f"High error rate in {level}",
                        "description": f"Error rate in {level} is {error_rate:.2%}. This indicates connectivity or resource issues.",
                        "action": f"Check {level} backend health and connectivity"
                    })
            
            # Analyze access patterns
            hot_keys = stats.get("access_patterns", {}).get("hot_keys", [])
            if len(hot_keys) > 50:
                recommendations.append({
                    "type": "optimization",
                    "priority": "medium",
                    "title": "Many hot keys detected",
                    "description": f"Detected {len(hot_keys)} frequently accessed keys. Consider cache warming.",
                    "action": "Implement proactive cache warming for hot keys"
                })
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append({
                "type": "system",
                "priority": "low",
                "title": "Monitoring system issue",
                "description": f"Could not analyze cache performance: {e}",
                "action": "Check monitoring system health"
            })
        
        return recommendations
    
    async def export_performance_report(self, hours: int = 24, format: str = "json") -> Dict[str, Any]:
        """Export comprehensive performance report."""
        try:
            report = {
                "report_generated": datetime.now().isoformat(),
                "time_range_hours": hours,
                "summary": {},
                "detailed_analysis": {},
                "alerts": [alert.to_dict() for alert in list(self.alerts)[-100:]],  # Last 100 alerts
                "recommendations": self._generate_recommendations()
            }
            
            # Generate summary for each level
            for level in ["l1_memory", "l2_redis", "l3_persistent", "global"]:
                try:
                    trends = self.analytics.analyze_trends(level, hours=hours)
                    if "error" not in trends:
                        report["summary"][level] = {
                            "current_hit_rate": trends["current_metrics"]["hit_rate"],
                            "avg_response_time_ms": trends["current_metrics"]["avg_response_time_ms"],
                            "hit_rate_trend": trends["hit_rate_trend"]["interpretation"],
                            "performance_stability": trends["statistics"]["hit_rate"]["std_dev"]
                        }
                        
                        report["detailed_analysis"][level] = trends
                        
                except Exception as e:
                    logger.debug(f"Could not analyze {level} for report: {e}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}


# Example alert handlers
async def log_alert_handler(alert: CacheAlert):
    """Log alert to file."""
    logger.warning(f"CACHE ALERT [{alert.severity.value.upper()}] {alert.message}")


async def email_alert_handler(alert: CacheAlert):
    """Send email alert (placeholder implementation)."""
    if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
        # This would integrate with actual email service
        logger.info(f"Would send email alert: {alert.message}")


async def webhook_alert_handler(alert: CacheAlert):
    """Send webhook alert (placeholder implementation)."""
    if alert.severity == AlertSeverity.CRITICAL:
        # This would integrate with webhook service (Slack, Discord, etc.)
        logger.info(f"Would send webhook alert: {alert.message}")