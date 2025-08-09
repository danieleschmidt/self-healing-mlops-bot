"""Enhanced health monitoring system for the self-healing bot."""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import deque, defaultdict
from statistics import mean, median
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ServiceDependency:
    """Represents a service dependency."""
    name: str
    endpoint: Optional[str] = None
    health_check_function: Optional[Callable] = None
    timeout: int = 10
    critical: bool = False
    retry_attempts: int = 2
    expected_response_codes: Set[int] = field(default_factory=lambda: {200, 201, 204})
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Enhanced health check configuration."""
    name: str
    check_function: Callable
    interval: int = 60  # Check interval in seconds
    timeout: int = 10   # Timeout for check in seconds
    critical: bool = False  # If True, failure affects overall health
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)
    
    # Advanced configuration
    retry_attempts: int = 1
    degraded_threshold: float = 0.8  # Response time threshold for degraded status
    unhealthy_threshold: float = 2.0  # Response time threshold for unhealthy status
    failure_threshold: int = 3  # Consecutive failures before marking unhealthy
    recovery_threshold: int = 2  # Consecutive successes after failure to mark healthy
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Other checks this depends on
    
    # Runtime state
    last_check: float = 0
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=10))
    historical_data: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class PerformanceMetrics:
    """Performance metrics for health monitoring."""
    average_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    success_rate: float = 100.0
    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    last_updated: float = 0.0


@dataclass
class Alert:
    """Health alert configuration."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    check_name: str
    condition: str  # The condition that triggered the alert
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    acknowledgements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Comprehensive health check report."""
    overall_status: HealthStatus
    health_score: float  # 0-100 score
    timestamp: float
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)
    performance_metrics: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    service_dependencies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class HealthMonitor:
    """Enhanced health monitor with comprehensive system monitoring."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.checks: Dict[str, HealthCheck] = {}
        self.service_dependencies: Dict[str, ServiceDependency] = {}
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.failure_threshold = self.config.get('failure_threshold', 3)
        self.alert_cooldown = self.config.get('alert_cooldown', 300)  # 5 minutes
        
        # Enhanced features
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.historical_reports: deque = deque(maxlen=100)
        self.notification_handlers: List[Callable] = []
        
        # Threading for sync health checks
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Health score calculation
        self.health_score_weights = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.7,
            HealthStatus.UNHEALTHY: 0.3,
            HealthStatus.CRITICAL: 0.0,
            HealthStatus.UNKNOWN: 0.5
        }
        
        # Register built-in checks
        self._register_builtin_checks()
    
    def register_check(self, health_check: HealthCheck):
        """Register a health check."""
        self.checks[health_check.name] = health_check
        self.performance_metrics[health_check.name] = PerformanceMetrics()
        logger.info(f"Registered health check: {health_check.name}")
    
    def add_check(
        self,
        name: str,
        check_function: Callable,
        interval: int = 60,
        timeout: int = 10,
        critical: bool = False,
        enabled: bool = True,
        tags: Optional[Set[str]] = None,
        **kwargs
    ):
        """Add a health check with parameters."""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            critical=critical,
            enabled=enabled,
            tags=tags or set(),
            **kwargs
        )
        self.register_check(health_check)
    
    def register_service_dependency(self, dependency: ServiceDependency):
        """Register a service dependency."""
        self.service_dependencies[dependency.name] = dependency
        
        # Create health check for this dependency if it has an endpoint
        if dependency.endpoint and not dependency.health_check_function:
            async def dependency_check():
                return await self._check_service_endpoint(dependency)
            
            self.add_check(
                name=f"service_{dependency.name}",
                check_function=dependency_check,
                critical=dependency.critical,
                timeout=dependency.timeout,
                tags={"service_dependency"}
            )
        
        logger.info(f"Registered service dependency: {dependency.name}")
    
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    async def _check_service_endpoint(self, dependency: ServiceDependency) -> Dict[str, Any]:
        """Check a service endpoint health."""
        try:
            import aiohttp
            
            timeout = aiohttp.ClientTimeout(total=dependency.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    dependency.endpoint,
                    headers=dependency.headers
                ) as response:
                    response_time = time.time()
                    
                    if response.status in dependency.expected_response_codes:
                        status = HealthStatus.HEALTHY
                        message = f"Service {dependency.name} is healthy"
                        
                        # Check response time for degradation
                        if hasattr(response, 'response_time'):
                            if response.response_time > 2.0:
                                status = HealthStatus.DEGRADED
                                message = f"Service {dependency.name} is slow (response time: {response.response_time:.2f}s)"
                    else:
                        status = HealthStatus.UNHEALTHY
                        message = f"Service {dependency.name} returned status {response.status}"
                    
                    return {
                        "status": status.value,
                        "message": message,
                        "data": {
                            "status_code": response.status,
                            "response_time": getattr(response, 'response_time', 0),
                            "endpoint": dependency.endpoint
                        }
                    }
                    
        except asyncio.TimeoutError:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Service {dependency.name} timed out",
                "data": {"timeout": dependency.timeout}
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Service {dependency.name} check failed: {str(e)}",
                "data": {"error": str(e)}
            }
    
    async def run_check(self, check: HealthCheck) -> Dict[str, Any]:
        """Run a single health check with enhanced error handling and metrics."""
        if not check.enabled:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "Check disabled",
                "timestamp": time.time(),
                "duration": 0
            }
        
        start_time = time.time()
        result = None
        
        for attempt in range(check.retry_attempts + 1):
            try:
                # Run the check with timeout
                if asyncio.iscoroutinefunction(check.check_function):
                    result = await asyncio.wait_for(
                        check.check_function(),
                        timeout=check.timeout
                    )
                else:
                    # Run sync function in thread pool
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        check.check_function
                    )
                
                break  # Success, exit retry loop
                
            except asyncio.TimeoutError:
                if attempt < check.retry_attempts:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Progressive backoff
                    continue
                else:
                    result = None
                    break
            except Exception as e:
                if attempt < check.retry_attempts:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    logger.exception(f"Health check {check.name} failed after retries: {e}")
                    result = None
                    break
        
        duration = time.time() - start_time
        current_time = time.time()
        
        # Process result
        if result is None:
            # Timeout or final failure
            check.consecutive_failures += 1
            check.consecutive_successes = 0
            check.last_status = HealthStatus.UNHEALTHY
            check.last_error = "Check failed or timed out"
            
            status = HealthStatus.CRITICAL if check.consecutive_failures >= check.failure_threshold else HealthStatus.UNHEALTHY
            
            check_result = {
                "status": status.value,
                "message": f"Check failed after {check.retry_attempts + 1} attempts",
                "timestamp": current_time,
                "duration": duration,
                "error": check.last_error,
                "consecutive_failures": check.consecutive_failures,
                "retry_attempts": check.retry_attempts + 1
            }
            
        else:
            # Success - process result
            check.consecutive_failures = 0
            check.consecutive_successes += 1
            check.last_error = None
            
            # Handle different result types
            if isinstance(result, dict):
                status_str = result.get("status", HealthStatus.HEALTHY.value)
                message = result.get("message", "OK")
                data = result.get("data", {})
            elif isinstance(result, bool):
                status_str = HealthStatus.HEALTHY.value if result else HealthStatus.UNHEALTHY.value
                message = "OK" if result else "Check failed"
                data = {}
            else:
                status_str = HealthStatus.HEALTHY.value
                message = str(result) if result else "OK"
                data = {}
            
            # Determine status based on response time if healthy
            status = HealthStatus(status_str)
            if status == HealthStatus.HEALTHY:
                if duration >= check.unhealthy_threshold:
                    status = HealthStatus.UNHEALTHY
                    message = f"{message} (slow response: {duration:.2f}s)"
                elif duration >= check.degraded_threshold:
                    status = HealthStatus.DEGRADED
                    message = f"{message} (degraded response: {duration:.2f}s)"
            
            check.last_status = status
            
            check_result = {
                "status": status.value,
                "message": message,
                "timestamp": current_time,
                "duration": duration,
                "data": data,
                "consecutive_successes": check.consecutive_successes
            }
        
        # Update metrics
        self._update_check_metrics(check, check_result)
        
        # Store in history
        check.historical_data.append({
            "timestamp": current_time,
            "status": check_result["status"],
            "duration": duration,
            "message": check_result["message"]
        })
        
        # Check for alerts
        await self._check_for_alerts(check, check_result)
        
        return check_result
    
    def _update_check_metrics(self, check: HealthCheck, result: Dict[str, Any]):
        """Update performance metrics for a check."""
        metrics = self.performance_metrics[check.name]
        
        metrics.total_checks += 1
        metrics.last_updated = time.time()
        
        duration = result.get('duration', 0)
        check.response_times.append(duration)
        
        if result["status"] in [HealthStatus.HEALTHY.value, HealthStatus.DEGRADED.value]:
            metrics.successful_checks += 1
        else:
            metrics.failed_checks += 1
        
        # Calculate response time metrics
        if check.response_times:
            response_times = list(check.response_times)
            metrics.average_response_time = mean(response_times)
            
            if len(response_times) >= 2:
                sorted_times = sorted(response_times)
                p95_index = int(0.95 * len(sorted_times))
                p99_index = int(0.99 * len(sorted_times))
                metrics.p95_response_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
                metrics.p99_response_time = sorted_times[min(p99_index, len(sorted_times) - 1)]
        
        # Calculate success rate
        if metrics.total_checks > 0:
            metrics.success_rate = (metrics.successful_checks / metrics.total_checks) * 100
    
    async def _check_for_alerts(self, check: HealthCheck, result: Dict[str, Any]):
        """Check if alerts should be generated."""
        status = HealthStatus(result["status"])
        
        # Critical status alert
        if status == HealthStatus.CRITICAL:
            await self._generate_alert(
                check_name=check.name,
                severity=AlertSeverity.CRITICAL,
                condition="Check status is CRITICAL",
                message=f"Health check '{check.name}' is in critical state: {result['message']}",
                metadata=result
            )
        
        # Consecutive failures alert
        elif check.consecutive_failures >= check.failure_threshold:
            await self._generate_alert(
                check_name=check.name,
                severity=AlertSeverity.ERROR if check.critical else AlertSeverity.WARNING,
                condition=f"Consecutive failures >= {check.failure_threshold}",
                message=f"Health check '{check.name}' has failed {check.consecutive_failures} times consecutively",
                metadata=result
            )
        
        # High response time alert
        elif result.get('duration', 0) > check.unhealthy_threshold:
            await self._generate_alert(
                check_name=check.name,
                severity=AlertSeverity.WARNING,
                condition="High response time",
                message=f"Health check '{check.name}' response time ({result['duration']:.2f}s) exceeds threshold ({check.unhealthy_threshold}s)",
                metadata=result
            )
        
        # Check if previous alerts should be resolved
        await self._check_alert_resolution(check, status)
    
    async def _generate_alert(
        self,
        check_name: str,
        severity: AlertSeverity,
        condition: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Generate a new alert."""
        alert_id = f"{check_name}_{condition}_{int(time.time())}"
        
        # Check if similar alert already exists and is unresolved
        existing_alerts = [
            alert for alert in self.alerts.values()
            if alert.check_name == check_name 
            and alert.condition == condition 
            and not alert.resolved
        ]
        
        if existing_alerts:
            # Update existing alert timestamp instead of creating new one
            for alert in existing_alerts:
                alert.timestamp = time.time()
                alert.metadata.update(metadata or {})
            return
        
        alert = Alert(
            id=alert_id,
            name=f"{check_name}_{severity.value}",
            description=message,
            severity=severity,
            check_name=check_name,
            condition=condition,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Generated alert: {alert.name} - {alert.description}")
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")
    
    async def _check_alert_resolution(self, check: HealthCheck, current_status: HealthStatus):
        """Check if any alerts for this check should be resolved."""
        unresolved_alerts = [
            alert for alert in self.alerts.values()
            if alert.check_name == check.name and not alert.resolved
        ]
        
        for alert in unresolved_alerts:
            should_resolve = False
            
            # Resolve critical/failure alerts if check is healthy for recovery_threshold times
            if current_status == HealthStatus.HEALTHY:
                if check.consecutive_successes >= check.recovery_threshold:
                    should_resolve = True
            
            if should_resolve:
                alert.resolved = True
                alert.resolved_at = time.time()
                
                logger.info(f"Resolved alert: {alert.name}")
                
                # Send resolution notifications
                for handler in self.notification_handlers:
                    try:
                        await handler(alert)
                    except Exception as e:
                        logger.error(f"Failed to send alert resolution notification: {e}")
    
    async def run_all_checks(self) -> HealthReport:
        """Run all enabled health checks and generate comprehensive report."""
        start_time = time.time()
        current_time = time.time()
        check_results = {}
        
        # Run checks that need to be run based on their intervals
        tasks = []
        checks_to_run = []
        
        for check_name, check in self.checks.items():
            if check.enabled and (current_time - check.last_check) >= check.interval:
                # Check dependencies first
                if not await self._check_dependencies(check):
                    check_results[check_name] = {
                        "status": HealthStatus.UNKNOWN.value,
                        "message": "Dependencies not met",
                        "timestamp": current_time,
                        "duration": 0,
                        "dependency_failed": True
                    }
                    continue
                
                tasks.append(self.run_check(check))
                checks_to_run.append(check_name)
                check.last_check = current_time
        
        # Execute checks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                check_name = checks_to_run[i]
                if isinstance(result, Exception):
                    check_results[check_name] = {
                        "status": HealthStatus.UNHEALTHY.value,
                        "message": f"Check execution failed: {result}",
                        "timestamp": current_time,
                        "error": str(result)
                    }
                else:
                    check_results[check_name] = result
        
        # Include cached results from checks that didn't need to run
        for check_name, check in self.checks.items():
            if check_name not in check_results and check.last_status != HealthStatus.UNKNOWN:
                check_results[check_name] = {
                    "status": check.last_status.value,
                    "message": "Cached result",
                    "timestamp": check.last_check,
                    "cached": True,
                    "consecutive_failures": check.consecutive_failures,
                    "consecutive_successes": check.consecutive_successes
                }
        
        # Check service dependencies
        service_dependency_results = await self._check_service_dependencies()
        
        # Calculate overall health and health score
        overall_status, health_score = self._calculate_overall_health(check_results, service_dependency_results)
        
        # Generate summary and recommendations
        summary = self._generate_summary(check_results)
        recommendations = self._generate_recommendations(check_results, service_dependency_results)
        
        # Get current alerts
        current_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        
        report = HealthReport(
            overall_status=overall_status,
            health_score=health_score,
            timestamp=current_time,
            checks=check_results,
            summary=summary,
            alerts=current_alerts,
            performance_metrics={name: asdict(metrics) for name, metrics in self.performance_metrics.items()},
            service_dependencies=service_dependency_results,
            recommendations=recommendations
        )
        
        # Store in history
        self.historical_reports.append(report)
        
        return report
    
    async def _check_dependencies(self, check: HealthCheck) -> bool:
        """Check if all dependencies for a health check are satisfied."""
        for dep_name in check.dependencies:
            if dep_name in self.checks:
                dep_check = self.checks[dep_name]
                if dep_check.last_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    return False
            elif dep_name in self.service_dependencies:
                # Check service dependency (would need implementation)
                pass
        return True
    
    async def _check_service_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Check all registered service dependencies."""
        results = {}
        
        for name, dependency in self.service_dependencies.items():
            if dependency.health_check_function:
                try:
                    result = await dependency.health_check_function()
                    results[name] = result if isinstance(result, dict) else {"status": "healthy", "message": str(result)}
                except Exception as e:
                    results[name] = {
                        "status": "unhealthy",
                        "message": f"Dependency check failed: {str(e)}",
                        "error": str(e)
                    }
            elif dependency.endpoint:
                results[name] = await self._check_service_endpoint(dependency)
            else:
                results[name] = {
                    "status": "unknown",
                    "message": "No health check method configured"
                }
        
        return results
    
    def _calculate_overall_health(
        self, 
        check_results: Dict[str, Dict[str, Any]], 
        service_results: Dict[str, Dict[str, Any]]
    ) -> tuple[HealthStatus, float]:
        """Calculate overall system health status and score."""
        if not check_results and not service_results:
            return HealthStatus.UNKNOWN, 0.0
        
        # Collect all statuses with their weights
        statuses = []
        weights = []
        
        # Process health checks
        for check_name, result in check_results.items():
            check = self.checks.get(check_name)
            status = HealthStatus(result.get("status", HealthStatus.UNKNOWN.value))
            weight = 2.0 if (check and check.critical) else 1.0
            
            statuses.append(status)
            weights.append(weight)
        
        # Process service dependencies
        for service_name, result in service_results.items():
            dependency = self.service_dependencies.get(service_name)
            status_str = result.get("status", "unknown")
            
            # Map service status to health status
            status_mapping = {
                "healthy": HealthStatus.HEALTHY,
                "degraded": HealthStatus.DEGRADED,
                "unhealthy": HealthStatus.UNHEALTHY,
                "critical": HealthStatus.CRITICAL,
                "unknown": HealthStatus.UNKNOWN
            }
            status = status_mapping.get(status_str, HealthStatus.UNKNOWN)
            weight = 2.0 if (dependency and dependency.critical) else 1.0
            
            statuses.append(status)
            weights.append(weight)
        
        if not statuses:
            return HealthStatus.UNKNOWN, 0.0
        
        # Calculate weighted health score
        total_score = 0.0
        total_weight = 0.0
        
        for status, weight in zip(statuses, weights):
            score = self.health_score_weights.get(status, 0.5)
            total_score += score * weight
            total_weight += weight
        
        health_score = (total_score / total_weight) * 100 if total_weight > 0 else 0.0
        
        # Determine overall status
        critical_count = sum(1 for s in statuses if s == HealthStatus.CRITICAL)
        unhealthy_count = sum(1 for s in statuses if s == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for s in statuses if s == HealthStatus.DEGRADED)
        
        # Check for critical failures
        critical_failures = any(
            status == HealthStatus.CRITICAL and (
                check_name not in self.checks or self.checks[check_name].critical
            )
            for check_name, result in check_results.items()
            for status in [HealthStatus(result.get("status", HealthStatus.UNKNOWN.value))]
        )
        
        if critical_failures or critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        elif health_score >= 95:
            overall_status = HealthStatus.HEALTHY
        elif health_score >= 80:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNHEALTHY
        
        return overall_status, health_score
    
    def _generate_summary(self, check_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive health summary statistics."""
        total_checks = len(check_results)
        status_counts = defaultdict(int)
        total_duration = 0.0
        successful_checks = 0
        
        for result in check_results.values():
            status = result.get("status", HealthStatus.UNKNOWN.value)
            status_counts[status] += 1
            total_duration += result.get("duration", 0)
            
            if status in [HealthStatus.HEALTHY.value, HealthStatus.DEGRADED.value]:
                successful_checks += 1
        
        avg_response_time = total_duration / total_checks if total_checks > 0 else 0.0
        success_rate = (successful_checks / total_checks * 100) if total_checks > 0 else 0.0
        
        return {
            "total_checks": total_checks,
            "status_counts": dict(status_counts),
            "success_rate": round(success_rate, 2),
            "average_response_time": round(avg_response_time, 3),
            "critical_alerts": len([a for a in self.alerts.values() if not a.resolved and a.severity == AlertSeverity.CRITICAL]),
            "total_active_alerts": len([a for a in self.alerts.values() if not a.resolved]),
            "uptime_percentage": success_rate  # Simplified uptime calculation
        }
    
    def _generate_recommendations(
        self, 
        check_results: Dict[str, Dict[str, Any]], 
        service_results: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations based on health status."""
        recommendations = []
        
        # Analyze failing checks
        failing_checks = [
            name for name, result in check_results.items()
            if result.get("status") in [HealthStatus.UNHEALTHY.value, HealthStatus.CRITICAL.value]
        ]
        
        if failing_checks:
            recommendations.append(f"Investigate failing health checks: {', '.join(failing_checks)}")
        
        # Analyze slow checks
        slow_checks = [
            name for name, result in check_results.items()
            if result.get("duration", 0) > 5.0  # 5 second threshold
        ]
        
        if slow_checks:
            recommendations.append(f"Optimize performance for slow checks: {', '.join(slow_checks)}")
        
        # Check for too many consecutive failures
        high_failure_checks = [
            name for name, check in self.checks.items()
            if check.consecutive_failures >= 5
        ]
        
        if high_failure_checks:
            recommendations.append(f"Review checks with persistent failures: {', '.join(high_failure_checks)}")
        
        # Service dependency recommendations
        failing_services = [
            name for name, result in service_results.items()
            if result.get("status") in ["unhealthy", "critical"]
        ]
        
        if failing_services:
            recommendations.append(f"Check service dependencies: {', '.join(failing_services)}")
        
        # Alert recommendations
        unresolved_critical = [
            alert.check_name for alert in self.alerts.values()
            if not alert.resolved and alert.severity == AlertSeverity.CRITICAL
        ]
        
        if unresolved_critical:
            recommendations.append(f"Address critical alerts immediately: {', '.join(set(unresolved_critical))}")
        
        return recommendations
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring:
            logger.warning("Health monitoring is already running")
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Enhanced health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop with enhanced error handling."""
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while self.monitoring:
            try:
                report = await self.run_all_checks()
                consecutive_failures = 0
                
                # Log significant status changes
                if hasattr(self, '_last_overall_status'):
                    if self._last_overall_status != report.overall_status:
                        logger.info(
                            f"Overall health status changed: "
                            f"{self._last_overall_status.value} -> {report.overall_status.value} "
                            f"(Score: {report.health_score:.1f})"
                        )
                
                self._last_overall_status = report.overall_status
                
                # Log health score periodically
                if len(self.historical_reports) % 10 == 0:  # Every 10 reports
                    logger.info(
                        f"Health Status: {report.overall_status.value} "
                        f"(Score: {report.health_score:.1f}, "
                        f"Active Alerts: {len([a for a in report.alerts if not a.resolved])})"
                    )
                
                # Wait before next check cycle
                await asyncio.sleep(10)  # Check every 10 seconds for due checks
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_failures += 1
                logger.exception(f"Error in health monitoring loop: {e}")
                
                # If too many consecutive failures, increase sleep time
                sleep_time = min(30 + (consecutive_failures * 10), 300)  # Max 5 minutes
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(
                        f"Health monitoring has failed {consecutive_failures} times consecutively. "
                        f"Sleeping for {sleep_time} seconds."
                    )
                
                await asyncio.sleep(sleep_time)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health monitoring status with comprehensive information."""
        current_time = time.time()
        
        return {
            "monitoring": self.monitoring,
            "uptime": current_time - getattr(self, '_start_time', current_time),
            "total_checks": len(self.checks),
            "enabled_checks": sum(1 for check in self.checks.values() if check.enabled),
            "total_dependencies": len(self.service_dependencies),
            "total_alerts": len(self.alerts),
            "unresolved_alerts": len([a for a in self.alerts.values() if not a.resolved]),
            "critical_alerts": len([a for a in self.alerts.values() if not a.resolved and a.severity == AlertSeverity.CRITICAL]),
            "checks": {
                name: {
                    "enabled": check.enabled,
                    "interval": check.interval,
                    "critical": check.critical,
                    "last_status": check.last_status.value,
                    "last_check": check.last_check,
                    "consecutive_failures": check.consecutive_failures,
                    "consecutive_successes": check.consecutive_successes,
                    "tags": list(check.tags)
                }
                for name, check in self.checks.items()
            },
            "performance_metrics": {
                name: asdict(metrics) for name, metrics in self.performance_metrics.items()
            }
        }
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            asdict(report) for report in self.historical_reports
            if report.timestamp >= cutoff_time
        ]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledgements.append({
                "acknowledged_by": acknowledged_by,
                "timestamp": time.time()
            })
            return True
        return False
    
    def _register_builtin_checks(self):
        """Register built-in system health checks."""
        
        # Memory usage check
        async def memory_check():
            try:
                import psutil
                memory = psutil.virtual_memory()
                
                if memory.percent > 95:
                    return {
                        "status": HealthStatus.CRITICAL.value,
                        "message": f"Critical memory usage: {memory.percent:.1f}%",
                        "data": {"percent": memory.percent, "available_gb": memory.available / (1024**3)}
                    }
                elif memory.percent > 85:
                    return {
                        "status": HealthStatus.UNHEALTHY.value,
                        "message": f"High memory usage: {memory.percent:.1f}%",
                        "data": {"percent": memory.percent, "available_gb": memory.available / (1024**3)}
                    }
                elif memory.percent > 75:
                    return {
                        "status": HealthStatus.DEGRADED.value,
                        "message": f"Moderate memory usage: {memory.percent:.1f}%",
                        "data": {"percent": memory.percent, "available_gb": memory.available / (1024**3)}
                    }
                else:
                    return {
                        "status": HealthStatus.HEALTHY.value,
                        "message": f"Memory usage: {memory.percent:.1f}%",
                        "data": {"percent": memory.percent, "available_gb": memory.available / (1024**3)}
                    }
            except ImportError:
                return {
                    "status": HealthStatus.UNKNOWN.value,
                    "message": "psutil not available for memory monitoring"
                }
        
        self.add_check(
            name="system_memory",
            check_function=memory_check,
            interval=30,
            critical=True,
            tags={"system", "memory"}
        )
        
        # Disk usage check
        async def disk_check():
            try:
                import psutil
                disk = psutil.disk_usage('/')
                percent_used = (disk.used / disk.total) * 100
                
                if percent_used > 95:
                    return {
                        "status": HealthStatus.CRITICAL.value,
                        "message": f"Critical disk usage: {percent_used:.1f}%",
                        "data": {"percent": percent_used, "free_gb": disk.free / (1024**3)}
                    }
                elif percent_used > 90:
                    return {
                        "status": HealthStatus.UNHEALTHY.value,
                        "message": f"High disk usage: {percent_used:.1f}%",
                        "data": {"percent": percent_used, "free_gb": disk.free / (1024**3)}
                    }
                elif percent_used > 80:
                    return {
                        "status": HealthStatus.DEGRADED.value,
                        "message": f"Moderate disk usage: {percent_used:.1f}%",
                        "data": {"percent": percent_used, "free_gb": disk.free / (1024**3)}
                    }
                else:
                    return {
                        "status": HealthStatus.HEALTHY.value,
                        "message": f"Disk usage: {percent_used:.1f}%",
                        "data": {"percent": percent_used, "free_gb": disk.free / (1024**3)}
                    }
            except ImportError:
                return {
                    "status": HealthStatus.UNKNOWN.value,
                    "message": "psutil not available for disk monitoring"
                }
        
        self.add_check(
            name="system_disk",
            check_function=disk_check,
            interval=60,
            critical=True,
            tags={"system", "disk"}
        )
        
        # Basic connectivity check
        def connectivity_check():
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                return {
                    "status": HealthStatus.HEALTHY.value,
                    "message": "Internet connectivity available"
                }
            except OSError:
                return {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": "No internet connectivity"
                }
        
        self.add_check(
            name="internet_connectivity",
            check_function=connectivity_check,
            interval=120,
            critical=False,
            tags={"network", "connectivity"}
        )


# Global health monitor instance
health_monitor = HealthMonitor()


# Notification handlers
async def log_alert_handler(alert: Alert):
    """Log alert notifications."""
    if alert.resolved:
        logger.info(f"RESOLVED: {alert.name} - {alert.description}")
    else:
        log_level = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical
        }
        log_func = log_level.get(alert.severity, logger.info)
        log_func(f"ALERT: {alert.name} - {alert.description}")


# Register default notification handler
health_monitor.add_notification_handler(log_alert_handler)