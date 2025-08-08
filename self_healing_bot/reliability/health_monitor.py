"""Health monitoring system for the self-healing bot."""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    check_function: Callable
    interval: int = 60  # Check interval in seconds
    timeout: int = 10   # Timeout for check in seconds
    critical: bool = False  # If True, failure affects overall health
    enabled: bool = True
    
    # Runtime state
    last_check: float = 0
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_error: Optional[str] = None
    consecutive_failures: int = 0


@dataclass
class HealthReport:
    """Health check report."""
    overall_status: HealthStatus
    timestamp: float
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """Monitor system health with configurable checks."""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.failure_threshold = 3  # Consecutive failures before marking unhealthy
    
    def register_check(self, health_check: HealthCheck):
        """Register a health check."""
        self.checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
    
    def add_check(
        self,
        name: str,
        check_function: Callable,
        interval: int = 60,
        timeout: int = 10,
        critical: bool = False,
        enabled: bool = True
    ):
        """Add a health check with parameters."""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            critical=critical,
            enabled=enabled
        )
        self.register_check(health_check)
    
    async def run_check(self, check: HealthCheck) -> Dict[str, Any]:
        """Run a single health check."""
        if not check.enabled:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "Check disabled",
                "timestamp": time.time(),
                "duration": 0
            }
        
        start_time = time.time()
        
        try:
            # Run the check with timeout
            if asyncio.iscoroutinefunction(check.check_function):
                result = await asyncio.wait_for(
                    check.check_function(),
                    timeout=check.timeout
                )
            else:
                result = check.check_function()
            
            duration = time.time() - start_time
            
            # Reset consecutive failures on success
            check.consecutive_failures = 0
            check.last_status = HealthStatus.HEALTHY
            check.last_error = None
            
            # Handle different result types
            if isinstance(result, dict):
                status = result.get("status", HealthStatus.HEALTHY.value)
                message = result.get("message", "OK")
                data = result.get("data", {})
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY.value if result else HealthStatus.UNHEALTHY.value
                message = "OK" if result else "Check failed"
                data = {}
            else:
                status = HealthStatus.HEALTHY.value
                message = str(result) if result else "OK"
                data = {}
            
            return {
                "status": status,
                "message": message,
                "timestamp": time.time(),
                "duration": duration,
                "data": data,
                "consecutive_failures": check.consecutive_failures
            }
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            check.consecutive_failures += 1
            check.last_status = HealthStatus.UNHEALTHY
            check.last_error = "Timeout"
            
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Check timed out after {check.timeout}s",
                "timestamp": time.time(),
                "duration": duration,
                "error": "timeout",
                "consecutive_failures": check.consecutive_failures
            }
            
        except Exception as e:
            duration = time.time() - start_time
            check.consecutive_failures += 1
            check.last_status = HealthStatus.UNHEALTHY
            check.last_error = str(e)
            
            logger.exception(f"Health check {check.name} failed: {e}")
            
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Check failed: {str(e)}",
                "timestamp": time.time(),
                "duration": duration,
                "error": str(e),
                "consecutive_failures": check.consecutive_failures
            }
    
    async def run_all_checks(self) -> HealthReport:
        """Run all enabled health checks."""
        current_time = time.time()
        check_results = {}
        
        # Run checks that need to be run
        tasks = []
        checks_to_run = []
        
        for check_name, check in self.checks.items():
            if check.enabled and (current_time - check.last_check) >= check.interval:
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
        
        # Include recent results from checks that didn't run
        for check_name, check in self.checks.items():
            if check_name not in check_results and check.last_status != HealthStatus.UNKNOWN:
                # Use cached result if check didn't need to run
                check_results[check_name] = {
                    "status": check.last_status.value,
                    "message": "Cached result",
                    "timestamp": check.last_check,
                    "cached": True
                }
        
        # Calculate overall health
        overall_status = self._calculate_overall_status(check_results)
        
        # Generate summary
        summary = self._generate_summary(check_results)
        
        return HealthReport(
            overall_status=overall_status,
            timestamp=current_time,
            checks=check_results,
            summary=summary
        )
    
    def _calculate_overall_status(self, check_results: Dict[str, Any]) -> HealthStatus:
        """Calculate overall system health status."""
        if not check_results:
            return HealthStatus.UNKNOWN
        
        has_critical_failure = False
        has_non_critical_failure = False
        
        for check_name, result in check_results.items():
            status = result.get("status", HealthStatus.UNKNOWN.value)
            check = self.checks.get(check_name)
            
            if status in [HealthStatus.UNHEALTHY.value, HealthStatus.DEGRADED.value]:
                if check and check.critical:
                    has_critical_failure = True
                else:
                    has_non_critical_failure = True
        
        if has_critical_failure:
            return HealthStatus.UNHEALTHY
        elif has_non_critical_failure:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _generate_summary(self, check_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate health summary statistics."""
        total_checks = len(check_results)
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0
        unknown_count = 0
        
        for result in check_results.values():
            status = result.get("status", HealthStatus.UNKNOWN.value)
            if status == HealthStatus.HEALTHY.value:
                healthy_count += 1
            elif status == HealthStatus.DEGRADED.value:
                degraded_count += 1
            elif status == HealthStatus.UNHEALTHY.value:
                unhealthy_count += 1
            else:
                unknown_count += 1
        
        return {
            "total_checks": total_checks,
            "healthy": healthy_count,
            "degraded": degraded_count,
            "unhealthy": unhealthy_count,
            "unknown": unknown_count,
            "health_percentage": (healthy_count / total_checks * 100) if total_checks > 0 else 0
        }
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring:
            logger.warning("Health monitoring is already running")
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")
    
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
        
        logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                report = await self.run_all_checks()
                
                # Log overall status changes
                if hasattr(self, '_last_overall_status'):
                    if self._last_overall_status != report.overall_status:
                        logger.info(
                            f"Overall health status changed: "
                            f"{self._last_overall_status.value} -> {report.overall_status.value}"
                        )
                
                self._last_overall_status = report.overall_status
                
                # Wait before next check cycle
                await asyncio.sleep(10)  # Check every 10 seconds for due checks
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait longer after errors
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health monitoring status."""
        return {
            "monitoring": self.monitoring,
            "total_checks": len(self.checks),
            "enabled_checks": sum(1 for check in self.checks.values() if check.enabled),
            "checks": {
                name: {
                    "enabled": check.enabled,
                    "interval": check.interval,
                    "critical": check.critical,
                    "last_status": check.last_status.value,
                    "last_check": check.last_check,
                    "consecutive_failures": check.consecutive_failures
                }
                for name, check in self.checks.items()
            }
        }


# Built-in health checks
async def memory_usage_check() -> Dict[str, Any]:
    """Check system memory usage."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"High memory usage: {memory.percent}%"
        elif memory.percent > 75:
            status = HealthStatus.DEGRADED
            message = f"Moderate memory usage: {memory.percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage: {memory.percent}%"
        
        return {
            "status": status.value,
            "message": message,
            "data": {
                "percent": memory.percent,
                "available": memory.available,
                "total": memory.total
            }
        }
    except ImportError:
        return {
            "status": HealthStatus.UNKNOWN.value,
            "message": "psutil not available"
        }


async def disk_usage_check() -> Dict[str, Any]:
    """Check disk space usage."""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        
        percent_used = (disk.used / disk.total) * 100
        
        if percent_used > 90:
            status = HealthStatus.UNHEALTHY
            message = f"High disk usage: {percent_used:.1f}%"
        elif percent_used > 80:
            status = HealthStatus.DEGRADED
            message = f"Moderate disk usage: {percent_used:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage: {percent_used:.1f}%"
        
        return {
            "status": status.value,
            "message": message,
            "data": {
                "percent": percent_used,
                "free": disk.free,
                "total": disk.total
            }
        }
    except ImportError:
        return {
            "status": HealthStatus.UNKNOWN.value,
            "message": "psutil not available"
        }


def basic_connectivity_check() -> bool:
    """Basic connectivity check."""
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


# Global health monitor instance
health_monitor = HealthMonitor()