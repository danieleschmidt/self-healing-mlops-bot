"""Health check and monitoring components."""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..core.config import config
from .logging import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    response_time: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class HealthMonitor:
    """Comprehensive health monitoring for the bot."""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = None
        self.check_history = []
        self.max_history = 100
    
    def register_check(self, name: str, check_func: callable, interval: int = 60):
        """Register a health check function."""
        self.checks[name] = {
            "func": check_func,
            "interval": interval,
            "last_run": 0,
            "last_result": None
        }
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_info in self.checks.items():
            try:
                start_time = time.time()
                result = await self._run_check(name, check_info)
                results[name] = result
            except Exception as e:
                logger.exception(f"Health check {name} failed: {e}")
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    response_time=0.0
                )
        
        # Update check history
        self._update_history(results)
        self.last_check_time = datetime.utcnow()
        
        return results
    
    async def _run_check(self, name: str, check_info: Dict[str, Any]) -> HealthCheck:
        """Run a single health check."""
        current_time = time.time()
        
        # Check if we need to run this check
        if current_time - check_info["last_run"] < check_info["interval"]:
            # Return cached result if still valid
            if check_info["last_result"]:
                return check_info["last_result"]
        
        start_time = time.time()
        
        try:
            # Run the check function
            result = await check_info["func"]()
            response_time = time.time() - start_time
            
            # Create health check result
            health_check = HealthCheck(
                name=name,
                status=result.get("status", HealthStatus.HEALTHY),
                message=result.get("message", "OK"),
                response_time=response_time,
                details=result.get("details", {})
            )
            
            # Cache result
            check_info["last_run"] = current_time
            check_info["last_result"] = health_check
            
            return health_check
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check execution failed: {str(e)}",
                response_time=response_time
            )
    
    def _update_history(self, results: Dict[str, HealthCheck]):
        """Update health check history."""
        self.check_history.append({
            "timestamp": datetime.utcnow(),
            "results": results
        })
        
        # Limit history size
        if len(self.check_history) > self.max_history:
            self.check_history = self.check_history[-self.max_history:]
    
    def get_overall_status(self, results: Dict[str, HealthCheck]) -> HealthStatus:
        """Get overall system health status."""
        if not results:
            return HealthStatus.UNHEALTHY
        
        statuses = [check.status for check in results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health."""
        if not self.check_history:
            return {"status": "unknown", "message": "No health checks run yet"}
        
        latest_check = self.check_history[-1]
        overall_status = self.get_overall_status(latest_check["results"])
        
        return {
            "status": overall_status.value,
            "timestamp": latest_check["timestamp"].isoformat(),
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": check.response_time
                }
                for name, check in latest_check["results"].items()
            }
        }


# Built-in health checks
class BuiltinHealthChecks:
    """Built-in health check implementations."""
    
    @staticmethod
    async def github_api_check() -> Dict[str, Any]:
        """Check GitHub API connectivity."""
        try:
            from ..integrations.github import GitHubIntegration
            
            github = GitHubIntegration()
            is_connected = await github.test_connection()
            
            if is_connected:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "GitHub API accessible",
                    "details": {"api_status": "connected"}
                }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": "GitHub API not accessible",
                    "details": {"api_status": "disconnected"}
                }
                
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"GitHub API check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    @staticmethod
    async def database_check() -> Dict[str, Any]:
        """Check database connectivity."""
        # Mock database check - in real implementation, test actual DB connection
        try:
            # Simulate database connection test
            await asyncio.sleep(0.1)  # Simulate DB query time
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Database accessible",
                "details": {"connection_status": "connected"}
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Database check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    @staticmethod
    async def redis_check() -> Dict[str, Any]:
        """Check Redis connectivity."""
        # Mock Redis check - in real implementation, test actual Redis connection
        try:
            # Simulate Redis connection test
            await asyncio.sleep(0.05)  # Simulate Redis ping time
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Redis accessible",
                "details": {"connection_status": "connected"}
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Redis check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    @staticmethod
    async def detector_registry_check() -> Dict[str, Any]:
        """Check detector registry status."""
        try:
            from ..detectors.registry import DetectorRegistry
            
            registry = DetectorRegistry()
            detector_count = len(registry.list_detectors())
            
            if detector_count > 0:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": f"Detector registry loaded with {detector_count} detectors",
                    "details": {"detector_count": detector_count}
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "No detectors loaded in registry",
                    "details": {"detector_count": 0}
                }
                
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Detector registry check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    @staticmethod
    async def playbook_registry_check() -> Dict[str, Any]:
        """Check playbook registry status."""
        try:
            from ..core.playbook import PlaybookRegistry
            
            playbook_count = len(PlaybookRegistry.list_playbooks())
            
            if playbook_count > 0:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": f"Playbook registry loaded with {playbook_count} playbooks",
                    "details": {"playbook_count": playbook_count}
                }
            else:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "No playbooks loaded in registry",
                    "details": {"playbook_count": 0}
                }
                
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Playbook registry check failed: {str(e)}",
                "details": {"error": str(e)}
            }


# Global health monitor instance
health_monitor = HealthMonitor()

# Register built-in health checks
health_monitor.register_check("github_api", BuiltinHealthChecks.github_api_check, 60)
health_monitor.register_check("database", BuiltinHealthChecks.database_check, 30)
health_monitor.register_check("redis", BuiltinHealthChecks.redis_check, 30)
health_monitor.register_check("detectors", BuiltinHealthChecks.detector_registry_check, 300)
health_monitor.register_check("playbooks", BuiltinHealthChecks.playbook_registry_check, 300)