"""Graceful degradation handler for providing fallback functionality when services are unavailable."""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from functools import wraps
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ServiceLevel(Enum):
    """Service level definitions for degradation."""
    CRITICAL = "critical"      # Core functionality, must work
    IMPORTANT = "important"    # Important but can degrade
    OPTIONAL = "optional"      # Nice to have, can be disabled
    MONITORING = "monitoring"  # Monitoring/metrics, can be skipped


class DegradationMode(Enum):
    """Degradation mode types."""
    CIRCUIT_BREAKER = "circuit_breaker"    # Service circuit breaker is open
    TIMEOUT = "timeout"                    # Service timeouts
    ERROR_RATE = "error_rate"             # High error rate
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Resources unavailable
    MANUAL = "manual"                      # Manually triggered
    DEPENDENCY_FAILURE = "dependency_failure"    # Upstream dependency failed


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    service_name: str
    service_level: ServiceLevel = ServiceLevel.IMPORTANT
    
    # Fallback strategy
    fallback_type: str = "static"  # static, cached, reduced, disabled
    fallback_data: Any = None
    cache_ttl: float = 300.0  # Cache TTL in seconds
    
    # Degradation triggers
    max_error_rate: float = 0.5    # Trigger at 50% error rate
    max_response_time: float = 5.0  # Trigger at 5s response time
    min_success_rate: float = 0.7   # Trigger below 70% success rate
    
    # Recovery settings
    recovery_check_interval: float = 30.0  # Check recovery every 30s
    recovery_success_threshold: int = 3    # Need 3 successes to recover
    
    # Circuit breaker integration
    circuit_breaker_name: Optional[str] = None
    
    # Monitoring
    metrics_enabled: bool = True
    alert_on_degradation: bool = True


@dataclass
class DegradationState:
    """Current degradation state for a service."""
    service_name: str
    is_degraded: bool = False
    degradation_mode: Optional[DegradationMode] = None
    degradation_start: float = 0.0
    degradation_reason: str = ""
    
    # Recovery tracking
    recovery_attempts: int = 0
    consecutive_successes: int = 0
    last_recovery_check: float = 0.0
    
    # Metrics
    total_fallback_calls: int = 0
    successful_fallbacks: int = 0
    failed_fallbacks: int = 0
    
    # Performance tracking
    fallback_response_times: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=20))


@dataclass
class DegradationMetrics:
    """Comprehensive degradation metrics."""
    total_services: int = 0
    degraded_services: int = 0
    total_fallback_calls: int = 0
    successful_fallbacks: int = 0
    failed_fallbacks: int = 0
    
    # Performance metrics
    avg_fallback_response_time: float = 0.0
    fallback_success_rate: float = 100.0
    
    # Recovery metrics
    total_recoveries: int = 0
    avg_degradation_duration: float = 0.0
    
    # By service level
    degradation_by_level: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    last_updated: float = 0.0


class GracefulDegradationHandler:
    """Handler for graceful service degradation with intelligent fallbacks."""
    
    def __init__(self):
        self.services: Dict[str, FallbackConfig] = {}
        self.states: Dict[str, DegradationState] = {}
        self.metrics = DegradationMetrics()
        self.cache: Dict[str, Dict[str, Any]] = {}  # service_name -> cache_data
        
        # Global settings
        self.auto_recovery_enabled = True
        self.recovery_task_running = False
        
        # Fallback strategies
        self.fallback_strategies = {
            "static": self._static_fallback,
            "cached": self._cached_fallback,
            "reduced": self._reduced_functionality_fallback,
            "disabled": self._disabled_fallback,
            "default": self._default_fallback
        }
    
    def register_service(self, config: FallbackConfig):
        """Register a service for graceful degradation."""
        self.services[config.service_name] = config
        self.states[config.service_name] = DegradationState(service_name=config.service_name)
        self.cache[config.service_name] = {}
        
        logger.info(f"Registered service '{config.service_name}' for graceful degradation")
        
        # Start recovery monitoring if not already running
        if self.auto_recovery_enabled and not self.recovery_task_running:
            asyncio.create_task(self._recovery_monitor())
    
    async def execute_with_fallback(
        self,
        service_name: str,
        primary_func: Callable,
        *args,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Execute function with fallback support."""
        if service_name not in self.services:
            logger.warning(f"Service {service_name} not registered for degradation")
            return await self._execute_function(primary_func, *args, **kwargs)
        
        config = self.services[service_name]
        state = self.states[service_name]
        start_time = time.time()
        
        # Check if service is already degraded
        if state.is_degraded:
            logger.info(f"Service {service_name} is degraded, using fallback")
            return await self._execute_fallback(service_name, context, *args, **kwargs)
        
        # Try primary function
        try:
            result = await asyncio.wait_for(
                self._execute_function(primary_func, *args, **kwargs),
                timeout=config.max_response_time
            )
            
            # Record successful execution
            execution_time = time.time() - start_time
            self._record_success(service_name, execution_time)
            
            # Cache successful result if configured
            if config.fallback_type == "cached":
                self._cache_result(service_name, result, args, kwargs)
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Service {service_name} timed out, triggering degradation")
            await self._trigger_degradation(
                service_name, 
                DegradationMode.TIMEOUT, 
                f"Timeout after {config.max_response_time}s"
            )
            return await self._execute_fallback(service_name, context, *args, **kwargs)
            
        except Exception as e:
            logger.warning(f"Service {service_name} failed: {e}")
            self._record_error(service_name, e)
            
            # Check if we should degrade based on error rate
            if self._should_degrade_on_errors(service_name):
                await self._trigger_degradation(
                    service_name,
                    DegradationMode.ERROR_RATE,
                    f"High error rate: {e}"
                )
                return await self._execute_fallback(service_name, context, *args, **kwargs)
            else:
                # Single error, still raise it
                raise
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    async def _execute_fallback(
        self,
        service_name: str,
        context: Optional[Dict[str, Any]],
        *args,
        **kwargs
    ) -> Any:
        """Execute fallback strategy for a degraded service."""
        config = self.services[service_name]
        state = self.states[service_name]
        start_time = time.time()
        
        try:
            # Get fallback strategy
            strategy = self.fallback_strategies.get(
                config.fallback_type, 
                self.fallback_strategies["default"]
            )
            
            # Execute fallback
            result = await strategy(service_name, context, *args, **kwargs)
            
            # Record successful fallback
            execution_time = time.time() - start_time
            state.successful_fallbacks += 1
            state.total_fallback_calls += 1
            state.fallback_response_times.append(execution_time)
            
            logger.info(
                f"Fallback executed successfully for {service_name} in {execution_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            # Record failed fallback
            state.failed_fallbacks += 1
            state.total_fallback_calls += 1
            
            logger.error(f"Fallback failed for {service_name}: {e}")
            
            # For critical services, we might want to raise the error
            if config.service_level == ServiceLevel.CRITICAL:
                raise FallbackFailedException(
                    f"Critical service {service_name} fallback failed: {e}"
                ) from e
            
            # For non-critical services, return a safe default
            return self._get_safe_default(service_name, context)
    
    async def _static_fallback(
        self,
        service_name: str,
        context: Optional[Dict[str, Any]],
        *args,
        **kwargs
    ) -> Any:
        """Return static fallback data."""
        config = self.services[service_name]
        
        if config.fallback_data is not None:
            return config.fallback_data
        
        # Default static responses by service level
        if config.service_level == ServiceLevel.CRITICAL:
            return {"status": "degraded", "message": "Service temporarily unavailable"}
        elif config.service_level == ServiceLevel.IMPORTANT:
            return {"status": "limited", "message": "Limited functionality available"}
        else:
            return {"status": "disabled", "message": "Service disabled"}
    
    async def _cached_fallback(
        self,
        service_name: str,
        context: Optional[Dict[str, Any]],
        *args,
        **kwargs
    ) -> Any:
        """Return cached data if available."""
        cache_key = self._generate_cache_key(args, kwargs)
        service_cache = self.cache.get(service_name, {})
        
        if cache_key in service_cache:
            cache_entry = service_cache[cache_key]
            
            # Check if cache is still valid
            if time.time() - cache_entry["timestamp"] < self.services[service_name].cache_ttl:
                logger.info(f"Returning cached result for {service_name}")
                return cache_entry["data"]
        
        # No valid cache, fall back to static
        logger.warning(f"No valid cache for {service_name}, using static fallback")
        return await self._static_fallback(service_name, context, *args, **kwargs)
    
    async def _reduced_functionality_fallback(
        self,
        service_name: str,
        context: Optional[Dict[str, Any]],
        *args,
        **kwargs
    ) -> Any:
        """Provide reduced functionality."""
        # This would be customized per service
        # For now, return a basic response indicating reduced functionality
        return {
            "status": "reduced_functionality",
            "message": f"Service {service_name} operating in reduced mode",
            "available_features": ["basic_operations"],
            "unavailable_features": ["advanced_operations", "real_time_data"]
        }
    
    async def _disabled_fallback(
        self,
        service_name: str,
        context: Optional[Dict[str, Any]],
        *args,
        **kwargs
    ) -> Any:
        """Service is completely disabled."""
        return {
            "status": "disabled",
            "message": f"Service {service_name} is temporarily disabled",
            "retry_after": 300  # Suggest retry in 5 minutes
        }
    
    async def _default_fallback(
        self,
        service_name: str,
        context: Optional[Dict[str, Any]],
        *args,
        **kwargs
    ) -> Any:
        """Default fallback strategy."""
        return await self._static_fallback(service_name, context, *args, **kwargs)
    
    def _generate_cache_key(self, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate a cache key from function arguments."""
        # Simple string representation - in production, might want more sophisticated hashing
        return f"{str(args)}_{str(sorted(kwargs.items()))}"
    
    def _cache_result(
        self,
        service_name: str,
        result: Any,
        args: Tuple,
        kwargs: Dict[str, Any]
    ):
        """Cache a successful result."""
        cache_key = self._generate_cache_key(args, kwargs)
        
        if service_name not in self.cache:
            self.cache[service_name] = {}
        
        self.cache[service_name][cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        
        # Limit cache size (keep last 100 entries per service)
        if len(self.cache[service_name]) > 100:
            oldest_key = min(
                self.cache[service_name].keys(),
                key=lambda k: self.cache[service_name][k]["timestamp"]
            )
            del self.cache[service_name][oldest_key]
    
    def _record_success(self, service_name: str, execution_time: float):
        """Record successful execution."""
        state = self.states[service_name]
        
        # If service was degraded, increment consecutive successes
        if state.is_degraded:
            state.consecutive_successes += 1
            logger.debug(
                f"Service {service_name} success #{state.consecutive_successes} "
                f"towards recovery"
            )
    
    def _record_error(self, service_name: str, error: Exception):
        """Record error for a service."""
        state = self.states[service_name]
        state.recent_errors.append({
            "error": str(error),
            "timestamp": time.time()
        })
        
        # Reset consecutive successes if we were trying to recover
        if state.is_degraded:
            state.consecutive_successes = 0
    
    def _should_degrade_on_errors(self, service_name: str) -> bool:
        """Check if service should be degraded based on error rate."""
        config = self.services[service_name]
        state = self.states[service_name]
        
        if len(state.recent_errors) < 5:  # Need some errors to calculate rate
            return False
        
        # Calculate error rate in the last 60 seconds
        current_time = time.time()
        recent_errors = [
            e for e in state.recent_errors
            if current_time - e["timestamp"] < 60
        ]
        
        # If we have too many recent errors, degrade
        error_rate = len(recent_errors) / 60  # errors per second
        return error_rate > (config.max_error_rate / 60)
    
    async def _trigger_degradation(
        self,
        service_name: str,
        mode: DegradationMode,
        reason: str
    ):
        """Trigger degradation for a service."""
        state = self.states[service_name]
        config = self.services[service_name]
        
        if not state.is_degraded:
            state.is_degraded = True
            state.degradation_mode = mode
            state.degradation_start = time.time()
            state.degradation_reason = reason
            state.consecutive_successes = 0
            
            self.metrics.degraded_services += 1
            self.metrics.degradation_by_level[config.service_level.value] += 1
            
            logger.warning(
                f"Service {service_name} degraded: {mode.value} - {reason}"
            )
            
            if config.alert_on_degradation:
                await self._send_degradation_alert(service_name, mode, reason)
    
    async def _send_degradation_alert(
        self,
        service_name: str,
        mode: DegradationMode,
        reason: str
    ):
        """Send alert for service degradation."""
        # This would integrate with alerting systems
        # For now, just log at error level
        config = self.services[service_name]
        
        logger.error(
            f"ALERT: Service degradation - {service_name} "
            f"(level: {config.service_level.value}, mode: {mode.value}) - {reason}"
        )
    
    async def _recovery_monitor(self):
        """Background task to monitor service recovery."""
        self.recovery_task_running = True
        
        try:
            while self.auto_recovery_enabled:
                await self._check_service_recoveries()
                await asyncio.sleep(10)  # Check every 10 seconds
        except Exception as e:
            logger.error(f"Recovery monitor error: {e}")
        finally:
            self.recovery_task_running = False
    
    async def _check_service_recoveries(self):
        """Check if any degraded services can be recovered."""
        current_time = time.time()
        
        for service_name, state in self.states.items():
            if not state.is_degraded:
                continue
            
            config = self.services[service_name]
            
            # Check if it's time for a recovery check
            if (current_time - state.last_recovery_check) < config.recovery_check_interval:
                continue
            
            state.last_recovery_check = current_time
            state.recovery_attempts += 1
            
            # Try to recover the service
            await self._attempt_service_recovery(service_name)
    
    async def _attempt_service_recovery(self, service_name: str):
        """Attempt to recover a degraded service."""
        config = self.services[service_name]
        state = self.states[service_name]
        
        # Check if we have enough consecutive successes
        if state.consecutive_successes >= config.recovery_success_threshold:
            await self._recover_service(service_name)
        else:
            logger.debug(
                f"Service {service_name} needs {config.recovery_success_threshold - state.consecutive_successes} "
                f"more successes for recovery"
            )
    
    async def _recover_service(self, service_name: str):
        """Recover a service from degraded state."""
        state = self.states[service_name]
        config = self.services[service_name]
        
        degradation_duration = time.time() - state.degradation_start
        
        state.is_degraded = False
        state.degradation_mode = None
        state.degradation_start = 0.0
        state.degradation_reason = ""
        state.consecutive_successes = 0
        state.recovery_attempts = 0
        
        self.metrics.degraded_services -= 1
        self.metrics.total_recoveries += 1
        self.metrics.degradation_by_level[config.service_level.value] -= 1
        
        # Update average degradation duration
        if self.metrics.total_recoveries > 0:
            current_avg = self.metrics.avg_degradation_duration
            self.metrics.avg_degradation_duration = (
                (current_avg * (self.metrics.total_recoveries - 1) + degradation_duration) /
                self.metrics.total_recoveries
            )
        
        logger.info(
            f"Service {service_name} recovered after {degradation_duration:.1f}s "
            f"(attempts: {state.recovery_attempts})"
        )
    
    def _get_safe_default(self, service_name: str, context: Optional[Dict[str, Any]]) -> Any:
        """Get a safe default value when all fallbacks fail."""
        config = self.services[service_name]
        
        if config.service_level == ServiceLevel.CRITICAL:
            # For critical services, we might want to raise an error
            return None
        else:
            # For non-critical services, return a safe empty response
            return {
                "status": "unavailable",
                "message": f"Service {service_name} is temporarily unavailable",
                "service_level": config.service_level.value
            }
    
    def manually_degrade_service(
        self,
        service_name: str,
        reason: str = "Manual degradation"
    ):
        """Manually trigger service degradation."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not registered")
        
        asyncio.create_task(
            self._trigger_degradation(service_name, DegradationMode.MANUAL, reason)
        )
    
    def manually_recover_service(self, service_name: str):
        """Manually recover a service."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not registered")
        
        asyncio.create_task(self._recover_service(service_name))
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status for a specific service."""
        if service_name not in self.services:
            return {"error": f"Service {service_name} not registered"}
        
        config = self.services[service_name]
        state = self.states[service_name]
        
        status = {
            "service_name": service_name,
            "service_level": config.service_level.value,
            "is_degraded": state.is_degraded,
            "degradation_mode": state.degradation_mode.value if state.degradation_mode else None,
            "degradation_duration": (
                time.time() - state.degradation_start
                if state.is_degraded else 0.0
            ),
            "degradation_reason": state.degradation_reason,
            "consecutive_successes": state.consecutive_successes,
            "recovery_attempts": state.recovery_attempts,
            "fallback_stats": {
                "total_calls": state.total_fallback_calls,
                "successful_calls": state.successful_fallbacks,
                "failed_calls": state.failed_fallbacks,
                "success_rate": (
                    (state.successful_fallbacks / state.total_fallback_calls * 100)
                    if state.total_fallback_calls > 0 else 100.0
                )
            }
        }
        
        # Add performance metrics
        if state.fallback_response_times:
            status["performance"] = {
                "avg_response_time": sum(state.fallback_response_times) / len(state.fallback_response_times),
                "min_response_time": min(state.fallback_response_times),
                "max_response_time": max(state.fallback_response_times)
            }
        
        return status
    
    def get_all_services_status(self) -> Dict[str, Any]:
        """Get status for all registered services."""
        return {
            "services": {
                name: self.get_service_status(name)
                for name in self.services.keys()
            },
            "global_metrics": self._get_global_metrics()
        }
    
    def _get_global_metrics(self) -> Dict[str, Any]:
        """Get global degradation metrics."""
        self.metrics.total_services = len(self.services)
        self.metrics.last_updated = time.time()
        
        # Update success rates
        total_fallback_calls = sum(state.total_fallback_calls for state in self.states.values())
        successful_fallbacks = sum(state.successful_fallbacks for state in self.states.values())
        
        self.metrics.total_fallback_calls = total_fallback_calls
        self.metrics.successful_fallbacks = successful_fallbacks
        
        if total_fallback_calls > 0:
            self.metrics.fallback_success_rate = (successful_fallbacks / total_fallback_calls) * 100
        
        # Calculate average response time
        all_response_times = []
        for state in self.states.values():
            all_response_times.extend(state.fallback_response_times)
        
        if all_response_times:
            self.metrics.avg_fallback_response_time = sum(all_response_times) / len(all_response_times)
        
        return {
            "total_services": self.metrics.total_services,
            "degraded_services": self.metrics.degraded_services,
            "healthy_services": self.metrics.total_services - self.metrics.degraded_services,
            "degradation_rate": (
                (self.metrics.degraded_services / self.metrics.total_services * 100)
                if self.metrics.total_services > 0 else 0.0
            ),
            "total_fallback_calls": self.metrics.total_fallback_calls,
            "fallback_success_rate": self.metrics.fallback_success_rate,
            "avg_fallback_response_time": self.metrics.avg_fallback_response_time,
            "total_recoveries": self.metrics.total_recoveries,
            "avg_degradation_duration": self.metrics.avg_degradation_duration,
            "degradation_by_level": dict(self.metrics.degradation_by_level)
        }
    
    @asynccontextmanager
    async def service_context(self, service_name: str, context: Optional[Dict[str, Any]] = None):
        """Context manager for service operations with automatic fallback."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not registered")
        
        state = self.states[service_name]
        
        if state.is_degraded:
            logger.info(f"Service {service_name} is degraded, skipping context")
            yield None
            return
        
        try:
            yield service_name
        except Exception as e:
            logger.warning(f"Error in service context for {service_name}: {e}")
            self._record_error(service_name, e)
            
            if self._should_degrade_on_errors(service_name):
                await self._trigger_degradation(
                    service_name,
                    DegradationMode.ERROR_RATE,
                    f"Context error: {e}"
                )
            
            raise


class FallbackFailedException(Exception):
    """Exception raised when fallback mechanisms fail."""
    pass


# Global degradation handler instance
degradation_handler = GracefulDegradationHandler()


# Predefined configurations for common services
GITHUB_API_FALLBACK_CONFIG = FallbackConfig(
    service_name="github_api",
    service_level=ServiceLevel.IMPORTANT,
    fallback_type="cached",
    cache_ttl=600.0,
    max_error_rate=0.3,
    max_response_time=10.0,
    recovery_check_interval=60.0
)

DATABASE_FALLBACK_CONFIG = FallbackConfig(
    service_name="database",
    service_level=ServiceLevel.CRITICAL,
    fallback_type="cached",
    cache_ttl=300.0,
    max_error_rate=0.1,
    max_response_time=5.0,
    recovery_check_interval=30.0
)

ML_INFERENCE_FALLBACK_CONFIG = FallbackConfig(
    service_name="ml_inference",
    service_level=ServiceLevel.IMPORTANT,
    fallback_type="reduced",
    fallback_data={"model_version": "fallback", "confidence": 0.5},
    max_error_rate=0.4,
    max_response_time=30.0,
    recovery_check_interval=120.0
)

MONITORING_FALLBACK_CONFIG = FallbackConfig(
    service_name="monitoring",
    service_level=ServiceLevel.MONITORING,
    fallback_type="disabled",
    max_error_rate=0.8,
    max_response_time=3.0,
    recovery_check_interval=300.0
)


def graceful_degradation(
    service_name: str,
    fallback_config: Optional[FallbackConfig] = None
):
    """Decorator for graceful degradation of functions."""
    def decorator(func: Callable) -> Callable:
        # Register service if config provided
        if fallback_config:
            degradation_handler.register_service(fallback_config)
        elif service_name not in degradation_handler.services:
            # Create default config
            default_config = FallbackConfig(
                service_name=service_name,
                service_level=ServiceLevel.IMPORTANT,
                fallback_type="static",
                fallback_data={"status": "degraded", "message": "Service unavailable"}
            )
            degradation_handler.register_service(default_config)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await degradation_handler.execute_with_fallback(
                service_name, func, *args, **kwargs
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(async_wrapper(*args, **kwargs))
            else:
                return asyncio.run(
                    degradation_handler.execute_with_fallback(
                        service_name, func, *args, **kwargs
                    )
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Convenience decorators for common service types
def github_api_fallback(func: Callable) -> Callable:
    """Decorator for GitHub API calls with graceful degradation."""
    degradation_handler.register_service(GITHUB_API_FALLBACK_CONFIG)
    return graceful_degradation("github_api")(func)


def database_fallback(func: Callable) -> Callable:
    """Decorator for database calls with graceful degradation."""
    degradation_handler.register_service(DATABASE_FALLBACK_CONFIG)
    return graceful_degradation("database")(func)


def ml_inference_fallback(func: Callable) -> Callable:
    """Decorator for ML inference with graceful degradation."""
    degradation_handler.register_service(ML_INFERENCE_FALLBACK_CONFIG)
    return graceful_degradation("ml_inference")(func)


def monitoring_fallback(func: Callable) -> Callable:
    """Decorator for monitoring services with graceful degradation."""
    degradation_handler.register_service(MONITORING_FALLBACK_CONFIG)
    return graceful_degradation("monitoring")(func)