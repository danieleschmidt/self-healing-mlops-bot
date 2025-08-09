"""Comprehensive fault tolerance manager that coordinates all reliability patterns."""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
from functools import wraps

from .circuit_breaker import circuit_breaker_manager, CircuitBreakerConfig, CircuitState
from .retry_handler import retry_manager, RetryConfig, RetryStrategy
from .health_monitor import health_monitor, HealthStatus
from ..monitoring.metrics import prometheus_metrics

logger = logging.getLogger(__name__)


class FaultTolerancePolicy(Enum):
    """Fault tolerance policy levels."""
    BASIC = "basic"  # Basic retry and circuit breaker
    STANDARD = "standard"  # Standard with health monitoring
    AGGRESSIVE = "aggressive"  # All patterns with tight thresholds
    RELAXED = "relaxed"  # Loose thresholds for development
    CUSTOM = "custom"  # Custom configuration


@dataclass
class FaultToleranceConfig:
    """Comprehensive fault tolerance configuration."""
    name: str
    policy: FaultTolerancePolicy = FaultTolerancePolicy.STANDARD
    
    # Circuit breaker configuration
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    circuit_breaker_enabled: bool = True
    
    # Retry configuration
    retry_config: Optional[RetryConfig] = None
    retry_enabled: bool = True
    
    # Health monitoring
    health_monitoring_enabled: bool = True
    health_check_interval: int = 60
    health_degradation_threshold: float = 0.8
    
    # Timeout configuration
    timeout_enabled: bool = True
    default_timeout: float = 30.0
    adaptive_timeout: bool = False
    
    # Rate limiting
    rate_limiting_enabled: bool = False
    rate_limit_rps: Optional[int] = None
    
    # Bulkhead isolation
    bulkhead_enabled: bool = False
    max_concurrent_requests: Optional[int] = None
    
    # Metrics and monitoring
    metrics_enabled: bool = True
    alerting_enabled: bool = True
    
    # Graceful degradation
    fallback_enabled: bool = False
    fallback_function: Optional[Callable] = None
    
    def __post_init__(self):
        """Initialize default configurations based on policy."""
        if self.policy != FaultTolerancePolicy.CUSTOM:
            self._apply_policy_defaults()
    
    def _apply_policy_defaults(self):
        """Apply default configurations based on policy."""
        if self.policy == FaultTolerancePolicy.BASIC:
            # Basic configuration
            if not self.circuit_breaker_config:
                self.circuit_breaker_config = CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=60,
                    success_threshold=2
                )
            if not self.retry_config:
                self.retry_config = RetryConfig(
                    max_retries=3,
                    base_delay=1.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
                )
            
        elif self.policy == FaultTolerancePolicy.STANDARD:
            # Standard configuration with monitoring
            if not self.circuit_breaker_config:
                self.circuit_breaker_config = CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=30,
                    success_threshold=2,
                    adaptive_thresholds=True,
                    metrics_enabled=True
                )
            if not self.retry_config:
                self.retry_config = RetryConfig(
                    max_retries=3,
                    base_delay=1.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    circuit_breaker_aware=True,
                    adaptive_scaling=True
                )
            
        elif self.policy == FaultTolerancePolicy.AGGRESSIVE:
            # Aggressive configuration with tight thresholds
            if not self.circuit_breaker_config:
                self.circuit_breaker_config = CircuitBreakerConfig(
                    failure_threshold=2,
                    recovery_timeout=15,
                    success_threshold=3,
                    adaptive_thresholds=True,
                    failure_rate_threshold=0.3,
                    metrics_enabled=True
                )
            if not self.retry_config:
                self.retry_config = RetryConfig(
                    max_retries=5,
                    base_delay=0.5,
                    max_delay=20.0,
                    strategy=RetryStrategy.FIBONACCI_BACKOFF,
                    circuit_breaker_aware=True,
                    adaptive_scaling=True
                )
            self.rate_limiting_enabled = True
            self.bulkhead_enabled = True
            
        elif self.policy == FaultTolerancePolicy.RELAXED:
            # Relaxed configuration for development
            if not self.circuit_breaker_config:
                self.circuit_breaker_config = CircuitBreakerConfig(
                    failure_threshold=10,
                    recovery_timeout=120,
                    success_threshold=1,
                    adaptive_thresholds=False
                )
            if not self.retry_config:
                self.retry_config = RetryConfig(
                    max_retries=2,
                    base_delay=2.0,
                    strategy=RetryStrategy.LINEAR_BACKOFF,
                    circuit_breaker_aware=False
                )


class FaultToleranceManager:
    """Comprehensive fault tolerance manager that coordinates all reliability patterns."""
    
    def __init__(self):
        self.configurations: Dict[str, FaultToleranceConfig] = {}
        self.active_requests: Dict[str, int] = {}
        self.request_metrics: Dict[str, List[float]] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        
        # Global settings
        self.global_timeout = 30.0
        self.monitoring_enabled = True
        
        # Register default configurations
        self._register_default_configs()
    
    def _register_default_configs(self):
        """Register default fault tolerance configurations."""
        # Default configurations for common services
        configs = [
            FaultToleranceConfig(
                name="github_api",
                policy=FaultTolerancePolicy.STANDARD,
                circuit_breaker_config=CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=30,
                    adaptive_thresholds=True
                ),
                retry_config=RetryConfig(
                    max_retries=5,
                    base_delay=2.0,
                    exponential_base=1.5,
                    circuit_breaker_aware=True
                ),
                rate_limiting_enabled=True,
                rate_limit_rps=10
            ),
            FaultToleranceConfig(
                name="database",
                policy=FaultTolerancePolicy.STANDARD,
                circuit_breaker_config=CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=60
                ),
                retry_config=RetryConfig(
                    max_retries=2,
                    base_delay=0.5,
                    strategy=RetryStrategy.LINEAR_BACKOFF
                ),
                bulkhead_enabled=True,
                max_concurrent_requests=10
            ),
            FaultToleranceConfig(
                name="ml_inference",
                policy=FaultTolerancePolicy.AGGRESSIVE,
                circuit_breaker_config=CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=45,
                    timeout=120
                ),
                retry_config=RetryConfig(
                    max_retries=4,
                    base_delay=1.5,
                    strategy=RetryStrategy.ADAPTIVE,
                    timeout_multiplier=1.2
                ),
                default_timeout=120.0,
                adaptive_timeout=True
            ),
            FaultToleranceConfig(
                name="webhook_processing",
                policy=FaultTolerancePolicy.RELAXED,
                circuit_breaker_config=CircuitBreakerConfig(
                    failure_threshold=10,
                    recovery_timeout=120
                ),
                retry_config=RetryConfig(
                    max_retries=2,
                    base_delay=1.0
                ),
                bulkhead_enabled=True,
                max_concurrent_requests=20
            )
        ]
        
        for config in configs:
            self.register_config(config)
    
    def register_config(self, config: FaultToleranceConfig):
        """Register a fault tolerance configuration."""
        self.configurations[config.name] = config
        logger.info(f"Registered fault tolerance configuration: {config.name}")
        
        # Initialize circuit breaker if enabled
        if config.circuit_breaker_enabled and config.circuit_breaker_config:
            circuit_breaker_manager.get_breaker(config.name, config.circuit_breaker_config)
        
        # Initialize retry handler if enabled
        if config.retry_enabled and config.retry_config:
            retry_handler = retry_manager.get_handler("network")  # Default handler
            if retry_handler:
                retry_handler.set_context_config(config.name, config.retry_config)
    
    def get_config(self, name: str) -> Optional[FaultToleranceConfig]:
        """Get fault tolerance configuration by name."""
        return self.configurations.get(name)
    
    async def execute_with_fault_tolerance(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with comprehensive fault tolerance."""
        config = self.get_config(name)
        if not config:
            logger.warning(f"No fault tolerance configuration found for {name}, using defaults")
            config = FaultToleranceConfig(name=name, policy=FaultTolerancePolicy.STANDARD)
            self.register_config(config)
        
        start_time = time.time()
        
        # Check bulkhead limits
        if config.bulkhead_enabled:
            if not await self._check_bulkhead_capacity(name, config):
                raise BulkheadFullError(f"Bulkhead capacity exceeded for {name}")
        
        # Check circuit breaker state
        if config.circuit_breaker_enabled:
            breaker = circuit_breaker_manager.get_breaker(name)
            if breaker.state == CircuitState.OPEN:
                if config.fallback_enabled and config.fallback_function:
                    logger.info(f"Circuit breaker open for {name}, executing fallback")
                    return await self._execute_fallback(name, config, *args, **kwargs)
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker is open for {name}")
        
        # Check health status
        if config.health_monitoring_enabled:
            health_status = await self._check_health_status(name)
            if health_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                if config.fallback_enabled and config.fallback_function:
                    logger.warning(f"Service {name} is {health_status.value}, using fallback")
                    return await self._execute_fallback(name, config, *args, **kwargs)
        
        try:
            # Increment active requests for bulkhead
            if config.bulkhead_enabled:
                self.active_requests[name] = self.active_requests.get(name, 0) + 1
            
            # Execute with timeout
            if config.timeout_enabled:
                timeout = self._calculate_adaptive_timeout(name, config)
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                else:
                    result = func(*args, **kwargs)
            else:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            
            # Record successful execution
            execution_time = time.time() - start_time
            await self._record_execution_metrics(name, execution_time, True)
            
            return result
            
        except Exception as e:
            # Record failed execution
            execution_time = time.time() - start_time
            await self._record_execution_metrics(name, execution_time, False)
            
            # Try fallback if available
            if config.fallback_enabled and config.fallback_function:
                logger.warning(f"Execution failed for {name}, trying fallback: {e}")
                try:
                    return await self._execute_fallback(name, config, *args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {name}: {fallback_error}")
                    raise e  # Raise original exception
            
            raise
        
        finally:
            # Decrement active requests for bulkhead
            if config.bulkhead_enabled and name in self.active_requests:
                self.active_requests[name] = max(0, self.active_requests[name] - 1)
    
    async def _check_bulkhead_capacity(self, name: str, config: FaultToleranceConfig) -> bool:
        """Check if bulkhead has capacity for new request."""
        if not config.max_concurrent_requests:
            return True
        
        current_requests = self.active_requests.get(name, 0)
        return current_requests < config.max_concurrent_requests
    
    async def _check_health_status(self, name: str) -> HealthStatus:
        """Check health status for the service."""
        # Check if there's a specific health check for this service
        if hasattr(health_monitor, 'checks') and name in health_monitor.checks:
            check = health_monitor.checks[name]
            return check.last_status
        
        # Default to healthy if no specific check
        return HealthStatus.HEALTHY
    
    def _calculate_adaptive_timeout(self, name: str, config: FaultToleranceConfig) -> float:
        """Calculate adaptive timeout based on historical performance."""
        if not config.adaptive_timeout:
            return config.default_timeout
        
        # Get recent response times
        recent_times = self.request_metrics.get(name, [])
        if not recent_times:
            return config.default_timeout
        
        # Calculate adaptive timeout based on percentiles
        recent_times = sorted(recent_times[-20:])  # Last 20 requests
        if len(recent_times) >= 5:
            p95_time = recent_times[int(0.95 * len(recent_times))]
            # Set timeout to 2x the 95th percentile, with bounds
            adaptive_timeout = max(
                config.default_timeout * 0.5,  # Minimum 50% of default
                min(p95_time * 2, config.default_timeout * 3)  # Maximum 3x default
            )
            return adaptive_timeout
        
        return config.default_timeout
    
    async def _execute_fallback(
        self, 
        name: str, 
        config: FaultToleranceConfig, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute fallback function."""
        if not config.fallback_function:
            raise ValueError(f"No fallback function configured for {name}")
        
        try:
            if asyncio.iscoroutinefunction(config.fallback_function):
                return await config.fallback_function(*args, **kwargs)
            else:
                return config.fallback_function(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fallback execution failed for {name}: {e}")
            raise FallbackError(f"Fallback failed for {name}: {e}") from e
    
    async def _record_execution_metrics(self, name: str, execution_time: float, success: bool):
        """Record execution metrics."""
        if name not in self.request_metrics:
            self.request_metrics[name] = []
        
        # Store response time (only for successful requests)
        if success:
            self.request_metrics[name].append(execution_time)
            # Keep only recent metrics
            self.request_metrics[name] = self.request_metrics[name][-50:]
        
        # Record Prometheus metrics if available
        if prometheus_metrics and self.monitoring_enabled:
            prometheus_metrics.record_event_processed(
                event_type=name,
                repo="system",
                status="success" if success else "failed",
                duration=execution_time
            )
    
    def create_fault_tolerant_decorator(self, name: str, config: Optional[FaultToleranceConfig] = None):
        """Create a decorator for fault-tolerant execution."""
        if config:
            self.register_config(config)
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.execute_with_fault_tolerance(name, func, *args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if asyncio.iscoroutinefunction(func):
                    return asyncio.run(async_wrapper(*args, **kwargs))
                else:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self.execute_with_fault_tolerance(name, func, *args, **kwargs)
                        )
                    finally:
                        loop.close()
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    @asynccontextmanager
    async def fault_tolerant_context(self, name: str):
        """Context manager for fault-tolerant operations."""
        config = self.get_config(name)
        if not config:
            config = FaultToleranceConfig(name=name)
            self.register_config(config)
        
        start_time = time.time()
        
        # Check bulkhead limits
        if config.bulkhead_enabled:
            if not await self._check_bulkhead_capacity(name, config):
                raise BulkheadFullError(f"Bulkhead capacity exceeded for {name}")
            self.active_requests[name] = self.active_requests.get(name, 0) + 1
        
        try:
            yield config
            
            # Record successful operation
            execution_time = time.time() - start_time
            await self._record_execution_metrics(name, execution_time, True)
            
        except Exception as e:
            # Record failed operation
            execution_time = time.time() - start_time
            await self._record_execution_metrics(name, execution_time, False)
            raise
        
        finally:
            # Decrement active requests for bulkhead
            if config.bulkhead_enabled and name in self.active_requests:
                self.active_requests[name] = max(0, self.active_requests[name] - 1)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of fault tolerance manager."""
        status = {
            "total_configurations": len(self.configurations),
            "active_requests": dict(self.active_requests),
            "monitoring_enabled": self.monitoring_enabled,
            "configurations": {}
        }
        
        for name, config in self.configurations.items():
            # Get circuit breaker status
            cb_status = "not_configured"
            if config.circuit_breaker_enabled:
                breaker = circuit_breaker_manager.get_breaker(name)
                cb_status = breaker.state.value if breaker else "unknown"
            
            # Get recent metrics
            recent_times = self.request_metrics.get(name, [])
            avg_response_time = sum(recent_times) / len(recent_times) if recent_times else 0
            
            status["configurations"][name] = {
                "policy": config.policy.value,
                "circuit_breaker_enabled": config.circuit_breaker_enabled,
                "circuit_breaker_status": cb_status,
                "retry_enabled": config.retry_enabled,
                "health_monitoring_enabled": config.health_monitoring_enabled,
                "bulkhead_enabled": config.bulkhead_enabled,
                "active_requests": self.active_requests.get(name, 0),
                "max_concurrent": config.max_concurrent_requests,
                "recent_requests": len(recent_times),
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "fallback_enabled": config.fallback_enabled
            }
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        metrics = {
            "total_services": len(self.configurations),
            "total_active_requests": sum(self.active_requests.values()),
            "services": {}
        }
        
        for name in self.configurations:
            recent_times = self.request_metrics.get(name, [])
            metrics["services"][name] = {
                "total_requests": len(recent_times),
                "active_requests": self.active_requests.get(name, 0),
                "avg_response_time": sum(recent_times) / len(recent_times) if recent_times else 0,
                "min_response_time": min(recent_times) if recent_times else 0,
                "max_response_time": max(recent_times) if recent_times else 0
            }
        
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.request_metrics.clear()
        self.active_requests.clear()
        logger.info("Fault tolerance metrics reset")


# Custom exceptions
class BulkheadFullError(Exception):
    """Exception raised when bulkhead capacity is exceeded."""
    pass


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class FallbackError(Exception):
    """Exception raised when fallback execution fails."""
    pass


# Global fault tolerance manager instance
fault_tolerance_manager = FaultToleranceManager()


# Convenience decorators
def fault_tolerant(
    name: str, 
    policy: FaultTolerancePolicy = FaultTolerancePolicy.STANDARD,
    **config_kwargs
):
    """Decorator for fault-tolerant execution with custom configuration."""
    config = FaultToleranceConfig(name=name, policy=policy, **config_kwargs)
    return fault_tolerance_manager.create_fault_tolerant_decorator(name, config)


def github_api_fault_tolerant(name: str = "github_api"):
    """Decorator for GitHub API calls with fault tolerance."""
    return fault_tolerance_manager.create_fault_tolerant_decorator(name)


def database_fault_tolerant(name: str = "database"):
    """Decorator for database operations with fault tolerance."""
    return fault_tolerance_manager.create_fault_tolerant_decorator(name)


def ml_inference_fault_tolerant(name: str = "ml_inference"):
    """Decorator for ML inference operations with fault tolerance."""
    return fault_tolerance_manager.create_fault_tolerant_decorator(name)


def webhook_fault_tolerant(name: str = "webhook_processing"):
    """Decorator for webhook processing with fault tolerance."""
    return fault_tolerance_manager.create_fault_tolerant_decorator(name)