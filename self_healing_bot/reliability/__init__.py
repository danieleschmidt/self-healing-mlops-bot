"""Reliability and robustness components for the self-healing bot."""

# Enhanced core components
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerMetrics,
    FailureType,
    circuit_breaker_manager,
    circuit_breaker,
    CircuitBreakerOpenError
)

from .retry_handler import (
    RetryHandler,
    RetryConfig,
    RetryStrategy,
    FailureClassification,
    RetryContext,
    RetryMetrics,
    retry,
    retry_network,
    retry_database,
    retry_github_api,
    retry_ml_inference,
    retry_critical,
    retry_manager,
    RetryExhaustedException
)

from .health_monitor import (
    HealthMonitor,
    HealthMonitorConfig,
    HealthStatus,
    ServiceDependency,
    Alert,
    AlertSeverity,
    HealthMetrics,
    PerformanceMetrics,
    health_monitor,
    health_check
)

# New reliability components
from .fault_tolerance_manager import (
    FaultToleranceManager,
    FaultTolerancePolicy,
    FallbackHandler,
    fault_tolerance_manager,
    fault_tolerant
)

from .rate_limiter import (
    RateLimiter,
    RateLimiterConfig,
    RateLimitStrategy,
    RateLimitResult,
    TokenBucketLimiter,
    SlidingWindowLimiter,
    rate_limiter_manager,
    rate_limit
)

from .bulkhead_manager import (
    BulkheadManager,
    BulkheadConfig,
    PoolPriority,
    ResourcePool,
    bulkhead_manager,
    create_bulkhead_decorator,
    BulkheadFullError,
    BulkheadTimeoutError
)

from .graceful_degradation import (
    GracefulDegradationHandler,
    FallbackConfig,
    ServiceLevel,
    DegradationMode,
    degradation_handler,
    graceful_degradation,
    github_api_fallback,
    database_fallback,
    ml_inference_fallback,
    monitoring_fallback,
    FallbackFailedException
)

from .service_mesh import (
    ServiceMesh,
    ServiceMeshConfig,
    ServiceEndpoint,
    LoadBalancingStrategy,
    TrafficSplitStrategy,
    ServiceHealthStatus,
    RequestContext,
    service_mesh_manager,
    service_mesh_integration,
    CircuitBreakerOpenError as MeshCircuitBreakerOpenError,
    ServiceUnavailableError,
    RequestFailedException
)

# Legacy components (if they exist)
try:
    from .error_recovery import ErrorRecoverySystem
except ImportError:
    ErrorRecoverySystem = None

try:
    from .timeout_manager import TimeoutManager
except ImportError:
    TimeoutManager = None


class ReliabilityManager:
    """Centralized manager for all reliability components."""
    
    def __init__(self):
        self.circuit_breaker_manager = circuit_breaker_manager
        self.retry_manager = retry_manager
        self.health_monitor = health_monitor
        self.fault_tolerance_manager = fault_tolerance_manager
        self.rate_limiter_manager = rate_limiter_manager
        self.bulkhead_manager = bulkhead_manager
        self.degradation_handler = degradation_handler
        self.service_mesh_manager = service_mesh_manager
    
    def get_system_health(self):
        """Get comprehensive system health status."""
        return {
            "circuit_breakers": self.circuit_breaker_manager.get_all_stats(),
            "retry_handlers": self.retry_manager.get_global_metrics(),
            "health_monitors": self.health_monitor.get_system_health(),
            "fault_tolerance": self.fault_tolerance_manager.get_global_stats(),
            "rate_limiters": self.rate_limiter_manager.get_all_stats(),
            "bulkheads": self.bulkhead_manager.get_all_stats(),
            "degradation": self.degradation_handler.get_all_services_status(),
            "service_mesh": self.service_mesh_manager.get_all_status(),
            "timestamp": __import__('time').time()
        }
    
    def get_reliability_score(self):
        """Calculate overall system reliability score (0-100)."""
        try:
            health_data = self.get_system_health()
            
            # Circuit breaker score (30% weight)
            cb_stats = health_data.get("circuit_breakers", {})
            cb_score = 100.0
            if cb_stats.get("total_breakers", 0) > 0:
                open_breakers = cb_stats.get("open_breakers", 0)
                cb_score = max(0, 100 - (open_breakers / cb_stats["total_breakers"] * 100))
            
            # Retry handler score (20% weight)
            retry_stats = health_data.get("retry_handlers", {})
            retry_score = 100.0
            for handler_name, metrics in retry_stats.items():
                if isinstance(metrics, dict) and "success_rate" in metrics:
                    retry_score = min(retry_score, metrics["success_rate"])
            
            # Health monitor score (25% weight)
            health_stats = health_data.get("health_monitors", {})
            health_score = health_stats.get("overall_health_score", 100.0)
            
            # Service mesh score (25% weight)
            mesh_stats = health_data.get("service_mesh", {})
            mesh_score = 100.0
            services = mesh_stats.get("services", {})
            if services:
                service_scores = []
                for service_data in services.values():
                    metrics = service_data.get("metrics", {})
                    service_scores.append(metrics.get("success_rate", 100.0))
                mesh_score = sum(service_scores) / len(service_scores) if service_scores else 100.0
            
            # Calculate weighted average
            reliability_score = (
                cb_score * 0.30 +
                retry_score * 0.20 +
                health_score * 0.25 +
                mesh_score * 0.25
            )
            
            return min(100.0, max(0.0, reliability_score))
            
        except Exception:
            return 50.0  # Default score if calculation fails
    
    async def shutdown_all(self):
        """Gracefully shutdown all reliability components."""
        await self.bulkhead_manager.shutdown_all()
        # Add other component shutdowns as needed


# Global reliability manager instance
reliability_manager = ReliabilityManager()


# Export all components
__all__ = [
    # Core enhanced components
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "CircuitBreakerMetrics",
    "FailureType",
    "circuit_breaker_manager",
    "circuit_breaker",
    "CircuitBreakerOpenError",
    
    "RetryHandler",
    "RetryConfig",
    "RetryStrategy",
    "FailureClassification",
    "RetryContext",
    "RetryMetrics",
    "retry",
    "retry_network",
    "retry_database",
    "retry_github_api",
    "retry_ml_inference",
    "retry_critical",
    "retry_manager",
    "RetryExhaustedException",
    
    "HealthMonitor",
    "HealthMonitorConfig",
    "HealthStatus",
    "ServiceDependency",
    "Alert",
    "AlertSeverity",
    "HealthMetrics",
    "PerformanceMetrics",
    "health_monitor",
    "health_check",
    
    # New reliability components
    "FaultToleranceManager",
    "FaultTolerancePolicy",
    "FallbackHandler",
    "fault_tolerance_manager",
    "fault_tolerant",
    
    "RateLimiter",
    "RateLimiterConfig",
    "RateLimitStrategy",
    "RateLimitResult",
    "TokenBucketLimiter",
    "SlidingWindowLimiter",
    "rate_limiter_manager",
    "rate_limit",
    
    "BulkheadManager",
    "BulkheadConfig",
    "PoolPriority",
    "ResourcePool",
    "bulkhead_manager",
    "create_bulkhead_decorator",
    "BulkheadFullError",
    "BulkheadTimeoutError",
    
    "GracefulDegradationHandler",
    "FallbackConfig",
    "ServiceLevel",
    "DegradationMode",
    "degradation_handler",
    "graceful_degradation",
    "github_api_fallback",
    "database_fallback",
    "ml_inference_fallback",
    "monitoring_fallback",
    "FallbackFailedException",
    
    "ServiceMesh",
    "ServiceMeshConfig",
    "ServiceEndpoint",
    "LoadBalancingStrategy",
    "TrafficSplitStrategy",
    "ServiceHealthStatus",
    "RequestContext",
    "service_mesh_manager",
    "service_mesh_integration",
    "ServiceUnavailableError",
    "RequestFailedException",
    
    # Centralized management
    "ReliabilityManager",
    "reliability_manager",
    
    # Legacy components (if available)
    "ErrorRecoverySystem",
    "TimeoutManager",
]

# Remove None values from __all__
__all__ = [item for item in __all__ if globals().get(item) is not None]