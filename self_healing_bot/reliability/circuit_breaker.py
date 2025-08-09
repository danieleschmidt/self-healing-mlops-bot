"""Enhanced circuit breaker pattern implementation for fault tolerance."""

import asyncio
import json
import time
import logging
from typing import Callable, Any, Optional, Dict, List, Set, Union
from enum import Enum
from dataclasses import dataclass, asdict
from functools import wraps
from pathlib import Path
from collections import defaultdict, deque
from statistics import mean

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Test if service recovered


class FailureType(Enum):
    """Types of failures for classification."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_ERROR = "security_error"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    """Enhanced configuration for circuit breaker."""
    # Basic configuration
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before half-open attempt
    success_threshold: int = 2  # Successes to close from half-open
    timeout: int = 30  # Operation timeout in seconds
    
    # Advanced configuration
    adaptive_thresholds: bool = True  # Enable adaptive threshold adjustment
    failure_rate_threshold: float = 0.5  # Failure rate to trigger opening
    minimum_requests: int = 10  # Minimum requests before considering failure rate
    sliding_window_size: int = 20  # Size of sliding window for metrics
    
    # Failure classification
    critical_failure_types: Set[FailureType] = None  # Failures that immediately open circuit
    retriable_failure_types: Set[FailureType] = None  # Failures that count towards threshold
    
    # Monitoring and persistence
    metrics_enabled: bool = True  # Enable detailed metrics collection
    state_persistence_enabled: bool = False  # Enable state persistence
    state_file_path: Optional[str] = None  # Path for state persistence
    
    # Health check integration
    health_check_enabled: bool = True  # Enable health checks during half-open
    health_check_interval: int = 10  # Seconds between health checks
    
    def __post_init__(self):
        if self.critical_failure_types is None:
            self.critical_failure_types = {
                FailureType.SERVICE_UNAVAILABLE,
                FailureType.RESOURCE_EXHAUSTION
            }
        
        if self.retriable_failure_types is None:
            self.retriable_failure_types = {
                FailureType.TIMEOUT,
                FailureType.CONNECTION_ERROR,
                FailureType.SERVER_ERROR,
                FailureType.RATE_LIMIT
            }


@dataclass
class CircuitBreakerMetrics:
    """Comprehensive metrics for circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_opened_count: int = 0
    circuit_closed_count: int = 0
    half_open_attempts: int = 0
    avg_response_time: float = 0.0
    failure_rate: float = 0.0
    uptime_percentage: float = 100.0
    failure_types: Dict[str, int] = None
    
    def __post_init__(self):
        if self.failure_types is None:
            self.failure_types = defaultdict(int)


class CircuitBreaker:
    """Enhanced circuit breaker implementation for service fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self.created_time = time.time()
        
        # Enhanced features
        self.metrics = CircuitBreakerMetrics()
        self.sliding_window = deque(maxlen=self.config.sliding_window_size)
        self.response_times = deque(maxlen=self.config.sliding_window_size)
        self.failure_types_history = deque(maxlen=100)
        self.state_changes = deque(maxlen=50)  # Track state changes
        self.adaptive_threshold = self.config.failure_threshold
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Load persisted state if enabled
        if self.config.state_persistence_enabled:
            self._load_persisted_state()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        
        @wraps(func) 
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(self.call(func, *args, **kwargs))
            else:
                return self.call_sync(func, *args, **kwargs)
                
        return wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection (async)."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Next retry in {self._time_until_retry():.1f} seconds"
                )
        
        start_time = time.time()
        failure_type = FailureType.UNKNOWN
        
        try:
            # Execute the function with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=self.config.timeout
                )
            else:
                result = func(*args, **kwargs)
            
            response_time = time.time() - start_time
            self._on_success(response_time)
            return result
            
        except asyncio.TimeoutError as e:
            failure_type = FailureType.TIMEOUT
            self._on_failure(failure_type, e)
            raise CircuitBreakerTimeoutError(
                f"Operation timed out after {self.config.timeout}s"
            ) from e
            
        except ConnectionError as e:
            failure_type = FailureType.CONNECTION_ERROR
            self._on_failure(failure_type, e)
            raise
            
        except Exception as e:
            # Classify the exception
            failure_type = self._classify_exception(e)
            self._on_failure(failure_type, e)
            raise
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection (sync)."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Next retry in {self._time_until_retry():.1f} seconds"
                )
        
        start_time = time.time()
        failure_type = FailureType.UNKNOWN
        
        try:
            result = func(*args, **kwargs)
            response_time = time.time() - start_time
            self._on_success(response_time)
            return result
            
        except Exception as e:
            # Classify the exception
            failure_type = self._classify_exception(e)
            self._on_failure(failure_type, e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self, response_time: float = 0.0):
        """Handle successful operation with enhanced metrics."""
        current_time = time.time()
        self.failure_count = 0
        self.last_success_time = current_time
        
        # Update metrics
        self.metrics.successful_requests += 1
        self.metrics.total_requests += 1
        
        # Record success in sliding window
        self.sliding_window.append((current_time, True, None))
        if response_time > 0:
            self.response_times.append(response_time)
            self._update_average_response_time()
        
        # Handle half-open to closed transition
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit("Success threshold reached")
    
    def _on_failure(self, failure_type: FailureType = FailureType.UNKNOWN, error: Exception = None):
        """Handle failed operation with enhanced failure classification."""
        current_time = time.time()
        self.failure_count += 1
        self.last_failure_time = current_time
        
        # Update metrics
        self.metrics.failed_requests += 1
        self.metrics.total_requests += 1
        self.metrics.failure_types[failure_type.value] += 1
        
        # Record failure in sliding window
        self.sliding_window.append((current_time, False, failure_type))
        self.failure_types_history.append((current_time, failure_type, str(error) if error else None))
        
        # Check for critical failures that immediately open circuit
        if failure_type in self.config.critical_failure_types:
            self._open_circuit(f"Critical failure: {failure_type.value}")
            return
        
        # Update adaptive threshold
        if self.config.adaptive_thresholds:
            self._update_adaptive_threshold()
        
        # Check if should open based on failure count or rate
        should_open = (
            self.failure_count >= self.adaptive_threshold or
            self._should_open_by_failure_rate()
        )
        
        if should_open and self.state == CircuitState.CLOSED:
            self._open_circuit(f"Failure threshold exceeded: {self.failure_count} failures")
        
        # Handle half-open state failures
        if self.state == CircuitState.HALF_OPEN:
            self._open_circuit("Failure in HALF_OPEN state")
            self.success_count = 0
    
    def _open_circuit(self, reason: str):
        """Open the circuit breaker."""
        if self.state != CircuitState.OPEN:
            old_state = self.state
            self.state = CircuitState.OPEN
            self.metrics.circuit_opened_count += 1
            
            # Record state change
            self.state_changes.append({
                "timestamp": time.time(),
                "from_state": old_state.value,
                "to_state": CircuitState.OPEN.value,
                "reason": reason
            })
            
            logger.warning(
                f"Circuit breaker '{self.name}' OPENED: {reason}",
                extra={
                    "circuit_breaker": self.name,
                    "failure_count": self.failure_count,
                    "failure_rate": self.metrics.failure_rate,
                    "reason": reason
                }
            )
            
            # Start health checking if enabled
            if self.config.health_check_enabled:
                self._start_health_checking()
            
            # Persist state if enabled
            if self.config.state_persistence_enabled:
                self._persist_state()
    
    def _close_circuit(self, reason: str):
        """Close the circuit breaker."""
        if self.state != CircuitState.CLOSED:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.success_count = 0
            self.metrics.circuit_closed_count += 1
            
            # Record state change
            self.state_changes.append({
                "timestamp": time.time(),
                "from_state": old_state.value,
                "to_state": CircuitState.CLOSED.value,
                "reason": reason
            })
            
            logger.info(
                f"Circuit breaker '{self.name}' CLOSED: {reason}",
                extra={
                    "circuit_breaker": self.name,
                    "success_count": self.success_count,
                    "reason": reason
                }
            )
            
            # Stop health checking
            self._stop_health_checking()
            
            # Persist state if enabled
            if self.config.state_persistence_enabled:
                self._persist_state()
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.metrics.half_open_attempts += 1
        
        # Record state change
        self.state_changes.append({
            "timestamp": time.time(),
            "from_state": old_state.value,
            "to_state": CircuitState.HALF_OPEN.value,
            "reason": "Recovery timeout elapsed"
        })
        
        logger.info(
            f"Circuit breaker '{self.name}' entering HALF_OPEN state",
            extra={"circuit_breaker": self.name}
        )
    
    def _classify_exception(self, exception: Exception) -> FailureType:
        """Classify exception to determine failure type."""
        exception_name = exception.__class__.__name__.lower()
        
        if 'timeout' in exception_name:
            return FailureType.TIMEOUT
        elif 'connection' in exception_name:
            return FailureType.CONNECTION_ERROR
        elif 'ratelimit' in exception_name or 'rate_limit' in exception_name:
            return FailureType.RATE_LIMIT
        elif hasattr(exception, 'status_code'):
            status_code = getattr(exception, 'status_code')
            if 500 <= status_code < 600:
                return FailureType.SERVER_ERROR
            elif status_code == 503:
                return FailureType.SERVICE_UNAVAILABLE
            elif 400 <= status_code < 500:
                return FailureType.CLIENT_ERROR
        elif 'validation' in exception_name or 'invalid' in exception_name:
            return FailureType.VALIDATION_ERROR
        elif 'memory' in str(exception).lower() or 'resource' in str(exception).lower():
            return FailureType.RESOURCE_EXHAUSTION
        elif 'security' in exception_name or 'auth' in exception_name:
            return FailureType.SECURITY_ERROR
        
        return FailureType.UNKNOWN
    
    def _should_open_by_failure_rate(self) -> bool:
        """Check if circuit should open based on failure rate."""
        if len(self.sliding_window) < self.config.minimum_requests:
            return False
        
        current_time = time.time()
        recent_window = [entry for entry in self.sliding_window 
                        if current_time - entry[0] <= 60]  # Last minute
        
        if len(recent_window) < self.config.minimum_requests:
            return False
        
        failures = sum(1 for _, success, _ in recent_window if not success)
        failure_rate = failures / len(recent_window)
        
        return failure_rate >= self.config.failure_rate_threshold
    
    def _update_adaptive_threshold(self):
        """Update adaptive failure threshold based on historical performance."""
        if not self.config.adaptive_thresholds or len(self.sliding_window) < 10:
            return
        
        # Analyze recent performance trends
        recent_failures = [entry for entry in self.sliding_window 
                          if not entry[1]]  # Get failures
        
        if len(recent_failures) > 0:
            # Increase sensitivity if seeing sustained failures
            sustained_failure_count = len([f for f in recent_failures[-5:] if not f[1]])
            if sustained_failure_count >= 3:
                self.adaptive_threshold = max(2, self.adaptive_threshold - 1)
            else:
                # Gradually return to normal threshold
                self.adaptive_threshold = min(
                    self.config.failure_threshold,
                    self.adaptive_threshold + 1
                )
    
    def _time_until_retry(self) -> float:
        """Calculate time until next retry attempt."""
        if self.state != CircuitState.OPEN:
            return 0.0
        
        elapsed = time.time() - self.last_failure_time
        return max(0.0, self.config.recovery_timeout - elapsed)
    
    def _update_failure_rate(self):
        """Update current failure rate metric."""
        if self.metrics.total_requests == 0:
            self.metrics.failure_rate = 0.0
        else:
            self.metrics.failure_rate = (
                self.metrics.failed_requests / self.metrics.total_requests
            )
    
    def _update_average_response_time(self):
        """Update average response time metric."""
        if self.response_times:
            self.metrics.avg_response_time = mean(self.response_times)
    
    def _get_metrics_dict(self) -> dict:
        """Get metrics as dictionary."""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "failure_rate": self.metrics.failure_rate,
            "circuit_opened_count": self.metrics.circuit_opened_count,
            "circuit_closed_count": self.metrics.circuit_closed_count,
            "half_open_attempts": self.metrics.half_open_attempts,
            "avg_response_time_ms": self.metrics.avg_response_time * 1000,
            "failure_types": dict(self.metrics.failure_types)
        }
    
    def get_state(self) -> dict:
        """Get comprehensive current circuit breaker state."""
        current_time = time.time()
        uptime = current_time - self.created_time
        
        # Calculate current failure rate
        self._update_failure_rate()
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "time_until_retry": self._time_until_retry() if self.state == CircuitState.OPEN else 0,
            "uptime_seconds": uptime,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "adaptive_threshold": self.adaptive_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            },
            "metrics": self._get_metrics_dict(),
            "recent_failures": list(self.failure_types_history)[-10:],
            "state_changes": list(self.state_changes)[-10:]
        }
    
    def reset(self):
        """Manually reset circuit breaker to closed state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        
        # Record manual reset
        self.state_changes.append({
            "timestamp": time.time(),
            "from_state": old_state.value,
            "to_state": CircuitState.CLOSED.value,
            "reason": "Manual reset"
        })
        
        # Stop health checking
        self._stop_health_checking()
        
        logger.info(
            f"Circuit breaker '{self.name}' manually reset to CLOSED state",
            extra={"circuit_breaker": self.name, "previous_state": old_state.value}
        )
        
        # Persist state if enabled
        if self.config.state_persistence_enabled:
            self._persist_state()
    
    async def _start_health_checking(self):
        """Start periodic health checking during open state."""
        if not self.config.health_check_enabled or self.health_check_task:
            return
        
        self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    def _stop_health_checking(self):
        """Stop health checking."""
        if self.health_check_task:
            self.health_check_task.cancel()
            self.health_check_task = None
    
    async def _health_check_loop(self):
        """Periodic health checking loop."""
        try:
            while self.state == CircuitState.OPEN:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check if recovery timeout has elapsed
                if self._should_attempt_reset():
                    logger.info(
                        f"Circuit breaker '{self.name}' attempting recovery",
                        extra={"circuit_breaker": self.name}
                    )
                    break
        except asyncio.CancelledError:
            pass
    
    def _persist_state(self):
        """Persist circuit breaker state to disk."""
        if not self.config.state_file_path:
            return
        
        try:
            state_data = {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "last_success_time": self.last_success_time,
                "adaptive_threshold": self.adaptive_threshold,
                "metrics": asdict(self.metrics),
                "timestamp": time.time()
            }
            
            Path(self.config.state_file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.state_file_path, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.warning(
                f"Failed to persist circuit breaker state: {e}",
                extra={"circuit_breaker": self.name}
            )
    
    def _load_persisted_state(self):
        """Load persisted circuit breaker state from disk."""
        if not self.config.state_file_path or not Path(self.config.state_file_path).exists():
            return
        
        try:
            with open(self.config.state_file_path, 'r') as f:
                state_data = json.load(f)
            
            # Only restore if state is recent (within 1 hour)
            if time.time() - state_data.get('timestamp', 0) > 3600:
                return
            
            self.state = CircuitState(state_data.get('state', CircuitState.CLOSED.value))
            self.failure_count = state_data.get('failure_count', 0)
            self.success_count = state_data.get('success_count', 0)
            self.last_failure_time = state_data.get('last_failure_time', 0)
            self.last_success_time = state_data.get('last_success_time', 0)
            self.adaptive_threshold = state_data.get('adaptive_threshold', self.config.failure_threshold)
            
            logger.info(
                f"Restored circuit breaker '{self.name}' state: {self.state.value}",
                extra={"circuit_breaker": self.name}
            )
            
        except Exception as e:
            logger.warning(
                f"Failed to load persisted circuit breaker state: {e}",
                extra={"circuit_breaker": self.name}
            )


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerTimeoutError(Exception):
    """Exception raised when operation times out."""
    pass


class CircuitBreakerManager:
    """Enhanced manager for multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._global_metrics = {
            "total_breakers": 0,
            "open_breakers": 0,
            "half_open_breakers": 0,
            "closed_breakers": 0
        }
    
    def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
            self._global_metrics["total_breakers"] += 1
        return self._breakers[name]
    
    def remove_breaker(self, name: str) -> bool:
        """Remove a circuit breaker."""
        if name in self._breakers:
            breaker = self._breakers[name]
            breaker._stop_health_checking()
            del self._breakers[name]
            self._global_metrics["total_breakers"] -= 1
            return True
        return False
    
    def get_all_states(self) -> Dict[str, dict]:
        """Get state of all circuit breakers."""
        return {name: breaker.get_state() for name, breaker in self._breakers.items()}
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global circuit breaker metrics."""
        # Update state counts
        self._global_metrics.update({
            "open_breakers": sum(1 for b in self._breakers.values() if b.state == CircuitState.OPEN),
            "half_open_breakers": sum(1 for b in self._breakers.values() if b.state == CircuitState.HALF_OPEN),
            "closed_breakers": sum(1 for b in self._breakers.values() if b.state == CircuitState.CLOSED),
            "total_requests": sum(b.metrics.total_requests for b in self._breakers.values()),
            "total_failures": sum(b.metrics.failed_requests for b in self._breakers.values()),
            "average_failure_rate": mean([b.metrics.failure_rate for b in self._breakers.values()]) if self._breakers else 0.0
        })
        
        return self._global_metrics
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")
    
    def get_unhealthy_breakers(self) -> List[CircuitBreaker]:
        """Get list of unhealthy (open or half-open) circuit breakers."""
        return [
            breaker for breaker in self._breakers.values()
            if breaker.state in [CircuitState.OPEN, CircuitState.HALF_OPEN]
        ]
    
    def get_breaker_by_failure_rate(self, min_failure_rate: float = 0.1) -> List[CircuitBreaker]:
        """Get breakers with failure rate above threshold."""
        return [
            breaker for breaker in self._breakers.values()
            if breaker.metrics.failure_rate >= min_failure_rate
        ]


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator to add circuit breaker protection to a function."""
    def decorator(func: Callable) -> Callable:
        breaker = circuit_breaker_manager.get_breaker(name, config)
        return breaker(func)
    return decorator


# Enhanced predefined circuit breakers for common services
@circuit_breaker(
    "github_api", 
    CircuitBreakerConfig(
        failure_threshold=3, 
        recovery_timeout=30,
        adaptive_thresholds=True,
        failure_rate_threshold=0.3,
        metrics_enabled=True
    )
)
async def github_api_call(func: Callable, *args, **kwargs):
    """GitHub API call with enhanced circuit breaker protection."""
    return await func(*args, **kwargs)


@circuit_breaker(
    "database", 
    CircuitBreakerConfig(
        failure_threshold=5, 
        recovery_timeout=60,
        adaptive_thresholds=True,
        critical_failure_types={FailureType.RESOURCE_EXHAUSTION, FailureType.CONNECTION_ERROR}
    )
)
async def database_operation(func: Callable, *args, **kwargs):
    """Database operation with enhanced circuit breaker protection."""
    return await func(*args, **kwargs)


@circuit_breaker(
    "webhook_processing", 
    CircuitBreakerConfig(
        failure_threshold=10, 
        recovery_timeout=120,
        sliding_window_size=50,
        failure_rate_threshold=0.4
    )
)
async def webhook_process(func: Callable, *args, **kwargs):
    """Webhook processing with enhanced circuit breaker protection."""
    return await func(*args, **kwargs)


@circuit_breaker(
    "ml_inference", 
    CircuitBreakerConfig(
        failure_threshold=8, 
        recovery_timeout=90,
        timeout=120,  # ML operations can take longer
        adaptive_thresholds=True,
        critical_failure_types={FailureType.RESOURCE_EXHAUSTION, FailureType.TIMEOUT}
    )
)
async def ml_inference_call(func: Callable, *args, **kwargs):
    """ML inference call with circuit breaker protection."""
    return await func(*args, **kwargs)