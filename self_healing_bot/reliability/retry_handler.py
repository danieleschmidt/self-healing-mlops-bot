"""Enhanced retry handler with intelligent strategies and comprehensive fault tolerance."""

import asyncio
import logging
import random
import time
from typing import Callable, Any, Type, Tuple, Optional, Dict, List, Set, Union
from functools import wraps
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from statistics import mean

from .circuit_breaker import circuit_breaker_manager, CircuitBreakerOpenError, FailureType

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    POLYNOMIAL_BACKOFF = "polynomial_backoff"
    ADAPTIVE = "adaptive"


class FailureClassification(Enum):
    """Classification of failures for retry decisions."""
    TRANSIENT = "transient"  # Temporary failures that can be retried
    PERSISTENT = "persistent"  # Failures that are likely to persist
    FATAL = "fatal"  # Failures that should not be retried
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker is open
    RATE_LIMITED = "rate_limited"  # Rate limit exceeded
    TIMEOUT = "timeout"  # Operation timed out
    RESOURCE_EXHAUSTED = "resource_exhausted"  # No resources available
    AUTHENTICATION = "authentication"  # Authentication failed
    AUTHORIZATION = "authorization"  # Authorization failed
    VALIDATION = "validation"  # Input validation failed


@dataclass
class RetryConfig:
    """Enhanced configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add random jitter
    jitter_factor: float = 0.1  # Jitter factor (0.0 to 1.0)
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Enhanced retry configuration
    circuit_breaker_aware: bool = True  # Check circuit breaker state
    circuit_breaker_name: Optional[str] = None  # Circuit breaker to check
    adaptive_scaling: bool = False  # Adaptive delay scaling based on success rate
    
    # Failure classification
    retriable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retriable_exceptions: Tuple[Type[Exception], ...] = ()
    
    # Context-aware configuration
    context_key: Optional[str] = None  # Key for context-based configuration
    success_rate_threshold: float = 0.8  # Success rate threshold for adaptive scaling
    
    # Advanced features
    timeout_multiplier: float = 1.0  # Multiply timeout on each retry
    max_timeout: float = 300.0  # Maximum timeout
    backoff_cap_factor: float = 1.0  # Factor to cap exponential growth
    
    # Metrics and monitoring
    metrics_enabled: bool = True


@dataclass
class RetryMetrics:
    """Comprehensive retry metrics."""
    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    total_delay_time: float = 0.0
    average_attempts: float = 0.0
    success_rate: float = 100.0
    failure_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_updated: float = 0.0


@dataclass
class RetryContext:
    """Context for retry operations."""
    operation_name: str
    attempt_number: int
    total_attempts: int
    delay: float
    exception: Optional[Exception] = None
    start_time: float = 0.0
    elapsed_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetryHandler:
    """Enhanced retry handler with intelligent strategies and circuit breaker integration."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.metrics = RetryMetrics()
        self.execution_history = deque(maxlen=100)
        self.success_history = deque(maxlen=20)  # For adaptive scaling
        self.context_configs: Dict[str, RetryConfig] = {}
        
        # Fibonacci sequence for fibonacci backoff
        self.fibonacci_cache = [1, 1]
        
        # Adaptive scaling factors
        self.adaptive_factors: Dict[str, float] = defaultdict(lambda: 1.0)
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with enhanced retry logic."""
        operation_name = getattr(func, '__name__', 'unknown_operation')
        start_time = time.time()
        last_exception = None
        
        # Get context-specific configuration if available
        config = self._get_context_config(operation_name)
        
        # Check circuit breaker state if enabled
        if config.circuit_breaker_aware:
            circuit_breaker_name = config.circuit_breaker_name or operation_name
            breaker = circuit_breaker_manager.get_breaker(circuit_breaker_name)
            if breaker.state.value == "open":
                raise CircuitBreakerOpenError(f"Circuit breaker {circuit_breaker_name} is open")
        
        for attempt in range(config.max_retries + 1):
            try:
                result = await self._execute_with_timeout(func, attempt, config, *args, **kwargs)
                
                # Record successful execution
                execution_time = time.time() - start_time
                self._record_success(operation_name, attempt + 1, execution_time)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Classify the failure
                failure_class = self._classify_failure(e)
                
                # Record failure
                self._record_failure(operation_name, failure_class, attempt + 1)
                
                # Check if exception is retriable
                if not self._is_retriable(e, failure_class, config):
                    logger.error(f"Non-retriable exception in {operation_name}: {e}")
                    break
                
                # Don't retry on the last attempt
                if attempt < config.max_retries:
                    # Calculate delay with context
                    retry_context = RetryContext(
                        operation_name=operation_name,
                        attempt_number=attempt + 1,
                        total_attempts=config.max_retries + 1,
                        delay=0.0,
                        exception=e,
                        start_time=start_time,
                        elapsed_time=time.time() - start_time
                    )
                    
                    delay = self._calculate_delay(attempt, config, retry_context)
                    retry_context.delay = delay
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_retries + 1} failed for {operation_name}: {e}. "
                        f"Retrying in {delay:.2f}s (strategy: {config.strategy.value})"
                    )
                    
                    await asyncio.sleep(delay)
                    
                else:
                    logger.error(f"All {config.max_retries + 1} attempts failed for {operation_name}")
        
        # All retries exhausted
        self._record_exhaustion(operation_name, config.max_retries + 1)
        raise RetryExhaustedException(
            f"Max retries ({config.max_retries}) exceeded for {operation_name}"
        ) from last_exception
    
    def execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with enhanced retry logic."""
        operation_name = getattr(func, '__name__', 'unknown_operation')
        start_time = time.time()
        last_exception = None
        
        # Get context-specific configuration if available
        config = self._get_context_config(operation_name)
        
        # Check circuit breaker state if enabled
        if config.circuit_breaker_aware:
            circuit_breaker_name = config.circuit_breaker_name or operation_name
            breaker = circuit_breaker_manager.get_breaker(circuit_breaker_name)
            if breaker.state.value == "open":
                raise CircuitBreakerOpenError(f"Circuit breaker {circuit_breaker_name} is open")
        
        for attempt in range(config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # Record successful execution
                execution_time = time.time() - start_time
                self._record_success(operation_name, attempt + 1, execution_time)
                
                return result
                    
            except Exception as e:
                last_exception = e
                
                # Classify the failure
                failure_class = self._classify_failure(e)
                
                # Record failure
                self._record_failure(operation_name, failure_class, attempt + 1)
                
                # Check if exception is retriable
                if not self._is_retriable(e, failure_class, config):
                    logger.error(f"Non-retriable exception in {operation_name}: {e}")
                    break
                
                # Don't retry on the last attempt
                if attempt < config.max_retries:
                    # Calculate delay with context
                    retry_context = RetryContext(
                        operation_name=operation_name,
                        attempt_number=attempt + 1,
                        total_attempts=config.max_retries + 1,
                        delay=0.0,
                        exception=e,
                        start_time=start_time,
                        elapsed_time=time.time() - start_time
                    )
                    
                    delay = self._calculate_delay(attempt, config, retry_context)
                    retry_context.delay = delay
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_retries + 1} failed for {operation_name}: {e}. "
                        f"Retrying in {delay:.2f}s (strategy: {config.strategy.value})"
                    )
                    
                    time.sleep(delay)
                    
                else:
                    logger.error(f"All {config.max_retries + 1} attempts failed for {operation_name}")
        
        # All retries exhausted
        self._record_exhaustion(operation_name, config.max_retries + 1)
        raise RetryExhaustedException(
            f"Max retries ({config.max_retries}) exceeded for {operation_name}"
        ) from last_exception
    
    async def _execute_with_timeout(
        self, 
        func: Callable, 
        attempt: int, 
        config: RetryConfig, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute function with adaptive timeout."""
        timeout = getattr(func, '__timeout__', None)
        if timeout and config.timeout_multiplier > 1.0:
            # Increase timeout on retries
            adaptive_timeout = min(
                timeout * (config.timeout_multiplier ** attempt),
                config.max_timeout
            )
            
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(func(*args, **kwargs), timeout=adaptive_timeout)
            else:
                # For sync functions, we can't easily apply timeout
                return func(*args, **kwargs)
        else:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    def _classify_failure(self, exception: Exception) -> FailureClassification:
        """Classify failure type for retry decision."""
        exception_name = exception.__class__.__name__.lower()
        exception_str = str(exception).lower()
        
        # Circuit breaker failures
        if isinstance(exception, CircuitBreakerOpenError):
            return FailureClassification.CIRCUIT_OPEN
        
        # Timeout failures
        if 'timeout' in exception_name or 'timeout' in exception_str:
            return FailureClassification.TIMEOUT
        
        # Rate limiting
        if ('rate' in exception_str and 'limit' in exception_str) or \
           'ratelimit' in exception_name or \
           '429' in exception_str:
            return FailureClassification.RATE_LIMITED
        
        # Authentication/Authorization
        if 'auth' in exception_name or 'unauthorized' in exception_str or '401' in exception_str:
            return FailureClassification.AUTHENTICATION
        if 'forbidden' in exception_str or '403' in exception_str:
            return FailureClassification.AUTHORIZATION
        
        # Validation errors
        if 'validation' in exception_name or 'invalid' in exception_str or \
           '400' in exception_str or 'bad request' in exception_str:
            return FailureClassification.VALIDATION
        
        # Resource exhaustion
        if 'memory' in exception_str or 'resource' in exception_str or \
           'exhausted' in exception_str or '503' in exception_str:
            return FailureClassification.RESOURCE_EXHAUSTED
        
        # Network and connection errors (usually transient)
        if 'connection' in exception_name or 'network' in exception_str or \
           'socket' in exception_name or '500' in exception_str or \
           '502' in exception_str or '504' in exception_str:
            return FailureClassification.TRANSIENT
        
        # Default classification based on HTTP status codes
        if hasattr(exception, 'status_code'):
            status_code = getattr(exception, 'status_code')
            if 400 <= status_code < 500:
                if status_code in [408, 429]:  # Request timeout, too many requests
                    return FailureClassification.TRANSIENT
                else:
                    return FailureClassification.PERSISTENT
            elif 500 <= status_code < 600:
                return FailureClassification.TRANSIENT
        
        # Default to transient for unknown exceptions
        return FailureClassification.TRANSIENT
    
    def _is_retriable(
        self, 
        exception: Exception, 
        failure_class: FailureClassification, 
        config: RetryConfig
    ) -> bool:
        """Determine if an exception should be retried."""
        # Check non-retriable exceptions first
        for non_retriable in config.non_retriable_exceptions:
            if isinstance(exception, non_retriable):
                return False
        
        # Check retriable exceptions
        retriable_by_type = any(isinstance(exception, exc_type) for exc_type in config.retriable_exceptions)
        if not retriable_by_type:
            return False
        
        # Classification-based retry decisions
        non_retriable_classifications = {
            FailureClassification.FATAL,
            FailureClassification.AUTHENTICATION,
            FailureClassification.AUTHORIZATION,
            FailureClassification.VALIDATION
        }
        
        if failure_class in non_retriable_classifications:
            return False
        
        # Circuit breaker awareness
        if failure_class == FailureClassification.CIRCUIT_OPEN and config.circuit_breaker_aware:
            return False  # Let circuit breaker handle the retry timing
        
        return True
    
    def _calculate_delay(
        self, 
        attempt: int, 
        config: RetryConfig, 
        context: RetryContext
    ) -> float:
        """Calculate delay with intelligent strategy selection."""
        base_delay = config.base_delay
        
        # Apply adaptive scaling if enabled
        if config.adaptive_scaling:
            adaptive_factor = self._get_adaptive_factor(context.operation_name)
            base_delay *= adaptive_factor
        
        # Calculate delay based on strategy
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self._exponential_backoff(attempt, base_delay, config)
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self._linear_backoff(attempt, base_delay, config)
        elif config.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = base_delay
        elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self._fibonacci_backoff(attempt, base_delay, config)
        elif config.strategy == RetryStrategy.POLYNOMIAL_BACKOFF:
            delay = self._polynomial_backoff(attempt, base_delay, config)
        elif config.strategy == RetryStrategy.ADAPTIVE:
            delay = self._adaptive_backoff(attempt, context, config)
        else:
            delay = self._exponential_backoff(attempt, base_delay, config)
        
        # Apply jitter if enabled
        if config.jitter:
            jitter_amount = delay * config.jitter_factor
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        # Ensure delay is within bounds
        delay = max(0.1, min(delay, config.max_delay))
        
        return delay
    
    def _exponential_backoff(self, attempt: int, base_delay: float, config: RetryConfig) -> float:
        """Calculate exponential backoff delay."""
        delay = base_delay * (config.exponential_base ** attempt)
        
        # Apply backoff cap to prevent explosive growth
        if config.backoff_cap_factor < 1.0:
            max_uncapped_delay = base_delay * (config.exponential_base ** 5)  # Cap after 5 attempts
            if delay > max_uncapped_delay:
                delay = max_uncapped_delay * config.backoff_cap_factor
        
        return delay
    
    def _linear_backoff(self, attempt: int, base_delay: float, config: RetryConfig) -> float:
        """Calculate linear backoff delay."""
        return base_delay * (attempt + 1)
    
    def _fibonacci_backoff(self, attempt: int, base_delay: float, config: RetryConfig) -> float:
        """Calculate fibonacci backoff delay."""
        # Extend fibonacci sequence if needed
        while len(self.fibonacci_cache) <= attempt:
            next_fib = self.fibonacci_cache[-1] + self.fibonacci_cache[-2]
            self.fibonacci_cache.append(next_fib)
        
        return base_delay * self.fibonacci_cache[attempt]
    
    def _polynomial_backoff(self, attempt: int, base_delay: float, config: RetryConfig) -> float:
        """Calculate polynomial backoff delay."""
        return base_delay * ((attempt + 1) ** 2)
    
    def _adaptive_backoff(self, attempt: int, context: RetryContext, config: RetryConfig) -> float:
        """Calculate adaptive backoff based on context and history."""
        base_delay = config.base_delay
        
        # Factor in failure classification
        failure_class = self._classify_failure(context.exception) if context.exception else FailureClassification.TRANSIENT
        
        if failure_class == FailureClassification.RATE_LIMITED:
            # Longer delays for rate limiting
            return base_delay * (3 ** attempt)
        elif failure_class == FailureClassification.RESOURCE_EXHAUSTED:
            # Progressive delays for resource exhaustion
            return base_delay * (2 ** attempt) * 2
        elif failure_class == FailureClassification.TIMEOUT:
            # Moderate delays for timeouts
            return base_delay * (1.5 ** attempt)
        else:
            # Default exponential backoff
            return self._exponential_backoff(attempt, base_delay, config)
    
    def _get_adaptive_factor(self, operation_name: str) -> float:
        """Get adaptive scaling factor based on historical success rate."""
        success_rate = self._calculate_success_rate(operation_name)
        
        if success_rate < self.config.success_rate_threshold:
            # Increase delays when success rate is low
            factor = 2.0 - success_rate  # Scale between 1.0 and 2.0
        else:
            # Normal delays when success rate is good
            factor = 1.0
        
        self.adaptive_factors[operation_name] = factor
        return factor
    
    def _calculate_success_rate(self, operation_name: str) -> float:
        """Calculate recent success rate for an operation."""
        if not self.execution_history:
            return 1.0
        
        recent_executions = [
            entry for entry in self.execution_history
            if entry.get('operation_name') == operation_name
        ][-10:]  # Last 10 executions
        
        if not recent_executions:
            return 1.0
        
        successes = sum(1 for entry in recent_executions if entry.get('success', False))
        return successes / len(recent_executions)
    
    def _record_success(self, operation_name: str, attempts: int, execution_time: float):
        """Record successful execution."""
        self.metrics.total_attempts += attempts
        if attempts > 1:
            self.metrics.successful_retries += 1
        
        self.success_history.append(time.time())
        
        self.execution_history.append({
            'operation_name': operation_name,
            'attempts': attempts,
            'success': True,
            'execution_time': execution_time,
            'timestamp': time.time()
        })
        
        self._update_metrics()
    
    def _record_failure(self, operation_name: str, failure_class: FailureClassification, attempt: int):
        """Record failed attempt."""
        self.metrics.failure_types[failure_class.value] += 1
        
        self.execution_history.append({
            'operation_name': operation_name,
            'attempt': attempt,
            'failure_class': failure_class.value,
            'success': False,
            'timestamp': time.time()
        })
    
    def _record_exhaustion(self, operation_name: str, total_attempts: int):
        """Record retry exhaustion."""
        self.metrics.total_attempts += total_attempts
        self.metrics.failed_retries += 1
        
        self.execution_history.append({
            'operation_name': operation_name,
            'total_attempts': total_attempts,
            'success': False,
            'exhausted': True,
            'timestamp': time.time()
        })
        
        self._update_metrics()
    
    def _update_metrics(self):
        """Update comprehensive metrics."""
        current_time = time.time()
        self.metrics.last_updated = current_time
        
        # Calculate success rate
        total_operations = len([e for e in self.execution_history if e.get('success') is not None])
        if total_operations > 0:
            successes = len([e for e in self.execution_history if e.get('success', False)])
            self.metrics.success_rate = (successes / total_operations) * 100
        
        # Calculate average attempts
        attempts_data = [e.get('attempts', 1) for e in self.execution_history if e.get('attempts')]
        if attempts_data:
            self.metrics.average_attempts = mean(attempts_data)
    
    def _get_context_config(self, operation_name: str) -> RetryConfig:
        """Get context-specific configuration."""
        if self.config.context_key and operation_name in self.context_configs:
            return self.context_configs[operation_name]
        return self.config
    
    def set_context_config(self, operation_name: str, config: RetryConfig):
        """Set context-specific retry configuration."""
        self.context_configs[operation_name] = config
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive retry metrics."""
        return asdict(self.metrics)
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        operation_history = [
            entry for entry in self.execution_history
            if entry.get('operation_name') == operation_name
        ]
        
        if not operation_history:
            return {"operation_name": operation_name, "executions": 0}
        
        total_executions = len(operation_history)
        successful_executions = len([e for e in operation_history if e.get('success', False)])
        failed_executions = total_executions - successful_executions
        
        attempts_data = [e.get('attempts', 1) for e in operation_history if e.get('attempts')]
        avg_attempts = mean(attempts_data) if attempts_data else 1.0
        
        return {
            "operation_name": operation_name,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
            "average_attempts": avg_attempts,
            "adaptive_factor": self.adaptive_factors.get(operation_name, 1.0)
        }
    
    def reset_metrics(self):
        """Reset all metrics and history."""
        self.metrics = RetryMetrics()
        self.execution_history.clear()
        self.success_history.clear()
        self.adaptive_factors.clear()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to a function."""
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self.execute_async(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(self.execute_async(func, *args, **kwargs))
            else:
                return self.execute_sync(func, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts are exhausted."""
    pass


# Enhanced retry configurations for different scenarios
NETWORK_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    jitter=True,
    circuit_breaker_aware=True,
    adaptive_scaling=True,
    retriable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    ),
    non_retriable_exceptions=(
        ValueError,
        TypeError,
    )
)

DATABASE_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    base_delay=0.5,
    max_delay=10.0,
    strategy=RetryStrategy.LINEAR_BACKOFF,
    jitter=True,
    circuit_breaker_aware=True,
    retriable_exceptions=(
        ConnectionError,
        TimeoutError,
    )
)

GITHUB_API_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    exponential_base=1.5,
    jitter=True,
    circuit_breaker_aware=True,
    adaptive_scaling=True,
    retriable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    )
)

ML_INFERENCE_RETRY_CONFIG = RetryConfig(
    max_retries=4,
    base_delay=1.5,
    max_delay=120.0,
    strategy=RetryStrategy.ADAPTIVE,
    timeout_multiplier=1.2,
    max_timeout=300.0,
    jitter=True,
    circuit_breaker_aware=True,
    adaptive_scaling=True
)

CRITICAL_OPERATION_RETRY_CONFIG = RetryConfig(
    max_retries=7,
    base_delay=0.5,
    max_delay=30.0,
    strategy=RetryStrategy.FIBONACCI_BACKOFF,
    jitter=True,
    circuit_breaker_aware=True,
    adaptive_scaling=True,
    backoff_cap_factor=0.8
)


def retry(config: RetryConfig = None):
    """Decorator to add intelligent retry logic with custom configuration."""
    retry_handler = RetryHandler(config or RetryConfig())
    return retry_handler


# Convenience decorators for common scenarios
def retry_network(
    max_retries: int = 3, 
    base_delay: float = 1.0, 
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    circuit_breaker_aware: bool = True
):
    """Retry decorator for network operations with intelligent defaults."""
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        strategy=strategy,
        circuit_breaker_aware=circuit_breaker_aware,
        adaptive_scaling=True,
        retriable_exceptions=NETWORK_RETRY_CONFIG.retriable_exceptions
    )
    return retry(config)


def retry_database(
    max_retries: int = 2, 
    base_delay: float = 0.5,
    strategy: RetryStrategy = RetryStrategy.LINEAR_BACKOFF
):
    """Retry decorator for database operations."""
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        strategy=strategy,
        circuit_breaker_aware=True,
        retriable_exceptions=DATABASE_RETRY_CONFIG.retriable_exceptions
    )
    return retry(config)


def retry_github_api(
    max_retries: int = 5, 
    base_delay: float = 2.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
):
    """Retry decorator for GitHub API calls with rate limit awareness."""
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        strategy=strategy,
        exponential_base=1.5,
        circuit_breaker_aware=True,
        adaptive_scaling=True,
        retriable_exceptions=GITHUB_API_RETRY_CONFIG.retriable_exceptions
    )
    return retry(config)


def retry_ml_inference(
    max_retries: int = 4,
    base_delay: float = 1.5,
    strategy: RetryStrategy = RetryStrategy.ADAPTIVE
):
    """Retry decorator for ML inference operations with adaptive strategies."""
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        strategy=strategy,
        timeout_multiplier=1.2,
        max_timeout=300.0,
        circuit_breaker_aware=True,
        adaptive_scaling=True
    )
    return retry(config)


def retry_critical(
    max_retries: int = 7,
    base_delay: float = 0.5,
    strategy: RetryStrategy = RetryStrategy.FIBONACCI_BACKOFF
):
    """Retry decorator for critical operations with aggressive retry policy."""
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        strategy=strategy,
        circuit_breaker_aware=True,
        adaptive_scaling=True,
        backoff_cap_factor=0.8
    )
    return retry(config)


# Global retry handler instances
network_retry = RetryHandler(NETWORK_RETRY_CONFIG)
database_retry = RetryHandler(DATABASE_RETRY_CONFIG)
github_api_retry = RetryHandler(GITHUB_API_RETRY_CONFIG)
ml_inference_retry = RetryHandler(ML_INFERENCE_RETRY_CONFIG)
critical_operation_retry = RetryHandler(CRITICAL_OPERATION_RETRY_CONFIG)


class RetryManager:
    """Manager for multiple retry handlers with global configuration."""
    
    def __init__(self):
        self.handlers: Dict[str, RetryHandler] = {
            "network": network_retry,
            "database": database_retry,
            "github_api": github_api_retry,
            "ml_inference": ml_inference_retry,
            "critical": critical_operation_retry
        }
    
    def get_handler(self, name: str) -> Optional[RetryHandler]:
        """Get a retry handler by name."""
        return self.handlers.get(name)
    
    def register_handler(self, name: str, handler: RetryHandler):
        """Register a custom retry handler."""
        self.handlers[name] = handler
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get metrics from all retry handlers."""
        return {
            name: handler.get_metrics()
            for name, handler in self.handlers.items()
        }
    
    def reset_all_metrics(self):
        """Reset metrics for all handlers."""
        for handler in self.handlers.values():
            handler.reset_metrics()


# Global retry manager instance
retry_manager = RetryManager()