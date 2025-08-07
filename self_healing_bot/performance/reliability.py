"""Reliability and resilience features for the bot."""

import asyncio
import time
from typing import Callable, Any, Dict, Optional, List
from functools import wraps
from dataclasses import dataclass
from enum import Enum

from ..monitoring.logging import get_logger, performance_logger
from ..core.errors import error_handler, circuit_breakers

logger = get_logger(__name__)


class RetryStrategy(Enum):
    """Retry strategies."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIXED = "fixed"


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0


def retry_with_backoff(config: RetryConfig = None):
    """Decorator for adding retry logic with exponential backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    
                    # Log successful execution after retry
                    if attempt > 0:
                        duration = time.time() - start_time
                        performance_logger.log_execution_time(
                            operation=func.__name__,
                            duration=duration,
                            success=True,
                            retry_attempt=attempt + 1
                        )
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry
                    if not error_handler.should_retry(e, attempt, config.max_attempts):
                        break
                    
                    if attempt < config.max_attempts - 1:
                        # Calculate delay
                        if config.strategy == RetryStrategy.EXPONENTIAL:
                            delay = min(
                                config.base_delay * (config.backoff_multiplier ** attempt),
                                config.max_delay
                            )
                        elif config.strategy == RetryStrategy.LINEAR:
                            delay = min(
                                config.base_delay + (attempt * config.base_delay),
                                config.max_delay
                            )
                        else:  # FIXED
                            delay = config.base_delay
                        
                        logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s",
                            function=func.__name__,
                            attempt=attempt + 1,
                            delay=delay,
                            error=str(e)
                        )
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed",
                            function=func.__name__,
                            final_error=str(e)
                        )
            
            # All attempts failed
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in asyncio if needed
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def with_circuit_breaker(breaker_name: str, fallback: Callable = None):
    """Decorator for adding circuit breaker protection."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            breaker = circuit_breakers.get(breaker_name)
            if not breaker:
                # No circuit breaker configured, execute normally
                return await func(*args, **kwargs)
            
            if not breaker.can_execute():
                logger.warning(
                    f"Circuit breaker {breaker_name} is OPEN, blocking execution",
                    function=func.__name__,
                    breaker_state=breaker.state
                )
                
                if fallback:
                    logger.info(f"Using fallback for {func.__name__}")
                    return await fallback(*args, **kwargs)
                else:
                    raise Exception(f"Circuit breaker {breaker_name} is open")
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                logger.error(
                    f"Circuit breaker {breaker_name} recorded failure",
                    function=func.__name__,
                    error=str(e),
                    failure_count=breaker.failure_count
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else func
    
    return decorator


def with_timeout(seconds: float):
    """Decorator for adding timeout protection."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(
                    f"Function {func.__name__} timed out after {seconds}s",
                    function=func.__name__,
                    timeout=seconds
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else func
    
    return decorator


class HealthyExecutor:
    """Executor that tracks component health and implements reliability patterns."""
    
    def __init__(self, name: str):
        self.name = name
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0,
            "last_execution": None
        }
    
    async def execute(
        self,
        func: Callable,
        *args,
        retry_config: RetryConfig = None,
        timeout: float = None,
        circuit_breaker: str = None,
        **kwargs
    ) -> Any:
        """Execute function with reliability patterns."""
        start_time = time.time()
        
        try:
            # Apply decorators based on configuration
            execution_func = func
            
            if circuit_breaker:
                execution_func = with_circuit_breaker(circuit_breaker)(execution_func)
            
            if timeout:
                execution_func = with_timeout(timeout)(execution_func)
            
            if retry_config:
                execution_func = retry_with_backoff(retry_config)(execution_func)
            
            # Execute the function
            if asyncio.iscoroutinefunction(execution_func):
                result = await execution_func(*args, **kwargs)
            else:
                result = execution_func(*args, **kwargs)
            
            # Record success
            duration = time.time() - start_time
            self._record_execution(True, duration)
            
            return result
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            self._record_execution(False, duration)
            
            # Handle error
            error_handler.handle_error(e, context={
                "executor": self.name,
                "function": func.__name__,
                "duration": duration
            }, component=self.name)
            
            raise
    
    def _record_execution(self, success: bool, duration: float):
        """Record execution statistics."""
        self.stats["total_executions"] += 1
        self.stats["last_execution"] = time.time()
        
        if success:
            self.stats["successful_executions"] += 1
        else:
            self.stats["failed_executions"] += 1
        
        # Update running average duration
        total = self.stats["total_executions"]
        old_avg = self.stats["average_duration"]
        self.stats["average_duration"] = ((old_avg * (total - 1)) + duration) / total
    
    def get_health_info(self) -> Dict[str, Any]:
        """Get health information for this executor."""
        total = self.stats["total_executions"]
        if total == 0:
            success_rate = 0.0
        else:
            success_rate = self.stats["successful_executions"] / total
        
        return {
            "name": self.name,
            "success_rate": round(success_rate, 4),
            "total_executions": total,
            "average_duration_ms": round(self.stats["average_duration"] * 1000, 2),
            "last_execution_ago_seconds": (
                round(time.time() - self.stats["last_execution"], 2)
                if self.stats["last_execution"] else None
            )
        }


class BulkheadIsolation:
    """Implement bulkhead pattern for resource isolation."""
    
    def __init__(self):
        self.resource_pools = {}
    
    def create_pool(self, name: str, max_concurrent: int = 5):
        """Create a resource pool with limited concurrency."""
        self.resource_pools[name] = asyncio.Semaphore(max_concurrent)
        logger.info(f"Created resource pool '{name}' with {max_concurrent} slots")
    
    async def execute_in_pool(self, pool_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function within a resource pool."""
        if pool_name not in self.resource_pools:
            self.create_pool(pool_name)
        
        semaphore = self.resource_pools[pool_name]
        
        async with semaphore:
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                performance_logger.log_execution_time(
                    operation=func.__name__,
                    duration=duration,
                    success=True,
                    resource_pool=pool_name
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                performance_logger.log_execution_time(
                    operation=func.__name__,
                    duration=duration,
                    success=False,
                    resource_pool=pool_name,
                    error=str(e)
                )
                raise
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get status of all resource pools."""
        status = {}
        for name, semaphore in self.resource_pools.items():
            status[name] = {
                "available_slots": semaphore._value,
                "waiting_tasks": len(semaphore._waiters) if hasattr(semaphore, '_waiters') else 0
            }
        return status


# Global instances
executors = {
    "github_api": HealthyExecutor("github_api"),
    "detectors": HealthyExecutor("detectors"),
    "playbooks": HealthyExecutor("playbooks"),
    "notifications": HealthyExecutor("notifications")
}

bulkhead = BulkheadIsolation()

# Initialize resource pools
bulkhead.create_pool("github_api_requests", max_concurrent=10)
bulkhead.create_pool("detector_execution", max_concurrent=5)
bulkhead.create_pool("playbook_execution", max_concurrent=3)
bulkhead.create_pool("file_operations", max_concurrent=5)