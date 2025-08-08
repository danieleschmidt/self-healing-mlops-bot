"""Retry handler with exponential backoff and jitter."""

import asyncio
import logging
import random
import time
from typing import Callable, Any, Type, Tuple, Optional
from functools import wraps
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add random jitter
    retriable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


class RetryHandler:
    """Handle retries with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                # Check if exception is retriable
                if not self._is_retriable(e):
                    logger.error(f"Non-retriable exception: {e}")
                    raise e
                
                # Don't sleep on the last attempt
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries + 1} attempts failed")
        
        # All retries exhausted
        raise RetryExhaustedException(
            f"Max retries ({self.config.max_retries}) exceeded"
        ) from last_exception
    
    def execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                # Check if exception is retriable
                if not self._is_retriable(e):
                    logger.error(f"Non-retriable exception: {e}")
                    raise e
                
                # Don't sleep on the last attempt
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries + 1} attempts failed")
        
        # All retries exhausted
        raise RetryExhaustedException(
            f"Max retries ({self.config.max_retries}) exceeded"
        ) from last_exception
    
    def _is_retriable(self, exception: Exception) -> bool:
        """Check if exception is retriable."""
        return any(isinstance(exception, exc_type) 
                  for exc_type in self.config.retriable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        # Exponential backoff
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay
    
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


# Common retry configurations
NETWORK_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    retriable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    )
)

DATABASE_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    base_delay=0.5,
    max_delay=10.0,
    retriable_exceptions=(
        ConnectionError,
        TimeoutError,
    )
)

GITHUB_API_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
    retriable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    )
)


def retry(config: RetryConfig = None):
    """Decorator to add retry logic with custom configuration."""
    retry_handler = RetryHandler(config or RetryConfig())
    return retry_handler


# Convenience decorators for common scenarios
def retry_network(max_retries: int = 3, base_delay: float = 1.0):
    """Retry decorator for network operations."""
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        retriable_exceptions=NETWORK_RETRY_CONFIG.retriable_exceptions
    )
    return retry(config)


def retry_database(max_retries: int = 2, base_delay: float = 0.5):
    """Retry decorator for database operations."""
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        retriable_exceptions=DATABASE_RETRY_CONFIG.retriable_exceptions
    )
    return retry(config)


def retry_github_api(max_retries: int = 3, base_delay: float = 2.0):
    """Retry decorator for GitHub API calls."""
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        retriable_exceptions=GITHUB_API_RETRY_CONFIG.retriable_exceptions
    )
    return retry(config)


# Global retry handler instances
network_retry = RetryHandler(NETWORK_RETRY_CONFIG)
database_retry = RetryHandler(DATABASE_RETRY_CONFIG)
github_api_retry = RetryHandler(GITHUB_API_RETRY_CONFIG)