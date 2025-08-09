"""Enterprise-grade rate limiter with token bucket algorithm, burst handling, and adaptive limits."""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from statistics import mean
import hashlib
import json

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


class RateLimitScope(Enum):
    """Rate limiting scopes."""
    GLOBAL = "global"
    PER_CLIENT = "per_client"
    PER_ENDPOINT = "per_endpoint"
    PER_USER = "per_user"
    CUSTOM = "custom"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    name: str
    requests_per_second: float = 10.0
    burst_size: Optional[int] = None  # If None, defaults to requests_per_second
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    scope: RateLimitScope = RateLimitScope.GLOBAL
    
    # Advanced configuration
    adaptive_scaling: bool = False
    min_rate: float = 1.0  # Minimum rate for adaptive scaling
    max_rate: float = 100.0  # Maximum rate for adaptive scaling
    
    # Window configuration (for sliding/fixed window)
    window_size: int = 60  # Window size in seconds
    
    # Burst handling
    burst_penalty_factor: float = 1.5  # Penalty factor for burst usage
    burst_recovery_time: int = 300  # Time to recover from burst penalty
    
    # Distribution and clustering
    distributed: bool = False
    redis_key_prefix: str = "rate_limit"
    
    # Monitoring
    metrics_enabled: bool = True
    alert_on_limit_exceeded: bool = True
    
    def __post_init__(self):
        """Initialize derived values."""
        if self.burst_size is None:
            self.burst_size = max(1, int(self.requests_per_second))


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, rate: float, burst_size: int):
        self.rate = rate  # tokens per second
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self.total_requests = 0
        self.denied_requests = 0
    
    def consume(self, tokens: int = 1) -> RateLimitResult:
        """Consume tokens from the bucket."""
        current_time = time.time()
        
        # Add tokens based on elapsed time
        elapsed = current_time - self.last_update
        self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
        self.last_update = current_time
        
        self.total_requests += 1
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            remaining = int(self.tokens)
            
            # Calculate reset time (when bucket will be full again)
            time_to_full = (self.burst_size - self.tokens) / self.rate
            reset_time = current_time + time_to_full
            
            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                reset_time=reset_time,
                metadata={
                    "strategy": "token_bucket",
                    "current_tokens": self.tokens,
                    "rate": self.rate,
                    "burst_size": self.burst_size
                }
            )
        else:
            self.denied_requests += 1
            
            # Calculate retry after time
            tokens_needed = tokens - self.tokens
            retry_after = tokens_needed / self.rate
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=current_time + retry_after,
                retry_after=retry_after,
                reason="Rate limit exceeded",
                metadata={
                    "strategy": "token_bucket",
                    "tokens_needed": tokens_needed,
                    "current_tokens": self.tokens
                }
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bucket statistics."""
        total = self.total_requests
        success_rate = ((total - self.denied_requests) / total * 100) if total > 0 else 100.0
        
        return {
            "current_tokens": self.tokens,
            "rate": self.rate,
            "burst_size": self.burst_size,
            "total_requests": total,
            "denied_requests": self.denied_requests,
            "success_rate": success_rate,
            "last_update": self.last_update
        }


class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation."""
    
    def __init__(self, rate: float, window_size: int = 60):
        self.rate = rate
        self.window_size = window_size
        self.requests = deque()
        self.total_requests = 0
        self.denied_requests = 0
    
    def is_allowed(self, tokens: int = 1) -> RateLimitResult:
        """Check if request is allowed within sliding window."""
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # Remove old requests outside the window
        while self.requests and self.requests[0] <= window_start:
            self.requests.popleft()
        
        self.total_requests += 1
        current_count = len(self.requests)
        max_requests = int(self.rate * self.window_size)
        
        if current_count < max_requests:
            # Add current request timestamp
            for _ in range(tokens):
                self.requests.append(current_time)
            
            remaining = max_requests - current_count - tokens
            reset_time = current_time + self.window_size
            
            return RateLimitResult(
                allowed=True,
                remaining=max(0, remaining),
                reset_time=reset_time,
                metadata={
                    "strategy": "sliding_window",
                    "window_size": self.window_size,
                    "current_count": current_count,
                    "max_requests": max_requests
                }
            )
        else:
            self.denied_requests += 1
            
            # Calculate when the next slot will be available
            if self.requests:
                oldest_request = self.requests[0]
                retry_after = oldest_request + self.window_size - current_time
            else:
                retry_after = self.window_size
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=current_time + retry_after,
                retry_after=max(0, retry_after),
                reason="Rate limit exceeded",
                metadata={
                    "strategy": "sliding_window",
                    "current_count": current_count,
                    "max_requests": max_requests
                }
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sliding window statistics."""
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # Clean old requests for accurate stats
        while self.requests and self.requests[0] <= window_start:
            self.requests.popleft()
        
        total = self.total_requests
        success_rate = ((total - self.denied_requests) / total * 100) if total > 0 else 100.0
        
        return {
            "rate": self.rate,
            "window_size": self.window_size,
            "current_requests": len(self.requests),
            "max_requests": int(self.rate * self.window_size),
            "total_requests": total,
            "denied_requests": self.denied_requests,
            "success_rate": success_rate
        }


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts limits based on system load and performance."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.current_rate = config.requests_per_second
        self.token_bucket = TokenBucket(self.current_rate, config.burst_size)
        
        # Adaptive scaling metrics
        self.response_times = deque(maxlen=100)
        self.error_rates = deque(maxlen=50)
        self.adjustment_history = deque(maxlen=20)
        
        # Adjustment parameters
        self.last_adjustment = time.time()
        self.adjustment_interval = 30.0  # Adjust every 30 seconds
        self.performance_threshold = 1.0  # 1 second response time threshold
        self.error_rate_threshold = 0.05  # 5% error rate threshold
    
    def consume(self, tokens: int = 1, response_time: float = None, error: bool = False) -> RateLimitResult:
        """Consume tokens with adaptive adjustment."""
        # Record performance metrics
        if response_time is not None:
            self.response_times.append(response_time)
        
        self.error_rates.append(1.0 if error else 0.0)
        
        # Check if adjustment is needed
        current_time = time.time()
        if current_time - self.last_adjustment >= self.adjustment_interval:
            self._adjust_rate()
            self.last_adjustment = current_time
        
        return self.token_bucket.consume(tokens)
    
    def _adjust_rate(self):
        """Adjust rate based on performance metrics."""
        if not self.response_times and not self.error_rates:
            return
        
        current_time = time.time()
        
        # Calculate performance indicators
        avg_response_time = mean(self.response_times) if self.response_times else 0
        error_rate = mean(self.error_rates) if self.error_rates else 0
        
        # Determine adjustment
        adjustment_factor = 1.0
        reason = "no_change"
        
        if error_rate > self.error_rate_threshold:
            # High error rate - reduce rate
            adjustment_factor = 0.8
            reason = f"high_error_rate_{error_rate:.3f}"
        elif avg_response_time > self.performance_threshold:
            # High response time - reduce rate
            adjustment_factor = 0.9
            reason = f"high_response_time_{avg_response_time:.3f}"
        elif error_rate < 0.01 and avg_response_time < self.performance_threshold * 0.5:
            # Good performance - increase rate
            adjustment_factor = 1.1
            reason = f"good_performance"
        
        # Apply adjustment with bounds
        old_rate = self.current_rate
        self.current_rate = max(
            self.config.min_rate,
            min(self.config.max_rate, self.current_rate * adjustment_factor)
        )
        
        if abs(self.current_rate - old_rate) > 0.1:
            # Update token bucket rate
            self.token_bucket.rate = self.current_rate
            
            # Record adjustment
            self.adjustment_history.append({
                "timestamp": current_time,
                "old_rate": old_rate,
                "new_rate": self.current_rate,
                "factor": adjustment_factor,
                "reason": reason,
                "avg_response_time": avg_response_time,
                "error_rate": error_rate
            })
            
            logger.info(
                f"Adaptive rate limit adjusted: {old_rate:.2f} -> {self.current_rate:.2f} RPS "
                f"(reason: {reason})"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive rate limiter statistics."""
        bucket_stats = self.token_bucket.get_stats()
        
        stats = {
            **bucket_stats,
            "strategy": "adaptive",
            "current_rate": self.current_rate,
            "original_rate": self.config.requests_per_second,
            "min_rate": self.config.min_rate,
            "max_rate": self.config.max_rate,
            "recent_adjustments": list(self.adjustment_history)[-5:],
            "avg_response_time": mean(self.response_times) if self.response_times else 0,
            "error_rate": mean(self.error_rates) if self.error_rates else 0
        }
        
        return stats


class RateLimiter:
    """Enterprise-grade rate limiter with multiple strategies and advanced features."""
    
    def __init__(self):
        self.limiters: Dict[str, Dict[str, Any]] = {}
        self.global_metrics = defaultdict(int)
        
        # Distributed rate limiting (placeholder for Redis integration)
        self.distributed_backend = None
    
    def create_limiter(self, config: RateLimitConfig) -> str:
        """Create a new rate limiter with the given configuration."""
        limiter_id = self._generate_limiter_id(config)
        
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            implementation = TokenBucket(config.requests_per_second, config.burst_size)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            implementation = SlidingWindowRateLimiter(config.requests_per_second, config.window_size)
        elif config.strategy == RateLimitStrategy.ADAPTIVE:
            implementation = AdaptiveRateLimiter(config)
        else:
            # Default to token bucket
            implementation = TokenBucket(config.requests_per_second, config.burst_size)
        
        self.limiters[limiter_id] = {
            "config": config,
            "implementation": implementation,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "client_limiters": {}  # For per-client rate limiting
        }
        
        logger.info(f"Created rate limiter '{config.name}' with strategy {config.strategy.value}")
        return limiter_id
    
    def _generate_limiter_id(self, config: RateLimitConfig) -> str:
        """Generate unique limiter ID."""
        id_string = f"{config.name}_{config.strategy.value}_{config.scope.value}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def check_rate_limit(
        self, 
        limiter_id: str, 
        client_id: Optional[str] = None,
        tokens: int = 1,
        **metadata
    ) -> RateLimitResult:
        """Check if request is within rate limit."""
        if limiter_id not in self.limiters:
            logger.error(f"Rate limiter {limiter_id} not found")
            return RateLimitResult(
                allowed=True,
                remaining=1000,
                reset_time=time.time() + 3600,
                reason="limiter_not_found"
            )
        
        limiter_data = self.limiters[limiter_id]
        config = limiter_data["config"]
        limiter_data["last_accessed"] = time.time()
        
        # Handle different scopes
        if config.scope == RateLimitScope.GLOBAL:
            implementation = limiter_data["implementation"]
        elif config.scope == RateLimitScope.PER_CLIENT:
            if not client_id:
                client_id = "default"
            
            if client_id not in limiter_data["client_limiters"]:
                # Create per-client limiter
                if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                    client_limiter = TokenBucket(config.requests_per_second, config.burst_size)
                elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                    client_limiter = SlidingWindowRateLimiter(config.requests_per_second, config.window_size)
                elif config.strategy == RateLimitStrategy.ADAPTIVE:
                    client_limiter = AdaptiveRateLimiter(config)
                else:
                    client_limiter = TokenBucket(config.requests_per_second, config.burst_size)
                
                limiter_data["client_limiters"][client_id] = client_limiter
            
            implementation = limiter_data["client_limiters"][client_id]
        else:
            # Default to global scope
            implementation = limiter_data["implementation"]
        
        # Execute rate limit check
        if hasattr(implementation, 'consume'):
            # Token bucket or adaptive
            result = implementation.consume(tokens)
        else:
            # Sliding window
            result = implementation.is_allowed(tokens)
        
        # Record global metrics
        self.global_metrics["total_requests"] += 1
        if not result.allowed:
            self.global_metrics["denied_requests"] += 1
        
        # Add limiter info to result
        result.metadata.update({
            "limiter_name": config.name,
            "limiter_id": limiter_id,
            "client_id": client_id,
            "scope": config.scope.value,
            **metadata
        })
        
        return result
    
    async def check_rate_limit_async(
        self, 
        limiter_id: str, 
        client_id: Optional[str] = None,
        tokens: int = 1,
        **metadata
    ) -> RateLimitResult:
        """Async version of rate limit check."""
        # For now, just call the sync version
        # In a real implementation, this could handle distributed rate limiting with Redis
        return self.check_rate_limit(limiter_id, client_id, tokens, **metadata)
    
    def get_limiter_stats(self, limiter_id: str) -> Dict[str, Any]:
        """Get statistics for a specific rate limiter."""
        if limiter_id not in self.limiters:
            return {"error": "limiter_not_found"}
        
        limiter_data = self.limiters[limiter_id]
        config = limiter_data["config"]
        implementation = limiter_data["implementation"]
        
        stats = {
            "limiter_id": limiter_id,
            "name": config.name,
            "strategy": config.strategy.value,
            "scope": config.scope.value,
            "created_at": limiter_data["created_at"],
            "last_accessed": limiter_data["last_accessed"],
            "client_count": len(limiter_data["client_limiters"])
        }
        
        # Get implementation-specific stats
        if hasattr(implementation, 'get_stats'):
            stats.update(implementation.get_stats())
        
        # Get per-client stats if applicable
        if config.scope == RateLimitScope.PER_CLIENT and limiter_data["client_limiters"]:
            client_stats = {}
            for client_id, client_limiter in limiter_data["client_limiters"].items():
                if hasattr(client_limiter, 'get_stats'):
                    client_stats[client_id] = client_limiter.get_stats()
            stats["client_stats"] = client_stats
        
        return stats
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics."""
        total_requests = self.global_metrics["total_requests"]
        denied_requests = self.global_metrics["denied_requests"]
        success_rate = ((total_requests - denied_requests) / total_requests * 100) if total_requests > 0 else 100.0
        
        return {
            "total_limiters": len(self.limiters),
            "total_requests": total_requests,
            "denied_requests": denied_requests,
            "success_rate": success_rate,
            "limiters": {
                limiter_id: {
                    "name": data["config"].name,
                    "strategy": data["config"].strategy.value,
                    "last_accessed": data["last_accessed"]
                }
                for limiter_id, data in self.limiters.items()
            }
        }
    
    def cleanup_inactive_limiters(self, max_age: int = 3600):
        """Clean up inactive rate limiters."""
        current_time = time.time()
        inactive_limiters = []
        
        for limiter_id, data in self.limiters.items():
            if current_time - data["last_accessed"] > max_age:
                inactive_limiters.append(limiter_id)
        
        for limiter_id in inactive_limiters:
            del self.limiters[limiter_id]
            logger.info(f"Cleaned up inactive rate limiter: {limiter_id}")
        
        return len(inactive_limiters)
    
    def create_decorator(self, config: RateLimitConfig, client_id_func: Optional[Callable] = None):
        """Create a decorator for rate-limited functions."""
        limiter_id = self.create_limiter(config)
        
        def decorator(func: Callable) -> Callable:
            async def async_wrapper(*args, **kwargs):
                client_id = None
                if client_id_func:
                    try:
                        client_id = client_id_func(*args, **kwargs)
                    except Exception as e:
                        logger.warning(f"Failed to extract client_id: {e}")
                
                result = await self.check_rate_limit_async(limiter_id, client_id)
                
                if not result.allowed:
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for {config.name}",
                        retry_after=result.retry_after,
                        result=result
                    )
                
                return await func(*args, **kwargs)
            
            def sync_wrapper(*args, **kwargs):
                client_id = None
                if client_id_func:
                    try:
                        client_id = client_id_func(*args, **kwargs)
                    except Exception as e:
                        logger.warning(f"Failed to extract client_id: {e}")
                
                result = self.check_rate_limit(limiter_id, client_id)
                
                if not result.allowed:
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for {config.name}",
                        retry_after=result.retry_after,
                        result=result
                    )
                
                return func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None, result: Optional[RateLimitResult] = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.result = result


# Global rate limiter instance
rate_limiter = RateLimiter()


# Predefined rate limiter configurations
GITHUB_API_RATE_LIMIT = RateLimitConfig(
    name="github_api",
    requests_per_second=5.0,
    burst_size=10,
    strategy=RateLimitStrategy.TOKEN_BUCKET,
    scope=RateLimitScope.GLOBAL,
    adaptive_scaling=True,
    max_rate=20.0
)

DATABASE_RATE_LIMIT = RateLimitConfig(
    name="database",
    requests_per_second=50.0,
    burst_size=100,
    strategy=RateLimitStrategy.SLIDING_WINDOW,
    scope=RateLimitScope.GLOBAL,
    window_size=60
)

ML_INFERENCE_RATE_LIMIT = RateLimitConfig(
    name="ml_inference",
    requests_per_second=2.0,
    burst_size=5,
    strategy=RateLimitStrategy.ADAPTIVE,
    scope=RateLimitScope.PER_CLIENT,
    min_rate=0.5,
    max_rate=10.0
)

WEBHOOK_RATE_LIMIT = RateLimitConfig(
    name="webhook",
    requests_per_second=10.0,
    burst_size=20,
    strategy=RateLimitStrategy.TOKEN_BUCKET,
    scope=RateLimitScope.PER_CLIENT
)


def rate_limited(config: RateLimitConfig, client_id_func: Optional[Callable] = None):
    """Decorator for rate-limited functions."""
    return rate_limiter.create_decorator(config, client_id_func)


def github_api_rate_limited(client_id_func: Optional[Callable] = None):
    """Rate limiting decorator for GitHub API calls."""
    return rate_limiter.create_decorator(GITHUB_API_RATE_LIMIT, client_id_func)


def database_rate_limited():
    """Rate limiting decorator for database operations."""
    return rate_limiter.create_decorator(DATABASE_RATE_LIMIT)


def ml_inference_rate_limited(client_id_func: Optional[Callable] = None):
    """Rate limiting decorator for ML inference operations."""
    return rate_limiter.create_decorator(ML_INFERENCE_RATE_LIMIT, client_id_func)


def webhook_rate_limited(client_id_func: Optional[Callable] = None):
    """Rate limiting decorator for webhook processing."""
    return rate_limiter.create_decorator(WEBHOOK_RATE_LIMIT, client_id_func)