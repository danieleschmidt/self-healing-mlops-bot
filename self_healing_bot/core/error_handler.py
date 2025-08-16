"""Advanced error handling and recovery system for the self-healing bot."""

import asyncio
import logging
import traceback
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    GITHUB_API = "github_api"
    WEBHOOK = "webhook"
    PLAYBOOK = "playbook"
    DETECTOR = "detector"
    DATABASE = "database"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    error_id: str
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    resolved: bool = False


class RetryStrategy:
    """Configurable retry strategy."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        exponential_base: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.exponential_base = exponential_base
        self.max_delay = max_delay
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


class ErrorHandler:
    """Advanced error handling system with retry logic and recovery."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.metrics = {
            "total_errors": 0,
            "errors_by_category": {cat.value: 0 for cat in ErrorCategory},
            "errors_by_severity": {sev.value: 0 for sev in ErrorSeverity},
            "recovery_attempts": 0,
            "successful_recoveries": 0
        }
    
    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        strategy: Callable[[ErrorContext], bool]
    ) -> None:
        """Register a recovery strategy for an error category."""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        self.recovery_strategies[category].append(strategy)
    
    def categorize_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorCategory:
        """Categorize an error based on its type and context."""
        error_type = type(error).__name__
        context = context or {}
        
        # GitHub API errors
        if "github" in str(error).lower() or "PyGithub" in error_type:
            return ErrorCategory.GITHUB_API
        
        # Network errors
        if any(keyword in error_type.lower() for keyword in ["connection", "timeout", "network"]):
            return ErrorCategory.NETWORK
        
        # Authentication errors
        if any(keyword in str(error).lower() for keyword in ["auth", "token", "permission", "unauthorized"]):
            return ErrorCategory.AUTHENTICATION
        
        # Database errors
        if any(keyword in error_type.lower() for keyword in ["sql", "database", "db"]):
            return ErrorCategory.DATABASE
        
        # Validation errors
        if "validation" in error_type.lower() or "pydantic" in error_type.lower():
            return ErrorCategory.VALIDATION
        
        # Configuration errors
        if "config" in str(error).lower():
            return ErrorCategory.CONFIGURATION
        
        # Context-based categorization
        if context.get("source") == "webhook":
            return ErrorCategory.WEBHOOK
        elif context.get("source") == "playbook":
            return ErrorCategory.PLAYBOOK
        elif context.get("source") == "detector":
            return ErrorCategory.DETECTOR
        
        return ErrorCategory.UNKNOWN
    
    def determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on type and category."""
        error_type = type(error).__name__
        
        # Critical errors
        if any(keyword in error_type.lower() for keyword in ["memory", "disk", "system"]):
            return ErrorSeverity.CRITICAL
        
        # High severity
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.DATABASE]:
            return ErrorSeverity.HIGH
        
        # Medium severity
        if category in [ErrorCategory.GITHUB_API, ErrorCategory.NETWORK]:
            return ErrorSeverity.MEDIUM
        
        # Low severity for validation and configuration
        return ErrorSeverity.LOW
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        retry_strategy: Optional[RetryStrategy] = None
    ) -> ErrorContext:
        """Handle an error with automatic categorization and recovery."""
        import uuid
        
        context = context or {}
        category = self.categorize_error(error, context)
        severity = self.determine_severity(error, category)
        
        error_context = ErrorContext(
            error_id=str(uuid.uuid4()),
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            category=category,
            traceback=traceback.format_exc(),
            metadata=context
        )
        
        # Update metrics
        self.metrics["total_errors"] += 1
        self.metrics["errors_by_category"][category.value] += 1
        self.metrics["errors_by_severity"][severity.value] += 1
        
        # Log the error
        await self._log_error(error_context)
        
        # Store in history
        self.error_history.append(error_context)
        
        # Attempt recovery if strategies exist
        if category in self.recovery_strategies:
            await self._attempt_recovery(error_context, retry_strategy)
        
        return error_context
    
    async def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with structured logging."""
        log_data = {
            "error_id": error_context.error_id,
            "error_type": error_context.error_type,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "retry_count": error_context.retry_count,
            "metadata": error_context.metadata
        }
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", **log_data)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error("High severity error", **log_data)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error", **log_data)
        else:
            logger.info("Low severity error", **log_data)
    
    async def _attempt_recovery(
        self,
        error_context: ErrorContext,
        retry_strategy: Optional[RetryStrategy]
    ) -> None:
        """Attempt error recovery using registered strategies."""
        strategies = self.recovery_strategies.get(error_context.category, [])
        retry_strategy = retry_strategy or RetryStrategy()
        
        for strategy in strategies:
            self.metrics["recovery_attempts"] += 1
            
            try:
                success = await self._execute_with_retry(
                    strategy,
                    error_context,
                    retry_strategy
                )
                
                if success:
                    error_context.resolved = True
                    self.metrics["successful_recoveries"] += 1
                    logger.info(
                        "Error recovery successful",
                        error_id=error_context.error_id,
                        strategy=strategy.__name__
                    )
                    break
                    
            except Exception as recovery_error:
                logger.error(
                    "Recovery strategy failed",
                    error_id=error_context.error_id,
                    strategy=strategy.__name__,
                    recovery_error=str(recovery_error)
                )
    
    async def _execute_with_retry(
        self,
        strategy: Callable,
        error_context: ErrorContext,
        retry_strategy: RetryStrategy
    ) -> bool:
        """Execute a recovery strategy with retry logic."""
        for attempt in range(retry_strategy.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(strategy):
                    result = await strategy(error_context)
                else:
                    result = strategy(error_context)
                
                if result:
                    return True
                    
            except Exception as e:
                error_context.retry_count += 1
                logger.warning(
                    "Recovery attempt failed",
                    error_id=error_context.error_id,
                    attempt=attempt,
                    error=str(e)
                )
                
                if attempt < retry_strategy.max_retries:
                    delay = retry_strategy.get_delay(attempt)
                    await asyncio.sleep(delay)
        
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        recent_errors = [
            e for e in self.error_history[-100:]  # Last 100 errors
            if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        return {
            "total_errors": self.metrics["total_errors"],
            "errors_by_category": self.metrics["errors_by_category"],
            "errors_by_severity": self.metrics["errors_by_severity"],
            "recovery_attempts": self.metrics["recovery_attempts"],
            "successful_recoveries": self.metrics["successful_recoveries"],
            "recovery_rate": (
                self.metrics["successful_recoveries"] / max(self.metrics["recovery_attempts"], 1)
            ),
            "recent_errors_count": len(recent_errors),
            "unresolved_errors": len([e for e in self.error_history if not e.resolved]),
            "error_trends": self._calculate_error_trends()
        }
    
    def _calculate_error_trends(self) -> Dict[str, Any]:
        """Calculate error trends over time."""
        now = datetime.now(timezone.utc)
        
        # Errors in last hour vs previous hour
        last_hour = [
            e for e in self.error_history
            if (now - e.timestamp).total_seconds() <= 3600
        ]
        previous_hour = [
            e for e in self.error_history
            if 3600 < (now - e.timestamp).total_seconds() <= 7200
        ]
        
        return {
            "last_hour_count": len(last_hour),
            "previous_hour_count": len(previous_hour),
            "trend": "increasing" if len(last_hour) > len(previous_hour) else "decreasing",
            "change_rate": (
                (len(last_hour) - len(previous_hour)) / max(len(previous_hour), 1)
            ) * 100
        }


# Global error handler instance
error_handler = ErrorHandler()


# Decorator for automatic error handling
def handle_errors(
    category: Optional[ErrorCategory] = None,
    retry_strategy: Optional[RetryStrategy] = None,
    reraise: bool = False
):
    """Decorator for automatic error handling."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
                if category:
                    context["source"] = category.value
                
                error_context = await error_handler.handle_error(e, context, retry_strategy)
                
                if reraise:
                    raise
                
                return None
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
                if category:
                    context["source"] = category.value
                
                # For sync functions, we need to handle the async error handler
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, schedule the error handling
                    asyncio.create_task(error_handler.handle_error(e, context, retry_strategy))
                else:
                    # Run in new event loop
                    asyncio.run(error_handler.handle_error(e, context, retry_strategy))
                
                if reraise:
                    raise
                
                return None
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator