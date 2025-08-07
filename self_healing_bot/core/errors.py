"""Custom exceptions and error handling for the self-healing bot."""

from typing import Dict, Any, Optional, List
from enum import Enum
import traceback


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BotError(Exception):
    """Base exception for all bot errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: str = None,
        details: Dict[str, Any] = None,
        recoverable: bool = True
    ):
        self.message = message
        self.severity = severity
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.recoverable = recoverable
        super().__init__(message)


class ValidationError(BotError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None, **kwargs):
        self.field = field
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
            **kwargs
        )


class DetectorError(BotError):
    """Raised when a detector fails."""
    
    def __init__(self, detector_name: str, message: str, **kwargs):
        self.detector_name = detector_name
        super().__init__(
            message=f"Detector '{detector_name}' failed: {message}",
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class PlaybookError(BotError):
    """Raised when a playbook fails."""
    
    def __init__(self, playbook_name: str, action: str, message: str, **kwargs):
        self.playbook_name = playbook_name
        self.action = action
        super().__init__(
            message=f"Playbook '{playbook_name}' action '{action}' failed: {message}",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class GitHubAPIError(BotError):
    """Raised when GitHub API operations fail."""
    
    def __init__(self, endpoint: str, status_code: int, message: str, **kwargs):
        self.endpoint = endpoint
        self.status_code = status_code
        super().__init__(
            message=f"GitHub API error ({status_code}) at {endpoint}: {message}",
            severity=ErrorSeverity.HIGH if status_code >= 500 else ErrorSeverity.MEDIUM,
            **kwargs
        )


class SecurityError(BotError):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=f"Security violation: {message}",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )


class ConfigurationError(BotError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        self.config_key = config_key
        super().__init__(
            message=f"Configuration error: {message}",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )


class ResourceError(BotError):
    """Raised when system resources are exhausted."""
    
    def __init__(self, resource: str, message: str, **kwargs):
        self.resource = resource
        super().__init__(
            message=f"Resource exhausted ({resource}): {message}",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ErrorHandler:
    """Centralized error handler for the bot."""
    
    def __init__(self):
        self.error_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        component: str = "unknown"
    ) -> Dict[str, Any]:
        """Handle an error and return error information."""
        from datetime import datetime
        from ..monitoring.logging import get_logger, audit_logger
        from ..monitoring.metrics import metrics
        
        logger = get_logger(__name__)
        
        # Determine error details
        if isinstance(error, BotError):
            error_info = {
                "type": error.__class__.__name__,
                "message": error.message,
                "severity": error.severity.value,
                "error_code": error.error_code,
                "details": error.details,
                "recoverable": error.recoverable
            }
        else:
            error_info = {
                "type": error.__class__.__name__,
                "message": str(error),
                "severity": ErrorSeverity.HIGH.value,
                "error_code": error.__class__.__name__,
                "details": {},
                "recoverable": True
            }
        
        # Add context and timing
        error_record = {
            **error_info,
            "timestamp": datetime.utcnow().isoformat(),
            "component": component,
            "context": context or {},
            "traceback": traceback.format_exc() if not isinstance(error, BotError) else None
        }
        
        # Log error
        logger.error(
            f"Error in {component}: {error_info['message']}",
            error_type=error_info['type'],
            severity=error_info['severity'],
            component=component,
            context=context or {}
        )
        
        # Record security events separately
        if isinstance(error, SecurityError):
            audit_logger.log_security_event(
                event_type="security_error",
                severity=error_info['severity'],
                details={
                    "error_type": error_info['type'],
                    "message": error_info['message'],
                    "component": component,
                    "context": context or {}
                }
            )
        
        # Update metrics
        metrics.record_error(error_info['type'], component)
        
        # Store in history
        self.error_history.append(error_record)
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        
        return error_record
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours."""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_errors = [
            error for error in self.error_history
            if datetime.fromisoformat(error['timestamp']) > cutoff_time
        ]
        
        if not recent_errors:
            return {"total_errors": 0, "by_type": {}, "by_component": {}, "by_severity": {}}
        
        # Aggregate by type
        by_type = {}
        by_component = {}
        by_severity = {}
        
        for error in recent_errors:
            # By type
            error_type = error['type']
            by_type[error_type] = by_type.get(error_type, 0) + 1
            
            # By component
            component = error['component']
            by_component[component] = by_component.get(component, 0) + 1
            
            # By severity
            severity = error['severity']
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "by_type": by_type,
            "by_component": by_component,
            "by_severity": by_severity,
            "recent_errors": recent_errors[-10:]  # Last 10 errors
        }
    
    def should_retry(self, error: Exception, attempt: int, max_attempts: int = 3) -> bool:
        """Determine if an operation should be retried."""
        if attempt >= max_attempts:
            return False
        
        if isinstance(error, BotError):
            return error.recoverable
        
        # Common recoverable errors
        recoverable_types = {
            'TimeoutError', 'ConnectionError', 'HTTPError',
            'TemporaryFailure', 'ServiceUnavailable'
        }
        
        return type(error).__name__ in recoverable_types


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        import time
        
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record a successful execution."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record a failed execution."""
        import time
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


# Global error handler instance
error_handler = ErrorHandler()

# Circuit breakers for different components
circuit_breakers = {
    "github_api": CircuitBreaker(failure_threshold=5, recovery_timeout=60),
    "detectors": CircuitBreaker(failure_threshold=3, recovery_timeout=30),
    "playbooks": CircuitBreaker(failure_threshold=3, recovery_timeout=30),
}