"""Structured logging configuration."""

import logging
import sys
from typing import Dict, Any, Optional
import structlog
from datetime import datetime
import json

try:
    from ..core.config import config
except ImportError:
    # Fallback configuration for when config is not available
    class MockConfig:
        log_level = "INFO"
        environment = "development"
    config = MockConfig()


def setup_logging():
    """Setup structured logging with structlog."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if config.environment != "development" 
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.log_level.upper()),
    )


class BotLogger:
    """Enhanced logger for bot operations."""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)
    
    def with_context(self, **kwargs) -> "BotLogger":
        """Create logger with additional context."""
        new_logger = BotLogger(self.logger.name)
        new_logger.logger = self.logger.bind(**kwargs)
        return new_logger


class ContextualLogger:
    """Logger that automatically includes execution context."""
    
    def __init__(self, base_logger: BotLogger):
        self.base_logger = base_logger
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set logging context."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context."""
        self.context.clear()
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with context."""
        merged_context = {**self.context, **kwargs}
        getattr(self.base_logger, level)(message, **merged_context)
    
    def info(self, message: str, **kwargs):
        self._log_with_context("info", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context("error", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context("warning", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context("debug", message, **kwargs)


class AuditLogger:
    """Audit logger for tracking important bot actions."""
    
    def __init__(self):
        self.logger = BotLogger("audit")
    
    def log_repair_action(
        self,
        repo: str,
        playbook: str,
        action: str,
        success: bool,
        details: Dict[str, Any] = None
    ):
        """Log repair action for audit trail."""
        self.logger.info(
            "Repair action executed",
            event_type="repair_action",
            repo=repo,
            playbook=playbook,
            action=action,
            success=success,
            details=details or {}
        )
    
    def log_issue_detected(
        self,
        repo: str,
        issue_type: str,
        severity: str,
        detector: str,
        details: Dict[str, Any] = None
    ):
        """Log issue detection for audit trail."""
        self.logger.info(
            "Issue detected",
            event_type="issue_detected",
            repo=repo,
            issue_type=issue_type,
            severity=severity,
            detector=detector,
            details=details or {}
        )
    
    def log_pr_created(
        self,
        repo: str,
        pr_number: int,
        title: str,
        playbook: str,
        changes: list = None
    ):
        """Log pull request creation."""
        self.logger.info(
            "Pull request created",
            event_type="pr_created",
            repo=repo,
            pr_number=pr_number,
            title=title,
            playbook=playbook,
            changes=changes or []
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any]
    ):
        """Log security-related events."""
        self.logger.warning(
            "Security event",
            event_type="security",
            security_event_type=event_type,
            severity=severity,
            details=details
        )


class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self):
        self.logger = BotLogger("performance")
    
    def log_execution_time(
        self,
        operation: str,
        duration: float,
        success: bool,
        **kwargs
    ):
        """Log operation execution time."""
        self.logger.info(
            "Operation completed",
            event_type="performance",
            operation=operation,
            duration_seconds=duration,
            success=success,
            **kwargs
        )
    
    def log_resource_usage(
        self,
        component: str,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[float] = None,
        **kwargs
    ):
        """Log resource usage."""
        self.logger.debug(
            "Resource usage",
            event_type="resource_usage",
            component=component,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            **kwargs
        )


# Global logger instances
audit_logger = AuditLogger()
performance_logger = PerformanceLogger()


def get_logger(name: str) -> BotLogger:
    """Get a configured logger instance."""
    return BotLogger(name)