"""Base action interface and common functionality."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

from ..core.context import Context

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    message: str
    data: Dict[str, Any] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class BaseAction(ABC):
    """Base class for all automated actions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.timeout = self.config.get("timeout", 300)  # 5 minutes default
        self.dry_run = self.config.get("dry_run", False)
    
    @abstractmethod
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """
        Execute the action for the given issue.
        
        Args:
            context: Execution context
            issue_data: Data about the issue being addressed
            
        Returns:
            ActionResult with execution details
        """
        pass
    
    @abstractmethod
    def can_handle(self, issue_type: str) -> bool:
        """Check if this action can handle the given issue type."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if action is enabled."""
        return self.enabled
    
    def get_timeout(self) -> int:
        """Get action timeout in seconds."""
        return self.timeout
    
    def is_dry_run(self) -> bool:
        """Check if running in dry-run mode."""
        return self.dry_run
    
    def create_result(self, success: bool, message: str, data: Dict[str, Any] = None) -> ActionResult:
        """Create a standardized action result."""
        return ActionResult(
            success=success,
            message=message,
            data=data or {}
        )
    
    def log_action(self, context: Context, message: str, level: str = "info") -> None:
        """Log action execution with context."""
        log_message = f"[{context.repo_full_name}] {self.__class__.__name__}: {message}"
        
        if level == "debug":
            logger.debug(log_message)
        elif level == "info":
            logger.info(log_message)
        elif level == "warning":
            logger.warning(log_message)
        elif level == "error":
            logger.error(log_message)
        else:
            logger.info(log_message)