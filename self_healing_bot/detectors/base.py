"""Base detector interface and common functionality."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from datetime import datetime

from ..core.context import Context

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Base class for all issue detectors."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
    
    @abstractmethod
    def get_supported_events(self) -> List[str]:
        """Get list of GitHub event types this detector supports."""
        pass
    
    @abstractmethod
    async def detect(self, context: Context) -> List[Dict[str, Any]]:
        """
        Detect issues in the given context.
        
        Returns:
            List of issue dictionaries with keys:
            - type: Issue type identifier
            - severity: "low", "medium", "high", "critical"
            - message: Human-readable description
            - data: Additional issue-specific data
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if detector is enabled."""
        return self.enabled
    
    def supports_event(self, event_type: str) -> bool:
        """Check if detector supports the given event type."""
        return event_type in self.get_supported_events()
    
    def create_issue(
        self,
        issue_type: str,
        severity: str,
        message: str,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a standardized issue dictionary."""
        return {
            "type": issue_type,
            "severity": severity,
            "message": message,
            "data": data or {},
            "detector": self.__class__.__name__,
            "timestamp": datetime.utcnow().isoformat()
        }