"""Context management for bot operations."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class Context:
    """Execution context for bot operations."""
    
    # Repository information
    repo_owner: str
    repo_name: str
    repo_full_name: str
    
    # Event information
    event_type: str
    event_data: Dict[str, Any]
    
    # State management
    state: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    execution_id: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # File operations
    _file_changes: Dict[str, str] = field(default_factory=dict)
    
    def set_error(self, error_type: str, message: str, traceback: Optional[str] = None) -> None:
        """Set error information in the context."""
        self.error_type = error_type
        self.error_message = message
        self.error_traceback = traceback
    
    def has_error(self) -> bool:
        """Check if context has error information."""
        return self.error_message is not None
    
    def clear_error(self) -> None:
        """Clear error information from context."""
        self.error_type = None
        self.error_message = None
        self.error_traceback = None
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a state value."""
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self.state.get(key, default)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        # In a real implementation, this would interact with the GitHub API
        # For now, return a mock configuration
        return {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 100
        }
    
    def save_config(self, config_path: str, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        # Store the file change for later processing
        self._file_changes[config_path] = json.dumps(config, indent=2)
    
    def read_file(self, file_path: str) -> str:
        """Read file contents."""
        # In a real implementation, this would use the GitHub API
        return f"# Mock content for {file_path}"
    
    def write_file(self, file_path: str, content: str) -> None:
        """Write content to file."""
        # Store the file change for later processing
        self._file_changes[file_path] = content
    
    def get_file_changes(self) -> Dict[str, str]:
        """Get all file changes made during this context."""
        return self._file_changes.copy()
    
    def create_pull_request(self, title: str, body: str, branch: str) -> "MockPR":
        """Create a pull request with the changes."""
        # Mock PR creation - in real implementation, this would use GitHub API
        return MockPR(number=123, title=title, body=body, branch=branch)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "repo_full_name": self.repo_full_name,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "state": self.state,
            "execution_id": self.execution_id,
            "started_at": self.started_at.isoformat(),
            "error_message": self.error_message,
            "error_type": self.error_type,
            "has_error": self.has_error(),
            "file_changes": list(self._file_changes.keys())
        }


@dataclass
class MockPR:
    """Mock pull request object."""
    number: int
    title: str
    body: str
    branch: str