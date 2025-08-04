"""Input validation and sanitization."""

import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import json
import logging

from ..monitoring.logging import get_logger, audit_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class InputValidator:
    """Comprehensive input validation for bot operations."""
    
    # Regex patterns for validation
    GITHUB_REPO_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$')
    GITHUB_BRANCH_PATTERN = re.compile(r'^[a-zA-Z0-9_./\-]+$')
    FILE_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9_./\-]+$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        re.compile(r'\.\./', re.IGNORECASE),  # Path traversal
        re.compile(r'<script', re.IGNORECASE),  # XSS
        re.compile(r'javascript:', re.IGNORECASE),  # XSS
        re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
        re.compile(r'eval\s*\(', re.IGNORECASE),  # Code execution
        re.compile(r'exec\s*\(', re.IGNORECASE),  # Code execution
        re.compile(r'import\s+os', re.IGNORECASE),  # OS access
        re.compile(r'__import__', re.IGNORECASE),  # Dynamic imports
    ]
    
    @classmethod
    def validate_repo_name(cls, repo_name: str) -> str:
        """Validate GitHub repository name."""
        if not repo_name or not isinstance(repo_name, str):
            raise ValidationError("Repository name must be a non-empty string")
        
        if not cls.GITHUB_REPO_PATTERN.match(repo_name):
            raise ValidationError(f"Invalid repository name format: {repo_name}")
        
        if len(repo_name) > 100:
            raise ValidationError("Repository name too long")
        
        return repo_name.strip()
    
    @classmethod
    def validate_branch_name(cls, branch_name: str) -> str:
        """Validate Git branch name."""
        if not branch_name or not isinstance(branch_name, str):
            raise ValidationError("Branch name must be a non-empty string")
        
        if not cls.GITHUB_BRANCH_PATTERN.match(branch_name):
            raise ValidationError(f"Invalid branch name format: {branch_name}")
        
        if len(branch_name) > 250:
            raise ValidationError("Branch name too long")
        
        # Check for dangerous patterns
        cls._check_dangerous_patterns(branch_name, "branch name")
        
        return branch_name.strip()
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> str:
        """Validate file path for safety."""
        if not file_path or not isinstance(file_path, str):
            raise ValidationError("File path must be a non-empty string")
        
        if not cls.FILE_PATH_PATTERN.match(file_path):
            raise ValidationError(f"Invalid file path format: {file_path}")
        
        if len(file_path) > 500:
            raise ValidationError("File path too long")
        
        # Check for path traversal
        if '..' in file_path:
            raise ValidationError("Path traversal not allowed")
        
        # Check for absolute paths
        if file_path.startswith('/'):
            raise ValidationError("Absolute paths not allowed")
        
        return file_path.strip()
    
    @classmethod
    def validate_commit_message(cls, message: str) -> str:
        """Validate commit message."""
        if not message or not isinstance(message, str):
            raise ValidationError("Commit message must be a non-empty string")
        
        if len(message) > 500:
            raise ValidationError("Commit message too long")
        
        # Check for dangerous patterns
        cls._check_dangerous_patterns(message, "commit message")
        
        return message.strip()
    
    @classmethod
    def validate_pr_title(cls, title: str) -> str:
        """Validate pull request title."""
        if not title or not isinstance(title, str):
            raise ValidationError("PR title must be a non-empty string")
        
        if len(title) > 250:
            raise ValidationError("PR title too long")
        
        # Check for dangerous patterns
        cls._check_dangerous_patterns(title, "PR title")
        
        return title.strip()
    
    @classmethod
    def validate_pr_body(cls, body: str) -> str:
        """Validate pull request body."""
        if not isinstance(body, str):
            raise ValidationError("PR body must be a string")
        
        if len(body) > 10000:
            raise ValidationError("PR body too long")
        
        # Check for dangerous patterns
        cls._check_dangerous_patterns(body, "PR body")
        
        return body.strip()
    
    @classmethod
    def validate_webhook_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GitHub webhook payload."""
        if not isinstance(payload, dict):
            raise ValidationError("Webhook payload must be a dictionary")
        
        # Check payload size
        payload_size = len(json.dumps(payload))
        if payload_size > 1024 * 1024:  # 1MB limit
            raise ValidationError("Webhook payload too large")
        
        # Validate repository information
        if 'repository' in payload:
            repo_data = payload['repository']
            if 'full_name' in repo_data:
                cls.validate_repo_name(repo_data['full_name'])
        
        return payload
    
    @classmethod
    def validate_url(cls, url: str) -> str:
        """Validate URL for safety."""
        if not url or not isinstance(url, str):
            raise ValidationError("URL must be a non-empty string")
        
        try:
            parsed = urlparse(url)
        except Exception:
            raise ValidationError("Invalid URL format")
        
        # Only allow HTTPS for security
        if parsed.scheme not in ['https']:
            raise ValidationError("Only HTTPS URLs are allowed")
        
        # Validate hostname
        if not parsed.hostname:
            raise ValidationError("URL must have a hostname")
        
        # Block private/local addresses
        if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            raise ValidationError("Private/local URLs not allowed")
        
        return url.strip()
    
    @classmethod
    def validate_email(cls, email: str) -> str:
        """Validate email address."""
        if not email or not isinstance(email, str):
            raise ValidationError("Email must be a non-empty string")
        
        if not cls.EMAIL_PATTERN.match(email):
            raise ValidationError("Invalid email format")
        
        if len(email) > 254:
            raise ValidationError("Email too long")
        
        return email.strip().lower()
    
    @classmethod
    def sanitize_log_message(cls, message: str) -> str:
        """Sanitize message for safe logging."""
        if not isinstance(message, str):
            return str(message)
        
        # Remove potential log injection patterns
        sanitized = re.sub(r'[\r\n\t]', ' ', message)
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Truncate if too long
        if len(sanitized) > 1000:
            sanitized = sanitized[:997] + "..."
        
        return sanitized
    
    @classmethod
    def _check_dangerous_patterns(cls, text: str, field_name: str):
        """Check text for dangerous patterns."""
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.search(text):
                audit_logger.log_security_event(
                    event_type="dangerous_pattern_detected",
                    severity="high",
                    details={
                        "field": field_name,
                        "pattern": pattern.pattern,
                        "text_preview": text[:100]
                    }
                )
                raise ValidationError(f"Dangerous pattern detected in {field_name}")


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self):
        self.requests = {}  # IP -> [(timestamp, endpoint), ...]
        self.limits = {
            "default": (60, 60),  # 60 requests per 60 seconds
            "webhook": (100, 60),  # 100 webhooks per 60 seconds
            "manual_trigger": (10, 60),  # 10 manual triggers per 60 seconds
        }
    
    def is_allowed(self, identifier: str, endpoint: str = "default") -> bool:
        """Check if request is allowed based on rate limits."""
        import time
        
        current_time = time.time()
        limit, window = self.limits.get(endpoint, self.limits["default"])
        
        # Get request history for this identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        request_history = self.requests[identifier]
        
        # Remove old requests outside the window
        cutoff_time = current_time - window
        request_history[:] = [
            (timestamp, ep) for timestamp, ep in request_history
            if timestamp > cutoff_time
        ]
        
        # Count requests for this endpoint
        endpoint_requests = [
            req for req in request_history
            if req[1] == endpoint
        ]
        
        if len(endpoint_requests) >= limit:
            return False
        
        # Add current request
        request_history.append((current_time, endpoint))
        return True


# Global instances
input_validator = InputValidator()
rate_limiter = RateLimiter()