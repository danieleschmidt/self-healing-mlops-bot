"""Input validation and sanitization."""

import re
import time
import asyncio
import hashlib
import hmac
import ipaddress
import html
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from urllib.parse import urlparse, unquote
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from fastapi import Request
from ..monitoring.logging import get_logger, audit_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    def __init__(self, message: str, error_type: str = "validation_error", severity: str = "medium"):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity

class SecurityThreat(Enum):
    """Security threat types."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    CODE_INJECTION = "code_injection"
    LDAP_INJECTION = "ldap_injection"
    XML_INJECTION = "xml_injection"
    SSRF = "ssrf"
    HEADER_INJECTION = "header_injection"
    LOG_INJECTION = "log_injection"


class InputValidator:
    """Comprehensive input validation for bot operations."""
    
    # Regex patterns for validation
    GITHUB_REPO_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$')
    GITHUB_BRANCH_PATTERN = re.compile(r'^[a-zA-Z0-9_./\-]+$')
    FILE_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9_./\-]+$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    # Advanced threat detection patterns
    SECURITY_PATTERNS = {
        SecurityThreat.SQL_INJECTION: [
            re.compile(r"\b(?:union|select|insert|update|delete|drop|create|alter)\s*\(", re.IGNORECASE),
            re.compile(r"\b(?:or|and)\s+\d+\s*=\s*\d+", re.IGNORECASE),
            re.compile(r"'\s*(?:or|and|union|select)\s+", re.IGNORECASE),
            re.compile(r"\d+\s*;\s*(?:drop|delete|update|insert)", re.IGNORECASE),
            re.compile(r"\b(?:benchmark|sleep|waitfor)\s*\(", re.IGNORECASE),
            re.compile(r"\b(?:information_schema|sys\.)", re.IGNORECASE),
            re.compile(r"--\s*$", re.MULTILINE),  # SQL comments
            re.compile(r"/\*.*\*/", re.DOTALL),  # SQL comments
        ],
        SecurityThreat.XSS: [
            re.compile(r'<script[^>]*>', re.IGNORECASE),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'<iframe[^>]*>', re.IGNORECASE),
            re.compile(r'<object[^>]*>', re.IGNORECASE),
            re.compile(r'<embed[^>]*>', re.IGNORECASE),
            re.compile(r'<img[^>]*onerror', re.IGNORECASE),
            re.compile(r'data:text/html', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'expression\s*\(', re.IGNORECASE),
        ],
        SecurityThreat.COMMAND_INJECTION: [
            re.compile(r'[;&|`$()]', re.IGNORECASE),
            re.compile(r'\b(?:sh|bash|cmd|powershell|wget|curl)\b', re.IGNORECASE),
            re.compile(r'\$\(.*\)', re.IGNORECASE),
            re.compile(r'`.*`', re.IGNORECASE),
            re.compile(r'\|\s*\w+', re.IGNORECASE),
        ],
        SecurityThreat.PATH_TRAVERSAL: [
            re.compile(r'\.\./'),
            re.compile(r'\.\.\\'),
            re.compile(r'%2e%2e%2f', re.IGNORECASE),
            re.compile(r'%2e%2e\\', re.IGNORECASE),
            re.compile(r'/etc/passwd'),
            re.compile(r'\\windows\\system32', re.IGNORECASE),
        ],
        SecurityThreat.CODE_INJECTION: [
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'exec\s*\(', re.IGNORECASE),
            re.compile(r'__import__', re.IGNORECASE),
            re.compile(r'import\s+os', re.IGNORECASE),
            re.compile(r'import\s+subprocess', re.IGNORECASE),
            re.compile(r'getattr\s*\(', re.IGNORECASE),
            re.compile(r'setattr\s*\(', re.IGNORECASE),
        ],
        SecurityThreat.LDAP_INJECTION: [
            re.compile(r'[()=*|&]', re.IGNORECASE),
            re.compile(r'\*\)', re.IGNORECASE),
        ],
        SecurityThreat.XML_INJECTION: [
            re.compile(r'<!\[CDATA\[', re.IGNORECASE),
            re.compile(r'<!DOCTYPE', re.IGNORECASE),
            re.compile(r'<!ENTITY', re.IGNORECASE),
        ],
        SecurityThreat.SSRF: [
            re.compile(r'https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|10\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[01]\.|192\.168\.)', re.IGNORECASE),
            re.compile(r'file://', re.IGNORECASE),
            re.compile(r'ftp://', re.IGNORECASE),
        ],
        SecurityThreat.HEADER_INJECTION: [
            re.compile(r'\r\n', re.IGNORECASE),
            re.compile(r'\n\r', re.IGNORECASE),
            re.compile(r'%0d%0a', re.IGNORECASE),
            re.compile(r'%0a%0d', re.IGNORECASE),
        ],
        SecurityThreat.LOG_INJECTION: [
            re.compile(r'\r\n.*(?:INFO|ERROR|DEBUG|WARN)', re.IGNORECASE),
            re.compile(r'\n.*(?:INFO|ERROR|DEBUG|WARN)', re.IGNORECASE),
        ]
    }
    
    # Whitelist patterns for safe content
    SAFE_PATTERNS = {
        'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
        'alphanumeric_underscore': re.compile(r'^[a-zA-Z0-9_]+$'),
        'alphanumeric_dash': re.compile(r'^[a-zA-Z0-9\-]+$'),
        'safe_filename': re.compile(r'^[a-zA-Z0-9._\-]+$'),
        'safe_path': re.compile(r'^[a-zA-Z0-9._/\-]+$'),
        'uuid': re.compile(r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$'),
        'semantic_version': re.compile(r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+)?$'),
    }
    
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
    def detect_security_threats(cls, text: str, field_name: str) -> List[Tuple[SecurityThreat, str]]:
        """Detect security threats in text."""
        threats = []
        
        for threat_type, patterns in cls.SECURITY_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    threats.append((threat_type, match.group()))
                    audit_logger.log_security_event(
                        event_type=f"security_threat_detected",
                        severity="high",
                        details={
                            "threat_type": threat_type.value,
                            "field": field_name,
                            "pattern": pattern.pattern,
                            "match": match.group()[:50],
                            "text_preview": text[:100]
                        }
                    )
        
        return threats
    
    @classmethod
    def _check_dangerous_patterns(cls, text: str, field_name: str):
        """Check text for dangerous patterns."""
        threats = cls.detect_security_threats(text, field_name)
        if threats:
            threat_types = [t[0].value for t in threats]
            raise ValidationError(
                f"Security threats detected in {field_name}: {', '.join(threat_types)}",
                error_type="security_threat",
                severity="high"
            )
    
    @classmethod
    def sanitize_html(cls, text: str) -> str:
        """Sanitize HTML content to prevent XSS."""
        if not isinstance(text, str):
            return str(text)
        
        # HTML entity encoding
        sanitized = html.escape(text, quote=True)
        
        # Additional XSS prevention
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @classmethod
    def sanitize_sql(cls, text: str) -> str:
        """Sanitize text to prevent SQL injection."""
        if not isinstance(text, str):
            return str(text)
        
        # Escape single quotes
        sanitized = text.replace("'", "''")
        
        # Remove SQL comments
        sanitized = re.sub(r'--.*$', '', sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)
        
        return sanitized
    
    @classmethod
    def sanitize_command(cls, text: str) -> str:
        """Sanitize text to prevent command injection."""
        if not isinstance(text, str):
            return str(text)
        
        # Remove dangerous characters
        dangerous_chars = [';', '&', '|', '`', '$', '(', ')', '\n', '\r']
        sanitized = text
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized
    
    @classmethod
    def validate_against_whitelist(cls, text: str, pattern_name: str) -> bool:
        """Validate text against whitelist patterns."""
        if pattern_name not in cls.SAFE_PATTERNS:
            raise ValidationError(f"Unknown whitelist pattern: {pattern_name}")
        
        pattern = cls.SAFE_PATTERNS[pattern_name]
        return bool(pattern.match(text))
    
    @classmethod
    def validate_data_integrity(cls, data: Dict[str, Any], expected_hash: Optional[str] = None) -> bool:
        """Validate data integrity using hash comparison."""
        if expected_hash is None:
            return True
        
        # Create hash of the data
        data_str = json.dumps(data, sort_keys=True)
        calculated_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        is_valid = hmac.compare_digest(calculated_hash, expected_hash)
        
        if not is_valid:
            audit_logger.log_security_event(
                event_type="data_integrity_violation",
                severity="critical",
                details={
                    "expected_hash": expected_hash,
                    "calculated_hash": calculated_hash
                }
            )
        
        return is_valid
    
    @classmethod
    def validate_ip_address(cls, ip_str: str, allow_private: bool = False) -> str:
        """Validate and normalize IP address."""
        try:
            ip = ipaddress.ip_address(ip_str)
            
            if not allow_private and ip.is_private:
                raise ValidationError(f"Private IP address not allowed: {ip}")
            
            if ip.is_loopback:
                raise ValidationError(f"Loopback IP address not allowed: {ip}")
            
            if ip.is_multicast:
                raise ValidationError(f"Multicast IP address not allowed: {ip}")
            
            return str(ip)
            
        except ipaddress.AddressValueError:
            raise ValidationError(f"Invalid IP address: {ip_str}")
    
    @classmethod
    def validate_json_payload(cls, payload: str, max_depth: int = 10, max_size: int = 1024*1024) -> Dict[str, Any]:
        """Validate JSON payload with size and depth limits."""
        if len(payload) > max_size:
            raise ValidationError(f"JSON payload too large: {len(payload)} bytes")
        
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}")
        
        # Check nesting depth
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                raise ValidationError(f"JSON nesting too deep: {current_depth}")
            
            if isinstance(obj, dict):
                for value in obj.values():
                    check_depth(value, current_depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, current_depth + 1)
        
        check_depth(data)
        return data
    
    @classmethod
    def validate_base64(cls, data: str, max_size: int = 1024*1024) -> bytes:
        """Validate and decode base64 data."""
        if len(data) > max_size * 4 // 3:  # Base64 overhead
            raise ValidationError(f"Base64 data too large")
        
        try:
            import base64
            decoded = base64.b64decode(data, validate=True)
            return decoded
        except Exception as e:
            raise ValidationError(f"Invalid base64 data: {e}")
    
    @classmethod
    def validate_regex_pattern(cls, pattern: str) -> re.Pattern:
        """Validate and compile regex pattern safely."""
        # Check for ReDoS patterns
        redos_patterns = [
            r'\(\?\!\.\*\)',  # Catastrophic backtracking
            r'\(\?\=\.\*\)',  # Excessive lookaheads
            r'\(\?\<\=\.\*\)', # Excessive lookbehinds
            r'\(\.\*\)\+',    # Nested quantifiers
        ]
        
        for redos in redos_patterns:
            if re.search(redos, pattern):
                raise ValidationError("Potentially dangerous regex pattern detected")
        
        try:
            compiled = re.compile(pattern)
            return compiled
        except re.error as e:
            raise ValidationError(f"Invalid regex pattern: {e}")
    
    @classmethod
    def validate_mime_type(cls, content: bytes, allowed_types: Set[str]) -> str:
        """Validate file content MIME type."""
        import mimetypes
        
        # Simple magic byte detection
        magic_signatures = {
            b'\x89PNG\r\n\x1a\n': 'image/png',
            b'\xff\xd8\xff': 'image/jpeg',
            b'GIF87a': 'image/gif',
            b'GIF89a': 'image/gif',
            b'%PDF': 'application/pdf',
            b'PK\x03\x04': 'application/zip',
        }
        
        detected_type = 'application/octet-stream'  # Default
        
        for signature, mime_type in magic_signatures.items():
            if content.startswith(signature):
                detected_type = mime_type
                break
        
        if detected_type not in allowed_types:
            raise ValidationError(f"MIME type not allowed: {detected_type}")
        
        return detected_type
    
    @classmethod
    def validate_content_length(cls, content: Union[str, bytes], max_length: int) -> Union[str, bytes]:
        """Validate content length."""
        length = len(content)
        if length > max_length:
            raise ValidationError(f"Content too large: {length} > {max_length}")
        return content


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    requests_per_minute: int
    burst_allowance: int = field(default_factory=lambda: 10)
    block_duration_minutes: int = field(default_factory=lambda: 5)

@dataclass
class ClientInfo:
    """Client request tracking information."""
    request_times: deque = field(default_factory=deque)
    blocked_until: Optional[datetime] = None
    total_requests: int = 0
    violations: int = 0

class RateLimiter:
    """Advanced rate limiting for API endpoints with burst protection."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.clients: Dict[str, ClientInfo] = defaultdict(ClientInfo)
        self.rules = {
            "default": RateLimitRule(requests_per_minute=requests_per_minute),
            "webhook": RateLimitRule(requests_per_minute=100, burst_allowance=20),
            "manual_trigger": RateLimitRule(requests_per_minute=10, burst_allowance=5),
            "health": RateLimitRule(requests_per_minute=120, burst_allowance=30),
            "metrics": RateLimitRule(requests_per_minute=30, burst_allowance=10),
        }
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self, client_id: str, endpoint: str = "default") -> bool:
        """Check if request is allowed based on rate limits."""
        async with self._lock:
            now = datetime.utcnow()
            client = self.clients[client_id]
            rule = self.rules.get(endpoint, self.rules["default"])
            
            # Check if client is currently blocked
            if client.blocked_until and now < client.blocked_until:
                audit_logger.log_security_event(
                    "rate_limit_blocked_request", "warning",
                    {
                        "client_id": client_id,
                        "endpoint": endpoint,
                        "blocked_until": client.blocked_until.isoformat()
                    }
                )
                return False
            
            # Clear expired block
            if client.blocked_until and now >= client.blocked_until:
                client.blocked_until = None
                logger.info(f"Rate limit block expired for client {client_id}")
            
            # Clean old requests (older than 1 minute)
            cutoff_time = now - timedelta(minutes=1)
            while client.request_times and client.request_times[0] < cutoff_time:
                client.request_times.popleft()
            
            # Count recent requests
            recent_requests = len(client.request_times)
            
            # Check rate limit
            if recent_requests >= rule.requests_per_minute:
                client.violations += 1
                
                # Block client if too many violations
                if client.violations >= 3:
                    client.blocked_until = now + timedelta(minutes=rule.block_duration_minutes)
                    logger.warning(
                        f"Client {client_id} blocked for {rule.block_duration_minutes} minutes due to rate limit violations"
                    )
                
                audit_logger.log_security_event(
                    "rate_limit_exceeded", "warning",
                    {
                        "client_id": client_id,
                        "endpoint": endpoint,
                        "recent_requests": recent_requests,
                        "limit": rule.requests_per_minute,
                        "violations": client.violations
                    }
                )
                return False
            
            # Check burst limit
            burst_window = now - timedelta(seconds=10)
            burst_requests = sum(1 for req_time in client.request_times if req_time > burst_window)
            
            if burst_requests >= rule.burst_allowance:
                audit_logger.log_security_event(
                    "rate_limit_burst_exceeded", "warning",
                    {
                        "client_id": client_id,
                        "endpoint": endpoint,
                        "burst_requests": burst_requests,
                        "burst_limit": rule.burst_allowance
                    }
                )
                return False
            
            # Allow request and record it
            client.request_times.append(now)
            client.total_requests += 1
            return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        async with self._lock:
            stats = {
                "total_clients": len(self.clients),
                "blocked_clients": sum(1 for c in self.clients.values() if c.blocked_until),
                "total_requests": sum(c.total_requests for c in self.clients.values()),
                "total_violations": sum(c.violations for c in self.clients.values()),
                "rules": {
                    name: {
                        "requests_per_minute": rule.requests_per_minute,
                        "burst_allowance": rule.burst_allowance,
                        "block_duration_minutes": rule.block_duration_minutes
                    }
                    for name, rule in self.rules.items()
                }
            }
            return stats
    
    async def reset_client(self, client_id: str):
        """Reset rate limiting for a specific client."""
        async with self._lock:
            if client_id in self.clients:
                self.clients[client_id] = ClientInfo()
                logger.info(f"Rate limit reset for client {client_id}")

class SecurityValidator:
    """Enhanced security validation for web requests."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.blocked_ips = set()
        self.suspicious_user_agents = {
            'bot', 'crawler', 'spider', 'scraper', 'scanner',
            'sqlmap', 'nikto', 'nessus', 'burp', 'owasp'
        }
        self.allowed_content_types = {
            'application/json',
            'application/x-www-form-urlencoded',
            'multipart/form-data',
            'text/plain'
        }
        self.csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        self.request_anomaly_threshold = 10  # Anomaly detection threshold
    
    async def validate_request(self, request: Request) -> bool:
        """Validate incoming request for security issues."""
        client_ip = request.client.host if request.client else "unknown"
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            audit_logger.log_security_event(
                "blocked_ip_access_attempt", "critical",
                {"client_ip": client_ip, "path": request.url.path}
            )
            raise ValidationError(f"IP {client_ip} is blocked", "access_denied", "critical")
        
        # Validate IP address
        try:
            self.input_validator.validate_ip_address(client_ip, allow_private=True)
        except ValidationError as e:
            audit_logger.log_security_event(
                "invalid_client_ip", "warning",
                {"client_ip": client_ip, "error": str(e)}
            )
        
        # Check request size
        content_length = int(request.headers.get("content-length", 0))
        if content_length > self.max_request_size:
            audit_logger.log_security_event(
                "request_too_large", "warning",
                {"client_ip": client_ip, "size": content_length, "path": request.url.path}
            )
            raise ValidationError("Request too large", "request_size_exceeded", "medium")
        
        # Validate content type
        content_type = request.headers.get("content-type", "").split(";")[0].strip()
        if content_type and content_type not in self.allowed_content_types:
            audit_logger.log_security_event(
                "invalid_content_type", "warning",
                {"client_ip": client_ip, "content_type": content_type}
            )
            raise ValidationError(f"Content type not allowed: {content_type}", "invalid_content_type", "medium")
        
        # Validate headers
        await self._validate_headers(request)
        
        # Check for security headers
        self._check_security_headers(request)
        
        # Validate URL path
        self._validate_path(request.url.path)
        
        # Validate query parameters
        for key, value in request.query_params.items():
            self._validate_parameter(key, str(value))
        
        # Detect anomalous requests
        await self._detect_request_anomalies(request)
        
        return True
    
    async def _validate_headers(self, request: Request):
        """Validate request headers."""
        dangerous_headers = {
            "x-forwarded-for": self._validate_forwarded_header,
            "user-agent": self._validate_user_agent,
            "referer": self._validate_referer,
        }
        
        for header_name, validator in dangerous_headers.items():
            if header_name in request.headers:
                validator(request.headers[header_name])
    
    def _validate_forwarded_header(self, value: str):
        """Validate X-Forwarded-For header."""
        # Basic IP validation
        ips = [ip.strip() for ip in value.split(",")]
        for ip in ips:
            if not re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ip):
                if not re.match(r'^[0-9a-fA-F:]+$', ip):  # IPv6 basic check
                    raise ValueError(f"Invalid IP in X-Forwarded-For: {ip}")
    
    def _validate_user_agent(self, value: str):
        """Validate User-Agent header."""
        if len(value) > 500:
            raise ValidationError("User-Agent header too long", "invalid_header", "medium")
        
        # Check for suspicious user agents
        value_lower = value.lower()
        for suspicious in self.suspicious_user_agents:
            if suspicious in value_lower:
                audit_logger.log_security_event(
                    "suspicious_user_agent", "high",
                    {"user_agent": value[:100]}
                )
                raise ValidationError("Suspicious User-Agent detected", "suspicious_user_agent", "high")
        
        # Check for threat patterns
        threats = self.input_validator.detect_security_threats(value, "user-agent")
        if threats:
            raise ValidationError("Security threats in User-Agent", "security_threat", "high")
    
    def _validate_referer(self, value: str):
        """Validate Referer header."""
        if value and len(value) > 1000:
            raise ValueError("Referer header too long")
        
        # Basic URL validation if present
        if value:
            try:
                parsed = urlparse(value)
                if parsed.scheme not in ['http', 'https', '']:
                    raise ValueError("Invalid referer scheme")
            except Exception:
                raise ValueError("Invalid referer URL")
    
    def _validate_path(self, path: str):
        """Validate URL path."""
        if len(path) > 1000:
            raise ValidationError("URL path too long", "invalid_path", "medium")
        
        # URL decode path to catch encoded attacks
        decoded_path = urllib.parse.unquote(path)
        
        # Check for threat patterns in both encoded and decoded paths
        for path_to_check in [path, decoded_path]:
            threats = self.input_validator.detect_security_threats(path_to_check, "url_path")
            if threats:
                raise ValidationError(f"Security threats in URL path: {[t[0].value for t in threats]}", "security_threat", "high")
    
    def _validate_parameter(self, name: str, value: str):
        """Validate query parameter."""
        if len(name) > 100 or len(value) > 1000:
            raise ValidationError("Parameter name or value too long", "invalid_parameter", "medium")
        
        # URL decode parameter value to catch encoded attacks
        decoded_value = urllib.parse.unquote(value)
        
        # Check for threat patterns in both encoded and decoded values
        for value_to_check in [value, decoded_value]:
            threats = self.input_validator.detect_security_threats(value_to_check, f"parameter_{name}")
            if threats:
                raise ValidationError(f"Security threats in parameter {name}: {[t[0].value for t in threats]}", "security_threat", "high")
    
    def block_ip(self, ip: str, reason: str = ""):
        """Block an IP address."""
        self.blocked_ips.add(ip)
        audit_logger.log_security_event(
            "ip_blocked", "critical",
            {"ip": ip, "reason": reason}
        )
        logger.warning(f"IP {ip} blocked: {reason}")
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address."""
        self.blocked_ips.discard(ip)
        logger.info(f"IP {ip} unblocked")
    
    def _check_security_headers(self, request: Request):
        """Check for required security headers."""
        security_headers = {
            'x-content-type-options': 'nosniff',
            'x-frame-options': 'DENY',
            'x-xss-protection': '1; mode=block',
            'strict-transport-security': 'max-age=31536000; includeSubDomains',
        }
        
        missing_headers = []
        for header_name, expected_value in security_headers.items():
            if header_name not in request.headers:
                missing_headers.append(header_name)
        
        if missing_headers:
            audit_logger.log_security_event(
                "missing_security_headers", "info",
                {
                    "missing_headers": missing_headers,
                    "client_ip": request.client.host if request.client else "unknown"
                }
            )
    
    async def _detect_request_anomalies(self, request: Request):
        """Detect anomalous request patterns."""
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for rapid-fire requests (basic anomaly detection)
        # In production, this would use more sophisticated ML-based detection
        
        # Check for unusual request patterns
        anomaly_indicators = []
        
        # Unusual path patterns
        path = request.url.path
        if len(path.split('/')) > 10:
            anomaly_indicators.append("deep_path_nesting")
        
        # Unusual query parameter count
        if len(request.query_params) > 20:
            anomaly_indicators.append("excessive_query_params")
        
        # Unusual header count
        if len(request.headers) > 50:
            anomaly_indicators.append("excessive_headers")
        
        # Check for base64 encoded content in parameters
        for key, value in request.query_params.items():
            try:
                if len(value) > 20 and len(value) % 4 == 0:
                    import base64
                    base64.b64decode(value, validate=True)
                    anomaly_indicators.append("base64_parameter")
            except Exception:
                pass
        
        if anomaly_indicators:
            severity = "high" if len(anomaly_indicators) >= self.request_anomaly_threshold else "medium"
            audit_logger.log_security_event(
                "request_anomaly_detected", severity,
                {
                    "client_ip": client_ip,
                    "path": path,
                    "anomaly_indicators": anomaly_indicators,
                    "indicator_count": len(anomaly_indicators)
                }
            )
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get recommended security headers for responses."""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': self.csp_policy,
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
        }
    
    def validate_webhook_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Validate GitHub webhook signature."""
        if not signature.startswith('sha256='):
            return False
        
        expected_signature = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        received_signature = signature.split('=')[1]
        
        return hmac.compare_digest(expected_signature, received_signature)


# Global instances
input_validator = InputValidator()