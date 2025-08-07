"""Security validation and threat detection system."""

import re
import ast
import hashlib
import hmac
import base64
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import secrets
import json

from ..monitoring.logging import get_logger, security_logger
from ..core.config import config

logger = get_logger(__name__)


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    threat_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    remediation_steps: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of security validation."""
    is_valid: bool
    threats: List[SecurityThreat] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None


class SecurityValidator:
    """Comprehensive security validation for bot inputs and actions."""
    
    def __init__(self):
        self.dangerous_patterns = self._load_dangerous_patterns()
        self.sensitive_patterns = self._load_sensitive_patterns()
        self.allowed_file_extensions = {'.py', '.yaml', '.yml', '.json', '.md', '.txt', '.sh', '.dockerfile'}
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.rate_limiters = {}
        
    def _load_dangerous_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns for dangerous code/commands."""
        return [
            {
                "pattern": r"(rm\s+-rf|rm\s+-r|rmdir)",
                "type": "dangerous_command",
                "severity": "critical",
                "description": "Potentially destructive file deletion command"
            },
            {
                "pattern": r"(exec\s*\(|eval\s*\(|__import__)",
                "type": "code_injection",
                "severity": "critical", 
                "description": "Code injection attempt detected"
            },
            {
                "pattern": r"(DROP\s+TABLE|DELETE\s+FROM|TRUNCATE)",
                "type": "sql_injection",
                "severity": "high",
                "description": "Potentially dangerous SQL command"
            },
            {
                "pattern": r"(<script|javascript:|vbscript:|onload=|onerror=)",
                "type": "xss_attempt",
                "severity": "high",
                "description": "Cross-site scripting attempt"
            },
            {
                "pattern": r"(\.\./|\.\.\\\\|/etc/passwd|/etc/shadow)",
                "type": "path_traversal",
                "severity": "high",
                "description": "Path traversal attempt detected"
            },
            {
                "pattern": r"(curl\s+.*\|.*sh|wget\s+.*\|.*sh)",
                "type": "remote_execution",
                "severity": "critical",
                "description": "Remote code execution attempt"
            }
        ]
    
    def _load_sensitive_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns for sensitive data."""
        return [
            {
                "pattern": r"[A-Za-z0-9]{20,}",  # API keys
                "type": "api_key",
                "severity": "medium",
                "description": "Potential API key detected"
            },
            {
                "pattern": r"-----BEGIN [A-Z]+ PRIVATE KEY-----",
                "type": "private_key",
                "severity": "critical",
                "description": "Private key detected"
            },
            {
                "pattern": r"(password|passwd|pwd)\s*[:=]\s*[\"']?[^\\s\"']+",
                "type": "password",
                "severity": "high",
                "description": "Password in plaintext detected"
            },
            {
                "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
                "type": "email",
                "severity": "low",
                "description": "Email address detected"
            },
            {
                "pattern": r"\\b\\d{3}-\\d{2}-\\d{4}\\b",
                "type": "ssn",
                "severity": "critical",
                "description": "Social Security Number detected"
            }
        ]
    
    async def validate_webhook_payload(self, payload: bytes, signature: str) -> ValidationResult:
        """Validate GitHub webhook payload authenticity."""
        threats = []
        
        try:
            # Verify HMAC signature
            expected_signature = self._calculate_signature(payload)
            
            if not hmac.compare_digest(signature, expected_signature):
                threats.append(SecurityThreat(
                    threat_type="invalid_signature",
                    severity="critical",
                    description="Webhook signature verification failed",
                    evidence={"provided_signature": signature[:20] + "..."},
                    remediation_steps=["Reject the webhook", "Log the attempt", "Check webhook configuration"]
                ))
                
                security_logger.log_authentication("webhook_signature_failed", False, {
                    "provided_signature": signature[:20] + "...",
                    "payload_size": len(payload)
                })
                
                return ValidationResult(is_valid=False, threats=threats)
            
            # Parse and validate payload structure
            try:
                payload_data = json.loads(payload.decode('utf-8'))
                structure_validation = await self._validate_payload_structure(payload_data)
                
                if not structure_validation.is_valid:
                    threats.extend(structure_validation.threats)
                
            except json.JSONDecodeError as e:
                threats.append(SecurityThreat(
                    threat_type="malformed_json",
                    severity="high",
                    description="Malformed JSON in webhook payload",
                    evidence={"error": str(e)},
                    remediation_steps=["Reject the payload", "Log the malformed data"]
                ))
            
            security_logger.log_authentication("webhook_signature_verified", True, {
                "payload_size": len(payload)
            })
            
            return ValidationResult(is_valid=len(threats) == 0, threats=threats)
            
        except Exception as e:
            logger.exception(f"Error validating webhook payload: {e}")
            threats.append(SecurityThreat(
                threat_type="validation_error",
                severity="medium",
                description=f"Error during payload validation: {str(e)}",
                remediation_steps=["Review validation logic", "Log the error"]
            ))
            
            return ValidationResult(is_valid=False, threats=threats)
    
    def _calculate_signature(self, payload: bytes) -> str:
        """Calculate HMAC signature for webhook payload."""
        secret = config.github_webhook_secret.encode('utf-8')
        signature = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        return f"sha256={signature}"
    
    async def _validate_payload_structure(self, payload_data: Dict[str, Any]) -> ValidationResult:
        """Validate the structure and content of webhook payload."""
        threats = []
        warnings = []
        
        # Check for required fields based on event type
        if 'repository' not in payload_data:
            threats.append(SecurityThreat(
                threat_type="missing_required_field",
                severity="medium",
                description="Repository information missing from payload",
                remediation_steps=["Verify webhook configuration"]
            ))
        
        # Validate repository information
        if 'repository' in payload_data:
            repo_validation = self._validate_repository_info(payload_data['repository'])
            threats.extend(repo_validation.threats)
            warnings.extend(repo_validation.warnings)
        
        # Check payload size
        payload_str = json.dumps(payload_data)
        if len(payload_str.encode('utf-8')) > 1024 * 1024:  # 1MB limit
            threats.append(SecurityThreat(
                threat_type="oversized_payload",
                severity="medium",
                description="Webhook payload exceeds size limit",
                evidence={"size": len(payload_str)},
                remediation_steps=["Check for data injection", "Implement payload size limits"]
            ))
        
        return ValidationResult(is_valid=len(threats) == 0, threats=threats, warnings=warnings)
    
    def _validate_repository_info(self, repo_data: Dict[str, Any]) -> ValidationResult:
        """Validate repository information."""
        threats = []
        warnings = []
        
        # Check for suspicious repository names
        if 'full_name' in repo_data:
            full_name = repo_data['full_name']
            
            # Check for path traversal attempts
            if '../' in full_name or '..\\\\' in full_name:
                threats.append(SecurityThreat(
                    threat_type="path_traversal",
                    severity="high",
                    description="Path traversal attempt in repository name",
                    evidence={"repo_name": full_name},
                    remediation_steps=["Reject the request", "Sanitize repository name"]
                ))
            
            # Check for overly long names (potential buffer overflow)
            if len(full_name) > 255:
                threats.append(SecurityThreat(
                    threat_type="buffer_overflow_attempt",
                    severity="medium",
                    description="Repository name exceeds reasonable length",
                    evidence={"repo_name_length": len(full_name)},
                    remediation_steps=["Truncate repository name", "Implement length limits"]
                ))
        
        return ValidationResult(is_valid=len(threats) == 0, threats=threats, warnings=warnings)
    
    async def validate_file_content(self, file_path: str, content: str) -> ValidationResult:
        """Validate file content for security threats."""
        threats = []
        warnings = []
        
        # Check file extension
        file_ext = self._get_file_extension(file_path)
        if file_ext not in self.allowed_file_extensions:
            threats.append(SecurityThreat(
                threat_type="forbidden_file_type",
                severity="medium",
                description=f"File type not allowed: {file_ext}",
                evidence={"file_path": file_path, "extension": file_ext},
                remediation_steps=["Review file type whitelist", "Reject file upload"]
            ))
        
        # Check file size
        if len(content.encode('utf-8')) > self.max_file_size:
            threats.append(SecurityThreat(
                threat_type="oversized_file",
                severity="medium",
                description="File exceeds size limit",
                evidence={"file_path": file_path, "size": len(content)},
                remediation_steps=["Implement file size limits", "Chunk large files"]
            ))
        
        # Scan for dangerous patterns
        dangerous_matches = self._scan_for_patterns(content, self.dangerous_patterns)
        for match in dangerous_matches:
            threats.append(SecurityThreat(
                threat_type=match["type"],
                severity=match["severity"],
                description=match["description"],
                evidence={"file_path": file_path, "matched_text": match["text"][:100]},
                remediation_steps=["Review code changes", "Sanitize input", "Manual review required"]
            ))
        
        # Scan for sensitive data
        sensitive_matches = self._scan_for_patterns(content, self.sensitive_patterns)
        for match in sensitive_matches:
            if match["severity"] in ["high", "critical"]:
                threats.append(SecurityThreat(
                    threat_type=match["type"],
                    severity=match["severity"],
                    description=match["description"],
                    evidence={"file_path": file_path, "pattern_type": match["type"]},
                    remediation_steps=["Remove sensitive data", "Use environment variables", "Encrypt secrets"]
                ))
            else:
                warnings.append(f"Potential sensitive data detected: {match['description']}")
        
        # Validate Python code syntax if it's a Python file
        if file_ext == '.py':
            python_validation = self._validate_python_code(content, file_path)
            threats.extend(python_validation.threats)
            warnings.extend(python_validation.warnings)
        
        return ValidationResult(is_valid=len(threats) == 0, threats=threats, warnings=warnings)
    
    def _get_file_extension(self, file_path: str) -> str:
        """Get file extension from path."""
        return '.' + file_path.split('.')[-1].lower() if '.' in file_path else ''
    
    def _scan_for_patterns(self, content: str, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scan content for dangerous patterns."""
        matches = []
        
        for pattern_config in patterns:
            pattern = pattern_config["pattern"]
            
            try:
                for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                    matches.append({
                        "type": pattern_config["type"],
                        "severity": pattern_config["severity"],
                        "description": pattern_config["description"],
                        "text": match.group(0),
                        "position": match.span()
                    })
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")
        
        return matches
    
    def _validate_python_code(self, content: str, file_path: str) -> ValidationResult:
        """Validate Python code for security issues."""
        threats = []
        warnings = []
        
        try:
            # Parse the Python code
            tree = ast.parse(content)
            
            # Walk through the AST to find dangerous constructs
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in ['exec', 'eval', '__import__', 'compile']:
                            threats.append(SecurityThreat(
                                threat_type="dangerous_function",
                                severity="critical",
                                description=f"Use of dangerous function: {func_name}",
                                evidence={"function": func_name, "file_path": file_path},
                                remediation_steps=["Remove dangerous function call", "Use safer alternatives"]
                            ))
                
                # Check for subprocess calls
                elif isinstance(node, ast.Attribute):
                    if node.attr in ['system', 'popen', 'spawn']:
                        threats.append(SecurityThreat(
                            threat_type="subprocess_call",
                            severity="high",
                            description=f"Subprocess call detected: {node.attr}",
                            evidence={"method": node.attr, "file_path": file_path},
                            remediation_steps=["Review subprocess usage", "Sanitize inputs", "Use safe alternatives"]
                        ))
                
                # Check for import statements
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['os', 'subprocess', 'sys', 'shutil']:
                            warnings.append(f"Import of potentially dangerous module: {alias.name}")
        
        except SyntaxError as e:
            warnings.append(f"Python syntax error: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing Python code: {e}")
        
        return ValidationResult(is_valid=len(threats) == 0, threats=threats, warnings=warnings)
    
    async def validate_configuration(self, config_data: Dict[str, Any]) -> ValidationResult:
        """Validate configuration changes for security."""
        threats = []
        warnings = []
        
        # Check for hardcoded secrets
        config_str = json.dumps(config_data, indent=2)
        sensitive_matches = self._scan_for_patterns(config_str, self.sensitive_patterns)
        
        for match in sensitive_matches:
            if match["severity"] in ["high", "critical"]:
                threats.append(SecurityThreat(
                    threat_type=match["type"],
                    severity=match["severity"],
                    description=f"Sensitive data in configuration: {match['description']}",
                    evidence={"pattern_type": match["type"]},
                    remediation_steps=["Use environment variables", "Encrypt sensitive values", "Remove from config"]
                ))
        
        # Check for dangerous configuration values
        dangerous_configs = [
            "debug: true",
            "ssl_verify: false",
            "disable_auth",
            "allow_all",
            "public_access"
        ]
        
        for dangerous_config in dangerous_configs:
            if dangerous_config.lower() in config_str.lower():
                threats.append(SecurityThreat(
                    threat_type="insecure_configuration",
                    severity="medium",
                    description=f"Potentially insecure configuration: {dangerous_config}",
                    remediation_steps=["Review security implications", "Use secure defaults"]
                ))
        
        return ValidationResult(is_valid=len(threats) == 0, threats=threats, warnings=warnings)
    
    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data by removing dangerous content."""
        if isinstance(data, str):
            # Remove potential script tags and dangerous characters
            sanitized = re.sub(r'<script.*?</script>', '', data, flags=re.IGNORECASE | re.DOTALL)
            sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'on\\w+\\s*=', '', sanitized, flags=re.IGNORECASE)
            return sanitized
        elif isinstance(data, dict):
            return {key: self.sanitize_input(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        else:
            return data
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> Dict[str, str]:
        """Hash sensitive data with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_value = hashlib.pbkdf2_hmac(
            'sha256',
            data.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return {
            "hash": base64.b64encode(hash_value).decode('utf-8'),
            "salt": salt
        }
    
    def verify_hash(self, data: str, hash_info: Dict[str, str]) -> bool:
        """Verify hashed data."""
        computed_hash = self.hash_sensitive_data(data, hash_info["salt"])
        return hmac.compare_digest(computed_hash["hash"], hash_info["hash"])


class RateLimiter:
    """Rate limiting for API endpoints and operations."""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests: Dict[str, List[float]] = {}
        self.blocked_until: Dict[str, float] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the given identifier."""
        current_time = datetime.utcnow().timestamp()
        
        # Check if currently blocked
        if identifier in self.blocked_until:
            if current_time < self.blocked_until[identifier]:
                return False
            else:
                # Unblock
                del self.blocked_until[identifier]
        
        # Clean old requests
        if identifier in self.requests:
            cutoff_time = current_time - self.window_seconds
            self.requests[identifier] = [
                timestamp for timestamp in self.requests[identifier]
                if timestamp > cutoff_time
            ]
        
        # Check rate limit
        request_count = len(self.requests.get(identifier, []))
        
        if request_count >= self.max_requests:
            # Block for window duration
            self.blocked_until[identifier] = current_time + self.window_seconds
            
            security_logger.log_access_attempt(
                f"rate_limit_exceeded_{identifier}",
                False,
                {"request_count": request_count, "window_seconds": self.window_seconds}
            )
            
            return False
        
        # Add current request
        if identifier not in self.requests:
            self.requests[identifier] = []
        self.requests[identifier].append(current_time)
        
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        current_time = datetime.utcnow().timestamp()
        
        # Check if blocked
        if identifier in self.blocked_until and current_time < self.blocked_until[identifier]:
            return 0
        
        # Clean old requests
        if identifier in self.requests:
            cutoff_time = current_time - self.window_seconds
            self.requests[identifier] = [
                timestamp for timestamp in self.requests[identifier]
                if timestamp > cutoff_time
            ]
        
        request_count = len(self.requests.get(identifier, []))
        return max(0, self.max_requests - request_count)


# Global security validator instance
security_validator = SecurityValidator()

# Rate limiters for different operations
webhook_rate_limiter = RateLimiter(max_requests=100, window_minutes=5)  # 100 webhooks per 5 minutes
api_rate_limiter = RateLimiter(max_requests=1000, window_minutes=1)     # 1000 API calls per minute