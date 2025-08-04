"""Secure secrets management."""

import os
import base64
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

from ..core.config import config
from ..monitoring.logging import get_logger, audit_logger

logger = get_logger(__name__)


class SecretsManager:
    """Secure secrets management for the bot."""
    
    def __init__(self):
        self._cipher = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption using the configured key."""
        try:
            # Use the encryption key from config
            key = config.encryption_key.encode()
            
            # Ensure key is proper length for Fernet (32 bytes)
            if len(key) != 32:
                # Derive key using PBKDF2
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'self_healing_bot_salt',
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(key))
            else:
                key = base64.urlsafe_b64encode(key)
            
            self._cipher = Fernet(key)
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value."""
        try:
            encrypted = self._cipher.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            raise
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a sensitive value."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self._cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            raise
    
    def store_secret(self, key: str, value: str, encrypted: bool = True) -> None:
        """Store a secret (in production, this would use a proper secret store)."""
        if encrypted:
            value = self.encrypt_value(value)
        
        # In production, this would store in a proper secrets management system
        # For now, we'll just log that we would store it
        audit_logger.log_security_event(
            event_type="secret_stored",
            severity="medium",
            details={"key": key, "encrypted": encrypted}
        )
    
    def get_secret(self, key: str, encrypted: bool = True) -> Optional[str]:
        """Retrieve a secret (in production, this would use a proper secret store)."""
        # In production, this would retrieve from a proper secrets management system
        # For now, return None to indicate secret not found
        return None
    
    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in dictionaries for logging."""
        sensitive_keys = {
            'password', 'token', 'key', 'secret', 'auth', 'credential',
            'private_key', 'access_token', 'refresh_token', 'api_key'
        }
        
        masked_data = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if key contains sensitive information
            is_sensitive = any(sensitive_word in key_lower for sensitive_word in sensitive_keys)
            
            if is_sensitive:
                if isinstance(value, str) and len(value) > 0:
                    # Show first 4 and last 4 characters, mask the rest
                    if len(value) > 8:
                        masked_data[key] = f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"
                    else:
                        masked_data[key] = "*" * len(value)
                else:
                    masked_data[key] = "***"
            elif isinstance(value, dict):
                # Recursively mask nested dictionaries
                masked_data[key] = self.mask_sensitive_data(value)
            else:
                masked_data[key] = value
        
        return masked_data


class SecretScanner:
    """Scanner for detecting secrets in code and logs."""
    
    # Patterns for detecting secrets
    SECRET_PATTERNS = [
        # GitHub tokens
        (r'ghp_[0-9a-zA-Z]{36}', 'GitHub Personal Access Token'),
        (r'ghs_[0-9a-zA-Z]{36}', 'GitHub App Installation Token'),
        (r'gho_[0-9a-zA-Z]{36}', 'GitHub OAuth Token'),
        
        # AWS credentials
        (r'AKIA[0-9A-Z]{16}', 'AWS Access Key ID'),
        (r'[0-9a-zA-Z/+]{40}', 'AWS Secret Access Key (potential)'),
        
        # API keys
        (r'sk-[0-9a-zA-Z]{48}', 'OpenAI API Key'),
        (r'xoxb-[0-9a-zA-Z-]{72}', 'Slack Bot Token'),
        (r'xoxp-[0-9a-zA-Z-]{72}', 'Slack User Token'),
        
        # Generic patterns
        (r'["\'][0-9a-zA-Z]{32,}["\']', 'Generic Long String (potential secret)'),
        (r'password\s*=\s*["\'][^"\']+["\']', 'Password in assignment'),
        (r'api_key\s*=\s*["\'][^"\']+["\']', 'API Key in assignment'),
    ]
    
    def __init__(self):
        import re
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), description)
            for pattern, description in self.SECRET_PATTERNS
        ]
    
    def scan_text(self, text: str) -> list:
        """Scan text for potential secrets."""
        findings = []
        
        for pattern, description in self.compiled_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                findings.append({
                    'type': description,
                    'match': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'line': text[:match.start()].count('\n') + 1
                })
        
        return findings
    
    def scan_file_content(self, file_path: str, content: str) -> list:
        """Scan file content for secrets."""
        findings = self.scan_text(content)
        
        if findings:
            audit_logger.log_security_event(
                event_type="secrets_detected_in_file",
                severity="high",
                details={
                    "file_path": file_path,
                    "findings_count": len(findings),
                    "secret_types": [f['type'] for f in findings]
                }
            )
        
        return findings
    
    def validate_pr_changes(self, file_changes: Dict[str, str]) -> Dict[str, list]:
        """Validate PR changes for secrets."""
        all_findings = {}
        
        for file_path, content in file_changes.items():
            findings = self.scan_file_content(file_path, content)
            if findings:
                all_findings[file_path] = findings
        
        return all_findings


# Environment variable protection
class EnvironmentProtector:
    """Protect against environment variable leakage."""
    
    SENSITIVE_ENV_VARS = {
        'GITHUB_PRIVATE_KEY', 'GITHUB_PRIVATE_KEY_PATH', 'GITHUB_WEBHOOK_SECRET',
        'GITHUB_TOKEN', 'SECRET_KEY', 'ENCRYPTION_KEY', 'DATABASE_URL',
        'REDIS_URL', 'SLACK_WEBHOOK_URL', 'SLACK_BOT_TOKEN', 'WANDB_API_KEY'
    }
    
    @classmethod
    def filter_environment(cls, env_dict: Dict[str, str]) -> Dict[str, str]:
        """Filter out sensitive environment variables."""
        filtered = {}
        
        for key, value in env_dict.items():
            key_upper = key.upper()
            
            if key_upper in cls.SENSITIVE_ENV_VARS:
                filtered[key] = "***REDACTED***"
            elif any(sensitive in key_upper for sensitive in ['TOKEN', 'KEY', 'SECRET', 'PASSWORD']):
                filtered[key] = "***REDACTED***"
            else:
                filtered[key] = value
        
        return filtered
    
    @classmethod
    def safe_environment_dump(cls) -> Dict[str, str]:
        """Get a safe dump of environment variables for debugging."""
        return cls.filter_environment(dict(os.environ))


# Global instances
secrets_manager = SecretsManager()
secret_scanner = SecretScanner()