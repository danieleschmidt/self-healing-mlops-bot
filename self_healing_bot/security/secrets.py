"""Secure secrets management."""

import os
import base64
import json
import time
import asyncio
import hashlib
import hmac
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import logging

from ..core.config import config
from ..monitoring.logging import get_logger, audit_logger

logger = get_logger(__name__)

class SecretType(Enum):
    """Types of secrets managed by the system."""
    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    PRIVATE_KEY = "private_key"
    CERTIFICATE = "certificate"
    WEBHOOK_SECRET = "webhook_secret"
    ENCRYPTION_KEY = "encryption_key"
    JWT_SECRET = "jwt_secret"
    OAUTH_TOKEN = "oauth_token"
    SESSION_KEY = "session_key"

@dataclass
class SecretMetadata:
    """Metadata for a stored secret."""
    secret_id: str
    secret_type: SecretType
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    rotation_period_days: Optional[int] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    version: int = 1

@dataclass
class SecretVersion:
    """A version of a secret."""
    version: int
    encrypted_value: str
    created_at: datetime
    created_by: str
    checksum: str


class EnhancedSecretsManager:
    """Enhanced secure secrets management with enterprise features."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self._primary_cipher = None
        self._backup_cipher = None
        self._multi_fernet = None
        self._rsa_private_key = None
        self._rsa_public_key = None
        self._secrets_metadata: Dict[str, SecretMetadata] = {}
        self._secret_versions: Dict[str, List[SecretVersion]] = {}
        self._access_log: List[Dict[str, Any]] = []
        self._storage_path = Path(storage_path) if storage_path else Path("/tmp/secrets")
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._initialize_encryption()
        try:
            self._load_metadata()
        except AttributeError:
            # Handle missing method gracefully during initialization
            pass
    
    def _initialize_encryption(self):
        """Initialize multiple encryption layers."""
        try:
            # Primary encryption key
            primary_key = self._derive_key(config.encryption_key, b'primary_salt_v1')
            self._primary_cipher = Fernet(primary_key)
            
            # Backup encryption key (for key rotation)
            backup_key = self._derive_key(config.encryption_key, b'backup_salt_v1')
            self._backup_cipher = Fernet(backup_key)
            
            # Multi-Fernet for key rotation support
            self._multi_fernet = MultiFernet([self._primary_cipher, self._backup_cipher])
            
            # Generate RSA key pair for asymmetric encryption
            self._generate_rsa_keys()
            
            audit_logger.log_security_event(
                "encryption_initialized", "info",
                {"encryption_method": "fernet_multi_layer", "rsa_key_size": 2048}
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            audit_logger.log_security_event(
                "encryption_initialization_failed", "critical",
                {"error": str(e)}
            )
            raise
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode())
        return base64.urlsafe_b64encode(key)
    
    def _generate_rsa_keys(self):
        """Generate RSA key pair for asymmetric encryption."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        self._rsa_private_key = private_key
        self._rsa_public_key = private_key.public_key()
        
        # Save keys to secure storage
        self._save_rsa_keys()
    
    def _save_rsa_keys(self):
        """Save RSA keys to secure storage."""
        # Serialize private key
        private_pem = self._rsa_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(
                config.encryption_key.encode()
            )
        )
        
        # Serialize public key
        public_pem = self._rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Save to files
        private_key_path = self._storage_path / "rsa_private.pem"
        public_key_path = self._storage_path / "rsa_public.pem"
        
        private_key_path.write_bytes(private_pem)
        public_key_path.write_bytes(public_pem)
        
        # Set restrictive permissions
        private_key_path.chmod(0o600)
        public_key_path.chmod(0o644)
    
    def encrypt_value(self, value: str, use_asymmetric: bool = False) -> str:
        """Encrypt a sensitive value with multiple encryption options."""
        try:
            if use_asymmetric:
                # Use RSA for small values (like keys)
                encrypted = self._rsa_public_key.encrypt(
                    value.encode(),
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return base64.urlsafe_b64encode(encrypted).decode()
            else:
                # Use Fernet for larger values
                encrypted = self._multi_fernet.encrypt(value.encode())
                return base64.urlsafe_b64encode(encrypted).decode()
                
        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            audit_logger.log_security_event(
                "encryption_failed", "high",
                {"error": str(e), "use_asymmetric": use_asymmetric}
            )
            raise
    
    def decrypt_value(self, encrypted_value: str, use_asymmetric: bool = False) -> str:
        """Decrypt a sensitive value."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            
            if use_asymmetric:
                # Use RSA private key
                decrypted = self._rsa_private_key.decrypt(
                    encrypted_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return decrypted.decode()
            else:
                # Use Fernet (tries multiple keys for rotation)
                decrypted = self._multi_fernet.decrypt(encrypted_bytes)
                return decrypted.decode()
                
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            audit_logger.log_security_event(
                "decryption_failed", "high",
                {"error": str(e), "use_asymmetric": use_asymmetric}
            )
            raise
    
    def store_secret(self, secret_id: str, value: str, secret_type: SecretType, 
                    expires_in_days: Optional[int] = None, tags: Optional[Dict[str, str]] = None,
                    rotation_period_days: Optional[int] = None, created_by: str = "system") -> str:
        """Store a secret with comprehensive metadata and versioning."""
        try:
            # Encrypt the value
            encrypted_value = self.encrypt_value(value)
            
            # Calculate checksum for integrity
            checksum = hashlib.sha256(value.encode()).hexdigest()
            
            # Create metadata
            now = datetime.utcnow()
            expires_at = now + timedelta(days=expires_in_days) if expires_in_days else None
            
            metadata = SecretMetadata(
                secret_id=secret_id,
                secret_type=secret_type,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                rotation_period_days=rotation_period_days,
                tags=tags or {},
                version=1
            )
            
            # Create version entry
            version = SecretVersion(
                version=1,
                encrypted_value=encrypted_value,
                created_at=now,
                created_by=created_by,
                checksum=checksum
            )
            
            # Store metadata and version
            self._secrets_metadata[secret_id] = metadata
            self._secret_versions[secret_id] = [version]
            
            # Save to persistent storage
            self._save_metadata()
            self._save_secret_version(secret_id, version)
            
            # Log the event
            audit_logger.log_security_event(
                "secret_stored", "info",
                {
                    "secret_id": secret_id,
                    "secret_type": secret_type.value,
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    "created_by": created_by,
                    "has_rotation_period": rotation_period_days is not None
                }
            )
            
            return secret_id
            
        except Exception as e:
            logger.error(f"Failed to store secret {secret_id}: {e}")
            audit_logger.log_security_event(
                "secret_storage_failed", "high",
                {"secret_id": secret_id, "error": str(e)}
            )
            raise
    
    def get_secret(self, secret_id: str, version: Optional[int] = None) -> Optional[str]:
        """Retrieve a secret with access logging and validation."""
        try:
            # Check if secret exists
            if secret_id not in self._secrets_metadata:
                audit_logger.log_security_event(
                    "secret_access_denied", "warning",
                    {"secret_id": secret_id, "reason": "secret_not_found"}
                )
                return None
            
            metadata = self._secrets_metadata[secret_id]
            
            # Check if secret has expired
            if metadata.expires_at and datetime.utcnow() > metadata.expires_at:
                audit_logger.log_security_event(
                    "secret_access_denied", "warning",
                    {"secret_id": secret_id, "reason": "secret_expired"}
                )
                return None
            
            # Get the requested version (latest if not specified)
            versions = self._secret_versions.get(secret_id, [])
            if not versions:
                return None
            
            if version is None:
                target_version = versions[-1]  # Latest version
            else:
                target_version = next((v for v in versions if v.version == version), None)
                if not target_version:
                    return None
            
            # Load version data if not in memory
            if not hasattr(target_version, 'encrypted_value') or not target_version.encrypted_value:
                target_version = self._load_secret_version(secret_id, target_version.version)
            
            # Decrypt the value
            decrypted_value = self.decrypt_value(target_version.encrypted_value)
            
            # Verify integrity
            calculated_checksum = hashlib.sha256(decrypted_value.encode()).hexdigest()
            if calculated_checksum != target_version.checksum:
                audit_logger.log_security_event(
                    "secret_integrity_violation", "critical",
                    {"secret_id": secret_id, "version": target_version.version}
                )
                raise ValueError("Secret integrity check failed")
            
            # Update access tracking
            metadata.last_accessed = datetime.utcnow()
            metadata.access_count += 1
            
            # Log access
            self._log_secret_access(secret_id, target_version.version, "read")
            
            return decrypted_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_id}: {e}")
            audit_logger.log_security_event(
                "secret_retrieval_failed", "high",
                {"secret_id": secret_id, "error": str(e)}
            )
            return None
    
    def rotate_secret(self, secret_id: str, new_value: str, created_by: str = "system") -> bool:
        """Rotate a secret by creating a new version."""
        try:
            if secret_id not in self._secrets_metadata:
                return False
            
            metadata = self._secrets_metadata[secret_id]
            versions = self._secret_versions[secret_id]
            
            # Encrypt new value
            encrypted_value = self.encrypt_value(new_value)
            checksum = hashlib.sha256(new_value.encode()).hexdigest()
            
            # Create new version
            new_version_num = max(v.version for v in versions) + 1
            new_version = SecretVersion(
                version=new_version_num,
                encrypted_value=encrypted_value,
                created_at=datetime.utcnow(),
                created_by=created_by,
                checksum=checksum
            )
            
            # Add to versions list
            versions.append(new_version)
            
            # Update metadata
            metadata.updated_at = datetime.utcnow()
            metadata.version = new_version_num
            
            # Save changes
            self._save_metadata()
            self._save_secret_version(secret_id, new_version)
            
            # Log rotation
            audit_logger.log_security_event(
                "secret_rotated", "info",
                {
                    "secret_id": secret_id,
                    "old_version": new_version_num - 1,
                    "new_version": new_version_num,
                    "rotated_by": created_by
                }
            )
            
            # Clean up old versions (keep last 3)
            if len(versions) > 3:
                old_versions = versions[:-3]
                for old_version in old_versions:
                    self._delete_secret_version(secret_id, old_version.version)
                self._secret_versions[secret_id] = versions[-3:]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate secret {secret_id}: {e}")
            audit_logger.log_security_event(
                "secret_rotation_failed", "high",
                {"secret_id": secret_id, "error": str(e)}
            )
            return False
    
    def delete_secret(self, secret_id: str, reason: str = "") -> bool:
        """Delete a secret and all its versions."""
        try:
            if secret_id not in self._secrets_metadata:
                return False
            
            # Delete all versions
            if secret_id in self._secret_versions:
                for version in self._secret_versions[secret_id]:
                    self._delete_secret_version(secret_id, version.version)
                del self._secret_versions[secret_id]
            
            # Delete metadata
            del self._secrets_metadata[secret_id]
            
            # Save changes
            self._save_metadata()
            
            # Log deletion
            audit_logger.log_security_event(
                "secret_deleted", "warning",
                {"secret_id": secret_id, "reason": reason}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_id}: {e}")
            return False
    
    def list_secrets(self, secret_type: Optional[SecretType] = None, 
                    include_expired: bool = False) -> List[Dict[str, Any]]:
        """List secrets with filtering options."""
        results = []
        now = datetime.utcnow()
        
        for secret_id, metadata in self._secrets_metadata.items():
            # Filter by type
            if secret_type and metadata.secret_type != secret_type:
                continue
            
            # Filter expired secrets
            is_expired = metadata.expires_at and now > metadata.expires_at
            if is_expired and not include_expired:
                continue
            
            results.append({
                "secret_id": secret_id,
                "secret_type": metadata.secret_type.value,
                "created_at": metadata.created_at.isoformat(),
                "updated_at": metadata.updated_at.isoformat(),
                "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                "is_expired": is_expired,
                "version": metadata.version,
                "access_count": metadata.access_count,
                "last_accessed": metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                "tags": metadata.tags,
                "needs_rotation": self._needs_rotation(metadata)
            })
        
        return results
    
    def get_secrets_needing_rotation(self) -> List[str]:
        """Get list of secrets that need rotation."""
        needing_rotation = []
        
        for secret_id, metadata in self._secrets_metadata.items():
            if self._needs_rotation(metadata):
                needing_rotation.append(secret_id)
        
        return needing_rotation
    
    def _needs_rotation(self, metadata: SecretMetadata) -> bool:
        """Check if a secret needs rotation."""
        if not metadata.rotation_period_days:
            return False
        
        days_since_update = (datetime.utcnow() - metadata.updated_at).days
        return days_since_update >= metadata.rotation_period_days
    
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
    
    @classmethod
    def validate_environment_security(cls) -> Dict[str, Any]:
        """Validate environment for security issues."""
        issues = []
        warnings = []
        
        # Check for sensitive variables in plain text
        for key, value in os.environ.items():
            key_upper = key.upper()
            
            # Check for common security issues
            if any(sensitive in key_upper for sensitive in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']):
                if len(value) < 8:
                    issues.append(f"Environment variable {key} appears to have a weak value")
                
                # Check if it looks like a default/example value
                if value.lower() in ['password', 'secret', 'changeme', '123456', 'admin']:
                    issues.append(f"Environment variable {key} has a default/weak value")
        
        # Check for required security variables
        required_vars = ['ENCRYPTION_KEY', 'GITHUB_WEBHOOK_SECRET']
        for var in required_vars:
            if var not in os.environ:
                warnings.append(f"Required security variable {var} not set")
        
        return {
            "security_issues": issues,
            "warnings": warnings,
            "total_env_vars": len(os.environ),
            "sensitive_vars_count": sum(1 for k in os.environ.keys() 
                                      if any(s in k.upper() for s in cls.SENSITIVE_ENV_VARS))
        }


    def _log_secret_access(self, secret_id: str, version: int, operation: str):
        """Log secret access for auditing."""
        access_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "secret_id": secret_id,
            "version": version,
            "operation": operation,
            "client_info": self._get_client_info()
        }
        
        self._access_log.append(access_entry)
        
        # Keep only last 1000 entries
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]
        
        audit_logger.log_security_event(
            "secret_accessed", "info",
            access_entry
        )
    
    def _get_client_info(self) -> Dict[str, str]:
        """Get client information for audit logging."""
        # In a real implementation, this would extract client info from request context
        return {
            "user_agent": "self_healing_bot",
            "source_ip": "localhost",
            "process_id": str(os.getpid())
        }
    
    def _save_metadata(self):
        """Save secrets metadata to persistent storage."""
        metadata_file = self._storage_path / "metadata.json"
        metadata_dict = {}
        
        for secret_id, metadata in self._secrets_metadata.items():
            metadata_dict[secret_id] = {
                "secret_id": metadata.secret_id,
                "secret_type": metadata.secret_type.value,
                "created_at": metadata.created_at.isoformat(),
                "updated_at": metadata.updated_at.isoformat(),
                "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                "rotation_period_days": metadata.rotation_period_days,
                "last_accessed": metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                "access_count": metadata.access_count,
                "tags": metadata.tags,
                "version": metadata.version
            }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _load_metadata(self):
        """Load secrets metadata from persistent storage."""
        metadata_file = self._storage_path / "metadata.json"
        
        if not metadata_file.exists():
            return
        
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            for secret_id, data in metadata_dict.items():
                metadata = SecretMetadata(
                    secret_id=data["secret_id"],
                    secret_type=SecretType(data["secret_type"]),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    updated_at=datetime.fromisoformat(data["updated_at"]),
                    expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
                    rotation_period_days=data["rotation_period_days"],
                    last_accessed=datetime.fromisoformat(data["last_accessed"]) if data["last_accessed"] else None,
                    access_count=data["access_count"],
                    tags=data["tags"],
                    version=data["version"]
                )
                self._secrets_metadata[secret_id] = metadata
                
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
    
    def _save_secret_version(self, secret_id: str, version: SecretVersion):
        """Save a secret version to persistent storage."""
        version_file = self._storage_path / f"{secret_id}_v{version.version}.enc"
        
        version_data = {
            "version": version.version,
            "encrypted_value": version.encrypted_value,
            "created_at": version.created_at.isoformat(),
            "created_by": version.created_by,
            "checksum": version.checksum
        }
        
        with open(version_file, 'w') as f:
            json.dump(version_data, f)
        
        # Set restrictive permissions
        version_file.chmod(0o600)
    
    def _load_secret_version(self, secret_id: str, version: int) -> SecretVersion:
        """Load a secret version from persistent storage."""
        version_file = self._storage_path / f"{secret_id}_v{version}.enc"
        
        if not version_file.exists():
            raise ValueError(f"Secret version {secret_id} v{version} not found")
        
        with open(version_file, 'r') as f:
            version_data = json.load(f)
        
        return SecretVersion(
            version=version_data["version"],
            encrypted_value=version_data["encrypted_value"],
            created_at=datetime.fromisoformat(version_data["created_at"]),
            created_by=version_data["created_by"],
            checksum=version_data["checksum"]
        )
    
    def _delete_secret_version(self, secret_id: str, version: int):
        """Delete a secret version from persistent storage."""
        version_file = self._storage_path / f"{secret_id}_v{version}.enc"
        
        if version_file.exists():
            version_file.unlink()


# External secret managers integration
class ExternalSecretManager:
    """Base class for external secret manager integrations."""
    
    async def get_secret(self, secret_id: str) -> Optional[str]:
        """Get secret from external manager."""
        raise NotImplementedError
    
    async def store_secret(self, secret_id: str, value: str) -> bool:
        """Store secret in external manager."""
        raise NotImplementedError


class HashiCorpVaultManager(ExternalSecretManager):
    """HashiCorp Vault integration."""
    
    def __init__(self, vault_url: str, vault_token: str, mount_path: str = "secret"):
        self.vault_url = vault_url.rstrip("/")
        self.vault_token = vault_token
        self.mount_path = mount_path
        self.headers = {
            "X-Vault-Token": vault_token,
            "Content-Type": "application/json"
        }
    
    async def get_secret(self, secret_id: str) -> Optional[str]:
        """Get secret from HashiCorp Vault."""
        try:
            import aiohttp
            
            url = f"{self.vault_url}/v1/{self.mount_path}/data/{secret_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["data"]["data"].get("value")
                    elif response.status == 404:
                        return None
                    else:
                        response.raise_for_status()
                        
        except Exception as e:
            logger.error(f"Failed to get secret from Vault: {e}")
            return None
    
    async def store_secret(self, secret_id: str, value: str) -> bool:
        """Store secret in HashiCorp Vault."""
        try:
            import aiohttp
            
            url = f"{self.vault_url}/v1/{self.mount_path}/data/{secret_id}"
            payload = {
                "data": {
                    "value": value,
                    "created_by": "self_healing_bot",
                    "created_at": datetime.utcnow().isoformat()
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=payload) as response:
                    return response.status in [200, 204]
                    
        except Exception as e:
            logger.error(f"Failed to store secret in Vault: {e}")
            return False


class AWSSecretsManager(ExternalSecretManager):
    """AWS Secrets Manager integration."""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self._client = None
    
    def _get_client(self):
        """Get boto3 client for Secrets Manager."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('secretsmanager', region_name=self.region_name)
            except ImportError:
                logger.error("boto3 not available for AWS Secrets Manager")
                raise
        return self._client
    
    async def get_secret(self, secret_id: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            client = self._get_client()
            response = client.get_secret_value(SecretId=secret_id)
            return response.get('SecretString')
            
        except Exception as e:
            if "ResourceNotFoundException" in str(e):
                return None
            logger.error(f"Failed to get secret from AWS: {e}")
            return None
    
    async def store_secret(self, secret_id: str, value: str) -> bool:
        """Store secret in AWS Secrets Manager."""
        try:
            client = self._get_client()
            
            # Try to update existing secret first
            try:
                client.update_secret(
                    SecretId=secret_id,
                    SecretString=value,
                    Description="Updated by self-healing MLOps bot"
                )
                return True
            except Exception:
                # Create new secret
                client.create_secret(
                    Name=secret_id,
                    SecretString=value,
                    Description="Created by self-healing MLOps bot"
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to store secret in AWS: {e}")
            return False


# Global instances
secrets_manager = EnhancedSecretsManager()
secret_scanner = SecretScanner()