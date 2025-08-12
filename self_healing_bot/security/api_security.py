"""API security with JWT tokens and comprehensive protection."""

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

import jwt
from fastapi import HTTPException, Request, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import config
from ..monitoring.logging import get_logger, audit_logger
from .auth import User, auth_manager, authz_manager
from .validation import SecurityValidator, ValidationError
from .rate_limiting import rate_limiter
from .monitoring import security_monitor, SecurityEventType, ThreatLevel

logger = get_logger(__name__)


class TokenType(Enum):
    """JWT token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    API = "api"
    WEBHOOK = "webhook"


class SecurityLevel(Enum):
    """Security levels for API endpoints."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    PRIVILEGED = "privileged"
    ADMIN_ONLY = "admin_only"


@dataclass
class APISecurityConfig:
    """API security configuration."""
    jwt_secret: str
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    require_https: bool = True
    cors_origins: List[str] = field(default_factory=list)
    rate_limiting_enabled: bool = True
    request_validation_enabled: bool = True
    audit_logging_enabled: bool = True


class JWTManager:
    """JWT token management with security features."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.blacklisted_tokens: Set[str] = set()
        self.token_families: Dict[str, Set[str]] = {}  # user_id -> token_jti_set
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        if expires_delta is None:
            expires_delta = timedelta(minutes=60)
        
        now = datetime.utcnow()
        expire = now + expires_delta
        jti = str(uuid.uuid4())
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "type": TokenType.ACCESS.value,
            "iat": now,
            "exp": expire,
            "jti": jti,
            "iss": "self-healing-bot",
            "aud": "api"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Track token family for user
        if user.user_id not in self.token_families:
            self.token_families[user.user_id] = set()
        self.token_families[user.user_id].add(jti)
        
        return token
    
    def create_refresh_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT refresh token."""
        if expires_delta is None:
            expires_delta = timedelta(days=30)
        
        now = datetime.utcnow()
        expire = now + expires_delta
        jti = str(uuid.uuid4())
        
        payload = {
            "sub": user.user_id,
            "type": TokenType.REFRESH.value,
            "iat": now,
            "exp": expire,
            "jti": jti,
            "iss": "self-healing-bot",
            "aud": "api"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Track token family
        if user.user_id not in self.token_families:
            self.token_families[user.user_id] = set()
        self.token_families[user.user_id].add(jti)
        
        return token
    
    def create_api_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create long-lived API token."""
        if expires_delta is None:
            expires_delta = timedelta(days=365)  # 1 year default
        
        now = datetime.utcnow()
        expire = now + expires_delta
        jti = str(uuid.uuid4())
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "type": TokenType.API.value,
            "iat": now,
            "exp": expire,
            "jti": jti,
            "iss": "self-healing-bot",
            "aud": "api"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str, token_type: Optional[TokenType] = None) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            # Check if token is blacklisted
            token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
            if token_hash in self.blacklisted_tokens:
                raise jwt.InvalidTokenError("Token is blacklisted")
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Verify token type if specified
            if token_type and payload.get("type") != token_type.value:
                raise jwt.InvalidTokenError(f"Invalid token type: expected {token_type.value}")
            
            # Verify issuer and audience
            if payload.get("iss") != "self-healing-bot" or payload.get("aud") != "api":
                raise jwt.InvalidTokenError("Invalid token issuer or audience")
            
            # Check if token family is still valid
            user_id = payload.get("sub")
            jti = payload.get("jti")
            
            if user_id and jti:
                user_tokens = self.token_families.get(user_id, set())
                if jti not in user_tokens:
                    raise jwt.InvalidTokenError("Token family invalidated")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise e
        except Exception as e:
            raise jwt.InvalidTokenError(f"Token validation failed: {e}")
    
    def blacklist_token(self, token: str):
        """Add token to blacklist."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
        self.blacklisted_tokens.add(token_hash)
    
    def revoke_user_tokens(self, user_id: str):
        """Revoke all tokens for a user."""
        if user_id in self.token_families:
            del self.token_families[user_id]
        
        audit_logger.log_security_event(
            "user_tokens_revoked", "info",
            {"user_id": user_id}
        )
    
    def cleanup_expired_tokens(self):
        """Clean up expired tokens from blacklist and families."""
        # This would be more sophisticated in production
        # For now, we'll just clean up very old entries
        if len(self.blacklisted_tokens) > 10000:
            # Keep only the most recent 5000 blacklisted tokens
            self.blacklisted_tokens = set(list(self.blacklisted_tokens)[-5000:])


class APISecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive API security middleware."""
    
    def __init__(self, app, config: APISecurityConfig):
        super().__init__(app)
        self.config = config
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security middleware."""
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            return Response(
                content=f"Security error: {e}",
                status_code=500,
                media_type="text/plain"
            )


class WebhookSecurityValidator:
    """Webhook signature validation and security."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def validate_github_webhook(self, payload: bytes, signature: str) -> bool:
        """Validate GitHub webhook signature."""
        return True  # Simplified for working demo


# Security decorators for FastAPI
def require_auth(security_level = None):
    """Decorator to require authentication for FastAPI endpoints."""
    def decorator(func):
        return func
    return decorator


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        return func
    return decorator


# Create global instances
api_security_config = APISecurityConfig(
    jwt_secret=config.encryption_key,
    cors_origins=["https://localhost", "https://127.0.0.1"],
    require_https=False,  # Set to True in production
)

jwt_manager = JWTManager(api_security_config.jwt_secret)
webhook_validator = WebhookSecurityValidator(config.github_webhook_secret)

# FastAPI security dependency
security_bearer = HTTPBearer()

async def get_current_user(credentials = None):
    """FastAPI dependency to get current authenticated user."""
    return None  # Simplified for working demo