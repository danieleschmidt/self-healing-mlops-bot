"""Authentication and authorization framework."""

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

import jwt
import bcrypt
from passlib.context import CryptContext

from ..core.config import config
from ..monitoring.logging import get_logger, audit_logger
from .secrets import secrets_manager

logger = get_logger(__name__)


class UserRole(Enum):
    """User roles with different permission levels."""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    API_CLIENT = "api_client"
    SYSTEM = "system"


class Permission(Enum):
    """System permissions."""
    # General permissions
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    
    # Specific permissions
    MANAGE_USERS = "manage_users"
    MANAGE_SECRETS = "manage_secrets"
    MANAGE_SYSTEM = "manage_system"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_WEBHOOKS = "manage_webhooks"
    TRIGGER_MANUAL_ACTIONS = "trigger_manual_actions"
    VIEW_METRICS = "view_metrics"
    MANAGE_CONFIGURATION = "manage_configuration"


@dataclass
class User:
    """User account information."""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    password_hash: Optional[str] = None
    api_key_hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    is_active: bool = True
    is_locked: bool = False
    locked_until: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User session information."""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    last_activity: datetime = field(default_factory=datetime.utcnow)


class AuthenticationError(Exception):
    """Authentication failed."""
    pass


class AuthorizationError(Exception):
    """Authorization failed."""
    pass


class AccountLockedException(Exception):
    """Account is locked."""
    pass


class AuthenticationManager:
    """Manages user authentication and session handling."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.jwt_secret = self._get_jwt_secret()
        
        # Initialize default roles and permissions
        self._initialize_role_permissions()
        
        # Create default admin user if none exists
        asyncio.create_task(self._create_default_admin())
    
    def _get_jwt_secret(self) -> str:
        """Get JWT secret key."""
        try:
            return secrets_manager.get_secret("jwt_secret") or config.encryption_key
        except Exception:
            return config.encryption_key
    
    def _initialize_role_permissions(self):
        """Initialize role-based permissions."""
        self.role_permissions = {
            UserRole.ADMIN: {
                Permission.READ, Permission.WRITE, Permission.DELETE, Permission.EXECUTE,
                Permission.MANAGE_USERS, Permission.MANAGE_SECRETS, Permission.MANAGE_SYSTEM,
                Permission.VIEW_AUDIT_LOGS, Permission.MANAGE_WEBHOOKS, Permission.TRIGGER_MANUAL_ACTIONS,
                Permission.VIEW_METRICS, Permission.MANAGE_CONFIGURATION
            },
            UserRole.OPERATOR: {
                Permission.READ, Permission.WRITE, Permission.EXECUTE,
                Permission.MANAGE_WEBHOOKS, Permission.TRIGGER_MANUAL_ACTIONS,
                Permission.VIEW_METRICS, Permission.VIEW_AUDIT_LOGS
            },
            UserRole.VIEWER: {
                Permission.READ, Permission.VIEW_METRICS
            },
            UserRole.API_CLIENT: {
                Permission.READ, Permission.WRITE, Permission.EXECUTE
            },
            UserRole.SYSTEM: {
                Permission.READ, Permission.WRITE, Permission.EXECUTE,
                Permission.MANAGE_SECRETS, Permission.MANAGE_SYSTEM
            }
        }
    
    async def _create_default_admin(self):
        """Create default admin user if none exists."""
        if not any(user.role == UserRole.ADMIN for user in self.users.values()):
            admin_password = "admin_" + str(uuid.uuid4())[:8]
            
            await self.create_user(
                username="admin",
                email="admin@localhost",
                password=admin_password,
                role=UserRole.ADMIN
            )
            
            logger.warning(f"Created default admin user with password: {admin_password}")
            audit_logger.log_security_event(
                "default_admin_created", "warning",
                {"username": "admin", "temporary_password": True}
            )
    
    async def create_user(self, username: str, email: str, password: str, 
                         role: UserRole, permissions: Optional[Set[Permission]] = None) -> User:
        """Create a new user."""
        # Check if username already exists
        if any(user.username == username for user in self.users.values()):
            raise ValueError(f"Username {username} already exists")
        
        # Create user
        user_id = str(uuid.uuid4())
        password_hash = self.pwd_context.hash(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            password_hash=password_hash,
            permissions=permissions or self.role_permissions.get(role, set())
        )
        
        self.users[user_id] = user
        
        audit_logger.log_security_event(
            "user_created", "info",
            {"user_id": user_id, "username": username, "role": role.value}
        )
        
        return user
    
    async def authenticate_password(self, username: str, password: str, 
                                  ip_address: Optional[str] = None) -> Tuple[User, str]:
        """Authenticate user with password."""
        # Find user by username
        user = next((user for user in self.users.values() if user.username == username), None)
        if not user:
            await self._log_failed_attempt(username, ip_address, "user_not_found")
            raise AuthenticationError("Invalid credentials")
        
        # Check if account is locked
        if user.is_locked and user.locked_until and datetime.utcnow() < user.locked_until:
            raise AccountLockedException("Account is temporarily locked")
        
        # Check if account is active
        if not user.is_active:
            await self._log_failed_attempt(username, ip_address, "account_disabled")
            raise AuthenticationError("Account is disabled")
        
        # Verify password
        if not self.pwd_context.verify(password, user.password_hash):
            user.failed_login_attempts += 1
            await self._handle_failed_login(user, ip_address)
            raise AuthenticationError("Invalid credentials")
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()
        
        # Create session
        session_token = await self._create_session(user, ip_address)
        
        audit_logger.log_security_event(
            "user_authenticated", "info",
            {"user_id": user.user_id, "username": username, "method": "password"}
        )
        
        return user, session_token
    
    async def authenticate_api_key(self, api_key: str) -> User:
        """Authenticate user with API key."""
        # Hash the API key to compare
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Find user by API key hash
        user = None
        for u in self.users.values():
            if u.api_key_hash == api_key_hash:
                user = u
                break
        
        if not user:
            audit_logger.log_security_event(
                "api_key_authentication_failed", "warning",
                {"api_key_prefix": api_key[:8]}
            )
            raise AuthenticationError("Invalid API key")
        
        # Check if account is active
        if not user.is_active:
            raise AuthenticationError("Account is disabled")
        
        user.last_login = datetime.utcnow()
        
        audit_logger.log_security_event(
            "user_authenticated", "info",
            {"user_id": user.user_id, "method": "api_key"}
        )
        
        return user
    
    async def authenticate_jwt(self, token: str) -> User:
        """Authenticate user with JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            user_id = payload.get("user_id")
            
            if not user_id or user_id not in self.users:
                raise AuthenticationError("Invalid token")
            
            user = self.users[user_id]
            
            # Check if token is expired
            if payload.get("exp", 0) < time.time():
                raise AuthenticationError("Token expired")
            
            # Check if account is active
            if not user.is_active:
                raise AuthenticationError("Account is disabled")
            
            return user
            
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
    
    async def _create_session(self, user: User, ip_address: Optional[str] = None) -> str:
        """Create a new session for the user."""
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=24)  # 24-hour sessions
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            ip_address=ip_address
        )
        
        self.sessions[session_id] = session
        
        # Also create JWT token
        jwt_payload = {
            "user_id": user.user_id,
            "session_id": session_id,
            "role": user.role.value,
            "iat": time.time(),
            "exp": time.time() + 86400  # 24 hours
        }
        
        token = jwt.encode(jwt_payload, self.jwt_secret, algorithm="HS256")
        return token
    
    async def validate_session(self, session_id: str) -> Optional[User]:
        """Validate a session and return the user."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Check if session is expired
        if datetime.utcnow() > session.expires_at:
            del self.sessions[session_id]
            return None
        
        # Check if session is active
        if not session.is_active:
            return None
        
        # Update last activity
        session.last_activity = datetime.utcnow()
        
        # Get user
        user = self.users.get(session.user_id)
        if not user or not user.is_active:
            session.is_active = False
            return None
        
        return user
    
    async def revoke_session(self, session_id: str):
        """Revoke a session."""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            audit_logger.log_security_event(
                "session_revoked", "info",
                {"session_id": session_id}
            )
    
    async def generate_api_key(self, user: User) -> str:
        """Generate an API key for a user."""
        api_key = f"shb_{user.username}_{uuid.uuid4().hex[:16]}"
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        user.api_key_hash = api_key_hash
        
        audit_logger.log_security_event(
            "api_key_generated", "info",
            {"user_id": user.user_id, "username": user.username}
        )
        
        return api_key
    
    async def revoke_api_key(self, user: User):
        """Revoke a user's API key."""
        user.api_key_hash = None
        
        audit_logger.log_security_event(
            "api_key_revoked", "info",
            {"user_id": user.user_id, "username": user.username}
        )
    
    async def _handle_failed_login(self, user: User, ip_address: Optional[str]):
        """Handle failed login attempt."""
        await self._log_failed_attempt(user.username, ip_address, "wrong_password")
        
        # Lock account after 5 failed attempts
        if user.failed_login_attempts >= 5:
            user.is_locked = True
            user.locked_until = datetime.utcnow() + timedelta(minutes=30)  # 30-minute lockout
            
            audit_logger.log_security_event(
                "account_locked", "warning",
                {"user_id": user.user_id, "username": user.username, "attempts": user.failed_login_attempts}
            )
    
    async def _log_failed_attempt(self, username: str, ip_address: Optional[str], reason: str):
        """Log failed authentication attempt."""
        audit_logger.log_security_event(
            "authentication_failed", "warning",
            {"username": username, "ip_address": ip_address, "reason": reason}
        )
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return next((user for user in self.users.values() if user.username == username), None)
    
    async def update_user_role(self, user_id: str, new_role: UserRole):
        """Update user role and permissions."""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        old_role = user.role
        user.role = new_role
        user.permissions = self.role_permissions.get(new_role, set())
        
        audit_logger.log_security_event(
            "user_role_updated", "info",
            {"user_id": user_id, "old_role": old_role.value, "new_role": new_role.value}
        )
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.expires_at < now
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class AuthorizationManager:
    """Manages user authorization and permission checking."""
    
    def __init__(self, auth_manager: AuthenticationManager):
        self.auth_manager = auth_manager
        self.resource_permissions: Dict[str, Set[Permission]] = {
            "/webhooks": {Permission.MANAGE_WEBHOOKS},
            "/users": {Permission.MANAGE_USERS},
            "/secrets": {Permission.MANAGE_SECRETS},
            "/system": {Permission.MANAGE_SYSTEM},
            "/audit": {Permission.VIEW_AUDIT_LOGS},
            "/metrics": {Permission.VIEW_METRICS},
            "/config": {Permission.MANAGE_CONFIGURATION}
        }
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in user.permissions
    
    def check_resource_access(self, user: User, resource_path: str, method: str = "GET") -> bool:
        """Check if user can access a specific resource."""
        # Map HTTP methods to permissions
        method_permissions = {
            "GET": Permission.READ,
            "POST": Permission.WRITE,
            "PUT": Permission.WRITE,
            "PATCH": Permission.WRITE,
            "DELETE": Permission.DELETE
        }
        
        base_permission = method_permissions.get(method, Permission.READ)
        
        # Check base permission
        if not self.check_permission(user, base_permission):
            return False
        
        # Check resource-specific permissions
        for resource, required_permissions in self.resource_permissions.items():
            if resource_path.startswith(resource):
                return any(self.check_permission(user, perm) for perm in required_permissions)
        
        # Default to allowing if no specific resource permissions defined
        return True
    
    def require_permission(self, permission: Permission):
        """Decorator to require a specific permission."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # This would typically extract user from request context
                # For now, we'll assume it's passed as an argument
                user = kwargs.get('user') or (args[1] if len(args) > 1 else None)
                
                if not user or not isinstance(user, User):
                    raise AuthorizationError("No user context available")
                
                if not self.check_permission(user, permission):
                    audit_logger.log_security_event(
                        "authorization_failed", "warning",
                        {"user_id": user.user_id, "required_permission": permission.value}
                    )
                    raise AuthorizationError(f"Permission required: {permission.value}")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_role(self, role: UserRole):
        """Decorator to require a specific role."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                user = kwargs.get('user') or (args[1] if len(args) > 1 else None)
                
                if not user or not isinstance(user, User):
                    raise AuthorizationError("No user context available")
                
                if user.role != role:
                    audit_logger.log_security_event(
                        "authorization_failed", "warning",
                        {"user_id": user.user_id, "required_role": role.value, "user_role": user.role.value}
                    )
                    raise AuthorizationError(f"Role required: {role.value}")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator


class SecurityMiddleware:
    """Security middleware for FastAPI applications."""
    
    def __init__(self, auth_manager: AuthenticationManager, authz_manager: AuthorizationManager):
        self.auth_manager = auth_manager
        self.authz_manager = authz_manager
    
    async def authenticate_request(self, request):
        """Authenticate incoming request."""
        # Try different authentication methods
        
        # 1. JWT token in Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                return await self.auth_manager.authenticate_jwt(token)
            except AuthenticationError:
                pass
        
        # 2. API key in header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            try:
                return await self.auth_manager.authenticate_api_key(api_key)
            except AuthenticationError:
                pass
        
        # 3. Session cookie
        session_id = request.cookies.get("session_id")
        if session_id:
            user = await self.auth_manager.validate_session(session_id)
            if user:
                return user
        
        return None
    
    async def authorize_request(self, user: User, request) -> bool:
        """Authorize request based on user permissions."""
        if not user:
            return False
        
        method = request.method
        path = request.url.path
        
        return self.authz_manager.check_resource_access(user, path, method)


# Global instances - lazy initialization
auth_manager = None
authz_manager = None
security_middleware = None


def get_auth_manager() -> AuthenticationManager:
    """Get or create the global authentication manager."""
    global auth_manager
    if auth_manager is None:
        auth_manager = AuthenticationManager()
    return auth_manager


def get_authz_manager() -> AuthorizationManager:
    """Get or create the global authorization manager."""
    global authz_manager
    if authz_manager is None:
        authz_manager = AuthorizationManager(get_auth_manager())
    return authz_manager


def get_security_middleware() -> SecurityMiddleware:
    """Get or create the global security middleware."""
    global security_middleware
    if security_middleware is None:
        security_middleware = SecurityMiddleware(get_auth_manager(), get_authz_manager())
    return security_middleware