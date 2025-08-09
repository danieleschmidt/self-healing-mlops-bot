"""Security components for the self-healing bot."""

from .validation import (
    InputValidator, 
    SecurityValidator, 
    RateLimiter,
    ValidationError,
    SecurityThreat,
    input_validator
)

from .secrets import (
    EnhancedSecretsManager,
    SecretScanner,
    EnvironmentProtector,
    SecretType,
    SecretMetadata,
    HashiCorpVaultManager,
    AWSSecretsManager,
    secrets_manager,
    secret_scanner
)

from .monitoring import (
    SecurityMonitor,
    ComplianceMonitor,
    SecurityEvent,
    SecurityEventType,
    ThreatLevel,
    ThreatIndicator,
    security_monitor,
    compliance_monitor
)

from .auth import (
    AuthenticationManager,
    AuthorizationManager,
    SecurityMiddleware,
    User,
    UserRole,
    Permission,
    Session,
    AuthenticationError,
    AuthorizationError,
    AccountLockedException,
    auth_manager,
    authz_manager,
    security_middleware
)

from .rate_limiting import (
    SecurityAwareRateLimiter,
    RateLimitRule,
    RateLimitType,
    RateLimitScope,
    RateLimitViolation,
    rate_limiter
)

from .api_security import (
    JWTManager,
    APISecurityMiddleware,
    WebhookSecurityValidator,
    TokenType,
    SecurityLevel,
    APISecurityConfig,
    jwt_manager,
    webhook_validator,
    api_security_config,
    get_current_user,
    require_auth,
    require_permission
)

from .scanning import (
    SecurityScanner,
    DependencyScanner,
    CodeScanner,
    ConfigurationScanner,
    Vulnerability,
    VulnerabilityType,
    SeverityLevel,
    ScanResult,
    security_scanner
)

from .threat_intelligence import (
    ThreatIntelligenceManager,
    ThreatIntelligence,
    ThreatFeed,
    ThreatType,
    IndicatorType,
    threat_intelligence
)

from .incident_response import (
    IncidentResponseManager,
    IncidentTicket,
    PlaybookRule,
    ResponseAction,
    IncidentSeverity,
    IncidentStatus,
    ResponseActionType,
    incident_response
)

__all__ = [
    # Validation
    'InputValidator', 'SecurityValidator', 'RateLimiter', 'ValidationError', 
    'SecurityThreat', 'input_validator',
    
    # Secrets Management
    'EnhancedSecretsManager', 'SecretScanner', 'EnvironmentProtector',
    'SecretType', 'SecretMetadata', 'HashiCorpVaultManager', 'AWSSecretsManager',
    'secrets_manager', 'secret_scanner',
    
    # Security Monitoring
    'SecurityMonitor', 'ComplianceMonitor', 'SecurityEvent', 'SecurityEventType',
    'ThreatLevel', 'ThreatIndicator', 'security_monitor', 'compliance_monitor',
    
    # Authentication & Authorization
    'AuthenticationManager', 'AuthorizationManager', 'SecurityMiddleware',
    'User', 'UserRole', 'Permission', 'Session', 'AuthenticationError',
    'AuthorizationError', 'AccountLockedException', 'auth_manager', 
    'authz_manager', 'security_middleware',
    
    # Rate Limiting
    'SecurityAwareRateLimiter', 'RateLimitRule', 'RateLimitType', 
    'RateLimitScope', 'RateLimitViolation', 'rate_limiter',
    
    # API Security
    'JWTManager', 'APISecurityMiddleware', 'WebhookSecurityValidator',
    'TokenType', 'SecurityLevel', 'APISecurityConfig', 'jwt_manager',
    'webhook_validator', 'api_security_config', 'get_current_user',
    'require_auth', 'require_permission',
    
    # Security Scanning
    'SecurityScanner', 'DependencyScanner', 'CodeScanner', 'ConfigurationScanner',
    'Vulnerability', 'VulnerabilityType', 'SeverityLevel', 'ScanResult',
    'security_scanner',
    
    # Threat Intelligence
    'ThreatIntelligenceManager', 'ThreatIntelligence', 'ThreatFeed',
    'ThreatType', 'IndicatorType', 'threat_intelligence',
    
    # Incident Response
    'IncidentResponseManager', 'IncidentTicket', 'PlaybookRule', 'ResponseAction',
    'IncidentSeverity', 'IncidentStatus', 'ResponseActionType', 'incident_response'
]