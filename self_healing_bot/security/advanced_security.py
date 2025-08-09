"""Advanced security features for Generation 2 reliability."""

import asyncio
import logging
import json
import hmac
import hashlib
import base64
import secrets
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import re
import ipaddress
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    AUTHENTICATION_FAILURE = "auth_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_PAYLOAD = "malicious_payload"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: str
    user_agent: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    mitigated: bool = False
    mitigation_actions: List[str] = None


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    name: str
    max_requests: int
    time_window: int  # seconds
    burst_allowance: int
    penalty_duration: int  # seconds
    scope: str  # "ip", "user", "endpoint"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    allowed_origins: List[str]
    blocked_ips: Set[str]
    rate_limits: List[RateLimitRule]
    payload_size_limit: int
    require_signature_verification: bool
    enable_geo_blocking: bool
    blocked_countries: List[str]
    suspicious_patterns: List[str]


class AdvancedSecurityManager:
    """Advanced security management system."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        
        # Security state
        self.security_events: deque = deque(maxlen=10000)
        self.blocked_ips: Set[str] = set()
        self.rate_limit_counters: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.suspicious_actors: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Configuration
        self.security_policy = self._load_default_security_policy()
        self.threat_detection_rules = self._load_threat_detection_rules()
        
        # Statistics
        self.security_stats = {
            "total_events": 0,
            "blocked_requests": 0,
            "threats_detected": 0,
            "threats_mitigated": 0,
            "false_positives": 0
        }
        
        # Cache for performance
        self._ip_reputation_cache: Dict[str, Tuple[str, datetime]] = {}
        self._signature_cache: Dict[str, bool] = {}
        
        # Monitoring
        self._monitoring_active = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def _load_default_security_policy(self) -> SecurityPolicy:
        """Load default security policy."""
        return SecurityPolicy(
            allowed_origins=[
                "https://github.com",
                "https://*.github.com",
                "https://api.github.com"
            ],
            blocked_ips=set(),
            rate_limits=[
                RateLimitRule(
                    name="webhook_rate_limit",
                    max_requests=100,
                    time_window=60,
                    burst_allowance=20,
                    penalty_duration=300,
                    scope="ip"
                ),
                RateLimitRule(
                    name="api_rate_limit",
                    max_requests=1000,
                    time_window=3600,
                    burst_allowance=100,
                    penalty_duration=900,
                    scope="user"
                )
            ],
            payload_size_limit=10 * 1024 * 1024,  # 10MB
            require_signature_verification=True,
            enable_geo_blocking=False,
            blocked_countries=[],
            suspicious_patterns=[
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"eval\s*\(",
                r"document\.cookie",
                r"window\.location",
                r"\.\./\.\./",
                r"proc/self/environ",
                r"/etc/passwd",
                r"cmd\.exe",
                r"powershell",
                r"wget\s+http",
                r"curl\s+http"
            ]
        )
    
    def _load_threat_detection_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load threat detection rules."""
        return {
            "brute_force": {
                "pattern": r"repeated authentication failures",
                "threshold": 5,
                "time_window": 300,  # 5 minutes
                "threat_level": ThreatLevel.HIGH,
                "auto_block": True
            },
            "payload_injection": {
                "patterns": [
                    r"<script[^>]*>",
                    r"javascript:",
                    r"eval\s*\(",
                    r"union\s+select",
                    r"drop\s+table",
                    r"\.\./",
                    r"proc/self"
                ],
                "threat_level": ThreatLevel.MEDIUM,
                "auto_block": False
            },
            "rate_limit_abuse": {
                "threshold": 1000,
                "time_window": 60,
                "threat_level": ThreatLevel.MEDIUM,
                "auto_block": True
            },
            "suspicious_user_agent": {
                "patterns": [
                    r"bot",
                    r"crawler",
                    r"scanner",
                    r"sqlmap",
                    r"nmap",
                    r"dirb",
                    r"nikto"
                ],
                "whitelist": [
                    r"github-hookshot",
                    r"github-actions"
                ],
                "threat_level": ThreatLevel.LOW,
                "auto_block": False
            },
            "geo_anomaly": {
                "enabled": False,
                "suspicious_countries": ["CN", "RU", "KP"],
                "threat_level": ThreatLevel.LOW,
                "auto_block": False
            }
        }
    
    async def start_monitoring(self):
        """Start security monitoring."""
        try:
            if self._monitoring_active:
                logger.warning("Security monitoring already active")
                return
            
            self._monitoring_active = True
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Load persistent data if available
            await self._load_persistent_data()
            
            logger.info("Advanced security monitoring started")
            
        except Exception as e:
            logger.exception(f"Error starting security monitoring: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop security monitoring."""
        try:
            self._monitoring_active = False
            
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Save persistent data
            await self._save_persistent_data()
            
            logger.info("Advanced security monitoring stopped")
            
        except Exception as e:
            logger.exception(f"Error stopping security monitoring: {e}")
    
    async def analyze_request(
        self,
        source_ip: str,
        user_agent: Optional[str],
        headers: Dict[str, str],
        payload: Optional[str],
        endpoint: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive request security analysis."""
        analysis_result = {
            "allowed": True,
            "threat_level": ThreatLevel.LOW,
            "threats_detected": [],
            "mitigation_actions": [],
            "security_score": 100.0
        }
        
        try:
            # IP-based checks
            ip_threats = await self._analyze_ip(source_ip)
            if ip_threats:
                analysis_result["threats_detected"].extend(ip_threats)
                analysis_result["security_score"] -= 20
            
            # Rate limiting checks
            rate_limit_threats = await self._check_rate_limits(
                source_ip, user_id, endpoint
            )
            if rate_limit_threats:
                analysis_result["threats_detected"].extend(rate_limit_threats)
                analysis_result["allowed"] = False
                analysis_result["mitigation_actions"].append("rate_limit_block")
            
            # User agent analysis
            if user_agent:
                ua_threats = await self._analyze_user_agent(user_agent)
                if ua_threats:
                    analysis_result["threats_detected"].extend(ua_threats)
                    analysis_result["security_score"] -= 10
            
            # Payload analysis
            if payload:
                payload_threats = await self._analyze_payload(payload)
                if payload_threats:
                    analysis_result["threats_detected"].extend(payload_threats)
                    analysis_result["security_score"] -= 30
            
            # Headers analysis
            header_threats = await self._analyze_headers(headers)
            if header_threats:
                analysis_result["threats_detected"].extend(header_threats)
                analysis_result["security_score"] -= 15
            
            # Determine overall threat level
            threat_levels = [t.get("level", ThreatLevel.LOW) for t in analysis_result["threats_detected"]]
            if ThreatLevel.CRITICAL in threat_levels:
                analysis_result["threat_level"] = ThreatLevel.CRITICAL
                analysis_result["allowed"] = False
                analysis_result["mitigation_actions"].append("immediate_block")
            elif ThreatLevel.HIGH in threat_levels:
                analysis_result["threat_level"] = ThreatLevel.HIGH
                if analysis_result["security_score"] < 60:
                    analysis_result["allowed"] = False
                    analysis_result["mitigation_actions"].append("temporary_block")
            elif ThreatLevel.MEDIUM in threat_levels:
                analysis_result["threat_level"] = ThreatLevel.MEDIUM
                if analysis_result["security_score"] < 40:
                    analysis_result["allowed"] = False
                    analysis_result["mitigation_actions"].append("enhanced_monitoring")
            
            # Log security event if threats detected
            if analysis_result["threats_detected"]:
                await self._log_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    analysis_result["threat_level"],
                    source_ip,
                    user_agent,
                    {
                        "endpoint": endpoint,
                        "user_id": user_id,
                        "threats": analysis_result["threats_detected"],
                        "security_score": analysis_result["security_score"]
                    }
                )
            
            return analysis_result
            
        except Exception as e:
            logger.exception(f"Error analyzing request: {e}")
            # Fail secure - block on analysis error
            return {
                "allowed": False,
                "threat_level": ThreatLevel.HIGH,
                "threats_detected": [{"type": "analysis_error", "description": str(e)}],
                "mitigation_actions": ["analysis_error_block"],
                "security_score": 0.0
            }
    
    async def _analyze_ip(self, ip: str) -> List[Dict[str, Any]]:
        """Analyze IP address for threats."""
        threats = []
        
        try:
            # Check if IP is blocked
            if ip in self.blocked_ips or ip in self.security_policy.blocked_ips:
                threats.append({
                    "type": "blocked_ip",
                    "level": ThreatLevel.HIGH,
                    "description": f"IP {ip} is in blocked list"
                })
                return threats
            
            # Check IP reputation (cached)
            reputation = await self._get_ip_reputation(ip)
            if reputation == "malicious":
                threats.append({
                    "type": "malicious_ip",
                    "level": ThreatLevel.HIGH,
                    "description": f"IP {ip} has malicious reputation"
                })
            elif reputation == "suspicious":
                threats.append({
                    "type": "suspicious_ip",
                    "level": ThreatLevel.MEDIUM,
                    "description": f"IP {ip} has suspicious reputation"
                })
            
            # Check for private/internal IP access
            try:
                ip_obj = ipaddress.ip_address(ip)
                if ip_obj.is_private and not self._is_allowed_private_ip(ip):
                    threats.append({
                        "type": "private_ip_access",
                        "level": ThreatLevel.MEDIUM,
                        "description": f"Access from private IP {ip}"
                    })
            except ValueError:
                threats.append({
                    "type": "invalid_ip",
                    "level": ThreatLevel.MEDIUM,
                    "description": f"Invalid IP address format: {ip}"
                })
            
            # Check for suspicious activity patterns
            if ip in self.suspicious_actors:
                actor_data = self.suspicious_actors[ip]
                if actor_data.get("threat_score", 0) > 50:
                    threats.append({
                        "type": "suspicious_actor",
                        "level": ThreatLevel.MEDIUM,
                        "description": f"IP {ip} has suspicious activity history"
                    })
            
        except Exception as e:
            logger.exception(f"Error analyzing IP {ip}: {e}")
        
        return threats
    
    async def _check_rate_limits(
        self,
        source_ip: str,
        user_id: Optional[str],
        endpoint: str
    ) -> List[Dict[str, Any]]:
        """Check rate limits for the request."""
        threats = []
        current_time = int(datetime.utcnow().timestamp())
        
        try:
            for rule in self.security_policy.rate_limits:
                # Determine the key based on rule scope
                if rule.scope == "ip":
                    key = f"rate_limit:{rule.name}:ip:{source_ip}"
                elif rule.scope == "user" and user_id:
                    key = f"rate_limit:{rule.name}:user:{user_id}"
                elif rule.scope == "endpoint":
                    key = f"rate_limit:{rule.name}:endpoint:{endpoint}"
                else:
                    continue
                
                # Get current counter
                if self.redis_client:
                    count = await self.redis_client.get(key)
                    count = int(count) if count else 0
                else:
                    # In-memory fallback
                    count = self.rate_limit_counters.get(key, {}).get("count", 0)
                    last_reset = self.rate_limit_counters.get(key, {}).get("last_reset", current_time)
                    
                    # Reset counter if window has passed
                    if current_time - last_reset >= rule.time_window:
                        count = 0
                        self.rate_limit_counters[key] = {
                            "count": 0,
                            "last_reset": current_time
                        }
                
                # Check if limit exceeded
                if count >= rule.max_requests:
                    threats.append({
                        "type": "rate_limit_exceeded",
                        "level": ThreatLevel.MEDIUM,
                        "description": f"Rate limit exceeded for {rule.name}: {count}/{rule.max_requests}",
                        "rule": rule.name,
                        "count": count,
                        "limit": rule.max_requests
                    })
                
                # Increment counter
                if self.redis_client:
                    await self.redis_client.incr(key)
                    await self.redis_client.expire(key, rule.time_window)
                else:
                    if key not in self.rate_limit_counters:
                        self.rate_limit_counters[key] = {
                            "count": 0,
                            "last_reset": current_time
                        }
                    self.rate_limit_counters[key]["count"] += 1
        
        except Exception as e:
            logger.exception(f"Error checking rate limits: {e}")
        
        return threats
    
    async def _analyze_user_agent(self, user_agent: str) -> List[Dict[str, Any]]:
        """Analyze user agent for threats."""
        threats = []
        
        try:
            rule = self.threat_detection_rules.get("suspicious_user_agent", {})
            
            # Check whitelist first
            whitelist = rule.get("whitelist", [])
            for pattern in whitelist:
                if re.search(pattern, user_agent, re.IGNORECASE):
                    return threats  # Whitelisted, no threats
            
            # Check suspicious patterns
            patterns = rule.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, user_agent, re.IGNORECASE):
                    threats.append({
                        "type": "suspicious_user_agent",
                        "level": rule.get("threat_level", ThreatLevel.LOW),
                        "description": f"Suspicious user agent pattern detected: {pattern}",
                        "user_agent": user_agent
                    })
                    break  # Only report first match
        
        except Exception as e:
            logger.exception(f"Error analyzing user agent: {e}")
        
        return threats
    
    async def _analyze_payload(self, payload: str) -> List[Dict[str, Any]]:
        """Analyze request payload for threats."""
        threats = []
        
        try:
            # Check payload size
            if len(payload.encode()) > self.security_policy.payload_size_limit:
                threats.append({
                    "type": "oversized_payload",
                    "level": ThreatLevel.MEDIUM,
                    "description": f"Payload size exceeds limit: {len(payload.encode())} bytes"
                })
            
            # Check for suspicious patterns
            for pattern in self.security_policy.suspicious_patterns:
                matches = re.findall(pattern, payload, re.IGNORECASE | re.MULTILINE)
                if matches:
                    threats.append({
                        "type": "malicious_payload",
                        "level": ThreatLevel.HIGH,
                        "description": f"Malicious pattern detected: {pattern}",
                        "matches": matches[:5]  # Limit to first 5 matches
                    })
            
            # Check for code injection patterns
            injection_patterns = self.threat_detection_rules.get("payload_injection", {}).get("patterns", [])
            for pattern in injection_patterns:
                if re.search(pattern, payload, re.IGNORECASE):
                    threats.append({
                        "type": "code_injection",
                        "level": ThreatLevel.HIGH,
                        "description": f"Potential code injection detected: {pattern}"
                    })
            
            # Check for sensitive data patterns (API keys, passwords, etc.)
            sensitive_patterns = [
                r"password\s*[=:]\s*['\"][^'\"]+['\"]",
                r"api[_-]?key\s*[=:]\s*['\"][^'\"]+['\"]",
                r"secret\s*[=:]\s*['\"][^'\"]+['\"]",
                r"token\s*[=:]\s*['\"][^'\"]+['\"]"
            ]
            
            for pattern in sensitive_patterns:
                if re.search(pattern, payload, re.IGNORECASE):
                    threats.append({
                        "type": "sensitive_data_leak",
                        "level": ThreatLevel.MEDIUM,
                        "description": "Potential sensitive data in payload"
                    })
                    break  # Don't report multiple sensitive data patterns
        
        except Exception as e:
            logger.exception(f"Error analyzing payload: {e}")
        
        return threats
    
    async def _analyze_headers(self, headers: Dict[str, str]) -> List[Dict[str, Any]]:
        """Analyze request headers for threats."""
        threats = []
        
        try:
            # Check for missing required headers
            required_headers = ["user-agent", "host"]
            for header in required_headers:
                if header.lower() not in [h.lower() for h in headers.keys()]:
                    threats.append({
                        "type": "missing_required_header",
                        "level": ThreatLevel.LOW,
                        "description": f"Missing required header: {header}"
                    })
            
            # Check for suspicious header values
            for header_name, header_value in headers.items():
                # Check for injection in headers
                for pattern in self.security_policy.suspicious_patterns:
                    if re.search(pattern, header_value, re.IGNORECASE):
                        threats.append({
                            "type": "header_injection",
                            "level": ThreatLevel.MEDIUM,
                            "description": f"Suspicious pattern in header {header_name}",
                            "header": header_name
                        })
                        break
                
                # Check for overly long header values
                if len(header_value) > 8192:  # 8KB limit
                    threats.append({
                        "type": "oversized_header",
                        "level": ThreatLevel.MEDIUM,
                        "description": f"Oversized header {header_name}: {len(header_value)} bytes"
                    })
            
            # Check origin header if present
            origin = headers.get("origin") or headers.get("Origin")
            if origin and self.security_policy.allowed_origins:
                if not self._is_origin_allowed(origin):
                    threats.append({
                        "type": "unauthorized_origin",
                        "level": ThreatLevel.MEDIUM,
                        "description": f"Unauthorized origin: {origin}"
                    })
        
        except Exception as e:
            logger.exception(f"Error analyzing headers: {e}")
        
        return threats
    
    async def _get_ip_reputation(self, ip: str) -> str:
        """Get IP reputation (with caching)."""
        try:
            # Check cache first
            if ip in self._ip_reputation_cache:
                reputation, cached_at = self._ip_reputation_cache[ip]
                if datetime.utcnow() - cached_at < timedelta(hours=1):
                    return reputation
            
            # Placeholder for actual IP reputation service
            # In a real implementation, this would query external services
            reputation = "clean"  # Default to clean
            
            # Simple heuristics based on IP characteristics
            try:
                ip_obj = ipaddress.ip_address(ip)
                if ip_obj.is_private:
                    reputation = "internal"
                elif str(ip_obj).startswith(("10.", "192.168.", "172.")):
                    reputation = "internal"
                # Add more sophisticated reputation logic here
            except ValueError:
                reputation = "suspicious"
            
            # Cache the result
            self._ip_reputation_cache[ip] = (reputation, datetime.utcnow())
            
            return reputation
        
        except Exception as e:
            logger.exception(f"Error getting IP reputation for {ip}: {e}")
            return "unknown"
    
    def _is_allowed_private_ip(self, ip: str) -> bool:
        """Check if private IP is allowed."""
        # Add logic for allowed private IP ranges
        allowed_private_ranges = [
            "127.0.0.1",  # localhost
            "::1"         # IPv6 localhost
        ]
        return ip in allowed_private_ranges
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        for allowed_origin in self.security_policy.allowed_origins:
            if allowed_origin == "*":
                return True
            if allowed_origin.endswith("*"):
                if origin.startswith(allowed_origin[:-1]):
                    return True
            elif origin == allowed_origin:
                return True
        return False
    
    async def _log_security_event(
        self,
        event_type: SecurityEventType,
        threat_level: ThreatLevel,
        source_ip: str,
        user_agent: Optional[str],
        details: Dict[str, Any]
    ):
        """Log a security event."""
        try:
            event = SecurityEvent(
                event_id=secrets.token_urlsafe(16),
                event_type=event_type,
                threat_level=threat_level,
                source_ip=source_ip,
                user_agent=user_agent,
                timestamp=datetime.utcnow(),
                details=details
            )
            
            # Add to memory storage
            self.security_events.append(event)
            
            # Store in Redis if available
            if self.redis_client:
                key = f"security_events:{event.event_id}"
                await self.redis_client.setex(
                    key,
                    86400 * 7,  # 7 days TTL
                    json.dumps(asdict(event), default=str)
                )
            
            # Update statistics
            self.security_stats["total_events"] += 1
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self.security_stats["threats_detected"] += 1
            
            # Update suspicious actors
            if source_ip not in self.suspicious_actors:
                self.suspicious_actors[source_ip] = {
                    "first_seen": datetime.utcnow(),
                    "threat_score": 0,
                    "events": []
                }
            
            actor_data = self.suspicious_actors[source_ip]
            actor_data["events"].append(event.event_id)
            actor_data["last_seen"] = datetime.utcnow()
            
            # Increase threat score based on event severity
            score_increase = {
                ThreatLevel.LOW: 5,
                ThreatLevel.MEDIUM: 15,
                ThreatLevel.HIGH: 30,
                ThreatLevel.CRITICAL: 50
            }.get(threat_level, 0)
            
            actor_data["threat_score"] += score_increase
            
            # Auto-block if threat score is too high
            if actor_data["threat_score"] > 100:
                await self.block_ip(source_ip, f"High threat score: {actor_data['threat_score']}")
            
            logger.warning(
                f"Security event logged: {event_type.value} from {source_ip} "
                f"(threat level: {threat_level.value})"
            )
            
        except Exception as e:
            logger.exception(f"Error logging security event: {e}")
    
    async def block_ip(self, ip: str, reason: str, duration: Optional[int] = None):
        """Block an IP address."""
        try:
            self.blocked_ips.add(ip)
            
            # Store in Redis with TTL if duration specified
            if self.redis_client:
                key = f"blocked_ips:{ip}"
                value = {
                    "blocked_at": datetime.utcnow().isoformat(),
                    "reason": reason,
                    "duration": duration
                }
                
                if duration:
                    await self.redis_client.setex(key, duration, json.dumps(value))
                else:
                    await self.redis_client.set(key, json.dumps(value))
            
            # Log the blocking event
            await self._log_security_event(
                SecurityEventType.UNAUTHORIZED_ACCESS,
                ThreatLevel.HIGH,
                ip,
                None,
                {"action": "ip_blocked", "reason": reason, "duration": duration}
            )
            
            self.security_stats["blocked_requests"] += 1
            
            logger.warning(f"IP {ip} blocked: {reason}")
            
        except Exception as e:
            logger.exception(f"Error blocking IP {ip}: {e}")
    
    async def unblock_ip(self, ip: str):
        """Unblock an IP address."""
        try:
            self.blocked_ips.discard(ip)
            
            if self.redis_client:
                key = f"blocked_ips:{ip}"
                await self.redis_client.delete(key)
            
            # Reset suspicious actor data
            if ip in self.suspicious_actors:
                self.suspicious_actors[ip]["threat_score"] = 0
            
            logger.info(f"IP {ip} unblocked")
            
        except Exception as e:
            logger.exception(f"Error unblocking IP {ip}: {e}")
    
    async def verify_webhook_signature(
        self,
        payload: str,
        signature: str,
        secret: str
    ) -> bool:
        """Verify GitHub webhook signature."""
        try:
            # Check cache first
            cache_key = hashlib.md5(f"{payload}{signature}".encode()).hexdigest()
            if cache_key in self._signature_cache:
                return self._signature_cache[cache_key]
            
            if not signature.startswith("sha256="):
                result = False
            else:
                expected_signature = "sha256=" + hmac.new(
                    secret.encode(),
                    payload.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                result = hmac.compare_digest(signature, expected_signature)
            
            # Cache result (with limited cache size)
            if len(self._signature_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self._signature_cache.keys())[:100]
                for key in oldest_keys:
                    del self._signature_cache[key]
            
            self._signature_cache[cache_key] = result
            
            if not result:
                logger.warning(f"Invalid webhook signature detected")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error verifying webhook signature: {e}")
            return False
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data."""
        while self._monitoring_active:
            try:
                current_time = datetime.utcnow()
                
                # Clean up old rate limit counters
                expired_keys = []
                for key, data in self.rate_limit_counters.items():
                    if current_time.timestamp() - data.get("last_reset", 0) > 3600:  # 1 hour
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.rate_limit_counters[key]
                
                # Clean up old suspicious actors (reduce threat scores over time)
                for ip, data in self.suspicious_actors.items():
                    last_seen = data.get("last_seen", current_time)
                    if isinstance(last_seen, str):
                        last_seen = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
                    
                    hours_since_activity = (current_time - last_seen).total_seconds() / 3600
                    
                    if hours_since_activity > 24:  # Reduce score after 24 hours of inactivity
                        data["threat_score"] = max(0, data["threat_score"] - 10)
                    
                    # Remove actors with zero threat score after 7 days
                    if data["threat_score"] == 0 and hours_since_activity > 168:  # 7 days
                        expired_keys.append(ip)
                
                for ip in expired_keys:
                    if ip in self.suspicious_actors:
                        del self.suspicious_actors[ip]
                
                # Clean up IP reputation cache
                expired_ips = []
                for ip, (reputation, cached_at) in self._ip_reputation_cache.items():
                    if current_time - cached_at > timedelta(hours=24):
                        expired_ips.append(ip)
                
                for ip in expired_ips:
                    del self._ip_reputation_cache[ip]
                
                # Clean up signature cache
                if len(self._signature_cache) > 5000:
                    # Keep only the most recent 2500 entries
                    keys_to_remove = list(self._signature_cache.keys())[:-2500]
                    for key in keys_to_remove:
                        del self._signature_cache[key]
                
                await asyncio.sleep(3600)  # Run cleanup every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in security cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _load_persistent_data(self):
        """Load persistent security data."""
        try:
            if not self.redis_client:
                return
            
            # Load blocked IPs
            pattern = "blocked_ips:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                ip = key.split(":", 2)[-1]
                self.blocked_ips.add(ip)
            
            logger.info(f"Loaded {len(self.blocked_ips)} blocked IPs from persistence")
            
        except Exception as e:
            logger.exception(f"Error loading persistent security data: {e}")
    
    async def _save_persistent_data(self):
        """Save persistent security data."""
        try:
            # Security data is automatically persisted via Redis operations
            logger.info("Security data persisted")
            
        except Exception as e:
            logger.exception(f"Error saving persistent security data: {e}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "monitoring_active": self._monitoring_active,
            "blocked_ips_count": len(self.blocked_ips),
            "suspicious_actors_count": len(self.suspicious_actors),
            "recent_events_count": len(self.security_events),
            "security_stats": self.security_stats.copy(),
            "rate_limit_rules_count": len(self.security_policy.rate_limits),
            "cache_sizes": {
                "ip_reputation": len(self._ip_reputation_cache),
                "signature_verification": len(self._signature_cache),
                "rate_limit_counters": len(self.rate_limit_counters)
            },
            "threat_detection_rules_count": len(self.threat_detection_rules),
            "policy": {
                "signature_verification_required": self.security_policy.require_signature_verification,
                "payload_size_limit": self.security_policy.payload_size_limit,
                "geo_blocking_enabled": self.security_policy.enable_geo_blocking,
                "allowed_origins_count": len(self.security_policy.allowed_origins)
            }
        }
    
    def get_recent_security_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent security events."""
        recent_events = list(self.security_events)[-limit:]
        return [asdict(event) for event in recent_events]
    
    def get_top_suspicious_actors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top suspicious actors by threat score."""
        sorted_actors = sorted(
            self.suspicious_actors.items(),
            key=lambda x: x[1].get("threat_score", 0),
            reverse=True
        )
        
        return [
            {
                "ip": ip,
                "threat_score": data.get("threat_score", 0),
                "first_seen": data.get("first_seen").isoformat() if data.get("first_seen") else None,
                "last_seen": data.get("last_seen").isoformat() if data.get("last_seen") else None,
                "events_count": len(data.get("events", []))
            }
            for ip, data in sorted_actors[:limit]
        ]