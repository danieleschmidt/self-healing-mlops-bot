"""Advanced rate limiting with security awareness."""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

from ..monitoring.logging import get_logger, audit_logger
from .monitoring import SecurityEventType, ThreatLevel, security_monitor

logger = get_logger(__name__)


class RateLimitType(Enum):
    """Types of rate limits."""
    REQUESTS_PER_SECOND = "requests_per_second"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    BANDWIDTH_PER_SECOND = "bandwidth_per_second"
    CONCURRENT_CONNECTIONS = "concurrent_connections"


class RateLimitScope(Enum):
    """Scope of rate limiting."""
    GLOBAL = "global"
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_ENDPOINT = "per_endpoint"
    PER_API_KEY = "per_api_key"


@dataclass
class RateLimitRule:
    """Rate limit rule definition."""
    name: str
    limit_type: RateLimitType
    scope: RateLimitScope
    limit: int
    window_seconds: int
    burst_allowance: int = 0
    block_duration_seconds: int = 300  # 5 minutes default
    grace_period_seconds: int = 0
    enabled: bool = True
    priority: int = 1
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitViolation:
    """Rate limit violation record."""
    violation_id: str
    rule_name: str
    client_identifier: str
    timestamp: datetime
    current_rate: float
    limit: int
    severity: ThreatLevel
    blocked_until: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)


class SecurityAwareRateLimiter:
    """Advanced rate limiter with security intelligence."""
    
    def __init__(self):
        self.rules: Dict[str, RateLimitRule] = {}
        self.client_state: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'requests': deque(),
            'violations': [],
            'trust_score': 1.0,
            'last_activity': None,
            'blocked_until': None,
            'bandwidth_used': deque(),
            'concurrent_connections': 0
        })
        
        self.global_state = {
            'total_requests': deque(),
            'active_connections': 0,
            'total_bandwidth': deque()
        }
        
        self.violation_history: deque = deque(maxlen=10000)
        self.adaptive_limits: Dict[str, float] = {}
        self.threat_multipliers: Dict[ThreatLevel, float] = {
            ThreatLevel.LOW: 1.0,
            ThreatLevel.MEDIUM: 0.7,
            ThreatLevel.HIGH: 0.4,
            ThreatLevel.CRITICAL: 0.1
        }
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default rate limiting rules."""
        default_rules = [
            RateLimitRule(
                name="global_requests_per_second",
                limit_type=RateLimitType.REQUESTS_PER_SECOND,
                scope=RateLimitScope.GLOBAL,
                limit=1000,
                window_seconds=1,
                burst_allowance=100,
                priority=1
            ),
            RateLimitRule(
                name="per_ip_requests_per_minute",
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                scope=RateLimitScope.PER_IP,
                limit=100,
                window_seconds=60,
                burst_allowance=20,
                block_duration_seconds=300,
                priority=2
            ),
            RateLimitRule(
                name="per_user_requests_per_minute",
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                scope=RateLimitScope.PER_USER,
                limit=200,
                window_seconds=60,
                burst_allowance=50,
                priority=2
            ),
            RateLimitRule(
                name="webhook_endpoint_rate_limit",
                limit_type=RateLimitType.REQUESTS_PER_SECOND,
                scope=RateLimitScope.PER_ENDPOINT,
                limit=10,
                window_seconds=1,
                burst_allowance=5,
                priority=3,
                conditions={"endpoint_pattern": "/webhooks/*"}
            ),
            RateLimitRule(
                name="api_bandwidth_limit",
                limit_type=RateLimitType.BANDWIDTH_PER_SECOND,
                scope=RateLimitScope.PER_IP,
                limit=1024*1024,  # 1MB/s
                window_seconds=1,
                burst_allowance=1024*1024*2,  # 2MB burst
                priority=2
            ),
            RateLimitRule(
                name="concurrent_connections_per_ip",
                limit_type=RateLimitType.CONCURRENT_CONNECTIONS,
                scope=RateLimitScope.PER_IP,
                limit=10,
                window_seconds=0,  # Not applicable for concurrent limits
                block_duration_seconds=60,
                priority=1
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.name] = rule
    
    def add_rule(self, rule: RateLimitRule):
        """Add a new rate limiting rule."""
        self.rules[rule.name] = rule
        logger.info(f"Added rate limiting rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a rate limiting rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed rate limiting rule: {rule_name}")
    
    async def check_rate_limit(self, client_ip: str, endpoint: str = "/", 
                             user_id: Optional[str] = None, api_key: Optional[str] = None,
                             request_size: int = 0) -> Tuple[bool, Optional[RateLimitViolation]]:
        """Check if request should be rate limited."""
        now = datetime.utcnow()
        current_time = time.time()
        
        # Update client activity
        client_state = self.client_state[client_ip]
        client_state['last_activity'] = now
        
        # Check if client is currently blocked
        if client_state['blocked_until'] and now < client_state['blocked_until']:
            violation = RateLimitViolation(
                violation_id=self._generate_violation_id(client_ip, "blocked"),
                rule_name="client_blocked",
                client_identifier=client_ip,
                timestamp=now,
                current_rate=0,
                limit=0,
                severity=ThreatLevel.HIGH,
                blocked_until=client_state['blocked_until'],
                details={"reason": "client_blocked", "blocked_until": client_state['blocked_until'].isoformat()}
            )
            return False, violation
        
        # Get client trust score and threat level
        trust_score = await self._calculate_trust_score(client_ip)
        threat_level = await self._assess_threat_level(client_ip, endpoint, user_id)
        
        # Check each applicable rule
        violated_rule = None
        for rule in sorted(self.rules.values(), key=lambda r: r.priority, reverse=True):
            if not rule.enabled:
                continue
            
            # Check if rule applies to this request
            if not self._rule_applies(rule, client_ip, endpoint, user_id, api_key):
                continue
            
            # Calculate effective limit based on trust score and threat level
            effective_limit = self._calculate_effective_limit(rule, trust_score, threat_level)
            
            # Check the specific rate limit
            is_violated, current_rate = await self._check_specific_limit(
                rule, client_ip, endpoint, user_id, api_key, request_size, effective_limit
            )
            
            if is_violated:
                violation = await self._handle_violation(
                    rule, client_ip, current_rate, effective_limit, threat_level, user_id
                )
                violated_rule = violation
                break
        
        # Record successful request if no violations
        if not violated_rule:
            await self._record_successful_request(client_ip, endpoint, user_id, request_size)
        
        return violated_rule is None, violated_rule
    
    def _rule_applies(self, rule: RateLimitRule, client_ip: str, endpoint: str,
                     user_id: Optional[str], api_key: Optional[str]) -> bool:
        """Check if a rule applies to the current request."""
        # Check scope
        if rule.scope == RateLimitScope.GLOBAL:
            pass  # Always applies
        elif rule.scope == RateLimitScope.PER_IP and not client_ip:
            return False
        elif rule.scope == RateLimitScope.PER_USER and not user_id:
            return False
        elif rule.scope == RateLimitScope.PER_API_KEY and not api_key:
            return False
        
        # Check conditions
        if rule.conditions:
            if "endpoint_pattern" in rule.conditions:
                pattern = rule.conditions["endpoint_pattern"]
                if pattern.endswith("*"):
                    if not endpoint.startswith(pattern[:-1]):
                        return False
                elif pattern != endpoint:
                    return False
        
        return True
    
    async def _check_specific_limit(self, rule: RateLimitRule, client_ip: str, endpoint: str,
                                  user_id: Optional[str], api_key: Optional[str], 
                                  request_size: int, effective_limit: int) -> Tuple[bool, float]:
        """Check a specific rate limit rule."""
        now = datetime.utcnow()
        current_time = time.time()
        
        # Get the appropriate state based on scope
        if rule.scope == RateLimitScope.GLOBAL:
            state = self.global_state
            identifier = "global"
        elif rule.scope == RateLimitScope.PER_IP:
            state = self.client_state[client_ip]
            identifier = client_ip
        elif rule.scope == RateLimitScope.PER_USER:
            state = self.client_state[user_id or client_ip]
            identifier = user_id or client_ip
        elif rule.scope == RateLimitScope.PER_API_KEY:
            state = self.client_state[api_key or client_ip]
            identifier = api_key or client_ip
        elif rule.scope == RateLimitScope.PER_ENDPOINT:
            endpoint_key = f"{client_ip}:{endpoint}"
            state = self.client_state[endpoint_key]
            identifier = endpoint_key
        else:
            state = self.client_state[client_ip]
            identifier = client_ip
        
        if rule.limit_type == RateLimitType.REQUESTS_PER_SECOND:
            return self._check_request_rate(state, rule, effective_limit, 1)
        elif rule.limit_type == RateLimitType.REQUESTS_PER_MINUTE:
            return self._check_request_rate(state, rule, effective_limit, 60)
        elif rule.limit_type == RateLimitType.REQUESTS_PER_HOUR:
            return self._check_request_rate(state, rule, effective_limit, 3600)
        elif rule.limit_type == RateLimitType.BANDWIDTH_PER_SECOND:
            return self._check_bandwidth_rate(state, rule, effective_limit, request_size)
        elif rule.limit_type == RateLimitType.CONCURRENT_CONNECTIONS:
            return self._check_concurrent_connections(state, rule, effective_limit)
        
        return False, 0.0
    
    def _check_request_rate(self, state: Dict[str, Any], rule: RateLimitRule, 
                          limit: int, window_seconds: int) -> Tuple[bool, float]:
        """Check request rate limit."""
        now = time.time()
        window_start = now - window_seconds
        
        # Clean old requests
        requests = state['requests']
        while requests and requests[0] < window_start:
            requests.popleft()
        
        current_rate = len(requests)
        
        # Check against limit (including burst allowance)
        effective_limit_with_burst = limit + rule.burst_allowance
        
        return current_rate >= effective_limit_with_burst, current_rate
    
    def _check_bandwidth_rate(self, state: Dict[str, Any], rule: RateLimitRule,
                            limit: int, request_size: int) -> Tuple[bool, float]:
        """Check bandwidth rate limit."""
        now = time.time()
        window_start = now - rule.window_seconds
        
        # Clean old bandwidth records
        bandwidth_used = state['bandwidth_used']
        while bandwidth_used and bandwidth_used[0][0] < window_start:
            bandwidth_used.popleft()
        
        # Calculate current bandwidth usage
        current_bandwidth = sum(size for _, size in bandwidth_used) + request_size
        
        effective_limit_with_burst = limit + rule.burst_allowance
        
        return current_bandwidth > effective_limit_with_burst, current_bandwidth
    
    def _check_concurrent_connections(self, state: Dict[str, Any], rule: RateLimitRule,
                                    limit: int) -> Tuple[bool, float]:
        """Check concurrent connections limit."""
        current_connections = state.get('concurrent_connections', 0)
        return current_connections >= limit, current_connections
    
    async def _calculate_trust_score(self, client_ip: str) -> float:
        """Calculate trust score for a client."""
        client_state = self.client_state[client_ip]
        
        # Base trust score
        trust_score = 1.0
        
        # Reduce trust based on violations
        violations = client_state.get('violations', [])
        recent_violations = [v for v in violations 
                           if datetime.utcnow() - v.timestamp < timedelta(hours=1)]
        
        if recent_violations:
            trust_score *= max(0.1, 1.0 - (len(recent_violations) * 0.2))
        
        # Reduce trust based on threat indicators
        threat_indicator = security_monitor.check_threat_indicators("ip", client_ip)
        if threat_indicator:
            confidence_multiplier = 1.0 - threat_indicator.confidence
            trust_score *= max(0.1, confidence_multiplier)
        
        return max(0.1, trust_score)  # Minimum trust score of 0.1
    
    async def _assess_threat_level(self, client_ip: str, endpoint: str, 
                                 user_id: Optional[str]) -> ThreatLevel:
        """Assess threat level for a request."""
        # Check if IP is in threat intelligence
        threat_indicator = security_monitor.check_threat_indicators("ip", client_ip)
        if threat_indicator:
            return threat_indicator.threat_level
        
        # Check recent security events from this IP
        recent_events = security_monitor.get_recent_events(minutes=60)
        ip_events = [event for event in recent_events if event.get('source_ip') == client_ip]
        
        if ip_events:
            high_severity_events = [event for event in ip_events 
                                  if event.get('threat_level') in ['high', 'critical']]
            if high_severity_events:
                return ThreatLevel.HIGH
            elif len(ip_events) > 10:  # Many events from same IP
                return ThreatLevel.MEDIUM
        
        return ThreatLevel.LOW
    
    def _calculate_effective_limit(self, rule: RateLimitRule, trust_score: float, 
                                 threat_level: ThreatLevel) -> int:
        """Calculate effective rate limit based on trust score and threat level."""
        base_limit = rule.limit
        
        # Apply trust score multiplier
        trust_multiplier = trust_score
        
        # Apply threat level multiplier
        threat_multiplier = self.threat_multipliers[threat_level]
        
        # Calculate effective limit
        effective_limit = int(base_limit * trust_multiplier * threat_multiplier)
        
        # Ensure minimum limit
        return max(1, effective_limit)
    
    async def _handle_violation(self, rule: RateLimitRule, client_ip: str, 
                              current_rate: float, limit: int, threat_level: ThreatLevel,
                              user_id: Optional[str]) -> RateLimitViolation:
        """Handle a rate limit violation."""
        now = datetime.utcnow()
        
        # Create violation record
        violation = RateLimitViolation(
            violation_id=self._generate_violation_id(client_ip, rule.name),
            rule_name=rule.name,
            client_identifier=client_ip,
            timestamp=now,
            current_rate=current_rate,
            limit=limit,
            severity=threat_level,
            details={
                "rule_type": rule.limit_type.value,
                "scope": rule.scope.value,
                "user_id": user_id,
                "threat_level": threat_level.value
            }
        )
        
        # Determine blocking duration based on severity and violations
        client_state = self.client_state[client_ip]
        recent_violations = len([v for v in client_state.get('violations', [])
                               if now - v.timestamp < timedelta(hours=1)])
        
        # Progressive blocking
        base_duration = rule.block_duration_seconds
        progressive_multiplier = min(5, 1.5 ** recent_violations)  # Cap at 5x
        
        if threat_level == ThreatLevel.CRITICAL:
            block_duration = base_duration * 10  # 10x for critical threats
        elif threat_level == ThreatLevel.HIGH:
            block_duration = base_duration * 3 * progressive_multiplier
        elif threat_level == ThreatLevel.MEDIUM:
            block_duration = base_duration * progressive_multiplier
        else:
            block_duration = base_duration
        
        # Block the client
        blocked_until = now + timedelta(seconds=block_duration)
        client_state['blocked_until'] = blocked_until
        violation.blocked_until = blocked_until
        
        # Record violation
        client_state.setdefault('violations', []).append(violation)
        self.violation_history.append(violation)
        
        # Log security event
        await security_monitor.log_security_event(
            SecurityEventType.POLICY_VIOLATION,
            threat_level,
            source_ip=client_ip,
            endpoint=rule.conditions.get('endpoint_pattern', 'unknown'),
            user_id=user_id,
            details={
                "rule_violated": rule.name,
                "current_rate": current_rate,
                "limit": limit,
                "blocked_for_seconds": block_duration,
                "violation_id": violation.violation_id
            }
        )
        
        logger.warning(f"Rate limit violation: {rule.name} by {client_ip} "
                      f"({current_rate}/{limit}) - blocked for {block_duration}s")
        
        return violation
    
    async def _record_successful_request(self, client_ip: str, endpoint: str,
                                       user_id: Optional[str], request_size: int):
        """Record a successful request for rate limiting calculations."""
        now = time.time()
        
        # Record in client state
        client_state = self.client_state[client_ip]
        client_state['requests'].append(now)
        
        if request_size > 0:
            client_state['bandwidth_used'].append((now, request_size))
        
        # Record in global state
        self.global_state['total_requests'].append(now)
        if request_size > 0:
            self.global_state['total_bandwidth'].append((now, request_size))
    
    def _generate_violation_id(self, client_ip: str, rule_name: str) -> str:
        """Generate a unique violation ID."""
        data = f"{client_ip}_{rule_name}_{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    async def increment_concurrent_connections(self, client_ip: str):
        """Increment concurrent connection count for a client."""
        self.client_state[client_ip]['concurrent_connections'] += 1
        self.global_state['active_connections'] += 1
    
    async def decrement_concurrent_connections(self, client_ip: str):
        """Decrement concurrent connection count for a client."""
        client_state = self.client_state[client_ip]
        if client_state['concurrent_connections'] > 0:
            client_state['concurrent_connections'] -= 1
        
        if self.global_state['active_connections'] > 0:
            self.global_state['active_connections'] -= 1
    
    def get_client_status(self, client_ip: str) -> Dict[str, Any]:
        """Get current status for a client."""
        client_state = self.client_state[client_ip]
        now = datetime.utcnow()
        
        # Calculate current rates
        current_time = time.time()
        recent_requests = [t for t in client_state['requests'] if current_time - t < 60]
        recent_bandwidth = [(t, s) for t, s in client_state['bandwidth_used'] 
                           if current_time - t < 60]
        
        return {
            "client_ip": client_ip,
            "is_blocked": client_state.get('blocked_until', datetime.min) > now,
            "blocked_until": client_state.get('blocked_until'),
            "trust_score": asyncio.run(self._calculate_trust_score(client_ip)),
            "current_request_rate_per_minute": len(recent_requests),
            "current_bandwidth_per_minute": sum(s for _, s in recent_bandwidth),
            "concurrent_connections": client_state.get('concurrent_connections', 0),
            "total_violations": len(client_state.get('violations', [])),
            "last_activity": client_state.get('last_activity')
        }
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global rate limiting metrics."""
        now = time.time()
        
        # Calculate current global rates
        recent_requests = [t for t in self.global_state['total_requests'] if now - t < 60]
        recent_bandwidth = [(t, s) for t, s in self.global_state['total_bandwidth'] 
                           if now - t < 60]
        
        return {
            "active_clients": len(self.client_state),
            "active_connections": self.global_state['active_connections'],
            "current_request_rate_per_minute": len(recent_requests),
            "current_bandwidth_per_minute": sum(s for _, s in recent_bandwidth),
            "total_violations_last_hour": len([v for v in self.violation_history
                                             if datetime.utcnow() - v.timestamp < timedelta(hours=1)]),
            "blocked_clients": len([c for c in self.client_state.values() 
                                  if c.get('blocked_until', datetime.min) > datetime.utcnow()]),
            "rules_enabled": len([r for r in self.rules.values() if r.enabled]),
            "rules_total": len(self.rules)
        }
    
    async def unblock_client(self, client_ip: str, reason: str = "manual_unblock"):
        """Manually unblock a client."""
        client_state = self.client_state[client_ip]
        if 'blocked_until' in client_state:
            del client_state['blocked_until']
        
        logger.info(f"Unblocked client {client_ip}: {reason}")
        
        await security_monitor.log_security_event(
            SecurityEventType.POLICY_VIOLATION,
            ThreatLevel.LOW,
            source_ip=client_ip,
            details={"action": "client_unblocked", "reason": reason}
        )
    
    async def cleanup_old_data(self):
        """Clean up old rate limiting data."""
        now = time.time()
        cutoff_time = now - 3600  # 1 hour
        
        # Clean client state
        for client_ip, state in list(self.client_state.items()):
            # Clean old requests
            while state['requests'] and state['requests'][0] < cutoff_time:
                state['requests'].popleft()
            
            # Clean old bandwidth data
            while state['bandwidth_used'] and state['bandwidth_used'][0][0] < cutoff_time:
                state['bandwidth_used'].popleft()
            
            # Clean old violations
            cutoff_datetime = datetime.utcnow() - timedelta(hours=24)
            state['violations'] = [v for v in state.get('violations', [])
                                 if v.timestamp > cutoff_datetime]
            
            # Remove clients with no recent activity
            if (not state['requests'] and not state['bandwidth_used'] 
                and state.get('concurrent_connections', 0) == 0):
                del self.client_state[client_ip]
        
        # Clean global state
        while self.global_state['total_requests'] and self.global_state['total_requests'][0] < cutoff_time:
            self.global_state['total_requests'].popleft()
        
        while self.global_state['total_bandwidth'] and self.global_state['total_bandwidth'][0][0] < cutoff_time:
            self.global_state['total_bandwidth'].popleft()


# Global instance
rate_limiter = SecurityAwareRateLimiter()