"""Automated incident response system."""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..core.config import config
from ..monitoring.logging import get_logger, audit_logger
from .monitoring import SecurityEvent, SecurityEventType, ThreatLevel, security_monitor
from .auth import auth_manager, User, UserRole
from .rate_limiting import rate_limiter
from .threat_intelligence import threat_intelligence, IndicatorType, ThreatType

logger = get_logger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"


class ResponseActionType(Enum):
    """Types of automated response actions."""
    BLOCK_IP = "block_ip"
    DISABLE_USER = "disable_user"
    REVOKE_TOKENS = "revoke_tokens"
    RATE_LIMIT = "rate_limit"
    ALERT_ADMINS = "alert_admins"
    ISOLATE_SYSTEM = "isolate_system"
    BACKUP_DATA = "backup_data"
    PATCH_SYSTEM = "patch_system"
    COLLECT_EVIDENCE = "collect_evidence"
    NOTIFY_EXTERNAL = "notify_external"


@dataclass
class ResponseAction:
    """Automated response action."""
    action_id: str
    action_type: ResponseActionType
    parameters: Dict[str, Any]
    executed_at: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


@dataclass
class IncidentTicket:
    """Security incident ticket."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    source_events: List[str] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    indicators_of_compromise: List[Dict[str, str]] = field(default_factory=list)
    response_actions: List[ResponseAction] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    lessons_learned: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlaybookRule:
    """Incident response playbook rule."""
    rule_id: str
    name: str
    description: str
    event_types: Set[SecurityEventType]
    threat_levels: Set[ThreatLevel]
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    cooldown_minutes: int = 60
    enabled: bool = True
    priority: int = 1


class IncidentResponseManager:
    """Automated incident response and management system."""
    
    def __init__(self):
        self.active_incidents: Dict[str, IncidentTicket] = {}
        self.incident_history: List[IncidentTicket] = []
        self.playbook_rules: Dict[str, PlaybookRule] = {}
        self.response_actions: Dict[ResponseActionType, Callable] = {}
        self.rule_execution_history: Dict[str, datetime] = {}  # rule_id -> last_execution
        
        # Initialize default playbooks
        self._initialize_default_playbooks()
        
        # Register response actions
        self._register_response_actions()
        
        # Register with security monitor
        self._register_event_handlers()
    
    def _initialize_default_playbooks(self):
        """Initialize default incident response playbooks."""
        default_playbooks = [
            PlaybookRule(
                rule_id="critical_auth_failure",
                name="Critical Authentication Failures",
                description="Respond to critical authentication failures",
                event_types={SecurityEventType.AUTHENTICATION_FAILURE},
                threat_levels={ThreatLevel.CRITICAL},
                conditions={"failed_attempts_threshold": 10},
                actions=[
                    {"type": ResponseActionType.BLOCK_IP.value, "params": {"duration_minutes": 60}},
                    {"type": ResponseActionType.ALERT_ADMINS.value, "params": {"priority": "high"}},
                    {"type": ResponseActionType.COLLECT_EVIDENCE.value, "params": {}}
                ],
                cooldown_minutes=30
            ),
            PlaybookRule(
                rule_id="brute_force_attack",
                name="Brute Force Attack Response",
                description="Respond to brute force attacks",
                event_types={SecurityEventType.BRUTE_FORCE_ATTEMPT},
                threat_levels={ThreatLevel.HIGH, ThreatLevel.CRITICAL},
                actions=[
                    {"type": ResponseActionType.BLOCK_IP.value, "params": {"duration_minutes": 120}},
                    {"type": ResponseActionType.RATE_LIMIT.value, "params": {"factor": 0.1}},
                    {"type": ResponseActionType.ALERT_ADMINS.value, "params": {}}
                ],
                cooldown_minutes=15
            ),
            PlaybookRule(
                rule_id="data_breach_attempt",
                name="Data Breach Response",
                description="Respond to potential data breaches",
                event_types={SecurityEventType.DATA_BREACH_ATTEMPT},
                threat_levels={ThreatLevel.HIGH, ThreatLevel.CRITICAL},
                actions=[
                    {"type": ResponseActionType.ISOLATE_SYSTEM.value, "params": {}},
                    {"type": ResponseActionType.BACKUP_DATA.value, "params": {}},
                    {"type": ResponseActionType.ALERT_ADMINS.value, "params": {"priority": "critical"}},
                    {"type": ResponseActionType.NOTIFY_EXTERNAL.value, "params": {"authorities": True}}
                ],
                cooldown_minutes=5
            ),
            PlaybookRule(
                rule_id="malware_detected",
                name="Malware Detection Response",
                description="Respond to malware detection",
                event_types={SecurityEventType.MALWARE_DETECTED},
                threat_levels={ThreatLevel.HIGH, ThreatLevel.CRITICAL},
                actions=[
                    {"type": ResponseActionType.ISOLATE_SYSTEM.value, "params": {}},
                    {"type": ResponseActionType.COLLECT_EVIDENCE.value, "params": {"forensics": True}},
                    {"type": ResponseActionType.PATCH_SYSTEM.value, "params": {}},
                    {"type": ResponseActionType.ALERT_ADMINS.value, "params": {}}
                ],
                cooldown_minutes=10
            ),
            PlaybookRule(
                rule_id="privilege_escalation",
                name="Privilege Escalation Response",
                description="Respond to privilege escalation attempts",
                event_types={SecurityEventType.PRIVILEGE_ESCALATION},
                threat_levels={ThreatLevel.HIGH, ThreatLevel.CRITICAL},
                actions=[
                    {"type": ResponseActionType.DISABLE_USER.value, "params": {"temporary": True}},
                    {"type": ResponseActionType.REVOKE_TOKENS.value, "params": {}},
                    {"type": ResponseActionType.ALERT_ADMINS.value, "params": {"priority": "high"}},
                    {"type": ResponseActionType.COLLECT_EVIDENCE.value, "params": {}}
                ],
                cooldown_minutes=5
            )
        ]
        
        for playbook in default_playbooks:
            self.playbook_rules[playbook.rule_id] = playbook
    
    def _register_response_actions(self):
        """Register automated response action handlers."""
        self.response_actions = {
            ResponseActionType.BLOCK_IP: self._action_block_ip,
            ResponseActionType.DISABLE_USER: self._action_disable_user,
            ResponseActionType.REVOKE_TOKENS: self._action_revoke_tokens,
            ResponseActionType.RATE_LIMIT: self._action_adjust_rate_limit,
            ResponseActionType.ALERT_ADMINS: self._action_alert_admins,
            ResponseActionType.ISOLATE_SYSTEM: self._action_isolate_system,
            ResponseActionType.BACKUP_DATA: self._action_backup_data,
            ResponseActionType.PATCH_SYSTEM: self._action_patch_system,
            ResponseActionType.COLLECT_EVIDENCE: self._action_collect_evidence,
            ResponseActionType.NOTIFY_EXTERNAL: self._action_notify_external
        }
    
    def _register_event_handlers(self):
        """Register event handlers with security monitor."""
        for event_type in SecurityEventType:
            security_monitor.register_event_handler(event_type, self._handle_security_event)
    
    async def _handle_security_event(self, event: SecurityEvent):
        """Handle incoming security event and trigger response if needed."""
        try:
            # Find matching playbook rules
            matching_rules = self._find_matching_rules(event)
            
            for rule in matching_rules:
                # Check cooldown
                if not self._check_rule_cooldown(rule.rule_id):
                    continue
                
                # Check additional conditions
                if not await self._check_rule_conditions(rule, event):
                    continue
                
                # Execute automated response
                await self._execute_playbook(rule, event)
                
                # Update rule execution history
                self.rule_execution_history[rule.rule_id] = datetime.utcnow()
                
        except Exception as e:
            logger.error(f"Error handling security event {event.event_id}: {e}")
    
    def _find_matching_rules(self, event: SecurityEvent) -> List[PlaybookRule]:
        """Find playbook rules that match the security event."""
        matching_rules = []
        
        for rule in self.playbook_rules.values():
            if not rule.enabled:
                continue
            
            # Check event type
            if event.event_type not in rule.event_types:
                continue
            
            # Check threat level
            if event.threat_level not in rule.threat_levels:
                continue
            
            matching_rules.append(rule)
        
        # Sort by priority (higher priority first)
        return sorted(matching_rules, key=lambda r: r.priority, reverse=True)
    
    def _check_rule_cooldown(self, rule_id: str) -> bool:
        """Check if rule is not in cooldown period."""
        if rule_id not in self.rule_execution_history:
            return True
        
        rule = self.playbook_rules[rule_id]
        last_execution = self.rule_execution_history[rule_id]
        cooldown_expires = last_execution + timedelta(minutes=rule.cooldown_minutes)
        
        return datetime.utcnow() > cooldown_expires
    
    async def _check_rule_conditions(self, rule: PlaybookRule, event: SecurityEvent) -> bool:
        """Check additional rule conditions."""
        if not rule.conditions:
            return True
        
        # Check failed attempts threshold
        if "failed_attempts_threshold" in rule.conditions:
            threshold = rule.conditions["failed_attempts_threshold"]
            if event.source_ip:
                recent_events = security_monitor.get_recent_events(minutes=60)
                ip_failures = len([
                    e for e in recent_events
                    if e.get('source_ip') == event.source_ip and 
                       e.get('event_type') == SecurityEventType.AUTHENTICATION_FAILURE.value
                ])
                if ip_failures < threshold:
                    return False
        
        # Add more condition checks as needed
        
        return True
    
    async def _execute_playbook(self, rule: PlaybookRule, event: SecurityEvent):
        """Execute automated response playbook."""
        logger.info(f"Executing playbook rule: {rule.name} for event {event.event_id}")
        
        # Create or update incident
        incident = await self._create_or_update_incident(rule, event)
        
        # Execute response actions
        for action_config in rule.actions:
            try:
                action = await self._execute_response_action(action_config, event, incident)
                incident.response_actions.append(action)
                
                # Add to timeline
                incident.timeline.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "response_action_executed",
                    "action_type": action.action_type.value,
                    "success": action.success,
                    "details": action.parameters
                })
                
            except Exception as e:
                logger.error(f"Failed to execute response action {action_config}: {e}")
        
        # Update incident
        incident.updated_at = datetime.utcnow()
        
        # Log incident response
        audit_logger.log_security_event(
            "incident_response_executed", "info",
            {
                "incident_id": incident.incident_id,
                "rule_name": rule.name,
                "event_id": event.event_id,
                "actions_executed": len(incident.response_actions)
            }
        )
    
    async def _create_or_update_incident(self, rule: PlaybookRule, event: SecurityEvent) -> IncidentTicket:
        """Create new incident or update existing one."""
        # Check if there's an existing related incident
        existing_incident = self._find_related_incident(event)
        
        if existing_incident:
            # Update existing incident
            existing_incident.source_events.append(event.event_id)
            existing_incident.updated_at = datetime.utcnow()
            
            # Escalate severity if needed
            if event.threat_level == ThreatLevel.CRITICAL:
                existing_incident.severity = IncidentSeverity.CRITICAL
            elif event.threat_level == ThreatLevel.HIGH and existing_incident.severity == IncidentSeverity.MEDIUM:
                existing_incident.severity = IncidentSeverity.HIGH
            
            return existing_incident
        
        # Create new incident
        incident_id = f"INC_{int(time.time())}_{hashlib.md5(event.event_id.encode()).hexdigest()[:8]}"
        
        # Map threat level to incident severity
        severity_mapping = {
            ThreatLevel.CRITICAL: IncidentSeverity.CRITICAL,
            ThreatLevel.HIGH: IncidentSeverity.HIGH,
            ThreatLevel.MEDIUM: IncidentSeverity.MEDIUM,
            ThreatLevel.LOW: IncidentSeverity.LOW
        }
        
        incident = IncidentTicket(
            incident_id=incident_id,
            title=f"Security Incident: {rule.name}",
            description=f"Automated incident created for {event.event_type.value} event",
            severity=severity_mapping.get(event.threat_level, IncidentSeverity.MEDIUM),
            status=IncidentStatus.INVESTIGATING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            source_events=[event.event_id],
            affected_systems=[event.endpoint] if event.endpoint else [],
            timeline=[
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "incident_created",
                    "details": {"trigger_event": event.event_id, "playbook_rule": rule.name}
                }
            ]
        )
        
        # Add indicators of compromise
        if event.source_ip:
            incident.indicators_of_compromise.append({
                "type": "ip",
                "value": event.source_ip,
                "source": "event_data"
            })
        
        # Store incident
        self.active_incidents[incident_id] = incident
        
        logger.info(f"Created incident {incident_id} for event {event.event_id}")
        
        return incident
    
    def _find_related_incident(self, event: SecurityEvent) -> Optional[IncidentTicket]:
        """Find existing incident related to this event."""
        # Look for incidents from same IP in last 4 hours
        if event.source_ip:
            cutoff_time = datetime.utcnow() - timedelta(hours=4)
            
            for incident in self.active_incidents.values():
                if incident.created_at < cutoff_time:
                    continue
                
                # Check if same IP is in indicators
                for ioc in incident.indicators_of_compromise:
                    if ioc.get("type") == "ip" and ioc.get("value") == event.source_ip:
                        return incident
        
        return None
    
    async def _execute_response_action(self, action_config: Dict[str, Any], 
                                     event: SecurityEvent, incident: IncidentTicket) -> ResponseAction:
        """Execute a single response action."""
        action_type = ResponseActionType(action_config["type"])
        parameters = action_config.get("params", {})
        
        # Add event and incident context
        parameters.update({
            "event_id": event.event_id,
            "incident_id": incident.incident_id,
            "source_ip": event.source_ip,
            "user_id": event.user_id
        })
        
        action = ResponseAction(
            action_id=f"{action_type.value}_{int(time.time())}_{hashlib.md5(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:8]}",
            action_type=action_type,
            parameters=parameters
        )
        
        start_time = time.time()
        
        try:
            # Execute the action
            handler = self.response_actions.get(action_type)
            if handler:
                await handler(parameters)
                action.success = True
            else:
                raise Exception(f"No handler for action type: {action_type.value}")
                
        except Exception as e:
            action.error_message = str(e)
            action.success = False
            logger.error(f"Response action failed: {action_type.value} - {e}")
        
        action.executed_at = datetime.utcnow()
        action.execution_time_ms = (time.time() - start_time) * 1000
        
        return action
    
    # Response action implementations
    async def _action_block_ip(self, params: Dict[str, Any]):
        """Block IP address."""
        ip = params.get("source_ip")
        duration_minutes = params.get("duration_minutes", 60)
        reason = f"Automated block: incident {params.get('incident_id')}"
        
        if ip:
            await rate_limiter.unblock_client(ip)  # First unblock to reset
            # Then block by adding to security monitor
            await security_monitor.block_ip(ip, reason)
            
            # Also add to threat intelligence
            threat_intelligence.add_custom_indicator(
                IndicatorType.IP_ADDRESS,
                ip,
                {ThreatType.MALICIOUS_IP},
                0.8,
                f"Blocked by incident response: {reason}",
                {"automated_block", "incident_response"}
            )
            
            logger.info(f"Blocked IP {ip} for {duration_minutes} minutes")
    
    async def _action_disable_user(self, params: Dict[str, Any]):
        """Disable user account."""
        user_id = params.get("user_id")
        temporary = params.get("temporary", True)
        
        if user_id:
            user = auth_manager.get_user_by_id(user_id)
            if user:
                user.is_active = False
                if temporary:
                    user.locked_until = datetime.utcnow() + timedelta(hours=1)
                
                logger.info(f"Disabled user {user.username} ({'temporarily' if temporary else 'permanently'})")
    
    async def _action_revoke_tokens(self, params: Dict[str, Any]):
        """Revoke user tokens."""
        user_id = params.get("user_id")
        
        if user_id:
            auth_manager.revoke_user_tokens(user_id)
            logger.info(f"Revoked all tokens for user {user_id}")
    
    async def _action_adjust_rate_limit(self, params: Dict[str, Any]):
        """Adjust rate limiting."""
        ip = params.get("source_ip")
        factor = params.get("factor", 0.5)  # Reduce by 50%
        
        if ip:
            # This would implement dynamic rate limit adjustment
            logger.info(f"Adjusted rate limit for {ip} by factor {factor}")
    
    async def _action_alert_admins(self, params: Dict[str, Any]):
        """Alert system administrators."""
        priority = params.get("priority", "medium")
        incident_id = params.get("incident_id")
        
        # Get admin users
        admin_users = [user for user in auth_manager.users.values() if user.role == UserRole.ADMIN]
        
        alert_message = f"Security incident {incident_id} requires attention (Priority: {priority})"
        
        # In a real implementation, this would send notifications via email, Slack, etc.
        for admin in admin_users:
            logger.warning(f"ALERT for {admin.username}: {alert_message}")
        
        # Log the alert
        audit_logger.log_security_event(
            "admin_alert_sent", priority,
            {"incident_id": incident_id, "recipients": [admin.username for admin in admin_users]}
        )
    
    async def _action_isolate_system(self, params: Dict[str, Any]):
        """Isolate affected system."""
        # This would implement system isolation logic
        logger.critical(f"SYSTEM ISOLATION TRIGGERED: {params}")
        
        # In production, this might:
        # - Disconnect from network
        # - Stop non-essential services
        # - Enable emergency mode
    
    async def _action_backup_data(self, params: Dict[str, Any]):
        """Create emergency data backup."""
        # This would implement emergency backup logic
        logger.info(f"EMERGENCY BACKUP INITIATED: {params}")
        
        # In production, this might:
        # - Snapshot databases
        # - Archive critical files
        # - Store in secure location
    
    async def _action_patch_system(self, params: Dict[str, Any]):
        """Apply security patches."""
        # This would implement automated patching
        logger.info(f"AUTOMATED PATCHING INITIATED: {params}")
        
        # In production, this might:
        # - Download security updates
        # - Apply patches
        # - Restart services if needed
    
    async def _action_collect_evidence(self, params: Dict[str, Any]):
        """Collect forensic evidence."""
        forensics = params.get("forensics", False)
        
        logger.info(f"EVIDENCE COLLECTION INITIATED (forensics={forensics}): {params}")
        
        # In production, this might:
        # - Capture memory dumps
        # - Save log files
        # - Document system state
        # - Preserve network traffic
    
    async def _action_notify_external(self, params: Dict[str, Any]):
        """Notify external parties."""
        authorities = params.get("authorities", False)
        
        if authorities:
            logger.critical(f"EXTERNAL NOTIFICATION (AUTHORITIES) TRIGGERED: {params}")
        else:
            logger.warning(f"EXTERNAL NOTIFICATION TRIGGERED: {params}")
        
        # In production, this might:
        # - Send notifications to SOC
        # - Alert law enforcement if required
        # - Notify business partners
        # - Update threat intelligence providers
    
    def add_playbook_rule(self, rule: PlaybookRule):
        """Add custom playbook rule."""
        self.playbook_rules[rule.rule_id] = rule
        logger.info(f"Added playbook rule: {rule.name}")
    
    def remove_playbook_rule(self, rule_id: str) -> bool:
        """Remove playbook rule."""
        if rule_id in self.playbook_rules:
            del self.playbook_rules[rule_id]
            logger.info(f"Removed playbook rule: {rule_id}")
            return True
        return False
    
    def get_incident(self, incident_id: str) -> Optional[IncidentTicket]:
        """Get incident by ID."""
        return self.active_incidents.get(incident_id)
    
    def list_active_incidents(self) -> List[IncidentTicket]:
        """List all active incidents."""
        return list(self.active_incidents.values())
    
    def close_incident(self, incident_id: str, resolution_notes: str = ""):
        """Close an incident."""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            incident.status = IncidentStatus.CLOSED
            incident.updated_at = datetime.utcnow()
            incident.lessons_learned = resolution_notes
            
            # Move to history
            self.incident_history.append(incident)
            del self.active_incidents[incident_id]
            
            # Keep only last 1000 incidents in history
            if len(self.incident_history) > 1000:
                self.incident_history = self.incident_history[-1000:]
            
            logger.info(f"Closed incident {incident_id}")
            
            audit_logger.log_security_event(
                "incident_closed", "info",
                {"incident_id": incident_id, "resolution_notes": resolution_notes}
            )
    
    def get_incident_statistics(self) -> Dict[str, Any]:
        """Get incident response statistics."""
        active_incidents = list(self.active_incidents.values())
        
        stats = {
            "active_incidents": len(active_incidents),
            "total_incidents_handled": len(self.incident_history),
            "playbook_rules": len(self.playbook_rules),
            "enabled_rules": len([r for r in self.playbook_rules.values() if r.enabled]),
            "incidents_by_severity": {},
            "incidents_by_status": {},
            "average_response_time_minutes": 0,
            "most_triggered_rules": [],
            "recent_activity": []
        }
        
        # Count by severity and status
        for incident in active_incidents:
            severity = incident.severity.value
            status = incident.status.value
            
            stats["incidents_by_severity"][severity] = stats["incidents_by_severity"].get(severity, 0) + 1
            stats["incidents_by_status"][status] = stats["incidents_by_status"].get(status, 0) + 1
        
        # Calculate average response time from closed incidents
        closed_incidents = self.incident_history[-100:]  # Last 100 closed incidents
        if closed_incidents:
            response_times = []
            for incident in closed_incidents:
                if incident.response_actions:
                    first_response = min(action.executed_at for action in incident.response_actions if action.executed_at)
                    response_time = (first_response - incident.created_at).total_seconds() / 60
                    response_times.append(response_time)
            
            if response_times:
                stats["average_response_time_minutes"] = sum(response_times) / len(response_times)
        
        return stats


# Global instance
incident_response = IncidentResponseManager()