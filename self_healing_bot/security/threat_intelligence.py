"""Threat intelligence integration and analysis."""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

import aiohttp
import ipaddress

from ..core.config import config
from ..monitoring.logging import get_logger, audit_logger
from .monitoring import security_monitor, ThreatLevel, SecurityEventType

logger = get_logger(__name__)


class ThreatType(Enum):
    """Types of threats."""
    MALWARE = "malware"
    BOTNET = "botnet"
    C2_SERVER = "c2_server"
    PHISHING = "phishing"
    MALICIOUS_IP = "malicious_ip"
    MALICIOUS_DOMAIN = "malicious_domain"
    MALICIOUS_URL = "malicious_url"
    VULNERABILITY = "vulnerability"
    EXPLOIT = "exploit"
    APT_GROUP = "apt_group"


class IndicatorType(Enum):
    """Types of threat indicators."""
    IP_ADDRESS = "ip"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH = "hash"
    EMAIL = "email"
    USER_AGENT = "user_agent"
    CVE = "cve"
    YARA_RULE = "yara"


@dataclass
class ThreatIntelligence:
    """Threat intelligence data structure."""
    indicator_id: str
    indicator_type: IndicatorType
    indicator_value: str
    threat_types: Set[ThreatType]
    confidence: float  # 0.0 to 1.0
    severity: ThreatLevel
    first_seen: datetime
    last_seen: datetime
    source: str
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    false_positive: bool = False
    active: bool = True


@dataclass
class ThreatFeed:
    """Threat intelligence feed configuration."""
    feed_id: str
    name: str
    url: str
    feed_type: str  # json, csv, xml, stix
    api_key: Optional[str] = None
    update_interval_hours: int = 24
    last_update: Optional[datetime] = None
    enabled: bool = True
    parser: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreatIntelligenceManager:
    """Manages threat intelligence feeds and analysis."""
    
    def __init__(self):
        self.threat_indicators: Dict[str, ThreatIntelligence] = {}
        self.threat_feeds: Dict[str, ThreatFeed] = {}
        self.ip_reputation_cache: Dict[str, Dict[str, Any]] = {}
        self.domain_reputation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_hours = 24
        
        # Initialize default threat feeds
        self._initialize_default_feeds()
    
    def _initialize_default_feeds(self):
        """Initialize default threat intelligence feeds."""
        default_feeds = [
            ThreatFeed(
                feed_id="abuse_ch_malware",
                name="Abuse.ch Malware Bazaar",
                url="https://bazaar.abuse.ch/export/json/recent/",
                feed_type="json",
                update_interval_hours=6,
                parser="abuse_ch_malware"
            ),
            ThreatFeed(
                feed_id="abuse_ch_feodotracker",
                name="Abuse.ch Feodo Tracker",
                url="https://feodotracker.abuse.ch/downloads/ipblocklist_recommended.json",
                feed_type="json",
                update_interval_hours=6,
                parser="abuse_ch_feodo"
            ),
            ThreatFeed(
                feed_id="tor_exit_nodes",
                name="Tor Exit Nodes",
                url="https://www.dan.me.uk/torlist/?exit",
                feed_type="txt",
                update_interval_hours=12,
                parser="tor_exit_nodes"
            ),
            ThreatFeed(
                feed_id="misp_feed",
                name="MISP Threat Feed",
                url="https://www.circl.lu/doc/misp/feed-osint/manifest.json",
                feed_type="json",
                update_interval_hours=12,
                parser="misp_feed"
            )
        ]
        
        for feed in default_feeds:
            self.threat_feeds[feed.feed_id] = feed
    
    async def update_all_feeds(self):
        """Update all enabled threat intelligence feeds."""
        tasks = []
        
        for feed in self.threat_feeds.values():
            if feed.enabled and self._should_update_feed(feed):
                task = asyncio.create_task(self._update_feed(feed))
                tasks.append(task)
        
        if tasks:
            logger.info(f"Updating {len(tasks)} threat intelligence feeds")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for result in results if not isinstance(result, Exception))
            logger.info(f"Successfully updated {success_count}/{len(tasks)} feeds")
    
    def _should_update_feed(self, feed: ThreatFeed) -> bool:
        """Check if a feed should be updated."""
        if not feed.last_update:
            return True
        
        time_since_update = datetime.utcnow() - feed.last_update
        return time_since_update.total_seconds() > feed.update_interval_hours * 3600
    
    async def _update_feed(self, feed: ThreatFeed):
        """Update a single threat intelligence feed."""
        try:
            logger.info(f"Updating threat feed: {feed.name}")
            
            headers = {}
            if feed.api_key:
                headers['Authorization'] = f"Bearer {feed.api_key}"
            
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(feed.url, headers=headers) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
                    content = await response.text()
                    
                    # Parse feed based on type
                    indicators = await self._parse_feed_content(feed, content)
                    
                    # Add indicators to database
                    added_count = 0
                    updated_count = 0
                    
                    for indicator in indicators:
                        if await self._add_or_update_indicator(indicator):
                            added_count += 1
                        else:
                            updated_count += 1
                    
                    feed.last_update = datetime.utcnow()
                    
                    logger.info(f"Updated {feed.name}: {added_count} new, {updated_count} updated indicators")
                    
                    # Log security event
                    await security_monitor.log_security_event(
                        SecurityEventType.POLICY_VIOLATION,  # Using existing event type
                        ThreatLevel.LOW,
                        details={
                            "action": "threat_feed_updated",
                            "feed_name": feed.name,
                            "indicators_added": added_count,
                            "indicators_updated": updated_count
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Failed to update threat feed {feed.name}: {e}")
            audit_logger.log_security_event(
                "threat_feed_update_failed", "warning",
                {"feed_name": feed.name, "error": str(e)}
            )
    
    async def _parse_feed_content(self, feed: ThreatFeed, content: str) -> List[ThreatIntelligence]:
        """Parse feed content based on feed type and parser."""
        try:
            if feed.parser == "abuse_ch_malware":
                return await self._parse_abuse_ch_malware(content)
            elif feed.parser == "abuse_ch_feodo":
                return await self._parse_abuse_ch_feodo(content)
            elif feed.parser == "tor_exit_nodes":
                return await self._parse_tor_exit_nodes(content)
            elif feed.parser == "misp_feed":
                return await self._parse_misp_feed(content)
            else:
                return await self._parse_generic_feed(feed, content)
                
        except Exception as e:
            logger.error(f"Failed to parse feed {feed.name}: {e}")
            return []
    
    async def _parse_abuse_ch_malware(self, content: str) -> List[ThreatIntelligence]:
        """Parse Abuse.ch malware feed."""
        indicators = []
        
        try:
            data = json.loads(content)
            
            for item in data.get('data', []):
                if 'sha256_hash' in item:
                    indicator = ThreatIntelligence(
                        indicator_id=f"abuse_ch_{item['sha256_hash']}",
                        indicator_type=IndicatorType.FILE_HASH,
                        indicator_value=item['sha256_hash'],
                        threat_types={ThreatType.MALWARE},
                        confidence=0.9,
                        severity=ThreatLevel.HIGH,
                        first_seen=datetime.utcnow(),
                        last_seen=datetime.utcnow(),
                        source="Abuse.ch Malware Bazaar",
                        description=f"Malware: {item.get('signature', 'Unknown')}",
                        tags={item.get('signature', '').lower()},
                        metadata=item
                    )
                    indicators.append(indicator)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Abuse.ch malware feed: {e}")
        
        return indicators
    
    async def _parse_abuse_ch_feodo(self, content: str) -> List[ThreatIntelligence]:
        """Parse Abuse.ch Feodo Tracker feed."""
        indicators = []
        
        try:
            data = json.loads(content)
            
            for item in data:
                if 'ip_address' in item:
                    indicator = ThreatIntelligence(
                        indicator_id=f"feodo_{item['ip_address']}",
                        indicator_type=IndicatorType.IP_ADDRESS,
                        indicator_value=item['ip_address'],
                        threat_types={ThreatType.BOTNET, ThreatType.C2_SERVER},
                        confidence=0.95,
                        severity=ThreatLevel.HIGH,
                        first_seen=datetime.fromisoformat(item.get('first_seen', datetime.utcnow().isoformat())),
                        last_seen=datetime.fromisoformat(item.get('last_seen', datetime.utcnow().isoformat())),
                        source="Abuse.ch Feodo Tracker",
                        description=f"Feodo botnet C2 server on port {item.get('port', 'unknown')}",
                        tags={'feodo', 'botnet', 'c2'},
                        metadata=item
                    )
                    indicators.append(indicator)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Abuse.ch Feodo feed: {e}")
        
        return indicators
    
    async def _parse_tor_exit_nodes(self, content: str) -> List[ThreatIntelligence]:
        """Parse Tor exit nodes list."""
        indicators = []
        
        lines = content.strip().split('\n')
        for line in lines:
            ip = line.strip()
            if ip and self._is_valid_ip(ip):
                indicator = ThreatIntelligence(
                    indicator_id=f"tor_exit_{ip}",
                    indicator_type=IndicatorType.IP_ADDRESS,
                    indicator_value=ip,
                    threat_types={ThreatType.MALICIOUS_IP},
                    confidence=0.8,
                    severity=ThreatLevel.MEDIUM,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    source="Tor Exit Nodes List",
                    description="Tor exit node IP address",
                    tags={'tor', 'exit_node'},
                    metadata={'tor_exit_node': True}
                )
                indicators.append(indicator)
        
        return indicators
    
    async def _parse_misp_feed(self, content: str) -> List[ThreatIntelligence]:
        """Parse MISP threat feed."""
        indicators = []
        
        try:
            data = json.loads(content)
            
            # This is a simplified MISP parser - real implementation would be more complex
            for event in data.get('events', []):
                for attribute in event.get('Attribute', []):
                    indicator_type_map = {
                        'ip-dst': IndicatorType.IP_ADDRESS,
                        'ip-src': IndicatorType.IP_ADDRESS,
                        'domain': IndicatorType.DOMAIN,
                        'hostname': IndicatorType.DOMAIN,
                        'url': IndicatorType.URL,
                        'md5': IndicatorType.FILE_HASH,
                        'sha1': IndicatorType.FILE_HASH,
                        'sha256': IndicatorType.FILE_HASH,
                    }
                    
                    attr_type = attribute.get('type')
                    if attr_type in indicator_type_map:
                        indicator = ThreatIntelligence(
                            indicator_id=f"misp_{attribute.get('uuid', hashlib.md5(attribute.get('value', '').encode()).hexdigest())}",
                            indicator_type=indicator_type_map[attr_type],
                            indicator_value=attribute.get('value'),
                            threat_types={ThreatType.MALWARE},  # Simplified
                            confidence=0.8,
                            severity=self._map_misp_threat_level(event.get('threat_level_id', '3')),
                            first_seen=datetime.utcnow(),
                            last_seen=datetime.utcnow(),
                            source="MISP Feed",
                            description=attribute.get('comment', ''),
                            tags=set(tag.get('name', '') for tag in attribute.get('Tag', [])),
                            metadata={'misp_event_id': event.get('id'), 'attribute': attribute}
                        )
                        indicators.append(indicator)
                        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MISP feed: {e}")
        
        return indicators
    
    def _map_misp_threat_level(self, threat_level_id: str) -> ThreatLevel:
        """Map MISP threat level to our threat levels."""
        mapping = {
            '1': ThreatLevel.HIGH,
            '2': ThreatLevel.MEDIUM,
            '3': ThreatLevel.LOW,
            '4': ThreatLevel.LOW
        }
        return mapping.get(threat_level_id, ThreatLevel.MEDIUM)
    
    async def _parse_generic_feed(self, feed: ThreatFeed, content: str) -> List[ThreatIntelligence]:
        """Parse generic feed content."""
        indicators = []
        
        if feed.feed_type == "json":
            try:
                data = json.loads(content)
                # Basic JSON parsing - would need customization per feed
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'indicator' in item:
                            # Basic indicator extraction
                            pass
            except json.JSONDecodeError:
                pass
        
        return indicators
    
    def _is_valid_ip(self, ip_str: str) -> bool:
        """Check if string is a valid IP address."""
        try:
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False
    
    async def _add_or_update_indicator(self, indicator: ThreatIntelligence) -> bool:
        """Add or update threat indicator. Returns True if new, False if updated."""
        existing_key = f"{indicator.indicator_type.value}:{indicator.indicator_value}"
        
        if existing_key in self.threat_indicators:
            # Update existing indicator
            existing = self.threat_indicators[existing_key]
            existing.last_seen = indicator.last_seen
            existing.confidence = max(existing.confidence, indicator.confidence)
            existing.threat_types.update(indicator.threat_types)
            existing.tags.update(indicator.tags)
            existing.references.extend(r for r in indicator.references if r not in existing.references)
            return False
        else:
            # Add new indicator
            self.threat_indicators[existing_key] = indicator
            
            # Add to security monitor
            await security_monitor.add_threat_indicator(
                indicator.indicator_type.value,
                indicator.indicator_value,
                indicator.severity,
                indicator.source,
                indicator.confidence,
                list(indicator.tags)
            )
            
            return True
    
    async def check_ip_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Check IP address reputation."""
        # Check cache first
        if ip_address in self.ip_reputation_cache:
            cache_entry = self.ip_reputation_cache[ip_address]
            cache_time = cache_entry.get('timestamp', datetime.min)
            if datetime.utcnow() - cache_time < timedelta(hours=self.cache_ttl_hours):
                return cache_entry['data']
        
        reputation = {
            "ip": ip_address,
            "is_malicious": False,
            "threat_types": [],
            "confidence": 0.0,
            "sources": [],
            "last_seen": None,
            "tags": []
        }
        
        # Check against threat indicators
        indicator_key = f"{IndicatorType.IP_ADDRESS.value}:{ip_address}"
        if indicator_key in self.threat_indicators:
            indicator = self.threat_indicators[indicator_key]
            if indicator.active and not indicator.false_positive:
                reputation.update({
                    "is_malicious": True,
                    "threat_types": [t.value for t in indicator.threat_types],
                    "confidence": indicator.confidence,
                    "sources": [indicator.source],
                    "last_seen": indicator.last_seen.isoformat(),
                    "tags": list(indicator.tags),
                    "description": indicator.description
                })
        
        # Check external reputation services (if configured)
        external_checks = await self._check_external_ip_reputation(ip_address)
        if external_checks:
            if external_checks.get('is_malicious'):
                reputation['is_malicious'] = True
                reputation['sources'].extend(external_checks.get('sources', []))
                reputation['confidence'] = max(reputation['confidence'], external_checks.get('confidence', 0.0))
        
        # Cache result
        self.ip_reputation_cache[ip_address] = {
            'data': reputation,
            'timestamp': datetime.utcnow()
        }
        
        return reputation
    
    async def _check_external_ip_reputation(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """Check external IP reputation services."""
        # This would integrate with services like VirusTotal, AbuseIPDB, etc.
        # For now, we'll return None (no external check)
        return None
    
    async def check_domain_reputation(self, domain: str) -> Dict[str, Any]:
        """Check domain reputation."""
        # Check cache first
        if domain in self.domain_reputation_cache:
            cache_entry = self.domain_reputation_cache[domain]
            cache_time = cache_entry.get('timestamp', datetime.min)
            if datetime.utcnow() - cache_time < timedelta(hours=self.cache_ttl_hours):
                return cache_entry['data']
        
        reputation = {
            "domain": domain,
            "is_malicious": False,
            "threat_types": [],
            "confidence": 0.0,
            "sources": [],
            "last_seen": None,
            "tags": []
        }
        
        # Check against threat indicators
        indicator_key = f"{IndicatorType.DOMAIN.value}:{domain}"
        if indicator_key in self.threat_indicators:
            indicator = self.threat_indicators[indicator_key]
            if indicator.active and not indicator.false_positive:
                reputation.update({
                    "is_malicious": True,
                    "threat_types": [t.value for t in indicator.threat_types],
                    "confidence": indicator.confidence,
                    "sources": [indicator.source],
                    "last_seen": indicator.last_seen.isoformat(),
                    "tags": list(indicator.tags),
                    "description": indicator.description
                })
        
        # Cache result
        self.domain_reputation_cache[domain] = {
            'data': reputation,
            'timestamp': datetime.utcnow()
        }
        
        return reputation
    
    async def check_file_hash_reputation(self, file_hash: str) -> Dict[str, Any]:
        """Check file hash reputation."""
        reputation = {
            "hash": file_hash,
            "is_malicious": False,
            "threat_types": [],
            "confidence": 0.0,
            "sources": [],
            "last_seen": None,
            "tags": []
        }
        
        # Check against threat indicators
        indicator_key = f"{IndicatorType.FILE_HASH.value}:{file_hash}"
        if indicator_key in self.threat_indicators:
            indicator = self.threat_indicators[indicator_key]
            if indicator.active and not indicator.false_positive:
                reputation.update({
                    "is_malicious": True,
                    "threat_types": [t.value for t in indicator.threat_types],
                    "confidence": indicator.confidence,
                    "sources": [indicator.source],
                    "last_seen": indicator.last_seen.isoformat(),
                    "tags": list(indicator.tags),
                    "description": indicator.description
                })
        
        return reputation
    
    def add_custom_indicator(self, indicator_type: IndicatorType, value: str,
                           threat_types: Set[ThreatType], confidence: float,
                           description: str = "", tags: Set[str] = None) -> str:
        """Add custom threat indicator."""
        indicator_id = f"custom_{hashlib.md5(f'{indicator_type.value}_{value}'.encode()).hexdigest()[:16]}"
        
        indicator = ThreatIntelligence(
            indicator_id=indicator_id,
            indicator_type=indicator_type,
            indicator_value=value,
            threat_types=threat_types,
            confidence=confidence,
            severity=ThreatLevel.MEDIUM,  # Default
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
            source="Custom",
            description=description,
            tags=tags or set()
        )
        
        indicator_key = f"{indicator_type.value}:{value}"
        self.threat_indicators[indicator_key] = indicator
        
        logger.info(f"Added custom threat indicator: {indicator_type.value}={value}")
        
        return indicator_id
    
    def remove_indicator(self, indicator_type: IndicatorType, value: str) -> bool:
        """Remove threat indicator."""
        indicator_key = f"{indicator_type.value}:{value}"
        
        if indicator_key in self.threat_indicators:
            del self.threat_indicators[indicator_key]
            logger.info(f"Removed threat indicator: {indicator_type.value}={value}")
            return True
        
        return False
    
    def mark_false_positive(self, indicator_type: IndicatorType, value: str) -> bool:
        """Mark indicator as false positive."""
        indicator_key = f"{indicator_type.value}:{value}"
        
        if indicator_key in self.threat_indicators:
            self.threat_indicators[indicator_key].false_positive = True
            logger.info(f"Marked false positive: {indicator_type.value}={value}")
            return True
        
        return False
    
    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get threat intelligence statistics."""
        active_indicators = [i for i in self.threat_indicators.values() if i.active and not i.false_positive]
        
        stats = {
            "total_indicators": len(self.threat_indicators),
            "active_indicators": len(active_indicators),
            "false_positives": len([i for i in self.threat_indicators.values() if i.false_positive]),
            "by_type": {},
            "by_threat_type": {},
            "by_severity": {},
            "feeds_configured": len(self.threat_feeds),
            "feeds_enabled": len([f for f in self.threat_feeds.values() if f.enabled]),
            "last_feed_update": None
        }
        
        # Count by indicator type
        for indicator in active_indicators:
            indicator_type = indicator.indicator_type.value
            stats["by_type"][indicator_type] = stats["by_type"].get(indicator_type, 0) + 1
            
            # Count by threat type
            for threat_type in indicator.threat_types:
                threat_type_name = threat_type.value
                stats["by_threat_type"][threat_type_name] = stats["by_threat_type"].get(threat_type_name, 0) + 1
            
            # Count by severity
            severity = indicator.severity.value
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
        
        # Find most recent feed update
        feed_updates = [f.last_update for f in self.threat_feeds.values() if f.last_update]
        if feed_updates:
            stats["last_feed_update"] = max(feed_updates).isoformat()
        
        return stats
    
    async def cleanup_old_indicators(self, max_age_days: int = 90):
        """Clean up old threat indicators."""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        
        old_indicators = [
            key for key, indicator in self.threat_indicators.items()
            if indicator.last_seen < cutoff_date
        ]
        
        for key in old_indicators:
            del self.threat_indicators[key]
        
        if old_indicators:
            logger.info(f"Cleaned up {len(old_indicators)} old threat indicators")
        
        # Also cleanup caches
        current_time = datetime.utcnow()
        
        # Clean IP reputation cache
        old_ip_entries = [
            ip for ip, data in self.ip_reputation_cache.items()
            if current_time - data['timestamp'] > timedelta(hours=self.cache_ttl_hours * 2)
        ]
        
        for ip in old_ip_entries:
            del self.ip_reputation_cache[ip]
        
        # Clean domain reputation cache
        old_domain_entries = [
            domain for domain, data in self.domain_reputation_cache.items()
            if current_time - data['timestamp'] > timedelta(hours=self.cache_ttl_hours * 2)
        ]
        
        for domain in old_domain_entries:
            del self.domain_reputation_cache[domain]
        
        if old_ip_entries or old_domain_entries:
            logger.info(f"Cleaned up {len(old_ip_entries)} IP and {len(old_domain_entries)} domain cache entries")


# Global instance
threat_intelligence = ThreatIntelligenceManager()