"""Security scanning integration and vulnerability assessment."""

import asyncio
import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..core.config import config
from ..monitoring.logging import get_logger, audit_logger
from .monitoring import security_monitor, SecurityEventType, ThreatLevel
from .secrets import secret_scanner

logger = get_logger(__name__)


class VulnerabilityType(Enum):
    """Types of vulnerabilities."""
    DEPENDENCY = "dependency"
    CODE = "code"
    CONFIGURATION = "configuration"
    INFRASTRUCTURE = "infrastructure"
    CONTAINER = "container"
    SECRET = "secret"
    PERMISSION = "permission"


class SeverityLevel(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Vulnerability:
    """Vulnerability information."""
    vulnerability_id: str
    title: str
    description: str
    severity: SeverityLevel
    vulnerability_type: VulnerabilityType
    cve_id: Optional[str] = None
    affected_component: Optional[str] = None
    affected_version: Optional[str] = None
    fixed_version: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    remediation: Optional[str] = None
    references: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    false_positive: bool = False
    risk_score: float = 0.0
    exploitability: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Security scan result."""
    scan_id: str
    scan_type: str
    target: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class DependencyScanner:
    """Scan dependencies for known vulnerabilities."""
    
    def __init__(self):
        self.vulnerability_databases = {
            "python": self._scan_python_dependencies,
            "node": self._scan_node_dependencies,
            "docker": self._scan_docker_dependencies
        }
    
    async def scan_dependencies(self, project_path: Path, scan_type: str = "python") -> List[Vulnerability]:
        """Scan project dependencies for vulnerabilities."""
        try:
            if scan_type in self.vulnerability_databases:
                scanner = self.vulnerability_databases[scan_type]
                return await scanner(project_path)
            else:
                logger.warning(f"Unsupported dependency scan type: {scan_type}")
                return []
                
        except Exception as e:
            logger.error(f"Dependency scan failed: {e}")
            return []
    
    async def _scan_python_dependencies(self, project_path: Path) -> List[Vulnerability]:
        """Scan Python dependencies using safety."""
        vulnerabilities = []
        
        try:
            # Look for requirements.txt or pyproject.toml
            requirements_files = [
                project_path / "requirements.txt",
                project_path / "pyproject.toml",
                project_path / "Pipfile",
                project_path / "setup.py"
            ]
            
            found_file = None
            for req_file in requirements_files:
                if req_file.exists():
                    found_file = req_file
                    break
            
            if not found_file:
                logger.info("No Python dependency file found")
                return vulnerabilities
            
            # Try to use safety if available
            try:
                result = subprocess.run(
                    ["safety", "check", "--json", "--file", str(found_file)],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    # No vulnerabilities found
                    return vulnerabilities
                
                # Parse safety output
                try:
                    safety_data = json.loads(result.stdout)
                    for vuln_data in safety_data:
                        vulnerability = Vulnerability(
                            vulnerability_id=f"safety_{vuln_data.get('id', 'unknown')}",
                            title=vuln_data.get('advisory', 'Unknown vulnerability'),
                            description=vuln_data.get('advisory', ''),
                            severity=self._map_safety_severity(vuln_data.get('severity', 'medium')),
                            vulnerability_type=VulnerabilityType.DEPENDENCY,
                            affected_component=vuln_data.get('package_name'),
                            affected_version=vuln_data.get('analyzed_version'),
                            fixed_version=">=".join(vuln_data.get('vulnerable_versions', [])),
                            remediation=f"Update {vuln_data.get('package_name')} to a safe version",
                            metadata={"safety_data": vuln_data}
                        )
                        vulnerabilities.append(vulnerability)
                        
                except json.JSONDecodeError:
                    # Fallback to text parsing
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if "vulnerability" in line.lower():
                            # Basic vulnerability detection
                            vulnerability = Vulnerability(
                                vulnerability_id=hashlib.md5(line.encode()).hexdigest()[:16],
                                title="Dependency vulnerability detected",
                                description=line.strip(),
                                severity=SeverityLevel.MEDIUM,
                                vulnerability_type=VulnerabilityType.DEPENDENCY
                            )
                            vulnerabilities.append(vulnerability)
                            
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Fallback to manual parsing
                vulnerabilities.extend(await self._manual_dependency_check(found_file))
                
        except Exception as e:
            logger.error(f"Python dependency scan error: {e}")
        
        return vulnerabilities
    
    def _map_safety_severity(self, safety_severity: str) -> SeverityLevel:
        """Map safety severity to our severity levels."""
        mapping = {
            "high": SeverityLevel.HIGH,
            "medium": SeverityLevel.MEDIUM,
            "low": SeverityLevel.LOW
        }
        return mapping.get(safety_severity.lower(), SeverityLevel.MEDIUM)
    
    async def _manual_dependency_check(self, requirements_file: Path) -> List[Vulnerability]:
        """Manual dependency vulnerability check using known patterns."""
        vulnerabilities = []
        
        try:
            content = requirements_file.read_text()
            
            # Known vulnerable patterns
            vulnerable_patterns = [
                (r"django\\s*[<>=]*\\s*[12]\\.", "Django < 3.0 has known vulnerabilities", SeverityLevel.HIGH),
                (r"flask\\s*[<>=]*\\s*0\\.", "Flask < 1.0 has known vulnerabilities", SeverityLevel.MEDIUM),
                (r"requests\\s*[<>=]*\\s*2\\.1[0-9]\\.", "Requests < 2.20 has known vulnerabilities", SeverityLevel.MEDIUM),
                (r"pillow\\s*[<>=]*\\s*[5-7]\\.", "Pillow < 8.0 may have vulnerabilities", SeverityLevel.LOW),
                (r"pyyaml\\s*[<>=]*\\s*[3-5]\\.", "PyYAML < 6.0 has vulnerabilities", SeverityLevel.MEDIUM)
            ]
            
            for pattern, description, severity in vulnerable_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    vulnerability = Vulnerability(
                        vulnerability_id=hashlib.md5(f"{requirements_file}_{match.group()}".encode()).hexdigest()[:16],
                        title="Potentially vulnerable dependency",
                        description=description,
                        severity=severity,
                        vulnerability_type=VulnerabilityType.DEPENDENCY,
                        affected_component=match.group().split()[0],
                        file_path=str(requirements_file),
                        remediation="Update to latest version"
                    )
                    vulnerabilities.append(vulnerability)
                    
        except Exception as e:
            logger.error(f"Manual dependency check error: {e}")
        
        return vulnerabilities
    
    async def _scan_node_dependencies(self, project_path: Path) -> List[Vulnerability]:
        """Scan Node.js dependencies."""
        vulnerabilities = []
        
        package_json = project_path / "package.json"
        if not package_json.exists():
            return vulnerabilities
        
        try:
            # Try npm audit
            result = subprocess.run(
                ["npm", "audit", "--json"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                try:
                    audit_data = json.loads(result.stdout)
                    for vuln_id, vuln_info in audit_data.get("vulnerabilities", {}).items():
                        vulnerability = Vulnerability(
                            vulnerability_id=f"npm_{vuln_id}",
                            title=vuln_info.get("title", "Node.js vulnerability"),
                            description=vuln_info.get("overview", ""),
                            severity=self._map_npm_severity(vuln_info.get("severity", "moderate")),
                            vulnerability_type=VulnerabilityType.DEPENDENCY,
                            affected_component=vuln_info.get("module_name"),
                            cve_id=vuln_info.get("cves", [])[0] if vuln_info.get("cves") else None,
                            remediation=vuln_info.get("recommendation", "Update dependency"),
                            references=vuln_info.get("references", [])
                        )
                        vulnerabilities.append(vulnerability)
                        
                except json.JSONDecodeError:
                    pass
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("npm not available for dependency scanning")
        
        return vulnerabilities
    
    def _map_npm_severity(self, npm_severity: str) -> SeverityLevel:
        """Map npm severity to our severity levels."""
        mapping = {
            "critical": SeverityLevel.CRITICAL,
            "high": SeverityLevel.HIGH,
            "moderate": SeverityLevel.MEDIUM,
            "low": SeverityLevel.LOW,
            "info": SeverityLevel.INFO
        }
        return mapping.get(npm_severity.lower(), SeverityLevel.MEDIUM)
    
    async def _scan_docker_dependencies(self, project_path: Path) -> List[Vulnerability]:
        """Scan Docker images for vulnerabilities."""
        vulnerabilities = []
        
        dockerfile = project_path / "Dockerfile"
        if not dockerfile.exists():
            return vulnerabilities
        
        try:
            content = dockerfile.read_text()
            
            # Check for vulnerable base images
            base_image_patterns = [
                (r"FROM\\s+ubuntu:1[4-8]\\.", "Ubuntu 14-18 have known vulnerabilities", SeverityLevel.HIGH),
                (r"FROM\\s+debian:[7-9]\\.", "Debian 7-9 have known vulnerabilities", SeverityLevel.MEDIUM),
                (r"FROM\\s+centos:[5-7]\\.", "CentOS 5-7 have known vulnerabilities", SeverityLevel.MEDIUM),
                (r"FROM\\s+alpine:[2-3]\\.[0-9]", "Alpine < 3.10 may have vulnerabilities", SeverityLevel.LOW)
            ]
            
            for pattern, description, severity in base_image_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    vulnerability = Vulnerability(
                        vulnerability_id=hashlib.md5(f"docker_{match.group()}".encode()).hexdigest()[:16],
                        title="Vulnerable base image",
                        description=description,
                        severity=severity,
                        vulnerability_type=VulnerabilityType.CONTAINER,
                        affected_component=match.group().split()[1],
                        file_path=str(dockerfile),
                        remediation="Update to latest base image"
                    )
                    vulnerabilities.append(vulnerability)
                    
        except Exception as e:
            logger.error(f"Docker dependency scan error: {e}")
        
        return vulnerabilities


class CodeScanner:
    """Scan source code for security vulnerabilities."""
    
    def __init__(self):
        self.security_patterns = [
            # SQL Injection patterns
            (r"execute\\s*\\(.*%.*\\)", "Potential SQL injection", SeverityLevel.HIGH, VulnerabilityType.CODE),
            (r"cursor\\.execute\\s*\\([^?]*\\+", "SQL injection via string concatenation", SeverityLevel.HIGH, VulnerabilityType.CODE),
            
            # Command injection patterns
            (r"os\\.system\\s*\\(", "Command injection risk", SeverityLevel.HIGH, VulnerabilityType.CODE),
            (r"subprocess\\.call\\s*\\(", "Potential command injection", SeverityLevel.MEDIUM, VulnerabilityType.CODE),
            
            # Crypto issues
            (r"md5\\(\\)", "MD5 is cryptographically broken", SeverityLevel.MEDIUM, VulnerabilityType.CODE),
            (r"sha1\\(\\)", "SHA1 is weak for cryptographic use", SeverityLevel.MEDIUM, VulnerabilityType.CODE),
            
            # Hardcoded secrets
            (r"password\\s*=\\s*['\"][^'\"]{8,}['\"]", "Hardcoded password", SeverityLevel.HIGH, VulnerabilityType.SECRET),
            (r"api_key\\s*=\\s*['\"][^'\"]{20,}['\"]", "Hardcoded API key", SeverityLevel.HIGH, VulnerabilityType.SECRET),
            
            # File operations
            (r"open\\s*\\(.*input\\s*\\(", "Path traversal risk", SeverityLevel.MEDIUM, VulnerabilityType.CODE),
            
            # Deserialization
            (r"pickle\\.loads\\s*\\(", "Unsafe deserialization", SeverityLevel.HIGH, VulnerabilityType.CODE),
            (r"yaml\\.load\\s*\\([^,]*\\)", "Unsafe YAML loading", SeverityLevel.HIGH, VulnerabilityType.CODE),
            
            # Debug/Development patterns
            (r"app\\.run\\s*\\(.*debug\\s*=\\s*True", "Debug mode enabled in production", SeverityLevel.MEDIUM, VulnerabilityType.CONFIGURATION)
        ]
    
    async def scan_directory(self, directory: Path, extensions: Set[str] = None) -> List[Vulnerability]:
        """Scan directory for code vulnerabilities."""
        if extensions is None:
            extensions = {".py", ".js", ".ts", ".java", ".php", ".rb", ".go"}
        
        vulnerabilities = []
        
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file() and file_path.suffix in extensions:
                    file_vulns = await self._scan_file(file_path)
                    vulnerabilities.extend(file_vulns)
                    
        except Exception as e:
            logger.error(f"Directory scan error: {e}")
        
        return vulnerabilities
    
    async def _scan_file(self, file_path: Path) -> List[Vulnerability]:
        """Scan a single file for vulnerabilities."""
        vulnerabilities = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern, description, severity, vuln_type in self.security_patterns:
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        vulnerability = Vulnerability(
                            vulnerability_id=hashlib.md5(f"{file_path}_{line_num}_{match.group()}".encode()).hexdigest()[:16],
                            title=description,
                            description=f"{description}: {match.group()}",
                            severity=severity,
                            vulnerability_type=vuln_type,
                            file_path=str(file_path),
                            line_number=line_num,
                            remediation=self._get_remediation_advice(pattern, description),
                            metadata={"pattern": pattern, "match": match.group()}
                        )
                        vulnerabilities.append(vulnerability)
                        
        except Exception as e:
            logger.error(f"File scan error for {file_path}: {e}")
        
        return vulnerabilities
    
    def _get_remediation_advice(self, pattern: str, description: str) -> str:
        """Get remediation advice for a vulnerability pattern."""
        remediation_map = {
            "SQL injection": "Use parameterized queries or ORM methods",
            "Command injection": "Validate input and use subprocess with shell=False",
            "MD5": "Use SHA-256 or stronger hash functions",
            "SHA1": "Use SHA-256 or stronger hash functions",
            "Hardcoded": "Use environment variables or secure credential storage",
            "Path traversal": "Validate file paths and use os.path.join()",
            "Unsafe deserialization": "Use safe serialization formats like JSON",
            "Debug mode": "Set debug=False in production"
        }
        
        for key, advice in remediation_map.items():
            if key.lower() in description.lower():
                return advice
        
        return "Review code for security best practices"


class ConfigurationScanner:
    """Scan configuration files for security issues."""
    
    def __init__(self):
        self.config_patterns = [
            # Database configurations
            (r"password\\s*[=:]\\s*['\"]?\\w+['\"]?", "Database password in config", SeverityLevel.HIGH),
            (r"host\\s*[=:]\\s*['\"]?0\\.0\\.0\\.0['\"]?", "Binding to all interfaces", SeverityLevel.MEDIUM),
            
            # SSL/TLS configurations
            (r"ssl_verify\\s*[=:]\\s*false", "SSL verification disabled", SeverityLevel.HIGH),
            (r"verify_ssl\\s*[=:]\\s*false", "SSL verification disabled", SeverityLevel.HIGH),
            
            # Debug configurations
            (r"debug\\s*[=:]\\s*true", "Debug mode enabled", SeverityLevel.MEDIUM),
            (r"log_level\\s*[=:]\\s*debug", "Debug logging enabled", SeverityLevel.LOW),
            
            # Default credentials
            (r"username\\s*[=:]\\s*['\"]?admin['\"]?", "Default admin username", SeverityLevel.MEDIUM),
            (r"password\\s*[=:]\\s*['\"]?(admin|password|123456)['\"]?", "Default password", SeverityLevel.HIGH)
        ]
    
    async def scan_configs(self, directory: Path) -> List[Vulnerability]:
        """Scan configuration files for security issues."""
        config_files = [
            "*.ini", "*.conf", "*.config", "*.yaml", "*.yml",
            "*.json", "*.toml", "*.env", ".env*", "docker-compose.yml"
        ]
        
        vulnerabilities = []
        
        for pattern in config_files:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    file_vulns = await self._scan_config_file(file_path)
                    vulnerabilities.extend(file_vulns)
        
        return vulnerabilities
    
    async def _scan_config_file(self, file_path: Path) -> List[Vulnerability]:
        """Scan a configuration file."""
        vulnerabilities = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern, description, severity in self.config_patterns:
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        vulnerability = Vulnerability(
                            vulnerability_id=hashlib.md5(f"{file_path}_{line_num}_{match.group()}".encode()).hexdigest()[:16],
                            title=f"Configuration issue: {description}",
                            description=f"{description} in {file_path.name}",
                            severity=severity,
                            vulnerability_type=VulnerabilityType.CONFIGURATION,
                            file_path=str(file_path),
                            line_number=line_num,
                            remediation="Review and secure configuration",
                            metadata={"pattern": pattern, "match": match.group()}
                        )
                        vulnerabilities.append(vulnerability)
                        
        except Exception as e:
            logger.error(f"Config file scan error for {file_path}: {e}")
        
        return vulnerabilities


class SecurityScanner:
    """Comprehensive security scanner."""
    
    def __init__(self):
        self.dependency_scanner = DependencyScanner()
        self.code_scanner = CodeScanner()
        self.config_scanner = ConfigurationScanner()
        self.secret_scanner = secret_scanner
        self.scan_history: List[ScanResult] = []
    
    async def run_comprehensive_scan(self, target_path: Path, scan_types: Optional[List[str]] = None) -> ScanResult:
        """Run comprehensive security scan."""
        if scan_types is None:
            scan_types = ["dependencies", "code", "config", "secrets"]
        
        scan_id = f"scan_{int(time.time())}_{hashlib.md5(str(target_path).encode()).hexdigest()[:8]}"
        
        scan_result = ScanResult(
            scan_id=scan_id,
            scan_type="comprehensive",
            target=str(target_path),
            started_at=datetime.utcnow(),
            status="running"
        )
        
        try:
            all_vulnerabilities = []
            
            # Run dependency scan
            if "dependencies" in scan_types:
                logger.info(f"Running dependency scan on {target_path}")
                dep_vulns = await self.dependency_scanner.scan_dependencies(target_path)
                all_vulnerabilities.extend(dep_vulns)
                scan_result.metadata["dependencies_scanned"] = True
            
            # Run code scan
            if "code" in scan_types:
                logger.info(f"Running code scan on {target_path}")
                code_vulns = await self.code_scanner.scan_directory(target_path)
                all_vulnerabilities.extend(code_vulns)
                scan_result.metadata["code_scanned"] = True
            
            # Run configuration scan
            if "config" in scan_types:
                logger.info(f"Running configuration scan on {target_path}")
                config_vulns = await self.config_scanner.scan_configs(target_path)
                all_vulnerabilities.extend(config_vulns)
                scan_result.metadata["config_scanned"] = True
            
            # Run secrets scan
            if "secrets" in scan_types:
                logger.info(f"Running secrets scan on {target_path}")
                secret_vulns = await self._scan_for_secrets(target_path)
                all_vulnerabilities.extend(secret_vulns)
                scan_result.metadata["secrets_scanned"] = True
            
            # Process and deduplicate vulnerabilities
            unique_vulnerabilities = self._deduplicate_vulnerabilities(all_vulnerabilities)
            scan_result.vulnerabilities = unique_vulnerabilities
            
            # Calculate risk scores
            for vuln in scan_result.vulnerabilities:
                vuln.risk_score = self._calculate_risk_score(vuln)
            
            # Sort by risk score
            scan_result.vulnerabilities.sort(key=lambda v: v.risk_score, reverse=True)
            
            # Generate summary
            scan_result.summary = self._generate_summary(scan_result.vulnerabilities)
            
            scan_result.status = "completed"
            scan_result.completed_at = datetime.utcnow()
            
            # Log security events for high/critical vulnerabilities
            await self._log_high_severity_vulnerabilities(scan_result)
            
        except Exception as e:
            logger.error(f"Comprehensive scan failed: {e}")
            scan_result.status = "failed"
            scan_result.error_message = str(e)
            scan_result.completed_at = datetime.utcnow()
        
        # Store scan result
        self.scan_history.append(scan_result)
        
        # Keep only last 100 scan results
        if len(self.scan_history) > 100:
            self.scan_history = self.scan_history[-100:]
        
        return scan_result
    
    async def _scan_for_secrets(self, target_path: Path) -> List[Vulnerability]:
        """Scan for hardcoded secrets."""
        vulnerabilities = []
        
        try:
            for file_path in target_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in {".py", ".js", ".java", ".env", ".ini", ".conf"}:
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        findings = self.secret_scanner.scan_text(content)
                        
                        for finding in findings:
                            vulnerability = Vulnerability(
                                vulnerability_id=hashlib.md5(f"{file_path}_{finding['start']}_{finding['type']}".encode()).hexdigest()[:16],
                                title=f"Secret detected: {finding['type']}",
                                description=f"{finding['type']} found in {file_path.name}",
                                severity=SeverityLevel.HIGH,
                                vulnerability_type=VulnerabilityType.SECRET,
                                file_path=str(file_path),
                                line_number=finding['line'],
                                remediation="Move secret to environment variables or secure storage",
                                metadata={"secret_type": finding['type'], "finding": finding}
                            )
                            vulnerabilities.append(vulnerability)
                            
                    except Exception as e:
                        logger.debug(f"Error scanning {file_path} for secrets: {e}")
                        
        except Exception as e:
            logger.error(f"Secrets scan error: {e}")
        
        return vulnerabilities
    
    def _deduplicate_vulnerabilities(self, vulnerabilities: List[Vulnerability]) -> List[Vulnerability]:
        """Remove duplicate vulnerabilities."""
        seen = set()
        unique_vulns = []
        
        for vuln in vulnerabilities:
            # Create a hash based on key attributes
            hash_key = f"{vuln.title}_{vuln.file_path}_{vuln.line_number}_{vuln.affected_component}"
            hash_digest = hashlib.md5(hash_key.encode()).hexdigest()
            
            if hash_digest not in seen:
                seen.add(hash_digest)
                unique_vulns.append(vuln)
        
        return unique_vulns
    
    def _calculate_risk_score(self, vulnerability: Vulnerability) -> float:
        """Calculate risk score for vulnerability."""
        # Base score from severity
        severity_scores = {
            SeverityLevel.CRITICAL: 10.0,
            SeverityLevel.HIGH: 7.5,
            SeverityLevel.MEDIUM: 5.0,
            SeverityLevel.LOW: 2.5,
            SeverityLevel.INFO: 1.0
        }
        
        base_score = severity_scores.get(vulnerability.severity, 5.0)
        
        # Adjust based on vulnerability type
        type_multipliers = {
            VulnerabilityType.SECRET: 1.5,
            VulnerabilityType.CODE: 1.3,
            VulnerabilityType.DEPENDENCY: 1.2,
            VulnerabilityType.CONFIGURATION: 1.1,
            VulnerabilityType.CONTAINER: 1.0,
            VulnerabilityType.INFRASTRUCTURE: 1.0,
            VulnerabilityType.PERMISSION: 0.9
        }
        
        type_multiplier = type_multipliers.get(vulnerability.vulnerability_type, 1.0)
        
        # Adjust based on exploitability (if available)
        exploitability_bonus = vulnerability.exploitability * 2.0
        
        risk_score = (base_score * type_multiplier) + exploitability_bonus
        return min(10.0, risk_score)  # Cap at 10.0
    
    def _generate_summary(self, vulnerabilities: List[Vulnerability]) -> Dict[str, int]:
        """Generate vulnerability summary."""
        summary = {
            "total": len(vulnerabilities),
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        }
        
        for vuln in vulnerabilities:
            summary[vuln.severity.value] += 1
        
        return summary
    
    async def _log_high_severity_vulnerabilities(self, scan_result: ScanResult):
        """Log high severity vulnerabilities as security events."""
        high_severity_vulns = [
            v for v in scan_result.vulnerabilities 
            if v.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
        ]
        
        for vuln in high_severity_vulns:
            threat_level = ThreatLevel.CRITICAL if vuln.severity == SeverityLevel.CRITICAL else ThreatLevel.HIGH
            
            await security_monitor.log_security_event(
                SecurityEventType.VULNERABILITY_EXPLOIT,
                threat_level,
                details={
                    "scan_id": scan_result.scan_id,
                    "vulnerability_id": vuln.vulnerability_id,
                    "title": vuln.title,
                    "severity": vuln.severity.value,
                    "type": vuln.vulnerability_type.value,
                    "file_path": vuln.file_path,
                    "risk_score": vuln.risk_score
                }
            )
    
    def get_scan_history(self, limit: int = 10) -> List[ScanResult]:
        """Get recent scan history."""
        return sorted(self.scan_history, key=lambda s: s.started_at, reverse=True)[:limit]
    
    def get_vulnerability_stats(self) -> Dict[str, Any]:
        """Get vulnerability statistics."""
        if not self.scan_history:
            return {"total_scans": 0, "total_vulnerabilities": 0}
        
        latest_scan = self.scan_history[-1]
        
        return {
            "total_scans": len(self.scan_history),
            "latest_scan_id": latest_scan.scan_id,
            "latest_scan_date": latest_scan.started_at.isoformat(),
            "total_vulnerabilities": len(latest_scan.vulnerabilities),
            "severity_breakdown": latest_scan.summary,
            "high_risk_count": len([
                v for v in latest_scan.vulnerabilities 
                if v.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
            ])
        }


# Global instance
security_scanner = SecurityScanner()