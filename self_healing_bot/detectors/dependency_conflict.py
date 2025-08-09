"""Dependency conflict detection for package management."""

from typing import List, Dict, Any, Optional, Set, Tuple
import logging
import re
import json
from datetime import datetime, timedelta
from collections import defaultdict, namedtuple
import subprocess
import tempfile
import os
from packaging import version, specifiers

from .base import BaseDetector
from ..core.context import Context

logger = logging.getLogger(__name__)

# Data structures for dependency analysis
Dependency = namedtuple('Dependency', ['name', 'version', 'specifier', 'source'])
ConflictInfo = namedtuple('ConflictInfo', ['packages', 'versions', 'reason', 'severity'])


class DependencyConflictDetector(BaseDetector):
    """Detect dependency conflicts, version incompatibilities, and missing dependencies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Package managers to analyze
        self.supported_managers = self.config.get("supported_managers", [
            "pip", "npm", "yarn", "poetry", "pipenv", "conda", "maven", "gradle"
        ])
        
        # Conflict detection settings
        self.check_version_conflicts = self.config.get("check_version_conflicts", True)
        self.check_missing_dependencies = self.config.get("check_missing_dependencies", True)
        self.check_deprecated_packages = self.config.get("check_deprecated_packages", True)
        self.check_license_conflicts = self.config.get("check_license_conflicts", True)
        
        # Version resolution settings
        self.max_resolution_depth = self.config.get("max_resolution_depth", 5)
        self.ignore_dev_dependencies = self.config.get("ignore_dev_dependencies", False)
        
        # Known problematic combinations
        self.known_conflicts = self.config.get("known_conflicts", {})
        self.deprecated_packages = self.config.get("deprecated_packages", set())
        
        # Caching
        self.resolution_cache = {}
        self.package_info_cache = {}
        self.cache_ttl_hours = self.config.get("cache_ttl_hours", 24)
        
        # Analysis thresholds
        self.max_conflicts_per_file = self.config.get("max_conflicts_per_file", 20)
        self.conflict_severity_threshold = self.config.get("conflict_severity_threshold", "medium")
    
    def get_supported_events(self) -> List[str]:
        return ["push", "pull_request", "dependency_update", "schedule", "workflow_run"]
    
    async def detect(self, context: Context) -> List[Dict[str, Any]]:
        """Detect dependency conflicts across multiple package managers."""
        issues = []
        
        try:
            # Analyze different dependency files
            for manager in self.supported_managers:
                manager_issues = await self._analyze_package_manager(context, manager)
                issues.extend(manager_issues)
            
            # Cross-manager conflict detection
            cross_manager_issues = await self._detect_cross_manager_conflicts(context)
            issues.extend(cross_manager_issues)
            
            # Dependency security and compliance
            security_issues = await self._check_dependency_security_compliance(context)
            issues.extend(security_issues)
            
            # Dependency freshness analysis
            freshness_issues = await self._analyze_dependency_freshness(context)
            issues.extend(freshness_issues)
            
        except Exception as e:
            logger.exception(f"Error in dependency conflict detection: {e}")
            issues.append(self.create_issue(
                issue_type="dependency_analysis_error",
                severity="medium",
                message=f"Dependency conflict detection failed: {str(e)}",
                data={"error_details": str(e)}
            ))
        
        return issues
    
    async def _analyze_package_manager(self, context: Context, manager: str) -> List[Dict[str, Any]]:
        """Analyze dependencies for a specific package manager."""
        issues = []
        
        # Get dependency files for the manager
        dependency_files = await self._get_dependency_files_for_manager(context, manager)
        
        for dep_file in dependency_files:
            file_path = dep_file["path"]
            dependencies = dep_file["dependencies"]
            dev_dependencies = dep_file.get("dev_dependencies", [])
            
            logger.info(f"Analyzing {len(dependencies)} dependencies in {file_path}")
            
            # Version conflict detection
            if self.check_version_conflicts:
                version_conflicts = await self._detect_version_conflicts(
                    dependencies, dev_dependencies, manager, file_path
                )
                issues.extend(version_conflicts)
            
            # Missing dependency detection
            if self.check_missing_dependencies:
                missing_deps = await self._detect_missing_dependencies(
                    dependencies, manager, file_path
                )
                issues.extend(missing_deps)
            
            # Deprecated package detection
            if self.check_deprecated_packages:
                deprecated_issues = await self._detect_deprecated_packages(
                    dependencies, manager, file_path
                )
                issues.extend(deprecated_issues)
            
            # Dependency resolution simulation
            resolution_issues = await self._simulate_dependency_resolution(
                dependencies, dev_dependencies, manager, file_path
            )
            issues.extend(resolution_issues)
        
        return issues
    
    async def _detect_version_conflicts(
        self, 
        dependencies: List[Dict[str, Any]], 
        dev_dependencies: List[Dict[str, Any]], 
        manager: str, 
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Detect version conflicts between dependencies."""
        issues = []
        
        # Build dependency graph
        all_deps = dependencies + ([] if self.ignore_dev_dependencies else dev_dependencies)
        dep_graph = await self._build_dependency_graph(all_deps, manager)
        
        # Find conflicts
        conflicts = self._find_version_conflicts_in_graph(dep_graph)
        
        for conflict in conflicts:
            severity = self._calculate_conflict_severity(conflict, manager)
            
            issues.append(self.create_issue(
                issue_type="dependency_version_conflict",
                severity=severity,
                message=f"Version conflict in {file_path}: {conflict.reason}",
                data={
                    "file_path": file_path,
                    "package_manager": manager,
                    "conflicting_packages": conflict.packages,
                    "conflicting_versions": conflict.versions,
                    "conflict_reason": conflict.reason,
                    "affected_dependencies": self._get_affected_dependencies(conflict, dep_graph),
                    "recommendation": self._get_conflict_resolution_recommendation(conflict, manager)
                }
            ))
        
        return issues
    
    async def _detect_missing_dependencies(
        self, dependencies: List[Dict[str, Any]], manager: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Detect missing transitive dependencies."""
        issues = []
        
        # Simulate installation to find missing deps
        missing_deps = await self._find_missing_transitive_dependencies(dependencies, manager)
        
        for missing_dep in missing_deps:
            issues.append(self.create_issue(
                issue_type="missing_dependency",
                severity="medium",
                message=f"Missing transitive dependency: {missing_dep['name']}",
                data={
                    "file_path": file_path,
                    "package_manager": manager,
                    "missing_package": missing_dep["name"],
                    "required_by": missing_dep["required_by"],
                    "suggested_version": missing_dep.get("suggested_version", "latest"),
                    "recommendation": f"Add {missing_dep['name']} to dependencies or update {missing_dep['required_by']}"
                }
            ))
        
        return issues
    
    async def _detect_deprecated_packages(
        self, dependencies: List[Dict[str, Any]], manager: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Detect usage of deprecated packages."""
        issues = []
        
        deprecated_info = await self._check_for_deprecated_packages(dependencies, manager)
        
        for dep_info in deprecated_info:
            severity = "high" if dep_info.get("security_risk", False) else "medium"
            
            issues.append(self.create_issue(
                issue_type="deprecated_dependency",
                severity=severity,
                message=f"Deprecated package: {dep_info['name']}",
                data={
                    "file_path": file_path,
                    "package_manager": manager,
                    "package_name": dep_info["name"],
                    "current_version": dep_info["version"],
                    "deprecation_reason": dep_info.get("reason", "Package deprecated"),
                    "replacement_package": dep_info.get("replacement", ""),
                    "deprecation_date": dep_info.get("deprecated_date", ""),
                    "recommendation": self._get_deprecation_recommendation(dep_info)
                }
            ))
        
        return issues
    
    async def _simulate_dependency_resolution(
        self, 
        dependencies: List[Dict[str, Any]], 
        dev_dependencies: List[Dict[str, Any]], 
        manager: str, 
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Simulate dependency resolution to find potential issues."""
        issues = []
        
        # Attempt to resolve dependency tree
        try:
            resolution_result = await self._resolve_dependency_tree(
                dependencies, dev_dependencies, manager
            )
            
            # Check for resolution failures
            if resolution_result["status"] == "failed":
                issues.append(self.create_issue(
                    issue_type="dependency_resolution_failure",
                    severity="high",
                    message=f"Dependency resolution failed for {file_path}",
                    data={
                        "file_path": file_path,
                        "package_manager": manager,
                        "resolution_error": resolution_result["error"],
                        "problematic_packages": resolution_result.get("problematic_packages", []),
                        "recommendation": "Review dependency versions and resolve conflicts"
                    }
                ))
            
            # Check for circular dependencies
            circular_deps = resolution_result.get("circular_dependencies", [])
            if circular_deps:
                issues.append(self.create_issue(
                    issue_type="circular_dependency",
                    severity="high",
                    message=f"Circular dependency detected in {file_path}",
                    data={
                        "file_path": file_path,
                        "package_manager": manager,
                        "circular_path": circular_deps,
                        "recommendation": "Refactor dependencies to break circular reference"
                    }
                ))
            
            # Check for oversized dependency trees
            total_deps = resolution_result.get("total_dependencies", 0)
            if total_deps > 500:  # Threshold for large dependency trees
                issues.append(self.create_issue(
                    issue_type="oversized_dependency_tree",
                    severity="medium",
                    message=f"Large dependency tree: {total_deps} total dependencies",
                    data={
                        "file_path": file_path,
                        "package_manager": manager,
                        "total_dependencies": total_deps,
                        "recommendation": "Consider reducing dependencies or using dependency bundling"
                    }
                ))
        
        except Exception as e:
            logger.exception(f"Error simulating dependency resolution: {e}")
        
        return issues
    
    async def _detect_cross_manager_conflicts(self, context: Context) -> List[Dict[str, Any]]:
        """Detect conflicts between different package managers."""
        issues = []
        
        # Get all dependency files
        all_dep_files = {}
        for manager in self.supported_managers:
            files = await self._get_dependency_files_for_manager(context, manager)
            if files:
                all_dep_files[manager] = files
        
        # Check for overlapping packages
        package_managers_map = defaultdict(list)
        
        for manager, files in all_dep_files.items():
            for file_info in files:
                for dep in file_info["dependencies"]:
                    package_name = dep["name"]
                    package_managers_map[package_name].append({
                        "manager": manager,
                        "version": dep.get("version", ""),
                        "file": file_info["path"]
                    })
        
        # Find packages managed by multiple managers
        for package_name, managers_info in package_managers_map.items():
            if len(managers_info) > 1:
                versions = [info["version"] for info in managers_info]
                unique_versions = set(v for v in versions if v)
                
                if len(unique_versions) > 1:
                    issues.append(self.create_issue(
                        issue_type="cross_manager_version_conflict",
                        severity="medium",
                        message=f"Package '{package_name}' has different versions across managers",
                        data={
                            "package_name": package_name,
                            "managers_info": managers_info,
                            "versions": list(unique_versions),
                            "recommendation": f"Standardize {package_name} version across all package managers"
                        }
                    ))
        
        return issues
    
    async def _check_dependency_security_compliance(self, context: Context) -> List[Dict[str, Any]]:
        """Check dependencies for security and compliance issues."""
        issues = []
        
        # Get all dependencies across managers
        all_dependencies = await self._get_all_dependencies(context)
        
        # Check against security advisories
        security_issues = await self._check_security_advisories(all_dependencies)
        issues.extend(security_issues)
        
        # Check license compatibility
        if self.check_license_conflicts:
            license_issues = await self._check_license_compatibility(all_dependencies)
            issues.extend(license_issues)
        
        return issues
    
    async def _analyze_dependency_freshness(self, context: Context) -> List[Dict[str, Any]]:
        """Analyze how up-to-date dependencies are."""
        issues = []
        
        all_dependencies = await self._get_all_dependencies(context)
        
        outdated_packages = []
        
        for dep_info in all_dependencies:
            package_name = dep_info["name"]
            current_version = dep_info["version"]
            manager = dep_info["manager"]
            
            # Get latest version info
            latest_info = await self._get_latest_package_info(package_name, manager)
            
            if latest_info and current_version:
                try:
                    current_ver = version.parse(current_version)
                    latest_ver = version.parse(latest_info["version"])
                    
                    # Calculate age and version difference
                    version_behind = self._calculate_version_distance(current_ver, latest_ver)
                    
                    if version_behind["major"] > 0 or version_behind["minor"] > 5:
                        outdated_packages.append({
                            "package": package_name,
                            "current_version": current_version,
                            "latest_version": latest_info["version"],
                            "versions_behind": version_behind,
                            "manager": manager,
                            "file_path": dep_info["file_path"],
                            "last_updated": latest_info.get("last_updated", ""),
                            "changelog_url": latest_info.get("changelog_url", "")
                        })
                
                except Exception as e:
                    logger.debug(f"Version comparison failed for {package_name}: {e}")
        
        if outdated_packages:
            # Group by severity
            severely_outdated = [p for p in outdated_packages if p["versions_behind"]["major"] > 0]
            moderately_outdated = [p for p in outdated_packages if p["versions_behind"]["minor"] > 3]
            
            if severely_outdated:
                issues.append(self.create_issue(
                    issue_type="severely_outdated_dependencies",
                    severity="medium",
                    message=f"{len(severely_outdated)} dependencies are severely outdated",
                    data={
                        "outdated_packages": severely_outdated,
                        "recommendation": "Update major version dependencies - review changelog for breaking changes"
                    }
                ))
            
            if moderately_outdated:
                issues.append(self.create_issue(
                    issue_type="outdated_dependencies",
                    severity="low",
                    message=f"{len(moderately_outdated)} dependencies are moderately outdated",
                    data={
                        "outdated_packages": moderately_outdated,
                        "recommendation": "Consider updating minor version dependencies"
                    }
                ))
        
        return issues
    
    async def _get_dependency_files_for_manager(self, context: Context, manager: str) -> List[Dict[str, Any]]:
        """Get dependency files for a specific package manager."""
        # Mock implementation - in production, would scan actual repository
        mock_files = {
            "pip": [
                {
                    "path": "requirements.txt",
                    "dependencies": [
                        {"name": "flask", "version": "2.0.1", "specifier": "==2.0.1"},
                        {"name": "numpy", "version": "1.21.0", "specifier": ">=1.21.0"},
                        {"name": "pandas", "version": "1.3.0", "specifier": "~=1.3.0"},
                        {"name": "scikit-learn", "version": "0.24.2", "specifier": "==0.24.2"},
                        {"name": "deprecated-pkg", "version": "1.0.0", "specifier": "==1.0.0"}
                    ],
                    "dev_dependencies": [
                        {"name": "pytest", "version": "6.2.4", "specifier": ">=6.0.0"},
                        {"name": "black", "version": "21.6b0", "specifier": ">=21.0.0"}
                    ]
                }
            ],
            "npm": [
                {
                    "path": "package.json",
                    "dependencies": [
                        {"name": "lodash", "version": "4.17.20", "specifier": "^4.17.20"},
                        {"name": "express", "version": "4.17.1", "specifier": "^4.17.1"},
                        {"name": "react", "version": "17.0.2", "specifier": "^17.0.2"},
                        {"name": "moment", "version": "2.29.1", "specifier": "^2.29.1"}  # Deprecated
                    ],
                    "dev_dependencies": [
                        {"name": "webpack", "version": "5.40.0", "specifier": "^5.40.0"},
                        {"name": "babel-loader", "version": "8.2.2", "specifier": "^8.2.2"}
                    ]
                }
            ]
        }
        
        return mock_files.get(manager, [])
    
    async def _build_dependency_graph(self, dependencies: List[Dict[str, Any]], manager: str) -> Dict[str, Any]:
        """Build a dependency graph for conflict detection."""
        graph = {
            "nodes": {},
            "edges": [],
            "conflicts": []
        }
        
        # Add primary dependencies
        for dep in dependencies:
            dep_name = dep["name"]
            dep_version = dep.get("version", "")
            dep_specifier = dep.get("specifier", "")
            
            graph["nodes"][dep_name] = {
                "version": dep_version,
                "specifier": dep_specifier,
                "type": "direct",
                "transitive_deps": await self._get_transitive_dependencies(dep_name, dep_version, manager)
            }
        
        # Find potential conflicts
        conflicts = self._analyze_graph_for_conflicts(graph)
        graph["conflicts"] = conflicts
        
        return graph
    
    def _find_version_conflicts_in_graph(self, dep_graph: Dict[str, Any]) -> List[ConflictInfo]:
        """Find version conflicts in the dependency graph."""
        conflicts = []
        
        # Check for direct version conflicts
        version_requirements = defaultdict(list)
        
        for node_name, node_info in dep_graph["nodes"].items():
            version_spec = node_info.get("specifier", "")
            if version_spec:
                version_requirements[node_name].append(version_spec)
                
                # Check transitive dependencies
                for trans_dep in node_info.get("transitive_deps", []):
                    trans_name = trans_dep["name"]
                    trans_spec = trans_dep.get("specifier", "")
                    if trans_spec:
                        version_requirements[trans_name].append(trans_spec)
        
        # Find conflicting requirements
        for package_name, specs in version_requirements.items():
            if len(specs) > 1:
                # Check if specifications are compatible
                if not self._are_version_specs_compatible(specs):
                    conflicts.append(ConflictInfo(
                        packages=[package_name],
                        versions=specs,
                        reason=f"Incompatible version requirements for {package_name}: {', '.join(specs)}",
                        severity="high"
                    ))
        
        return conflicts
    
    def _are_version_specs_compatible(self, specs: List[str]) -> bool:
        """Check if version specifications are compatible."""
        try:
            # Use packaging library to check compatibility
            spec_set = specifiers.SpecifierSet(",".join(specs))
            
            # Try to find a version that satisfies all specs
            # This is a simplified check - in practice, would check against available versions
            return len(str(spec_set)) > 0
        
        except Exception:
            # If parsing fails, assume incompatible
            return False
    
    async def _get_transitive_dependencies(self, package_name: str, package_version: str, manager: str) -> List[Dict[str, Any]]:
        """Get transitive dependencies for a package."""
        # Mock implementation - in production, would query package registries
        mock_transitive_deps = {
            "flask": [
                {"name": "werkzeug", "version": "2.0.1", "specifier": ">=2.0.0"},
                {"name": "jinja2", "version": "3.0.1", "specifier": ">=3.0.0"},
                {"name": "click", "version": "8.0.1", "specifier": ">=7.0.0"}
            ],
            "pandas": [
                {"name": "numpy", "version": "1.21.0", "specifier": ">=1.20.0"},
                {"name": "python-dateutil", "version": "2.8.2", "specifier": ">=2.7.0"},
                {"name": "pytz", "version": "2021.1", "specifier": ">=2020.1"}
            ],
            "react": [
                {"name": "loose-envify", "version": "1.4.0", "specifier": "^1.1.0"},
                {"name": "object-assign", "version": "4.1.1", "specifier": "^4.1.1"}
            ]
        }
        
        return mock_transitive_deps.get(package_name, [])
    
    def _analyze_graph_for_conflicts(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze dependency graph for various types of conflicts."""
        conflicts = []
        
        # Check for known problematic combinations
        for conflict_pattern, conflict_info in self.known_conflicts.items():
            if self._matches_conflict_pattern(graph, conflict_pattern):
                conflicts.append({
                    "type": "known_conflict",
                    "pattern": conflict_pattern,
                    "description": conflict_info.get("description", "Known problematic combination"),
                    "severity": conflict_info.get("severity", "medium")
                })
        
        return conflicts
    
    def _matches_conflict_pattern(self, graph: Dict[str, Any], pattern: str) -> bool:
        """Check if dependency graph matches a known conflict pattern."""
        # Simple pattern matching - could be more sophisticated
        pattern_packages = pattern.split("+")
        graph_packages = set(graph["nodes"].keys())
        
        return all(pkg.strip() in graph_packages for pkg in pattern_packages)
    
    async def _find_missing_transitive_dependencies(self, dependencies: List[Dict[str, Any]], manager: str) -> List[Dict[str, Any]]:
        """Find missing transitive dependencies."""
        missing_deps = []
        
        # Mock implementation - would use actual package resolution
        for dep in dependencies:
            if dep["name"] == "flask":
                # Simulate missing werkzeug dependency
                missing_deps.append({
                    "name": "werkzeug",
                    "required_by": "flask",
                    "suggested_version": "2.0.1",
                    "reason": "Required by flask but not explicitly declared"
                })
        
        return missing_deps
    
    async def _check_for_deprecated_packages(self, dependencies: List[Dict[str, Any]], manager: str) -> List[Dict[str, Any]]:
        """Check for deprecated packages."""
        deprecated_info = []
        
        # Mock deprecated package database
        deprecated_packages = {
            "deprecated-pkg": {
                "reason": "Package is no longer maintained",
                "replacement": "new-pkg",
                "deprecated_date": "2021-01-01",
                "security_risk": True
            },
            "moment": {
                "reason": "Project is in maintenance mode",
                "replacement": "dayjs or date-fns",
                "deprecated_date": "2020-09-15",
                "security_risk": False
            }
        }
        
        for dep in dependencies:
            if dep["name"] in deprecated_packages:
                info = deprecated_packages[dep["name"]].copy()
                info["name"] = dep["name"]
                info["version"] = dep.get("version", "")
                deprecated_info.append(info)
        
        return deprecated_info
    
    async def _resolve_dependency_tree(
        self, 
        dependencies: List[Dict[str, Any]], 
        dev_dependencies: List[Dict[str, Any]], 
        manager: str
    ) -> Dict[str, Any]:
        """Simulate dependency tree resolution."""
        # Mock implementation of dependency resolution
        all_deps = dependencies + dev_dependencies
        total_deps = len(all_deps)
        
        # Simulate some resolution scenarios
        if any(dep["name"] == "problematic-package" for dep in all_deps):
            return {
                "status": "failed",
                "error": "Cannot resolve version conflicts",
                "problematic_packages": ["problematic-package"]
            }
        
        # Check for circular dependencies (mock)
        if any(dep["name"] == "circular-a" for dep in all_deps):
            return {
                "status": "success",
                "total_dependencies": total_deps * 5,  # Simulate transitive deps
                "circular_dependencies": ["circular-a", "circular-b", "circular-a"]
            }
        
        return {
            "status": "success",
            "total_dependencies": total_deps * 3,  # Simulate transitive expansion
            "resolved_versions": {dep["name"]: dep.get("version", "latest") for dep in all_deps}
        }
    
    async def _get_all_dependencies(self, context: Context) -> List[Dict[str, Any]]:
        """Get all dependencies across all package managers."""
        all_deps = []
        
        for manager in self.supported_managers:
            files = await self._get_dependency_files_for_manager(context, manager)
            for file_info in files:
                for dep in file_info["dependencies"]:
                    all_deps.append({
                        **dep,
                        "manager": manager,
                        "file_path": file_info["path"]
                    })
        
        return all_deps
    
    async def _check_security_advisories(self, dependencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check dependencies against security advisory databases."""
        issues = []
        
        # Mock security advisory check
        vulnerable_packages = {
            "lodash": {
                "advisory_id": "GHSA-p6mc-m468-83gw",
                "severity": "high",
                "description": "Prototype pollution vulnerability",
                "affected_versions": "<4.17.21",
                "patched_versions": ">=4.17.21"
            }
        }
        
        for dep in dependencies:
            if dep["name"] in vulnerable_packages:
                vuln_info = vulnerable_packages[dep["name"]]
                issues.append(self.create_issue(
                    issue_type="dependency_security_vulnerability",
                    severity=vuln_info["severity"],
                    message=f"Security vulnerability in {dep['name']}: {vuln_info['description']}",
                    data={
                        "package_name": dep["name"],
                        "current_version": dep.get("version", ""),
                        "advisory_id": vuln_info["advisory_id"],
                        "affected_versions": vuln_info["affected_versions"],
                        "patched_versions": vuln_info["patched_versions"],
                        "file_path": dep["file_path"],
                        "package_manager": dep["manager"],
                        "recommendation": f"Update {dep['name']} to {vuln_info['patched_versions']}"
                    }
                ))
        
        return issues
    
    async def _check_license_compatibility(self, dependencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for license compatibility issues."""
        issues = []
        
        # Mock license information
        license_info = {
            "flask": "BSD-3-Clause",
            "numpy": "BSD-3-Clause", 
            "pandas": "BSD-3-Clause",
            "problematic-gpl-pkg": "GPL-3.0"  # Potentially problematic
        }
        
        # Check for incompatible licenses
        incompatible_licenses = ["GPL-3.0", "AGPL-3.0"]
        
        for dep in dependencies:
            dep_license = license_info.get(dep["name"], "Unknown")
            
            if dep_license in incompatible_licenses:
                issues.append(self.create_issue(
                    issue_type="license_compatibility_issue",
                    severity="medium",
                    message=f"Potentially incompatible license: {dep['name']} ({dep_license})",
                    data={
                        "package_name": dep["name"],
                        "license": dep_license,
                        "file_path": dep["file_path"],
                        "package_manager": dep["manager"],
                        "recommendation": f"Review license compatibility for {dep['name']}"
                    }
                ))
        
        return issues
    
    async def _get_latest_package_info(self, package_name: str, manager: str) -> Optional[Dict[str, Any]]:
        """Get latest version information for a package."""
        # Mock implementation - would query package registries
        mock_latest_versions = {
            "flask": {"version": "2.1.2", "last_updated": "2022-04-01"},
            "numpy": {"version": "1.22.3", "last_updated": "2022-03-15"},
            "lodash": {"version": "4.17.21", "last_updated": "2021-02-20"},
            "express": {"version": "4.18.1", "last_updated": "2022-04-25"}
        }
        
        return mock_latest_versions.get(package_name)
    
    def _calculate_version_distance(self, current_version: version.Version, latest_version: version.Version) -> Dict[str, int]:
        """Calculate the distance between current and latest versions."""
        return {
            "major": latest_version.major - current_version.major,
            "minor": latest_version.minor - current_version.minor,
            "patch": latest_version.micro - current_version.micro
        }
    
    def _calculate_conflict_severity(self, conflict: ConflictInfo, manager: str) -> str:
        """Calculate severity of a dependency conflict."""
        if "major" in conflict.reason.lower() or "incompatible" in conflict.reason.lower():
            return "high"
        elif "minor" in conflict.reason.lower():
            return "medium"
        else:
            return "low"
    
    def _get_affected_dependencies(self, conflict: ConflictInfo, dep_graph: Dict[str, Any]) -> List[str]:
        """Get list of dependencies affected by the conflict."""
        affected = []
        
        for package in conflict.packages:
            # Find which dependencies depend on this package
            for node_name, node_info in dep_graph["nodes"].items():
                transitive_deps = node_info.get("transitive_deps", [])
                if any(dep["name"] == package for dep in transitive_deps):
                    affected.append(node_name)
        
        return affected
    
    def _get_conflict_resolution_recommendation(self, conflict: ConflictInfo, manager: str) -> str:
        """Get recommendation for resolving a dependency conflict."""
        base_recommendations = {
            "version_conflict": f"Update conflicting packages to compatible versions: {', '.join(conflict.packages)}",
            "missing_dependency": f"Add missing dependencies: {', '.join(conflict.packages)}",
            "circular_dependency": f"Refactor to break circular dependency: {' -> '.join(conflict.packages)}"
        }
        
        # Default recommendation based on conflict type
        if "version" in conflict.reason.lower():
            return base_recommendations["version_conflict"]
        elif "missing" in conflict.reason.lower():
            return base_recommendations["missing_dependency"]
        elif "circular" in conflict.reason.lower():
            return base_recommendations["circular_dependency"]
        else:
            return f"Review and resolve dependency issue: {conflict.reason}"
    
    def _get_deprecation_recommendation(self, dep_info: Dict[str, Any]) -> str:
        """Get recommendation for handling deprecated packages."""
        replacement = dep_info.get("replacement", "")
        
        if replacement:
            return f"Replace {dep_info['name']} with {replacement}"
        else:
            return f"Find alternative to deprecated package {dep_info['name']}"