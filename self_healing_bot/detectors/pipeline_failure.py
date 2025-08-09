"""Pipeline and CI/CD failure detection with comprehensive analysis."""

from typing import List, Dict, Any, Optional, Tuple
import re
import logging
import json
from datetime import datetime, timedelta
from collections import defaultdict

from .base import BaseDetector
from ..core.context import Context

logger = logging.getLogger(__name__)


class PipelineFailureDetector(BaseDetector):
    """Detect pipeline and CI/CD failures with comprehensive analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.failure_history = defaultdict(list)
        self.failure_rate_threshold = self.config.get("failure_rate_threshold", 0.3)  # 30%
        self.window_hours = self.config.get("window_hours", 24)
        self.max_retry_attempts = self.config.get("max_retry_attempts", 3)
        self.critical_workflows = set(self.config.get("critical_workflows", [
            "production-deployment", "release", "security-scan", "main-build"
        ]))
    
    def get_supported_events(self) -> List[str]:
        return ["workflow_run", "check_run", "status", "push", "pull_request"]
    
    async def detect(self, context: Context) -> List[Dict[str, Any]]:
        """Detect pipeline failures and categorize them."""
        issues = []
        
        if context.event_type == "workflow_run":
            issues.extend(await self._detect_workflow_failures(context))
            if context.event_data.get("workflow_run", {}).get("conclusion") == "failure":
                issues.extend(await self._detect_failure_patterns(context))
                issues.extend(await self._detect_critical_workflow_failures(context))
        elif context.event_type == "check_run":
            issues.extend(await self._detect_check_failures(context))
        elif context.event_type == "status":
            issues.extend(await self._detect_status_failures(context))
        elif context.event_type in ["push", "pull_request"]:
            # Proactive failure prediction based on code changes
            issues.extend(await self._detect_potential_failures(context))
        
        return issues
    
    async def _detect_workflow_failures(self, context: Context) -> List[Dict[str, Any]]:
        """Detect GitHub Actions workflow failures with detailed analysis."""
        issues = []
        
        workflow_data = context.event_data.get("workflow_run", {})
        conclusion = workflow_data.get("conclusion")
        workflow_name = workflow_data.get("name", "unknown")
        
        if conclusion == "failure":
            # Categorize failure type based on workflow name and logs
            failure_type = await self._categorize_workflow_failure(context, workflow_data)
            
            # Get additional failure context
            failure_context = await self._get_failure_context(context, workflow_data)
            
            issues.append(self.create_issue(
                issue_type=f"workflow_failure_{failure_type}",
                severity=self._get_failure_severity(failure_type),
                message=f"Workflow '{workflow_name}' failed with {failure_type}",
                data={
                    "workflow_id": workflow_data.get("id"),
                    "workflow_name": workflow_name,
                    "failure_type": failure_type,
                    "run_url": workflow_data.get("html_url"),
                    "head_sha": workflow_data.get("head_sha"),
                    "run_number": workflow_data.get("run_number"),
                    "run_attempt": workflow_data.get("run_attempt", 1),
                    "failure_context": failure_context,
                    "recommendation": self._get_failure_recommendation(failure_type, failure_context)
                }
            ))
        
        elif conclusion == "cancelled":
            issues.append(self.create_issue(
                issue_type="workflow_cancelled",
                severity="medium",
                message=f"Workflow '{workflow_name}' was cancelled",
                data={
                    "workflow_id": workflow_data.get("id"),
                    "workflow_name": workflow_name,
                    "run_number": workflow_data.get("run_number"),
                    "recommendation": "Investigate cancellation cause - may indicate resource constraints or manual intervention"
                }
            ))
        
        elif conclusion == "timed_out":
            issues.append(self.create_issue(
                issue_type="workflow_timeout",
                severity="high",
                message=f"Workflow '{workflow_name}' timed out",
                data={
                    "workflow_id": workflow_data.get("id"),
                    "workflow_name": workflow_name,
                    "run_number": workflow_data.get("run_number"),
                    "recommendation": "Review workflow timeout settings and optimize long-running steps"
                }
            ))
        
        return issues
    
    async def _categorize_workflow_failure(self, context: Context, workflow_data: Dict[str, Any]) -> str:
        """Categorize the type of workflow failure with comprehensive analysis."""
        workflow_name = workflow_data.get("name", "").lower()
        workflow_id = workflow_data.get("id")
        
        # Get workflow logs and jobs for detailed analysis
        failure_details = await self._analyze_workflow_logs(context, workflow_id)
        
        # Primary categorization based on workflow name patterns
        if any(keyword in workflow_name for keyword in ["test", "ci", "check"]):
            return await self._categorize_test_failure(failure_details)
        elif any(keyword in workflow_name for keyword in ["build", "compile"]):
            return await self._categorize_build_failure(failure_details)
        elif any(keyword in workflow_name for keyword in ["deploy", "release"]):
            return await self._categorize_deployment_failure(failure_details)
        elif any(keyword in workflow_name for keyword in ["train", "ml", "model"]):
            return await self._categorize_training_failure(failure_details)
        elif any(keyword in workflow_name for keyword in ["lint", "format", "style"]):
            return "code_quality_failure"
        elif any(keyword in workflow_name for keyword in ["security", "scan", "audit"]):
            return "security_failure"
        
        # Secondary categorization based on failure patterns
        if failure_details:
            return await self._categorize_by_error_patterns(failure_details)
        
        return "unknown_failure"
    
    async def _analyze_workflow_logs(self, context: Context, workflow_id: str) -> Dict[str, Any]:
        """Analyze workflow logs for detailed failure categorization."""
        try:
            # Mock implementation - in production, this would fetch actual logs
            # via GitHub API: /repos/{owner}/{repo}/actions/runs/{run_id}/logs
            return {
                "error_patterns": self._get_mock_error_patterns(),
                "failed_steps": self._get_mock_failed_steps(),
                "resource_usage": self._get_mock_resource_usage(),
                "exit_codes": {"setup": 0, "test": 1, "build": 0, "deploy": 2},
                "duration_minutes": 45,
                "retry_count": 2
            }
        except Exception as e:
            logger.exception(f"Failed to analyze workflow logs for {workflow_id}: {e}")
            return {}
    
    async def _categorize_test_failure(self, failure_details: Dict[str, Any]) -> str:
        """Categorize test failures with detailed analysis."""
        if not failure_details:
            return "test_failure"
        
        error_patterns = failure_details.get("error_patterns", [])
        failed_steps = failure_details.get("failed_steps", [])
        
        # Check for specific test failure patterns
        if "timeout_error" in error_patterns:
            return "test_timeout_failure"
        elif "assertion_error" in error_patterns:
            return "test_assertion_failure"
        elif "setup_error" in error_patterns:
            return "test_setup_failure"
        elif "fixture_error" in error_patterns:
            return "test_fixture_failure"
        elif "integration_error" in error_patterns:
            return "test_integration_failure"
        elif any("unit" in step for step in failed_steps):
            return "unit_test_failure"
        elif any("e2e" in step or "end-to-end" in step for step in failed_steps):
            return "e2e_test_failure"
        elif "flaky_test" in error_patterns:
            return "flaky_test_failure"
        
        return "test_failure"
    
    async def _categorize_build_failure(self, failure_details: Dict[str, Any]) -> str:
        """Categorize build failures with detailed analysis."""
        if not failure_details:
            return "build_failure"
        
        error_patterns = failure_details.get("error_patterns", [])
        
        if "compilation_error" in error_patterns:
            return "compilation_failure"
        elif "dependency_error" in error_patterns:
            return "dependency_build_failure"
        elif "memory_error" in error_patterns or "gpu_oom" in error_patterns:
            return "build_resource_failure"
        elif "docker_error" in error_patterns:
            return "docker_build_failure"
        elif "npm_error" in error_patterns or "yarn_error" in error_patterns:
            return "frontend_build_failure"
        elif "webpack_error" in error_patterns:
            return "webpack_build_failure"
        elif "linting_error" in error_patterns:
            return "linting_build_failure"
        
        return "build_failure"
    
    async def _categorize_deployment_failure(self, failure_details: Dict[str, Any]) -> str:
        """Categorize deployment failures with detailed analysis."""
        if not failure_details:
            return "deployment_failure"
        
        error_patterns = failure_details.get("error_patterns", [])
        
        if "kubernetes_error" in error_patterns:
            return "k8s_deployment_failure"
        elif "docker_error" in error_patterns:
            return "container_deployment_failure"
        elif "authentication_error" in error_patterns:
            return "deployment_auth_failure"
        elif "network_error" in error_patterns:
            return "deployment_network_failure"
        elif "health_check_error" in error_patterns:
            return "deployment_health_failure"
        elif "rollback_error" in error_patterns:
            return "deployment_rollback_failure"
        elif "terraform_error" in error_patterns:
            return "infrastructure_deployment_failure"
        
        return "deployment_failure"
    
    async def _categorize_training_failure(self, failure_details: Dict[str, Any]) -> str:
        """Categorize ML training failures with detailed analysis."""
        if not failure_details:
            return "training_failure"
        
        error_patterns = failure_details.get("error_patterns", [])
        resource_usage = failure_details.get("resource_usage", {})
        
        if "gpu_oom" in error_patterns:
            return "gpu_oom_training_failure"
        elif "cuda_error" in error_patterns:
            return "cuda_training_failure"
        elif "data_loading_error" in error_patterns:
            return "data_training_failure"
        elif "convergence_error" in error_patterns:
            return "convergence_training_failure"
        elif "checkpoint_error" in error_patterns:
            return "checkpoint_training_failure"
        elif "distributed_training_error" in error_patterns:
            return "distributed_training_failure"
        elif resource_usage.get("memory_usage", 0) > 0.9:
            return "memory_training_failure"
        elif "validation_error" in error_patterns:
            return "validation_training_failure"
        
        return "training_failure"
    
    async def _categorize_by_error_patterns(self, failure_details: Dict[str, Any]) -> str:
        """Categorize failures based on error patterns when workflow name is unclear."""
        error_patterns = failure_details.get("error_patterns", [])
        
        if "gpu_oom" in error_patterns or "cuda_error" in error_patterns:
            return "resource_failure"
        elif "authentication_error" in error_patterns:
            return "auth_failure"
        elif "network_error" in error_patterns or "timeout_error" in error_patterns:
            return "network_failure"
        elif "dependency_error" in error_patterns or "import_error" in error_patterns:
            return "dependency_failure"
        elif "permission_error" in error_patterns:
            return "permission_failure"
        elif "disk_space_error" in error_patterns:
            return "storage_failure"
        
        return "unknown_failure"
    
    async def _get_failure_context(self, context: Context, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional context about the failure."""
        return {
            "actor": workflow_data.get("actor", {}).get("login", "unknown"),
            "event": workflow_data.get("event", "unknown"),
            "branch": workflow_data.get("head_branch", "unknown"),
            "commit_message": workflow_data.get("head_commit", {}).get("message", "")[:100],
            "previous_run_conclusion": "success",  # Mock - would fetch from API
            "failure_count_last_24h": len(self.failure_history.get(workflow_data.get("name", ""), [])),
            "workflow_path": workflow_data.get("path", "unknown")
        }
    
    async def _detect_failure_patterns(self, context: Context) -> List[Dict[str, Any]]:
        """Detect recurring failure patterns and trends."""
        issues = []
        
        # Analyze failure history for patterns
        workflow_name = context.event_data.get("workflow_run", {}).get("name", "unknown")
        
        # Track failure in history
        current_time = datetime.utcnow()
        self.failure_history[workflow_name].append({
            "timestamp": current_time,
            "type": "failure"
        })
        
        # Clean old entries
        cutoff_time = current_time - timedelta(hours=self.window_hours)
        self.failure_history[workflow_name] = [
            entry for entry in self.failure_history[workflow_name]
            if entry["timestamp"] > cutoff_time
        ]
        
        # Calculate failure rate
        recent_failures = len(self.failure_history[workflow_name])
        
        if recent_failures >= 3:  # Minimum failures to detect pattern
            failure_rate = recent_failures / self.window_hours  # failures per hour
            
            if failure_rate > self.failure_rate_threshold:
                issues.append(self.create_issue(
                    issue_type="recurring_failure_pattern",
                    severity="high",
                    message=f"High failure rate detected for {workflow_name}: {failure_rate:.2f} failures/hour",
                    data={
                        "workflow_name": workflow_name,
                        "failure_rate": failure_rate,
                        "recent_failures": recent_failures,
                        "threshold": self.failure_rate_threshold,
                        "window_hours": self.window_hours,
                        "recommendation": "Investigate recurring issue and consider workflow maintenance"
                    }
                ))
        
        return issues
    
    async def _detect_critical_workflow_failures(self, context: Context) -> List[Dict[str, Any]]:
        """Detect failures in critical workflows that require immediate attention."""
        issues = []
        
        workflow_data = context.event_data.get("workflow_run", {})
        workflow_name = workflow_data.get("name", "")
        
        if workflow_name in self.critical_workflows and workflow_data.get("conclusion") == "failure":
            issues.append(self.create_issue(
                issue_type="critical_workflow_failure",
                severity="critical",
                message=f"Critical workflow '{workflow_name}' failed - immediate attention required",
                data={
                    "workflow_name": workflow_name,
                    "workflow_id": workflow_data.get("id"),
                    "run_url": workflow_data.get("html_url"),
                    "head_sha": workflow_data.get("head_sha"),
                    "recommendation": "Immediate investigation and fix required for critical workflow"
                }
            ))
        
        return issues
    
    async def _detect_potential_failures(self, context: Context) -> List[Dict[str, Any]]:
        """Detect potential failures based on code changes."""
        issues = []
        
        # Mock implementation - analyze changed files for potential issues
        changed_files = self._get_mock_changed_files(context)
        
        risk_analysis = self._analyze_change_risk(changed_files)
        
        if risk_analysis["overall_risk"] == "high":
            issues.append(self.create_issue(
                issue_type="potential_failure_risk",
                severity="medium",
                message=f"High-risk changes detected: {risk_analysis['high_risk_count']} high-risk files",
                data={
                    "changed_files": changed_files,
                    "risk_analysis": risk_analysis,
                    "recommendation": "Consider additional testing and gradual deployment"
                }
            ))
        
        return issues
    
    def _analyze_change_risk(self, changed_files: List[str]) -> Dict[str, Any]:
        """Analyze risk level of file changes."""
        risk_counts = {"high": 0, "medium": 0, "low": 0}
        
        for file_path in changed_files:
            risk = self._assess_file_risk(file_path)
            risk_counts[risk] += 1
        
        total_files = len(changed_files)
        high_risk_ratio = risk_counts["high"] / total_files if total_files > 0 else 0
        
        overall_risk = "high" if high_risk_ratio > 0.3 else "medium" if high_risk_ratio > 0.1 else "low"
        
        return {
            "overall_risk": overall_risk,
            "high_risk_count": risk_counts["high"],
            "medium_risk_count": risk_counts["medium"],
            "low_risk_count": risk_counts["low"],
            "high_risk_ratio": high_risk_ratio,
            "total_files": total_files
        }
    
    async def _detect_check_failures(self, context: Context) -> List[Dict[str, Any]]:
        """Detect check run failures with enhanced analysis."""
        issues = []
        
        check_run = context.event_data.get("check_run", {})
        conclusion = check_run.get("conclusion")
        check_name = check_run.get("name", "unknown")
        
        if conclusion == "failure":
            # Analyze check failure details
            failure_analysis = await self._analyze_check_failure(check_run)
            
            issues.append(self.create_issue(
                issue_type="check_failure",
                severity=self._get_check_failure_severity(check_name, failure_analysis),
                message=f"Check '{check_name}' failed: {failure_analysis.get('summary', 'Unknown error')}",
                data={
                    "check_id": check_run.get("id"),
                    "check_name": check_name,
                    "details_url": check_run.get("details_url"),
                    "failure_analysis": failure_analysis,
                    "recommendation": self._get_check_failure_recommendation(check_name, failure_analysis)
                }
            ))
        
        return issues
    
    async def _analyze_check_failure(self, check_run: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze check run failure for detailed information."""
        # Mock analysis - in production, would analyze actual check details
        return {
            "summary": "Test suite failed with 3 failing tests",
            "error_type": "test_failure",
            "failed_count": 3,
            "total_count": 150,
            "duration_seconds": 120
        }
    
    def _get_check_failure_severity(self, check_name: str, failure_analysis: Dict[str, Any]) -> str:
        """Get severity for check failures."""
        if "security" in check_name.lower():
            return "high"
        elif failure_analysis.get("failed_count", 0) > 10:
            return "high"
        elif "required" in check_name.lower():
            return "medium"
        else:
            return "low"
    
    def _get_check_failure_recommendation(self, check_name: str, failure_analysis: Dict[str, Any]) -> str:
        """Get recommendation for check failures."""
        if "security" in check_name.lower():
            return "Address security issues before merging"
        elif failure_analysis.get("failed_count", 0) > 10:
            return "Multiple test failures detected - review changes carefully"
        else:
            return "Review and fix failing checks before proceeding"
    
    async def _detect_status_failures(self, context: Context) -> List[Dict[str, Any]]:
        """Detect status check failures with enhanced context."""
        issues = []
        
        status_data = context.event_data
        state = status_data.get("state")
        status_context = status_data.get("context", "unknown")
        
        if state == "failure":
            issues.append(self.create_issue(
                issue_type="status_failure",
                severity=self._get_status_failure_severity(status_context),
                message=f"Status check '{status_context}' failed: {status_data.get('description', 'Unknown error')}",
                data={
                    "status_context": status_context,
                    "description": status_data.get("description"),
                    "target_url": status_data.get("target_url"),
                    "commit_sha": status_data.get("sha"),
                    "recommendation": self._get_status_failure_recommendation(status_context)
                }
            ))
        
        return issues
    
    def _get_status_failure_severity(self, status_context: str) -> str:
        """Get severity for status failures."""
        if any(keyword in status_context.lower() for keyword in ["security", "deploy", "production"]):
            return "high"
        elif any(keyword in status_context.lower() for keyword in ["test", "build"]):
            return "medium"
        else:
            return "low"
    
    def _get_status_failure_recommendation(self, status_context: str) -> str:
        """Get recommendation for status failures."""
        if "security" in status_context.lower():
            return "Review security scan results and address vulnerabilities"
        elif "deploy" in status_context.lower():
            return "Check deployment configuration and target environment"
        else:
            return "Review status check details and resolve issues"
    
    def _get_failure_recommendation(self, failure_type: str, failure_context: Dict[str, Any]) -> str:
        """Get specific recommendation based on failure type and context."""
        recommendations = {
            "gpu_oom_training_failure": "Reduce batch size, enable gradient checkpointing, or use larger GPU instance",
            "test_timeout_failure": "Optimize test performance or increase timeout limits",
            "dependency_build_failure": "Check dependency versions and compatibility matrix",
            "k8s_deployment_failure": "Verify Kubernetes configuration and cluster resources",
            "security_failure": "Address security vulnerabilities before proceeding",
            "flaky_test_failure": "Investigate test stability and add proper waits/assertions"
        }
        
        base_recommendation = recommendations.get(failure_type, "Review logs and error details for resolution")
        
        # Add context-specific recommendations
        if failure_context.get("failure_count_last_24h", 0) > 3:
            base_recommendation += ". Consider disabling or fixing this frequently failing workflow."
        
        return base_recommendation
    
    def _get_failure_severity(self, failure_type: str) -> str:
        """Get severity level for different failure types."""
        severity_map = {
            # Critical failures
            "deployment_failure": "critical",
            "k8s_deployment_failure": "critical",
            "container_deployment_failure": "critical",
            "deployment_health_failure": "critical",
            "critical_workflow_failure": "critical",
            "infrastructure_deployment_failure": "critical",
            
            # High severity failures
            "training_failure": "high",
            "gpu_oom_training_failure": "high",
            "cuda_training_failure": "high",
            "build_failure": "high",
            "compilation_failure": "high",
            "test_failure": "high",
            "security_failure": "high",
            "recurring_failure_pattern": "high",
            
            # Medium severity failures
            "test_timeout_failure": "medium",
            "test_setup_failure": "medium",
            "code_quality_failure": "medium",
            "dependency_failure": "medium",
            "auth_failure": "medium",
            "network_failure": "medium",
            "workflow_cancelled": "medium",
            "flaky_test_failure": "medium",
            
            # Low severity failures
            "test_assertion_failure": "low",
            "permission_failure": "low",
            "linting_build_failure": "low",
            "unknown_failure": "medium"
        }
        return severity_map.get(failure_type, "medium")
    
    def _get_mock_changed_files(self, context: Context) -> List[str]:
        """Mock implementation to get changed files."""
        # In production, this would analyze the actual commit/PR changes
        return [
            "src/model/training.py",
            "config/model_config.json",
            ".github/workflows/ci.yml",
            "requirements.txt",
            "Dockerfile",
            "tests/test_model.py"
        ]
    
    def _assess_file_risk(self, file_path: str) -> str:
        """Assess risk level of changes to specific files."""
        high_risk_patterns = [
            "requirements.txt", "Dockerfile", "docker-compose.yml",
            ".github/workflows/", "config/", "setup.py", "pyproject.toml",
            "kubernetes/", "terraform/", "infrastructure/"
        ]
        
        medium_risk_patterns = [
            "src/model/", "src/training/", "src/deployment/",
            "tests/", "scripts/", "migrations/"
        ]
        
        if any(pattern in file_path for pattern in high_risk_patterns):
            return "high"
        elif any(pattern in file_path for pattern in medium_risk_patterns):
            return "medium"
        else:
            return "low"
    
    def _get_mock_error_patterns(self) -> List[str]:
        """Generate mock error patterns for testing."""
        import random
        patterns = [
            "gpu_oom", "import_error", "timeout_error", "assertion_error", 
            "compilation_error", "dependency_error", "network_error",
            "authentication_error", "cuda_error", "docker_error"
        ]
        return random.sample(patterns, k=random.randint(1, 3))
    
    def _get_mock_failed_steps(self) -> List[str]:
        """Generate mock failed steps for testing."""
        return ["setup-environment", "run-unit-tests", "build-docker-image"]
    
    def _get_mock_resource_usage(self) -> Dict[str, float]:
        """Generate mock resource usage data."""
        import random
        return {
            "memory_usage": random.uniform(0.5, 1.0),
            "cpu_usage": random.uniform(0.3, 0.9),
            "disk_usage": random.uniform(0.1, 0.8),
            "gpu_memory_usage": random.uniform(0.6, 1.0)
        }


class ErrorPatternDetector:
    """Enhanced error pattern detector for comprehensive log analysis."""
    
    PATTERNS = {
        "gpu_oom": [
            r"CUDA out of memory",
            r"RuntimeError.*out of memory",
            r"GPU memory.*insufficient",
            r"OutOfMemoryError.*GPU"
        ],
        "cuda_error": [
            r"CUDA error",
            r"CUDA runtime error",
            r"cuDNN error",
            r"NVIDIA driver error"
        ],
        "import_error": [
            r"ImportError",
            r"ModuleNotFoundError", 
            r"No module named",
            r"cannot import name"
        ],
        "dependency_error": [
            r"Could not find a version that satisfies",
            r"No matching distribution found",
            r"Package.*not found",
            r"dependency resolution failed"
        ],
        "timeout_error": [
            r"TimeoutError",
            r"Request timeout",
            r"Connection timed out",
            r"Socket timeout"
        ],
        "authentication_error": [
            r"Authentication failed",
            r"Invalid credentials",
            r"Permission denied",
            r"Access token.*invalid"
        ],
        "compilation_error": [
            r"SyntaxError",
            r"compilation terminated",
            r"fatal error",
            r"build failed"
        ],
        "docker_error": [
            r"Docker.*failed",
            r"container.*exited",
            r"image.*not found",
            r"build context.*error"
        ],
        "kubernetes_error": [
            r"kubectl.*error",
            r"pods.*failed",
            r"deployment.*failed",
            r"service.*unavailable"
        ],
        "network_error": [
            r"Connection refused",
            r"Network.*unreachable",
            r"DNS.*resolution.*failed",
            r"HTTP.*error.*[45]\d\d"
        ],
        "memory_error": [
            r"MemoryError",
            r"out of memory",
            r"memory allocation.*failed",
            r"OOMKilled"
        ],
        "disk_space_error": [
            r"No space left on device",
            r"disk.*full",
            r"storage.*exceeded",
            r"quota.*exceeded"
        ]
    }
    
    @classmethod
    def detect_patterns(cls, log_content: str) -> List[str]:
        """Detect error patterns in log content with enhanced analysis."""
        detected_patterns = []
        
        for pattern_name, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, log_content, re.IGNORECASE | re.MULTILINE):
                    detected_patterns.append(pattern_name)
                    break
        
        return detected_patterns
    
    @classmethod
    def extract_error_context(cls, log_content: str, pattern: str, context_lines: int = 3) -> str:
        """Extract context around error patterns."""
        lines = log_content.split('\n')
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                return '\n'.join(lines[start:end])
        return ""
    
    @classmethod
    def categorize_error_severity(cls, patterns: List[str]) -> str:
        """Categorize error severity based on detected patterns."""
        critical_patterns = {"gpu_oom", "cuda_error", "docker_error", "kubernetes_error"}
        high_patterns = {"compilation_error", "dependency_error", "memory_error"}
        
        if any(p in critical_patterns for p in patterns):
            return "critical"
        elif any(p in high_patterns for p in patterns):
            return "high"
        elif patterns:
            return "medium"
        else:
            return "low"