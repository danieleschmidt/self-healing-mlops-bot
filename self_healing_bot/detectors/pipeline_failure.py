"""Pipeline and CI/CD failure detection."""

from typing import List, Dict, Any
import re
import logging

from .base import BaseDetector
from ..core.context import Context

logger = logging.getLogger(__name__)


class PipelineFailureDetector(BaseDetector):
    """Detect pipeline and CI/CD failures."""
    
    def get_supported_events(self) -> List[str]:
        return ["workflow_run", "check_run", "status"]
    
    async def detect(self, context: Context) -> List[Dict[str, Any]]:
        """Detect pipeline failures and categorize them."""
        issues = []
        
        if context.event_type == "workflow_run":
            issues.extend(await self._detect_workflow_failures(context))
        elif context.event_type == "check_run":
            issues.extend(await self._detect_check_failures(context))
        elif context.event_type == "status":
            issues.extend(await self._detect_status_failures(context))
        
        return issues
    
    async def _detect_workflow_failures(self, context: Context) -> List[Dict[str, Any]]:
        """Detect GitHub Actions workflow failures."""
        issues = []
        
        workflow_data = context.event_data.get("workflow_run", {})
        conclusion = workflow_data.get("conclusion")
        
        if conclusion == "failure":
            # Categorize failure type based on workflow name and logs
            failure_type = await self._categorize_workflow_failure(context, workflow_data)
            
            issues.append(self.create_issue(
                issue_type=f"workflow_failure_{failure_type}",
                severity=self._get_failure_severity(failure_type),
                message=f"Workflow '{workflow_data.get('name', 'unknown')}' failed with {failure_type}",
                data={
                    "workflow_id": workflow_data.get("id"),
                    "workflow_name": workflow_data.get("name"),
                    "failure_type": failure_type,
                    "run_url": workflow_data.get("html_url"),
                    "head_sha": workflow_data.get("head_sha")
                }
            ))
        
        elif conclusion == "cancelled":
            issues.append(self.create_issue(
                issue_type="workflow_cancelled",
                severity="medium",
                message=f"Workflow '{workflow_data.get('name', 'unknown')}' was cancelled",
                data={
                    "workflow_id": workflow_data.get("id"),
                    "workflow_name": workflow_data.get("name")
                }
            ))
        
        return issues
    
    async def _categorize_workflow_failure(self, context: Context, workflow_data: Dict[str, Any]) -> str:
        """Categorize the type of workflow failure."""
        workflow_name = workflow_data.get("name", "").lower()
        
        # Analyze workflow name patterns
        if any(keyword in workflow_name for keyword in ["test", "ci", "check"]):
            return "test_failure"
        elif any(keyword in workflow_name for keyword in ["build", "compile"]):
            return "build_failure"
        elif any(keyword in workflow_name for keyword in ["deploy", "release"]):
            return "deployment_failure"
        elif any(keyword in workflow_name for keyword in ["train", "ml", "model"]):
            return "training_failure"
        elif any(keyword in workflow_name for keyword in ["lint", "format", "style"]):
            return "code_quality_failure"
        
        # TODO: In a real implementation, we would fetch and analyze the logs
        # For now, return a generic failure type
        return "unknown_failure"
    
    async def _detect_check_failures(self, context: Context) -> List[Dict[str, Any]]:
        """Detect check run failures."""
        issues = []
        
        check_run = context.event_data.get("check_run", {})
        conclusion = check_run.get("conclusion")
        
        if conclusion == "failure":
            issues.append(self.create_issue(
                issue_type="check_failure",
                severity="medium",
                message=f"Check '{check_run.get('name', 'unknown')}' failed",
                data={
                    "check_id": check_run.get("id"),
                    "check_name": check_run.get("name"),
                    "details_url": check_run.get("details_url")
                }
            ))
        
        return issues
    
    async def _detect_status_failures(self, context: Context) -> List[Dict[str, Any]]:
        """Detect status check failures."""
        issues = []
        
        status_data = context.event_data
        state = status_data.get("state")
        
        if state == "failure":
            issues.append(self.create_issue(
                issue_type="status_failure",
                severity="medium",
                message=f"Status check '{status_data.get('context', 'unknown')}' failed",
                data={
                    "status_context": status_data.get("context"),
                    "description": status_data.get("description"),
                    "target_url": status_data.get("target_url")
                }
            ))
        
        return issues
    
    def _get_failure_severity(self, failure_type: str) -> str:
        """Get severity level for different failure types."""
        severity_map = {
            "deployment_failure": "critical",
            "training_failure": "high",
            "test_failure": "high",
            "build_failure": "high",
            "code_quality_failure": "medium",
            "unknown_failure": "medium"
        }
        return severity_map.get(failure_type, "medium")


class ErrorPatternDetector:
    """Helper class to detect common error patterns in logs."""
    
    PATTERNS = {
        "gpu_oom": [
            r"CUDA out of memory",
            r"RuntimeError.*out of memory",
            r"GPU memory.*insufficient"
        ],
        "import_error": [
            r"ImportError",
            r"ModuleNotFoundError",
            r"No module named"
        ],
        "dependency_error": [
            r"Could not find a version that satisfies",
            r"No matching distribution found",
            r"Package.*not found"
        ],
        "timeout_error": [
            r"TimeoutError",
            r"Request timeout",
            r"Connection timed out"
        ],
        "authentication_error": [
            r"Authentication failed",
            r"Invalid credentials",
            r"Permission denied"
        ]
    }
    
    @classmethod
    def detect_patterns(cls, log_content: str) -> List[str]:
        """Detect error patterns in log content."""
        detected_patterns = []
        
        for pattern_name, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, log_content, re.IGNORECASE):
                    detected_patterns.append(pattern_name)
                    break
        
        return detected_patterns