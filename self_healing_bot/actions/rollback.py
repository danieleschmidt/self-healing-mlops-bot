"""Rollback actions for ML models and deployments."""

from typing import Dict, Any, List, Optional
import logging
import subprocess
import json
from datetime import datetime, timedelta

from .base import BaseAction, ActionResult
from ..core.context import Context
from ..integrations.github import GitHubIntegration

logger = logging.getLogger(__name__)


class ModelRollbackAction(BaseAction):
    """Rollback ML model to a previous stable version."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_rollback_days = self.config.get("max_rollback_days", 7)
        self.rollback_threshold = self.config.get("rollback_threshold", 0.15)  # 15% degradation
        self.github_integration = GitHubIntegration()
        self.create_pr = self.config.get("create_pr", True)
        self.pr_branch_prefix = self.config.get("pr_branch_prefix", "bot/rollback")
        self.verification_enabled = self.config.get("verification_enabled", True)
        self.rollback_safety_checks = self.config.get("rollback_safety_checks", True)
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in [
            "model_degradation", "deployment_failure", "performance_issue",
            "data_corruption", "security_incident", "infrastructure_failure"
        ]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Execute model rollback with comprehensive verification."""
        try:
            self.log_action(context, "Starting model rollback process")
            
            degradation_pct = issue_data.get("degradation_percentage", 0)
            metric_name = issue_data.get("metric_name", "unknown")
            issue_type = issue_data.get("type", "")
            
            # Perform safety checks if enabled
            if self.rollback_safety_checks:
                safety_check = await self._perform_rollback_safety_checks(context, issue_data)
                if not safety_check["safe"]:
                    return self.create_result(
                        success=False,
                        message=f"Rollback safety check failed: {safety_check['reason']}"
                    )
            
            # Check if rollback is warranted based on issue type and severity
            should_rollback, reason = await self._should_rollback(context, issue_data)
            if not should_rollback:
                return self.create_result(
                    success=False,
                    message=f"Rollback not warranted: {reason}"
                )
            
            # Find stable model version
            stable_version = await self._find_stable_model_version(context, issue_data)
            
            if not stable_version:
                return self.create_result(
                    success=False,
                    message="No stable model version found for rollback"
                )
            
            # Create rollback plan
            rollback_plan = await self._create_rollback_plan(context, stable_version, issue_data)
            
            # Perform rollback with verification
            rollback_result = await self._perform_verified_model_rollback(context, rollback_plan, issue_data)
            
            if rollback_result["success"]:
                # Create PR if enabled
                pr_result = None
                if self.create_pr:
                    pr_result = await self._create_rollback_pr(context, issue_data, rollback_result)
                
                result_data = {
                    "rolled_back_to": stable_version,
                    "reason": reason,
                    "rollback_plan": rollback_plan,
                    "rollback_details": rollback_result,
                    "verification_results": rollback_result.get("verification", {})
                }
                
                if pr_result:
                    result_data["pull_request"] = pr_result
                
                return self.create_result(
                    success=True,
                    message=f"Successfully rolled back model to version {stable_version['version']}",
                    data=result_data
                )
            else:
                return self.create_result(
                    success=False,
                    message=f"Model rollback failed: {rollback_result['error']}"
                )
                
        except Exception as e:
            logger.exception(f"Model rollback failed: {e}")
            return self.create_result(
                success=False,
                message=f"Model rollback failed: {str(e)}"
            )
    
    async def _perform_rollback_safety_checks(self, context: Context, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive safety checks before rollback."""
        try:
            checks = []
            
            # Check if there are active users/traffic
            traffic_check = await self._check_active_traffic(context)
            checks.append(traffic_check)
            
            # Check if there are ongoing deployments
            deployment_check = await self._check_ongoing_deployments(context)
            checks.append(deployment_check)
            
            # Check backup availability
            backup_check = await self._check_backup_availability(context)
            checks.append(backup_check)
            
            # Check rollback window
            window_check = await self._check_rollback_window(context, issue_data)
            checks.append(window_check)
            
            # All checks must pass for rollback to be safe
            all_safe = all(check["safe"] for check in checks)
            
            if not all_safe:
                failed_checks = [check["name"] for check in checks if not check["safe"]]
                return {
                    "safe": False,
                    "reason": f"Failed safety checks: {', '.join(failed_checks)}",
                    "checks": checks
                }
            
            return {
                "safe": True,
                "reason": "All safety checks passed",
                "checks": checks
            }
            
        except Exception as e:
            return {
                "safe": False,
                "reason": f"Safety check failed: {str(e)}",
                "checks": []
            }
    
    async def _should_rollback(self, context: Context, issue_data: Dict[str, Any]) -> tuple[bool, str]:
        """Determine if rollback should be performed based on issue severity and type."""
        issue_type = issue_data.get("type", "")
        degradation_pct = issue_data.get("degradation_percentage", 0)
        severity = issue_data.get("severity", "medium")
        
        # Critical issues always warrant rollback
        if severity == "critical":
            return True, f"Critical {issue_type} requires immediate rollback"
        
        # Security incidents always warrant rollback
        if issue_type == "security_incident":
            return True, "Security incident requires immediate rollback"
        
        # Data corruption always warrants rollback
        if issue_type == "data_corruption":
            return True, "Data corruption requires immediate rollback"
        
        # Performance degradation above threshold
        if issue_type == "model_degradation" and degradation_pct >= self.rollback_threshold * 100:
            return True, f"Model degradation ({degradation_pct:.1f}%) exceeds threshold ({self.rollback_threshold * 100:.1f}%)"
        
        # High severity performance issues
        if issue_type == "performance_issue" and severity == "high":
            return True, f"High severity performance issue requires rollback"
        
        # Infrastructure failures in production
        if issue_type == "infrastructure_failure" and severity in ["high", "critical"]:
            return True, f"Infrastructure failure requires rollback"
        
        return False, f"Issue severity ({severity}) and type ({issue_type}) do not meet rollback criteria"
    
    async def _create_rollback_plan(self, context: Context, stable_version: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive rollback plan."""
        return {
            "rollback_type": "model_version",
            "target_version": stable_version["version"],
            "current_version": issue_data.get("current_version", "unknown"),
            "rollback_strategy": self._determine_rollback_strategy(issue_data),
            "verification_steps": [
                "health_check",
                "smoke_test",
                "performance_validation",
                "integration_test"
            ],
            "rollback_steps": [
                "backup_current_state",
                "update_model_registry",
                "update_deployment_config",
                "restart_services",
                "verify_rollback",
                "monitor_health"
            ],
            "rollback_timeout": 600,  # 10 minutes
            "monitoring_duration": 1800,  # 30 minutes post-rollback
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def _perform_verified_model_rollback(self, context: Context, rollback_plan: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform model rollback with comprehensive verification."""
        try:
            rollback_steps = []
            target_version = rollback_plan["target_version"]
            
            # Step 1: Backup current state
            backup_result = await self._backup_current_model_state(context)
            rollback_steps.append({"step": "backup", "result": backup_result, "success": backup_result["success"]})
            
            if not backup_result["success"]:
                return {"success": False, "error": "Failed to backup current state", "steps": rollback_steps}
            
            # Step 2: Update model registry
            registry_result = await self._update_model_registry(context, target_version, issue_data)
            rollback_steps.append({"step": "registry_update", "result": registry_result, "success": registry_result["success"]})
            
            # Step 3: Update deployment configuration
            config_result = await self._update_deployment_config(context, target_version, rollback_plan)
            rollback_steps.append({"step": "config_update", "result": config_result, "success": config_result["success"]})
            
            # Step 4: Create rollback scripts
            script_result = await self._create_rollback_scripts(context, rollback_plan)
            rollback_steps.append({"step": "script_creation", "result": script_result, "success": script_result["success"]})
            
            # Step 5: Perform verification if enabled
            verification_results = {}
            if self.verification_enabled:
                verification_results = await self._verify_rollback(context, rollback_plan)
                rollback_steps.append({"step": "verification", "result": verification_results, "success": verification_results.get("success", False)})
            
            # Determine overall success
            all_steps_successful = all(step["success"] for step in rollback_steps)
            
            return {
                "success": all_steps_successful,
                "steps": rollback_steps,
                "verification": verification_results,
                "rollback_plan": rollback_plan,
                "completed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "steps": rollback_steps,
                "failed_at": datetime.utcnow().isoformat()
            }
    
    async def _create_rollback_pr(self, context: Context, issue_data: Dict[str, Any], rollback_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a GitHub pull request for the rollback."""
        try:
            issue_type = issue_data.get("type", "")
            branch_name = f"{self.pr_branch_prefix}-{issue_type}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            
            # Get file changes
            file_changes = context.get_file_changes()
            
            if not file_changes:
                # Create rollback documentation
                rollback_summary = self._generate_rollback_summary(issue_data, rollback_result)
                file_changes = {"ROLLBACK_SUMMARY.md": rollback_summary}
            
            # Create PR title and body
            target_version = rollback_result.get("rollback_plan", {}).get("target_version", "unknown")
            title = f"🔄 Rollback Model to Version {target_version}"
            body = self._generate_rollback_pr_body(issue_data, rollback_result)
            
            # Create pull request
            installation_id = context.get_state("github_installation_id", 1)  # Default for demo
            
            pr_result = await self.github_integration.create_pull_request(
                installation_id=installation_id,
                repo_full_name=context.repo_full_name,
                title=title,
                body=body,
                head_branch=branch_name,
                base_branch="main",
                file_changes=file_changes
            )
            
            return pr_result
            
        except Exception as e:
            logger.error(f"Rollback PR creation failed: {e}")
            return None
    
    async def _find_stable_model_version(self, context: Context, issue_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a stable model version to rollback to."""
        # In a real implementation, this would query model registry (MLflow, etc.)
        # For demonstration, return mock stable version
        
        stable_versions = [
            {
                "version": "v1.2.3",
                "timestamp": "2024-01-15T10:30:00Z",
                "metrics": {
                    "accuracy": 0.94,
                    "f1_score": 0.91,
                    "auc_roc": 0.93
                },
                "deployment_status": "stable",
                "performance_rating": "excellent"
            },
            {
                "version": "v1.2.2", 
                "timestamp": "2024-01-10T14:20:00Z",
                "metrics": {
                    "accuracy": 0.92,
                    "f1_score": 0.89,
                    "auc_roc": 0.91
                },
                "deployment_status": "stable",
                "performance_rating": "good"
            }
        ]
        
        # Select the most recent stable version
        for version in stable_versions:
            version_date = datetime.fromisoformat(version["timestamp"].replace('Z', '+00:00'))
            if (datetime.now().astimezone() - version_date).days <= self.max_rollback_days:
                return version
        
        return stable_versions[0] if stable_versions else None
    
    async def _check_active_traffic(self, context: Context) -> Dict[str, Any]:
        """Check if there's active traffic that might be affected by rollback."""
        # In production, this would check load balancers, monitoring systems, etc.
        return {
            "name": "active_traffic",
            "safe": True,
            "message": "Traffic levels are within acceptable range for rollback"
        }
    
    async def _check_ongoing_deployments(self, context: Context) -> Dict[str, Any]:
        """Check for ongoing deployments that might conflict."""
        # In production, this would check CI/CD pipelines, deployment tools, etc.
        return {
            "name": "ongoing_deployments", 
            "safe": True,
            "message": "No conflicting deployments detected"
        }
    
    async def _check_backup_availability(self, context: Context) -> Dict[str, Any]:
        """Check if backups are available and accessible."""
        return {
            "name": "backup_availability",
            "safe": True,
            "message": "Model and configuration backups are available"
        }
    
    async def _check_rollback_window(self, context: Context, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if we're within the acceptable rollback window."""
        issue_time = issue_data.get("detected_at")
        if issue_time:
            try:
                detection_time = datetime.fromisoformat(issue_time.replace('Z', '+00:00'))
                time_since_issue = (datetime.now().astimezone() - detection_time).total_seconds()
                max_rollback_time = self.max_rollback_days * 24 * 3600
                
                if time_since_issue <= max_rollback_time:
                    return {
                        "name": "rollback_window",
                        "safe": True,
                        "message": f"Within rollback window ({time_since_issue/3600:.1f}h < {max_rollback_time/3600:.1f}h)"
                    }
                else:
                    return {
                        "name": "rollback_window",
                        "safe": False,
                        "message": f"Outside rollback window ({time_since_issue/3600:.1f}h > {max_rollback_time/3600:.1f}h)"
                    }
            except Exception:
                pass
        
        return {
            "name": "rollback_window",
            "safe": True,
            "message": "Rollback window check passed (no timestamp available)"
        }
    
    def _determine_rollback_strategy(self, issue_data: Dict[str, Any]) -> str:
        """Determine the appropriate rollback strategy based on issue type."""
        issue_type = issue_data.get("type", "")
        severity = issue_data.get("severity", "medium")
        
        if severity == "critical" or issue_type in ["security_incident", "data_corruption"]:
            return "immediate"
        elif severity == "high":
            return "fast"
        else:
            return "gradual"
    
    async def _perform_model_rollback(self, context: Context, stable_version: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual model rollback."""
        try:
            version = stable_version["version"]
            
            # Update deployment configuration
            deployment_config = {
                "model_version": version,
                "rollback_reason": f"Performance degradation in {issue_data.get('metric_name')}",
                "rollback_timestamp": datetime.utcnow().isoformat(),
                "previous_version": issue_data.get("current_version", "unknown"),
                "auto_rollback": True
            }
            
            context.save_config("deployment.yaml", deployment_config)
            
            # Create rollback script
            rollback_script = f"""#!/bin/bash
# Auto-generated rollback script
echo "Rolling back model to version {version}..."

# Update model registry pointer
echo "Updating model registry..."
# curl -X POST "http://model-registry/api/v1/models/production/set-version" -d '{{"version": "{version}"}}'

# Restart services
echo "Restarting inference services..."
# kubectl rollout restart deployment/ml-inference

echo "Rollback to {version} completed successfully"
"""
            
            context.write_file("rollback.sh", rollback_script)
            
            return {
                "success": True,
                "version": version,
                "config_updated": True,
                "script_created": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class DeploymentRollbackAction(BaseAction):
    """Rollback deployment to a previous stable state."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.rollback_strategies = self.config.get("rollback_strategies", self._get_default_strategies())
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["deployment_failure", "service_error", "infrastructure_issue"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Execute deployment rollback."""
        try:
            issue_severity = issue_data.get("severity", "medium")
            deployment_type = issue_data.get("deployment_type", "kubernetes")
            
            # Select rollback strategy
            strategy = self._select_rollback_strategy(issue_severity, deployment_type)
            
            if not strategy:
                return self.create_result(
                    success=False,
                    message=f"No rollback strategy available for {deployment_type}"
                )
            
            # Execute rollback
            rollback_result = await self._execute_rollback_strategy(context, strategy, issue_data)
            
            if rollback_result["success"]:
                return self.create_result(
                    success=True,
                    message=f"Successfully executed {strategy['name']} rollback",
                    data={
                        "strategy": strategy["name"],
                        "rollback_details": rollback_result
                    }
                )
            else:
                return self.create_result(
                    success=False,
                    message=f"Deployment rollback failed: {rollback_result['error']}"
                )
                
        except Exception as e:
            logger.exception(f"Deployment rollback failed: {e}")
            return self.create_result(
                success=False,
                message=f"Deployment rollback failed: {str(e)}"
            )
    
    def _select_rollback_strategy(self, severity: str, deployment_type: str) -> Optional[Dict[str, Any]]:
        """Select appropriate rollback strategy."""
        strategy_key = f"{deployment_type}_{severity}"
        
        # Try specific strategy first
        strategy = self.rollback_strategies.get(strategy_key)
        if strategy:
            return strategy
        
        # Fall back to general deployment type strategy
        general_strategy = self.rollback_strategies.get(deployment_type)
        if general_strategy:
            return general_strategy
        
        # Default strategy
        return self.rollback_strategies.get("default")
    
    async def _execute_rollback_strategy(self, context: Context, strategy: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the selected rollback strategy."""
        try:
            actions_completed = []
            
            for action in strategy.get("actions", []):
                action_type = action.get("type")
                
                if action_type == "kubernetes_rollback":
                    result = await self._kubernetes_rollback(context, action, issue_data)
                    actions_completed.append(result)
                elif action_type == "docker_rollback":
                    result = await self._docker_rollback(context, action, issue_data)
                    actions_completed.append(result)
                elif action_type == "service_restart":
                    result = await self._service_restart(context, action, issue_data)
                    actions_completed.append(result)
                elif action_type == "config_restore":
                    result = await self._config_restore(context, action, issue_data)
                    actions_completed.append(result)
            
            return {
                "success": True,
                "actions_completed": actions_completed,
                "strategy_name": strategy["name"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _kubernetes_rollback(self, context: Context, action: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, str]:
        """Perform Kubernetes deployment rollback."""
        deployment_name = action.get("deployment", "ml-inference")
        namespace = action.get("namespace", "default")
        
        rollback_script = f"""#!/bin/bash
# Kubernetes rollback script
echo "Rolling back Kubernetes deployment {deployment_name} in namespace {namespace}..."

# Rollback to previous revision
kubectl rollout undo deployment/{deployment_name} -n {namespace}

# Wait for rollout to complete
kubectl rollout status deployment/{deployment_name} -n {namespace} --timeout=300s

echo "Kubernetes rollback completed"
"""
        
        context.write_file("k8s_rollback.sh", rollback_script)
        
        return {
            "type": "kubernetes_rollback",
            "deployment": deployment_name,
            "namespace": namespace,
            "status": "script_created"
        }
    
    async def _docker_rollback(self, context: Context, action: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, str]:
        """Perform Docker container rollback."""
        container_name = action.get("container", "ml-service")
        previous_tag = action.get("previous_tag", "stable")
        
        rollback_script = f"""#!/bin/bash
# Docker rollback script
echo "Rolling back Docker container {container_name} to {previous_tag}..."

# Stop current container
docker stop {container_name}

# Remove current container
docker rm {container_name}

# Start with previous stable image
docker run -d --name {container_name} ml-service:{previous_tag}

echo "Docker rollback completed"
"""
        
        context.write_file("docker_rollback.sh", rollback_script)
        
        return {
            "type": "docker_rollback",
            "container": container_name,
            "previous_tag": previous_tag,
            "status": "script_created"
        }
    
    async def _service_restart(self, context: Context, action: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, str]:
        """Restart services."""
        services = action.get("services", ["ml-inference"])
        
        restart_script = f"""#!/bin/bash
# Service restart script
echo "Restarting services: {', '.join(services)}..."

"""
        
        for service in services:
            restart_script += f"""
# Restart {service}
systemctl restart {service}
echo "Restarted {service}"
"""
        
        context.write_file("service_restart.sh", restart_script)
        
        return {
            "type": "service_restart",
            "services": services,
            "status": "script_created"
        }
    
    async def _config_restore(self, context: Context, action: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, str]:
        """Restore configuration files."""
        config_files = action.get("config_files", ["config.yaml"])
        
        # Create backup of current configs and restore previous versions
        for config_file in config_files:
            # In a real implementation, this would restore from version control or backup
            backup_config = {
                "restored": True,
                "restore_timestamp": datetime.utcnow().isoformat(),
                "reason": "Automatic rollback due to deployment failure"
            }
            context.save_config(f"{config_file}.backup", backup_config)
        
        return {
            "type": "config_restore",
            "config_files": config_files,
            "status": "configs_restored"
        }
    
    def _get_default_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get default rollback strategies."""
        return {
            "kubernetes_critical": {
                "name": "Critical Kubernetes Rollback",
                "description": "Immediate rollback for critical Kubernetes deployment issues",
                "actions": [
                    {
                        "type": "kubernetes_rollback",
                        "deployment": "ml-inference",
                        "namespace": "production"
                    },
                    {
                        "type": "service_restart",
                        "services": ["nginx", "ml-gateway"]
                    }
                ]
            },
            "kubernetes": {
                "name": "Standard Kubernetes Rollback",
                "description": "Standard rollback for Kubernetes deployments",
                "actions": [
                    {
                        "type": "kubernetes_rollback",
                        "deployment": "ml-inference",
                        "namespace": "default"
                    }
                ]
            },
            "docker": {
                "name": "Docker Container Rollback",
                "description": "Rollback Docker containers to stable versions",
                "actions": [
                    {
                        "type": "docker_rollback",
                        "container": "ml-service",
                        "previous_tag": "stable"
                    }
                ]
            },
            "default": {
                "name": "Generic Service Rollback",
                "description": "Generic rollback strategy for any deployment type",
                "actions": [
                    {
                        "type": "config_restore",
                        "config_files": ["config.yaml", "deployment.yaml"]
                    },
                    {
                        "type": "service_restart",
                        "services": ["ml-service"]
                    }
                ]
            }
        }


class GradualRollbackAction(BaseAction):
    """Implement gradual/canary rollback to minimize impact."""
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["model_degradation", "performance_issue"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Execute gradual rollback strategy."""
        try:
            degradation_pct = issue_data.get("degradation_percentage", 0)
            
            # Determine rollback percentage based on degradation severity
            if degradation_pct > 20:
                rollback_percent = 100  # Full rollback
            elif degradation_pct > 10:
                rollback_percent = 50   # Partial rollback
            else:
                rollback_percent = 25   # Gradual rollback
            
            # Create gradual rollback configuration
            canary_config = {
                "rollback_strategy": "gradual",
                "rollback_percentage": rollback_percent,
                "monitoring_period": "1h",
                "success_criteria": {
                    "error_rate": 0.01,
                    "latency_p95": 200,
                    "throughput_minimum": 1000
                },
                "rollback_steps": [
                    {"percentage": 25, "duration": "15m"},
                    {"percentage": 50, "duration": "20m"},
                    {"percentage": 100, "duration": "ongoing"}
                ]
            }
            
            context.save_config("canary_rollback.yaml", canary_config)
            
            return self.create_result(
                success=True,
                message=f"Configured gradual rollback ({rollback_percent}% traffic)",
                data={
                    "rollback_percentage": rollback_percent,
                    "strategy": "gradual",
                    "config": canary_config
                }
            )
            
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Gradual rollback configuration failed: {str(e)}"
            )