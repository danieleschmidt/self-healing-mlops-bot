"""Rollback actions for ML models and deployments."""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from .base import BaseAction, ActionResult
from ..core.context import Context

logger = logging.getLogger(__name__)


class ModelRollbackAction(BaseAction):
    """Rollback ML model to a previous stable version."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_rollback_days = self.config.get("max_rollback_days", 7)
        self.rollback_threshold = self.config.get("rollback_threshold", 0.15)  # 15% degradation
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["model_degradation", "deployment_failure", "performance_issue"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Execute model rollback."""
        try:
            degradation_pct = issue_data.get("degradation_percentage", 0)
            metric_name = issue_data.get("metric_name", "unknown")
            
            # Check if rollback is warranted
            if degradation_pct < self.rollback_threshold * 100:
                return self.create_result(
                    success=False,
                    message=f"Degradation ({degradation_pct:.1f}%) below rollback threshold ({self.rollback_threshold * 100:.1f}%)"
                )
            
            # Find stable model version
            stable_version = await self._find_stable_model_version(context, issue_data)
            
            if not stable_version:
                return self.create_result(
                    success=False,
                    message="No stable model version found for rollback"
                )
            
            # Perform rollback
            rollback_result = await self._perform_model_rollback(context, stable_version, issue_data)
            
            if rollback_result["success"]:
                return self.create_result(
                    success=True,
                    message=f"Successfully rolled back model to version {stable_version['version']}",
                    data={
                        "rolled_back_to": stable_version,
                        "reason": f"{metric_name} degraded by {degradation_pct:.1f}%",
                        "rollback_details": rollback_result
                    }
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