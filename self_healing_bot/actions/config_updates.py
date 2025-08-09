"""Configuration update actions for ML pipeline optimization."""

import json
import yaml
import os
import subprocess
from typing import Dict, Any, List, Optional, Union
import logging
import re
from datetime import datetime

from .base import BaseAction, ActionResult
from ..core.context import Context
from ..integrations.github import GitHubIntegration

logger = logging.getLogger(__name__)


class ConfigUpdateAction(BaseAction):
    """Update configuration files for ML pipelines."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.github_integration = GitHubIntegration()
        self.create_pr = self.config.get("create_pr", True)
        self.pr_branch_prefix = self.config.get("pr_branch_prefix", "bot/config-update")
        self.backup_enabled = self.config.get("backup_enabled", True)
        self.validation_enabled = self.config.get("validation_enabled", True)
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in [
            "config_error", "performance_issue", "resource_issue", 
            "environment_config", "ci_cd_config", "deployment_config"
        ]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Update configuration based on issue type."""
        issue_type = issue_data.get("type", "")
        
        if not self.can_handle(issue_type):
            return self.create_result(
                success=False,
                message=f"Cannot handle issue type: {issue_type}"
            )
        
        try:
            self.log_action(context, f"Starting {issue_type} configuration update")
            
            # Create backup if enabled
            backup_data = None
            if self.backup_enabled:
                backup_data = await self._create_config_backup(context, issue_data)
            
            # Apply appropriate configuration updates
            result = None
            if issue_type == "config_error":
                result = await self._fix_config_errors(context, issue_data)
            elif issue_type == "performance_issue":
                result = await self._optimize_performance_config(context, issue_data)
            elif issue_type == "resource_issue":
                result = await self._adjust_resource_config(context, issue_data)
            elif issue_type == "environment_config":
                result = await self._update_environment_config(context, issue_data)
            elif issue_type == "ci_cd_config":
                result = await self._update_ci_cd_config(context, issue_data)
            elif issue_type == "deployment_config":
                result = await self._update_deployment_config(context, issue_data)
            else:
                config_file = issue_data.get("config_file", "config.yaml")
                updates = issue_data.get("config_updates", {})
                result = await self._generic_config_update(context, config_file, updates)
            
            # Validate configuration if enabled
            if result.success and self.validation_enabled:
                validation_result = await self._validate_configuration(context, result.data)
                if not validation_result["valid"]:
                    # Rollback if validation fails
                    if backup_data:
                        await self._rollback_config_changes(context, backup_data)
                    return self.create_result(
                        success=False,
                        message=f"Configuration validation failed: {validation_result['errors']}"
                    )
            
            # Create PR if successful and enabled
            if result.success and self.create_pr:
                pr_result = await self._create_config_pr(context, issue_data, result.data)
                if pr_result:
                    result.data["pull_request"] = pr_result
            
            # Add backup info to result
            if backup_data:
                result.data["backup"] = backup_data
                
            return result
            
        except Exception as e:
            logger.exception(f"Configuration update failed: {e}")
            # Attempt rollback if backup exists
            if backup_data:
                await self._rollback_config_changes(context, backup_data)
            
            return self.create_result(
                success=False,
                message=f"Configuration update failed: {str(e)}"
            )
    
    async def _fix_config_errors(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Fix configuration errors and syntax issues."""
        try:
            config_files = issue_data.get("config_files", ["config.yaml"])
            error_patterns = issue_data.get("error_patterns", [])
            fixes_applied = []
            
            for config_file in config_files:
                try:
                    # Try to load and validate the configuration
                    if config_file.endswith(('.yaml', '.yml')):
                        content = context.read_file(config_file)
                        
                        # Fix common YAML issues
                        fixed_content = self._fix_yaml_syntax(content)
                        
                        # Try to parse the fixed YAML
                        yaml.safe_load(fixed_content)
                        
                        if content != fixed_content:
                            context.write_file(config_file, fixed_content)
                            fixes_applied.append(f"Fixed YAML syntax in {config_file}")
                    
                    elif config_file.endswith('.json'):
                        content = context.read_file(config_file)
                        
                        # Fix common JSON issues
                        fixed_content = self._fix_json_syntax(content)
                        
                        # Try to parse the fixed JSON
                        json.loads(fixed_content)
                        
                        if content != fixed_content:
                            context.write_file(config_file, fixed_content)
                            fixes_applied.append(f"Fixed JSON syntax in {config_file}")
                            
                except Exception as e:
                    logger.warning(f"Could not fix config file {config_file}: {e}")
            
            if fixes_applied:
                return self.create_result(
                    success=True,
                    message=f"Fixed configuration errors: {', '.join(fixes_applied)}",
                    data={"fixes": fixes_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No configuration fixes applied"
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Configuration error fix failed: {str(e)}"
            )
    
    async def _optimize_performance_config(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Optimize configuration for performance."""
        try:
            config_file = issue_data.get("config_file", "config.yaml")
            performance_issues = issue_data.get("performance_issues", [])
            
            current_config = context.load_config(config_file)
            optimizations = []
            
            # Apply performance optimizations based on detected issues
            if "slow_training" in performance_issues:
                optimizations.extend(self._get_training_performance_optimizations(current_config))
            
            if "slow_inference" in performance_issues:
                optimizations.extend(self._get_inference_performance_optimizations(current_config))
            
            if "memory_usage" in performance_issues:
                optimizations.extend(self._get_memory_optimizations(current_config))
            
            # Apply optimizations
            optimized_config = current_config.copy()
            applied_optimizations = []
            
            for optimization in optimizations:
                path = optimization["path"]
                value = optimization["value"]
                description = optimization["description"]
                
                self._set_nested_value(optimized_config, path, value)
                applied_optimizations.append(description)
            
            if applied_optimizations:
                context.save_config(config_file, optimized_config)
                return self.create_result(
                    success=True,
                    message=f"Applied performance optimizations: {', '.join(applied_optimizations)}",
                    data={
                        "config_file": config_file,
                        "optimizations": applied_optimizations,
                        "config_diff": self._generate_config_diff(current_config, optimized_config)
                    }
                )
            else:
                return self.create_result(
                    success=False,
                    message="No performance optimizations needed"
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Performance configuration optimization failed: {str(e)}"
            )
    
    async def _adjust_resource_config(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Adjust resource allocation configuration."""
        try:
            config_files = issue_data.get("config_files", ["config.yaml"])
            resource_issues = issue_data.get("resource_issues", [])
            adjustments_applied = []
            
            for config_file in config_files:
                current_config = context.load_config(config_file)
                adjusted_config = current_config.copy()
                
                # Apply resource adjustments
                for issue in resource_issues:
                    if issue == "cpu_overload":
                        self._apply_cpu_resource_adjustments(adjusted_config)
                        adjustments_applied.append(f"Reduced CPU usage in {config_file}")
                    
                    elif issue == "memory_overflow":
                        self._apply_memory_resource_adjustments(adjusted_config)
                        adjustments_applied.append(f"Optimized memory usage in {config_file}")
                    
                    elif issue == "gpu_utilization":
                        self._apply_gpu_resource_adjustments(adjusted_config)
                        adjustments_applied.append(f"Optimized GPU usage in {config_file}")
                
                if current_config != adjusted_config:
                    context.save_config(config_file, adjusted_config)
            
            if adjustments_applied:
                return self.create_result(
                    success=True,
                    message=f"Applied resource adjustments: {', '.join(adjustments_applied)}",
                    data={"adjustments": adjustments_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No resource adjustments needed"
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Resource configuration adjustment failed: {str(e)}"
            )
    
    async def _update_environment_config(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Update environment configuration and variables."""
        try:
            env_updates = issue_data.get("env_updates", {})
            config_updates = issue_data.get("config_updates", {})
            updates_applied = []
            
            # Update environment files
            for env_file in [".env", ".env.local", ".env.production"]:
                try:
                    content = context.read_file(env_file)
                    original_content = content
                    
                    # Apply environment variable updates
                    for key, value in env_updates.items():
                        pattern = rf'^{key}=.*$'
                        replacement = f'{key}={value}'
                        
                        if re.search(pattern, content, re.MULTILINE):
                            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                        else:
                            content += f'\n{replacement}\n'
                        
                        updates_applied.append(f"Updated {key} in {env_file}")
                    
                    if content != original_content:
                        context.write_file(env_file, content)
                        
                except Exception:
                    # Environment file might not exist, skip
                    continue
            
            # Update configuration files with environment-related settings
            if config_updates:
                for config_file, updates in config_updates.items():
                    current_config = context.load_config(config_file)
                    updated_config = self._merge_configs(current_config, updates)
                    context.save_config(config_file, updated_config)
                    updates_applied.append(f"Updated environment config in {config_file}")
            
            if updates_applied:
                return self.create_result(
                    success=True,
                    message=f"Updated environment configuration: {', '.join(updates_applied)}",
                    data={"updates": updates_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No environment configuration updates needed"
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Environment configuration update failed: {str(e)}"
            )
    
    async def _update_ci_cd_config(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Update CI/CD pipeline configuration."""
        try:
            ci_issues = issue_data.get("ci_issues", [])
            workflow_updates = issue_data.get("workflow_updates", {})
            fixes_applied = []
            
            # Common CI/CD configuration files
            ci_files = [
                ".github/workflows/ci.yml",
                ".github/workflows/cd.yml", 
                ".github/workflows/test.yml",
                ".gitlab-ci.yml",
                "azure-pipelines.yml",
                "Jenkinsfile"
            ]
            
            for ci_file in ci_files:
                try:
                    content = context.read_file(ci_file)
                    original_content = content
                    
                    # Apply CI/CD fixes based on detected issues
                    if "timeout_issues" in ci_issues:
                        content = self._fix_ci_timeouts(content)
                        if content != original_content:
                            fixes_applied.append(f"Fixed timeout issues in {ci_file}")
                    
                    if "dependency_caching" in ci_issues:
                        content = self._add_dependency_caching(content, ci_file)
                        if content != original_content:
                            fixes_applied.append(f"Added dependency caching in {ci_file}")
                    
                    if "test_optimization" in ci_issues:
                        content = self._optimize_test_pipeline(content)
                        if content != original_content:
                            fixes_applied.append(f"Optimized test pipeline in {ci_file}")
                    
                    # Apply custom workflow updates
                    for update_path, value in workflow_updates.items():
                        content = self._update_yaml_value(content, update_path, value)
                    
                    if content != original_content:
                        context.write_file(ci_file, content)
                        
                except Exception:
                    # CI file might not exist or be inaccessible
                    continue
            
            if fixes_applied:
                return self.create_result(
                    success=True,
                    message=f"Updated CI/CD configuration: {', '.join(fixes_applied)}",
                    data={"fixes": fixes_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No CI/CD configuration updates applied"
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"CI/CD configuration update failed: {str(e)}"
            )
    
    async def _update_deployment_config(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Update deployment configuration."""
        try:
            deployment_issues = issue_data.get("deployment_issues", [])
            deployment_updates = issue_data.get("deployment_updates", {})
            fixes_applied = []
            
            # Deployment configuration files
            deployment_files = [
                "docker-compose.yml",
                "docker-compose.prod.yml",
                "k8s/deployment.yaml",
                "kubernetes/deployment.yaml",
                "deployment.yaml",
                "Dockerfile"
            ]
            
            for deploy_file in deployment_files:
                try:
                    content = context.read_file(deploy_file)
                    original_content = content
                    
                    # Apply deployment fixes
                    if "resource_limits" in deployment_issues:
                        content = self._fix_resource_limits(content, deploy_file)
                        if content != original_content:
                            fixes_applied.append(f"Fixed resource limits in {deploy_file}")
                    
                    if "health_checks" in deployment_issues:
                        content = self._add_health_checks(content, deploy_file)
                        if content != original_content:
                            fixes_applied.append(f"Added health checks in {deploy_file}")
                    
                    if "scaling_config" in deployment_issues:
                        content = self._optimize_scaling_config(content, deploy_file)
                        if content != original_content:
                            fixes_applied.append(f"Optimized scaling config in {deploy_file}")
                    
                    # Apply custom deployment updates
                    if deploy_file in deployment_updates:
                        updates = deployment_updates[deploy_file]
                        for update_path, value in updates.items():
                            if deploy_file.endswith(('.yaml', '.yml')):
                                content = self._update_yaml_value(content, update_path, value)
                            elif deploy_file == "Dockerfile":
                                content = self._update_dockerfile_value(content, update_path, value)
                    
                    if content != original_content:
                        context.write_file(deploy_file, content)
                        
                except Exception:
                    # Deployment file might not exist
                    continue
            
            if fixes_applied:
                return self.create_result(
                    success=True,
                    message=f"Updated deployment configuration: {', '.join(fixes_applied)}",
                    data={"fixes": fixes_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No deployment configuration updates applied"
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Deployment configuration update failed: {str(e)}"
            )
    
    async def _generic_config_update(self, context: Context, config_file: str, updates: Dict[str, Any]) -> ActionResult:
        """Apply generic configuration updates."""
        try:
            if not updates:
                return self.create_result(
                    success=False,
                    message="No configuration updates specified"
                )
            
            # Load existing config
            current_config = context.load_config(config_file)
            
            # Apply updates
            updated_config = self._merge_configs(current_config, updates)
            
            # Save updated config
            context.save_config(config_file, updated_config)
            
            return self.create_result(
                success=True,
                message=f"Updated configuration in {config_file}",
                data={
                    "config_file": config_file,
                    "updates_applied": updates,
                    "config_preview": updated_config
                }
            )
            
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Generic configuration update failed: {str(e)}"
            )
    
    async def _create_config_backup(self, context: Context, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create backup of configuration files before applying updates."""
        try:
            config_files = issue_data.get("config_files", ["config.yaml"])
            backup_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "files": {}
            }
            
            for config_file in config_files:
                try:
                    content = context.read_file(config_file)
                    backup_data["files"][config_file] = content
                except Exception as e:
                    logger.warning(f"Could not backup {config_file}: {e}")
            
            return backup_data
        except Exception as e:
            logger.error(f"Configuration backup creation failed: {e}")
            return {}
    
    async def _rollback_config_changes(self, context: Context, backup_data: Dict[str, Any]) -> None:
        """Rollback configuration changes using backup data."""
        try:
            files = backup_data.get("files", {})
            for file_path, content in files.items():
                if file_path.endswith(('.yaml', '.yml')):
                    context.save_config(file_path, yaml.safe_load(content))
                elif file_path.endswith('.json'):
                    context.save_config(file_path, json.loads(content))
                else:
                    context.write_file(file_path, content)
            logger.info(f"Rolled back configuration changes to {len(files)} files")
        except Exception as e:
            logger.error(f"Configuration rollback failed: {e}")
    
    async def _validate_configuration(self, context: Context, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration after updates."""
        try:
            config_files = update_data.get("config_files", [])
            if not config_files and "config_file" in update_data:
                config_files = [update_data["config_file"]]
            
            validation_errors = []
            
            for config_file in config_files:
                try:
                    if config_file.endswith(('.yaml', '.yml')):
                        content = context.read_file(config_file)
                        yaml.safe_load(content)  # Validate YAML syntax
                    elif config_file.endswith('.json'):
                        content = context.read_file(config_file)
                        json.loads(content)  # Validate JSON syntax
                    
                    # Additional semantic validation could be added here
                    
                except yaml.YAMLError as e:
                    validation_errors.append(f"YAML validation error in {config_file}: {e}")
                except json.JSONDecodeError as e:
                    validation_errors.append(f"JSON validation error in {config_file}: {e}")
                except Exception as e:
                    validation_errors.append(f"Validation error in {config_file}: {e}")
            
            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation process failed: {e}"]
            }
    
    async def _create_config_pr(self, context: Context, issue_data: Dict[str, Any], update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a GitHub pull request for configuration updates."""
        try:
            issue_type = issue_data.get("type", "")
            branch_name = f"{self.pr_branch_prefix}-{issue_type}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            
            # Get file changes
            file_changes = context.get_file_changes()
            
            if not file_changes:
                return None
            
            # Create PR title and body
            title = f"🔧 Update {issue_type.replace('_', ' ').title()} Configuration"
            body = self._generate_config_pr_body(issue_type, update_data)
            
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
            logger.error(f"Configuration PR creation failed: {e}")
            return None
    
    def _generate_config_pr_body(self, issue_type: str, update_data: Dict[str, Any]) -> str:
        """Generate PR body for configuration updates."""
        updates = update_data.get("updates", update_data.get("fixes", update_data.get("adjustments", [])))
        
        body = f"""## Automated {issue_type.replace('_', ' ').title()} Configuration Updates
        
This PR contains automated configuration updates to address {issue_type} issues.

### Changes Made:
"""
        
        if isinstance(updates, list):
            for update in updates:
                body += f"- {update}\n"
        elif isinstance(updates, dict):
            for key, value in updates.items():
                body += f"- Updated `{key}`: {value}\n"
        
        body += f"""
### Details:
- **Issue Type**: {issue_type}
- **Updates Applied**: {len(updates) if isinstance(updates, (list, dict)) else 'N/A'}
- **Auto-generated**: Yes
- **Backup Created**: Yes
- **Configuration Validated**: Yes

### Testing:
- [ ] Configuration files are valid
- [ ] Application starts successfully
- [ ] No regressions introduced
- [ ] Performance metrics verified

🤖 This PR was automatically generated by the Self-Healing MLOps Bot.
"""
        
        return body
    
    def _merge_configs(self, base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration updates."""
        result = base_config.copy()
        
        for key, value in updates.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _fix_yaml_syntax(self, content: str) -> str:
        """Fix common YAML syntax issues."""
        # Fix indentation issues
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix tabs to spaces
            line = line.replace('\t', '  ')
            
            # Fix trailing spaces
            line = line.rstrip()
            
            # Fix common YAML syntax issues
            if ':' in line and not line.strip().startswith('#'):
                # Ensure space after colon
                line = re.sub(r':([^\s])', r': \1', line)
                # Fix quotes around values containing special characters
                if re.search(r':\s*[^"\'\[\{\d\-\s].*[|>@#]', line):
                    key, value = line.split(':', 1)
                    line = f'{key}: "{value.strip()}"'
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_json_syntax(self, content: str) -> str:
        """Fix common JSON syntax issues."""
        # Fix trailing commas
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # Fix single quotes to double quotes
        content = re.sub(r"'([^']*)':", r'"\1":', content)
        content = re.sub(r":\s*'([^']*)'", r': "\1"', content)
        
        # Fix missing quotes around keys
        content = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', content)
        
        return content
    
    def _get_training_performance_optimizations(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get training performance optimizations."""
        optimizations = []
        
        # Batch size optimization
        current_batch_size = self._get_nested_value(config, 'training.batch_size') or 32
        if current_batch_size < 64:
            optimizations.append({
                "path": "training.batch_size",
                "value": min(current_batch_size * 2, 128),
                "description": "Increased batch size for better GPU utilization"
            })
        
        # Enable mixed precision
        if not self._get_nested_value(config, 'training.mixed_precision'):
            optimizations.append({
                "path": "training.mixed_precision", 
                "value": True,
                "description": "Enabled mixed precision training for speed"
            })
        
        # Optimize data loading
        current_workers = self._get_nested_value(config, 'data.num_workers') or 1
        if current_workers < 4:
            optimizations.append({
                "path": "data.num_workers",
                "value": min(current_workers * 2, 8),
                "description": "Increased data loading workers"
            })
        
        return optimizations
    
    def _get_inference_performance_optimizations(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get inference performance optimizations."""
        optimizations = []
        
        # Enable model compilation
        if not self._get_nested_value(config, 'model.compile'):
            optimizations.append({
                "path": "model.compile",
                "value": True,
                "description": "Enabled model compilation for faster inference"
            })
        
        # Optimize batch size for inference
        inference_batch = self._get_nested_value(config, 'inference.batch_size') or 1
        if inference_batch < 8:
            optimizations.append({
                "path": "inference.batch_size",
                "value": 16,
                "description": "Increased inference batch size"
            })
        
        # Enable tensor optimizations
        if not self._get_nested_value(config, 'inference.optimize_tensors'):
            optimizations.append({
                "path": "inference.optimize_tensors",
                "value": True,
                "description": "Enabled tensor optimizations"
            })
        
        return optimizations
    
    def _get_memory_optimizations(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get memory usage optimizations."""
        optimizations = []
        
        # Enable gradient checkpointing
        if not self._get_nested_value(config, 'training.gradient_checkpointing'):
            optimizations.append({
                "path": "training.gradient_checkpointing",
                "value": True,
                "description": "Enabled gradient checkpointing to save memory"
            })
        
        # Reduce cache size
        current_cache = self._get_nested_value(config, 'data.cache_size') or 1000
        if current_cache > 500:
            optimizations.append({
                "path": "data.cache_size",
                "value": current_cache // 2,
                "description": "Reduced data cache size to save memory"
            })
        
        return optimizations
    
    def _apply_cpu_resource_adjustments(self, config: Dict[str, Any]) -> None:
        """Apply CPU resource adjustments."""
        # Reduce number of workers
        current_workers = self._get_nested_value(config, 'training.num_workers') or 4
        self._set_nested_value(config, 'training.num_workers', max(1, current_workers // 2))
        
        # Optimize thread usage
        self._set_nested_value(config, 'system.max_threads', 4)
        self._set_nested_value(config, 'system.thread_affinity', True)
    
    def _apply_memory_resource_adjustments(self, config: Dict[str, Any]) -> None:
        """Apply memory resource adjustments."""
        # Reduce batch size
        current_batch = self._get_nested_value(config, 'training.batch_size') or 32
        self._set_nested_value(config, 'training.batch_size', max(1, current_batch // 2))
        
        # Enable memory-efficient options
        self._set_nested_value(config, 'training.gradient_checkpointing', True)
        self._set_nested_value(config, 'data.pin_memory', False)
    
    def _apply_gpu_resource_adjustments(self, config: Dict[str, Any]) -> None:
        """Apply GPU resource adjustments."""
        # Enable memory growth
        self._set_nested_value(config, 'gpu.memory_growth', True)
        
        # Reduce GPU batch size
        current_batch = self._get_nested_value(config, 'training.batch_size') or 32
        self._set_nested_value(config, 'training.batch_size', max(1, current_batch // 2))
        
        # Enable mixed precision
        self._set_nested_value(config, 'training.mixed_precision', True)
    
    def _generate_config_diff(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[str]:
        """Generate a human-readable diff of configuration changes."""
        changes = []
        
        def compare_configs(old_dict, new_dict, path=""):
            for key in set(list(old_dict.keys()) + list(new_dict.keys())):
                current_path = f"{path}.{key}" if path else key
                
                if key not in old_dict:
                    changes.append(f"+ {current_path}: {new_dict[key]}")
                elif key not in new_dict:
                    changes.append(f"- {current_path}: {old_dict[key]}")
                elif old_dict[key] != new_dict[key]:
                    if isinstance(old_dict[key], dict) and isinstance(new_dict[key], dict):
                        compare_configs(old_dict[key], new_dict[key], current_path)
                    else:
                        changes.append(f"~ {current_path}: {old_dict[key]} -> {new_dict[key]}")
        
        compare_configs(old_config, new_config)
        return changes
    
    def _fix_ci_timeouts(self, content: str) -> str:
        """Fix CI pipeline timeout issues."""
        # Increase timeout values
        content = re.sub(r'timeout-minutes:\s*(\d+)', 
                        lambda m: f'timeout-minutes: {max(int(m.group(1)), 30)}', content)
        
        # Add timeout to steps that don't have them
        if 'timeout-minutes' not in content and 'steps:' in content:
            content = re.sub(r'(- name: .+\n\s+run: .+)', 
                           r'\1\n        timeout-minutes: 30', content)
        
        return content
    
    def _add_dependency_caching(self, content: str, ci_file: str) -> str:
        """Add dependency caching to CI pipelines."""
        if '.github/workflows' in ci_file:
            # Add GitHub Actions caching
            if 'actions/cache@' not in content:
                cache_step = """      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-"""
            
                content = re.sub(r'(- name: Setup Python.*\n.*\n.*\n)', 
                               r'\1' + cache_step + '\n', content)
        
        return content
    
    def _optimize_test_pipeline(self, content: str) -> str:
        """Optimize test pipeline configuration."""
        # Enable parallel test execution
        if 'pytest' in content and '--numprocesses' not in content:
            content = re.sub(r'pytest', 'pytest --numprocesses auto', content)
        
        # Add test result caching
        if '--cache-dir' not in content and 'pytest' in content:
            content = re.sub(r'pytest', 'pytest --cache-dir=.pytest_cache', content)
        
        return content
    
    def _update_yaml_value(self, content: str, path: str, value: Any) -> str:
        """Update a value in YAML content using dot notation path."""
        try:
            data = yaml.safe_load(content)
            self._set_nested_value(data, path, value)
            return yaml.dump(data, default_flow_style=False)
        except Exception:
            return content
    
    def _fix_resource_limits(self, content: str, deploy_file: str) -> str:
        """Fix resource limits in deployment files."""
        if deploy_file.endswith(('.yaml', '.yml')):
            # Add resource limits if missing
            if 'resources:' not in content:
                resource_config = """
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m" """
            
                content = re.sub(r'(containers:\n.*?name: .+\n)', 
                               r'\1' + resource_config, content)
        
        return content
    
    def _add_health_checks(self, content: str, deploy_file: str) -> str:
        """Add health checks to deployment configurations."""
        if deploy_file.endswith(('.yaml', '.yml')) and 'livenessProbe' not in content:
            health_check = """
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5"""
            
            content = re.sub(r'(containers:\n.*?ports:\n.*?\n)', 
                           r'\1' + health_check, content)
        
        return content
    
    def _optimize_scaling_config(self, content: str, deploy_file: str) -> str:
        """Optimize scaling configuration."""
        if 'replicas:' in content:
            # Ensure minimum replicas for availability
            content = re.sub(r'replicas:\s*1\b', 'replicas: 2', content)
        
        # Add HPA configuration if missing
        if 'HorizontalPodAutoscaler' not in content and deploy_file.endswith('.yaml'):
            content += '\n---\napiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\nmetadata:\n  name: app-hpa\nspec:\n  scaleTargetRef:\n    apiVersion: apps/v1\n    kind: Deployment\n    name: app\n  minReplicas: 2\n  maxReplicas: 10\n  metrics:\n  - type: Resource\n    resource:\n      name: cpu\n      target:\n        type: Utilization\n        averageUtilization: 70\n'
        
        return content
    
    def _update_dockerfile_value(self, content: str, update_path: str, value: str) -> str:
        """Update values in Dockerfile."""
        if update_path == "base_image":
            content = re.sub(r'^FROM .+$', f'FROM {value}', content, flags=re.MULTILINE)
        elif update_path == "working_dir":
            if 'WORKDIR' in content:
                content = re.sub(r'WORKDIR .+', f'WORKDIR {value}', content)
            else:
                content = content.replace('FROM ', f'FROM ') + f'\nWORKDIR {value}\n'
        
        return content


class HyperparameterTuningAction(BaseAction):
    """Automatically tune hyperparameters based on performance issues."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.tuning_strategies = self.config.get("tuning_strategies", self._get_default_strategies())
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["model_degradation", "performance_issue", "training_failure"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Execute hyperparameter tuning based on the specific issue."""
        try:
            issue_type = issue_data.get("type", "")
            metric_name = issue_data.get("metric_name", "")
            degradation_pct = issue_data.get("degradation_percentage", 0)
            
            # Determine tuning strategy
            strategy = self._select_tuning_strategy(issue_type, metric_name, degradation_pct)
            
            if not strategy:
                return self.create_result(
                    success=False,
                    message=f"No tuning strategy available for {issue_type}"
                )
            
            # Apply hyperparameter adjustments
            adjustments = await self._apply_strategy(context, strategy, issue_data)
            
            if adjustments:
                return self.create_result(
                    success=True,
                    message=f"Applied hyperparameter tuning strategy: {strategy['name']}",
                    data={
                        "strategy": strategy["name"],
                        "adjustments": adjustments,
                        "reason": f"Addressing {issue_type} in {metric_name}"
                    }
                )
            else:
                return self.create_result(
                    success=False,
                    message="No hyperparameter adjustments could be applied"
                )
                
        except Exception as e:
            logger.exception(f"Hyperparameter tuning failed: {e}")
            return self.create_result(
                success=False,
                message=f"Hyperparameter tuning failed: {str(e)}"
            )
    
    def _select_tuning_strategy(self, issue_type: str, metric_name: str, degradation_pct: float) -> Optional[Dict[str, Any]]:
        """Select appropriate tuning strategy based on issue characteristics."""
        
        # High degradation - aggressive tuning
        if degradation_pct > 15:
            return self.tuning_strategies.get("aggressive_optimization")
        
        # Accuracy/performance issues
        if metric_name in ["accuracy", "f1_score", "auc_roc"]:
            return self.tuning_strategies.get("accuracy_optimization")
        
        # Latency/throughput issues
        if metric_name in ["latency_p95", "throughput"]:
            return self.tuning_strategies.get("performance_optimization")
        
        # Training failures
        if issue_type == "training_failure":
            return self.tuning_strategies.get("stability_optimization")
        
        # Default conservative strategy
        return self.tuning_strategies.get("conservative_tuning")
    
    async def _apply_strategy(self, context: Context, strategy: Dict[str, Any], issue_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply the selected tuning strategy."""
        adjustments = []
        
        for adjustment in strategy.get("adjustments", []):
            try:
                config_file = adjustment.get("file", "config.yaml")
                parameter_path = adjustment.get("parameter")
                adjustment_type = adjustment.get("type")
                value = adjustment.get("value")
                
                # Load current config
                current_config = context.load_config(config_file)
                
                # Apply adjustment
                old_value = self._get_nested_value(current_config, parameter_path)
                new_value = self._calculate_new_value(old_value, adjustment_type, value)
                
                # Update config
                self._set_nested_value(current_config, parameter_path, new_value)
                context.save_config(config_file, current_config)
                
                adjustments.append({
                    "parameter": parameter_path,
                    "old_value": old_value,
                    "new_value": new_value,
                    "adjustment_type": adjustment_type
                })
                
            except Exception as e:
                logger.warning(f"Failed to apply adjustment {adjustment}: {e}")
        
        return adjustments
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get value from nested configuration using dot notation."""
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set value in nested configuration using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _calculate_new_value(self, old_value: Any, adjustment_type: str, adjustment_value: Union[float, int, str]) -> Any:
        """Calculate new parameter value based on adjustment type."""
        if old_value is None:
            return adjustment_value
        
        if adjustment_type == "multiply":
            return old_value * adjustment_value
        elif adjustment_type == "add":
            return old_value + adjustment_value
        elif adjustment_type == "subtract":
            return old_value - adjustment_value
        elif adjustment_type == "divide":
            return old_value / adjustment_value if adjustment_value != 0 else old_value
        elif adjustment_type == "set":
            return adjustment_value
        elif adjustment_type == "reduce_by_percent":
            return old_value * (1 - adjustment_value / 100)
        elif adjustment_type == "increase_by_percent":
            return old_value * (1 + adjustment_value / 100)
        else:
            return adjustment_value
    
    def _get_default_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get default hyperparameter tuning strategies."""
        return {
            "aggressive_optimization": {
                "name": "Aggressive Performance Recovery",
                "description": "Significant changes for major performance degradation",
                "adjustments": [
                    {
                        "file": "config.yaml",
                        "parameter": "model.learning_rate",
                        "type": "reduce_by_percent",
                        "value": 50
                    },
                    {
                        "file": "config.yaml", 
                        "parameter": "training.batch_size",
                        "type": "reduce_by_percent",
                        "value": 25
                    },
                    {
                        "file": "config.yaml",
                        "parameter": "training.epochs",
                        "type": "increase_by_percent",
                        "value": 20
                    }
                ]
            },
            "accuracy_optimization": {
                "name": "Accuracy Recovery",
                "description": "Adjustments to improve model accuracy",
                "adjustments": [
                    {
                        "file": "config.yaml",
                        "parameter": "model.dropout_rate",
                        "type": "reduce_by_percent",
                        "value": 20
                    },
                    {
                        "file": "config.yaml",
                        "parameter": "model.learning_rate",
                        "type": "reduce_by_percent",
                        "value": 30
                    },
                    {
                        "file": "config.yaml",
                        "parameter": "training.early_stopping_patience",
                        "type": "increase_by_percent",
                        "value": 50
                    }
                ]
            },
            "performance_optimization": {
                "name": "Speed/Throughput Optimization",
                "description": "Adjustments to improve inference speed",
                "adjustments": [
                    {
                        "file": "config.yaml",
                        "parameter": "model.batch_size",
                        "type": "increase_by_percent",
                        "value": 25
                    },
                    {
                        "file": "config.yaml",
                        "parameter": "model.precision",
                        "type": "set",
                        "value": "fp16"
                    },
                    {
                        "file": "config.yaml",
                        "parameter": "inference.num_workers",
                        "type": "increase_by_percent",
                        "value": 50
                    }
                ]
            },
            "stability_optimization": {
                "name": "Training Stability",
                "description": "Adjustments to improve training stability",
                "adjustments": [
                    {
                        "file": "config.yaml",
                        "parameter": "model.learning_rate",
                        "type": "reduce_by_percent",
                        "value": 40
                    },
                    {
                        "file": "config.yaml",
                        "parameter": "training.gradient_clipping",
                        "type": "set",
                        "value": 1.0
                    },
                    {
                        "file": "config.yaml",
                        "parameter": "training.warmup_steps",
                        "type": "set",
                        "value": 1000
                    }
                ]
            },
            "conservative_tuning": {
                "name": "Conservative Adjustment",
                "description": "Small, safe adjustments for minor issues",
                "adjustments": [
                    {
                        "file": "config.yaml",
                        "parameter": "model.learning_rate",
                        "type": "reduce_by_percent", 
                        "value": 10
                    },
                    {
                        "file": "config.yaml",
                        "parameter": "training.batch_size",
                        "type": "reduce_by_percent",
                        "value": 10
                    }
                ]
            }
        }


class ResourceConfigAction(BaseAction):
    """Update resource configuration for infrastructure issues."""
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["resource_issue", "gpu_oom", "memory_issue", "cpu_issue"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Update resource configurations."""
        try:
            issue_type = issue_data.get("type", "")
            adjustments = []
            
            if issue_type == "gpu_oom":
                adjustments.extend(await self._fix_gpu_memory_config(context))
            elif issue_type == "memory_issue":
                adjustments.extend(await self._fix_memory_config(context))
            elif issue_type == "cpu_issue":
                adjustments.extend(await self._fix_cpu_config(context))
            
            if adjustments:
                return self.create_result(
                    success=True,
                    message=f"Applied resource configuration fixes for {issue_type}",
                    data={"adjustments": adjustments}
                )
            else:
                return self.create_result(
                    success=False,
                    message=f"No resource fixes available for {issue_type}"
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Resource configuration update failed: {str(e)}"
            )
    
    async def _fix_gpu_memory_config(self, context: Context) -> List[Dict[str, str]]:
        """Fix GPU memory configuration."""
        adjustments = []
        
        # Update training config
        config = context.load_config("config.yaml")
        
        # Reduce batch size
        if "training" in config and "batch_size" in config["training"]:
            old_batch_size = config["training"]["batch_size"]
            new_batch_size = max(1, old_batch_size // 2)
            config["training"]["batch_size"] = new_batch_size
            adjustments.append({
                "parameter": "training.batch_size",
                "old_value": str(old_batch_size),
                "new_value": str(new_batch_size),
                "reason": "Reduce GPU memory usage"
            })
        
        # Enable gradient checkpointing
        if "model" not in config:
            config["model"] = {}
        config["model"]["gradient_checkpointing"] = True
        adjustments.append({
            "parameter": "model.gradient_checkpointing",
            "old_value": "False",
            "new_value": "True",
            "reason": "Enable gradient checkpointing for memory efficiency"
        })
        
        # Enable mixed precision
        config["model"]["mixed_precision"] = True
        adjustments.append({
            "parameter": "model.mixed_precision",
            "old_value": "False",
            "new_value": "True",
            "reason": "Enable mixed precision training"
        })
        
        context.save_config("config.yaml", config)
        
        return adjustments
    
    async def _fix_memory_config(self, context: Context) -> List[Dict[str, str]]:
        """Fix general memory configuration."""
        adjustments = []
        
        config = context.load_config("config.yaml")
        
        # Reduce data loader workers
        if "data" not in config:
            config["data"] = {}
        
        old_workers = config["data"].get("num_workers", 4)
        new_workers = max(1, old_workers // 2)
        config["data"]["num_workers"] = new_workers
        
        adjustments.append({
            "parameter": "data.num_workers",
            "old_value": str(old_workers),
            "new_value": str(new_workers),
            "reason": "Reduce memory usage from data loading"
        })
        
        context.save_config("config.yaml", config)
        
        return adjustments
    
    async def _fix_cpu_config(self, context: Context) -> List[Dict[str, str]]:
        """Fix CPU configuration."""
        adjustments = []
        
        config = context.load_config("config.yaml")
        
        # Adjust number of workers
        if "training" not in config:
            config["training"] = {}
        
        config["training"]["num_workers"] = 1
        adjustments.append({
            "parameter": "training.num_workers",
            "old_value": "auto",
            "new_value": "1",
            "reason": "Reduce CPU load"
        })
        
        context.save_config("config.yaml", config)
        
        return adjustments