"""Configuration update actions for ML pipeline optimization."""

import json
import yaml
from typing import Dict, Any, List, Optional, Union
import logging
import re

from .base import BaseAction, ActionResult
from ..core.context import Context

logger = logging.getLogger(__name__)


class ConfigUpdateAction(BaseAction):
    """Update configuration files for ML pipelines."""
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["config_error", "performance_issue", "resource_issue"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Update configuration based on issue type."""
        try:
            config_file = issue_data.get("config_file", "config.yaml")
            updates = issue_data.get("config_updates", {})
            
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
            logger.exception(f"Configuration update failed: {e}")
            return self.create_result(
                success=False,
                message=f"Configuration update failed: {str(e)}"
            )
    
    def _merge_configs(self, base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration updates."""
        result = base_config.copy()
        
        for key, value in updates.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result


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