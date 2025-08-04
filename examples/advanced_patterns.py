#!/usr/bin/env python3
"""
Advanced usage patterns for the Self-Healing MLOps Bot.

This example demonstrates:
1. Custom metrics and monitoring
2. Multi-stage repair workflows
3. Integration with external ML platforms
4. Advanced performance optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

from self_healing_bot import SelfHealingBot
from self_healing_bot.core.context import Context
from self_healing_bot.core.playbook import Playbook, Action, PlaybookRegistry
from self_healing_bot.detectors.base import BaseDetector
from self_healing_bot.performance.caching import cache_result

logger = logging.getLogger(__name__)


class MLModelPerformanceDetector(BaseDetector):
    """Advanced detector for ML model performance degradation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.baseline_metrics = self._load_baseline_metrics()
        self.performance_threshold = self.config.get("performance_threshold", 0.05)
    
    def get_supported_events(self):
        return ["schedule", "workflow_run", "push"]
    
    async def detect(self, context: Context):
        """Detect model performance issues using advanced analytics."""
        issues = []
        
        # Get current model metrics
        current_metrics = await self._fetch_model_metrics(context)
        if not current_metrics:
            return issues
        
        # Advanced anomaly detection
        anomalies = self._detect_anomalies(current_metrics)
        
        for metric_name, anomaly_info in anomalies.items():
            if anomaly_info["is_anomaly"]:
                issues.append({
                    "type": "model_performance_anomaly",
                    "severity": self._calculate_severity(anomaly_info["severity_score"]),
                    "message": f"Performance anomaly in {metric_name}: {anomaly_info['description']}",
                    "data": {
                        "metric_name": metric_name,
                        "current_value": anomaly_info["current_value"],
                        "expected_value": anomaly_info["expected_value"],
                        "anomaly_score": anomaly_info["severity_score"],
                        "historical_data": anomaly_info["historical_data"]
                    }
                })
        
        return issues
    
    @cache_result(ttl=300, namespace="model_metrics")
    async def _fetch_model_metrics(self, context: Context):
        """Fetch model metrics from monitoring systems."""
        # Mock advanced metrics fetching
        return {
            "accuracy": np.random.normal(0.92, 0.02),
            "precision": np.random.normal(0.90, 0.03),
            "recall": np.random.normal(0.88, 0.025),
            "f1_score": np.random.normal(0.89, 0.02),
            "inference_latency": np.random.normal(150, 20),
            "throughput": np.random.normal(1000, 100),
            "memory_usage": np.random.normal(2048, 200),
            "error_rate": np.random.exponential(0.01)
        }
    
    def _detect_anomalies(self, current_metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Advanced anomaly detection using statistical methods."""
        anomalies = {}
        
        for metric_name, current_value in current_metrics.items():
            baseline = self.baseline_metrics.get(metric_name, {})
            
            if not baseline:
                continue
            
            # Z-score based anomaly detection
            mean = baseline.get("mean", current_value)
            std = baseline.get("std", 0.1)
            z_score = abs(current_value - mean) / (std + 1e-8)
            
            # Seasonal decomposition (simplified)
            seasonal_factor = self._get_seasonal_factor(metric_name)
            adjusted_z_score = z_score / seasonal_factor
            
            is_anomaly = adjusted_z_score > 2.5
            
            anomalies[metric_name] = {
                "is_anomaly": is_anomaly,
                "current_value": current_value,
                "expected_value": mean,
                "z_score": z_score,
                "adjusted_z_score": adjusted_z_score,
                "severity_score": min(adjusted_z_score / 2.5, 1.0),
                "description": self._get_anomaly_description(metric_name, adjusted_z_score),
                "historical_data": baseline
            }
        
        return anomalies
    
    def _get_seasonal_factor(self, metric_name: str) -> float:
        """Get seasonal adjustment factor for metric."""
        # Mock seasonal patterns
        hour = datetime.now().hour
        if metric_name in ["throughput", "inference_latency"]:
            # Higher variance during business hours
            return 0.8 if 9 <= hour <= 17 else 1.2
        return 1.0
    
    def _get_anomaly_description(self, metric_name: str, z_score: float) -> str:
        """Generate human-readable anomaly description."""
        severity = "severe" if z_score > 4 else "moderate" if z_score > 3 else "mild"
        direction = "degradation" if metric_name in ["accuracy", "precision", "recall"] else "spike"
        return f"{severity} {direction} detected"
    
    def _calculate_severity(self, severity_score: float) -> str:
        """Calculate issue severity based on anomaly score."""
        if severity_score > 0.8:
            return "critical"
        elif severity_score > 0.6:
            return "high"
        elif severity_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _load_baseline_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load baseline metrics for comparison."""
        # Mock baseline data
        return {
            "accuracy": {"mean": 0.92, "std": 0.01},
            "precision": {"mean": 0.90, "std": 0.015},
            "recall": {"mean": 0.88, "std": 0.012},
            "f1_score": {"mean": 0.89, "std": 0.01},
            "inference_latency": {"mean": 150, "std": 15},
            "throughput": {"mean": 1000, "std": 50},
            "memory_usage": {"mean": 2048, "std": 100},
            "error_rate": {"mean": 0.01, "std": 0.005}
        }


@PlaybookRegistry.register("intelligent_model_recovery")
class IntelligentModelRecovery(Playbook):
    """Multi-stage intelligent model recovery workflow."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.recovery_strategies = {
            "accuracy": ["retrain_with_more_data", "hyperparameter_tuning", "model_rollback"],
            "latency": ["model_optimization", "batch_size_adjustment", "resource_scaling"],
            "memory": ["model_pruning", "quantization", "resource_scaling"],
            "error_rate": ["input_validation", "model_rollback", "circuit_breaker"]
        }
    
    def should_trigger(self, context: Context):
        return any(
            issue["type"] == "model_performance_anomaly" 
            for issue in context.get_state("detected_issues", [])
        )
    
    @Action(order=1, timeout=300)
    async def immediate_response(self, context: Context):
        """Immediate response to critical issues."""
        issues = context.get_state("detected_issues", [])
        critical_issues = [i for i in issues if i["data"]["anomaly_score"] > 0.8]
        
        responses = []
        
        for issue in critical_issues:
            metric_name = issue["data"]["metric_name"]
            
            if metric_name == "error_rate" and issue["data"]["current_value"] > 0.1:
                # Enable circuit breaker
                await self._enable_circuit_breaker(context)
                responses.append("Enabled circuit breaker for high error rate")
            
            elif metric_name == "inference_latency" and issue["data"]["current_value"] > 500:
                # Scale up resources immediately
                await self._emergency_scale_up(context)
                responses.append("Emergency resource scaling initiated")
        
        context.set_state("immediate_responses", responses)
        return f"Immediate responses: {'; '.join(responses)}"
    
    @Action(order=2, timeout=600)
    async def root_cause_analysis(self, context: Context):
        """Perform intelligent root cause analysis."""
        issues = context.get_state("detected_issues", [])
        
        analysis_results = {}
        
        for issue in issues:
            metric_name = issue["data"]["metric_name"]
            
            # Multi-dimensional analysis
            root_causes = await self._analyze_root_causes(context, metric_name, issue["data"])
            analysis_results[metric_name] = root_causes
        
        # Generate recovery plan
        recovery_plan = self._generate_recovery_plan(analysis_results)
        context.set_state("recovery_plan", recovery_plan)
        
        return f"Root cause analysis completed. Recovery plan generated with {len(recovery_plan)} strategies."
    
    @Action(order=3, timeout=1200)
    async def execute_recovery_strategy(self, context: Context):
        """Execute the intelligent recovery strategy."""
        recovery_plan = context.get_state("recovery_plan", {})
        execution_results = []
        
        for metric_name, strategies in recovery_plan.items():
            for strategy in strategies[:2]:  # Execute top 2 strategies
                try:
                    result = await self._execute_strategy(context, strategy, metric_name)
                    execution_results.append(f"{strategy}: {result}")
                except Exception as e:
                    execution_results.append(f"{strategy}: Failed - {str(e)}")
        
        context.set_state("execution_results", execution_results)
        return f"Executed {len(execution_results)} recovery strategies"
    
    @Action(order=4, timeout=300)
    async def validation_and_monitoring(self, context: Context):
        """Validate recovery and set up enhanced monitoring."""
        execution_results = context.get_state("execution_results", [])
        
        # Set up enhanced monitoring
        monitoring_config = {
            "enhanced_metrics": True,
            "alert_sensitivity": "high",
            "monitoring_duration": "24h",
            "rollback_threshold": 0.1
        }
        
        context.set_state("enhanced_monitoring", monitoring_config)
        
        # Create comprehensive report
        report = self._generate_recovery_report(context)
        
        # Create PR with recovery changes
        pr = context.create_pull_request(
            title="🤖 Intelligent Model Recovery",
            body=report,
            branch="recovery/intelligent-model-recovery"
        )
        
        return f"Recovery validation completed. Monitoring enhanced. PR created: #{pr.number}"
    
    async def _enable_circuit_breaker(self, context: Context):
        """Enable circuit breaker pattern."""
        circuit_breaker_config = """
# Circuit breaker configuration
circuit_breaker:
  failure_threshold: 5
  timeout: 30
  monitor_requests: 100
"""
        context.write_file("config/circuit_breaker.yml", circuit_breaker_config)
    
    async def _emergency_scale_up(self, context: Context):
        """Emergency resource scaling."""
        scaling_config = """
# Emergency scaling configuration
scaling:
  replicas: 5
  cpu: "1000m"
  memory: "4Gi"
  gpu: 1
"""
        context.write_file("config/emergency_scaling.yml", scaling_config)
    
    async def _analyze_root_causes(self, context: Context, metric_name: str, issue_data: Dict[str, Any]) -> List[str]:
        """Advanced root cause analysis."""
        possible_causes = []
        
        if metric_name == "accuracy":
            if issue_data["current_value"] < issue_data["expected_value"] * 0.9:
                possible_causes.extend(["data_drift", "model_degradation", "feature_corruption"])
        
        elif metric_name == "inference_latency":
            if issue_data["current_value"] > issue_data["expected_value"] * 1.5:
                possible_causes.extend(["resource_contention", "model_complexity", "batch_processing"])
        
        elif metric_name == "memory_usage":
            if issue_data["current_value"] > issue_data["expected_value"] * 1.3:
                possible_causes.extend(["memory_leak", "inefficient_operations", "batch_size"])
        
        return possible_causes
    
    def _generate_recovery_plan(self, analysis_results: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Generate intelligent recovery plan."""
        recovery_plan = {}
        
        for metric_name, causes in analysis_results.items():
            strategies = []
            
            for cause in causes:
                if cause in ["data_drift", "model_degradation"]:
                    strategies.append("automated_retraining")
                elif cause in ["resource_contention", "batch_processing"]:
                    strategies.append("resource_optimization")
                elif cause in ["memory_leak", "inefficient_operations"]:
                    strategies.append("code_optimization")
            
            # Add metric-specific strategies
            if metric_name in self.recovery_strategies:
                strategies.extend(self.recovery_strategies[metric_name])
            
            # Remove duplicates and prioritize
            recovery_plan[metric_name] = list(dict.fromkeys(strategies))
        
        return recovery_plan
    
    async def _execute_strategy(self, context: Context, strategy: str, metric_name: str) -> str:
        """Execute specific recovery strategy."""
        if strategy == "automated_retraining":
            return await self._trigger_retraining(context)
        elif strategy == "resource_optimization":
            return await self._optimize_resources(context)
        elif strategy == "hyperparameter_tuning":
            return await self._tune_hyperparameters(context)
        elif strategy == "model_rollback":
            return await self._rollback_model(context)
        else:
            return f"Strategy {strategy} not implemented"
    
    async def _trigger_retraining(self, context: Context) -> str:
        """Trigger automated model retraining."""
        retraining_config = """
# Automated retraining configuration
retraining:
  trigger: "performance_degradation"
  data_window: "30d"
  validation_split: 0.2
  early_stopping: true
  hyperparameter_search: true
"""
        context.write_file("config/retraining.yml", retraining_config)
        return "Automated retraining pipeline configured and triggered"
    
    async def _optimize_resources(self, context: Context) -> str:
        """Optimize computational resources."""
        optimization_config = """
# Resource optimization
optimization:
  cpu_request: "500m"
  cpu_limit: "2000m"
  memory_request: "1Gi"
  memory_limit: "4Gi"
  auto_scaling:
    min_replicas: 3
    max_replicas: 10
    target_cpu_utilization: 70
"""
        context.write_file("config/resource_optimization.yml", optimization_config)
        return "Resource optimization configuration applied"
    
    async def _tune_hyperparameters(self, context: Context) -> str:
        """Automated hyperparameter tuning."""
        tuning_config = """
# Hyperparameter tuning configuration
hyperparameter_tuning:
  method: "bayesian_optimization"
  trials: 50
  parameters:
    learning_rate: [0.0001, 0.01]
    batch_size: [16, 32, 64, 128]
    dropout_rate: [0.1, 0.5]
    l2_regularization: [0.0001, 0.01]
"""
        context.write_file("config/hyperparameter_tuning.yml", tuning_config)
        return "Hyperparameter tuning pipeline initiated"
    
    async def _rollback_model(self, context: Context) -> str:
        """Rollback to previous stable model version."""
        rollback_config = """
# Model rollback configuration
rollback:
  target_version: "previous_stable"
  validation_required: true
  gradual_rollout: true
  rollout_percentage: 50
"""
        context.write_file("config/rollback.yml", rollback_config)
        return "Model rollback to previous stable version initiated"
    
    def _generate_recovery_report(self, context: Context) -> str:
        """Generate comprehensive recovery report."""
        issues = context.get_state("detected_issues", [])
        recovery_plan = context.get_state("recovery_plan", {})
        execution_results = context.get_state("execution_results", [])
        
        return f"""
## 🤖 Intelligent Model Recovery Report

### Issues Detected
{chr(10).join([f"- **{issue['data']['metric_name']}**: {issue['message']}" for issue in issues])}

### Recovery Plan Executed
{chr(10).join([f"- {result}" for result in execution_results])}

### Enhanced Monitoring
- 24-hour enhanced monitoring enabled
- Alert sensitivity increased
- Automatic rollback configured

### Next Steps
1. Monitor model performance for next 24 hours
2. Validate recovery effectiveness
3. Consider long-term improvements if issues persist

### Generated Configurations
- Circuit breaker settings
- Resource optimization
- Automated retraining pipeline
- Hyperparameter tuning setup

---
*This recovery was automatically executed by the self-healing MLOps bot*
        """


class MLPlatformIntegration:
    """Integration with external ML platforms."""
    
    @staticmethod
    async def integrate_with_mlflow(context: Context):
        """Integrate with MLflow for experiment tracking."""
        integration_code = """
import mlflow
from self_healing_bot import BotIntegration

# MLflow integration for self-healing bot
class MLflowBotIntegration(BotIntegration):
    def __init__(self, tracking_uri):
        self.client = mlflow.tracking.MlflowClient(tracking_uri)
    
    async def get_model_metrics(self, model_name, stage="Production"):
        \"\"\"Get model metrics from MLflow.\"\"\"
        model_version = self.client.get_latest_versions(
            model_name, stages=[stage]
        )[0]
        
        run = self.client.get_run(model_version.run_id)
        return run.data.metrics
    
    async def trigger_retraining(self, experiment_name):
        \"\"\"Trigger model retraining experiment.\"\"\"
        experiment = self.client.get_experiment_by_name(experiment_name)
        # Trigger retraining logic here
        pass
"""
        
        context.write_file("integrations/mlflow_integration.py", integration_code)
        return "MLflow integration configured"
    
    @staticmethod
    async def integrate_with_wandb(context: Context):
        """Integrate with Weights & Biases."""
        integration_code = """
import wandb
from self_healing_bot import BotIntegration

class WandBBotIntegration(BotIntegration):
    def __init__(self, project_name):
        self.project = project_name
    
    async def get_model_metrics(self, run_id):
        \"\"\"Get model metrics from W&B.\"\"\"
        api = wandb.Api()
        run = api.run(f"{self.project}/{run_id}")
        return run.summary
    
    async def create_alert(self, metric_name, threshold):
        \"\"\"Create W&B alert for model monitoring.\"\"\"
        # W&B alert creation logic
        pass
"""
        
        context.write_file("integrations/wandb_integration.py", integration_code)
        return "W&B integration configured"


async def advanced_example():
    """Run advanced usage example."""
    logger.info("Starting advanced Self-Healing MLOps Bot example")
    
    # Initialize bot with advanced configuration
    bot = SelfHealingBot()
    
    # Register advanced detector
    advanced_detector = MLModelPerformanceDetector({
        "performance_threshold": 0.05,
        "anomaly_sensitivity": 2.5,
        "seasonal_adjustment": True
    })
    bot.detector_registry.register_detector("ml_performance", advanced_detector)
    
    # Simulate model performance issue
    performance_event = {
        "schedule": "0 */6 * * *",  # Every 6 hours
        "repository": {
            "full_name": "ml-team/production-model",
            "name": "production-model"
        }
    }
    
    try:
        # Process performance monitoring event
        context = await bot.process_event("schedule", performance_event)
        
        if context:
            logger.info("Advanced model recovery workflow completed")
            
            # Demonstrate platform integration
            await MLPlatformIntegration.integrate_with_mlflow(context)
            await MLPlatformIntegration.integrate_with_wandb(context)
            
            logger.info("ML platform integrations configured")
    
    except Exception as e:
        logger.error(f"Error in advanced example: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(advanced_example())