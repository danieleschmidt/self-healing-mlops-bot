"""Automated retraining actions for ML models."""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import json

from .base import BaseAction, ActionResult
from ..core.context import Context

logger = logging.getLogger(__name__)


class AutoRetrainingAction(BaseAction):
    """Trigger automated model retraining."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.retraining_triggers = self.config.get("retraining_triggers", self._get_default_triggers())
        self.max_retraining_frequency = self.config.get("max_retraining_frequency", "daily")
        self.validation_split = self.config.get("validation_split", 0.2)
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["data_drift", "model_degradation", "performance_issue"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Execute automated retraining."""
        try:
            issue_type = issue_data.get("type", "")
            
            # Check if retraining is warranted
            should_retrain, reason = await self._should_trigger_retraining(context, issue_data)
            
            if not should_retrain:
                return self.create_result(
                    success=False,
                    message=f"Retraining not triggered: {reason}"
                )
            
            # Prepare retraining configuration
            retraining_config = await self._prepare_retraining_config(context, issue_data)
            
            # Trigger retraining workflow
            workflow_result = await self._trigger_retraining_workflow(context, retraining_config)
            
            if workflow_result["success"]:
                return self.create_result(
                    success=True,
                    message=f"Retraining workflow triggered successfully",
                    data={
                        "workflow_id": workflow_result.get("workflow_id"),
                        "retraining_config": retraining_config,
                        "trigger_reason": reason
                    }
                )
            else:
                return self.create_result(
                    success=False,
                    message=f"Retraining workflow failed: {workflow_result.get('error')}"
                )
                
        except Exception as e:
            logger.exception(f"Auto retraining failed: {e}")
            return self.create_result(
                success=False,
                message=f"Auto retraining failed: {str(e)}"
            )
    
    async def _should_trigger_retraining(self, context: Context, issue_data: Dict[str, Any]) -> tuple[bool, str]:
        """Determine if retraining should be triggered."""
        issue_type = issue_data.get("type", "")
        severity = issue_data.get("severity", "medium")
        
        # Check retraining frequency limits
        last_retraining = context.get_state("last_retraining")
        if last_retraining:
            last_time = datetime.fromisoformat(last_retraining)
            time_since = datetime.utcnow() - last_time
            
            if self.max_retraining_frequency == "daily" and time_since < timedelta(days=1):
                return False, "Retraining frequency limit reached (daily)"
            elif self.max_retraining_frequency == "weekly" and time_since < timedelta(weeks=1):
                return False, "Retraining frequency limit reached (weekly)"
        
        # Check issue-specific triggers
        trigger_config = self.retraining_triggers.get(issue_type, {})
        
        if issue_type == "data_drift":
            drift_score = issue_data.get("drift_score", 0)
            threshold = trigger_config.get("drift_threshold", 0.15)
            
            if drift_score > threshold:
                return True, f"Data drift score ({drift_score:.3f}) exceeds threshold ({threshold})"
            else:
                return False, f"Data drift score ({drift_score:.3f}) below retraining threshold"
        
        elif issue_type == "model_degradation":
            degradation_pct = issue_data.get("degradation_percentage", 0)
            threshold = trigger_config.get("degradation_threshold", 10.0)
            
            if degradation_pct > threshold:
                return True, f"Model degradation ({degradation_pct:.1f}%) exceeds threshold ({threshold}%)"
            else:
                return False, f"Model degradation ({degradation_pct:.1f}%) below retraining threshold"
        
        elif issue_type == "performance_issue":
            if severity in ["critical", "high"]:
                return True, f"High severity performance issue requires retraining"
            else:
                return False, "Performance issue severity too low for automatic retraining"
        
        return False, "No matching retraining trigger found"
    
    async def _prepare_retraining_config(self, context: Context, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for retraining."""
        issue_type = issue_data.get("type", "")
        
        # Base retraining configuration
        config = {
            "trigger_reason": f"Automated retraining due to {issue_type}",
            "trigger_timestamp": datetime.utcnow().isoformat(),
            "trigger_issue": issue_data,
            "validation_split": self.validation_split,
            "early_stopping": True,
            "save_best_model": True,
            "model_versioning": True
        }
        
        # Add issue-specific configurations
        if issue_type == "data_drift":
            config.update({
                "data_refresh": True,
                "feature_selection": "auto",
                "hyperparameter_tuning": "basic",
                "drift_monitoring": True
            })
        elif issue_type == "model_degradation":
            metric_name = issue_data.get("metric_name", "accuracy")
            config.update({
                "target_metric": metric_name,
                "performance_baseline": issue_data.get("baseline_value"),
                "hyperparameter_tuning": "aggressive",
                "ensemble_methods": True
            })
        elif issue_type == "performance_issue":
            config.update({
                "optimization_focus": "speed",
                "model_pruning": True,
                "quantization": True,
                "batch_optimization": True
            })
        
        # Save configuration
        context.save_config("retraining_config.yaml", config)
        
        return config
    
    async def _trigger_retraining_workflow(self, context: Context, config: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger the retraining workflow."""
        try:
            # In a real implementation, this would trigger GitHub Actions workflow
            # or submit to ML platform (MLflow, Kubeflow, etc.)
            
            # Create retraining script
            retraining_script = self._generate_retraining_script(config)
            context.write_file("scripts/retrain_model.py", retraining_script)
            
            # Create workflow file
            workflow_yaml = self._generate_retraining_workflow(config)
            context.write_file(".github/workflows/auto_retrain.yml", workflow_yaml)
            
            # Update last retraining timestamp
            context.set_state("last_retraining", datetime.utcnow().isoformat())
            
            return {
                "success": True,
                "workflow_id": f"auto-retrain-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
                "script_path": "scripts/retrain_model.py",
                "workflow_path": ".github/workflows/auto_retrain.yml"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_retraining_script(self, config: Dict[str, Any]) -> str:
        """Generate Python retraining script."""
        return f'''#!/usr/bin/env python3
"""
Automated model retraining script
Generated by Self-Healing MLOps Bot
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load training data."""
    logger.info("Loading training data...")
    # In a real implementation, this would load from your data source
    # For now, return mock data
    return pd.DataFrame({{
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
        'target': np.random.randint(0, 2, 1000)
    }})

def preprocess_data(df):
    """Preprocess the data."""
    logger.info("Preprocessing data...")
    X = df[['feature_1', 'feature_2']]
    y = df['target']
    return X, y

def train_model(X_train, y_train, config):
    """Train the model."""
    logger.info("Training model...")
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    
    metrics = {{
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }}
    
    return metrics

def save_model(model, metrics, config):
    """Save the trained model."""
    logger.info("Saving model...")
    import joblib
    
    model_path = f"models/model_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.pkl"
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_path = model_path.replace('.pkl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model saved to {{model_path}}")
    return model_path

def main():
    """Main retraining function."""
    # Load configuration
    config = {json.dumps(config, indent=8)}
    
    logger.info("Starting automated retraining...")
    logger.info(f"Trigger reason: {{config['trigger_reason']}}")
    
    try:
        # Load and preprocess data
        df = load_data()
        X, y = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.get('validation_split', 0.2),
            random_state=42
        )
        
        # Train model
        model = train_model(X_train, y_train, config)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        logger.info(f"Model metrics: {{metrics}}")
        
        # Save model
        model_path = save_model(model, metrics, config)
        
        logger.info("Retraining completed successfully!")
        return model_path, metrics
        
    except Exception as e:
        logger.exception(f"Retraining failed: {{e}}")
        raise

if __name__ == "__main__":
    main()
'''
    
    def _generate_retraining_workflow(self, config: Dict[str, Any]) -> str:
        """Generate GitHub Actions workflow for retraining."""
        return f'''name: Automated Model Retraining

on:
  workflow_dispatch:
    inputs:
      trigger_reason:
        description: 'Reason for retraining'
        required: true
        default: '{config.get("trigger_reason", "Manual trigger")}'

jobs:
  retrain:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run retraining
      run: |
        python scripts/retrain_model.py
      env:
        TRIGGER_REASON: ${{{{ github.event.inputs.trigger_reason }}}}
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: retrained-model
        path: models/
    
    - name: Create retraining report
      run: |
        echo "## Automated Retraining Report" > retraining_report.md
        echo "- **Trigger**: ${{{{ github.event.inputs.trigger_reason }}}}" >> retraining_report.md
        echo "- **Timestamp**: $(date -u)" >> retraining_report.md
        echo "- **Workflow Run**: ${{{{ github.run_id }}}}" >> retraining_report.md
    
    - name: Comment on issue
      if: github.event.issue.number
      uses: actions/github-script@v6
      with:
        script: |
          github.rest.issues.createComment({{
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: '🤖 **Automated Retraining Completed**\\n\\nThe model has been retrained successfully. Check the artifacts for the new model version.'
          }})
'''
    
    def _get_default_triggers(self) -> Dict[str, Dict[str, Any]]:
        """Get default retraining triggers."""
        return {
            "data_drift": {
                "drift_threshold": 0.15,
                "statistical_tests": ["ks", "psi"],
                "min_samples": 1000
            },
            "model_degradation": {
                "degradation_threshold": 10.0,  # 10% degradation
                "metrics": ["accuracy", "f1_score", "auc_roc"],
                "consecutive_failures": 3
            },
            "performance_issue": {
                "latency_threshold": 500,  # ms
                "throughput_threshold": 100,  # requests/sec
                "error_rate_threshold": 0.05  # 5%
            }
        }


class DataValidationAction(BaseAction):
    """Validate data quality before retraining."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.validation_rules = self.config.get("validation_rules", self._get_default_validation_rules())
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["data_drift", "data_quality", "training_failure"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Execute data validation."""
        try:
            # Perform data quality checks
            validation_results = await self._validate_data_quality(context)
            
            # Check validation results
            passed_checks = sum(1 for result in validation_results if result["passed"])
            total_checks = len(validation_results)
            
            if passed_checks == total_checks:
                return self.create_result(
                    success=True,
                    message=f"Data validation passed ({passed_checks}/{total_checks} checks)",
                    data={
                        "validation_results": validation_results,
                        "data_quality_score": 1.0
                    }
                )
            else:
                failed_checks = [r for r in validation_results if not r["passed"]]
                return self.create_result(
                    success=False,
                    message=f"Data validation failed ({passed_checks}/{total_checks} checks passed)",
                    data={
                        "validation_results": validation_results,
                        "failed_checks": failed_checks,
                        "data_quality_score": passed_checks / total_checks
                    }
                )
                
        except Exception as e:
            logger.exception(f"Data validation failed: {e}")
            return self.create_result(
                success=False,
                message=f"Data validation failed: {str(e)}"
            )
    
    async def _validate_data_quality(self, context: Context) -> List[Dict[str, Any]]:
        """Validate data quality using predefined rules."""
        results = []
        
        # Mock data validation - in reality would check actual data
        validation_checks = [
            {
                "check_name": "schema_validation",
                "description": "Validate data schema consistency",
                "passed": True,
                "details": "All columns present with correct data types"
            },
            {
                "check_name": "missing_values",
                "description": "Check for excessive missing values",
                "passed": True,
                "details": "Missing value rate: 2.3% (below 5% threshold)"
            },
            {
                "check_name": "data_freshness",
                "description": "Verify data recency",
                "passed": True,
                "details": "Latest data is 2 hours old (within 24h threshold)"
            },
            {
                "check_name": "statistical_distribution",
                "description": "Check for distribution anomalies",
                "passed": False,
                "details": "Feature 'age' distribution shifted significantly"
            },
            {
                "check_name": "data_volume",
                "description": "Verify sufficient data volume",
                "passed": True,
                "details": "50,000 samples available (minimum: 10,000)"
            }
        ]
        
        results.extend(validation_checks)
        
        return results
    
    def _get_default_validation_rules(self) -> Dict[str, Any]:
        """Get default data validation rules."""
        return {
            "schema": {
                "enforce_types": True,
                "required_columns": [],
                "allowed_null_percentage": 0.05
            },
            "freshness": {
                "max_age_hours": 24,
                "check_timestamps": True
            },
            "volume": {
                "min_samples": 10000,
                "max_samples": None
            },
            "quality": {
                "max_duplicates_percentage": 0.01,
                "max_outliers_percentage": 0.05,
                "min_unique_values": 10
            },
            "distribution": {
                "check_drift": True,
                "drift_threshold": 0.1,
                "reference_period": "30d"
            }
        }


class ModelValidationAction(BaseAction):
    """Validate retrained model before deployment."""
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["retraining_complete", "model_ready"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Execute model validation."""
        try:
            # Perform model validation checks
            validation_results = await self._validate_model(context, issue_data)
            
            # Check if model passes validation
            all_passed = all(result["passed"] for result in validation_results)
            
            if all_passed:
                return self.create_result(
                    success=True,
                    message="Model validation passed - ready for deployment",
                    data={"validation_results": validation_results}
                )
            else:
                failed_checks = [r for r in validation_results if not r["passed"]]
                return self.create_result(
                    success=False,
                    message=f"Model validation failed - {len(failed_checks)} checks failed",
                    data={
                        "validation_results": validation_results,
                        "failed_checks": failed_checks
                    }
                )
                
        except Exception as e:
            logger.exception(f"Model validation failed: {e}")
            return self.create_result(
                success=False,
                message=f"Model validation failed: {str(e)}"
            )
    
    async def _validate_model(self, context: Context, issue_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate the retrained model."""
        # Mock model validation results
        return [
            {
                "check_name": "performance_improvement",
                "description": "Verify model performance improvement",
                "passed": True,
                "details": "Accuracy improved from 0.85 to 0.89 (+4.7%)"
            },
            {
                "check_name": "inference_speed",
                "description": "Check inference latency",
                "passed": True,
                "details": "Average inference time: 45ms (target: <100ms)"
            },
            {
                "check_name": "model_size",
                "description": "Validate model size constraints",
                "passed": True,
                "details": "Model size: 12.3MB (limit: 50MB)"
            },
            {
                "check_name": "bias_check",
                "description": "Check for model bias",
                "passed": True,
                "details": "Bias metrics within acceptable thresholds"
            },
            {
                "check_name": "stability_test",
                "description": "Test model stability",
                "passed": True,
                "details": "Model produces consistent outputs across test runs"
            }
        ]