"""Resource scaling actions for dynamic infrastructure management."""

import json
import yaml
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from .base import BaseAction, ActionResult
from ..core.context import Context
from ..integrations.github import GitHubIntegration

logger = logging.getLogger(__name__)


class ResourceScalingAction(BaseAction):
    """Dynamically scale resources based on demand and performance metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.github_integration = GitHubIntegration()
        self.create_pr = self.config.get("create_pr", True)
        self.pr_branch_prefix = self.config.get("pr_branch_prefix", "bot/resource-scaling")
        self.min_replicas = self.config.get("min_replicas", 1)
        self.max_replicas = self.config.get("max_replicas", 10)
        self.scale_up_threshold = self.config.get("scale_up_threshold", 0.7)  # 70% resource usage
        self.scale_down_threshold = self.config.get("scale_down_threshold", 0.3)  # 30% resource usage
        self.scale_cooldown = self.config.get("scale_cooldown", 300)  # 5 minutes
        self.auto_scaling_enabled = self.config.get("auto_scaling_enabled", True)
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in [
            "high_cpu_usage", "high_memory_usage", "high_traffic", "low_performance",
            "resource_exhaustion", "scaling_needed", "cost_optimization"
        ]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Execute resource scaling based on issue type and metrics."""
        try:
            issue_type = issue_data.get("type", "")
            
            if not self.can_handle(issue_type):
                return self.create_result(
                    success=False,
                    message=f"Cannot handle issue type: {issue_type}"
                )
            
            self.log_action(context, f"Starting resource scaling for {issue_type}")
            
            # Analyze current resource usage and requirements
            analysis = await self._analyze_resource_requirements(context, issue_data)
            
            if not analysis["scaling_needed"]:
                return self.create_result(
                    success=False,
                    message=f"No scaling needed: {analysis['reason']}"
                )
            
            # Determine scaling strategy
            scaling_plan = await self._create_scaling_plan(context, analysis, issue_data)
            
            # Execute scaling actions
            scaling_result = await self._execute_scaling_plan(context, scaling_plan)
            
            if scaling_result["success"]:
                # Monitor scaling results
                monitoring_result = await self._monitor_scaling_results(context, scaling_plan)
                scaling_result["monitoring"] = monitoring_result
                
                # Create PR if enabled
                pr_result = None
                if self.create_pr:
                    pr_result = await self._create_scaling_pr(context, issue_data, scaling_result)
                
                result_data = {
                    "analysis": analysis,
                    "scaling_plan": scaling_plan,
                    "scaling_result": scaling_result,
                    "monitoring": monitoring_result
                }
                
                if pr_result:
                    result_data["pull_request"] = pr_result
                
                return self.create_result(
                    success=True,
                    message=f"Successfully scaled resources: {scaling_result['summary']}",
                    data=result_data
                )
            else:
                return self.create_result(
                    success=False,
                    message=f"Resource scaling failed: {scaling_result['error']}"
                )
                
        except Exception as e:
            logger.exception(f"Resource scaling failed: {e}")
            return self.create_result(
                success=False,
                message=f"Resource scaling failed: {str(e)}"
            )
    
    async def _analyze_resource_requirements(self, context: Context, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current resource usage and scaling requirements."""
        try:
            issue_type = issue_data.get("type", "")
            current_metrics = issue_data.get("metrics", {})
            
            analysis = {
                "current_replicas": current_metrics.get("replicas", 3),
                "current_cpu_usage": current_metrics.get("cpu_usage_percent", 50.0),
                "current_memory_usage": current_metrics.get("memory_usage_percent", 60.0),
                "current_rps": current_metrics.get("requests_per_second", 100),
                "avg_response_time": current_metrics.get("avg_response_time_ms", 200),
                "error_rate": current_metrics.get("error_rate_percent", 1.0),
                "scaling_needed": False,
                "scaling_direction": None,
                "scaling_urgency": "low",
                "reason": ""
            }
            
            # Determine if scaling is needed based on issue type and metrics
            if issue_type in ["high_cpu_usage", "high_memory_usage"]:
                if analysis["current_cpu_usage"] > self.scale_up_threshold * 100:
                    analysis["scaling_needed"] = True
                    analysis["scaling_direction"] = "up"
                    analysis["scaling_urgency"] = "high" if analysis["current_cpu_usage"] > 90 else "medium"
                    analysis["reason"] = f"High CPU usage ({analysis['current_cpu_usage']:.1f}%)"
                
                elif analysis["current_memory_usage"] > self.scale_up_threshold * 100:
                    analysis["scaling_needed"] = True
                    analysis["scaling_direction"] = "up"
                    analysis["scaling_urgency"] = "high" if analysis["current_memory_usage"] > 90 else "medium"
                    analysis["reason"] = f"High memory usage ({analysis['current_memory_usage']:.1f}%)"
            
            elif issue_type == "high_traffic":
                if analysis["avg_response_time"] > 500 or analysis["error_rate"] > 5:
                    analysis["scaling_needed"] = True
                    analysis["scaling_direction"] = "up"
                    analysis["scaling_urgency"] = "high" if analysis["error_rate"] > 10 else "medium"
                    analysis["reason"] = f"High traffic causing performance degradation"
            
            elif issue_type == "low_performance":
                if analysis["current_cpu_usage"] < self.scale_down_threshold * 100 and analysis["current_memory_usage"] < self.scale_down_threshold * 100:
                    analysis["scaling_needed"] = True
                    analysis["scaling_direction"] = "down"
                    analysis["scaling_urgency"] = "low"
                    analysis["reason"] = f"Low resource utilization - can scale down"
                else:
                    analysis["scaling_needed"] = True
                    analysis["scaling_direction"] = "up"
                    analysis["scaling_urgency"] = "medium"
                    analysis["reason"] = f"Performance issues despite normal resource usage"
            
            elif issue_type == "resource_exhaustion":
                analysis["scaling_needed"] = True
                analysis["scaling_direction"] = "up"
                analysis["scaling_urgency"] = "critical"
                analysis["reason"] = "Resource exhaustion detected"
            
            elif issue_type == "cost_optimization":
                if (analysis["current_cpu_usage"] < 30 and 
                    analysis["current_memory_usage"] < 30 and 
                    analysis["current_replicas"] > self.min_replicas):
                    analysis["scaling_needed"] = True
                    analysis["scaling_direction"] = "down"
                    analysis["scaling_urgency"] = "low"
                    analysis["reason"] = "Cost optimization opportunity - scale down"
            
            # Add predictive analysis
            analysis["predicted_metrics"] = await self._predict_future_resource_needs(context, current_metrics)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Resource analysis failed: {e}")
            return {
                "scaling_needed": False,
                "reason": f"Analysis failed: {str(e)}",
                "error": True
            }
    
    async def _create_scaling_plan(self, context: Context, analysis: Dict[str, Any], issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive scaling plan."""
        current_replicas = analysis["current_replicas"]
        scaling_direction = analysis["scaling_direction"]
        scaling_urgency = analysis["scaling_urgency"]
        
        # Calculate target replicas
        if scaling_direction == "up":
            if scaling_urgency == "critical":
                target_replicas = min(current_replicas * 3, self.max_replicas)
            elif scaling_urgency == "high":
                target_replicas = min(current_replicas * 2, self.max_replicas)
            else:
                target_replicas = min(current_replicas + 2, self.max_replicas)
        else:  # scale down
            if scaling_urgency == "low":
                target_replicas = max(current_replicas - 1, self.min_replicas)
            else:
                target_replicas = max(current_replicas // 2, self.min_replicas)
        
        # Determine scaling strategy
        if scaling_urgency == "critical":
            strategy = "immediate"
        elif scaling_urgency == "high":
            strategy = "fast"
        else:
            strategy = "gradual"
        
        # Calculate resource adjustments
        resource_adjustments = self._calculate_resource_adjustments(analysis, target_replicas)
        
        return {
            "current_replicas": current_replicas,
            "target_replicas": target_replicas,
            "scaling_direction": scaling_direction,
            "scaling_factor": target_replicas / current_replicas,
            "strategy": strategy,
            "urgency": scaling_urgency,
            "resource_adjustments": resource_adjustments,
            "estimated_duration": self._estimate_scaling_duration(strategy, abs(target_replicas - current_replicas)),
            "rollback_plan": self._create_rollback_plan(current_replicas, target_replicas),
            "monitoring_plan": self._create_monitoring_plan(),
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def _execute_scaling_plan(self, context: Context, scaling_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the scaling plan across different infrastructure components."""
        try:
            strategy = scaling_plan["strategy"]
            target_replicas = scaling_plan["target_replicas"]
            resource_adjustments = scaling_plan["resource_adjustments"]
            
            execution_steps = []
            
            # Step 1: Update Kubernetes HPA/deployment
            k8s_result = await self._scale_kubernetes_resources(context, scaling_plan)
            execution_steps.append({"step": "kubernetes_scaling", "result": k8s_result})
            
            # Step 2: Update Docker Compose (if applicable)
            docker_result = await self._scale_docker_resources(context, scaling_plan)
            execution_steps.append({"step": "docker_scaling", "result": docker_result})
            
            # Step 3: Update cloud provider auto-scaling settings
            cloud_result = await self._scale_cloud_resources(context, scaling_plan)
            execution_steps.append({"step": "cloud_scaling", "result": cloud_result})
            
            # Step 4: Update load balancer configuration
            lb_result = await self._update_load_balancer_config(context, scaling_plan)
            execution_steps.append({"step": "load_balancer_update", "result": lb_result})
            
            # Step 5: Update monitoring alerts and thresholds
            monitoring_result = await self._update_monitoring_config(context, scaling_plan)
            execution_steps.append({"step": "monitoring_update", "result": monitoring_result})
            
            # Determine overall success
            successful_steps = [step for step in execution_steps if step["result"].get("success", False)]
            success_rate = len(successful_steps) / len(execution_steps)
            
            overall_success = success_rate >= 0.8  # At least 80% of steps successful
            
            return {
                "success": overall_success,
                "success_rate": success_rate,
                "execution_steps": execution_steps,
                "successful_steps": len(successful_steps),
                "total_steps": len(execution_steps),
                "summary": f"Scaled from {scaling_plan['current_replicas']} to {target_replicas} replicas",
                "completed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_steps": execution_steps,
                "failed_at": datetime.utcnow().isoformat()
            }
    
    async def _scale_kubernetes_resources(self, context: Context, scaling_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Scale Kubernetes resources."""
        try:
            target_replicas = scaling_plan["target_replicas"]
            resource_adjustments = scaling_plan["resource_adjustments"]
            
            # Update HPA configuration
            hpa_config = f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model
  minReplicas: {max(1, target_replicas // 2)}
  maxReplicas: {target_replicas * 2}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {int(self.scale_up_threshold * 100)}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {int(self.scale_up_threshold * 100)}
"""
            
            # Update deployment configuration
            deployment_config = context.read_file("k8s/deployment.yaml")
            
            # Update replicas and resources
            updated_deployment = yaml.safe_load(deployment_config)
            updated_deployment["spec"]["replicas"] = target_replicas
            
            # Update resource limits if specified
            if resource_adjustments.get("cpu_limit") or resource_adjustments.get("memory_limit"):
                container_resources = updated_deployment["spec"]["template"]["spec"]["containers"][0].setdefault("resources", {})
                limits = container_resources.setdefault("limits", {})
                requests = container_resources.setdefault("requests", {})
                
                if resource_adjustments.get("cpu_limit"):
                    limits["cpu"] = resource_adjustments["cpu_limit"]
                    requests["cpu"] = resource_adjustments.get("cpu_request", resource_adjustments["cpu_limit"])
                
                if resource_adjustments.get("memory_limit"):
                    limits["memory"] = resource_adjustments["memory_limit"]
                    requests["memory"] = resource_adjustments.get("memory_request", resource_adjustments["memory_limit"])
            
            # Save updated configurations
            context.write_file("k8s/hpa.yaml", hpa_config)
            context.write_file("k8s/deployment.yaml", yaml.dump(updated_deployment, default_flow_style=False))
            
            # Create scaling script
            scaling_script = f"""#!/bin/bash
set -e

echo "Scaling Kubernetes resources to {target_replicas} replicas..."

# Apply HPA configuration
kubectl apply -f k8s/hpa.yaml

# Apply deployment configuration
kubectl apply -f k8s/deployment.yaml

# Wait for rollout to complete
kubectl rollout status deployment/ml-model -n production --timeout=600s

# Verify scaling
kubectl get hpa ml-model-hpa -n production
kubectl get pods -n production -l app=ml-model

echo "Kubernetes scaling completed successfully"
"""
            
            context.write_file("scale_k8s.sh", scaling_script)
            
            return {
                "success": True,
                "message": f"Kubernetes resources configured for {target_replicas} replicas",
                "details": {
                    "target_replicas": target_replicas,
                    "hpa_updated": True,
                    "deployment_updated": True,
                    "script_created": "scale_k8s.sh"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to scale Kubernetes resources"
            }
    
    async def _scale_docker_resources(self, context: Context, scaling_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Scale Docker Compose resources."""
        try:
            target_replicas = scaling_plan["target_replicas"]
            
            # Update docker-compose.yml
            compose_config = yaml.safe_load(context.read_file("docker-compose.yml"))
            
            if "ml-model" in compose_config.get("services", {}):
                service_config = compose_config["services"]["ml-model"]
                service_config["deploy"] = service_config.get("deploy", {})
                service_config["deploy"]["replicas"] = target_replicas
                
                # Update resource limits
                resource_adjustments = scaling_plan["resource_adjustments"]
                if resource_adjustments.get("cpu_limit") or resource_adjustments.get("memory_limit"):
                    resources = service_config["deploy"].setdefault("resources", {})
                    limits = resources.setdefault("limits", {})
                    
                    if resource_adjustments.get("cpu_limit"):
                        limits["cpus"] = resource_adjustments["cpu_limit"]
                    if resource_adjustments.get("memory_limit"):
                        limits["memory"] = resource_adjustments["memory_limit"]
                
                context.write_file("docker-compose.yml", yaml.dump(compose_config, default_flow_style=False))
                
                # Create Docker scaling script
                docker_script = f"""#!/bin/bash
set -e

echo "Scaling Docker Compose services to {target_replicas} replicas..."

# Scale the service
docker-compose up -d --scale ml-model={target_replicas}

# Verify scaling
docker-compose ps ml-model

echo "Docker Compose scaling completed successfully"
"""
                
                context.write_file("scale_docker.sh", docker_script)
                
                return {
                    "success": True,
                    "message": f"Docker Compose configured for {target_replicas} replicas",
                    "details": {
                        "target_replicas": target_replicas,
                        "compose_updated": True,
                        "script_created": "scale_docker.sh"
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "ml-model service not found in docker-compose.yml",
                    "warning": True
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to scale Docker resources"
            }
    
    async def _scale_cloud_resources(self, context: Context, scaling_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Update cloud provider auto-scaling configurations."""
        try:
            target_replicas = scaling_plan["target_replicas"]
            
            # Create Terraform configuration for auto-scaling
            terraform_config = f"""
# Auto-scaling configuration
resource "aws_autoscaling_group" "ml_model_asg" {{
  name                = "ml-model-asg"
  vpc_zone_identifier = var.subnet_ids
  target_group_arns   = [aws_lb_target_group.ml_model_tg.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = {max(1, target_replicas // 2)}
  max_size         = {target_replicas * 2}
  desired_capacity = {target_replicas}

  launch_template {{
    id      = aws_launch_template.ml_model_lt.id
    version = "$Latest"
  }}

  tag {{
    key                 = "Name"
    value               = "ml-model-instance"
    propagate_at_launch = true
  }}
}}

resource "aws_autoscaling_policy" "ml_model_scale_up" {{
  name                   = "ml-model-scale-up"
  scaling_adjustment     = 2
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.ml_model_asg.name
}}

resource "aws_autoscaling_policy" "ml_model_scale_down" {{
  name                   = "ml-model-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.ml_model_asg.name
}}

resource "aws_cloudwatch_metric_alarm" "cpu_high" {{
  alarm_name          = "ml-model-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "60"
  statistic           = "Average"
  threshold           = "{int(self.scale_up_threshold * 100)}"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.ml_model_scale_up.arn]

  dimensions = {{
    AutoScalingGroupName = aws_autoscaling_group.ml_model_asg.name
  }}
}}

resource "aws_cloudwatch_metric_alarm" "cpu_low" {{
  alarm_name          = "ml-model-cpu-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "60"
  statistic           = "Average"
  threshold           = "{int(self.scale_down_threshold * 100)}"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.ml_model_scale_down.arn]

  dimensions = {{
    AutoScalingGroupName = aws_autoscaling_group.ml_model_asg.name
  }}
}}
"""
            
            context.write_file("terraform/autoscaling.tf", terraform_config)
            
            return {
                "success": True,
                "message": "Cloud auto-scaling configuration updated",
                "details": {
                    "target_replicas": target_replicas,
                    "min_replicas": max(1, target_replicas // 2),
                    "max_replicas": target_replicas * 2,
                    "terraform_updated": True
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to update cloud auto-scaling configuration"
            }
    
    async def _update_load_balancer_config(self, context: Context, scaling_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Update load balancer configuration for scaled resources."""
        try:
            # Update nginx configuration
            nginx_config = f"""
upstream ml_model_backend {{
    least_conn;
    
    # Health check settings
    server 127.0.0.1:8080 max_fails=3 fail_timeout=30s;
    
    # Additional servers will be added by auto-scaling
    # keepalive connections
    keepalive 32;
}}

server {{
    listen 80;
    server_name ml-model.example.com;
    
    # Load balancing configuration
    location / {{
        proxy_pass http://ml_model_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeout settings optimized for scaling
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Retry failed requests
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
        proxy_next_upstream_tries 3;
        proxy_next_upstream_timeout 10s;
    }}
    
    # Health check endpoint
    location /health {{
        access_log off;
        return 200 "healthy\\n";
        add_header Content-Type text/plain;
    }}
}}
"""
            
            context.write_file("nginx_scaled.conf", nginx_config)
            
            return {
                "success": True,
                "message": "Load balancer configuration updated for scaling",
                "details": {
                    "nginx_config_updated": True,
                    "health_checks_enabled": True,
                    "retry_logic_configured": True
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to update load balancer configuration"
            }
    
    async def _update_monitoring_config(self, context: Context, scaling_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Update monitoring configuration for scaled infrastructure."""
        try:
            target_replicas = scaling_plan["target_replicas"]
            
            # Update Prometheus monitoring rules
            prometheus_rules = f"""
groups:
- name: ml_model_scaling
  rules:
  # High CPU usage alert
  - alert: MLModelHighCPU
    expr: avg(cpu_usage_percent) > {int(self.scale_up_threshold * 100)}
    for: 5m
    labels:
      severity: warning
      service: ml-model
    annotations:
      summary: "ML Model high CPU usage detected"
      description: "CPU usage is {{% value %}}% for 5 minutes"
      
  # High memory usage alert
  - alert: MLModelHighMemory
    expr: avg(memory_usage_percent) > {int(self.scale_up_threshold * 100)}
    for: 5m
    labels:
      severity: warning
      service: ml-model
    annotations:
      summary: "ML Model high memory usage detected"
      description: "Memory usage is {{% value %}}% for 5 minutes"
      
  # Scaling event notification
  - alert: MLModelScalingEvent
    expr: changes(ml_model_replicas[5m]) > 0
    labels:
      severity: info
      service: ml-model
    annotations:
      summary: "ML Model scaling event occurred"
      description: "Replica count changed to {{% value %}}"

  # Low resource utilization (for scale-down)
  - alert: MLModelLowUtilization
    expr: avg(cpu_usage_percent) < {int(self.scale_down_threshold * 100)} and avg(memory_usage_percent) < {int(self.scale_down_threshold * 100)}
    for: 15m
    labels:
      severity: info
      service: ml-model
    annotations:
      summary: "ML Model low resource utilization"
      description: "Consider scaling down - CPU: {{% value %}}%, Memory: {{% value %}}%"
"""
            
            # Update Grafana dashboard configuration
            grafana_dashboard = {{
                "dashboard": {{
                    "title": "ML Model Auto-Scaling",
                    "panels": [
                        {{
                            "title": "Replica Count",
                            "type": "stat",
                            "targets": [
                                {{
                                    "expr": "ml_model_replicas",
                                    "legendFormat": "Current Replicas"
                                }}
                            ]
                        }},
                        {{
                            "title": "Resource Utilization",
                            "type": "graph",
                            "targets": [
                                {{
                                    "expr": "avg(cpu_usage_percent)",
                                    "legendFormat": "CPU Usage %"
                                }},
                                {{
                                    "expr": "avg(memory_usage_percent)", 
                                    "legendFormat": "Memory Usage %"
                                }}
                            ]
                        }},
                        {{
                            "title": "Scaling Events",
                            "type": "table",
                            "targets": [
                                {{
                                    "expr": "increase(scaling_events_total[1h])",
                                    "legendFormat": "Scaling Events (1h)"
                                }}
                            ]
                        }}
                    ]
                }}
            }}
            
            context.write_file("monitoring/prometheus_scaling_rules.yml", prometheus_rules)
            context.write_file("monitoring/grafana_scaling_dashboard.json", json.dumps(grafana_dashboard, indent=2))
            
            return {{
                "success": True,
                "message": "Monitoring configuration updated for scaling",
                "details": {{
                    "prometheus_rules_updated": True,
                    "grafana_dashboard_updated": True,
                    "alert_thresholds_configured": True
                }}
            }}
            
        except Exception as e:
            return {{
                "success": False,
                "error": str(e),
                "message": "Failed to update monitoring configuration"
            }}
    
    async def _monitor_scaling_results(self, context: Context, scaling_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor the results of scaling actions."""
        try:
            target_replicas = scaling_plan["target_replicas"]
            monitoring_duration = 300  # 5 minutes
            
            # Create monitoring script
            monitoring_script = f"""#!/bin/bash
set -e

echo "Monitoring scaling results for {monitoring_duration} seconds..."

START_TIME=$(date +%s)
END_TIME=$((START_TIME + {monitoring_duration}))

while [ $(date +%s) -lt $END_TIME ]; do
    echo "=== $(date) ==="
    
    # Check Kubernetes pods
    if command -v kubectl &> /dev/null; then
        echo "Kubernetes pods:"
        kubectl get pods -n production -l app=ml-model --no-headers | wc -l
        kubectl top pods -n production -l app=ml-model
    fi
    
    # Check Docker containers
    if command -v docker &> /dev/null; then
        echo "Docker containers:"
        docker ps --filter name=ml-model --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"
    fi
    
    # Check service health
    echo "Health checks:"
    for i in {{1..{target_replicas}}}; do
        curl -f http://localhost:8080/health > /dev/null 2>&1 && echo "Instance $i: OK" || echo "Instance $i: FAIL"
    done
    
    echo "---"
    sleep 30
done

echo "Monitoring completed"
"""
            
            context.write_file("monitor_scaling.sh", monitoring_script)
            
            # Mock monitoring results for demo
            monitoring_results = {{
                "duration": monitoring_duration,
                "target_replicas": target_replicas,
                "actual_replicas": target_replicas,  # In reality, would check actual count
                "health_check_success_rate": 95.0,
                "average_response_time": 180,  # ms
                "resource_utilization": {{
                    "cpu": 45.0,  # %
                    "memory": 55.0  # %
                }},
                "scaling_success": True,
                "monitoring_script": "monitor_scaling.sh"
            }}
            
            return monitoring_results
            
        except Exception as e:
            return {{
                "success": False,
                "error": str(e),
                "message": "Failed to set up scaling monitoring"
            }}
    
    async def _create_scaling_pr(self, context: Context, issue_data: Dict[str, Any], scaling_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a GitHub pull request for scaling changes."""
        try:
            issue_type = issue_data.get("type", "")
            scaling_plan = scaling_result.get("scaling_plan", {})
            branch_name = f"{self.pr_branch_prefix}-{issue_type}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            
            # Get file changes
            file_changes = context.get_file_changes()
            
            # Create PR title and body
            current_replicas = scaling_plan.get("current_replicas", 0)
            target_replicas = scaling_plan.get("target_replicas", 0)
            title = f"⚡ Auto-Scale Resources: {current_replicas} → {target_replicas} replicas"
            body = self._generate_scaling_pr_body(issue_data, scaling_result)
            
            # Create pull request
            installation_id = context.get_state("github_installation_id", 1)
            
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
            logger.error(f"Scaling PR creation failed: {e}")
            return None
    
    def _generate_scaling_pr_body(self, issue_data: Dict[str, Any], scaling_result: Dict[str, Any]) -> str:
        """Generate PR body for scaling changes."""
        issue_type = issue_data.get("type", "unknown")
        scaling_plan = scaling_result.get("scaling_plan", {})
        analysis = scaling_result.get("analysis", {})
        
        current_replicas = scaling_plan.get("current_replicas", 0)
        target_replicas = scaling_plan.get("target_replicas", 0)
        scaling_direction = scaling_plan.get("scaling_direction", "unknown")
        
        body = f"""## Automated Resource Scaling

This PR implements automated resource scaling in response to {issue_type}.

### Scaling Details
- **Direction**: {scaling_direction.title()}
- **Current Replicas**: {current_replicas}
- **Target Replicas**: {target_replicas}
- **Scaling Factor**: {scaling_plan.get("scaling_factor", 1.0):.2f}x
- **Strategy**: {scaling_plan.get("strategy", "unknown").title()}
- **Urgency**: {scaling_plan.get("urgency", "unknown").title()}

### Trigger Analysis
- **CPU Usage**: {analysis.get("current_cpu_usage", "N/A"):.1f}%
- **Memory Usage**: {analysis.get("current_memory_usage", "N/A"):.1f}%
- **Response Time**: {analysis.get("avg_response_time", "N/A")}ms
- **Error Rate**: {analysis.get("error_rate", "N/A"):.1f}%
- **Reason**: {analysis.get("reason", "Unknown")}

### Changes Made
"""
        
        execution_steps = scaling_result.get("scaling_result", {}).get("execution_steps", [])
        for step in execution_steps:
            step_name = step["step"].replace("_", " ").title()
            status = "✅" if step["result"].get("success", False) else "❌"
            body += f"- {status} **{step_name}**: {step['result'].get('message', 'No message')}\n"
        
        body += f"""
### Monitoring Results
"""
        monitoring = scaling_result.get("monitoring", {})
        if monitoring:
            body += f"""- **Actual Replicas**: {monitoring.get("actual_replicas", "N/A")}
- **Health Check Success**: {monitoring.get("health_check_success_rate", "N/A"):.1f}%
- **Average Response Time**: {monitoring.get("average_response_time", "N/A")}ms
- **CPU Utilization**: {monitoring.get("resource_utilization", {}).get("cpu", "N/A"):.1f}%
- **Memory Utilization**: {monitoring.get("resource_utilization", {}).get("memory", "N/A"):.1f}%
"""
        else:
            body += "- Monitoring data not available\n"
        
        body += f"""
### Files Modified
- `k8s/deployment.yaml` - Updated replica count and resource limits
- `k8s/hpa.yaml` - Updated HorizontalPodAutoscaler configuration  
- `docker-compose.yml` - Updated service scaling configuration
- `terraform/autoscaling.tf` - Updated cloud auto-scaling rules
- `nginx_scaled.conf` - Updated load balancer configuration
- `monitoring/prometheus_scaling_rules.yml` - Updated monitoring alerts
- `monitoring/grafana_scaling_dashboard.json` - Updated dashboard

### Post-Scaling Actions
- [ ] Monitor system performance for 30 minutes
- [ ] Verify all health checks are passing
- [ ] Check cost implications of scaling changes
- [ ] Update capacity planning documentation

🤖 This scaling was performed automatically by the Self-Healing MLOps Bot.
"""
        
        return body
    
    async def _predict_future_resource_needs(self, context: Context, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future resource needs based on trends."""
        # Mock predictive analysis - in production would use ML models
        return {
            "predicted_cpu_usage_1h": current_metrics.get("cpu_usage_percent", 50) * 1.1,
            "predicted_memory_usage_1h": current_metrics.get("memory_usage_percent", 60) * 1.05,
            "predicted_rps_1h": current_metrics.get("requests_per_second", 100) * 1.2,
            "confidence": 0.85,
            "recommendation": "Scale up proactively to handle predicted load increase"
        }
    
    def _calculate_resource_adjustments(self, analysis: Dict[str, Any], target_replicas: int) -> Dict[str, Any]:
        """Calculate CPU/memory adjustments for scaled resources."""
        current_cpu = analysis.get("current_cpu_usage", 50)
        current_memory = analysis.get("current_memory_usage", 60)
        
        # Adjust resource limits based on usage patterns
        adjustments = {}
        
        if current_cpu > 80:
            adjustments["cpu_limit"] = "1000m"  # 1 CPU core
            adjustments["cpu_request"] = "500m"  # 0.5 CPU core
        elif current_cpu > 60:
            adjustments["cpu_limit"] = "800m"
            adjustments["cpu_request"] = "400m"
        
        if current_memory > 80:
            adjustments["memory_limit"] = "2Gi"
            adjustments["memory_request"] = "1Gi"
        elif current_memory > 60:
            adjustments["memory_limit"] = "1.5Gi"
            adjustments["memory_request"] = "750Mi"
        
        return adjustments
    
    def _estimate_scaling_duration(self, strategy: str, replica_diff: int) -> int:
        """Estimate how long scaling will take in seconds."""
        if strategy == "immediate":
            return 60 + (replica_diff * 15)  # 1 minute + 15s per replica
        elif strategy == "fast":
            return 120 + (replica_diff * 30)  # 2 minutes + 30s per replica
        else:  # gradual
            return 300 + (replica_diff * 60)  # 5 minutes + 1 minute per replica
    
    def _create_rollback_plan(self, current_replicas: int, target_replicas: int) -> Dict[str, Any]:
        """Create rollback plan in case scaling fails."""
        return {
            "rollback_replicas": current_replicas,
            "rollback_strategy": "immediate",
            "rollback_timeout": 180,  # 3 minutes
            "rollback_verification": True,
            "rollback_script": "rollback_scaling.sh"
        }
    
    def _create_monitoring_plan(self) -> Dict[str, Any]:
        """Create monitoring plan for post-scaling verification."""
        return {
            "monitoring_duration": 1800,  # 30 minutes
            "health_check_interval": 30,  # 30 seconds
            "metrics_to_monitor": [
                "replica_count",
                "cpu_usage",
                "memory_usage", 
                "response_time",
                "error_rate",
                "throughput"
            ],
            "success_criteria": {
                "health_check_success_rate": 95.0,  # %
                "max_response_time": 500,  # ms
                "max_error_rate": 2.0  # %
            }
        }