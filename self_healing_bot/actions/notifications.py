"""Notification actions for alerting and communication."""

import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .base import BaseAction, ActionResult
from ..core.context import Context

logger = logging.getLogger(__name__)


class SlackNotificationAction(BaseAction):
    """Send notifications to Slack channels."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.webhook_url = self.config.get("webhook_url")
        self.default_channel = self.config.get("default_channel", "#ml-ops-alerts")
        self.username = self.config.get("username", "Self-Healing Bot")
        self.emoji = self.config.get("emoji", ":robot_face:")
    
    def can_handle(self, issue_type: str) -> bool:
        return True  # Can notify about any issue type
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Send Slack notification."""
        try:
            if not self.webhook_url:
                return self.create_result(
                    success=False,
                    message="Slack webhook URL not configured"
                )
            
            # Build notification message
            message = self._build_slack_message(context, issue_data)
            
            # In a real implementation, this would make an HTTP POST to Slack
            # For now, we'll just log the message and store it
            logger.info(f"Slack notification: {message['text']}")
            
            # Store notification for tracking
            notification_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "platform": "slack",
                "channel": message.get("channel", self.default_channel),
                "message": message["text"],
                "context": context.to_dict()
            }
            
            context.set_state("slack_notification", notification_record)
            
            return self.create_result(
                success=True,
                message=f"Slack notification sent to {message.get('channel', self.default_channel)}",
                data={"notification": notification_record}
            )
            
        except Exception as e:
            logger.exception(f"Slack notification failed: {e}")
            return self.create_result(
                success=False,
                message=f"Slack notification failed: {str(e)}"
            )
    
    def _build_slack_message(self, context: Context, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build Slack message payload."""
        issue_type = issue_data.get("type", "unknown")
        severity = issue_data.get("severity", "medium")
        message_text = issue_data.get("message", "Issue detected")
        
        # Select channel based on severity
        channel = self._get_channel_for_severity(severity)
        
        # Build message
        emoji_map = {
            "critical": ":red_circle:",
            "high": ":orange_circle:",
            "medium": ":yellow_circle:",
            "low": ":green_circle:"
        }
        
        severity_emoji = emoji_map.get(severity, ":white_circle:")
        
        message = {
            "channel": channel,
            "username": self.username,
            "icon_emoji": self.emoji,
            "text": f"{severity_emoji} *{severity.upper()} Alert*: {message_text}",
            "attachments": [
                {
                    "color": self._get_color_for_severity(severity),
                    "fields": [
                        {
                            "title": "Repository",
                            "value": context.repo_full_name,
                            "short": True
                        },
                        {
                            "title": "Issue Type",
                            "value": issue_type.replace("_", " ").title(),
                            "short": True
                        },
                        {
                            "title": "Event Type",
                            "value": context.event_type,
                            "short": True
                        },
                        {
                            "title": "Execution ID",
                            "value": context.execution_id,
                            "short": True
                        }
                    ],
                    "footer": "Self-Healing MLOps Bot",
                    "ts": int(datetime.utcnow().timestamp())
                }
            ]
        }
        
        # Add issue-specific details
        if issue_type == "data_drift":
            drift_score = issue_data.get("drift_score", 0)
            message["attachments"][0]["fields"].append({
                "title": "Drift Score",
                "value": f"{drift_score:.3f}",
                "short": True
            })
        elif issue_type == "model_degradation":
            degradation_pct = issue_data.get("degradation_percentage", 0)
            message["attachments"][0]["fields"].append({
                "title": "Degradation",
                "value": f"{degradation_pct:.1f}%",
                "short": True
            })
        
        return message
    
    def _get_channel_for_severity(self, severity: str) -> str:
        """Get appropriate Slack channel based on severity."""
        channel_map = {
            "critical": "#ml-ops-critical",
            "high": "#ml-ops-alerts",
            "medium": "#ml-ops-alerts",
            "low": "#ml-ops-activity"
        }
        return channel_map.get(severity, self.default_channel)
    
    def _get_color_for_severity(self, severity: str) -> str:
        """Get color code for severity level."""
        color_map = {
            "critical": "#ff0000",  # Red
            "high": "#ff8800",      # Orange
            "medium": "#ffaa00",    # Yellow
            "low": "#00aa00"        # Green
        }
        return color_map.get(severity, "#808080")  # Gray default


class IssueCreationAction(BaseAction):
    """Create GitHub issues for tracking problems."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.auto_assign = self.config.get("auto_assign", True)
        self.default_assignee = self.config.get("default_assignee")
        self.label_prefix = self.config.get("label_prefix", "ml-ops")
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["model_degradation", "data_drift", "pipeline_failure"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Create GitHub issue for the problem."""
        try:
            issue_details = self._build_issue_details(context, issue_data)
            
            # In a real implementation, this would use GitHub API
            # For now, we'll simulate issue creation
            mock_issue = {
                "number": 123,
                "title": issue_details["title"],
                "body": issue_details["body"],
                "labels": issue_details["labels"],
                "assignees": issue_details.get("assignees", []),
                "created_at": datetime.utcnow().isoformat(),
                "html_url": f"https://github.com/{context.repo_full_name}/issues/123"
            }
            
            context.set_state("created_issue", mock_issue)
            
            return self.create_result(
                success=True,
                message=f"Created GitHub issue #{mock_issue['number']}: {issue_details['title']}",
                data={"issue": mock_issue}
            )
            
        except Exception as e:
            logger.exception(f"Issue creation failed: {e}")
            return self.create_result(
                success=False,
                message=f"Issue creation failed: {str(e)}"
            )
    
    def _build_issue_details(self, context: Context, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build GitHub issue details."""
        issue_type = issue_data.get("type", "unknown")
        severity = issue_data.get("severity", "medium")
        message = issue_data.get("message", "Issue detected")
        
        # Generate title
        title = f"[{severity.upper()}] {issue_type.replace('_', ' ').title()}: {message}"
        
        # Generate body
        body = self._generate_issue_body(context, issue_data)
        
        # Generate labels
        labels = self._generate_labels(issue_type, severity)
        
        # Assign if configured
        assignees = []
        if self.auto_assign and self.default_assignee:
            assignees = [self.default_assignee]
        
        return {
            "title": title,
            "body": body,
            "labels": labels,
            "assignees": assignees
        }
    
    def _generate_issue_body(self, context: Context, issue_data: Dict[str, Any]) -> str:
        """Generate comprehensive issue body."""
        issue_type = issue_data.get("type", "unknown")
        
        body = f"""## Problem Description
{issue_data.get('message', 'Issue detected by self-healing MLOps bot')}

## Issue Details
- **Type**: {issue_type.replace('_', ' ').title()}
- **Severity**: {issue_data.get('severity', 'medium').title()}
- **Repository**: {context.repo_full_name}
- **Event Type**: {context.event_type}
- **Detection Time**: {datetime.utcnow().isoformat()}
- **Execution ID**: {context.execution_id}

"""
        
        # Add issue-specific details
        if issue_type == "data_drift":
            body += f"""## Data Drift Details
- **Drift Score**: {issue_data.get('drift_score', 0):.3f}
- **Threshold**: {issue_data.get('threshold', 0.1):.3f}
- **Detection Method**: {issue_data.get('detection_method', 'unknown')}
- **Affected Features**: {', '.join(issue_data.get('affected_features', []))}
- **Recommendation**: {issue_data.get('recommendation', 'Monitor and consider retraining')}

"""
        elif issue_type == "model_degradation":
            body += f"""## Model Degradation Details
- **Metric**: {issue_data.get('metric_name', 'unknown')}
- **Current Value**: {issue_data.get('current_value', 'unknown')}
- **Baseline Value**: {issue_data.get('baseline_value', 'unknown')}
- **Degradation**: {issue_data.get('degradation_percentage', 0):.1f}%
- **Recommendation**: {issue_data.get('recommendation', 'Investigate and consider rollback')}

"""
        elif issue_type == "pipeline_failure":
            body += f"""## Pipeline Failure Details
- **Workflow**: {issue_data.get('workflow_name', 'unknown')}
- **Failure Type**: {issue_data.get('failure_type', 'unknown')}
- **Run URL**: {issue_data.get('run_url', 'N/A')}
- **Head SHA**: {issue_data.get('head_sha', 'N/A')}

"""
        
        # Add automated actions section
        body += f"""## Automated Actions Attempted
_The self-healing bot will attempt to resolve this issue automatically and update this issue with results._

## Next Steps
- [ ] Review automated fix attempts
- [ ] Manual investigation if automated fixes fail
- [ ] Update monitoring thresholds if needed
- [ ] Close issue when resolved

---
*This issue was automatically created by the Self-Healing MLOps Bot*
"""
        
        return body
    
    def _generate_labels(self, issue_type: str, severity: str) -> List[str]:
        """Generate appropriate labels for the issue."""
        labels = [
            f"{self.label_prefix}:auto-generated",
            f"{self.label_prefix}:{issue_type.replace('_', '-')}",
            f"severity:{severity}",
            "ml-ops"
        ]
        
        # Add type-specific labels
        if issue_type == "data_drift":
            labels.append("data-quality")
        elif issue_type == "model_degradation":
            labels.append("model-performance")
        elif issue_type == "pipeline_failure":
            labels.append("ci-cd")
        
        return labels


class EmailNotificationAction(BaseAction):
    """Send email notifications for critical issues."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.smtp_config = self.config.get("smtp_config", {})
        self.default_recipients = self.config.get("default_recipients", [])
        self.critical_recipients = self.config.get("critical_recipients", [])
    
    def can_handle(self, issue_type: str) -> bool:
        return True
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Send email notification."""
        try:
            severity = issue_data.get("severity", "medium")
            
            # Only send emails for high/critical issues
            if severity not in ["high", "critical"]:
                return self.create_result(
                    success=True,
                    message=f"No email notification needed for {severity} severity"
                )
            
            # Build email
            email_details = self._build_email(context, issue_data)
            
            # In a real implementation, this would send actual email via SMTP
            # For now, simulate email sending
            email_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "recipients": email_details["recipients"],
                "subject": email_details["subject"],
                "body_preview": email_details["body"][:200] + "...",
                "sent": True
            }
            
            context.set_state("email_notification", email_record)
            
            return self.create_result(
                success=True,
                message=f"Email notification sent to {len(email_details['recipients'])} recipients",
                data={"email": email_record}
            )
            
        except Exception as e:
            logger.exception(f"Email notification failed: {e}")
            return self.create_result(
                success=False,
                message=f"Email notification failed: {str(e)}"
            )
    
    def _build_email(self, context: Context, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build email notification details."""
        issue_type = issue_data.get("type", "unknown")
        severity = issue_data.get("severity", "medium")
        message = issue_data.get("message", "Issue detected")
        
        # Select recipients based on severity
        recipients = self.critical_recipients if severity == "critical" else self.default_recipients
        
        # Build subject
        subject = f"[{severity.upper()}] ML Pipeline Alert: {issue_type.replace('_', ' ').title()} - {context.repo_full_name}"
        
        # Build body
        body = f"""
ML Pipeline Alert - {severity.upper()} Severity

Repository: {context.repo_full_name}
Issue Type: {issue_type.replace('_', ' ').title()}
Message: {message}
Detection Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
Execution ID: {context.execution_id}

The self-healing MLOps bot detected an issue in your ML pipeline and is attempting automated repairs.

Issue Details:
{json.dumps(issue_data, indent=2)}

You can monitor the automated resolution progress in the repository's Actions tab.

--
This email was sent by the Self-Healing MLOps Bot
"""
        
        return {
            "recipients": recipients,
            "subject": subject,
            "body": body
        }


class WebhookNotificationAction(BaseAction):
    """Send notifications via custom webhooks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.webhook_url = self.config.get("webhook_url")
        self.custom_headers = self.config.get("custom_headers", {})
    
    def can_handle(self, issue_type: str) -> bool:
        return bool(self.webhook_url)
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Send webhook notification."""
        try:
            if not self.webhook_url:
                return self.create_result(
                    success=False,
                    message="Webhook URL not configured"
                )
            
            # Build webhook payload
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "repository": context.repo_full_name,
                "event_type": context.event_type,
                "execution_id": context.execution_id,
                "issue": issue_data,
                "bot_version": "1.0.0"
            }
            
            # In a real implementation, this would make HTTP POST request
            # For now, simulate webhook sending
            webhook_record = {
                "url": self.webhook_url,
                "payload": payload,
                "headers": self.custom_headers,
                "sent_at": datetime.utcnow().isoformat(),
                "status": "success"
            }
            
            context.set_state("webhook_notification", webhook_record)
            
            return self.create_result(
                success=True,
                message=f"Webhook notification sent to {self.webhook_url}",
                data={"webhook": webhook_record}
            )
            
        except Exception as e:
            logger.exception(f"Webhook notification failed: {e}")
            return self.create_result(
                success=False,
                message=f"Webhook notification failed: {str(e)}"
            )