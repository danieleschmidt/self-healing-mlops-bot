"""GitHub webhook handler with secure verification and event routing."""

import asyncio
import hmac
import hashlib
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import HTTPException, Request, BackgroundTasks, status
from pydantic import BaseModel, Field
import structlog

from ..core.bot import SelfHealingBot
from ..core.config import config
from ..monitoring.logging import get_logger, audit_logger, performance_logger
from ..monitoring.metrics import prometheus_metrics


logger = get_logger(__name__)


class WebhookPayload(BaseModel):
    """GitHub webhook payload model with validation."""
    action: Optional[str] = None
    repository: Optional[Dict[str, Any]] = None
    workflow_run: Optional[Dict[str, Any]] = None
    check_run: Optional[Dict[str, Any]] = None
    installation: Optional[Dict[str, Any]] = None
    sender: Optional[Dict[str, Any]] = None
    organization: Optional[Dict[str, Any]] = None


class WebhookResponse(BaseModel):
    """Webhook response model."""
    status: str
    event_type: str
    repo_full_name: Optional[str] = None
    execution_id: Optional[str] = None
    message: str
    timestamp: str


class WebhookHandler:
    """Secure GitHub webhook handler with event routing and processing."""
    
    def __init__(self, bot: SelfHealingBot):
        self.bot = bot
        self.webhook_secret = config.github_webhook_secret
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._background_processor_task: Optional[asyncio.Task] = None
        self._supported_events = {
            "push", "pull_request", "workflow_run", "check_run",
            "issues", "issue_comment", "repository", "installation",
            "installation_repositories", "ping"
        }
        
        # Start background processor
        self._start_background_processor()
    
    def _start_background_processor(self):
        """Start background webhook processor."""
        if not self._background_processor_task or self._background_processor_task.done():
            self._background_processor_task = asyncio.create_task(
                self._background_webhook_processor()
            )
            logger.info("Started background webhook processor")
    
    async def _background_webhook_processor(self):
        """Background processor for webhook events."""
        logger.info("Background webhook processor started")
        
        while True:
            try:
                # Get webhook event from queue with timeout
                webhook_data = await asyncio.wait_for(
                    self._processing_queue.get(),
                    timeout=30.0
                )
                
                await self._process_webhook_event(
                    webhook_data["event_type"],
                    webhook_data["payload"],
                    webhook_data["metadata"]
                )
                
                self._processing_queue.task_done()
                
            except asyncio.TimeoutError:
                # Timeout is expected when queue is empty
                continue
            except asyncio.CancelledError:
                logger.info("Background webhook processor cancelled")
                break
            except Exception as e:
                logger.exception(
                    "Error in background webhook processor",
                    error=str(e)
                )
                # Continue processing other events
                continue
    
    async def handle_webhook(self, request: Request) -> Dict[str, Any]:
        """Handle incoming GitHub webhook with security verification."""
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        try:
            # Verify webhook signature
            await self._verify_webhook_signature(request)
            
            # Extract event information
            event_type = request.headers.get("X-GitHub-Event")
            if not event_type:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing X-GitHub-Event header"
                )
            
            # Validate event type
            if event_type not in self._supported_events:
                logger.warning(
                    "Unsupported webhook event type",
                    event_type=event_type,
                    client_ip=client_ip
                )
                return {
                    "status": "ignored",
                    "event_type": event_type,
                    "message": f"Event type '{event_type}' not supported",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Parse webhook payload
            try:
                payload = await request.json()
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON payload: {str(e)}"
                )
            
            # Validate payload structure
            validated_payload = WebhookPayload(**payload)
            
            # Extract repository information
            repo_data = payload.get("repository", {})
            repo_full_name = repo_data.get("full_name", "unknown/unknown")
            
            # Handle ping event immediately
            if event_type == "ping":
                return {
                    "status": "pong",
                    "event_type": event_type,
                    "repo_full_name": repo_full_name,
                    "message": "Webhook endpoint is working correctly",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Log webhook received
            logger.info(
                "Webhook received",
                event_type=event_type,
                repo=repo_full_name,
                client_ip=client_ip
            )
            
            # Add to processing queue for background processing
            webhook_metadata = {
                "client_ip": client_ip,
                "received_at": datetime.utcnow().isoformat(),
                "delivery_id": request.headers.get("X-GitHub-Delivery", "unknown")
            }
            
            await self._processing_queue.put({
                "event_type": event_type,
                "payload": payload,
                "metadata": webhook_metadata
            })
            
            # Log audit event
            audit_logger.log_security_event(
                "webhook_received", "info",
                {
                    "event_type": event_type,
                    "repo": repo_full_name,
                    "client_ip": client_ip,
                    "delivery_id": webhook_metadata["delivery_id"]
                }
            )
            
            # Track performance
            performance_logger.log_execution_time(
                "webhook_handling",
                time.time() - start_time,
                True,
                event_type=event_type,
                repo=repo_full_name
            )
            
            return {
                "status": "accepted",
                "event_type": event_type,
                "repo_full_name": repo_full_name,
                "message": "Webhook queued for processing",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except HTTPException:
            # Re-raise HTTP exceptions
            performance_logger.log_execution_time(
                "webhook_handling",
                time.time() - start_time,
                False,
                client_ip=client_ip
            )
            raise
        except Exception as e:
            performance_logger.log_execution_time(
                "webhook_handling",
                time.time() - start_time,
                False,
                error=str(e),
                client_ip=client_ip
            )
            logger.exception(
                "Unexpected error processing webhook",
                error=str(e),
                client_ip=client_ip
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def _verify_webhook_signature(self, request: Request) -> bool:
        """Verify GitHub webhook signature for security."""
        if not self.webhook_secret:
            if config.environment == "development":
                logger.warning(
                    "Webhook secret not configured, skipping signature verification in development"
                )
                return True
            else:
                audit_logger.log_security_event(
                    "webhook_signature_missing", "critical",
                    {"environment": config.environment}
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Webhook secret not configured"
                )
        
        # Get signature from headers
        signature = request.headers.get("X-Hub-Signature-256")
        if not signature:
            audit_logger.log_security_event(
                "webhook_signature_missing", "warning",
                {"client_ip": request.client.host if request.client else "unknown"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing webhook signature"
            )
        
        # Verify signature
        try:
            body = await request.body()
            expected_signature = "sha256=" + hmac.new(
                self.webhook_secret.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                audit_logger.log_security_event(
                    "webhook_signature_invalid", "critical",
                    {
                        "client_ip": request.client.host if request.client else "unknown",
                        "provided_signature": signature[:20] + "..."  # Log partial signature
                    }
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature"
                )
            
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error verifying webhook signature", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Signature verification failed"
            )
    
    async def _process_webhook_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        """Process webhook event in background."""
        start_time = time.time()
        repo_full_name = payload.get("repository", {}).get("full_name", "unknown/unknown")
        execution_id = None
        
        try:
            logger.info(
                "Processing webhook event",
                event_type=event_type,
                repo=repo_full_name,
                delivery_id=metadata.get("delivery_id")
            )
            
            # Filter events that should trigger bot processing
            if not self._should_process_event(event_type, payload):
                logger.debug(
                    "Event filtered out",
                    event_type=event_type,
                    repo=repo_full_name,
                    reason="Event does not require bot processing"
                )
                return
            
            # Process event through bot
            context = await self.bot.process_event(event_type, payload)
            
            if context:
                execution_id = context.execution_id
                logger.info(
                    "Webhook event processed successfully",
                    event_type=event_type,
                    repo=repo_full_name,
                    execution_id=execution_id
                )
                
                # Log successful processing
                audit_logger.log_security_event(
                    "webhook_processed", "info",
                    {
                        "event_type": event_type,
                        "repo": repo_full_name,
                        "execution_id": execution_id,
                        "delivery_id": metadata.get("delivery_id")
                    }
                )
            else:
                logger.warning(
                    "No context returned from bot processing",
                    event_type=event_type,
                    repo=repo_full_name
                )
            
            # Track metrics
            prometheus_metrics.webhook_events_processed.labels(
                event_type=event_type,
                status="success"
            ).inc()
            
            performance_logger.log_execution_time(
                "webhook_event_processing",
                time.time() - start_time,
                True,
                event_type=event_type,
                repo=repo_full_name,
                execution_id=execution_id
            )
            
        except Exception as e:
            logger.exception(
                "Error processing webhook event",
                event_type=event_type,
                repo=repo_full_name,
                error=str(e),
                delivery_id=metadata.get("delivery_id")
            )
            
            # Log error in audit trail
            audit_logger.log_security_event(
                "webhook_processing_error", "error",
                {
                    "event_type": event_type,
                    "repo": repo_full_name,
                    "error": str(e),
                    "delivery_id": metadata.get("delivery_id")
                }
            )
            
            # Track error metrics
            prometheus_metrics.webhook_events_processed.labels(
                event_type=event_type,
                status="error"
            ).inc()
            
            performance_logger.log_execution_time(
                "webhook_event_processing",
                time.time() - start_time,
                False,
                event_type=event_type,
                repo=repo_full_name,
                error=str(e)
            )
    
    def _should_process_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """Determine if an event should be processed by the bot."""
        
        # Always ignore ping events after responding
        if event_type == "ping":
            return False
        
        # Process workflow run failures
        if event_type == "workflow_run":
            workflow_run = payload.get("workflow_run", {})
            conclusion = workflow_run.get("conclusion")
            return conclusion in ["failure", "cancelled", "timed_out"]
        
        # Process check run failures
        if event_type == "check_run":
            check_run = payload.get("check_run", {})
            conclusion = check_run.get("conclusion")
            return conclusion in ["failure", "cancelled", "timed_out"]
        
        # Process push events to main/master branches
        if event_type == "push":
            ref = payload.get("ref", "")
            return ref in ["refs/heads/main", "refs/heads/master"]
        
        # Process pull request events (opened, synchronize, closed)
        if event_type == "pull_request":
            action = payload.get("action")
            return action in ["opened", "synchronize", "closed", "ready_for_review"]
        
        # Process repository events (created, archived, etc.)
        if event_type == "repository":
            action = payload.get("action")
            return action in ["created", "archived", "unarchived"]
        
        # Process installation events for app management
        if event_type in ["installation", "installation_repositories"]:
            return True
        
        # Process issue-related events
        if event_type in ["issues", "issue_comment"]:
            action = payload.get("action")
            return action in ["opened", "created", "edited"]
        
        # Default: don't process unknown events
        logger.debug(
            "Event type not configured for processing",
            event_type=event_type,
            payload_keys=list(payload.keys())
        )
        return False
    
    async def get_processing_queue_size(self) -> int:
        """Get current processing queue size."""
        return self._processing_queue.qsize()
    
    async def shutdown(self):
        """Gracefully shutdown the webhook handler."""
        logger.info("Shutting down webhook handler")
        
        # Cancel background processor
        if self._background_processor_task and not self._background_processor_task.done():
            self._background_processor_task.cancel()
            try:
                await self._background_processor_task
            except asyncio.CancelledError:
                pass
        
        # Wait for queue to be processed
        if not self._processing_queue.empty():
            logger.info(
                "Waiting for webhook processing queue to empty",
                queue_size=self._processing_queue.qsize()
            )
            await self._processing_queue.join()
        
        logger.info("Webhook handler shutdown complete")