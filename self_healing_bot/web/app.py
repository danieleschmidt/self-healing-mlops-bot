"""FastAPI web application."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hmac
import hashlib

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..core.bot import SelfHealingBot
from ..core.config import config

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Self-Healing MLOps Bot",
    description="Autonomous ML Pipeline Repair and Drift Detection",
    version="1.0.0",
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.debug else ["https://*.github.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global bot instance
bot = SelfHealingBot()


# Pydantic models
class WebhookPayload(BaseModel):
    """GitHub webhook payload model."""
    action: Optional[str] = None
    repository: Optional[Dict[str, Any]] = None
    workflow_run: Optional[Dict[str, Any]] = None
    check_run: Optional[Dict[str, Any]] = None
    installation: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    components: Dict[str, str]
    active_executions: int


class ExecutionStatusResponse(BaseModel):
    """Execution status response model."""
    execution_id: str
    repo: str
    event_type: str
    started_at: str
    has_error: bool
    state: Dict[str, Any]


# Dependency functions
async def verify_webhook_signature(request: Request) -> bool:
    """Verify GitHub webhook signature."""
    if not config.github_webhook_secret:
        logger.warning("Webhook secret not configured, skipping signature verification")
        return True
    
    signature = request.headers.get("X-Hub-Signature-256")
    if not signature:
        raise HTTPException(status_code=401, detail="Missing signature")
    
    body = await request.body()
    expected_signature = "sha256=" + hmac.new(
        config.github_webhook_secret.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    return True


# Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Self-Healing MLOps Bot API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    health_data = await bot.health_check()
    return HealthResponse(**health_data)


@app.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    signature_valid: bool = Depends(verify_webhook_signature)
):
    """Handle GitHub webhook events."""
    try:
        # Get event type from headers
        event_type = request.headers.get("X-GitHub-Event")
        if not event_type:
            raise HTTPException(status_code=400, detail="Missing event type header")
        
        # Parse webhook payload
        payload = await request.json()
        
        # Log webhook received
        repo_name = payload.get("repository", {}).get("full_name", "unknown")
        logger.info(f"Received {event_type} webhook for {repo_name}")
        
        # Process event in background
        background_tasks.add_task(process_webhook_event, event_type, payload)
        
        return {"status": "accepted", "event_type": event_type}
        
    except Exception as e:
        logger.exception(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def process_webhook_event(event_type: str, payload: Dict[str, Any]):
    """Process webhook event in background."""
    try:
        context = await bot.process_event(event_type, payload)
        if context:
            logger.info(f"Successfully processed {event_type} event for {context.repo_full_name}")
        else:
            logger.warning(f"No context returned for {event_type} event")
    except Exception as e:
        logger.exception(f"Error in background webhook processing: {e}")


@app.get("/executions", response_model=List[ExecutionStatusResponse])
async def list_executions():
    """List active executions."""
    executions = bot.list_active_executions()
    return [ExecutionStatusResponse(**execution) for execution in executions if execution]


@app.get("/executions/{execution_id}", response_model=ExecutionStatusResponse)
async def get_execution_status(execution_id: str):
    """Get status of specific execution."""
    status = bot.get_execution_status(execution_id)
    if not status:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    return ExecutionStatusResponse(**status)


@app.post("/manual-trigger")
async def manual_trigger(
    repo_full_name: str,
    event_type: str = "manual",
    event_data: Dict[str, Any] = None
):
    """Manually trigger bot processing for a repository."""
    try:
        # Create mock event data
        mock_payload = {
            "repository": {
                "full_name": repo_full_name,
                "name": repo_full_name.split("/")[-1],
                "owner": {"login": repo_full_name.split("/")[0]}
            },
            **(event_data or {})
        }
        
        context = await bot.process_event(event_type, mock_payload)
        
        return {
            "status": "completed",
            "execution_id": context.execution_id if context else None,
            "repo": repo_full_name,
            "event_type": event_type
        }
        
    except Exception as e:
        logger.exception(f"Error in manual trigger: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get bot metrics for monitoring."""
    health_data = await bot.health_check()
    
    # Add additional metrics
    metrics = {
        "bot_health": health_data,
        "uptime_seconds": "unknown",  # TODO: Implement uptime tracking
        "total_events_processed": "unknown",  # TODO: Implement event counter
        "successful_repairs": "unknown",  # TODO: Implement repair counter
        "failed_repairs": "unknown"  # TODO: Implement failure counter
    }
    
    return metrics


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.exception(f"Unhandled exception in {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("Starting Self-Healing MLOps Bot API")
    
    # Perform initial health check
    try:
        health_data = await bot.health_check()
        logger.info(f"Bot health check: {health_data['status']}")
    except Exception as e:
        logger.error(f"Initial health check failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("Shutting down Self-Healing MLOps Bot API")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "self_healing_bot.web.app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower()
    )