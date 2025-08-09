"""FastAPI web application with production-ready features."""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import hmac
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import structlog

from ..core.bot import SelfHealingBot
from ..core.config import config
from ..monitoring.logging import setup_logging, get_logger, audit_logger, performance_logger
from ..monitoring.metrics import prometheus_metrics
from ..security.validation import RateLimiter, SecurityValidator
from .webhooks import WebhookHandler
from .middleware import add_middleware

# Setup structured logging
setup_logging()
logger = get_logger(__name__)

# Global state
app_state = {
    "bot": None,
    "webhook_handler": None,
    "rate_limiter": None,
    "security_validator": None,
    "startup_time": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Self-Healing MLOps Bot API")
    app_state["startup_time"] = time.time()
    
    # Initialize components
    app_state["bot"] = SelfHealingBot()
    app_state["webhook_handler"] = WebhookHandler(app_state["bot"])
    app_state["rate_limiter"] = RateLimiter(
        requests_per_minute=config.rate_limit_per_minute
    )
    app_state["security_validator"] = SecurityValidator()
    
    # Perform initial health check
    try:
        health_data = await app_state["bot"].health_check()
        logger.info("Bot health check passed", status=health_data["status"])
        audit_logger.log_security_event(
            "startup", "info", {"health_status": health_data["status"]}
        )
    except Exception as e:
        logger.error("Initial health check failed", error=str(e))
        audit_logger.log_security_event(
            "startup_failure", "critical", {"error": str(e)}
        )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Self-Healing MLOps Bot API")
    audit_logger.log_security_event("shutdown", "info", {})

# Initialize FastAPI app with lifespan management
app = FastAPI(
    title="Self-Healing MLOps Bot",
    description="Autonomous ML Pipeline Repair and Drift Detection",
    version="1.0.0",
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None,
    lifespan=lifespan
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if config.debug else ["localhost", "*.github.com", "*.githubapp.com"]
)

# Add custom middleware (includes CORS, security, logging, etc.)
add_middleware(app)

# Security scheme for protected endpoints
security = HTTPBearer(auto_error=False)



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
    uptime_seconds: float
    version: str = "1.0.0"


class ExecutionStatusResponse(BaseModel):
    """Execution status response model."""
    execution_id: str
    repo: str
    event_type: str
    started_at: str
    has_error: bool
    state: Dict[str, Any]
    duration_seconds: Optional[float] = None

class MetricsResponse(BaseModel):
    """Metrics response model."""
    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    active_executions: int
    bot_health: Dict[str, Any]
    rate_limit_stats: Dict[str, Any]

class ManualTriggerRequest(BaseModel):
    """Manual trigger request model."""
    repo_full_name: str = Field(..., description="Repository full name (owner/repo)")
    event_type: str = Field(default="manual", description="Event type to simulate")
    event_data: Optional[Dict[str, Any]] = Field(default=None, description="Additional event data")


# Dependency functions
async def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance."""
    return app_state["rate_limiter"]

async def get_security_validator() -> SecurityValidator:
    """Get security validator instance."""
    return app_state["security_validator"]

async def get_bot() -> SelfHealingBot:
    """Get bot instance."""
    if not app_state["bot"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bot service not available"
        )
    return app_state["bot"]

async def get_webhook_handler() -> WebhookHandler:
    """Get webhook handler instance."""
    if not app_state["webhook_handler"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Webhook handler not available"
        )
    return app_state["webhook_handler"]

async def verify_rate_limit(request: Request, rate_limiter: RateLimiter = Depends(get_rate_limiter)) -> bool:
    """Verify request is within rate limits."""
    client_ip = request.client.host if request.client else "unknown"
    
    if not await rate_limiter.check_rate_limit(client_ip):
        audit_logger.log_security_event(
            "rate_limit_exceeded", "warning", {"client_ip": client_ip}
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    return True

async def validate_security(
    request: Request,
    security_validator: SecurityValidator = Depends(get_security_validator)
) -> bool:
    """Validate request security."""
    try:
        await security_validator.validate_request(request)
        return True
    except Exception as e:
        audit_logger.log_security_event(
            "security_validation_failed", "warning", 
            {"error": str(e), "path": str(request.url.path)}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Security validation failed"
        )


# Routes
@app.get("/", tags=["system"])
async def root():
    """Root endpoint with basic API information."""
    uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
    
    return {
        "message": "Self-Healing MLOps Bot API",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": uptime,
        "environment": config.environment,
        "documentation": "/docs" if config.debug else None
    }

@app.get("/ping", response_class=PlainTextResponse, tags=["system"])
async def ping():
    """Simple ping endpoint for load balancer health checks."""
    return "pong"


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(bot: SelfHealingBot = Depends(get_bot)):
    """Comprehensive health check endpoint."""
    start_time = time.time()
    
    try:
        health_data = await bot.health_check()
        uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
        
        health_response = HealthResponse(
            **health_data,
            uptime_seconds=uptime
        )
        
        # Log performance metrics
        performance_logger.log_execution_time(
            "health_check",
            time.time() - start_time,
            True
        )
        
        return health_response
        
    except Exception as e:
        performance_logger.log_execution_time(
            "health_check",
            time.time() - start_time,
            False,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


@app.post("/webhook", tags=["webhooks"])
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    _rate_limit: bool = Depends(verify_rate_limit),
    _security: bool = Depends(validate_security),
    webhook_handler: WebhookHandler = Depends(get_webhook_handler)
):
    """Handle GitHub webhook events with enhanced security and monitoring."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        # Process webhook through dedicated handler
        response = await webhook_handler.handle_webhook(request)
        
        # Track metrics
        prometheus_metrics.webhook_requests_total.labels(
            event_type=response.get("event_type", "unknown"),
            status="success"
        ).inc()
        
        performance_logger.log_execution_time(
            "webhook_processing",
            time.time() - start_time,
            True,
            event_type=response.get("event_type"),
            client_ip=client_ip
        )
        
        return response
        
    except HTTPException:
        prometheus_metrics.webhook_requests_total.labels(
            event_type="unknown",
            status="error"
        ).inc()
        raise
    except Exception as e:
        prometheus_metrics.webhook_requests_total.labels(
            event_type="unknown",
            status="error"
        ).inc()
        
        performance_logger.log_execution_time(
            "webhook_processing",
            time.time() - start_time,
            False,
            error=str(e),
            client_ip=client_ip
        )
        
        logger.exception("Error processing webhook", error=str(e), client_ip=client_ip)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )




@app.get("/executions", response_model=List[ExecutionStatusResponse], tags=["executions"])
async def list_executions(
    bot: SelfHealingBot = Depends(get_bot),
    _rate_limit: bool = Depends(verify_rate_limit)
):
    """List active executions."""
    try:
        executions = bot.list_active_executions()
        return [ExecutionStatusResponse(**execution) for execution in executions if execution]
    except Exception as e:
        logger.error("Error listing executions", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list executions"
        )


@app.get("/executions/{execution_id}", response_model=ExecutionStatusResponse, tags=["executions"])
async def get_execution_status(
    execution_id: str,
    bot: SelfHealingBot = Depends(get_bot),
    _rate_limit: bool = Depends(verify_rate_limit)
):
    """Get status of specific execution."""
    try:
        execution_status = bot.get_execution_status(execution_id)
        if not execution_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Execution not found"
            )
        
        return ExecutionStatusResponse(**execution_status)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting execution status", execution_id=execution_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get execution status"
        )


@app.post("/manual-trigger", tags=["operations"])
async def manual_trigger(
    request: ManualTriggerRequest,
    background_tasks: BackgroundTasks,
    bot: SelfHealingBot = Depends(get_bot),
    _rate_limit: bool = Depends(verify_rate_limit),
    _security: bool = Depends(validate_security)
):
    """Manually trigger bot processing for a repository."""
    start_time = time.time()
    
    try:
        # Validate repository name format
        if "/" not in request.repo_full_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Repository name must be in format 'owner/repo'"
            )
        
        # Create mock event data
        mock_payload = {
            "repository": {
                "full_name": request.repo_full_name,
                "name": request.repo_full_name.split("/")[-1],
                "owner": {"login": request.repo_full_name.split("/")[0]}
            },
            **(request.event_data or {})
        }
        
        # Log manual trigger
        audit_logger.log_security_event(
            "manual_trigger", "info",
            {
                "repo": request.repo_full_name,
                "event_type": request.event_type
            }
        )
        
        # Process in background to avoid timeout
        background_tasks.add_task(
            process_manual_trigger,
            request.event_type,
            mock_payload,
            bot
        )
        
        performance_logger.log_execution_time(
            "manual_trigger_request",
            time.time() - start_time,
            True,
            repo=request.repo_full_name
        )
        
        return {
            "status": "accepted",
            "repo": request.repo_full_name,
            "event_type": request.event_type,
            "message": "Manual trigger request accepted and processing in background"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        performance_logger.log_execution_time(
            "manual_trigger_request",
            time.time() - start_time,
            False,
            error=str(e)
        )
        logger.exception("Error in manual trigger", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Manual trigger failed"
        )

async def process_manual_trigger(
    event_type: str,
    mock_payload: Dict[str, Any],
    bot: SelfHealingBot
):
    """Process manual trigger in background."""
    start_time = time.time()
    repo_name = mock_payload.get("repository", {}).get("full_name", "unknown")
    
    try:
        context = await bot.process_event(event_type, mock_payload)
        
        performance_logger.log_execution_time(
            "manual_trigger_execution",
            time.time() - start_time,
            True,
            repo=repo_name,
            execution_id=context.execution_id if context else None
        )
        
        logger.info(
            "Manual trigger completed successfully",
            repo=repo_name,
            event_type=event_type,
            execution_id=context.execution_id if context else None
        )
        
    except Exception as e:
        performance_logger.log_execution_time(
            "manual_trigger_execution",
            time.time() - start_time,
            False,
            error=str(e),
            repo=repo_name
        )
        logger.exception(
            "Manual trigger execution failed",
            repo=repo_name,
            event_type=event_type,
            error=str(e)
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
async def get_metrics(
    bot: SelfHealingBot = Depends(get_bot),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    """Get comprehensive bot metrics for monitoring."""
    try:
        health_data = await bot.health_check()
        uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
        
        # Get rate limiter stats
        rate_stats = await rate_limiter.get_stats()
        
        metrics = MetricsResponse(
            uptime_seconds=uptime,
            total_requests=prometheus_metrics.webhook_requests_total._value.sum(),
            successful_requests=prometheus_metrics.webhook_requests_total.labels(status="success")._value.sum(),
            failed_requests=prometheus_metrics.webhook_requests_total.labels(status="error")._value.sum(),
            active_executions=health_data.get("active_executions", 0),
            bot_health=health_data,
            rate_limit_stats=rate_stats
        )
        
        return metrics
        
    except Exception as e:
        logger.error("Error getting metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metrics"
        )

@app.get("/metrics/prometheus", response_class=PlainTextResponse, tags=["monitoring"])
async def prometheus_metrics_endpoint():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    try:
        return PlainTextResponse(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error("Error generating Prometheus metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate metrics"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with detailed logging."""
    client_ip = request.client.host if request.client else "unknown"
    
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=str(request.url.path),
        client_ip=client_ip
    )
    
    # Log security events for certain status codes
    if exc.status_code in [401, 403, 429]:
        audit_logger.log_security_event(
            "http_error", "warning",
            {
                "status_code": exc.status_code,
                "path": str(request.url.path),
                "client_ip": client_ip
            }
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with comprehensive error tracking."""
    client_ip = request.client.host if request.client else "unknown"
    
    logger.exception(
        "Unhandled exception occurred",
        path=str(request.url.path),
        method=request.method,
        client_ip=client_ip,
        error=str(exc)
    )
    
    # Log critical security event
    audit_logger.log_security_event(
        "unhandled_exception", "critical",
        {
            "path": str(request.url.path),
            "method": request.method,
            "client_ip": client_ip,
            "error": str(exc)
        }
    )
    
    # Track error in metrics
    prometheus_metrics.http_errors_total.labels(
        method=request.method,
        status_code=500
    ).inc()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )




if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    uvicorn.run(
        "self_healing_bot.web.app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower(),
        log_config=log_config,
        access_log=True
    )