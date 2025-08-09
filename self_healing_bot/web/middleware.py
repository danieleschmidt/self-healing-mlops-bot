"""Custom middleware for the FastAPI web application."""

import time
import uuid
from typing import Callable, Dict, Any
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import structlog

from ..monitoring.logging import get_logger, audit_logger, performance_logger
from ..monitoring.metrics import prometheus_metrics
from ..core.config import config


logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with detailed logging."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Get client information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request start
        start_time = time.time()
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=client_ip,
            user_agent=user_agent[:100],  # Truncate long user agents
            content_length=int(request.headers.get("content-length", 0))
        )
        
        # Process request
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log successful response
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration_seconds=duration,
                client_ip=client_ip
            )
            
            # Record metrics
            prometheus_metrics.record_http_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as exc:
            duration = time.time() - start_time
            
            # Log error
            logger.exception(
                "Request failed with exception",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                duration_seconds=duration,
                client_ip=client_ip,
                error=str(exc)
            )
            
            # Record error metrics
            prometheus_metrics.record_http_error(
                method=request.method,
                status_code=500
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
                headers={"X-Request-ID": request_id}
            )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self'"
            ),
            "Permissions-Policy": (
                "geolocation=(), microphone=(), camera=(), "
                "payment=(), usb=(), magnetometer=(), gyroscope=()"
            )
        }
        
        # Adjust security headers for development
        if config.environment == "development":
            self.security_headers["Content-Security-Policy"] = (
                "default-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "connect-src 'self' ws: wss:"
            )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add server identification (optional)
        if config.debug:
            response.headers["X-Powered-By"] = "Self-Healing-MLOps-Bot/1.0.0"
        
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and alerting."""

    def __init__(self, app):
        super().__init__(app)
        self.slow_request_threshold = 5.0  # seconds
        self.very_slow_request_threshold = 10.0  # seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance."""
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        # Log performance metrics
        performance_logger.log_execution_time(
            "http_request",
            duration,
            response.status_code < 400,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code
        )
        
        # Alert on slow requests
        if duration > self.very_slow_request_threshold:
            audit_logger.log_security_event(
                "very_slow_request", "warning",
                {
                    "method": request.method,
                    "path": str(request.url.path),
                    "duration": duration,
                    "status_code": response.status_code,
                    "client_ip": request.client.host if request.client else "unknown"
                }
            )
        elif duration > self.slow_request_threshold:
            logger.warning(
                "Slow request detected",
                method=request.method,
                path=str(request.url.path),
                duration=duration,
                status_code=response.status_code
            )
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors with proper logging and responses."""
        try:
            return await call_next(request)
            
        except ValueError as exc:
            # Handle validation errors
            logger.warning(
                "Validation error",
                path=str(request.url.path),
                method=request.method,
                error=str(exc)
            )
            
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "Validation error",
                    "message": str(exc),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except PermissionError as exc:
            # Handle permission errors
            audit_logger.log_security_event(
                "permission_denied", "warning",
                {
                    "path": str(request.url.path),
                    "method": request.method,
                    "client_ip": request.client.host if request.client else "unknown",
                    "error": str(exc)
                }
            )
            
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Permission denied",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except TimeoutError as exc:
            # Handle timeout errors
            logger.error(
                "Request timeout",
                path=str(request.url.path),
                method=request.method,
                error=str(exc)
            )
            
            return JSONResponse(
                status_code=504,
                content={
                    "detail": "Request timeout",
                    "message": "The request took too long to process",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as exc:
            # Handle all other exceptions
            logger.exception(
                "Unhandled exception",
                path=str(request.url.path),
                method=request.method,
                error=str(exc)
            )
            
            # Record error in metrics
            prometheus_metrics.record_error(
                error_type=type(exc).__name__,
                component="web_middleware"
            )
            
            # Don't expose internal error details in production
            if config.environment == "production":
                error_detail = "Internal server error"
                error_message = "An unexpected error occurred"
            else:
                error_detail = f"{type(exc).__name__}: {str(exc)}"
                error_message = str(exc)
            
            return JSONResponse(
                status_code=500,
                content={
                    "detail": error_detail,
                    "message": error_message,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )


class CORSHeadersMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with enhanced security."""

    def __init__(self, app):
        super().__init__(app)
        
        # Configure allowed origins based on environment
        if config.environment == "development":
            self.allowed_origins = ["*"]
        else:
            self.allowed_origins = [
                "https://github.com",
                "https://*.github.com",
                "https://*.githubapp.com"
            ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle CORS with security considerations."""
        origin = request.headers.get("origin", "")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            
            # Check if origin is allowed
            if self._is_origin_allowed(origin):
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, HEAD, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = (
                    "Accept, Accept-Language, Content-Language, Content-Type, "
                    "Authorization, X-GitHub-Event, X-GitHub-Delivery, X-Hub-Signature-256"
                )
                response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours
                
                if config.environment != "production":
                    response.headers["Access-Control-Allow-Credentials"] = "true"
            
            return response
        
        # Process the request
        response = await call_next(request)
        
        # Add CORS headers to actual responses
        if self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin if origin else "*"
            
            if config.environment != "production":
                response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if the origin is allowed."""
        if not origin:
            return config.environment == "development"
        
        if "*" in self.allowed_origins:
            return True
        
        for allowed in self.allowed_origins:
            if allowed.startswith("*."):
                # Wildcard subdomain matching
                domain = allowed[2:]
                if origin.endswith(f".{domain}") or origin == f"https://{domain}":
                    return True
            elif origin == allowed:
                return True
        
        return False


def add_middleware(app):
    """Add all middleware to the FastAPI app in the correct order."""
    
    # Order matters! Middleware is applied in reverse order of addition
    # (last added is executed first)
    
    # 1. CORS (outermost)
    app.add_middleware(CORSHeadersMiddleware)
    
    # 2. Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # 3. Performance monitoring
    app.add_middleware(PerformanceMonitoringMiddleware)
    
    # 4. Error handling
    app.add_middleware(ErrorHandlingMiddleware)
    
    # 5. Request logging (innermost)
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("All middleware components added to FastAPI app")