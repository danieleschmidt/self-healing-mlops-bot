# FastAPI Web Server Implementation

This document summarizes the robust FastAPI web server implementation for the self-healing MLOps bot.

## Components Implemented

### 1. Enhanced Core Web Application (`self_healing_bot/web/app.py`)

**Features:**
- Production-ready FastAPI application with lifespan management
- Comprehensive health checks with uptime tracking
- Enhanced webhook endpoint with security and monitoring
- Manual trigger endpoint with background processing
- Metrics endpoints (JSON and Prometheus format)
- Structured error handling with detailed logging
- Security middleware integration
- Request ID tracking for debugging

**Key Endpoints:**
- `GET /` - API information with uptime
- `GET /ping` - Simple health check for load balancers
- `GET /health` - Comprehensive health status
- `POST /webhook` - GitHub webhook handler with security
- `GET /executions` - List active executions
- `GET /executions/{id}` - Get specific execution status
- `POST /manual-trigger` - Manual event triggering
- `GET /metrics` - JSON metrics for monitoring
- `GET /metrics/prometheus` - Prometheus format metrics

### 2. GitHub Webhook Handler (`self_healing_bot/web/webhooks.py`)

**Features:**
- Secure webhook signature verification (HMAC-SHA256)
- Event type validation and filtering
- Background processing with async queue
- Comprehensive audit logging
- Rate limiting and security validation
- Support for all GitHub webhook events
- Graceful error handling and recovery
- Performance monitoring and metrics

**Supported Events:**
- Push events (main/master branches)
- Pull request events
- Workflow run failures
- Check run failures  
- Installation events
- Repository events
- Issue events

### 3. Production Server Entry Point (`self_healing_bot/server.py`)

**Features:**
- Graceful shutdown with signal handling
- Configuration validation
- Multiple deployment modes (development/production)
- Gunicorn integration for production
- Health check monitoring
- Resource cleanup on shutdown
- SSL/TLS support configuration
- Performance optimizations

**Deployment Modes:**
- Development: Single worker with reload
- Production: Multi-worker with Gunicorn
- Container: Optimized for Kubernetes/Docker

### 4. Advanced Middleware (`self_healing_bot/web/middleware.py`)

**Middleware Components:**

#### RequestLoggingMiddleware
- Unique request ID generation
- Comprehensive request/response logging
- Performance metrics collection
- Client IP and User-Agent tracking

#### SecurityHeadersMiddleware
- Security headers (HSTS, CSP, X-Frame-Options, etc.)
- Environment-specific configurations
- XSS and clickjacking protection

#### PerformanceMonitoringMiddleware
- Request duration tracking
- Slow request alerting
- Response time headers
- Performance metrics collection

#### ErrorHandlingMiddleware
- Centralized exception handling
- Proper HTTP status codes
- Error logging and metrics
- Development vs production error details

#### CORSHeadersMiddleware
- Enhanced CORS with security
- Origin validation
- Preflight request handling
- Wildcard subdomain support

### 5. Enhanced Security & Rate Limiting (`self_healing_bot/security/validation.py`)

**RateLimiter Features:**
- Per-client rate limiting with burst protection
- Endpoint-specific limits
- Automatic client blocking for violations
- Comprehensive statistics and monitoring
- Async-safe implementation

**SecurityValidator Features:**
- Request size validation
- Header validation (X-Forwarded-For, User-Agent, etc.)
- Path traversal detection
- Suspicious pattern detection
- IP blocking capabilities

**InputValidator Features:**
- GitHub repository name validation
- Branch name validation
- File path safety checks
- Commit message sanitization
- URL validation with HTTPS enforcement

### 6. Enhanced Metrics (`self_healing_bot/monitoring/metrics.py`)

**Web-Specific Metrics:**
- `bot_webhook_requests_total` - Webhook request counter
- `bot_webhook_events_processed_total` - Event processing counter
- `bot_http_requests_total` - HTTP request counter
- `bot_http_errors_total` - HTTP error counter
- `bot_http_request_duration_seconds` - Request duration histogram
- `bot_rate_limit_hits_total` - Rate limit violation counter
- `bot_security_violations_total` - Security violation counter

## Security Features

### 1. Authentication & Authorization
- GitHub webhook signature verification
- Bearer token support for protected endpoints
- Request source validation

### 2. Rate Limiting
- Per-IP rate limiting with configurable limits
- Burst protection to prevent spam
- Automatic blocking of abusive clients
- Whitelist/blacklist support

### 3. Input Validation
- Comprehensive input sanitization
- Path traversal prevention
- XSS and injection attack protection
- Request size limits

### 4. Security Headers
- HTTPS enforcement
- Content Security Policy (CSP)
- X-Frame-Options for clickjacking protection
- HSTS for transport security

### 5. Audit Logging
- All security events logged
- Request tracking with unique IDs
- Failed authentication attempts logged
- Rate limit violations tracked

## Performance Optimizations

### 1. Async Processing
- Background webhook processing
- Non-blocking request handling
- Async database operations

### 2. Caching
- Rate limiter with in-memory cache
- Request response caching headers
- Static asset caching

### 3. Connection Management
- Keep-alive connections
- Connection pooling
- Timeout configurations

### 4. Resource Limits
- Request size limits
- Concurrent request limits
- Memory usage monitoring

## Monitoring & Observability

### 1. Structured Logging
- JSON logging for production
- Contextual logging with request IDs
- Performance logging
- Security event logging

### 2. Metrics Collection
- Prometheus metrics export
- Request/response metrics
- Error rate monitoring
- Performance histograms

### 3. Health Checks
- Deep health checks for all components
- Dependency health monitoring
- Uptime tracking
- Component status reporting

### 4. Alerting
- Slow request detection
- Error rate alerting
- Security violation alerts
- Service availability monitoring

## Configuration

The server is configured via environment variables:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8080
ENVIRONMENT=production
DEBUG=false

# GitHub Configuration  
GITHUB_APP_ID=your_app_id
GITHUB_PRIVATE_KEY_PATH=/path/to/private-key.pem
GITHUB_WEBHOOK_SECRET=your_webhook_secret

# Security Configuration
SECRET_KEY=your_secret_key
ENCRYPTION_KEY=your_encryption_key_32_bytes
RATE_LIMIT_PER_MINUTE=60

# Logging Configuration
LOG_LEVEL=INFO

# SSL Configuration (Production)
SSL_KEYFILE=/path/to/ssl.key
SSL_CERTFILE=/path/to/ssl.crt
SSL_CA_CERTS=/path/to/ca-bundle.crt
```

## Usage

### Development Mode
```bash
# Using CLI
self-healing-bot server --reload

# Using Python module
python -m self_healing_bot.server

# Using uvicorn directly
uvicorn self_healing_bot.web.app:app --reload
```

### Production Mode
```bash
# Using CLI with production flag
self-healing-bot server --production

# Using Gunicorn directly
gunicorn self_healing_bot.server:create_app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker Deployment
```bash
# Build image
docker build -t self-healing-bot .

# Run container
docker run -p 8080:8080 -e ENVIRONMENT=production self-healing-bot
```

## Testing

The implementation includes comprehensive error handling and can be tested with:

```bash
# Health check
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics

# Webhook (with proper signature)
curl -X POST http://localhost:8080/webhook \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: push" \
  -H "X-Hub-Signature-256: sha256=..." \
  -d '{"repository":{"full_name":"test/repo"}}'
```

## Integration with Existing Architecture

The web server integrates seamlessly with the existing bot architecture:

- **SelfHealingBot**: Core orchestration remains unchanged
- **Detectors**: Webhook events trigger detection pipelines  
- **Playbooks**: Repair actions executed via webhook processing
- **GitHub Integration**: Enhanced with webhook processing
- **Monitoring**: Extended with web-specific metrics
- **Security**: Enhanced input validation and rate limiting

This implementation provides a production-ready, secure, and highly observable web interface for the self-healing MLOps bot while maintaining compatibility with the existing codebase.