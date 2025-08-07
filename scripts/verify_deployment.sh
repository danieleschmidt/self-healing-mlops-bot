#!/bin/bash

# Self-Healing MLOps Bot - Deployment Verification Script
# This script verifies that all components are properly deployed and functional

set -e

echo "🔍 Self-Healing MLOps Bot - Deployment Verification"
echo "=================================================="

# Configuration
COMPOSE_FILE="${1:-docker-compose.yml}"
BASE_URL="${2:-http://localhost:8080}"
TIMEOUT=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are available
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    log_success "Docker and Docker Compose are available"
}

# Check service status
check_service_status() {
    log_info "Checking service status..."
    
    if ! docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        log_error "Some services are not running. Run: docker-compose -f $COMPOSE_FILE ps"
        exit 1
    fi
    
    log_success "All services are running"
}

# Test health endpoints
test_health_endpoints() {
    log_info "Testing health endpoints..."
    
    # Main health endpoint
    if curl -f -s --max-time $TIMEOUT "$BASE_URL/health" > /dev/null; then
        log_success "Main health endpoint is responding"
    else
        log_error "Main health endpoint is not responding at $BASE_URL/health"
        return 1
    fi
    
    # Detailed health check
    local health_response
    health_response=$(curl -s --max-time $TIMEOUT "$BASE_URL/health" || echo "{}")
    
    local status
    status=$(echo "$health_response" | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")
    
    if [ "$status" = "healthy" ]; then
        log_success "System status: $status"
    else
        log_warning "System status: $status"
        echo "$health_response" | jq . 2>/dev/null || echo "$health_response"
    fi
}

# Test metrics endpoint
test_metrics_endpoint() {
    log_info "Testing metrics endpoint..."
    
    if curl -f -s --max-time $TIMEOUT "$BASE_URL/metrics" > /dev/null; then
        log_success "Metrics endpoint is responding"
    else
        log_warning "Metrics endpoint is not responding (this may be expected in development)"
    fi
}

# Test webhook endpoint (should reject unauthorized requests)
test_webhook_endpoint() {
    log_info "Testing webhook endpoint security..."
    
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT -X POST "$BASE_URL/webhook" || echo "000")
    
    if [ "$response_code" = "401" ] || [ "$response_code" = "403" ]; then
        log_success "Webhook endpoint properly rejects unauthorized requests (HTTP $response_code)"
    elif [ "$response_code" = "422" ]; then
        log_success "Webhook endpoint is responding (HTTP $response_code - validation error expected)"
    else
        log_warning "Webhook endpoint response: HTTP $response_code"
    fi
}

# Check database connectivity
check_database() {
    log_info "Checking database connectivity..."
    
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U postgres -d selfhealingbot &> /dev/null; then
        log_success "Database is accessible"
    else
        log_error "Database connectivity check failed"
        return 1
    fi
}

# Check Redis connectivity
check_redis() {
    log_info "Checking Redis connectivity..."
    
    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
        log_success "Redis is accessible"
    else
        log_error "Redis connectivity check failed"
        return 1
    fi
}

# Check log output
check_logs() {
    log_info "Checking recent logs for errors..."
    
    local error_count
    error_count=$(docker-compose -f "$COMPOSE_FILE" logs --tail=100 self-healing-bot 2>/dev/null | grep -i "error\|exception\|failed" | wc -l || echo "0")
    
    if [ "$error_count" -eq 0 ]; then
        log_success "No recent errors found in logs"
    else
        log_warning "Found $error_count recent error messages in logs"
        log_info "Use 'docker-compose -f $COMPOSE_FILE logs self-healing-bot' to view details"
    fi
}

# Check resource usage
check_resources() {
    log_info "Checking resource usage..."
    
    # CPU and memory usage
    local stats
    stats=$(docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep self-healing-bot || echo "")
    
    if [ -n "$stats" ]; then
        log_success "Resource usage:"
        echo "$stats"
    else
        log_warning "Could not retrieve resource statistics"
    fi
}

# Performance test
run_performance_test() {
    log_info "Running basic performance test..."
    
    local start_time
    local end_time
    local duration
    
    start_time=$(date +%s.%3N)
    
    for i in {1..5}; do
        if ! curl -f -s --max-time 5 "$BASE_URL/health" > /dev/null; then
            log_error "Performance test failed on request $i"
            return 1
        fi
    done
    
    end_time=$(date +%s.%3N)
    duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "unknown")
    
    log_success "Performance test completed: 5 requests in ${duration}s"
}

# Security checks
run_security_checks() {
    log_info "Running basic security checks..."
    
    # Check for exposed secrets in environment variables
    local exposed_secrets
    exposed_secrets=$(docker-compose -f "$COMPOSE_FILE" config 2>/dev/null | grep -i "password\|secret\|key" | grep -v "REDACTED\|****" | wc -l || echo "0")
    
    if [ "$exposed_secrets" -gt 0 ]; then
        log_warning "Potential secrets exposure detected in compose configuration"
    else
        log_success "No obvious secrets exposure in configuration"
    fi
    
    # Check file permissions
    if [ -f "keys/private-key.pem" ]; then
        local key_perms
        key_perms=$(stat -c "%a" keys/private-key.pem 2>/dev/null || echo "unknown")
        
        if [ "$key_perms" = "600" ]; then
            log_success "Private key file permissions are secure"
        else
            log_warning "Private key file permissions: $key_perms (should be 600)"
        fi
    fi
}

# Main execution
main() {
    local exit_code=0
    
    echo "Starting deployment verification with:"
    echo "  Compose file: $COMPOSE_FILE"
    echo "  Base URL: $BASE_URL"
    echo ""
    
    # Run all checks
    check_prerequisites || exit_code=1
    check_service_status || exit_code=1
    
    # Give services time to fully start
    log_info "Waiting for services to stabilize..."
    sleep 10
    
    test_health_endpoints || exit_code=1
    test_metrics_endpoint || exit_code=1
    test_webhook_endpoint || exit_code=1
    check_database || exit_code=1
    check_redis || exit_code=1
    check_logs || exit_code=1
    check_resources || exit_code=1
    run_performance_test || exit_code=1
    run_security_checks || exit_code=1
    
    echo ""
    echo "=================================================="
    
    if [ $exit_code -eq 0 ]; then
        log_success "✅ All verification checks passed!"
        log_info "Your Self-Healing MLOps Bot deployment is ready for use."
        echo ""
        log_info "Quick commands:"
        echo "  Health check: curl $BASE_URL/health"
        echo "  View logs: docker-compose -f $COMPOSE_FILE logs -f self-healing-bot"
        echo "  Scale workers: docker-compose -f $COMPOSE_FILE up -d --scale celery-worker=3"
    else
        log_error "❌ Some verification checks failed!"
        log_info "Please review the errors above and check your deployment configuration."
    fi
    
    exit $exit_code
}

# Handle script arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [compose-file] [base-url]"
    echo ""
    echo "Arguments:"
    echo "  compose-file    Docker compose file to use (default: docker-compose.yml)"
    echo "  base-url        Base URL for API testing (default: http://localhost:8080)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 docker-compose.prod.yml https://your-domain.com"
    echo "  $0 docker-compose.yml http://localhost:8080"
    exit 0
fi

main "$@"