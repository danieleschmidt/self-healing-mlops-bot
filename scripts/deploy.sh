#!/bin/bash

# Self-Healing MLOps Bot Deployment Script
# Usage: ./scripts/deploy.sh [environment]

set -euo pipefail

ENVIRONMENT=${1:-production}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Deploying Self-Healing MLOps Bot to $ENVIRONMENT..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_warning ".env file not found. Creating from .env.example..."
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        log_warning "Please update .env file with your configuration before proceeding"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Validate environment configuration
validate_config() {
    log_info "Validating configuration..."
    
    # Check required environment variables
    required_vars=(
        "GITHUB_APP_ID"
        "GITHUB_PRIVATE_KEY_PATH"
        "GITHUB_WEBHOOK_SECRET"
        "SECRET_KEY"
        "ENCRYPTION_KEY"
        "POSTGRES_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" "$PROJECT_ROOT/.env" || grep -q "^${var}=$" "$PROJECT_ROOT/.env" || grep -q "^${var}=your-" "$PROJECT_ROOT/.env"; then
            log_error "Environment variable $var is not properly configured in .env file"
            exit 1
        fi
    done
    
    # Check if GitHub private key exists
    if [ ! -f "$PROJECT_ROOT/keys/private-key.pem" ]; then
        log_error "GitHub private key not found at keys/private-key.pem"
        log_info "Please place your GitHub App private key at keys/private-key.pem"
        exit 1
    fi
    
    log_success "Configuration validation passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build with build args for caching
    docker-compose -f docker-compose.prod.yml build \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --parallel
    
    log_success "Docker images built successfully"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    cd "$PROJECT_ROOT"
    
    # Start database first
    docker-compose -f docker-compose.prod.yml up -d db redis
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    timeout=60
    while ! docker-compose -f docker-compose.prod.yml exec -T db pg_isready -U postgres &> /dev/null; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            log_error "Database failed to start within 60 seconds"
            exit 1
        fi
    done
    
    # Run migrations
    docker-compose -f docker-compose.prod.yml run --rm app python -c "
from self_healing_bot.database.migrations import run_migrations
run_migrations()
print('Database migrations completed successfully')
"
    
    log_success "Database migrations completed"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Pull latest images for services that aren't built locally
    docker-compose -f docker-compose.prod.yml pull nginx postgres redis prometheus grafana
    
    # Stop existing services
    docker-compose -f docker-compose.prod.yml down
    
    # Start all services
    docker-compose -f docker-compose.prod.yml up -d
    
    log_success "Services deployed successfully"
}

# Health checks
perform_health_checks() {
    log_info "Performing health checks..."
    
    cd "$PROJECT_ROOT"
    
    # Wait for services to be ready
    services=("app" "db" "redis")
    for service in "${services[@]}"; do
        log_info "Checking $service health..."
        timeout=120
        while ! docker-compose -f docker-compose.prod.yml exec -T "$service" sh -c "exit 0" &> /dev/null; do
            sleep 5
            timeout=$((timeout - 5))
            if [ $timeout -le 0 ]; then
                log_error "$service failed to start within 2 minutes"
                docker-compose -f docker-compose.prod.yml logs "$service"
                exit 1
            fi
        done
        log_success "$service is healthy"
    done
    
    # Test application endpoint
    log_info "Testing application endpoint..."
    timeout=60
    while ! curl -f http://localhost:8080/health &> /dev/null; do
        sleep 5
        timeout=$((timeout - 5))
        if [ $timeout -le 0 ]; then
            log_error "Application health check failed"
            docker-compose -f docker-compose.prod.yml logs app
            exit 1
        fi
    done
    
    log_success "Application is responding to health checks"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    cd "$PROJECT_ROOT"
    
    # Create Grafana dashboards directory if it doesn't exist
    mkdir -p monitoring/grafana/dashboards monitoring/grafana/datasources
    
    # Wait for Grafana to be ready
    log_info "Waiting for Grafana to be ready..."
    timeout=60
    while ! curl -f http://localhost:3000/api/health &> /dev/null; do
        sleep 5
        timeout=$((timeout - 5))
        if [ $timeout -le 0 ]; then
            log_warning "Grafana health check timeout, but continuing..."
            break
        fi
    done
    
    log_success "Monitoring setup completed"
}

# Cleanup old images and containers
cleanup() {
    log_info "Cleaning up old images and containers..."
    
    # Remove dangling images
    docker image prune -f
    
    # Remove unused volumes (be careful with this in production)
    if [ "$ENVIRONMENT" != "production" ]; then
        docker volume prune -f
    fi
    
    log_success "Cleanup completed"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo ""
    
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose.prod.yml ps
    
    echo ""
    log_info "Service URLs:"
    echo "  🤖 Self-Healing Bot: https://localhost"
    echo "  📊 Grafana Dashboard: http://localhost:3000"
    echo "  📈 Prometheus: http://localhost:9091"
    echo "  🏥 Health Check: http://localhost:8080/health"
    echo ""
    
    log_success "🎉 Deployment completed successfully!"
    log_info "Monitor the logs with: docker-compose -f docker-compose.prod.yml logs -f"
}

# Main deployment flow
main() {
    log_info "Starting deployment to $ENVIRONMENT environment..."
    
    check_prerequisites
    validate_config
    build_images
    run_migrations
    deploy_services
    perform_health_checks
    setup_monitoring
    cleanup
    show_status
}

# Error handling
trap 'log_error "Deployment failed at line $LINENO"' ERR

# Run main function
main "$@"