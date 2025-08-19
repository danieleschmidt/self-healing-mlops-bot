#!/bin/bash
# TERRAGON Production Deployment Script

set -e

echo "🚀 Starting TERRAGON production deployment..."

# Check prerequisites
echo "🔍 Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Load environment variables
if [ -f .env.production ]; then
    source .env.production
    echo "✅ Environment variables loaded"
else
    echo "❌ .env.production file not found"
    exit 1
fi

# Build and deploy
echo "🏗️ Building production images..."
docker-compose -f docker-compose.prod.yml build --no-cache

echo "🚀 Starting production services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
timeout 300 bash -c 'until docker-compose -f docker-compose.prod.yml ps | grep -q "healthy"; do sleep 10; done'

# Run database migrations
echo "🗄️ Running database migrations..."
docker-compose -f docker-compose.prod.yml exec -T terragon-api python -m alembic upgrade head

# Verify deployment
echo "✅ Verifying deployment..."
curl -f http://localhost:8080/health || { echo "❌ Health check failed"; exit 1; }

echo "🎉 TERRAGON production deployment completed successfully!"
echo "🌐 Application available at: https://terragon-api.production.com"
echo "📊 Monitoring available at: http://localhost:3000 (Grafana)"
echo "📈 Metrics available at: http://localhost:9090 (Prometheus)"
