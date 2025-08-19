#!/bin/bash
# TERRAGON Production Verification Script

echo "🔍 Verifying TERRAGON production deployment..."

# Check service health
echo "🏥 Checking service health..."
curl -f http://localhost:8080/health/live || { echo "❌ Liveness check failed"; exit 1; }
curl -f http://localhost:8080/health/ready || { echo "❌ Readiness check failed"; exit 1; }
curl -f http://localhost:8080/health/startup || { echo "❌ Startup check failed"; exit 1; }

# Check database connection
echo "🗄️ Checking database connection..."
docker-compose -f docker-compose.prod.yml exec -T terragon-db pg_isready -U terragon -d terragon_prod || { echo "❌ Database check failed"; exit 1; }

# Check Redis connection
echo "🔄 Checking Redis connection..."
docker-compose -f docker-compose.prod.yml exec -T terragon-redis redis-cli ping || { echo "❌ Redis check failed"; exit 1; }

# Check monitoring
echo "📊 Checking monitoring..."
curl -f http://localhost:9090/-/healthy || { echo "❌ Prometheus check failed"; exit 1; }
curl -f http://localhost:3000/api/health || { echo "❌ Grafana check failed"; exit 1; }

# Check SSL certificate
echo "🔒 Checking SSL certificate..."
if command -v openssl >/dev/null 2>&1; then
    echo | openssl s_client -connect terragon-api.production.com:443 -servername terragon-api.production.com 2>/dev/null | openssl x509 -noout -dates
fi

echo "✅ All verification checks passed!"
echo "🎉 TERRAGON is running healthy in production!"
