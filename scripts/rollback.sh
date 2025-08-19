#!/bin/bash
# TERRAGON Production Rollback Script

set -e

echo "🔄 Starting TERRAGON production rollback..."

# Get the previous version
PREVIOUS_VERSION=$(docker images terragon/self-healing-mlops-bot --format "table {{.Tag}}" | grep -v "latest" | head -n 1)

if [ -z "$PREVIOUS_VERSION" ]; then
    echo "❌ No previous version found for rollback"
    exit 1
fi

echo "📦 Rolling back to version: $PREVIOUS_VERSION"

# Update image tag in docker-compose
sed -i "s/terragon\/self-healing-mlops-bot:latest/terragon\/self-healing-mlops-bot:$PREVIOUS_VERSION/g" docker-compose.prod.yml

# Deploy previous version
docker-compose -f docker-compose.prod.yml up -d

# Verify rollback
echo "✅ Verifying rollback..."
timeout 120 bash -c 'until curl -f http://localhost:8080/health; do sleep 5; done'

echo "🎉 Rollback completed successfully!"
