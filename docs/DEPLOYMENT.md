# Production Deployment Guide

This guide covers deploying the Self-Healing MLOps Bot to a production Kubernetes environment.

## 🚀 Quick Start

### Prerequisites

- Kubernetes cluster (v1.21+)
- kubectl configured to access your cluster
- Helm 3.x installed
- cert-manager for TLS certificates
- NGINX Ingress Controller
- Prometheus and Grafana for monitoring

### 1. Create GitHub App

1. Go to GitHub Settings > Developer settings > GitHub Apps
2. Create a new GitHub App with these permissions:
   - Repository permissions:
     - Actions: Read
     - Contents: Write
     - Issues: Write
     - Metadata: Read
     - Pull requests: Write
   - Subscribe to events:
     - Push
     - Pull request
     - Issues
     - Workflow run

3. Generate and download the private key
4. Install the app on your repositories

### 2. Configure Secrets

```bash
# Clone the repository
git clone <repository-url>
cd self-healing-mlops-bot

# Copy and edit the secrets template
cp kubernetes/production/secrets.yaml kubernetes/production/secrets-actual.yaml
```

Edit `secrets-actual.yaml` and replace all placeholder values:

```yaml
stringData:
  GITHUB_APP_ID: "123456"  # Your GitHub App ID
  GITHUB_WEBHOOK_SECRET: "your-webhook-secret"
  GITHUB_TOKEN: "ghp_your-token-here"
  SECRET_KEY: "$(openssl rand -hex 32)"
  ENCRYPTION_KEY: "$(openssl rand -hex 32)"
  DB_PASSWORD: "$(openssl rand -base64 32)"
  # ... other secrets
```

Encode your GitHub private key:
```bash
base64 -w 0 < your-github-app.pem
```

### 3. Deploy to Kubernetes

```bash
# Create namespace and apply configurations
kubectl apply -f kubernetes/production/namespace.yaml
kubectl apply -f kubernetes/production/secrets-actual.yaml
kubectl apply -f kubernetes/production/configmap.yaml
kubectl apply -f kubernetes/production/rbac.yaml

# Deploy stateful services first
kubectl apply -f kubernetes/production/statefulset.yaml
kubectl apply -f kubernetes/production/services.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n mlops-bot-prod --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n mlops-bot-prod --timeout=300s

# Deploy the application
kubectl apply -f kubernetes/production/deployment.yaml

# Configure auto-scaling and ingress
kubectl apply -f kubernetes/production/hpa.yaml
kubectl apply -f kubernetes/production/ingress.yaml

# Apply monitoring configuration
kubectl apply -f kubernetes/production/monitoring.yaml
```

### 4. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n mlops-bot-prod

# Check ingress is configured
kubectl get ingress -n mlops-bot-prod

# Test the application
curl -k https://mlops-bot.yourdomain.com/health
```

## 🔧 Configuration

### Environment Variables

The bot is configured through environment variables defined in `configmap.yaml`:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8080` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENVIRONMENT` | Deployment environment | `production` |
| `DATABASE_URL` | PostgreSQL connection string | - |
| `REDIS_URL` | Redis connection string | - |
| `ENABLE_AUTO_SCALING` | Enable auto-scaling | `true` |
| `MAX_WORKERS` | Maximum worker threads | `50` |
| `RATE_LIMIT_PER_MINUTE` | API rate limit | `1000` |

### Scaling Configuration

Horizontal Pod Autoscaler (HPA) configuration:

```yaml
# Main application pods
minReplicas: 3
maxReplicas: 10
targetCPUUtilization: 70%
targetMemoryUtilization: 80%

# Worker pods
minReplicas: 2
maxReplicas: 8
targetCPUUtilization: 75%
targetMemoryUtilization: 85%
```

### Resource Requirements

| Component | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-----------|-------------|----------------|-----------|--------------|
| MLOps Bot | 200m | 512Mi | 1 | 2Gi |
| Celery Worker | 100m | 256Mi | 500m | 1Gi |
| PostgreSQL | 200m | 512Mi | 1 | 2Gi |
| Redis | 100m | 256Mi | 500m | 1Gi |

## 🔍 Monitoring

### Metrics Endpoints

- Application metrics: `http://mlops-bot-service:9090/metrics`
- Health check: `http://mlops-bot-service:8080/health`
- Redis metrics: `http://redis-service:9121/metrics`

### Key Metrics

#### Application Metrics
- `bot_events_processed_total` - Total webhook events processed
- `bot_event_processing_duration_seconds` - Event processing latency
- `bot_issues_detected_total` - Total issues detected by type
- `bot_repairs_attempted_total` - Total repair attempts by status
- `bot_errors_total` - Total errors by component
- `bot_github_api_rate_limit_remaining` - GitHub API rate limit

#### System Metrics
- `container_cpu_usage_seconds_total` - CPU usage
- `container_memory_working_set_bytes` - Memory usage
- `kube_pod_container_status_restarts_total` - Pod restarts

### Alerts

Critical alerts configured in Prometheus:

1. **High Error Rate** (>10% for 2 minutes)
2. **High Latency** (95th percentile >30s for 5 minutes)  
3. **Pod Crashing** (Restart rate >0 for 1 minute)
4. **High Memory Usage** (>90% for 5 minutes)
5. **Database/Redis Down** (Health check failing for 1 minute)
6. **GitHub Rate Limit** (<100 requests remaining)

### Grafana Dashboard

A pre-configured Grafana dashboard is included showing:
- Event processing rate and error rate
- Response time percentiles
- Resource utilization
- Auto-scaling metrics
- Health status of all components

## 🔒 Security

### Network Security

- All traffic encrypted with TLS
- Internal services use ClusterIP (no external access)
- Rate limiting on ingress (100 RPS with burst)
- Webhook signature verification
- Security headers applied

### Pod Security

- Non-root containers (user ID 1000)
- Read-only root filesystem where possible
- No privilege escalation
- Capabilities dropped
- Security contexts enforced

### Secret Management

**⚠️ Important**: Use proper secret management for production:

1. **Sealed Secrets**: Encrypt secrets at rest
2. **External Secrets Operator**: Integrate with vault systems
3. **Azure Key Vault / AWS Secrets Manager**: Cloud-native solutions
4. **HashiCorp Vault**: Enterprise secret management

### RBAC

Minimal required permissions:
- Read pods, configmaps, services for health checks
- Create events for audit logging  
- Read secrets for application configuration
- No cluster-admin privileges

## 📊 Performance Tuning

### Database Optimization

PostgreSQL configuration:
```yaml
shared_buffers: 256MB
effective_cache_size: 1GB
work_mem: 4MB
maintenance_work_mem: 64MB
max_connections: 100
```

### Redis Optimization

Redis configuration:
```yaml
maxmemory: 800mb
maxmemory-policy: allkeys-lru
save: 900 1 300 10 60 10000
appendonly: yes
appendfsync: everysec
```

### Application Tuning

Key performance settings:
- Connection pooling enabled (10 connections per pod)
- Async I/O for all external calls
- Request/response caching (Redis)
- Batch processing for similar operations
- Circuit breakers for external service calls

## 🚨 Troubleshooting

### Common Issues

#### Pod Startup Issues
```bash
# Check pod events
kubectl describe pod <pod-name> -n mlops-bot-prod

# Check logs
kubectl logs <pod-name> -n mlops-bot-prod -f

# Check secrets are mounted correctly
kubectl exec <pod-name> -n mlops-bot-prod -- ls -la /etc/github/
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it postgres-0 -n mlops-bot-prod -- psql -U bot_user -d mlops_bot_prod -c \"SELECT 1;\"

# Check database logs
kubectl logs postgres-0 -n mlops-bot-prod
```

#### GitHub Integration Issues
```bash
# Verify webhook secret and private key
kubectl get secret mlops-bot-secrets -n mlops-bot-prod -o yaml

# Test GitHub API connectivity
kubectl exec <pod-name> -n mlops-bot-prod -- curl -H \"Authorization: Bearer <token>\" https://api.github.com/user
```

### Performance Issues

#### High Memory Usage
```bash
# Check memory metrics
kubectl top pods -n mlops-bot-prod

# Adjust memory limits
kubectl patch deployment mlops-bot -n mlops-bot-prod -p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"mlops-bot\",\"resources\":{\"limits\":{\"memory\":\"4Gi\"}}}]}}}}'
```

#### High CPU Usage
```bash
# Check CPU metrics
kubectl top pods -n mlops-bot-prod

# Scale up replicas temporarily
kubectl scale deployment mlops-bot -n mlops-bot-prod --replicas=5
```

### Logging

Application logs are structured JSON format:
```bash
# Follow application logs
kubectl logs -f deployment/mlops-bot -n mlops-bot-prod

# Filter by log level
kubectl logs deployment/mlops-bot -n mlops-bot-prod | jq 'select(.level==\"ERROR\")'

# Filter by component  
kubectl logs deployment/mlops-bot -n mlops-bot-prod | jq 'select(.name==\"security\")'
```

## 📈 Maintenance

### Backup Procedures

#### Database Backup
```bash
# Create backup
kubectl exec postgres-0 -n mlops-bot-prod -- pg_dump -U bot_user mlops_bot_prod > backup.sql

# Restore backup
kubectl exec -i postgres-0 -n mlops-bot-prod -- psql -U bot_user mlops_bot_prod < backup.sql
```

#### Redis Backup
```bash
# Redis automatically saves to persistent volume
# Manual backup trigger:
kubectl exec redis-0 -n mlops-bot-prod -- redis-cli BGSAVE
```

### Update Procedures

1. **Rolling Update**:
   ```bash
   kubectl set image deployment/mlops-bot mlops-bot=mlops-bot:v1.1.0 -n mlops-bot-prod
   ```

2. **Database Migrations**:
   ```bash
   # Run migrations
   kubectl create job --from=deployment/mlops-bot migrate-$(date +%s) -n mlops-bot-prod -- alembic upgrade head
   ```

3. **Configuration Updates**:
   ```bash
   # Update configmap
   kubectl apply -f kubernetes/production/configmap.yaml
   
   # Restart deployment to pick up changes
   kubectl rollout restart deployment/mlops-bot -n mlops-bot-prod
   ```

### Health Monitoring

Regular health checks:
```bash
# Overall cluster health
kubectl get pods -n mlops-bot-prod

# Application health endpoint
curl -k https://mlops-bot.yourdomain.com/health

# Check HPA status
kubectl get hpa -n mlops-bot-prod

# Check resource usage
kubectl top pods -n mlops-bot-prod
```

## 🔄 Disaster Recovery

### Backup Strategy
- Database: Daily automated backups to object storage
- Redis: Persistent volume snapshots
- Configuration: All K8s manifests in version control
- Secrets: Encrypted backup of secret values

### Recovery Procedures

1. **Complete Cluster Rebuild**:
   - Deploy from K8s manifests
   - Restore database from backup
   - Verify all services healthy

2. **Data Recovery**:
   - Restore database to point-in-time
   - Replay missed events from GitHub webhook logs
   - Validate data integrity

3. **Service Recovery**:
   - Scale down affected pods
   - Clear Redis cache if needed
   - Scale back up with health checks

## 📞 Support

For production support:
- Monitor Grafana dashboards for system health
- Check Prometheus alerts for critical issues
- Review application logs for error patterns
- Use kubectl commands for troubleshooting

Emergency escalation procedures should include:
1. Immediate incident response team notification
2. Runbook for common failure scenarios  
3. Contact information for on-call engineers
4. Disaster recovery checklist and procedures