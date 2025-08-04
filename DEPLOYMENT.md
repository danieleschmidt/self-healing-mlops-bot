# Self-Healing MLOps Bot - Production Deployment Guide

This guide covers deploying the Self-Healing MLOps Bot to a production environment using Docker Compose with full monitoring, security, and scalability features.

## 🏗️ Architecture Overview

The production deployment includes:

- **Application Layer**: FastAPI web server with Celery workers
- **Database Layer**: PostgreSQL with Redis for caching/queuing  
- **Reverse Proxy**: Nginx with SSL termination and rate limiting
- **Monitoring Stack**: Prometheus + Grafana with custom dashboards
- **Security**: Input validation, secrets management, audit logging

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ or CentOS 8+ (recommended)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: Minimum 50GB SSD
- **CPU**: 4+ cores recommended
- **Network**: Static IP with ports 80, 443, 22 open

### Software Dependencies
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install additional tools
sudo apt update
sudo apt install -y curl jq git nginx-utils
```

## 🔐 GitHub App Setup

### Create GitHub App
1. Go to GitHub Settings → Developer settings → GitHub Apps
2. Click "New GitHub App"
3. Configure the app:
   - **App name**: `YourOrg Self-Healing MLOps Bot`
   - **Homepage URL**: `https://your-domain.com`
   - **Webhook URL**: `https://your-domain.com/webhook`
   - **Webhook secret**: Generate a secure random string

### App Permissions
Grant the following permissions:
- **Repository permissions**:
  - Actions: Read & Write
  - Contents: Read & Write
  - Issues: Read & Write
  - Pull requests: Read & Write
  - Metadata: Read
- **Organization permissions**:
  - Actions: Read
- **Events**:
  - Push
  - Pull request
  - Workflow run
  - Check run

### Download Private Key
1. After creating the app, scroll down to "Private keys"
2. Click "Generate a private key"
3. Save the downloaded `.pem` file securely

## 🚀 Deployment Steps

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/your-org/self-healing-mlops-bot.git
cd self-healing-mlops-bot

# Create environment configuration
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` file with your values:

```bash
# GitHub App Configuration
GITHUB_APP_ID=123456
GITHUB_PRIVATE_KEY_PATH=/app/keys/private-key.pem
GITHUB_WEBHOOK_SECRET=your-webhook-secret-here

# Database Configuration
POSTGRES_PASSWORD=secure-postgres-password
DATABASE_URL=postgresql://postgres:secure-postgres-password@db:5432/self_healing_bot

# Security
SECRET_KEY=your-super-secret-jwt-key-here
ENCRYPTION_KEY=your-32-byte-encryption-key-here

# Monitoring
GRAFANA_PASSWORD=secure-grafana-password

# Optional: Slack notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
```

### 3. SSL Certificate Setup

```bash
# Create SSL directory
mkdir -p ssl

# Option A: Let's Encrypt (recommended for production)
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/key.pem

# Option B: Self-signed certificate (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/key.pem -out ssl/cert.pem \
    -subj "/C=US/ST=State/L=City/O=Org/CN=your-domain.com"
```

### 4. Prepare GitHub Private Key

```bash
# Create keys directory
mkdir -p keys

# Copy your GitHub App private key
cp /path/to/your/downloaded-key.pem keys/private-key.pem
chmod 600 keys/private-key.pem
```

### 5. Deploy with Script

```bash
# Make deployment script executable
chmod +x scripts/deploy.sh

# Run deployment
./scripts/deploy.sh production
```

### 6. Manual Deployment (Alternative)

If you prefer manual deployment:

```bash
# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f app
```

## 🔧 Post-Deployment Configuration

### 1. Configure GitHub Webhook

1. Go to your GitHub App settings
2. Update webhook URL to: `https://your-domain.com/webhook`
3. Test the webhook delivery

### 2. Install GitHub App

1. Go to GitHub App settings → Install App
2. Select repositories to monitor
3. Complete installation

### 3. Verify Installation

```bash
# Test health endpoint
curl https://your-domain.com/health

# Test webhook endpoint (should require proper headers)
curl -X POST https://your-domain.com/webhook

# Check application logs
docker-compose -f docker-compose.prod.yml logs app | tail -100
```

## 📊 Monitoring Setup

### Access Monitoring Dashboards

- **Grafana**: `http://your-domain.com:3000`
  - Username: `admin`
  - Password: Value from `GRAFANA_PASSWORD` in `.env`

- **Prometheus**: `http://your-domain.com:9091`

### Import Dashboards

1. Log into Grafana
2. Go to Configuration → Data Sources
3. Add Prometheus data source: `http://prometheus:9090`
4. Import dashboards from `monitoring/grafana/dashboards/`

### Key Metrics to Monitor

- **Application Metrics**:
  - Request rate and response time
  - Error rate and types
  - GitHub API rate limits
  - Queue length and processing time

- **System Metrics**:
  - CPU and memory usage
  - Disk space and I/O
  - Network bandwidth
  - Container health

- **Business Metrics**:
  - Issues detected and resolved
  - Repair success rate
  - Repository coverage

## 🔒 Security Considerations

### Network Security
```bash
# Configure firewall (UFW example)
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw --force enable
```

### Database Security
- Use strong passwords
- Enable SSL connections
- Regular backup schedule
- Monitor access logs

### Application Security
- Rotate secrets regularly
- Monitor audit logs
- Review user permissions
- Keep dependencies updated

## 🔄 Maintenance Tasks

### Daily Operations

```bash
# Check service health
./scripts/health-check.sh

# View recent logs
docker-compose -f docker-compose.prod.yml logs --tail=100 app

# Monitor resource usage
docker stats
```

### Weekly Maintenance

```bash
# Update Docker images
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# Clean up old images
docker image prune -f

# Backup database
docker-compose -f docker-compose.prod.yml exec db pg_dump -U postgres self_healing_bot > backup_$(date +%Y%m%d).sql
```

### Monthly Tasks

- Review and rotate secrets
- Update SSL certificates
- Analyze performance metrics
- Review security logs
- Update documentation

## 🚨 Troubleshooting

### Common Issues

**Service won't start**:
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs [service-name]

# Check disk space
df -h

# Check memory usage
free -h
```

**Database connection issues**:
```bash
# Test database connectivity
docker-compose -f docker-compose.prod.yml exec app python -c "
from self_healing_bot.core.config import config
import psycopg2
conn = psycopg2.connect(config.database_url)
print('Database connection successful')
"
```

**GitHub webhook failures**:
```bash
# Check webhook logs
docker-compose -f docker-compose.prod.yml logs app | grep webhook

# Test webhook endpoint
curl -X POST https://your-domain.com/health
```

### Log Locations

- **Application logs**: `docker-compose logs app`
- **Database logs**: `docker-compose logs db`  
- **Nginx logs**: `docker-compose logs nginx`
- **System logs**: `/var/log/syslog`

## 📈 Scaling Considerations

### Horizontal Scaling

```yaml
# In docker-compose.prod.yml
worker:
  deploy:
    replicas: 4  # Scale workers

app:
  deploy:
    replicas: 2  # Scale web servers
```

### Vertical Scaling

```yaml
# In docker-compose.prod.yml
app:
  deploy:
    resources:
      limits:
        cpus: '2.0'
        memory: 4G
      reservations:
        cpus: '1.0'
        memory: 2G
```

### Load Balancing

For multiple instances, use:
- External load balancer (AWS ALB, GCP Load Balancer)
- Database connection pooling
- Redis clustering for high availability

## 🔐 Backup Strategy

### Database Backups

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

docker-compose -f docker-compose.prod.yml exec -T db pg_dump -U postgres self_healing_bot > \
    "$BACKUP_DIR/db_backup_$DATE.sql"

# Keep only last 7 days
find $BACKUP_DIR -name "db_backup_*.sql" -mtime +7 -delete
```

### Configuration Backups

```bash
# Backup critical configurations
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
    .env \
    keys/ \
    ssl/ \
    nginx.conf \
    docker-compose.prod.yml
```

## 📞 Support and Monitoring

### Health Endpoints

- **Application Health**: `GET /health`
- **Database Health**: `GET /health/db`  
- **GitHub Integration**: `GET /health/github`
- **Metrics**: `GET /metrics`

### Alerting Setup

Configure alerts for:
- Service downtime
- High error rates
- Resource exhaustion
- Security events
- GitHub API rate limits

### Support Contacts

- **Technical Issues**: Create GitHub issue
- **Security Concerns**: Email security@yourorg.com
- **Emergency**: Follow incident response procedure

---

## 📝 Additional Resources

- [API Documentation](./API.md)
- [Development Guide](./DEVELOPMENT.md)
- [Security Guide](./SECURITY.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)

For questions or issues, please create a GitHub issue or contact the development team.