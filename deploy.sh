#!/bin/bash

# Self-Healing MLOps Bot Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="self-healing-bot"
DOCKER_IMAGE="ghcr.io/danieleschmidt/self-healing-mlops-bot:latest"
KUBECTL_CONTEXT=${KUBECTL_CONTEXT:-""}

# Helper functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        error "docker is not installed or not in PATH"
    fi
    
    # Check helm (optional)
    if ! command -v helm &> /dev/null; then
        warn "helm is not installed - some features may not be available"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    success "Prerequisites check passed"
}

# Deploy database and dependencies
deploy_dependencies() {
    log "Deploying dependencies (PostgreSQL, Redis)..."
    
    # Deploy PostgreSQL
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: ${NAMESPACE}
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: self_healing_bot
        - name: POSTGRES_USER
          value: bot_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: self-healing-bot-secrets
              key: DATABASE_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "250m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
EOF

    # Deploy Redis
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "128Mi"
            cpu: "50m"
          limits:
            memory: "256Mi"
            cpu: "100m"
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        emptyDir: {}
EOF

    success "Dependencies deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Deploy Prometheus ServiceMonitor
    cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: self-healing-bot-metrics
  namespace: ${NAMESPACE}
  labels:
    app: self-healing-bot
spec:
  selector:
    matchLabels:
      app: self-healing-bot
      component: api
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
EOF

    success "Monitoring stack deployed"
}

# Main deployment function
deploy_application() {
    log "Deploying Self-Healing MLOps Bot..."
    
    # Apply all Kubernetes manifests
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secrets.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/hpa.yaml
    kubectl apply -f k8s/pdb.yaml
    kubectl apply -f k8s/ingress.yaml
    
    success "Application manifests applied"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log "Waiting for deployment to be ready..."
    
    kubectl wait --for=condition=available --timeout=300s deployment/self-healing-bot -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/celery-worker -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/celery-beat -n ${NAMESPACE}
    
    success "All deployments are ready"
}

# Run health checks
health_check() {
    log "Running health checks..."
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get svc self-healing-bot-api -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
    
    # Port forward for health check
    kubectl port-forward svc/self-healing-bot-api 18080:80 -n ${NAMESPACE} &
    PF_PID=$!
    
    sleep 5
    
    # Health check
    if curl -f http://localhost:18080/health > /dev/null 2>&1; then
        success "Health check passed"
    else
        warn "Health check failed - check logs for details"
    fi
    
    # Clean up port forward
    kill $PF_PID 2>/dev/null || true
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo ""
    
    echo "Pods:"
    kubectl get pods -n ${NAMESPACE}
    echo ""
    
    echo "Services:"
    kubectl get svc -n ${NAMESPACE}
    echo ""
    
    echo "Ingress:"
    kubectl get ingress -n ${NAMESPACE}
    echo ""
    
    echo "HPA Status:"
    kubectl get hpa -n ${NAMESPACE}
    echo ""
}

# Cleanup function
cleanup() {
    if [[ "$1" == "--all" ]]; then
        log "Cleaning up all resources..."
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
        success "All resources cleaned up"
    else
        warn "Use --all flag to cleanup all resources"
    fi
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        check_prerequisites
        deploy_dependencies
        deploy_application
        deploy_monitoring
        wait_for_deployment
        health_check
        show_status
        success "Deployment completed successfully!"
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        cleanup $2
        ;;
    "health")
        health_check
        ;;
    *)
        echo "Usage: $0 {deploy|status|cleanup|health}"
        echo "  deploy  - Deploy the application"
        echo "  status  - Show deployment status"
        echo "  cleanup - Clean up resources (use --all for complete cleanup)"
        echo "  health  - Run health checks"
        exit 1
        ;;
esac