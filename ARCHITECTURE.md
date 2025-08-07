# Architecture Overview

The Self-Healing MLOps Bot is designed as a microservices-based system that automatically detects and resolves issues in ML pipelines through intelligent monitoring and automated repair actions.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Integration                        │
├─────────────────────────────────────────────────────────────────┤
│  Webhooks  │  API Client  │  Pull Requests  │  Issue Creation  │
└─────────────────┬───────────────────────────────────────────────┘
                  │
          ┌───────▼───────┐
          │   Ingress     │
          │   Controller  │
          └───────┬───────┘
                  │
    ┌─────────────▼─────────────┐
    │     Load Balancer         │
    │   (NGINX/HAProxy)         │
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │    API Gateway            │
    │  - Authentication         │
    │  - Rate Limiting          │
    │  - Request Routing        │
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │   Web Application         │
    │   (FastAPI)               │
    └─────────────┬─────────────┘
                  │
         ┌────────┼────────┐
         │        │        │
    ┌────▼───┐┌──▼───┐┌───▼────┐
    │ Event  ││Core  ││  Web   │
    │Process ││Logic ││  API   │
    └────┬───┘└──┬───┘└───┬────┘
         │       │        │
    ┌────▼───────▼────────▼────┐
    │    Message Queue          │
    │    (Redis/RabbitMQ)       │
    └────┬──────────────────┬───┘
         │                  │
    ┌────▼─────┐       ┌────▼─────┐
    │  Worker  │       │  Worker  │
    │  Nodes   │  ...  │  Nodes   │
    └────┬─────┘       └────┬─────┘
         │                  │
    ┌────▼──────────────────▼─────┐
    │      Data Layer              │
    │  ┌─────────┐  ┌─────────┐   │
    │  │PostgreSQL│  │  Redis  │   │
    │  └─────────┘  └─────────┘   │
    └──────────────────────────────┘
```

## 🔧 Core Components

### 1. Web Application Layer

**FastAPI Application** (`self_healing_bot/web/app.py`)
- HTTP server for webhook handling and REST API
- Async request processing with middleware stack
- Authentication and authorization
- Rate limiting and security headers
- Health check endpoints

### 2. Event Processing Engine

**Bot Core** (`self_healing_bot/core/bot.py`)
- Central orchestrator for all bot operations
- Event routing and processing pipeline
- Context management for execution tracking
- Integration with all subsystems

### 3. Detection System

**Detector Registry** (`self_healing_bot/detectors/registry.py`)
- Pluggable detector architecture
- Event-based trigger system
- Parallel detection execution
- Result aggregation and prioritization

**Built-in Detectors**:
- **Pipeline Failure Detector**: CI/CD pipeline monitoring
- **Data Drift Detector**: Statistical drift detection
- **Model Degradation Detector**: Performance monitoring

### 4. Action System

**Playbook Engine** (`self_healing_bot/core/playbook.py`)
- YAML-defined repair workflows
- Dynamic action chaining
- Rollback capabilities
- Success/failure tracking

### 5. Data Layer

**PostgreSQL Database**:
- Execution history and audit trails
- Repository configurations
- Issue tracking and resolution status

**Redis Cache/Queue**:
- Session and response caching
- Task queue for background processing
- Rate limiting counters

## 🔄 Processing Flow

### Webhook Processing Pipeline

1. **Webhook Validation**: HMAC signature verification
2. **Rate Limiting**: Request throttling per repository
3. **Event Parsing**: GitHub event deserialization
4. **Context Creation**: Execution tracking and metadata
5. **Security Scanning**: Malicious payload detection
6. **Detector Execution**: Parallel issue detection
7. **Playbook Selection**: Automated repair workflow selection
8. **Action Execution**: Repair action implementation
9. **Result Validation**: Fix effectiveness verification
10. **Notification**: Team communication and documentation

## 🏗️ Scalability Features

- **Horizontal Pod Autoscaling**: CPU/memory based scaling
- **Adaptive Concurrency**: Resource-aware task scheduling
- **Intelligent Caching**: Multi-level caching with adaptive TTL
- **Circuit Breakers**: Fault tolerance for external services
- **Connection Pooling**: Efficient resource utilization

## 🔒 Security Architecture

- **Multi-layer Authentication**: GitHub App + API keys
- **Input Validation**: Comprehensive security scanning
- **Network Security**: TLS encryption and network policies
- **RBAC**: Role-based access control
- **Audit Logging**: Complete security event tracking

This architecture provides enterprise-grade reliability, security, and scalability for automated MLOps pipeline healing.