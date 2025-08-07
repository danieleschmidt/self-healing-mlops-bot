# API Documentation

The Self-Healing MLOps Bot provides both webhook endpoints for GitHub integration and REST API endpoints for management and monitoring.

## 🔗 Base URLs

- **Production**: `https://mlops-bot.yourdomain.com`
- **Staging**: `https://staging-mlops-bot.yourdomain.com`  
- **Local Development**: `http://localhost:8080`

## 📡 Webhook Endpoints

### GitHub Webhook

**Endpoint**: `POST /webhooks/github`

Receives GitHub webhook events for repository monitoring and automated actions.

**Headers Required**:
```
Content-Type: application/json
X-GitHub-Event: <event-type>
X-GitHub-Delivery: <unique-delivery-id>
X-Hub-Signature-256: sha256=<signature>
```

**Supported Events**:
- `workflow_run` - CI/CD pipeline events
- `push` - Code push events  
- `pull_request` - PR lifecycle events
- `issues` - Issue management events
- `deployment` - Deployment status events

**Example Request**:
```bash
curl -X POST https://mlops-bot.yourdomain.com/webhooks/github \\
  -H \"Content-Type: application/json\" \\
  -H \"X-GitHub-Event: workflow_run\" \\
  -H \"X-GitHub-Delivery: 12345678-1234-1234-1234-123456789012\" \\
  -H \"X-Hub-Signature-256: sha256=abc123...\" \\
  -d '{\"action\":\"completed\",\"workflow_run\":{...}}'
```

**Response**:
```json
{
  \"status\": \"accepted\",
  \"execution_id\": \"550e8400-e29b-41d4-a716-446655440000\",
  \"message\": \"Webhook processed successfully\"
}
```

**Error Responses**:
- `400 Bad Request` - Invalid payload or missing headers
- `401 Unauthorized` - Invalid webhook signature
- `422 Unprocessable Entity` - Unsupported event type
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Processing error

## 🔧 Management API

### Authentication

All management endpoints require API key authentication:

**Header**: `Authorization: Bearer <api-key>`

### Health Endpoints

#### System Health Check
**Endpoint**: `GET /health`

Returns overall system health status.

**Response**:
```json
{
  \"status\": \"healthy\",
  \"timestamp\": \"2024-01-15T10:30:00Z\",
  \"uptime_seconds\": 86400,
  \"version\": \"1.0.0\",
  \"environment\": \"production\"
}
```

#### Detailed Health Check
**Endpoint**: `GET /health/detailed`

Returns comprehensive health information for all components.

**Response**:
```json
{
  \"status\": \"healthy\",
  \"timestamp\": \"2024-01-15T10:30:00Z\",
  \"uptime_seconds\": 86400,
  \"active_executions\": 5,
  \"components\": {
    \"database\": {
      \"status\": \"healthy\",
      \"message\": \"Database connection successful\",
      \"execution_time\": 0.045,
      \"details\": {
        \"connection_pool\": \"active\",
        \"active_connections\": 8
      }
    },
    \"redis\": {
      \"status\": \"healthy\", 
      \"message\": \"Redis connection successful\",
      \"execution_time\": 0.012
    },
    \"github_api\": {
      \"status\": \"healthy\",
      \"message\": \"GitHub API accessible\",
      \"execution_time\": 0.234,
      \"details\": {
        \"rate_limit_remaining\": 4850
      }
    }
  }
}
```

#### Readiness Check
**Endpoint**: `GET /health/ready`

Kubernetes readiness probe endpoint.

**Response**: `200 OK` if ready, `503 Service Unavailable` if not ready.

#### Liveness Check  
**Endpoint**: `GET /health/live`

Kubernetes liveness probe endpoint.

**Response**: `200 OK` if alive, `500 Internal Server Error` if not alive.

### Repository Management

#### List Monitored Repositories
**Endpoint**: `GET /api/v1/repositories`

**Query Parameters**:
- `limit` (int): Maximum number of results (default: 50)
- `offset` (int): Pagination offset (default: 0)
- `status` (string): Filter by status (`active`, `paused`, `error`)

**Response**:
```json
{
  \"repositories\": [
    {
      \"id\": \"123\",
      \"full_name\": \"owner/repo\",
      \"status\": \"active\",
      \"last_event\": \"2024-01-15T10:25:00Z\",
      \"issues_detected\": 12,
      \"repairs_attempted\": 8,
      \"success_rate\": 0.85
    }
  ],
  \"total\": 1,
  \"limit\": 50,
  \"offset\": 0
}
```

#### Get Repository Details
**Endpoint**: `GET /api/v1/repositories/{owner}/{repo}`

**Response**:
```json
{
  \"id\": \"123\",
  \"full_name\": \"owner/repo\",
  \"status\": \"active\",
  \"configuration\": {
    \"auto_fix_enabled\": true,
    \"notification_channels\": [\"slack\", \"email\"],
    \"detection_sensitivity\": \"medium\"
  },
  \"statistics\": {
    \"total_events\": 456,
    \"issues_detected\": 23,
    \"repairs_successful\": 18,
    \"repairs_failed\": 3,
    \"average_response_time\": 2.3,
    \"last_activity\": \"2024-01-15T10:25:00Z\"
  },
  \"recent_issues\": [
    {
      \"id\": \"issue-123\",
      \"type\": \"pipeline_failure\",
      \"severity\": \"high\",
      \"detected_at\": \"2024-01-15T10:20:00Z\",
      \"resolved\": true,
      \"resolution_time\": 180
    }
  ]
}
```

#### Update Repository Configuration
**Endpoint**: `PUT /api/v1/repositories/{owner}/{repo}/config`

**Request Body**:
```json
{
  \"auto_fix_enabled\": true,
  \"notification_channels\": [\"slack\", \"email\"],
  \"detection_sensitivity\": \"high\",
  \"custom_playbooks\": [\"custom-fix-1\", \"custom-fix-2\"]
}
```

**Response**:
```json
{
  \"message\": \"Configuration updated successfully\",
  \"repository\": \"owner/repo\",
  \"updated_at\": \"2024-01-15T10:30:00Z\"
}
```

### Issue Management

#### List Issues
**Endpoint**: `GET /api/v1/issues`

**Query Parameters**:
- `repository` (string): Filter by repository
- `type` (string): Filter by issue type
- `severity` (string): Filter by severity
- `status` (string): Filter by status (`open`, `resolved`, `failed`)
- `limit` (int): Maximum results (default: 50)
- `offset` (int): Pagination offset (default: 0)

**Response**:
```json
{
  \"issues\": [
    {
      \"id\": \"issue-123\",
      \"repository\": \"owner/repo\",
      \"type\": \"data_drift\",
      \"severity\": \"medium\", 
      \"status\": \"resolved\",
      \"detected_at\": \"2024-01-15T10:15:00Z\",
      \"resolved_at\": \"2024-01-15T10:18:00Z\",
      \"description\": \"Data drift detected with score 0.15\",
      \"actions_taken\": [
        {
          \"action\": \"retrain_model\",
          \"status\": \"completed\",
          \"executed_at\": \"2024-01-15T10:17:00Z\"
        }
      ]
    }
  ],
  \"total\": 1,
  \"limit\": 50,
  \"offset\": 0
}
```

#### Get Issue Details
**Endpoint**: `GET /api/v1/issues/{issue-id}`

**Response**:
```json
{
  \"id\": \"issue-123\",
  \"repository\": \"owner/repo\",
  \"type\": \"pipeline_failure\",
  \"severity\": \"high\",
  \"status\": \"resolved\",
  \"detected_at\": \"2024-01-15T10:15:00Z\",
  \"resolved_at\": \"2024-01-15T10:18:00Z\",
  \"description\": \"CI pipeline failed due to import error\",
  \"metadata\": {
    \"workflow_name\": \"CI\",
    \"workflow_run_id\": 789,
    \"failure_reason\": \"ModuleNotFoundError: No module named 'numpy'\",
    \"affected_files\": [\"src/train.py\"]
  },
  \"actions_taken\": [
    {
      \"action\": \"fix_import_error\",
      \"status\": \"completed\",
      \"executed_at\": \"2024-01-15T10:17:00Z\",
      \"details\": {
        \"changes_made\": [\"Added numpy to requirements.txt\"],
        \"pull_request\": \"https://github.com/owner/repo/pull/42\"
      }
    }
  ],
  \"timeline\": [
    {
      \"timestamp\": \"2024-01-15T10:15:00Z\",
      \"event\": \"issue_detected\",
      \"details\": \"Pipeline failure detected by webhook\"
    },
    {
      \"timestamp\": \"2024-01-15T10:16:30Z\",
      \"event\": \"analysis_started\",
      \"details\": \"Analyzing failure logs\"
    },
    {
      \"timestamp\": \"2024-01-15T10:17:00Z\",
      \"event\": \"action_executed\",
      \"details\": \"Applied import error fix\"
    },
    {
      \"timestamp\": \"2024-01-15T10:18:00Z\",
      \"event\": \"issue_resolved\",
      \"details\": \"Pipeline re-run successful\"
    }
  ]
}
```

#### Manually Trigger Issue Resolution
**Endpoint**: `POST /api/v1/issues/{issue-id}/resolve`

**Request Body**:
```json
{
  \"action\": \"retry_failed_action\",
  \"force\": false,
  \"parameters\": {
    \"custom_parameter\": \"value\"
  }
}
```

**Response**:
```json
{
  \"message\": \"Issue resolution triggered\",
  \"execution_id\": \"550e8400-e29b-41d4-a716-446655440001\",
  \"estimated_completion\": \"2024-01-15T10:35:00Z\"
}
```

### Playbook Management

#### List Playbooks
**Endpoint**: `GET /api/v1/playbooks`

**Response**:
```json
{
  \"playbooks\": [
    {
      \"name\": \"pipeline_failure_fix\",
      \"version\": \"1.2.0\",
      \"description\": \"Automatically fix common pipeline failures\",
      \"supported_issues\": [\"pipeline_failure\", \"test_failure\"],
      \"success_rate\": 0.89,
      \"enabled\": true,
      \"last_updated\": \"2024-01-10T15:30:00Z\"
    },
    {
      \"name\": \"data_drift_response\",
      \"version\": \"1.0.0\",
      \"description\": \"Handle data drift detection and model retraining\",
      \"supported_issues\": [\"data_drift\", \"model_degradation\"],
      \"success_rate\": 0.76,
      \"enabled\": true,
      \"last_updated\": \"2024-01-08T09:15:00Z\"
    }
  ],
  \"total\": 2
}
```

#### Get Playbook Details
**Endpoint**: `GET /api/v1/playbooks/{playbook-name}`

**Response**:
```json
{
  \"name\": \"pipeline_failure_fix\",
  \"version\": \"1.2.0\",
  \"description\": \"Automatically fix common pipeline failures\",
  \"author\": \"MLOps Team\",
  \"created_at\": \"2023-12-01T10:00:00Z\",
  \"last_updated\": \"2024-01-10T15:30:00Z\",
  \"supported_issues\": [\"pipeline_failure\", \"test_failure\"],
  \"configuration\": {
    \"timeout\": 1800,
    \"retry_count\": 3,
    \"rollback_on_failure\": true
  },
  \"statistics\": {
    \"total_executions\": 234,
    \"successful_executions\": 208,
    \"failed_executions\": 26,
    \"success_rate\": 0.89,
    \"average_execution_time\": 145.6
  },
  \"actions\": [
    {
      \"type\": \"code_analysis\",
      \"description\": \"Analyze failure logs and error messages\"
    },
    {
      \"type\": \"dependency_fix\", 
      \"description\": \"Fix missing dependencies\"
    },
    {
      \"type\": \"create_pull_request\",
      \"description\": \"Create PR with fixes\"
    }
  ]
}
```

#### Enable/Disable Playbook
**Endpoint**: `PATCH /api/v1/playbooks/{playbook-name}`

**Request Body**:
```json
{
  \"enabled\": false,
  \"reason\": \"Temporarily disabled for testing\"
}
```

**Response**:
```json
{
  \"message\": \"Playbook updated successfully\",
  \"playbook\": \"pipeline_failure_fix\",
  \"enabled\": false,
  \"updated_at\": \"2024-01-15T10:30:00Z\"
}
```

### Metrics and Analytics

#### System Metrics
**Endpoint**: `GET /metrics`

Returns Prometheus-format metrics for system monitoring.

**Response**: Prometheus format text
```
# HELP bot_events_processed_total Total number of events processed
# TYPE bot_events_processed_total counter
bot_events_processed_total{event_type=\"workflow_run\",repo=\"owner/repo\",status=\"success\"} 245

# HELP bot_event_processing_duration_seconds Time spent processing events
# TYPE bot_event_processing_duration_seconds histogram
bot_event_processing_duration_seconds_bucket{event_type=\"workflow_run\",repo=\"owner/repo\",le=\"1\"} 100
...
```

#### Analytics Dashboard Data
**Endpoint**: `GET /api/v1/analytics`

**Query Parameters**:
- `period` (string): Time period (`1h`, `24h`, `7d`, `30d`)
- `repository` (string): Filter by repository
- `metric_type` (string): Metric type filter

**Response**:
```json
{
  \"period\": \"24h\",
  \"summary\": {
    \"total_events\": 1245,
    \"issues_detected\": 23,
    \"issues_resolved\": 19,
    \"resolution_rate\": 0.83,
    \"average_resolution_time\": 145.6,
    \"most_common_issue_type\": \"pipeline_failure\"
  },
  \"timeseries\": {
    \"events_per_hour\": [
      {\"timestamp\": \"2024-01-15T00:00:00Z\", \"value\": 45},
      {\"timestamp\": \"2024-01-15T01:00:00Z\", \"value\": 38},
      // ... hourly data points
    ],
    \"resolution_times\": [
      {\"timestamp\": \"2024-01-15T00:00:00Z\", \"value\": 120},
      {\"timestamp\": \"2024-01-15T01:00:00Z\", \"value\": 156},
      // ... hourly averages
    ]
  },
  \"breakdown_by_type\": {
    \"pipeline_failure\": {\"count\": 12, \"resolved\": 10},
    \"data_drift\": {\"count\": 5, \"resolved\": 5},
    \"test_failure\": {\"count\": 6, \"resolved\": 4}
  },
  \"top_repositories\": [
    {\"repository\": \"owner/repo1\", \"events\": 234, \"issues\": 12},
    {\"repository\": \"owner/repo2\", \"events\": 187, \"issues\": 8}
  ]
}
```

### Configuration Management

#### Get System Configuration
**Endpoint**: `GET /api/v1/config`

**Response**:
```json
{
  \"version\": \"1.0.0\",
  \"environment\": \"production\",
  \"features\": {
    \"auto_scaling\": true,
    \"predictive_scaling\": true,
    \"advanced_analytics\": true
  },
  \"limits\": {
    \"max_concurrent_executions\": 50,
    \"rate_limit_per_minute\": 1000,
    \"max_file_size_mb\": 10
  },
  \"integrations\": {
    \"github\": {\"enabled\": true, \"rate_limit_remaining\": 4850},
    \"slack\": {\"enabled\": true, \"channels_configured\": 3},
    \"email\": {\"enabled\": false}
  }
}
```

#### Update System Configuration
**Endpoint**: `PUT /api/v1/config`

**Request Body**:
```json
{
  \"features\": {
    \"auto_scaling\": true,
    \"predictive_scaling\": false
  },
  \"limits\": {
    \"max_concurrent_executions\": 75,
    \"rate_limit_per_minute\": 1500
  }
}
```

**Response**:
```json
{
  \"message\": \"Configuration updated successfully\",
  \"updated_at\": \"2024-01-15T10:30:00Z\",
  \"restart_required\": false
}
```

## 🚨 Error Handling

### Error Response Format

All API errors follow this format:

```json
{
  \"error\": {
    \"code\": \"INVALID_REQUEST\",
    \"message\": \"The request is invalid\",
    \"details\": \"Missing required field 'repository'\",
    \"request_id\": \"550e8400-e29b-41d4-a716-446655440002\",
    \"timestamp\": \"2024-01-15T10:30:00Z\"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Request format is invalid |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `VALIDATION_FAILED` | 422 | Request validation failed |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## 📊 Rate Limiting

### Rate Limits

| Endpoint Category | Limit | Window |
|-------------------|-------|---------|
| Webhooks | 100 requests | 5 minutes |
| Management API | 1000 requests | 1 minute |
| Analytics | 100 requests | 1 minute |
| Health Checks | Unlimited | - |

### Rate Limit Headers

All responses include rate limit headers:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
X-RateLimit-Window: 60
```

When rate limit is exceeded:

```json
{
  \"error\": {
    \"code\": \"RATE_LIMITED\",
    \"message\": \"Rate limit exceeded\",
    \"details\": \"Try again in 60 seconds\",
    \"retry_after\": 60
  }
}
```

## 🔐 Security

### API Key Authentication

Generate API keys through the web interface or CLI:

```bash
# Generate new API key
mlops-bot generate-api-key --name \"monitoring-system\" --permissions \"read\"

# List API keys  
mlops-bot list-api-keys

# Revoke API key
mlops-bot revoke-api-key --id \"key-123\"
```

### Webhook Security

GitHub webhooks are secured with HMAC signatures:

1. Configure webhook secret in GitHub
2. Bot verifies `X-Hub-Signature-256` header
3. Requests with invalid signatures are rejected

### HTTPS Only

All API endpoints require HTTPS in production. HTTP requests are automatically redirected to HTTPS.

## 📈 Monitoring

### Health Check Endpoints

Use these endpoints for monitoring and alerting:

- **Liveness**: `GET /health/live` (Kubernetes liveness probe)
- **Readiness**: `GET /health/ready` (Kubernetes readiness probe)  
- **Health**: `GET /health` (Overall health status)
- **Detailed**: `GET /health/detailed` (Component-level health)

### Metrics Collection

Prometheus metrics available at `/metrics` endpoint include:

- Request rates and latencies
- Error rates by endpoint
- Resource utilization
- Business metrics (issues detected, resolved, etc.)
- External service health (GitHub API, database, etc.)

## 🧪 Testing

### API Testing

Example test requests:

```bash
# Health check
curl -f https://mlops-bot.yourdomain.com/health

# Get repository stats  
curl -H \"Authorization: Bearer your-api-key\" \\
     https://mlops-bot.yourdomain.com/api/v1/repositories/owner/repo

# List recent issues
curl -H \"Authorization: Bearer your-api-key\" \\
     \"https://mlops-bot.yourdomain.com/api/v1/issues?limit=10&status=resolved\"

# Trigger manual resolution
curl -X POST \\
     -H \"Authorization: Bearer your-api-key\" \\
     -H \"Content-Type: application/json\" \\
     -d '{\"action\": \"retry_failed_action\"}' \\
     https://mlops-bot.yourdomain.com/api/v1/issues/issue-123/resolve
```

### Mock Webhook Testing

Test webhook processing with mock payloads:

```bash
# Create test payload
cat > test_webhook.json << EOF
{
  \"action\": \"completed\",
  \"workflow_run\": {
    \"id\": 123,
    \"name\": \"CI\",
    \"conclusion\": \"failure\",
    \"html_url\": \"https://github.com/owner/repo/actions/runs/123\"
  },
  \"repository\": {
    \"full_name\": \"owner/repo\",
    \"name\": \"repo\",
    \"owner\": {\"login\": \"owner\"}
  }
}
EOF

# Calculate HMAC signature
signature=$(echo -n \"$(cat test_webhook.json)\" | openssl dgst -sha256 -hmac \"your-webhook-secret\" | sed 's/^.* //')

# Send test webhook
curl -X POST \\
     -H \"Content-Type: application/json\" \\
     -H \"X-GitHub-Event: workflow_run\" \\
     -H \"X-GitHub-Delivery: test-delivery-123\" \\
     -H \"X-Hub-Signature-256: sha256=$signature\" \\
     -d @test_webhook.json \\
     https://mlops-bot.yourdomain.com/webhooks/github
```

## 📚 SDK and Examples

### Python SDK

```python
from mlops_bot_sdk import MLOpsBotClient

# Initialize client
client = MLOpsBotClient(
    base_url=\"https://mlops-bot.yourdomain.com\",
    api_key=\"your-api-key\"
)

# Get repository statistics
repo_stats = client.repositories.get(\"owner/repo\")
print(f\"Success rate: {repo_stats.success_rate}\")

# List recent issues
issues = client.issues.list(status=\"resolved\", limit=10)
for issue in issues:
    print(f\"{issue.type}: {issue.description}\")

# Trigger manual resolution
result = client.issues.resolve(\"issue-123\", action=\"retry\")
print(f\"Resolution triggered: {result.execution_id}\")
```

### JavaScript SDK

```javascript
import { MLOpsBotClient } from '@mlops-bot/sdk';

const client = new MLOpsBotClient({
  baseURL: 'https://mlops-bot.yourdomain.com',
  apiKey: 'your-api-key'
});

// Get system health
const health = await client.health.getDetailed();
console.log(`System status: ${health.status}`);

// Get analytics
const analytics = await client.analytics.get({ period: '24h' });
console.log(`Issues resolved: ${analytics.summary.issues_resolved}`);
```

This comprehensive API documentation covers all endpoints, authentication, error handling, rate limiting, security, and provides practical examples for integration and testing.