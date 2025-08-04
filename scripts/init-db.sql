-- Self-Healing MLOps Bot Database Initialization
-- This script creates the necessary database schema and initial data

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS bot;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set search path
SET search_path = bot, public;

-- Execution contexts table
CREATE TABLE IF NOT EXISTS execution_contexts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id VARCHAR(255) UNIQUE NOT NULL,
    repo_owner VARCHAR(255) NOT NULL,
    repo_name VARCHAR(255) NOT NULL,
    repo_full_name VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    installation_id INTEGER,
    error_type VARCHAR(255),
    error_message TEXT,
    state JSONB DEFAULT '{}',
    file_changes JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Issues table
CREATE TABLE IF NOT EXISTS issues (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID REFERENCES execution_contexts(id) ON DELETE CASCADE,
    issue_type VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    data JSONB DEFAULT '{}',
    detector VARCHAR(255) NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    resolution_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Playbook executions table
CREATE TABLE IF NOT EXISTS playbook_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID REFERENCES execution_contexts(id) ON DELETE CASCADE,
    playbook_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL, -- 'running', 'completed', 'failed'
    results JSONB DEFAULT '[]',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

-- Action results table
CREATE TABLE IF NOT EXISTS action_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    playbook_execution_id UUID REFERENCES playbook_executions(id) ON DELETE CASCADE,
    action_name VARCHAR(255) NOT NULL,
    success BOOLEAN NOT NULL,
    message TEXT,
    data JSONB DEFAULT '{}',
    execution_time REAL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- GitHub operations table
CREATE TABLE IF NOT EXISTS github_operations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID REFERENCES execution_contexts(id) ON DELETE CASCADE,
    operation_type VARCHAR(100) NOT NULL, -- 'create_pr', 'create_issue', 'add_comment'
    repository VARCHAR(255) NOT NULL,
    github_id INTEGER, -- PR number, issue number, etc.
    url TEXT,
    status VARCHAR(50) NOT NULL,
    request_data JSONB,
    response_data JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Monitoring schema tables
SET search_path = monitoring, public;

-- Metrics table for storing custom metrics
CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value REAL NOT NULL,
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System health table
CREATE TABLE IF NOT EXISTS system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL, -- 'healthy', 'degraded', 'unhealthy'
    details JSONB DEFAULT '{}',
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance tracking table
CREATE TABLE IF NOT EXISTS performance_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operation VARCHAR(255) NOT NULL,
    duration REAL NOT NULL, -- in seconds
    success BOOLEAN NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit schema tables
SET search_path = audit, public;

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(255) NOT NULL,
    user_id VARCHAR(255), -- Could be 'system' for automated actions
    resource_type VARCHAR(255),
    resource_id VARCHAR(255),
    action VARCHAR(255) NOT NULL,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Security events table
CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    source VARCHAR(255),
    details JSONB NOT NULL,
    investigated BOOLEAN DEFAULT FALSE,
    resolution TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    investigated_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for better performance
SET search_path = bot, public;

-- Execution contexts indexes
CREATE INDEX IF NOT EXISTS idx_execution_contexts_repo ON execution_contexts(repo_full_name);
CREATE INDEX IF NOT EXISTS idx_execution_contexts_event_type ON execution_contexts(event_type);
CREATE INDEX IF NOT EXISTS idx_execution_contexts_created_at ON execution_contexts(created_at);
CREATE INDEX IF NOT EXISTS idx_execution_contexts_installation_id ON execution_contexts(installation_id);

-- Issues indexes
CREATE INDEX IF NOT EXISTS idx_issues_execution_id ON issues(execution_id);
CREATE INDEX IF NOT EXISTS idx_issues_type_severity ON issues(issue_type, severity);
CREATE INDEX IF NOT EXISTS idx_issues_resolved ON issues(resolved);
CREATE INDEX IF NOT EXISTS idx_issues_created_at ON issues(created_at);

-- Playbook executions indexes
CREATE INDEX IF NOT EXISTS idx_playbook_executions_execution_id ON playbook_executions(execution_id);
CREATE INDEX IF NOT EXISTS idx_playbook_executions_name_status ON playbook_executions(playbook_name, status);
CREATE INDEX IF NOT EXISTS idx_playbook_executions_started_at ON playbook_executions(started_at);

-- GitHub operations indexes
CREATE INDEX IF NOT EXISTS idx_github_operations_execution_id ON github_operations(execution_id);
CREATE INDEX IF NOT EXISTS idx_github_operations_repo ON github_operations(repository);
CREATE INDEX IF NOT EXISTS idx_github_operations_type ON github_operations(operation_type);

-- Monitoring indexes
SET search_path = monitoring, public;

CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_system_health_component ON system_health(component);
CREATE INDEX IF NOT EXISTS idx_system_health_checked_at ON system_health(checked_at);
CREATE INDEX IF NOT EXISTS idx_performance_logs_operation ON performance_logs(operation);
CREATE INDEX IF NOT EXISTS idx_performance_logs_created_at ON performance_logs(created_at);

-- Audit indexes
SET search_path = audit, public;

CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_security_events_type_severity ON security_events(event_type, severity);
CREATE INDEX IF NOT EXISTS idx_security_events_investigated ON security_events(investigated);

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to relevant tables
SET search_path = bot, public;

CREATE TRIGGER update_execution_contexts_updated_at 
    BEFORE UPDATE ON execution_contexts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW recent_executions AS
SELECT 
    ec.id,
    ec.execution_id,
    ec.repo_full_name,
    ec.event_type,
    ec.created_at,
    ec.error_type,
    COUNT(i.id) as issue_count,
    COUNT(pe.id) as playbook_count
FROM execution_contexts ec
LEFT JOIN issues i ON ec.id = i.execution_id
LEFT JOIN playbook_executions pe ON ec.id = pe.execution_id
WHERE ec.created_at >= NOW() - INTERVAL '24 hours'
GROUP BY ec.id, ec.execution_id, ec.repo_full_name, ec.event_type, ec.created_at, ec.error_type
ORDER BY ec.created_at DESC;

CREATE OR REPLACE VIEW issue_summary AS
SELECT 
    issue_type,
    severity,
    COUNT(*) as count,
    COUNT(CASE WHEN resolved THEN 1 END) as resolved_count,
    MAX(created_at) as last_seen
FROM bot.issues 
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY issue_type, severity
ORDER BY count DESC;

-- Insert initial system health record
SET search_path = monitoring, public;

INSERT INTO system_health (component, status, details) 
VALUES ('database', 'healthy', '{"message": "Database initialized successfully"}')
ON CONFLICT DO NOTHING;

-- Create bot user (for application use)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'bot_user') THEN
        CREATE ROLE bot_user WITH LOGIN PASSWORD 'change_this_password';
    END IF;
END
$$;

-- Grant permissions
GRANT USAGE ON SCHEMA bot TO bot_user;
GRANT USAGE ON SCHEMA monitoring TO bot_user;
GRANT USAGE ON SCHEMA audit TO bot_user;

GRANT ALL ON ALL TABLES IN SCHEMA bot TO bot_user;
GRANT ALL ON ALL TABLES IN SCHEMA monitoring TO bot_user;
GRANT ALL ON ALL TABLES IN SCHEMA audit TO bot_user;

GRANT ALL ON ALL SEQUENCES IN SCHEMA bot TO bot_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA monitoring TO bot_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA audit TO bot_user;

-- Log initialization completion
INSERT INTO audit.audit_logs (event_type, user_id, action, details)
VALUES ('system', 'database_init', 'initialize', '{"message": "Database schema initialized successfully"}');

-- Show completion message
DO $$
BEGIN
    RAISE NOTICE 'Self-Healing MLOps Bot database initialization completed successfully!';
    RAISE NOTICE 'Schemas created: bot, monitoring, audit';
    RAISE NOTICE 'Tables created: % total', (
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema IN ('bot', 'monitoring', 'audit')
    );
END
$$;