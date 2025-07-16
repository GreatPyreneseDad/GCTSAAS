-- GCT SaaS Database Schema
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Organizations table
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size_category VARCHAR(20),
    cultural_context JSONB,
    optimization_parameters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID REFERENCES organizations(id),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(100),
    baseline_measurements JSONB,
    privacy_settings JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Coherence measurements (time-series)
CREATE TABLE coherence_measurements (
    user_id UUID NOT NULL REFERENCES users(id),
    timestamp TIMESTAMPTZ NOT NULL,
    psi_score DECIMAL(5,4),
    rho_score DECIMAL(5,4),
    q_score DECIMAL(5,4),
    f_score DECIMAL(5,4),
    coherence_score DECIMAL(5,4),
    q_optimal_score DECIMAL(5,4),
    measurement_context JSONB,
    confidence DECIMAL(3,2) DEFAULT 0.95,
    PRIMARY KEY (user_id, timestamp)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('coherence_measurements', 'timestamp');

-- Coherence derivatives
CREATE TABLE coherence_derivatives (
    user_id UUID NOT NULL REFERENCES users(id),
    timestamp TIMESTAMPTZ NOT NULL,
    dC_dt DECIMAL(8,6),
    dPsi_dt DECIMAL(8,6),
    dRho_dt DECIMAL(8,6),
    dQ_dt DECIMAL(8,6),
    dF_dt DECIMAL(8,6),
    dQ_optimal_dt DECIMAL(8,6),
    confidence_lower DECIMAL(8,6),
    confidence_upper DECIMAL(8,6),
    PRIMARY KEY (user_id, timestamp)
);

-- Inflection points
CREATE TABLE coherence_inflections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    timestamp TIMESTAMPTZ NOT NULL,
    inflection_type VARCHAR(50) NOT NULL,
    magnitude DECIMAL(6,3),
    duration_estimate_days INTEGER,
    contributing_factors JSONB,
    confidence DECIMAL(3,2),
    prediction_horizon_days INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_coherence_user_time ON coherence_measurements (user_id, timestamp DESC);
CREATE INDEX idx_derivatives_user_time ON coherence_derivatives (user_id, timestamp DESC);
CREATE INDEX idx_inflections_user_type ON coherence_inflections (user_id, inflection_type);
CREATE INDEX idx_users_org ON users (org_id);
