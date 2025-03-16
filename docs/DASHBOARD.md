# Dashboard Guide

This guide provides comprehensive information on setting up, configuring, and using the Compunir Node Dashboard for monitoring your decentralized GPU node.

## Overview

The Node Dashboard provides real-time monitoring and management capabilities for GPU nodes participating in the decentralized training network. Key features include:

- Real-time GPU monitoring (utilization, temperature, memory)
- Job tracking and management
- Earnings visualization and reporting
- Performance metrics and history
- Verification statistics
- Sybil protection metrics

## Installation

### Prerequisites

- Node.js v18 or higher
- npm or yarn
- Access to a running GPU node

### Basic Installation

1. Clone the repository if you haven't already:
```bash
git clone https://github.com/ameritusweb/compunir
cd compunir/dashboard
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Create a configuration file:
```bash
cp .env.example .env
```

4. Edit the `.env` file with your node's connection details:
```
NODE_API_URL=http://your-node:8000
NODE_WS_URL=ws://your-node:8001
AUTH_TOKEN=your_auth_token
```

5. Start the development server:
```bash
npm run dev
# or
yarn dev
```

6. Access the dashboard at `http://localhost:3000`

### Production Deployment

For production deployment, use one of the following methods:

#### Using Docker (Recommended)

1. Build the Docker image:
```bash
cd deployment/production
docker build -t node-dashboard .
```

2. Run the container:
```bash
docker run -p 3000:3000 -v /path/to/config:/app/config node-dashboard
```

#### Traditional Deployment

1. Build the application:
```bash
npm run build
# or
yarn build
```

2. Set up a process manager:
```bash
npm install -g pm2
pm2 start ecosystem.config.js
```

3. Configure a reverse proxy (Nginx recommended):
```nginx
server {
    listen 80;
    server_name your-dashboard-domain.com;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## Configuration

The dashboard can be configured through environment variables and a configuration file.

### Environment Variables

Create a `.env` file in the root directory with the following options:

```env
# Node Connection
NODE_API_URL=http://localhost:8000       # URL of your node's API
NODE_WS_URL=ws://localhost:8001          # WebSocket URL for real-time updates
REFRESH_INTERVAL=5000                    # UI refresh interval in milliseconds

# Security
AUTH_TOKEN=your_auth_token               # Authentication token for API access
ENABLE_SSL=true                          # Enable SSL/TLS for secure connections
SSL_CERT_PATH=/path/to/cert.pem          # Path to SSL certificate
SSL_KEY_PATH=/path/to/key.pem            # Path to SSL private key

# Performance
MAX_CONCURRENT_CONNECTIONS=100           # Maximum number of WebSocket connections
WEBSOCKET_HEARTBEAT_INTERVAL=30000       # WebSocket heartbeat interval (ms)
API_TIMEOUT=5000                         # API request timeout (ms)

# Metrics
METRICS_RETENTION_DAYS=30                # How long to keep historical metrics
GPU_METRICS_INTERVAL=1000                # GPU metrics collection interval (ms)
EARNINGS_DECIMALS=8                      # Decimal precision for earnings display
```

### Dashboard Configuration File

Create a `dashboard.config.js` file to customize dashboard behavior:

```javascript
module.exports = {
  // GPU Monitoring
  gpu: {
    warningTemp: 80,         // Temperature warning threshold (Celsius)
    criticalTemp: 85,        // Critical temperature threshold
    utilizationTarget: 95,   // Target GPU utilization percentage
    memoryBuffer: 1024,      // Memory buffer in MB
    metricHistory: 100       // Number of historical metrics to retain
  },

  // Job Settings
  jobs: {
    maxActiveJobs: 5,        // Maximum concurrent jobs
    retryAttempts: 3,        // Number of retry attempts for failed jobs
    minVerifiers: 3,         // Minimum required verifiers per job
    autoRetryEnabled: true   // Auto-retry failed jobs
  },

  // Earnings Display
  earnings: {
    currency: 'XMR',
    updateInterval: 300000,  // 5 minutes
    historicalPeriods: ['24h', '7d', '30d', '90d']
  },

  // UI Customization
  ui: {
    theme: 'light',          // 'light' or 'dark'
    refreshInterval: 5000,   // UI refresh interval in milliseconds
    alertDuration: 5000,     // How long alerts are displayed
    maxTableRows: 10         // Maximum rows in data tables
  }
}