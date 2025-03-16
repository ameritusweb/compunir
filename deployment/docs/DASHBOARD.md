# Node Dashboard Documentation

## Overview
The Node Dashboard provides real-time monitoring and management capabilities for GPU nodes participating in the decentralized training network. This document covers configuration options and deployment procedures.

## Configuration Options

### Environment Variables
Create a `.env` file in the root directory with the following options:

```env
# Node Connection
NODE_API_URL=http://localhost:8000
NODE_WS_URL=ws://localhost:8001
REFRESH_INTERVAL=5000

# Security
AUTH_TOKEN=your_auth_token
ENABLE_SSL=true
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem

# Performance
MAX_CONCURRENT_CONNECTIONS=100
WEBSOCKET_HEARTBEAT_INTERVAL=30000
API_TIMEOUT=5000

# Metrics
METRICS_RETENTION_DAYS=30
GPU_METRICS_INTERVAL=1000
EARNINGS_DECIMALS=8
```

### Dashboard Configuration File
Create a `dashboard.config.js` file to customize dashboard behavior:

```javascript
module.exports = {
  // GPU Monitoring
  gpu: {
    warningTemp: 80, // Temperature warning threshold (Celsius)
    criticalTemp: 85, // Critical temperature threshold
    utilizationTarget: 95, // Target GPU utilization percentage
    memoryBuffer: 1024, // Memory buffer in MB
    metricHistory: 100 // Number of historical metrics to retain
  },

  // Job Settings
  jobs: {
    maxActiveJobs: 5, // Maximum concurrent jobs
    retryAttempts: 3, // Number of retry attempts for failed jobs
    minVerifiers: 3, // Minimum required verifiers per job
    autoRetryEnabled: true // Auto-retry failed jobs
  },

  // Earnings Display
  earnings: {
    currency: 'XMR',
    updateInterval: 300000, // 5 minutes
    historicalPeriods: ['24h', '7d', '30d', '90d']
  },

  // UI Customization
  ui: {
    theme: 'light', // 'light' or 'dark'
    refreshInterval: 5000,
    alertDuration: 5000,
    maxTableRows: 10
  }
}
```

## Deployment

### Prerequisites
- Node.js v18 or higher
- npm or yarn
- Access to a GPU node running the network client

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/node-dashboard
cd node-dashboard
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Build the application:
```bash
npm run build
# or
yarn build
```

### Production Deployment

#### Option 1: Docker Deployment

```dockerfile
# Dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
COPY . .

RUN npm install --production
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

Build and run the Docker container:
```bash
docker build -t node-dashboard .
docker run -p 3000:3000 -v /path/to/config:/app/config node-dashboard
```

#### Option 2: Traditional Deployment

1. Setup process manager:
```bash
npm install -g pm2
```

2. Create PM2 configuration:
```javascript
// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'node-dashboard',
    script: 'npm',
    args: 'start',
    env: {
      NODE_ENV: 'production',
      PORT: 3000
    },
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G'
  }]
}
```

3. Start the application:
```bash
pm2 start ecosystem.config.js
```

### SSL Configuration

1. Generate SSL certificates:
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout private.key -out certificate.crt
```

2. Update environment variables:
```env
ENABLE_SSL=true
SSL_CERT_PATH=/path/to/certificate.crt
SSL_KEY_PATH=/path/to/private.key
```

### Monitoring and Maintenance

#### Health Checks
The dashboard exposes health check endpoints:
- `/health` - Basic health check
- `/health/gpu` - GPU status
- `/health/metrics` - Metrics system status

#### Logging
Logs are stored in:
- `/logs/error.log` - Error logs
- `/logs/access.log` - Access logs
- `/logs/websocket.log` - WebSocket connection logs

Configure log rotation:
```javascript
// log-rotation.config.js
module.exports = {
  files: 'logs/*.log',
  size: '10M',
  compress: true,
  keep: 5
}
```

#### Backup
Regular backups of configuration and metrics:
```bash
#!/bin/bash
# backup-script.sh
backup_dir="/path/to/backups/$(date +%Y%m%d)"
mkdir -p "$backup_dir"
cp -r config/* "$backup_dir"/
cp -r metrics/* "$backup_dir"/
```

### Troubleshooting

Common issues and solutions:

1. **WebSocket Connection Failed**
   - Check NODE_WS_URL configuration
   - Verify network connectivity
   - Check SSL certificate if using HTTPS

2. **High Memory Usage**
   - Adjust METRICS_RETENTION_DAYS
   - Check for memory leaks using Node.js profiler
   - Adjust max_memory_restart in PM2 config

3. **GPU Metrics Not Updating**
   - Verify GPU driver installation
   - Check GPU_METRICS_INTERVAL setting
   - Ensure proper permissions for GPU access

4. **Dashboard Performance Issues**
   - Reduce REFRESH_INTERVAL
   - Adjust maxTableRows in UI config
   - Enable browser caching

## Security Considerations

1. **API Security**
   - Enable authentication
   - Use HTTPS
   - Implement rate limiting
   - Set secure CORS policies

2. **WebSocket Security**
   - Enable WebSocket authentication
   - Implement message validation
   - Set appropriate timeouts

3. **Access Control**
   - Implement role-based access
   - Set up IP whitelisting
   - Use secure session management

## API Reference

Documentation for the Node API endpoints used by the dashboard:

```typescript
interface NodeAPI {
  // GPU Metrics
  GET /api/gpu/metrics: {
    utilization: number;
    temperature: number;
    memory: {
      used: number;
      total: number;
    };
    power: number;
  };

  // Job Management
  GET /api/jobs: Job[];
  GET /api/jobs/:id: JobDetails;
  POST /api/jobs/:id/retry: { success: boolean };

  // Earnings
  GET /api/earnings/history: {
    date: string;
    earnings: number;
  }[];

  // Verification
  GET /api/verifications/stats: {
    total: number;
    success: number;
    rate: number;
  };
}
```

## Support and Updates

- GitHub Issues: Report bugs and feature requests
- Discord: Community support channel
- Documentation: Updated monthly
- Security Updates: Automated notifications via GitHub