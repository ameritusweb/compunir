# Node Dashboard Quick Start Guide

## 5-Minute Setup

1. **Download and Install**
```bash
# Clone repository
git clone https://github.com/your-repo/node-dashboard
cd node-dashboard

# Install dependencies
npm install

# Create configuration
cp .env.example .env
```

2. **Configure Node Connection**
Edit `.env`:
```env
NODE_API_URL=http://your-node:8000
NODE_WS_URL=ws://your-node:8001
AUTH_TOKEN=your_auth_token
```

3. **Start Dashboard**
```bash
npm run dev
```

Visit `http://localhost:3000` to access your dashboard.

## Basic Configuration

### Minimal dashboard.config.js
```javascript
module.exports = {
  gpu: {
    warningTemp: 80,
    utilizationTarget: 95
  },
  jobs: {
    maxActiveJobs: 3,
    autoRetryEnabled: true
  },
  ui: {
    theme: 'light',
    refreshInterval: 5000
  }
}
```

## Common Tasks

### Monitor GPU Status
- Check GPU temperature and utilization in real-time
- Set up alerts for high temperature
- View historical performance graphs

### Track Earnings
- View daily, weekly, and monthly earnings
- Monitor pending payments
- Export earnings reports

### Manage Jobs
- View active and completed jobs
- Retry failed jobs
- Check verification status

## Next Steps

1. Configure SSL for secure access
2. Set up automated backups
3. Configure monitoring alerts
4. Join the community Discord

For detailed configuration options and advanced features, refer to the full documentation.