# Installation Guide

This guide will walk you through the process of installing and setting up the Compunir decentralized GPU training network.

## Prerequisites

Before installation, ensure your system meets the following requirements:

### Hardware Requirements
- **GPU**: NVIDIA GPU with compute capability 6.0 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended)
- **Storage**: At least 10GB of free disk space
- **Network**: Stable internet connection with minimum 5Mbps upload/download

### Software Requirements
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 10.15+ (Ubuntu recommended)
- **Python**: 3.8 or higher
- **CUDA**: Version 11.0 or higher (for GPU support)
- **Git**: Latest version
- **Monero Wallet**: Feather Wallet or Official Monero CLI/GUI Wallet

## Basic Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/ameritusweb/compunir
cd compunir
```

### Step 2: Create and Activate Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS
source venv/bin/activate
# On Windows
.\venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install core package with development dependencies
pip install -e ".[dev]"

# Or, for production-only installation
pip install -e .
```

### Step 4: Configure Your Setup

```bash
# Copy default configuration
cp config/default_config.yml config/local_config.yml

# Edit configuration file with your settings
# (See Configuration Guide for details)
nano config/local_config.yml
```

### Step 5: Set Up Monero Wallet

1. Install a Monero wallet (Feather Wallet recommended for beginners)
2. Create a new wallet and securely store your recovery seed
3. Set up the wallet RPC if you plan to send/receive automatic payments
4. Update your `local_config.yml` with wallet address and RPC settings

## Advanced Installation

### Docker Installation

For containerized deployment, you can use Docker:

```bash
# Build the Docker image
docker build -t compunir .

# Run with mounted configuration
docker run -d \
  --name compunir-node \
  --gpus all \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  -p 50051:50051 \
  compunir
```

### Dashboard Installation

To install the dashboard for monitoring:

```bash
# Navigate to dashboard directory
cd dashboard

# Install dependencies
npm install

# Build for production
npm run build

# Or run development server
npm run dev
```

## Setting Up a Production Environment

For production deployments, we recommend the following additional steps:

### System Service

Create a systemd service for automatic startup:

```bash
# Create service file
sudo nano /etc/systemd/system/compunir.service
```

Add the following content:

```ini
[Unit]
Description=Compunir Decentralized GPU Node
After=network.target

[Service]
User=yourusername
WorkingDirectory=/path/to/compunir
ExecStart=/path/to/compunir/venv/bin/python -m decentralized_gpu.core.node_manager
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable compunir
sudo systemctl start compunir
```

### Firewall Configuration

If you're using a firewall, you'll need to open the required ports:

```bash
# For UFW (Ubuntu)
sudo ufw allow 50051/tcp  # gRPC communication
sudo ufw allow 8000/tcp   # API (if enabled)
sudo ufw allow 3000/tcp   # Dashboard (if running locally)
```

### Setting Up HTTPS

For secure dashboard access, set up HTTPS using the included scripts:

```bash
# Navigate to deployment/scripts
cd deployment/scripts

# Run setup script (includes SSL certificate generation)
./setup.sh
```

## Verification

To verify your installation is working correctly:

```bash
# Check node status
python -m decentralized_gpu.tools.status

# Run diagnostic test
python -m decentralized_gpu.tools.diagnostic
```

## Troubleshooting

### Common Issues

**GPU Not Detected**
- Ensure CUDA is properly installed: `nvidia-smi`
- Check that the NVIDIA drivers are up to date
- Verify that the `pynvml` package is installed

**Network Connection Issues**
- Check your firewall settings
- Verify the node address in your configuration file
- Ensure the bootstrap nodes are accessible

**Monero Wallet Connection**
- Check that the wallet RPC service is running
- Verify your wallet address and RPC credentials
- Ensure you have sufficient funds for staking

### Getting Help

If you encounter issues not covered here:

1. Check the logs in the `logs/` directory
2. Visit our [GitHub Issues](https://github.com/ameritusweb/compunir/issues)
3. Join our community on Discord for real-time support

## Next Steps

After installation:

1. Review the [Configuration Guide](./CONFIGURATION.md) for detailed settings
2. Learn about the [Payment System](./PAYMENT.md) for earning XMR
3. Check the [Dashboard Guide](./DASHBOARD.md) for monitoring

For quick usage examples, see the [Quick Start Guide](../QUICKSTART.md).