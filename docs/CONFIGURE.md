# Configuration Guide

This guide explains all configuration options for the Compunir decentralized GPU training network. Proper configuration is essential for optimal performance, security, and reliability.

## Configuration Overview

Configuration is managed through YAML files located in the `config/` directory:

- `default_config.yml`: Default settings (do not modify)
- `local_config.yml`: Your custom settings (create by copying default)

Settings in `local_config.yml` override the defaults. The configuration is organized into the following sections:

1. Network
2. Wallet
3. GPU
4. Job
5. Storage
6. Logging
7. Security
8. Performance
9. Sybil Protection

## Network Configuration

Controls how your node communicates with the network.

```yaml
network:
  address: "localhost:50051"  # Your node's address
  bootstrap_nodes:            # Initial nodes to connect to
    - "node1.example.com:50051"
    - "node2.example.com:50051"
  heartbeat_interval: 30      # Seconds between heartbeats
  reconnect_delay: 60         # Seconds to wait before reconnection attempts
  max_concurrent_connections: 100  # Maximum simultaneous connections
```

### Important Network Settings

- `address`: Your node's public address (IP or domain with port)
- `bootstrap_nodes`: List of known nodes for initial connection
- `heartbeat_interval`: Frequency of status updates (lower values increase responsiveness but consume more bandwidth)

## Wallet Configuration

Settings for Monero wallet integration.

```yaml
wallet:
  address: "your_monero_address_here"  # Your Monero wallet address
  rpc_url: "http://localhost:18082"     # Local wallet RPC
  rpc_username: "username"              # RPC authentication
  rpc_password: "password"              # RPC authentication
  min_payment: 0.01                     # Minimum XMR for payout
  base_rate: 0.001                      # Base XMR per GPU hour
```

### Important Wallet Settings

- `address`: Your Monero wallet address for receiving payments
- `rpc_url`: URL of your Monero wallet RPC service (needed for automatic payments)
- `min_payment`: Smallest amount that can be transferred (to avoid dust transactions)
- `base_rate`: Base payment rate for GPU computation time

## GPU Configuration

Controls how your GPU resources are utilized.

```yaml
gpu:
  max_temperature: 85        # Maximum GPU temperature (Celsius)
  max_power_usage: 250       # Maximum power usage (Watts)
  memory_buffer: 1024        # Memory buffer to maintain (MB)
  utilization_target: 95     # Target GPU utilization (%)
  monitoring_interval: 1     # GPU stats collection interval (seconds)
```

### Important GPU Settings

- `max_temperature`: Safety limit for GPU temperature
- `memory_buffer`: Amount of GPU memory to keep free for system operations
- `utilization_target`: Optimal GPU utilization percentage to aim for

## Job Configuration

Controls job execution preferences.

```yaml
job:
  frameworks:                # Supported ML frameworks
    - pytorch
  min_payment_rate: 0.0005   # Minimum XMR per hour
  max_job_duration: 86400    # Maximum job duration (seconds)
  preferred_job_types:       # Preferred types of jobs
    - training
    - fine-tuning
  max_concurrent_jobs: 1     # Maximum concurrent jobs
  auto_accept_jobs: true     # Automatically accept jobs matching criteria
```

### Important Job Settings

- `frameworks`: ML frameworks your node supports
- `max_job_duration`: Longest job you're willing to run (24 hours by default)
- `max_concurrent_jobs`: How many jobs to run simultaneously (limited by GPU memory)

## Storage Configuration

Controls disk usage for job data and results.

```yaml
storage:
  work_dir: "./work"         # Directory for job working files
  max_storage: 100           # Maximum storage usage (GB)
  cleanup_threshold: 90      # Storage cleanup threshold (%)
  data_retention_days: 7     # How long to keep job data
```

### Important Storage Settings

- `work_dir`: Where job data and results are stored
- `max_storage`: Maximum disk space to use
- `data_retention_days`: How long to keep job data after completion

## Logging Configuration

Controls logging behavior.

```yaml
logging:
  level: "INFO"              # Logging level
  file: "node.log"           # Log file name
  max_size: 100              # Maximum log file size (MB)
  backup_count: 5            # Number of backup log files
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Important Logging Settings

- `level`: Detail level of logs (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `max_size`: Maximum size before log rotation
- `backup_count`: Number of archived log files to keep

## Security Configuration

Controls security features.

```yaml
security:
  enable_secure_enclave: false  # Use secure enclave if available
  enable_encryption: true       # Encrypt data in transit
  allowed_ips: []               # List of allowed IP addresses (empty for any)
  require_auth: true            # Require authentication for API
  auth_token_validity: 86400    # Auth token validity period (seconds)
```

### Important Security Settings

- `enable_encryption`: Always keep this on for production
- `allowed_ips`: Restrict connections to specific IPs (for private networks)
- `require_auth`: Enforce authentication for API requests

## Performance Configuration

Tunes performance parameters.

```yaml
performance:
  max_concurrent_jobs: 1      # Maximum concurrent jobs
  batch_size_limit: 32        # Maximum batch size
  prefetch_factor: 2          # Data loading prefetch factor
  enable_amp: true            # Enable automatic mixed precision
  transfer_threads: 4         # Threads for data transfer
  max_memory_usage: 0.9       # Maximum system memory usage (0-1)
```

### Important Performance Settings

- `batch_size_limit`: Upper limit for training batch size
- `enable_amp`: Enables automatic mixed precision for faster training
- `max_memory_usage`: Fraction of system memory to use

## Sybil Protection Settings

Controls protection against Sybil attacks.

```yaml
sybil_protection:
  # Minimum requirements
  min_stake: 0.1                  # Minimum XMR stake
  base_pow_difficulty: 5          # Base PoW difficulty
  min_reputation: 0.3             # Minimum reputation to participate
  
  # Geographic protection
  max_geo_cluster_size: 5         # Maximum nodes per geographic area
  proximity_threshold_km: 100     # Distance threshold for clustering
  max_nodes_per_subnet: 3         # Maximum nodes per subnet
  
  # Network protection
  max_connections_per_minute: 100 # Rate limiting
  min_peer_entropy: 2.0           # Minimum peer distribution entropy
  
  # Reputation settings
  reputation_decay: 0.99          # Daily reputation decay factor
  max_verification_history: 1000  # Maximum verification history entries
```

### Important Sybil Protection Settings

- `min_stake`: Minimum Monero stake required to join the network
- `base_pow_difficulty`: Base difficulty for proof-of-work challenges
- `min_reputation`: Minimum reputation score to participate

## Dashboard Configuration

Controls the monitoring dashboard behavior.

```yaml
dashboard:
  gpu:
    warningTemp: 80            # Temperature warning threshold (Celsius)
    criticalTemp: 85           # Critical temperature threshold
    utilizationTarget: 95      # Target GPU utilization percentage
    memoryBuffer: 1024         # Memory buffer in MB
    metricHistory: 100         # Number of historical metrics to retain
  
  ui:
    theme: 'light'             # 'light' or 'dark'
    refreshInterval: 5000      # Dashboard refresh rate (ms)
    alertDuration: 5000        # Alert display duration (ms)
    maxTableRows: 10           # Maximum rows in tables
```

### Important Dashboard Settings

- `warningTemp`: Temperature threshold for warnings
- `refreshInterval`: How often the dashboard updates (milliseconds)

## Configuration Examples

### Minimal Configuration Example

```yaml
network:
  address: "localhost:50051"
  bootstrap_nodes:
    - "seed1.compunir.org:50051"
    - "seed2.compunir.org:50051"

wallet:
  address: "your_monero_address_here"
  
gpu:
  max_temperature: 85
  utilization_target: 95
```

### Production Configuration Example

```yaml
network:
  address: "your-public-ip:50051"
  bootstrap_nodes:
    - "seed1.compunir.org:50051"
    - "seed2.compunir.org:50051"
  heartbeat_interval: 30
  reconnect_delay: 60

wallet:
  address: "your_monero_address_here"
  rpc_url: "http://localhost:18082"
  rpc_username: "username"
  rpc_password: "password"
  min_payment: 0.01
  base_rate: 0.001

gpu:
  max_temperature: 80
  max_power_usage: 220
  memory_buffer: 1024
  utilization_target: 92

job:
  frameworks:
    - pytorch
  min_payment_rate: 0.0006
  max_job_duration: 43200
  preferred_job_types:
    - training
  max_concurrent_jobs: 1
  auto_accept_jobs: true

security:
  enable_encryption: true
  require_auth: true

sybil_protection:
  min_stake: 0.1
  base_pow_difficulty: 5

storage:
  max_storage: 50
  data_retention_days: 3

performance:
  enable_amp: true
  prefetch_factor: 3
```

### High-Performance Configuration Example

```yaml
gpu:
  max_temperature: 85
  max_power_usage: 300
  memory_buffer: 512
  utilization_target: 98

job:
  max_concurrent_jobs: 2

performance:
  batch_size_limit: 64
  prefetch_factor: 4
  enable_amp: true
  transfer_threads: 8
  max_memory_usage: 0.95
```

## Reloading Configuration

Changes to configuration require restarting the node:

```bash
# If running as a service
sudo systemctl restart compunir

# If running in terminal
# First stop with Ctrl+C, then restart:
python -m decentralized_gpu.core.node_manager
```

## Configuration Validation

Validate your configuration with the built-in tool:

```bash
python -m decentralized_gpu.tools.validate_config
```

## Environment Variables

You can override configuration values using environment variables with the prefix `COMPUNIR_`:

```bash
# Example: Override wallet address
export COMPUNIR_WALLET_ADDRESS="your_new_address"

# Example: Override maximum temperature
export COMPUNIR_GPU_MAX_TEMPERATURE=82
```

## Next Steps

After configuring your node:

1. Review the [Installation Guide](./INSTALL.md) for setup instructions
2. Learn about the [Payment System](./PAYMENT.md) for earning XMR
3. Explore the [Dashboard Setup](./DASHBOARD.md) for monitoring your node