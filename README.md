![Logo](https://raw.githubusercontent.com/ameritusweb/compunir/main/favlogo.png)

# Compunir: Decentralized GPU Training Network

A peer-to-peer network for distributed machine learning training with built-in verification and cryptocurrency payment systems.

## Features

- **Distributed GPU Resource Sharing**: Monetize idle GPU resources or access affordable compute
- **Verification System**: Ensure training integrity with multi-node verification
- **Monero-based Payments**: Private, secure transactions with low fees
- **Sybil Attack Protection**: Advanced protection against network manipulation
- **Data Distribution**: Efficient distribution of datasets across the network
- **Real-time Monitoring**: Comprehensive dashboard for system monitoring

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- Git
- Monero Wallet (Feather Wallet recommended)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/ameritusweb/compunir
cd compunir

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install development version
pip install -e ".[dev]"

# Create local configuration
cp config/default_config.yml config/local_config.yml
```

Edit `config/local_config.yml` with your settings. For detailed configuration options, see the [Configuration Guide](docs/CONFIGURE.md).

## Configuration

The system uses a unified configuration system that integrates multiple configuration files:

- `default_config.yml`: Default system settings (do not modify)
- `distribution_config.yml`: Data distribution settings
- `logging_config.yml`: Logging configuration
- `local_config.yml`: Your custom settings (overrides defaults)

All configurations are automatically loaded and merged when the system starts.

## Usage

### Starting a Node

Use the provided startup script:

```bash
# Make the script executable
chmod +x scripts/node_startup.py

# Run the node
./scripts/node_startup.py
```

Or run it directly with Python:

```bash
python scripts/node_startup.py
```

### Monitoring Your Node

Access the dashboard at http://localhost:3000 after starting your node.

### Submitting Training Jobs

You can submit jobs through the dashboard or programmatically:

```python
from compunir.core import JobExecutor

async def submit_job(model, dataset):
    executor = JobExecutor(config)
    job_id = await executor.submit_job(model, dataset)
    return job_id
```

## Architecture

### Core Components

- **Node Manager**: Manages the overall node operation
- **Verification System**: Ensures computational integrity
- **Data Distribution**: Manages dataset distribution and replication
- **Payment System**: Handles Monero transactions

### Key Modules

- **Sybil Protection**: Prevents network manipulation attacks
- **GPU Monitoring**: Tracks GPU resource usage
- **Job Execution**: Handles ML job execution
- **Network Interface**: Manages P2P communication

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/decentralized_gpu

# Run specific test categories
pytest tests/unit
pytest tests/integration
pytest tests/advanced
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Installation Guide](docs/INSTALL.md)
- [Configuration Guide](docs/CONFIGURE.md)
- [Dashboard Guide](docs/DASHBOARD.md)
- [Verification System](docs/VERIFICATION.md)
- [Payment System](docs/PAYMENT.md)

## Developer Resources

- **Data Distribution**: Learn about the data distribution system in [Data Distribution Guide](docs/DATA_DISTRIBUTION.md)
- **Verification**: Learn how the verification system works in [Verification System](docs/VERIFICATION.md)
- **Models**: The system includes key data models like `DataShard` and `DataVerificationProof`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.