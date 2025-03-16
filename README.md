# compunir
GPUs unite using secure and private crypto transactions to distribute compute to decentralized nodes.

![Logo](https://raw.githubusercontent.com/ameritusweb/compunir/main/favlogo.png)

# Decentralized GPU Training Network

A peer-to-peer network for distributed machine learning training with built-in verification and cryptocurrency payment systems.

## Features

- Distributed GPU resource sharing
- Verification system for training integrity
- Monero-based payment processing
- Advanced node selection algorithms
- Real-time GPU monitoring
- Comprehensive error handling

## Installation

```bash
# Clone the repository
git clone https://github.com/ameritusweb/compunir
cd compunir

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install development version
pip install -e ".[dev]"
```

## Configuration

Copy the default configuration and modify as needed:

```bash
cp config/default_config.yaml config/local_config.yaml
```

Edit `config/local_config.yaml` with your settings.

## Usage

### Starting a Node

```python
from decentralized_gpu.core import NodeManager
from decentralized_gpu.network import NetworkClient

async def main():
    config = load_config("config/local_config.yaml")
    node = NodeManager(config)
    await node.start()

if __name__ == "__main__":
    asyncio.run(main())
```

### Submitting Training Jobs

```python
from decentralized_gpu.core import JobExecutor

async def submit_job(model, dataset):
    executor = JobExecutor(config)
    job_id = await executor.submit_job(model, dataset)
    return job_id
```

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

- [Installation Guide](docs/installation.md)
- [Configuration Guide](docs/configuration.md)
- [Verification System](docs/verification.md)
- [Payment System](docs/payment.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
