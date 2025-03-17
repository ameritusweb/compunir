# Data Distribution Guide

This guide explains the data distribution system in the Compunir decentralized GPU training network. Understanding this system is important for developers who want to extend the project or optimize data handling for specific use cases.

## Overview

The data distribution system is responsible for efficiently distributing datasets across network nodes while ensuring data integrity, redundancy, and optimal utilization of network resources.

Key components include:
- Shard management and storage
- Data transfer protocols
- Distribution optimization
- Integration with verification

## Core Components

### DataDistributionManager

The central component that orchestrates data distribution across the network.

**Responsibilities:**
- Manages node registration and state tracking
- Handles dataset distribution to appropriate nodes
- Enables shard retrieval from optimal source nodes
- Redistributes shards when nodes join/leave
- Performs load balancing and shard rebalancing

**Key Methods:**
- `distribute_dataset()`: Distributes a dataset across available nodes
- `retrieve_shard()`: Retrieves a shard from the network
- `redistribute_shard()`: Moves a shard to new nodes
- `register_node()`: Registers a node with the distribution system

### ShardStorage

Manages the physical storage of data shards on each node.

**Responsibilities:**
- Stores and retrieves shards from disk
- Manages shard metadata
- Implements caching and efficient access
- Handles disk space management

**Key Methods:**
- `store_shard()`: Writes a shard to disk
- `get_shard()`: Retrieves a shard from disk
- `has_shard()`: Checks if a shard exists
- `delete_shard()`: Removes a shard from storage

### ShardTransferServer

Handles data transfer protocols between nodes.

**Responsibilities:**
- Implements efficient data transfer protocols
- Manages chunked transfers for large data
- Handles data encryption and decryption
- Verifies data integrity during transfers

**Key Methods:**
- `start()`: Starts the transfer server
- `_handle_client()`: Manages client connections
- `_handle_download()`: Processes shard download requests
- `_handle_transfer()`: Manages shard upload operations

### VerifiedDataDistribution

Integrates with the verification system to ensure data integrity.

**Responsibilities:**
- Verifies data shards before distribution
- Ensures shards are validated after transfers
- Integrates with the broader verification system

**Key Methods:**
- `distribute_verified_dataset()`: Distributes datasets with verification
- `retrieve_and_verify_shard()`: Retrieves and verifies shard integrity

## Key Data Structures

### DataShard

Represents a portion of a dataset with metadata.

```python
class DataShard:
    def __init__(self, shard_id, data, index_range, checksum, encryption_key=None):
        self.shard_id = shard_id            # Unique identifier
        self.data = data                    # PyArrow table
        self.index_range = index_range      # (start, end) indices
        self.checksum = checksum            # Data integrity hash
        self.encryption_key = encryption_key # Optional encryption
```

### DataVerificationProof

Contains verification information for a data shard.

```python
class DataVerificationProof:
    def __init__(self, data_id, source_hash, processing_steps, 
                 result_hash, metadata, timestamp=None):
        self.data_id = data_id              # Data identifier
        self.source_hash = source_hash      # Original data hash
        self.processing_steps = processing_steps # Processing history
        self.result_hash = result_hash      # Final data hash
        self.metadata = metadata            # Additional metadata
        self.timestamp = timestamp or time.time()
```

## Data Flow

### Dataset Distribution

1. Dataset is split into shards with associated verification proofs
2. `VerifiedDataDistribution` verifies each shard before distribution
3. `DataDistributionManager` assigns nodes for each shard based on capacity and network quality
4. Shards are transferred to assigned nodes using `ShardTransferServer`
5. Each node stores received shards using `ShardStorage`
6. Replication ensures multiple copies exist across the network

### Shard Retrieval

1. Node requests a shard by ID from `DataDistributionManager`
2. Manager identifies optimal source nodes that have the shard
3. Shard is retrieved from the best source node
4. Integrity is verified against the stored checksum
5. Shard is returned to the requesting node

## How Load Balancing Works

The distribution system uses several strategies for load balancing:

1. **Initial Assignment**: When distributing datasets, nodes are assigned based on:
   - Current node capacity
   - Existing load (number of shards)
   - Network quality
   - Geographic distribution

2. **Periodic Rebalancing**: The system periodically redistributes shards to:
   - Balance load across nodes
   - Compensate for network changes
   - Optimize access patterns

3. **Failure Handling**: When nodes fail or disconnect:
   - Affected shards are identified
   - Replacement nodes are selected
   - Shards are redistributed to maintain the replication factor

The load balancing algorithm aims to keep the ratio of shards to capacity relatively uniform across nodes while considering network topology and access patterns.

## Performance Monitoring

The `DistributionPerformanceMonitor` tracks:

- Node utilization and capacity
- Transfer rates and latency
- Replication health
- Network efficiency

This data helps optimize distribution parameters and identify bottlenecks.

## Integration with Other Components

### Verification System Integration

The distribution system integrates with verification through:
- `VerifiedDataDistribution` which connects distribution and verification
- Verification of data integrity before and after transfers
- Integration with Sybil protection for node selection

### Node Manager Integration

The distribution manager is started and managed by the node manager:
- The node manager initializes the distribution system
- It passes node registration/deregistration events
- It coordinates job execution with data availability

## Configuration

The distribution system is configured through `distribution_config.yml` or the distribution section in `local_config.yml`.

Key configuration options:

```yaml
distribution:
  # Network settings
  network:
    server_host: "0.0.0.0"
    server_port: 7000
    transfer_protocol: "socket"  # "http" or "socket"
    transfer_chunk_size: 1048576  # 1MB
  
  # Storage settings
  storage:
    storage_dir: "data/shards"
    max_cache_size: 1073741824  # 1GB
  
  # Distribution settings
  default_replication_factor: 2
  rebalance_interval: 3600  # 1 hour
```

## Common Operations

### Creating and Distributing a Dataset

```python
import pyarrow as pa
from compunir.data_distribution.models import create_data_shards

# Create dataset
table = pa.Table.from_pandas(your_dataframe)

# Split into shards
dataset_id = "my_dataset_001"
shards = create_data_shards(table, dataset_id)

# Distribute dataset
result = await verified_distribution.distribute_verified_dataset(
    dataset_id=dataset_id,
    shards=shards,
    replication_factor=3
)
```

### Retrieving a Shard

```python
# Get distribution manager instance
distribution_manager = app.distribution_manager

# Retrieve shard
shard_id = "my_dataset_001_shard_0"
shard = await distribution_manager.retrieve_shard(shard_id)

# Access data
if shard:
    df = shard.data.to_pandas()
```

### Monitoring Distribution Performance

```python
# Get performance monitor instance
performance_monitor = app.distribution_monitor

# Collect metrics
metrics = await performance_monitor.collect_metrics()

# Access specific metrics
node_stats = metrics['node_metrics']
distribution_stats = metrics['distribution_metrics']
```

## Troubleshooting

Common issues and solutions:

1. **Transfer failures**: Check network connectivity and firewall settings. Ensure transfer_chunk_size isn't too large for your network.

2. **Node not receiving shards**: Verify node registration was successful and the node has enough storage space.

3. **Slow distribution**: Consider adjusting chunk size, increasing max_concurrent_transfers, or switching transfer protocol.

4. **Missing shards**: Check replication factor and verify all nodes are online. Use the distribution monitor to check replication health.

5. **High disk usage**: Adjust cleanup_threshold or increase max_storage to prevent disk space issues.

## Next Steps

For more advanced topics like customizing the distribution algorithm or implementing your own transfer protocols, see the source code documentation.

If you're interested in optimizing data distribution for specific workloads, look at:
- Tuning replication factors for your use case
- Customizing node selection weights
- Implementing domain-specific data compression