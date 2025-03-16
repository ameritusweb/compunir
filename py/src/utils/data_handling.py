import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
import pyarrow as pa
import time
import hashlib
from dataclasses import dataclass

from ..data_distribution import DataDistributionManager
from ..verification import DataVerificationProof

@dataclass
class DistributedDataContext:
    """Context for distributed data operation"""
    dataset_id: str
    shards: List[Tuple]
    distribution_status: Dict
    verification_proofs: List[Any]
    metadata: Dict

class DistributedDataHandler:
    """Handle data with distribution system integration"""
    
    def __init__(self, config: Dict, distribution_manager: DataDistributionManager):
        self.config = config
        self.distribution_manager = distribution_manager
        self.dataset_contexts = {}
        
    async def process_and_distribute(self, 
                                   data_table: pa.Table, 
                                   dataset_id: str,
                                   num_shards: int,
                                   verification_data: Optional[Dict] = None) -> DistributedDataContext:
        """Process and distribute a dataset to the network"""
        try:
            logging.info(f"Processing and distributing dataset: {dataset_id}")
            
            # 1. Create shards
            shards = await self._create_data_shards(data_table, dataset_id, num_shards)
            
            # 2. Create verification proofs
            proofs = []
            if verification_data:
                for shard, _ in shards:
                    proof = await self._create_verification_proof(
                        shard, dataset_id, verification_data
                    )
                    proofs.append(proof)
            
            # 3. Distribute shards
            replication_factor = self.config.get("distribution", {}).get(
                "default_replication_factor", 2
            )
            
            verified_shards = [(shard, proof) for shard, _, proof in zip(
                [s[0] for s in shards], proofs
            )]
            
            distribution_result = await self.distribution_manager.distribute_dataset(
                dataset_id, verified_shards, replication_factor
            )
            
            # 4. Create context
            context = DistributedDataContext(
                dataset_id=dataset_id,
                shards=shards,
                distribution_status=distribution_result,
                verification_proofs=proofs,
                metadata={
                    'created_at': time.time(),
                    'num_shards': num_shards,
                    'total_rows': len(data_table),
                    'distribution_config': {
                        'replication_factor': replication_factor
                    }
                }
            )
            
            # Store context
            self.dataset_contexts[dataset_id] = context
            
            return context
            
        except Exception as e:
            logging.error(f"Error processing and distributing dataset {dataset_id}: {str(e)}")
            raise
            
    async def retrieve_dataset(self, dataset_id: str) -> Optional[pa.Table]:
        """Retrieve a distributed dataset from the network"""
        try:
            logging.info(f"Retrieving distributed dataset: {dataset_id}")
            
            # 1. Get dataset shards
            shard_ids = self.distribution_manager.dataset_shards.get(dataset_id, [])
            if not shard_ids:
                logging.warning(f"No shards found for dataset {dataset_id}")
                return None
                
            # 2. Retrieve shards
            retrieved_shards = []
            for shard_id in shard_ids:
                shard = await self.distribution_manager.retrieve_shard(shard_id)
                if shard:
                    retrieved_shards.append(shard)
                else:
                    logging.warning(f"Failed to retrieve shard {shard_id}")
            
            if not retrieved_shards:
                logging.error(f"Failed to retrieve any shards for dataset {dataset_id}")
                return None
                
            # 3. Sort shards by index range
            retrieved_shards.sort(key=lambda s: s.index_range[0])
            
            # 4. Combine into single table
            combined_data = pa.concat_tables([s.data for s in retrieved_shards])
            
            return combined_data
            
        except Exception as e:
            logging.error(f"Error retrieving dataset {dataset_id}: {str(e)}")
            return None
            
    async def fetch_distributed_metadata(self, dataset_id: str) -> Optional[Dict]:
        """Fetch metadata for distributed dataset"""
        try:
            # Check local cache first
            if dataset_id in self.dataset_contexts:
                return self.dataset_contexts[dataset_id].metadata
                
            # Try to reconstruct metadata from distribution manager
            if dataset_id in self.distribution_manager.dataset_shards:
                shard_ids = self.distribution_manager.dataset_shards[dataset_id]
                shard_statuses = {
                    shard_id: self.distribution_manager.shard_status.get(shard_id)
                    for shard_id in shard_ids
                }
                
                if not any(shard_statuses.values()):
                    return None
                    
                # Build metadata from available information
                total_size = sum(
                    status.size_bytes for status in shard_statuses.values() 
                    if status is not None
                )
                
                return {
                    'dataset_id': dataset_id,
                    'num_shards': len(shard_ids),
                    'total_size_bytes': total_size,
                    'distribution_time': min(
                        status.last_updated for status in shard_statuses.values()
                        if status is not None
                    ),
                    'is_complete': all(
                        status is not None for status in shard_statuses.values()
                    )
                }
                
            return None
            
        except Exception as e:
            logging.error(f"Error fetching metadata for dataset {dataset_id}: {str(e)}")
            return None
    
    async def _create_data_shards(self, data_table: pa.Table, dataset_id: str, num_shards: int) -> List[Tuple]:
        """Create shards from a data table"""
        try:
            # Calculate shard size
            total_rows = len(data_table)
            shard_size = total_rows // num_shards
            remaining = total_rows % num_shards
            
            shards = []
            start_idx = 0
            
            for i in range(num_shards):
                # Calculate shard range
                shard_rows = shard_size + (1 if i < remaining else 0)
                end_idx = start_idx + shard_rows
                
                # Create shard
                shard_data = data_table.slice(start_idx, shard_rows)
                shard_id = f"{dataset_id}_shard_{i}"
                
                # Calculate checksum
                checksum = self._calculate_data_checksum(shard_data)
                
                # Create shard record
                from ..data_distribution import DataShard
                shard = DataShard(
                    shard_id=shard_id,
                    data=shard_data,
                    index_range=(start_idx, end_idx),
                    checksum=checksum,
                    encryption_key=None  # Will be set by distribution system if needed
                )
                
                shards.append((shard, checksum))
                start_idx = end_idx
                
            return shards
            
        except Exception as e:
            logging.error(f"Error creating data shards: {str(e)}")
            raise
            
    def _calculate_data_checksum(self, data: pa.Table) -> str:
        """Calculate checksum for data"""
        try:
            # Serialize table to bytes
            sink = pa.BufferOutputStream()
            writer = pa.RecordBatchStreamWriter(sink, data.schema)
            writer.write_table(data)
            writer.close()
            serialized = sink.getvalue().to_pybytes()
            
            # Calculate SHA-256 hash
            return hashlib.sha256(serialized).hexdigest()
            
        except Exception as e:
            logging.error(f"Error calculating data checksum: {str(e)}")
            raise
            
    async def _create_verification_proof(self, shard, dataset_id: str, verification_data: Dict) -> DataVerificationProof:
        """Create verification proof for a shard"""
        try:
            from ..verification import DataVerificationProof
            
            # Create verification proof
            proof = DataVerificationProof(
                data_id=shard.shard_id,
                source_hash=verification_data.get("source_hash", ""),
                processing_steps=verification_data.get("processing_steps", []),
                result_hash=shard.checksum,
                metadata={
                    "dataset_id": dataset_id,
                    "shard_id": shard.shard_id,
                    "index_range": shard.index_range,
                    "creation_time": time.time()
                },
                timestamp=time.time()
            )
            
            return proof
            
        except Exception as e:
            logging.error(f"Error creating verification proof: {str(e)}")
            raise