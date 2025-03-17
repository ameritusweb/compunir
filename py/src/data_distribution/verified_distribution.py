import logging
from .distribution_manager import DataDistributionManager
from .data_shard import DataShard
from .data_verification_proof import DataVerificationProof
from typing import Dict, List, Tuple, Optional
import time

class VerifiedDataDistribution:
    """Integrate data distribution with verification system"""
    
    def __init__(self, config: Dict, verification_system, distribution_manager: DataDistributionManager):
        self.config = config
        self.verification_system = verification_system
        self.distribution_manager = distribution_manager
        
    async def distribute_verified_dataset(self, 
                                        dataset_id: str, 
                                        shards: List[Tuple[DataShard, DataVerificationProof]],
                                        replication_factor: int = 2) -> Dict:
        """Distribute dataset with verification checks"""
        try:
            # Verify all shards before distribution
            verified_shards = []
            for shard, proof in shards:
                # Verify shard
                verification_result = await self.verification_system.verify_data_flow(proof)
                
                if verification_result['is_valid']:
                    verified_shards.append((shard, proof))
                else:
                    logging.warning(f"Skipping invalid shard {shard.shard_id}: {verification_result.get('error', 'Unknown error')}")
                    
            if not verified_shards:
                raise ValueError("No valid shards to distribute")
                
            # Distribute verified shards
            result = await self.distribution_manager.distribute_dataset(
                dataset_id=dataset_id,
                shards=verified_shards,
                replication_factor=replication_factor
            )
            
            # Add verification status
            for shard_id, info in result.items():
                info['verified'] = True
                
            return result
            
        except Exception as e:
            logging.error(f"Error distributing verified dataset {dataset_id}: {str(e)}")
            raise
            
async def retrieve_and_verify_shard(self, shard_id: str, requester_node_id: Optional[str] = None) -> Tuple[Optional[DataShard], Optional[DataVerificationProof]]:
       """Retrieve and verify shard integrity"""
       try:
           # Retrieve shard
           shard = await self.distribution_manager.retrieve_shard(shard_id, requester_node_id)
           if not shard:
               return None, None
               
           # Generate verification proof
           proof = await self._generate_retrieval_proof(shard)
           
           # Verify shard integrity
           verification_result = await self.verification_system.verify_data_flow(proof)
           
           if not verification_result['is_valid']:
               logging.warning(f"Retrieved shard failed verification: {shard_id}")
               return None, None
               
           return shard, proof
           
       except Exception as e:
           logging.error(f"Error retrieving and verifying shard {shard_id}: {str(e)}")
           return None, None
           
async def _generate_retrieval_proof(self, shard: DataShard) -> DataVerificationProof:
    """Generate verification proof for retrieved shard"""
    try:
        # Create verification proof
        proof = DataVerificationProof(
            data_id=f"retrieved_{shard.shard_id}",
            source_hash=shard.checksum,
            processing_steps=[{
                "operation": "retrieve",
                "parameters": {
                    "shard_id": shard.shard_id,
                    "retrieval_time": time.time()
                }
            }],
            result_hash=await self.distribution_manager._calculate_shard_hash(shard),
            metadata={
                "shard_id": shard.shard_id,
                "index_range": shard.index_range,
                "retrieval_timestamp": time.time()
            },
            timestamp=time.time()
        )
        
        return proof
        
    except Exception as e:
        logging.error(f"Error generating retrieval proof: {str(e)}")
        raise
        
async def monitor_shard_health(self):
    """Monitor health of distributed shards"""
    try:
        # Get all shard statuses
        all_shards = {}
        for shard_id, status in self.distribution_manager.shard_status.items():
            all_shards[shard_id] = {
                'assigned_nodes': status.assigned_nodes,
                'replication_factor': status.replication_factor,
                'actual_replications': sum(1 for _, transfer_status in status.transfer_status.items() 
                                            if transfer_status == "completed"),
                'dataset_id': status.dataset_id
            }
            
        # Find under-replicated shards
        under_replicated = []
        for shard_id, info in all_shards.items():
            if info['actual_replications'] < info['replication_factor']:
                under_replicated.append(shard_id)
                
        # Handle under-replicated shards
        for shard_id in under_replicated:
            await self._repair_shard_replication(shard_id)
            
        return {
            'total_shards': len(all_shards),
            'healthy_shards': len(all_shards) - len(under_replicated),
            'under_replicated': len(under_replicated),
            'repaired': len(under_replicated)
        }
        
    except Exception as e:
        logging.error(f"Error monitoring shard health: {str(e)}")
        return {
            'error': str(e),
            'status': 'failed'
        }
        
async def _repair_shard_replication(self, shard_id: str):
    """Repair under-replicated shard"""
    try:
        if shard_id not in self.distribution_manager.shard_status:
            return False
            
        status = self.distribution_manager.shard_status[shard_id]
        
        # Calculate how many additional replicas needed
        completed_replicas = sum(1 for _, transfer_status in status.transfer_status.items() 
                                if transfer_status == "completed")
        
        needed_replicas = status.replication_factor - completed_replicas
        if needed_replicas <= 0:
            return True
            
        # Find new nodes
        new_nodes = await self.distribution_manager._find_replacement_nodes(
            shard_id, needed_replicas
        )
        
        if not new_nodes:
            logging.warning(f"No replacement nodes available for shard {shard_id}")
            return False
            
        # Redistribute shard
        return await self.distribution_manager.redistribute_shard(shard_id, new_nodes)
        
    except Exception as e:
        logging.error(f"Error repairing shard replication for {shard_id}: {str(e)}")
        return False