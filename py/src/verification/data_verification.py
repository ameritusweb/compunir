import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import time
import hashlib
from dataclasses import dataclass
import json

from ..data_distribution import DataDistributionManager, DataShard

@dataclass
class DataVerificationResult:
    """Result of data verification process"""
    is_valid: bool
    data_id: str
    verification_time: float
    details: Dict
    warnings: List[str]
    errors: List[str]

class DataVerificationSystem:
    """System for verifying distributed data integrity"""
    
    def __init__(self, config: Dict, verification_system):
        self.config = config
        self.verification_system = verification_system
        self.verification_history = {}
        
    async def verify_data_integrity(self, 
                                  shard: DataShard, 
                                  proof: Any) -> DataVerificationResult:
        """Verify integrity of data shard"""
        try:
            start_time = time.time()
            
            # 1. Verify basic integrity
            checksum_valid = await self._verify_checksum(shard)
            
            # 2. Verify against provided proof
            proof_valid = await self._verify_against_proof(shard, proof)
            
            # 3. Execute verification-specific logic
            advanced_verification = await self._perform_advanced_verification(shard, proof)
            
            # Combine results
            is_valid = checksum_valid and proof_valid and advanced_verification.get('is_valid', False)
            
            # Create result
            result = DataVerificationResult(
                is_valid=is_valid,
                data_id=shard.shard_id,
                verification_time=time.time() - start_time,
                details={
                    'checksum_valid': checksum_valid,
                    'proof_valid': proof_valid,
                    'advanced_verification': advanced_verification
                },
                warnings=[],
                errors=[]
            )
            
            # Collect warnings and errors
            if not checksum_valid:
                result.errors.append("Checksum validation failed")
                
            if not proof_valid:
                result.errors.append("Proof validation failed")
                
            if advanced_verification.get('warnings'):
                result.warnings.extend(advanced_verification['warnings'])
                
            if advanced_verification.get('errors'):
                result.errors.extend(advanced_verification['errors'])
                
            # Store verification result
            self._store_verification_result(shard.shard_id, result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error verifying data integrity: {str(e)}")
            return DataVerificationResult(
                is_valid=False,
                data_id=shard.shard_id,
                verification_time=time.time() - start_time,
                details={},
                warnings=[],
                errors=[f"Verification error: {str(e)}"]
            )
            
    async def verify_dataset_consistency(self, dataset_id: str, shards: List[DataShard]) -> DataVerificationResult:
        """Verify consistency across all shards in a dataset"""
        try:
            start_time = time.time()
            
            if not shards:
                return DataVerificationResult(
                    is_valid=False,
                    data_id=dataset_id,
                    verification_time=0,
                    details={},
                    warnings=[],
                    errors=["No shards provided for verification"]
                )
                
            # 1. Verify each shard individually
            shard_results = []
            for shard in shards:
                checksum_valid = await self._verify_checksum(shard)
                shard_results.append({
                    'shard_id': shard.shard_id,
                    'checksum_valid': checksum_valid
                })
                
            # 2. Verify shard relationships
            relationships_valid = await self._verify_shard_relationships(shards)
            
            # 3. Verify dataset completeness
            completeness_valid = await self._verify_dataset_completeness(dataset_id, shards)
            
            # Combine results
            all_shard_valid = all(r['checksum_valid'] for r in shard_results)
            is_valid = all_shard_valid and relationships_valid and completeness_valid
            
            # Create result
            result = DataVerificationResult(
                is_valid=is_valid,
                data_id=dataset_id,
                verification_time=time.time() - start_time,
                details={
                    'shard_results': shard_results,
                    'relationships_valid': relationships_valid,
                    'completeness_valid': completeness_valid,
                    'total_shards': len(shards)
                },
                warnings=[],
                errors=[]
            )
            
            # Collect errors
            if not all_shard_valid:
                invalid_shards = [r['shard_id'] for r in shard_results if not r['checksum_valid']]
                result.errors.append(f"Invalid checksums in shards: {', '.join(invalid_shards)}")
                
            if not relationships_valid:
                result.errors.append("Shard relationships verification failed")
                
            if not completeness_valid:
                result.errors.append("Dataset completeness verification failed")
                
            # Store verification result
            self._store_verification_result(dataset_id, result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error verifying dataset consistency: {str(e)}")
            return DataVerificationResult(
                is_valid=False,
                data_id=dataset_id,
                verification_time=time.time() - start_time,
                details={},
                warnings=[],
                errors=[f"Verification error: {str(e)}"]
            )
            
    async def verify_distribution_integrity(self, 
                                         distribution_manager: DataDistributionManager,
                                         dataset_id: str) -> DataVerificationResult:
        """Verify integrity of distributed dataset across nodes"""
        try:
            start_time = time.time()
            
            # 1. Get all shards for dataset
            shard_ids = distribution_manager.dataset_shards.get(dataset_id, [])
            if not shard_ids:
                return DataVerificationResult(
                    is_valid=False,
                    data_id=dataset_id,
                    verification_time=time.time() - start_time,
                    details={},
                    warnings=[],
                    errors=["No shards found for dataset"]
                )
                
            # 2. Verify distribution status
            distribution_status = {}
            for shard_id in shard_ids:
                status = distribution_manager.shard_status.get(shard_id)
                if not status:
                    distribution_status[shard_id] = {'found': False}
                    continue
                    
                # Count successful replications
                successful_replications = sum(
                    1 for _, transfer_status in status.transfer_status.items()
                    if transfer_status == "completed"
                )
                
                distribution_status[shard_id] = {
                    'found': True,
                    'replication_factor': status.replication_factor,
                    'successful_replications': successful_replications,
                    'is_fully_replicated': successful_replications >= status.replication_factor
                }
                
            # 3. Verify distribution completeness
            all_shards_found = all(status['found'] for status in distribution_status.values())
            all_shards_replicated = all(
                status.get('is_fully_replicated', False) 
                for status in distribution_status.values() 
                if status['found']
            )
            
            # 4. Sample verification of shard integrity
            sample_integrity = await self._verify_shard_sample_integrity(
                distribution_manager, shard_ids
            )
            
            # Combine results
            is_valid = all_shards_found and all_shards_replicated and sample_integrity['is_valid']
            
            # Create result
            result = DataVerificationResult(
                is_valid=is_valid,
                data_id=dataset_id,
                verification_time=time.time() - start_time,
                details={
                    'distribution_status': distribution_status,
                    'all_shards_found': all_shards_found,
                    'all_shards_replicated': all_shards_replicated,
                    'sample_integrity': sample_integrity
                },
                warnings=[],
                errors=[]
            )
            
            # Collect errors and warnings
            if not all_shards_found:
                missing_shards = [
                    shard_id for shard_id, status in distribution_status.items()
                    if not status['found']
                ]
                result.errors.append(f"Missing shards: {', '.join(missing_shards)}")
                
            if not all_shards_replicated:
                under_replicated = [
                    shard_id for shard_id, status in distribution_status.items()
                    if status['found'] and not status.get('is_fully_replicated', False)
                ]
                result.warnings.append(f"Under-replicated shards: {', '.join(under_replicated)}")
                
            if not sample_integrity['is_valid']:
                result.errors.append(f"Sample integrity check failed: {sample_integrity.get('error', '')}")
                
            # Store verification result
            self._store_verification_result(f"distribution_{dataset_id}", result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error verifying distribution integrity: {str(e)}")
            return DataVerificationResult(
                is_valid=False,
                data_id=dataset_id,
                verification_time=time.time() - start_time,
                details={},
                warnings=[],
                errors=[f"Verification error: {str(e)}"]
            )
    
    async def _verify_checksum(self, shard: DataShard) -> bool:
        """Verify shard data matches its checksum"""
        try:
            calculated_checksum = await self._calculate_shard_checksum(shard)
            return calculated_checksum == shard.checksum
            
        except Exception as e:
            logging.error(f"Error verifying checksum: {str(e)}")
            return False
            
    async def _verify_against_proof(self, shard: DataShard, proof: Any) -> bool:
        """Verify shard against provided proof"""
        try:
            # Basic validation
            if not proof:
                return False
                
            # Check if proof is for this shard
            if proof.data_id != shard.shard_id:
                return False
                
            # Check if result hash matches
            calculated_hash = await self._calculate_shard_checksum(shard)
            return calculated_hash == proof.result_hash
            
        except Exception as e:
            logging.error(f"Error verifying against proof: {str(e)}")
            return False
            
    async def _perform_advanced_verification(self, shard: DataShard, proof: Any) -> Dict:
        """Perform advanced verification checks"""
        try:
            # Use the main verification system for this
            result = await self.verification_system.verify_data_flow(proof)
            
            return {
                'is_valid': result.get('is_valid', False),
                'details': result.get('details', {}),
                'warnings': result.get('warnings', []),
                'errors': result.get('errors', [])
            }
            
        except Exception as e:
            logging.error(f"Error performing advanced verification: {str(e)}")
            return {
                'is_valid': False,
                'errors': [str(e)]
            }
            
    async def _verify_shard_relationships(self, shards: List[DataShard]) -> bool:
        """Verify relationships between shards"""
        try:
            if not shards:
                return False
                
            # Sort shards by index range
            sorted_shards = sorted(shards, key=lambda s: s.index_range[0])
            
            # Check for continuity
            for i in range(len(sorted_shards) - 1):
                current = sorted_shards[i]
                next_shard = sorted_shards[i + 1]
                
                # Check if ranges are contiguous
                if current.index_range[1] != next_shard.index_range[0]:
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error verifying shard relationships: {str(e)}")
            return False
            
    async def _verify_dataset_completeness(self, dataset_id: str, shards: List[DataShard]) -> bool:
        """Verify dataset completeness"""
        try:
            if not shards:
                return False
                
            # Check if all shards belong to the same dataset
            shard_prefixes = set()
            for shard in shards:
                # Expect shard_id format: "dataset_id_shard_X"
                parts = shard.shard_id.split('_shard_')
                if len(parts) != 2:
                    return False
                    
                shard_prefixes.add(parts[0])
                
            # All shards should have the same prefix
            if len(shard_prefixes) != 1:
                return False
                
            # Check if prefix matches dataset_id
            prefix = next(iter(shard_prefixes))
            if prefix != dataset_id:
                return False
                
            # Sort shards by index
            sorted_shards = sorted(shards, key=lambda s: s.index_range[0])
            
            # Check if first shard starts at 0
            if sorted_shards[0].index_range[0] != 0:
                return False
                
            # Check if ranges are contiguous and cover the entire dataset
            for i in range(len(sorted_shards) - 1):
                if sorted_shards[i].index_range[1] != sorted_shards[i + 1].index_range[0]:
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error verifying dataset completeness: {str(e)}")
            return False
            
    async def _verify_shard_sample_integrity(self, 
                                           distribution_manager: DataDistributionManager,
                                           shard_ids: List[str]) -> Dict:
        """Verify integrity of a sample of shards"""
        try:
            if not shard_ids:
                return {'is_valid': False, 'error': 'No shards to sample'}
                
            # Determine sample size
            sample_count = min(3, len(shard_ids))  # Sample up to 3 shards
            
            # Randomly select shards
            import random
            sample_ids = random.sample(shard_ids, sample_count)
            
            # Retrieve and verify sample shards
            sample_results = []
            for shard_id in sample_ids:
                # Get shard
                shard = await distribution_manager.retrieve_shard(shard_id)
                if not shard:
                    sample_results.append({
                        'shard_id': shard_id,
                        'is_valid': False,
                        'error': 'Failed to retrieve shard'
                    })
                    continue
                    
                # Verify checksum
                checksum_valid = await self._verify_checksum(shard)
                sample_results.append({
                    'shard_id': shard_id,
                    'is_valid': checksum_valid,
                    'error': None if checksum_valid else 'Checksum validation failed'
                })
                
            # Overall result
            is_valid = all(result['is_valid'] for result in sample_results)
            
            return {
                'is_valid': is_valid,
                'sample_size': sample_count,
                'results': sample_results
            }
            
        except Exception as e:
            logging.error(f"Error verifying shard sample integrity: {str(e)}")
            return {
                'is_valid': False,
                'error': str(e)
            }
    
    async def _calculate_shard_checksum(self, shard: DataShard) -> str:
        """Calculate checksum for shard data"""
        try:
            import pyarrow as pa
            
            # Serialize table to bytes
            sink = pa.BufferOutputStream()
            writer = pa.RecordBatchStreamWriter(sink, shard.data.schema)
            writer.write_table(shard.data)
            writer.close()
            serialized = sink.getvalue().to_pybytes()
            
            # Calculate SHA-256 hash
            return hashlib.sha256(serialized).hexdigest()
            
        except Exception as e:
            logging.error(f"Error calculating shard checksum: {str(e)}")
            raise
            
    def _store_verification_result(self, data_id: str, result: DataVerificationResult):
        """Store verification result for later reference"""
        if data_id not in self.verification_history:
            self.verification_history[data_id] = []
            
        self.verification_history[data_id].append({
            'timestamp': time.time(),
            'is_valid': result.is_valid,
            'verification_time': result.verification_time,
            'warnings': result.warnings,
            'errors': result.errors
        })
        
        # Limit history size
        max_history = self.config.get('max_verification_history', 100)
        if len(self.verification_history[data_id]) > max_history:
            self.verification_history[data_id] = self.verification_history[data_id][-max_history:]