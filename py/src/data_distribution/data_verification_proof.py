from typing import Dict, List
import time

class DataVerificationProof:
    """
    Proof that verifies the origin, processing, and integrity of a data shard.
    """
    def __init__(self,
                 data_id: str,
                 source_hash: str,
                 processing_steps: List[Dict],
                 result_hash: str,
                 metadata: Dict,
                 timestamp: float = None):
        """
        Initialize a data verification proof.
        
        Args:
            data_id: Unique identifier for the data being verified
            source_hash: Hash of the source data before processing
            processing_steps: List of processing operations applied to the data
            result_hash: Hash of the resulting data after processing
            metadata: Additional metadata for verification
            timestamp: When the proof was created (defaults to current time)
        """
        self.data_id = data_id
        self.source_hash = source_hash
        self.processing_steps = processing_steps
        self.result_hash = result_hash
        self.metadata = metadata
        self.timestamp = timestamp or time.time()