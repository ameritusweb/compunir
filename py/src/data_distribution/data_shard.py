from typing import Tuple, Optional
import pyarrow as pa

class DataShard:
    """
    Represents a shard of a distributed dataset with associated metadata.
    """
    def __init__(self, 
                 shard_id: str,
                 data: pa.Table, 
                 index_range: Tuple[int, int],
                 checksum: str,
                 encryption_key: Optional[bytes] = None):
        """
        Initialize a data shard.
        
        Args:
            shard_id: Unique identifier for the shard
            data: PyArrow table containing the actual data
            index_range: Tuple of (start_idx, end_idx) indicating the range of 
                         data in the original dataset
            checksum: SHA-256 hash of the serialized data for integrity verification
            encryption_key: Optional encryption key for sensitive data
        """
        self.shard_id = shard_id
        self.data = data
        self.index_range = index_range
        self.checksum = checksum
        self.encryption_key = encryption_key
    
    def get_size(self) -> int:
        """
        Get the size of the shard in bytes (approximate).
        """
        return sum(col.nbytes for col in self.data.columns)
        
    def get_num_rows(self) -> int:
        """
        Get the number of rows in the shard.
        """
        return self.data.num_rows
        
    def get_schema(self) -> pa.Schema:
        """
        Get the schema of the shard.
        """
        return self.data.schema