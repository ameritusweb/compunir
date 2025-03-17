import asyncio
import os
import logging
import json
import hashlib
from typing import Dict, Optional, List, Set
import aiofiles
import shutil
import time

class ShardStorage:
    """Store and manage shard data on disk"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_dir = config.get('storage_dir', 'data/shards')
        self.max_cache_size = config.get('max_cache_size', 1024 * 1024 * 1024)  # 1GB
        self.current_cache_size = 0
        self.cached_shards = {}  # shard_id -> metadata
        self.access_counts = {}  # shard_id -> count
        
        # Create directory if needed
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'metadata'), exist_ok=True)
        
        # Initialize cache
        self._init_cache()
        
    def _init_cache(self):
        """Initialize cache from disk"""
        try:
            # Scan metadata directory
            metadata_dir = os.path.join(self.base_dir, 'metadata')
            for filename in os.listdir(metadata_dir):
                if filename.endswith('.json'):
                    shard_id = filename[:-5]  # Remove .json
                    
                    # Load metadata
                    with open(os.path.join(metadata_dir, filename), 'r') as f:
                        metadata = json.load(f)
                        
                    # Add to cache
                    self.cached_shards[shard_id] = metadata
                    self.access_counts[shard_id] = 0
                    self.current_cache_size += metadata.get('size', 0)
                    
            logging.info(f"Initialized shard cache with {len(self.cached_shards)} shards, {self.current_cache_size} bytes")
                    
        except Exception as e:
            logging.error(f"Error initializing cache: {str(e)}")
            
    async def store_shard(self, shard_id: str, data: bytes) -> bool:
        """Store shard data on disk"""
        try:
            # Calculate metadata
            size = len(data)
            hash_value = hashlib.sha256(data).hexdigest()
            timestamp = time.time()
            
            # Create metadata
            metadata = {
                'shard_id': shard_id,
                'size': size,
                'hash': hash_value,
                'stored_at': timestamp
            }
            
            # Store data
            data_path = os.path.join(self.base_dir, shard_id)
            async with aiofiles.open(data_path, 'wb') as f:
                await f.write(data)
                
            # Store metadata
            metadata_path = os.path.join(self.base_dir, 'metadata', f"{shard_id}.json")
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata))
                
            # Update cache
            self.cached_shards[shard_id] = metadata
            self.access_counts[shard_id] = 0
            self.current_cache_size += size
            
            # Check cache size
            await self._manage_cache_size()
            
            return True
            
        except Exception as e:
            logging.error(f"Error storing shard {shard_id}: {str(e)}")
            return False
            
    async def get_shard(self, shard_id: str) -> Optional[bytes]:
        """Retrieve shard data from disk"""
        try:
            # Check if shard exists
            if not await self.has_shard(shard_id):
                return None
                
            # Get data
            data_path = os.path.join(self.base_dir, shard_id)
            async with aiofiles.open(data_path, 'rb') as f:
                data = await f.read()
                
            # Update access count
            self.access_counts[shard_id] = self.access_counts.get(shard_id, 0) + 1
            
            return data
            
        except Exception as e:
            logging.error(f"Error retrieving shard {shard_id}: {str(e)}")
            return None
            
    async def has_shard(self, shard_id: str) -> bool:
        """Check if shard exists in storage"""
        # Check cache first
        if shard_id in self.cached_shards:
            return True
            
        # Check disk
        data_path = os.path.join(self.base_dir, shard_id)
        metadata_path = os.path.join(self.base_dir, 'metadata', f"{shard_id}.json")
        
        return os.path.exists(data_path) and os.path.exists(metadata_path)
        
    async def delete_shard(self, shard_id: str) -> bool:
        """Delete shard from storage"""
        try:
            # Check if shard exists
            if not await self.has_shard(shard_id):
                return False
                
            # Get size for cache management
            size = 0
            if shard_id in self.cached_shards:
                size = self.cached_shards[shard_id].get('size', 0)
                
            # Delete files
            data_path = os.path.join(self.base_dir, shard_id)
            metadata_path = os.path.join(self.base_dir, 'metadata', f"{shard_id}.json")
            
            if os.path.exists(data_path):
                os.remove(data_path)
                
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            # Update cache
            if shard_id in self.cached_shards:
                del self.cached_shards[shard_id]
                
            if shard_id in self.access_counts:
                del self.access_counts[shard_id]
                
            self.current_cache_size -= size
            
            return True
            
        except Exception as e:
            logging.error(f"Error deleting shard {shard_id}: {str(e)}")
            return False
            
    async def get_shard_size(self, shard_id: str) -> int:
        """Get size of shard in bytes"""
        try:
            # Check cache
            if shard_id in self.cached_shards:
                return self.cached_shards[shard_id].get('size', 0)
                
            # Check disk
            metadata_path = os.path.join(self.base_dir, 'metadata', f"{shard_id}.json")
            if not os.path.exists(metadata_path):
                return 0
                
            # Load metadata
            async with aiofiles.open(metadata_path, 'r') as f:
                metadata = json.loads(await f.read())
                
            return metadata.get('size', 0)
            
        except Exception as e:
            logging.error(f"Error getting shard size for {shard_id}: {str(e)}")
            return 0
            
    async def get_all_shards(self) -> List[str]:
        """Get list of all shards in storage"""
        try:
            # Combine cached shards and disk shards
            result = set(self.cached_shards.keys())
            
            # Add from disk
            metadata_dir = os.path.join(self.base_dir, 'metadata')
            for filename in os.listdir(metadata_dir):
                if filename.endswith('.json'):
                    result.add(filename[:-5])  # Remove .json
                    
            return list(result)
            
        except Exception as e:
            logging.error(f"Error getting all shards: {str(e)}")
            return []
            
    async def _manage_cache_size(self):
        """Manage cache size by removing least accessed shards"""
        try:
            if self.current_cache_size <= self.max_cache_size:
                return
                
            # Sort shards by access count
            sorted_shards = sorted(
                self.access_counts.items(),
                key=lambda x: x[1]
            )
            
            # Remove shards until under limit
            for shard_id, _ in sorted_shards:
                if self.current_cache_size <= self.max_cache_size:
                    break
                    
                if shard_id in self.cached_shards:
                    size = self.cached_shards[shard_id].get('size', 0)
                    
                    # Remove from cache only (leave on disk)
                    del self.cached_shards[shard_id]
                    del self.access_counts[shard_id]
                    self.current_cache_size -= size
                    
        except Exception as e:
            logging.error(f"Error managing cache size: {str(e)}")