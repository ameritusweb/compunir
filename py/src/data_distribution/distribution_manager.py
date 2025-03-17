import asyncio
import logging
import time
from typing import Dict, List, Set, Optional, Tuple, Any
import random
import hashlib
import json
import numpy as np
from dataclasses import dataclass
import pyarrow as pa
from cryptography.fernet import Fernet
import aiohttp
import socket
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from .data_shard import DataShard
from .data_verification_proof import DataVerificationProof

@dataclass
class DistributionNode:
    """Information about a node in the distribution network"""
    node_id: str
    address: str
    capacity: float
    connection_quality: float
    last_active: float
    shard_assignments: Set[str]

@dataclass
class ShardStatus:
    """Status of a data shard in the distribution system"""
    shard_id: str
    dataset_id: str
    size_bytes: int
    assigned_nodes: List[str]
    replication_factor: int
    integrity_hash: str
    transfer_status: Dict[str, str]  # node_id -> status
    last_updated: float

class DataDistributionManager:
    """Manages data distribution across nodes in the network"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.nodes: Dict[str, DistributionNode] = {}
        self.shard_status: Dict[str, ShardStatus] = {}
        self.dataset_shards: Dict[str, List[str]] = {}  # dataset_id -> shard_ids
        self.transfer_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=config.get('transfer_threads', 4))
        self.transfer_tasks = set()
        self.shutdown_event = asyncio.Event()
        
        # Initialize encryption
        self.encryption_key = config.get('encryption_key', Fernet.generate_key())
        self.cipher = Fernet(self.encryption_key)
        
        # Network settings
        self.chunk_size = config.get('transfer_chunk_size', 1024 * 1024)  # 1MB chunks
        self.max_retry = config.get('max_transfer_retries', 5)
        
    async def start(self):
        """Start distribution manager and background tasks"""
        # Start background transfer worker
        self.transfer_tasks.add(asyncio.create_task(self._process_transfer_queue()))
        
        # Start node monitoring task
        self.transfer_tasks.add(asyncio.create_task(self._monitor_nodes()))
        
        # Start rebalancing task
        self.transfer_tasks.add(asyncio.create_task(self._periodic_rebalance()))
        
        logging.info("Data distribution manager started")
        
    async def stop(self):
        """Stop distribution manager and cleanup"""
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for tasks to complete
        if self.transfer_tasks:
            await asyncio.gather(*self.transfer_tasks, return_exceptions=True)
            
        # Shutdown executor
        self.executor.shutdown()
        
        logging.info("Data distribution manager stopped")
        
    async def register_node(self, node_id: str, node_info: Dict) -> bool:
        """Register a new node in the distribution network"""
        try:
            # Check if node already exists
            if node_id in self.nodes:
                # Update existing node
                self.nodes[node_id].address = node_info['address']
                self.nodes[node_id].capacity = node_info.get('capacity', 1.0)
                self.nodes[node_id].last_active = time.time()
                return True
            
            # Create new node
            self.nodes[node_id] = DistributionNode(
                node_id=node_id,
                address=node_info['address'],
                capacity=node_info.get('capacity', 1.0),
                connection_quality=1.0,  # Initial value
                last_active=time.time(),
                shard_assignments=set()
            )
            
            # Trigger rebalancing if necessary
            if len(self.shard_status) > 0:
                await self._queue_rebalance()
                
            return True
            
        except Exception as e:
            logging.error(f"Error registering node {node_id}: {str(e)}")
            return False
            
    async def distribute_dataset(self, 
                               dataset_id: str, 
                               shards: List[Tuple[DataShard, DataVerificationProof]],
                               replication_factor: int = 2) -> Dict:
        """Distribute dataset shards across available nodes"""
        try:
            if not self.nodes:
                raise ValueError("No nodes available for distribution")
                
            # Track dataset
            self.dataset_shards[dataset_id] = []
            
            # Process each shard
            distribution_results = {}
            for shard, proof in shards:
                # Assign nodes for this shard
                assigned_nodes = await self._assign_nodes_for_shard(
                    replication_factor=replication_factor
                )
                
                if not assigned_nodes:
                    raise ValueError("Failed to assign nodes for shard")
                
                # Calculate shard hash
                shard_hash = await self._calculate_shard_hash(shard)
                
                # Create shard status
                self.shard_status[shard.shard_id] = ShardStatus(
                    shard_id=shard.shard_id,
                    dataset_id=dataset_id,
                    size_bytes=await self._get_shard_size(shard),
                    assigned_nodes=assigned_nodes,
                    replication_factor=replication_factor,
                    integrity_hash=shard_hash,
                    transfer_status={node_id: "pending" for node_id in assigned_nodes},
                    last_updated=time.time()
                )
                
                # Update node assignments
                for node_id in assigned_nodes:
                    self.nodes[node_id].shard_assignments.add(shard.shard_id)
                
                # Queue shard transfers
                for node_id in assigned_nodes:
                    await self.transfer_queue.put({
                        'type': 'distribute',
                        'shard_id': shard.shard_id,
                        'node_id': node_id,
                        'shard': shard,
                        'proof': proof
                    })
                    
                # Add to dataset tracking
                self.dataset_shards[dataset_id].append(shard.shard_id)
                
                # Add to results
                distribution_results[shard.shard_id] = {
                    'assigned_nodes': assigned_nodes,
                    'status': 'pending'
                }
                
            return distribution_results
            
        except Exception as e:
            logging.error(f"Error distributing dataset {dataset_id}: {str(e)}")
            raise
            
    async def get_shard_locations(self, shard_id: str) -> List[str]:
        """Get list of nodes that have a copy of the shard"""
        try:
            if shard_id not in self.shard_status:
                return []
                
            # Find nodes with successful transfer
            status = self.shard_status[shard_id]
            return [
                node_id for node_id, transfer_status in status.transfer_status.items()
                if transfer_status == "completed"
            ]
            
        except Exception as e:
            logging.error(f"Error getting shard locations for {shard_id}: {str(e)}")
            return []
            
    async def retrieve_shard(self, 
                           shard_id: str, 
                           requester_node_id: Optional[str] = None) -> Optional[DataShard]:
        """Retrieve a shard from the network"""
        try:
            if shard_id not in self.shard_status:
                raise ValueError(f"Shard {shard_id} not found")
                
            # Get status
            status = self.shard_status[shard_id]
            
            # Find nodes with the shard
            available_nodes = [
                node_id for node_id, transfer_status in status.transfer_status.items()
                if transfer_status == "completed" and node_id != requester_node_id
            ]
            
            if not available_nodes:
                raise ValueError(f"No available nodes have shard {shard_id}")
                
            # Select best node to retrieve from
            source_node_id = await self._select_best_source_node(available_nodes)
            
            # Retrieve shard from selected node
            shard = await self._download_shard_from_node(shard_id, source_node_id)
            
            # Verify shard integrity
            if shard:
                shard_hash = await self._calculate_shard_hash(shard)
                if shard_hash != status.integrity_hash:
                    logging.error(f"Integrity check failed for shard {shard_id}")
                    return None
                    
            return shard
            
        except Exception as e:
            logging.error(f"Error retrieving shard {shard_id}: {str(e)}")
            return None
            
    async def redistribute_shard(self, shard_id: str, new_nodes: List[str]) -> bool:
        """Redistribute shard to a new set of nodes"""
        try:
            if shard_id not in self.shard_status:
                raise ValueError(f"Shard {shard_id} not found")
                
            # Get current status
            status = self.shard_status[shard_id]
            
            # Find source nodes
            source_nodes = [
                node_id for node_id, transfer_status in status.transfer_status.items()
                if transfer_status == "completed"
            ]
            
            if not source_nodes:
                raise ValueError(f"No source nodes available for shard {shard_id}")
                
            # Create redistribution tasks
            source_node_id = await self._select_best_source_node(source_nodes)
            
            # Retrieve shard
            shard = await self._download_shard_from_node(shard_id, source_node_id)
            if not shard:
                raise ValueError(f"Failed to retrieve shard {shard_id}")
                
            # Update status
            for node_id in new_nodes:
                if node_id not in status.assigned_nodes:
                    status.assigned_nodes.append(node_id)
                    status.transfer_status[node_id] = "pending"
                    
                    # Update node assignments
                    if node_id in self.nodes:
                        self.nodes[node_id].shard_assignments.add(shard_id)
                        
            # Queue transfers
            for node_id in new_nodes:
                await self.transfer_queue.put({
                    'type': 'redistribute',
                    'shard_id': shard_id,
                    'node_id': node_id,
                    'shard': shard
                })
                
            return True
            
        except Exception as e:
            logging.error(f"Error redistributing shard {shard_id}: {str(e)}")
            return False
            
    async def _process_transfer_queue(self):
        """Process pending transfers in queue"""
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Get next transfer task
                    transfer_task = await asyncio.wait_for(
                        self.transfer_queue.get(),
                        timeout=1.0
                    )
                    
                    # Process based on type
                    if transfer_task['type'] == 'distribute':
                        await self._transfer_shard_to_node(
                            transfer_task['shard_id'],
                            transfer_task['node_id'],
                            transfer_task['shard']
                        )
                    elif transfer_task['type'] == 'redistribute':
                        await self._transfer_shard_to_node(
                            transfer_task['shard_id'],
                            transfer_task['node_id'],
                            transfer_task['shard']
                        )
                        
                    # Mark task as done
                    self.transfer_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No tasks in queue
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logging.info("Transfer queue processor stopped")
        except Exception as e:
            logging.error(f"Error in transfer queue processor: {str(e)}")
            
    async def _transfer_shard_to_node(self, shard_id: str, node_id: str, shard: DataShard) -> bool:
        """Transfer shard to a specific node"""
        try:
            if shard_id not in self.shard_status:
                logging.error(f"Shard {shard_id} not found in status tracking")
                return False
                
            if node_id not in self.nodes:
                logging.error(f"Node {node_id} not registered")
                return False
                
            # Get node address
            node_address = self.nodes[node_id].address
            
            # Update status
            self.shard_status[shard_id].transfer_status[node_id] = "transferring"
            self.shard_status[shard_id].last_updated = time.time()
            
            # Serialize and encrypt shard
            shard_data = await self._serialize_and_encrypt_shard(shard)
            
            # Transfer to node
            success = await self._upload_shard_to_node(shard_id, node_id, node_address, shard_data)
            
            # Update status
            if success:
                self.shard_status[shard_id].transfer_status[node_id] = "completed"
            else:
                self.shard_status[shard_id].transfer_status[node_id] = "failed"
                
            self.shard_status[shard_id].last_updated = time.time()
            
            return success
            
        except Exception as e:
            logging.error(f"Error transferring shard {shard_id} to node {node_id}: {str(e)}")
            
            # Update status
            if shard_id in self.shard_status and node_id in self.shard_status[shard_id].transfer_status:
                self.shard_status[shard_id].transfer_status[node_id] = "failed"
                self.shard_status[shard_id].last_updated = time.time()
                
            return False
            
    async def _upload_shard_to_node(self, shard_id: str, node_id: str, node_address: str, shard_data: bytes) -> bool:
        """Upload shard data to node via HTTP or direct socket"""
        try:
            # Choose transfer protocol based on configuration
            protocol = self.config.get('transfer_protocol', 'http')
            
            # Calculate number of chunks
            total_chunks = (len(shard_data) + self.chunk_size - 1) // self.chunk_size
            
            # Retry loop
            for attempt in range(self.max_retry):
                try:
                    if protocol == 'http':
                        # HTTP-based transfer
                        async with aiohttp.ClientSession() as session:
                            # Initiate transfer
                            async with session.post(
                                f"{node_address}/api/transfers/init",
                                json={
                                    'shard_id': shard_id,
                                    'size': len(shard_data),
                                    'chunks': total_chunks,
                                    'hash': hashlib.sha256(shard_data).hexdigest()
                                }
                            ) as response:
                                if response.status != 200:
                                    continue  # Try again
                                    
                                transfer_id = (await response.json())['transfer_id']
                            
                            # Transfer chunks
                            for i in range(total_chunks):
                                start = i * self.chunk_size
                                end = min(start + self.chunk_size, len(shard_data))
                                chunk = shard_data[start:end]
                                
                                async with session.post(
                                    f"{node_address}/api/transfers/chunk",
                                    json={
                                        'transfer_id': transfer_id,
                                        'chunk_index': i,
                                        'data': chunk.hex()
                                    }
                                ) as response:
                                    if response.status != 200:
                                        raise RuntimeError(f"Chunk transfer failed: {response.status}")
                            
                            # Finalize transfer
                            async with session.post(
                                f"{node_address}/api/transfers/finalize",
                                json={'transfer_id': transfer_id}
                            ) as response:
                                return response.status == 200
                                
                    elif protocol == 'socket':
                        # Direct socket transfer (more efficient for large data)
                        return await self._socket_transfer(
                            node_address, shard_id, shard_data, total_chunks
                        )
                        
                    return False
                    
                except Exception as e:
                    logging.warning(f"Transfer attempt {attempt+1} failed: {str(e)}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            # All attempts failed
            return False
            
        except Exception as e:
            logging.error(f"Error uploading shard to node: {str(e)}")
            return False
            
    async def _socket_transfer(self, node_address: str, shard_id: str, shard_data: bytes, total_chunks: int) -> bool:
        """Transfer shard using direct socket connection"""
        try:
            # Parse address
            host, port_str = node_address.split(':')
            port = int(port_str)
            
            # Connect to node
            reader, writer = await asyncio.open_connection(host, port)
            
            try:
                # Send transfer init
                init_msg = {
                    'type': 'transfer_init',
                    'shard_id': shard_id,
                    'size': len(shard_data),
                    'chunks': total_chunks
                }
                writer.write(self._encode_message(init_msg))
                await writer.drain()
                
                # Read response
                response = await self._read_message(reader)
                if response.get('status') != 'ok':
                    return False
                    
                # Send chunks
                for i in range(total_chunks):
                    start = i * self.chunk_size
                    end = min(start + self.chunk_size, len(shard_data))
                    chunk = shard_data[start:end]
                    
                    chunk_msg = {
                        'type': 'chunk',
                        'index': i,
                        'data': chunk
                    }
                    writer.write(self._encode_message(chunk_msg))
                    await writer.drain()
                    
                    # Read ack
                    ack = await self._read_message(reader)
                    if ack.get('status') != 'ok':
                        return False
                        
                # Finalize
                final_msg = {'type': 'finalize'}
                writer.write(self._encode_message(final_msg))
                await writer.drain()
                
                # Read completion
                completion = await self._read_message(reader)
                return completion.get('status') == 'ok'
                
            finally:
                writer.close()
                await writer.wait_closed()
                
        except Exception as e:
            logging.error(f"Socket transfer error: {str(e)}")
            return False
            
    async def _download_shard_from_node(self, shard_id: str, node_id: str) -> Optional[DataShard]:
        """Download shard from node"""
        try:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not registered")
                
            # Get node address
            node_address = self.nodes[node_id].address
            
            # Choose protocol
            protocol = self.config.get('transfer_protocol', 'http')
            
            if protocol == 'http':
                # HTTP-based transfer
                async with aiohttp.ClientSession() as session:
                    # Request shard
                    async with session.get(
                        f"{node_address}/api/shards/{shard_id}"
                    ) as response:
                        if response.status != 200:
                            raise RuntimeError(f"Shard download failed: {response.status}")
                            
                        # Get data
                        shard_data = await response.read()
                        
            elif protocol == 'socket':
                # Socket-based transfer
                shard_data = await self._socket_download(node_address, shard_id)
                
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
                
            # Decrypt and deserialize
            if shard_data:
                return await self._decrypt_and_deserialize_shard(shard_data)
                
            return None
            
        except Exception as e:
            logging.error(f"Error downloading shard {shard_id} from node {node_id}: {str(e)}")
            return None
            
    async def _socket_download(self, node_address: str, shard_id: str) -> Optional[bytes]:
        """Download shard using direct socket connection"""
        try:
            # Parse address
            host, port_str = node_address.split(':')
            port = int(port_str)
            
            # Connect to node
            reader, writer = await asyncio.open_connection(host, port)
            
            try:
                # Send download request
                request = {
                    'type': 'download',
                    'shard_id': shard_id
                }
                writer.write(self._encode_message(request))
                await writer.drain()
                
                # Read response
                response = await self._read_message(reader)
                if response.get('status') != 'ok':
                    return None
                    
                # Get metadata
                total_chunks = response.get('chunks', 0)
                size = response.get('size', 0)
                
                # Receive chunks
                chunks = []
                for _ in range(total_chunks):
                    chunk_msg = await self._read_message(reader)
                    if chunk_msg.get('type') != 'chunk':
                        raise RuntimeError("Invalid message type")
                        
                    chunks.append(chunk_msg.get('data', b''))
                    
                    # Send ack
                    writer.write(self._encode_message({'type': 'ack', 'status': 'ok'}))
                    await writer.drain()
                    
                # Combine chunks
                return b''.join(chunks)
                
            finally:
                writer.close()
                await writer.wait_closed()
                
        except Exception as e:
            logging.error(f"Socket download error: {str(e)}")
            return None
            
    def _encode_message(self, message: Dict) -> bytes:
        """Encode message for socket transfer"""
        # Serialize message
        if 'data' in message and isinstance(message['data'], bytes):
            # Handle binary data specially
            data = message.pop('data')
            message_json = json.dumps(message).encode()
            
            # Format: [header size (4 bytes)][header][data size (4 bytes)][data]
            header_size = len(message_json)
            data_size = len(data)
            
            result = bytearray()
            result.extend(struct.pack('>I', header_size))
            result.extend(message_json)
            result.extend(struct.pack('>I', data_size))
            result.extend(data)
            
            return bytes(result)
        else:
            # Regular JSON message
            message_json = json.dumps(message).encode()
            header_size = len(message_json)
            
            result = bytearray()
            result.extend(struct.pack('>I', header_size))
            result.extend(message_json)
            result.extend(struct.pack('>I', 0))  # No data
            
            return bytes(result)
            
    async def _read_message(self, reader: asyncio.StreamReader) -> Dict:
        """Read message from socket transfer"""
        # Read header size
        header_size_bytes = await reader.readexactly(4)
        header_size = struct.unpack('>I', header_size_bytes)[0]
        
        # Read header
        header_bytes = await reader.readexactly(header_size)
        header = json.loads(header_bytes.decode())
        
        # Read data size
        data_size_bytes = await reader.readexactly(4)
        data_size = struct.unpack('>I', data_size_bytes)[0]
        
        # Read data if present
        if data_size > 0:
            data = await reader.readexactly(data_size)
            header['data'] = data
            
        return header
        
    async def _monitor_nodes(self):
        """Monitor node status and handle failures"""
        try:
            while not self.shutdown_event.is_set():
                current_time = time.time()
                inactive_timeout = self.config.get('node_inactive_timeout', 300)  # 5 minutes
                
                # Find inactive nodes
                inactive_nodes = []
                for node_id, node in list(self.nodes.items()):
                    if current_time - node.last_active > inactive_timeout:
                        inactive_nodes.append(node_id)
                        
                # Handle inactive nodes
                for node_id in inactive_nodes:
                    await self._handle_node_failure(node_id)
                    
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            logging.info("Node monitor stopped")
        except Exception as e:
            logging.error(f"Error in node monitor: {str(e)}")
            
    async def _handle_node_failure(self, node_id: str):
        """Handle node failure and rebalance shards"""
        try:
            logging.warning(f"Handling failure for node {node_id}")
            
            # Get shards assigned to this node
            affected_shards = []
            for shard_id, status in self.shard_status.items():
                if node_id in status.assigned_nodes:
                    affected_shards.append(shard_id)
                    
            # Remove node
            if node_id in self.nodes:
                del self.nodes[node_id]
                
            # Rebalance affected shards
            for shard_id in affected_shards:
                if shard_id in self.shard_status:
                    status = self.shard_status[shard_id]
                    
                    # Remove failed node
                    if node_id in status.assigned_nodes:
                        status.assigned_nodes.remove(node_id)
                        
                    if node_id in status.transfer_status:
                        del status.transfer_status[node_id]
                        
                    # Check if we need to redistribute
                    if len(status.assigned_nodes) < status.replication_factor:
                        # Find new nodes
                        new_nodes = await self._find_replacement_nodes(
                            shard_id,
                            status.replication_factor - len(status.assigned_nodes)
                        )
                        
                        if new_nodes:
                            # Trigger redistribution
                            await self.redistribute_shard(shard_id, new_nodes)
                            
        except Exception as e:
            logging.error(f"Error handling node failure: {str(e)}")
            
    async def _find_replacement_nodes(self, shard_id: str, count: int) -> List[str]:
        """Find replacement nodes for shard replication"""
        try:
            if shard_id not in self.shard_status:
                return []
                
            # Get current assignments
            status = self.shard_status[shard_id]
            current_assignments = set(status.assigned_nodes)
            
            # Find eligible nodes
            eligible_nodes = []
            for node_id, node in self.nodes.items():
                if node_id not in current_assignments:
                    # Calculate load factor
                    load_factor = len(node.shard_assignments) / max(1.0, node.capacity)
                    eligible_nodes.append((node_id, load_factor))
                    
            # Sort by load factor (ascending)
            eligible_nodes.sort(key=lambda x: x[1])
            
            # Return required number of nodes
            return [node_id for node_id, _ in eligible_nodes[:count]]
            
        except Exception as e:
            logging.error(f"Error finding replacement nodes: {str(e)}")
            return []
            
    async def _periodic_rebalance(self):
        """Periodically rebalance shard distribution"""
        try:
            while not self.shutdown_event.is_set():
                # Wait for rebalance interval
                rebalance_interval = self.config.get('rebalance_interval', 3600)  # 1 hour
                await asyncio.sleep(rebalance_interval)
                
                # Trigger rebalance
                await self._rebalance_shards()
                
        except asyncio.CancelledError:
            logging.info("Periodic rebalancer stopped")
        except Exception as e:
            logging.error(f"Error in periodic rebalancer: {str(e)}")
            
    async def _queue_rebalance(self):
        """Queue a rebalance operation"""
        # Create rebalance task
        asyncio.create_task(self._rebalance_shards())
            
    async def _rebalance_shards(self):
        """Rebalance shard distribution across nodes"""
        try:
            logging.info("Starting shard rebalancing")
            
            # Calculate node loads
            node_loads = {}
            for node_id, node in self.nodes.items():
                node_loads[node_id] = len(node.shard_assignments) / max(1.0, node.capacity)
                
            # Find imbalanced nodes
            if not node_loads:
                return
                
            avg_load = sum(node_loads.values()) / len(node_loads)
            threshold = self.config.get('rebalance_threshold', 0.2)
            
            overloaded = [node_id for node_id, load in node_loads.items() 
                        if load > avg_load * (1 + threshold)]
                        
            underloaded = [node_id for node_id, load in node_loads.items()
                        if load < avg_load * (1 - threshold)]
                        
            # Rebalance if needed
            if overloaded and underloaded:
                for over_node in overloaded:
                    # Get shards from overloaded node
                    if over_node not in self.nodes:
                        continue
                        
                    shards_to_move = list(self.nodes[over_node].shard_assignments)
                    shards_to_move.sort(key=lambda s: self.shard_status.get(s, ShardStatus(
                        shard_id=s, dataset_id="", size_bytes=0, assigned_nodes=[],
                        replication_factor=0, integrity_hash="", transfer_status={},
                        last_updated=0
                    )).size_bytes)
                    
                    # Calculate how many to move
                    move_count = max(1, int(len(shards_to_move) * threshold))
                    
                    # Move shards
                    for i in range(min(move_count, len(shards_to_move))):
                        shard_id = shards_to_move[i]
                        
                        # Find destination node
                        if not underloaded:
                            break
                            
                        dest_node = underloaded.pop(0)
                        
                        # Redistribute shard
                        await self.redistribute_shard(shard_id, [dest_node])
                        
                        # Update node assignments for tracking only
                        # (actual update happens in redistribute_shard)
                        if over_node in self.nodes and dest_node in self.nodes:
                            if shard_id in self.nodes[over_node].shard_assignments:
                                self.nodes[over_node].shard_assignments.remove(shard_id)
                                
                            self.nodes[dest_node].shard_assignments.add(shard_id)
                            
                        # Reinsert destination if still underloaded
                        updated_load = len(self.nodes[dest_node].shard_assignments) / max(1.0, self.nodes[dest_node].capacity)
                        if updated_load < avg_load:
                            underloaded.append(dest_node)
                
        except Exception as e:
            logging.error(f"Error rebalancing shards: {str(e)}")
            
    async def _assign_nodes_for_shard(self, replication_factor: int) -> List[str]:
        """Assign nodes for a new shard based on load balancing"""
        try:
            # Calculate node loads
            node_loads = {}
            for node_id, node in self.nodes.items():
                node_loads[node_id] = len(node.shard_assignments) / max(1.0, node.capacity)
                
            # Sort nodes by load (ascending)
            sorted_nodes = sorted(node_loads.items(), key=lambda x: x[1])
            
            # Select nodes
            assigned_nodes = []
            for i in range(min(replication_factor, len(sorted_nodes))):
                assigned_nodes.append(sorted_nodes[i][0])
                
            return assigned_nodes
            
        except Exception as e:
            logging.error(f"Error assigning nodes: {str(e)}")
            return []
            
    async def _select_best_source_node(self, node_ids: List[str]) -> str:
        """Select best source node for shard retrieval"""
        try:
            if not node_ids:
                raise ValueError("No nodes available")
                
            # Calculate scores
            node_scores = {}
            for node_id in node_ids:
                if node_id in self.nodes:
                    # Higher connection quality is better
                    quality_score = self.nodes[node_id].connection_quality
                    
                    # Higher capacity is better
                    capacity_score = self.nodes[node_id].capacity
                    
                    # Combine scores
                    node_scores[node_id] = quality_score * 0.7 + capacity_score * 0.3
                    
            # Find best node
            if not node_scores:
                return random.choice(node_ids)
                
            best_node = max(node_scores.items(), key=lambda x: x[1])[0]
            return best_node
            
        except Exception as e:
            logging.error(f"Error selecting source node: {str(e)}")
            # Fall back to random
            return random.choice(node_ids) if node_ids else None
            
    async def _serialize_and_encrypt_shard(self, shard: DataShard) -> bytes:
        """Serialize and encrypt shard data"""
        try:
            # Serialize shard to bytes
            serialized = await asyncio.to_thread(self._serialize_shard, shard)
            
            # Compress
            compressed = zlib.compress(serialized)
            
            # Encrypt if key available
            if self.cipher:
                return self.cipher.encrypt(compressed)
                
            return compressed
            
        except Exception as e:
            logging.error(f"Error serializing and encrypting shard: {str(e)}")
            raise
            
    async def _decrypt_and_deserialize_shard(self, data: bytes) -> Optional[DataShard]:
        """Decrypt and deserialize shard data"""
        try:
            # Decrypt if cipher available
            if self.cipher:
                decrypted = self.cipher.decrypt(data)
            else:
                decrypted = data
                
            # Decompress
            decompressed = zlib.decompress(decrypted)
            
            # Deserialize
            return await asyncio.to_thread(self._deserialize_shard, decompressed)
            
        except Exception as e:
            logging.error(f"Error decrypting and deserializing shard: {str(e)}")
            return None
            
    def _serialize_shard(self, shard: DataShard) -> bytes:
        """Serialize shard to bytes"""
        try:
            # Convert PyArrow table to bytes
            sink = pa.BufferOutputStream()
            writer = pa.RecordBatchStreamWriter(sink, shard.data.schema)
            writer.write_table(shard.data)
            writer.close()
            data_bytes = sink.getvalue().to_pybytes()
            
            # Create metadata dictionary
            metadata = {
                'shard_id': shard.shard_id,
                'index_range': shard.index_range,
                'checksum': shard.checksum
            }
            
            # Serialize metadata
            metadata_bytes = json.dumps(metadata).encode()
            
            # Create header with sizes
            metadata_size = len(metadata_bytes)
            data_size = len(data_bytes)
            
            header = struct.pack('>II', metadata_size, data_size)
            
            # Combine everything
            result = bytearray()
            result.extend(header)
            result.extend(metadata_bytes)
            result.extend(data_bytes)
            
            return bytes(result)
            
        except Exception as e:
            logging.error(f"Error serializing shard: {str(e)}")
            raise
            
    def _deserialize_shard(self, data: bytes) -> DataShard:
        """Deserialize bytes to shard"""
        try:
            # Parse header
            header_size = 8  # Two 4-byte integers
            header = struct.unpack('>II', data[:header_size])
            metadata_size, data_size = header
            
            # Extract metadata
            metadata_bytes = data[header_size:header_size + metadata_size]
            metadata = json.loads(metadata_bytes.decode())
            
            # Extract data
            data_bytes = data[header_size + metadata_size:header_size + metadata_size + data_size]
            
            # Parse Arrow table
            reader = pa.RecordBatchStreamReader(pa.py_buffer(data_bytes))
            table = reader.read_all()
            
            # Create shard
            return DataShard(
                shard_id=metadata['shard_id'],
                data=table,
                index_range=tuple(metadata['index_range']),
                checksum=metadata['checksum'],
                encryption_key=None  # Don't expose encryption key
            )
            
        except Exception as e:
            logging.error(f"Error deserializing shard: {str(e)}")
            raise
            
    async def _calculate_shard_hash(self, shard: DataShard) -> str:
        """Calculate hash of shard data for integrity checking"""
        try:
            # Get serialized data (without encryption)
            serialized = await asyncio.to_thread(self._serialize_shard, shard)
            
            # Calculate hash
            return hashlib.sha256(serialized).hexdigest()
            
        except Exception as e:
            logging.error(f"Error calculating shard hash: {str(e)}")
            raise
            
    async def _get_shard_size(self, shard: DataShard) -> int:
        """Get size of serialized shard in bytes"""
        try:
            # This is approximate since we don't fully serialize/compress
            return sum(col.nbytes for col in shard.data.columns)
            
        except Exception as e:
            logging.error(f"Error getting shard size: {str(e)}")
            return 0