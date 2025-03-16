import asyncio
import logging
import json
import struct
import hashlib
import zlib
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet

class ShardTransferServer:
    """Server for handling shard data transfers between nodes"""
    
    def __init__(self, config: Dict, shard_storage):
        self.config = config
        self.shard_storage = shard_storage  # Interface to store/retrieve shards
        self.server = None
        self.clients = set()
        self.transfers = {}
        self.shutdown_event = asyncio.Event()
        
        # Initialize encryption
        self.encryption_key = config.get('encryption_key')
        self.cipher = Fernet(self.encryption_key) if self.encryption_key else None
        
        # Set transfer parameters
        self.max_concurrent_transfers = config.get('max_concurrent_transfers', 10)
        self.chunk_size = config.get('transfer_chunk_size', 1024 * 1024)  # 1MB
        
    async def start(self, host: str, port: int):
        """Start the transfer server"""
        try:
            # Start server
            self.server = await asyncio.start_server(
                self._handle_client, host, port
            )
            
            # Log start
            addr = self.server.sockets[0].getsockname()
            logging.info(f"Transfer server started on {addr}")
            
            # Start serving
            asyncio.create_task(self._serve())
            
        except Exception as e:
            logging.error(f"Error starting transfer server: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the transfer server"""
        self.shutdown_event.set()
        
        # Close client connections
        for client in self.clients:
            client.close()
            
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        logging.info("Transfer server stopped")
        
    async def _serve(self):
        """Serve until shutdown"""
        try:
            async with self.server:
                await self.shutdown_event.wait()
        except Exception as e:
            logging.error(f"Server error: {str(e)}")
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client connection"""
        self.clients.add(writer)
        client_addr = writer.get_extra_info('peername')
        
        try:
            logging.info(f"New client connection from {client_addr}")
            
            while not self.shutdown_event.is_set():
                try:
                    # Read message
                    message = await self._read_message(reader)
                    if not message:
                        break
                        
                    # Process message
                    response = await self._process_message(message, client_addr)
                    
                    # Send response
                    writer.write(self._encode_message(response))
                    await writer.drain()
                    
                    # Handle command-specific actions
                    if message.get('type') == 'download':
                        await self._handle_download(message, reader, writer)
                    elif message.get('type') == 'transfer_init':
                        await self._handle_transfer(message, reader, writer)
                        
                except asyncio.IncompleteReadError:
                    # Client disconnected
                    break
                except Exception as e:
                    logging.error(f"Error handling client {client_addr}: {str(e)}")
                    response = {'status': 'error', 'message': str(e)}
                    writer.write(self._encode_message(response))
                    await writer.drain()
                    
        except Exception as e:
            logging.error(f"Client handler error for {client_addr}: {str(e)}")
        finally:
            # Clean up
            writer.close()
            try:
                await writer.wait_closed()
            except:
                pass
            self.clients.remove(writer)
            logging.info(f"Client {client_addr} disconnected")
            
    async def _process_message(self, message: Dict, client_addr) -> Dict:
        """Process incoming message and generate response"""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'transfer_init':
                # Initialize new transfer
                shard_id = message.get('shard_id')
                size = message.get('size', 0)
                chunks = message.get('chunks', 0)
                
                # Create transfer state
                transfer_id = hashlib.sha256(f"{shard_id}_{time.time()}".encode()).hexdigest()[:16]
                self.transfers[transfer_id] = {
                    'shard_id': shard_id,
                    'size': size,
                    'chunks': chunks,
                    'received_chunks': 0,
                    'data': bytearray(size),
                    'client': client_addr,
                    'start_time': time.time()
                }
                
                return {
                    'status': 'ok',
                    'transfer_id': transfer_id
                }
                
            elif msg_type == 'chunk':
                # Process chunk
                transfer_id = message.get('transfer_id')
                chunk_index = message.get('chunk_index', 0)
                chunk_data = message.get('data')
                
                if transfer_id not in self.transfers:
                    return {'status': 'error', 'message': 'Invalid transfer ID'}
                    
                transfer = self.transfers[transfer_id]
                
                # Convert hex to bytes if needed
                if isinstance(chunk_data, str):
                    chunk_data = bytes.fromhex(chunk_data)
                
                # Calculate chunk position
                start = chunk_index * self.chunk_size
                end = min(start + len(chunk_data), transfer['size'])
                
                # Store chunk
                transfer['data'][start:end] = chunk_data
                transfer['received_chunks'] += 1
                
                return {'status': 'ok'}
                
            elif msg_type == 'finalize':
                # Finalize transfer
                transfer_id = message.get('transfer_id')
                
                if transfer_id not in self.transfers:
                    return {'status': 'error', 'message': 'Invalid transfer ID'}
                    
                transfer = self.transfers[transfer_id]
                
                # Check if all chunks received
                if transfer['received_chunks'] != transfer['chunks']:
                    return {
                        'status': 'error', 
                        'message': f"Missing chunks: {transfer['chunks'] - transfer['received_chunks']}"
                    }
                    
                # Store shard data
                shard_data = bytes(transfer['data'])
                await self.shard_storage.store_shard(transfer['shard_id'], shard_data)
                
                # Calculate transfer stats
                transfer_time = time.time() - transfer['start_time']
                transfer_rate = transfer['size'] / transfer_time / 1024 / 1024  # MB/s
                
                # Cleanup
                del self.transfers[transfer_id]
                
                return {
                    'status': 'ok',
                    'transfer_time': transfer_time,
                    'transfer_rate': transfer_rate
                }
                
            elif msg_type == 'download':
                # Request to download shard
                shard_id = message.get('shard_id')
                
                # Check if shard exists
                if not await self.shard_storage.has_shard(shard_id):
                    return {'status': 'error', 'message': 'Shard not found'}
                    
                # Get shard size
                size = await self.shard_storage.get_shard_size(shard_id)
                chunks = (size + self.chunk_size - 1) // self.chunk_size
                
                return {
                    'status': 'ok',
                    'size': size,
                    'chunks': chunks
                }
                
            else:
                return {'status': 'error', 'message': 'Unknown message type'}
                
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    async def _handle_download(self, message: Dict, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle shard download request"""
        try:
            shard_id = message.get('shard_id')
            
            # Get shard data
            shard_data = await self.shard_storage.get_shard(shard_id)
            if not shard_data:
                writer.write(self._encode_message({'status': 'error', 'message': 'Shard not found'}))
                await writer.drain()
                return
                
            # Calculate chunks
            total_chunks = (len(shard_data) + self.chunk_size - 1) // self.chunk_size
            
            # Send chunks
            for i in range(total_chunks):
                start = i * self.chunk_size
                end = min(start + self.chunk_size, len(shard_data))
                chunk = shard_data[start:end]
                
                # Send chunk
                chunk_msg = {
                    'type': 'chunk',
                    'index': i,
                    'data': chunk
                }
                writer.write(self._encode_message(chunk_msg))
                await writer.drain()
                
                # Wait for ack
                ack = await self._read_message(reader)
                if ack.get('status') != 'ok':
                    raise RuntimeError("Chunk transfer failed")
                    
            # Send completion
            writer.write(self._encode_message({'status': 'ok', 'message': 'Download complete'}))
            await writer.drain()
            
        except Exception as e:
            logging.error(f"Error handling download: {str(e)}")
            writer.write(self._encode_message({'status': 'error', 'message': str(e)}))
            await writer.drain()
            
    async def _handle_transfer(self, message: Dict, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle shard upload transfer"""
        try:
            shard_id = message.get('shard_id')
            size = message.get('size')
            chunks = message.get('chunks')
            
            # Create buffer for shard data
            shard_buffer = bytearray(size)
            
            # Receive chunks
            for i in range(chunks):
                # Read chunk message
                chunk_msg = await self._read_message(reader)
                if chunk_msg.get('type') != 'chunk':
                    raise RuntimeError(f"Expected chunk message, got {chunk_msg.get('type')}")
                    
                # Get chunk data
                chunk_index = chunk_msg.get('index')
                chunk_data = chunk_msg.get('data')
                
                # Store chunk
                start = chunk_index * self.chunk_size
                end = min(start + len(chunk_data), size)
                shard_buffer[start:end] = chunk_data
                
                # Send ack
                writer.write(self._encode_message({'status': 'ok'}))
                await writer.drain()
                
            # Read finalize message
            final_msg = await self._read_message(reader)
            if final_msg.get('type') != 'finalize':
                raise RuntimeError(f"Expected finalize message, got {final_msg.get('type')}")
                
            # Store complete shard
            await self.shard_storage.store_shard(shard_id, bytes(shard_buffer))
            
            # Send completion
            writer.write(self._encode_message({'status': 'ok', 'message': 'Transfer complete'}))
            await writer.drain()
            
        except Exception as e:
            logging.error(f"Error handling transfer: {str(e)}")
            writer.write(self._encode_message({'status': 'error', 'message': str(e)}))
            await writer.drain()
            
    def _read_message(self, reader: asyncio.StreamReader) -> Dict:
        """Read message from stream"""
        # Implementation same as in DataDistributionManager._read_message
        
    def _encode_message(self, message: Dict) -> bytes:
        """Encode message for transmission"""
        # Implementation same as in DataDistributionManager._encode_message