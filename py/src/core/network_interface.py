import asyncio
import logging
from typing import Dict, Optional, List
import time
from dataclasses import dataclass

@dataclass
class NodeInfo:
    node_id: str
    address: str
    gpu_info: Dict
    last_heartbeat: float
    status: str = 'active'
    current_job: Optional[str] = None

class NetworkInterface:
    """Core network interface handling node communication and management"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.active_nodes: Dict[str, NodeInfo] = {}
        self.logger = logging.getLogger(__name__)
        self._monitoring_task = None

    async def start(self):
        """Start network interface and monitoring"""
        try:
            self._monitoring_task = asyncio.create_task(self._monitor_nodes())
            self.logger.info("Network interface started")
        except Exception as e:
            self.logger.error(f"Failed to start network interface: {str(e)}")
            raise

    async def stop(self):
        """Stop network interface and cleanup"""
        try:
            if self._monitoring_task:
                self._monitoring_task.cancel()
                await self._monitoring_task
            self.logger.info("Network interface stopped")
        except Exception as e:
            self.logger.error(f"Error stopping network interface: {str(e)}")

    async def add_node(self, node_info: Dict) -> str:
        """Add a new node to the network"""
        try:
            node_id = self._generate_node_id(node_info)
            
            self.active_nodes[node_id] = NodeInfo(
                node_id=node_id,
                address=node_info['address'],
                gpu_info=node_info['gpu_info'],
                last_heartbeat=time.time()
            )
            
            self.logger.info(f"Added new node: {node_id}")
            return node_id
            
        except Exception as e:
            self.logger.error(f"Failed to add node: {str(e)}")
            raise

    async def remove_node(self, node_id: str):
        """Remove a node from the network"""
        try:
            if node_id in self.active_nodes:
                node = self.active_nodes[node_id]
                if node.current_job:
                    await self._handle_job_interruption(node.current_job)
                del self.active_nodes[node_id]
                self.logger.info(f"Removed node: {node_id}")
        except Exception as e:
            self.logger.error(f"Failed to remove node: {str(e)}")
            raise

    async def update_node_status(self, node_id: str, status: Dict) -> Dict:
        """Update node status and get required actions"""
        try:
            if node_id not in self.active_nodes:
                raise ValueError(f"Unknown node: {node_id}")
            
            node = self.active_nodes[node_id]
            node.last_heartbeat = time.time()
            node.status = status.get('status', 'active')
            
            return {
                'actions': self._get_required_actions(node),
                'config_updates': self._get_config_updates(node)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update node status: {str(e)}")
            raise

    def get_active_nodes(self) -> List[str]:
        """Get list of currently active node IDs"""
        return [
            node_id for node_id, node in self.active_nodes.items()
            if self._is_node_active(node)
        ]

    def get_node_info(self, node_id: str) -> Optional[NodeInfo]:
        """Get information about a specific node"""
        return self.active_nodes.get(node_id)

    async def _monitor_nodes(self):
        """Monitor node health and cleanup inactive nodes"""
        while True:
            try:
                current_time = time.time()
                timeout = self.config.get('node_timeout', 300)
                
                # Find and remove inactive nodes
                inactive_nodes = [
                    node_id for node_id, node in self.active_nodes.items()
                    if current_time - node.last_heartbeat > timeout
                ]
                
                for node_id in inactive_nodes:
                    await self.remove_node(node_id)
                
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in node monitoring: {str(e)}")
                await asyncio.sleep(60)

    async def _handle_job_interruption(self, job_id: str):
        """Handle interrupted job when node disconnects"""
        # Implementation would handle job failure and cleanup
        self.logger.warning(f"Job {job_id} interrupted due to node disconnection")

    def _generate_node_id(self, node_info: Dict) -> str:
        """Generate unique node ID"""
        import hashlib
        import uuid
        
        node_data = f"{node_info['address']}:{uuid.uuid4()}"
        return hashlib.sha256(node_data.encode()).hexdigest()[:16]

    def _is_node_active(self, node: NodeInfo) -> bool:
        """Check if node is currently active"""
        timeout = self.config.get('node_timeout', 300)
        return time.time() - node.last_heartbeat <= timeout

    def _get_required_actions(self, node: NodeInfo) -> List[str]:
        """Get list of required actions for node"""
        actions = []
        
        # Check resource utilization
        if self._check_resource_usage(node):
            actions.append('REDUCE_LOAD')
            
        # Check for pending tasks
        if self._has_pending_tasks(node):
            actions.append('PROCESS_PENDING')
            
        return actions

    def _get_config_updates(self, node: NodeInfo) -> Dict:
        """Get any configuration updates for node"""
        return {
            'heartbeat_interval': self.config.get('heartbeat_interval', 30),
            'resource_limits': self.config.get('resource_limits', {})
        }

    def _check_resource_usage(self, node: NodeInfo) -> bool:
        """Check if node is exceeding resource limits"""
        # Implementation would check GPU metrics
        return False

    def _has_pending_tasks(self, node: NodeInfo) -> bool:
        """Check if node has pending tasks"""
        # Implementation would check task queue
        return False