import asyncio
import logging
from typing import Dict, Optional, List
import time
from dataclasses import dataclass
import torch

from ..utils.gpu_monitoring import GPUMonitor
from .network_interface import NetworkInterface
from .job_executor import JobExecutor
from ..verification import VerificationSystem
from ..payment import PaymentProcessor

@dataclass
class NodeState:
    status: str
    current_job: Optional[str]
    gpu_metrics: Dict
    last_heartbeat: float
    verification_status: Optional[str] = None

class NodeManager:
    """Manages node operations, resources, and job execution"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.gpu_monitor = GPUMonitor()
        self.network = NetworkInterface(config)
        self.job_executor = JobExecutor(config)
        self.verification_system = VerificationSystem(config)
        self.payment_processor = PaymentProcessor(config['payment'])
        
        # State tracking
        self.node_id: Optional[str] = None
        self.state = NodeState(
            status='initializing',
            current_job=None,
            gpu_metrics={},
            last_heartbeat=time.time()
        )
        
        # Task tracking
        self._monitoring_task = None
        self._heartbeat_task = None
        self._job_task = None

    async def start(self):
        """Start node manager and all components"""
        try:
            # Initialize GPU monitoring
            if not self.gpu_monitor.initialize():
                raise RuntimeError("Failed to initialize GPU monitoring")
            
            # Register with network
            self.node_id = await self._register_node()
            
            # Start components
            await self.network.start()
            
            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._monitor_gpu())
            self._heartbeat_task = asyncio.create_task(self._send_heartbeats())
            
            self.state.status = 'active'
            self.logger.info("Node manager started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start node manager: {str(e)}")
            await self.stop()
            raise

    async def stop(self):
        """Stop node manager and cleanup"""
        try:
            self.state.status = 'stopping'
            
            # Cancel background tasks
            for task in [self._monitoring_task, self._heartbeat_task, self._job_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Stop components
            await self.network.stop()
            self.gpu_monitor.cleanup()
            
            self.logger.info("Node manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping node manager: {str(e)}")
            raise

    async def execute_job(self, job_spec: Dict):
        """Execute a training job"""
        try:
            if self.state.current_job:
                raise RuntimeError("Node is already executing a job")
            
            # Update state
            self.state.current_job = job_spec['job_id']
            self.state.status = 'executing'
            
            # Start job execution
            self._job_task = asyncio.create_task(
                self.job_executor.execute_job(job_spec)
            )
            
            # Wait for completion
            await self._job_task
            
            # Reset state
            self.state.current_job = None
            self.state.status = 'active'
            
        except Exception as e:
            self.logger.error(f"Job execution failed: {str(e)}")
            self.state.status = 'error'
            raise
        finally:
            self._job_task = None

    async def handle_verification_request(self, request: Dict) -> Dict:
        """Handle verification request from another node"""
        try:
            result = await self.verification_system.verify_work(
                proof=request['proof'],
                model=request['model'],
                inputs=request['inputs']
            )
            return result
        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}")
            raise

    async def _register_node(self) -> str:
        """Register node with the network"""
        try:
            # Get GPU information
            gpu_info = self.gpu_monitor.get_gpu_info()
            
            # Register with network
            node_info = {
                'gpu_info': gpu_info,
                'address': self.config['network']['address']
            }
            
            node_id = await self.network.add_node(node_info)
            self.logger.info(f"Registered with network, node_id: {node_id}")
            return node_id
            
        except Exception as e:
            self.logger.error(f"Registration failed: {str(e)}")
            raise

    async def _monitor_gpu(self):
        """Monitor GPU metrics"""
        try:
            while True:
                metrics = self.gpu_monitor.get_current_stats()
                if metrics:
                    self.state.gpu_metrics = {
                        'utilization': metrics.utilization,
                        'memory_used': metrics.memory_used,
                        'temperature': metrics.temperature,
                        'power_usage': metrics.power_usage
                    }
                    
                    # Check resource limits
                    if self._check_resource_limits(metrics):
                        await self._handle_resource_violation()
                        
                await asyncio.sleep(self.config.get('monitoring_interval', 1))
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"GPU monitoring error: {str(e)}")
            raise

    async def _send_heartbeats(self):
        """Send periodic heartbeats to network"""
        try:
            while True:
                try:
                    status = {
                        'status': self.state.status,
                        'gpu_metrics': self.state.gpu_metrics,
                        'current_job': self.state.current_job
                    }
                    
                    result = await self.network.update_node_status(
                        self.node_id,
                        status
                    )
                    
                    # Handle required actions
                    await self._handle_required_actions(result.get('actions', []))
                    
                    self.state.last_heartbeat = time.time()
                    
                except Exception as e:
                    self.logger.error(f"Heartbeat failed: {str(e)}")
                    
                await asyncio.sleep(self.config.get('heartbeat_interval', 30))
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Heartbeat task error: {str(e)}")
            raise

    def _check_resource_limits(self, metrics) -> bool:
        """Check if GPU metrics exceed configured limits"""
        limits = self.config.get('gpu_limits', {})
        
        if metrics.temperature > limits.get('max_temperature', 85):
            return True
            
        if metrics.power_usage > limits.get('max_power_usage', 250):
            return True
            
        if metrics.utilization > limits.get('max_utilization', 95):
            return True
            
        return False

    async def _handle_resource_violation(self):
        """Handle GPU resource limit violation"""
        self.logger.warning("Resource limits exceeded")
        
        if self.state.current_job:
            # Pause or stop current job
            await self.job_executor.pause_job(self.state.current_job)
            
        self.state.status = 'throttled'

    async def _handle_required_actions(self, actions: List[str]):
        """Handle required actions from network"""
        for action in actions:
            try:
                if action == 'REDUCE_LOAD':
                    await self._handle_resource_violation()
                elif action == 'UPDATE_SOFTWARE':
                    # Implementation would handle software updates
                    pass
                elif action == 'VERIFICATION_NEEDED':
                    await self._handle_verification_request()
            except Exception as e:
                self.logger.error(f"Failed to handle action {action}: {str(e)}")

    async def _handle_verification_request(self):
        """Handle pending verification request"""
        if self.state.verification_status == 'in_progress':
            return
            
        self.state.verification_status = 'in_progress'
        try:
            # Implementation would handle verification
            pass
        finally:
            self.state.verification_status = None