import asyncio
import logging
import time
from typing import Dict, Optional, List
from decimal import Decimal
from dataclasses import dataclass

from ..utils.gpu_monitoring import GPUMonitor
from ..core.job_executor import JobExecutor
from .network_interface import NetworkInterface
from ..verification.verification_system import AdvancedVerificationSystem
from ..payment.monero_wallet import MoneroWallet
from ..payment.monero_payment_processor import MoneroPaymentProcessor
from ..verification.sybil_protection import SybilProtectionSystem

@dataclass
class NodeState:
    """Tracks current node state"""
    status: str
    current_job: Optional[str]
    gpu_metrics: Dict
    last_heartbeat: float
    verification_status: Optional[str] = None
    wallet_address: Optional[str] = None

class NodeManager:
    """Manages node operations, security, and job execution"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.gpu_monitor = GPUMonitor()
        self.network = NetworkInterface(config)
        self.job_executor = JobExecutor(config)
        self.verification_system = AdvancedVerificationSystem(config)
        self.payment_processor = MoneroPaymentProcessor(config['payment'])
        self.sybil_protection = SybilProtectionSystem(config)  # Sybil attack protection

        # Node state tracking
        self.node_id: Optional[str] = None
        self.nodes = {}  # Track registered nodes
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
        self._sybil_monitoring_task = None

    async def start(self):
        """Start node manager and all components"""
        try:
            # Initialize GPU monitoring
            if not self.gpu_monitor.initialize():
                raise RuntimeError("Failed to initialize GPU monitoring")
            
            # Register with network
            self.node_id = await self._register_node()
            
            # Start network operations
            await self.network.start()

            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._monitor_gpu())
            self._heartbeat_task = asyncio.create_task(self._send_heartbeats())
            self._sybil_monitoring_task = asyncio.create_task(self._monitor_suspicious_activity())

            self.state.status = 'active'
            self.logger.info("Unified Node Manager started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start node manager: {str(e)}")
            await self.stop()
            raise

    async def stop(self):
        """Stop node manager and cleanup"""
        try:
            self.state.status = 'stopping'
            
            # Cancel background tasks
            for task in [self._monitoring_task, self._heartbeat_task, self._job_task, self._sybil_monitoring_task]:
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

    async def register_node(self, node_info: Dict) -> Dict:
        """Register node with Sybil protection"""
        try:
            node_id = node_info.get('node_id')
            logging.info(f"Processing node registration with Sybil protection for {node_id}")
            wallet_address = node_info.get('wallet_address', None) 

            # 1. Verify basic node information
            if not self._validate_node_info(node_info):
                logging.warning(f"Invalid node information for {node_id}")
                return {
                    'status': 'error',
                    'error': 'Invalid node information'
                }
            
            if not wallet_address:
                logging.warning(f"Node {node_id} is missing a wallet address!")

            # 2. Verify Monero stake
            stake_verification = await self._verify_stake(
                node_info.get('wallet_address'),
                node_info.get('stake_transaction_id')
            )
            
            if not stake_verification['verified']:
                logging.warning(f"Stake verification failed for {node_id}: {stake_verification['error']}")
                return {
                    'status': 'error',
                    'error': f"Stake verification failed: {stake_verification['error']}"
                }
            
            # 3. Verify proof of work
            pow_verification = await self.sybil_protection.verify_node_identity(
                node_id=node_id,
                registration_data={
                    'stake_amount': stake_verification['amount'],
                    'geographic_data': node_info.get('geographic_data', {}),
                    'network_stats': node_info.get('network_stats', {}),
                    'hardware_info': node_info.get('gpu_info', {})
                },
                proof_of_work=node_info.get('pow_proof', '')
            )
            
            if not pow_verification['verified']:
                logging.warning(f"Identity verification failed for {node_id}: {pow_verification.get('error')}")
                return {
                    'status': 'error',
                    'error': f"Identity verification failed: {pow_verification.get('error')}"
                }
            
            # 4. Register node with parent class
            registration_result = await super().register_node(node_info)
            if registration_result.get('status') != 'success':
                logging.warning(f"Base registration failed for {node_id}: {registration_result.get('error')}")
                return registration_result
            
            # 5. Setup individual node monitoring
            asyncio.create_task(self._monitor_node_behavior(node_id))
            
            # Store node state including wallet address
            self.nodes[node_id] = NodeState(
                status='active',
                current_job=None,
                gpu_metrics={},
                last_heartbeat=time.time(),
                wallet_address=wallet_address
            )

            logging.info(f"Node {node_id} registered with wallet address: {wallet_address}")
            
            logging.info(f"Node {node_id} successfully registered with Sybil protection")
            return {
                'status': 'success',
                'node_id': node_id,
                'reputation_score': pow_verification['reputation_score'],
                'registration_time': registration_result['registration_time']
            }
            
        except Exception as e:
            logging.error(f"Error registering node with Sybil protection: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    async def _verify_stake(self, wallet_address: str, transaction_id: str) -> Dict:
        """Verify node's Monero stake"""
        try:
            if not transaction_id or not wallet_address:
                return {
                    'verified': False,
                    'error': 'Missing wallet address or transaction ID'
                }
                
            async with MoneroWallet(self.config['wallet']) as wallet:
                # Verify transaction
                tx_info = await wallet.get_transfer_by_txid(transaction_id)
                if not tx_info:
                    return {
                        'verified': False,
                        'error': 'Transaction not found'
                    }
                
                # Verify amount
                min_stake = self.sybil_protection.MIN_STAKE
                if tx_info['amount'] < min_stake:
                    return {
                        'verified': False,
                        'error': f'Insufficient stake amount. Minimum: {min_stake} XMR'
                    }
                
                # Verify destination
                if tx_info.get('address') != wallet_address:
                    return {
                        'verified': False,
                        'error': 'Invalid stake destination'
                    }
                
                # Verify confirmation count
                min_confirmations = self.config.get('min_stake_confirmations', 10)
                if tx_info.get('confirmations', 0) < min_confirmations:
                    return {
                        'verified': False,
                        'error': f'Insufficient confirmations. Required: {min_confirmations}'
                    }
                
                return {
                    'verified': True,
                    'amount': tx_info['amount'],
                    'confirmations': tx_info['confirmations']
                }
                
        except Exception as e:
            logging.error(f"Error verifying stake: {str(e)}")
            return {
                'verified': False,
                'error': str(e)
            }

    async def execute_job(self, job_spec: Dict):
        """Execute a distributed ML job"""
        try:
            if self.state.current_job:
                raise RuntimeError("Node is already executing a job")

            # Start job execution
            self.state.current_job = job_spec['job_id']
            self.state.status = 'executing'
            self._job_task = asyncio.create_task(self.job_executor.execute_job(job_spec))

            await self._job_task  # Wait for completion
            self.state.current_job = None
            self.state.status = 'active'

        except Exception as e:
            self.logger.error(f"Job execution failed: {str(e)}")
            self.state.status = 'error'
            raise
        finally:
            self._job_task = None

    async def _monitor_node_behavior(self, node_id: str):
        """Monitor individual node behavior for Sybil detection"""
        try:
            check_interval = self.config.get('sybil_protection', {}).get('node_check_interval', 300)  # 5 minutes
            
            while node_id in self.nodes:
                # Get recent verification performance
                verification_stats = await self._get_node_verification_stats(node_id)
                
                # Update reputation
                await self.sybil_protection.update_node_reputation(
                    node_id,
                    verification_stats
                )
                
                # Check for suspicious behavior
                if await self._detect_suspicious_behavior(node_id):
                    await self._handle_suspicious_node(node_id)
                    
                await asyncio.sleep(check_interval)
                
        except asyncio.CancelledError:
            # Task was cancelled, clean exit
            pass
        except Exception as e:
            logging.error(f"Error monitoring node {node_id}: {str(e)}")
            
    async def _monitor_suspicious_activity(self):
        """Monitor network-wide suspicious activity patterns"""
        try:
            check_interval = self.config.get('sybil_protection', {}).get('network_check_interval', 1800)  # 30 minutes
            
            while True:
                # Analyze network-wide verification patterns
                pattern_analysis = await self.sybil_protection.analyze_verification_patterns()
                
                if pattern_analysis.get('patterns_detected', False):
                    suspicious_nodes = pattern_analysis.get('suspicious_nodes', [])
                    if suspicious_nodes:
                        logging.warning(f"Detected suspicious verification patterns from {len(suspicious_nodes)} nodes")
                        await self._handle_suspicious_pattern(suspicious_nodes, pattern_analysis)
                
                # Analyze geographic distribution
                geo_analysis = await self._analyze_geographic_distribution()
                if geo_analysis.get('suspicious_clusters', False):
                    logging.warning("Detected suspicious geographic clustering")
                    await self._handle_geographic_clustering(geo_analysis)
                
                await asyncio.sleep(check_interval)
                
        except asyncio.CancelledError:
            # Task was cancelled, clean exit
            pass
        except Exception as e:
            logging.error(f"Error in suspicious activity monitoring: {str(e)}")

    async def _handle_suspicious_pattern(self, suspicious_nodes: List[str], pattern_analysis: Dict):
        """Handle detected suspicious verification patterns"""
        try:
            # 1. Apply reputation penalty to all suspicious nodes
            for node_id in suspicious_nodes:
                await self.sybil_protection.update_node_reputation(
                    node_id,
                    {'success_rate': 0.0, 'compute_quality': 0.0, 'response_time': 10.0}
                )
                
            # 2. Check patterns for potential collusion
            if 'result_patterns' in pattern_analysis and any('uniform_results' in patterns 
                                                          for patterns in pattern_analysis['result_patterns'].values()):
                logging.warning("Potential collusion detected in verification results")
                
                # 3. Take more severe action for collusion
                await self._handle_collusion(suspicious_nodes)
                
            # 4. Notify monitoring systems
            await self._notify_suspicious_pattern(pattern_analysis)
            
        except Exception as e:
            logging.error(f"Error handling suspicious pattern: {str(e)}")
            
    async def _analyze_geographic_distribution(self) -> Dict:
        """Analyze geographic distribution of nodes for clustering"""
        try:
            # Implementation would analyze the geographic data of nodes
            # to detect unusual clustering
            
            # This is a simplified example:
            geo_clusters = {}
            
            for node_id, node in self.nodes.items():
                if node_id in self.sybil_protection.nodes:
                    sybil_node = self.sybil_protection.nodes[node_id]
                    geo_data = sybil_node.geographic_data
                    
                    if geo_data:
                        region_key = geo_data.get('country_code', 'unknown')
                        if region_key not in geo_clusters:
                            geo_clusters[region_key] = []
                            
                        geo_clusters[region_key].append(node_id)
            
            # Find suspiciously large clusters
            max_cluster_size = self.config.get('sybil_protection', {}).get('max_geo_cluster_size', 5)
            suspicious_clusters = {
                region: nodes for region, nodes in geo_clusters.items()
                if len(nodes) > max_cluster_size
            }
            
            return {
                'clusters': geo_clusters,
                'suspicious_clusters': len(suspicious_clusters) > 0,
                'suspicious_regions': list(suspicious_clusters.keys()),
                'affected_nodes': [node for nodes in suspicious_clusters.values() for node in nodes]
            }
            
        except Exception as e:
            logging.error(f"Error analyzing geographic distribution: {str(e)}")
            return {'suspicious_clusters': False}
            
    async def _handle_geographic_clustering(self, geo_analysis: Dict):
        """Handle detected geographic clustering"""
        try:
            affected_nodes = geo_analysis.get('affected_nodes', [])
            if not affected_nodes:
                return
                
            logging.warning(f"Handling geographic clustering affecting {len(affected_nodes)} nodes")
            
            # 1. Increase verification requirements for affected nodes
            for node_id in affected_nodes:
                if node_id in self.sybil_protection.nodes:
                    # Increase PoW difficulty for these nodes
                    self.sybil_protection.nodes[node_id].pow_difficulty += 2
            
            # 2. Apply moderate reputation penalty
            for node_id in affected_nodes:
                await self.sybil_protection.update_node_reputation(
                    node_id,
                    {'success_rate': 0.3, 'compute_quality': 0.3, 'response_time': 5.0}
                )
            
            # 3. Notify monitoring systems
            await self._notify_geographic_clustering(geo_analysis)
            
        except Exception as e:
            logging.error(f"Error handling geographic clustering: {str(e)}")

    async def suspend_node(self, node_id: str, reason: str) -> bool:
        """Suspend a node from the network"""
        try:
            if node_id not in self.nodes:
                return False
                
            # Mark node as suspended
            self.nodes[node_id].status = "suspended"
            self.nodes[node_id].suspension_reason = reason
            self.nodes[node_id].suspension_time = time.time()
            
            # Log suspension
            logging.warning(f"Node {node_id} suspended: {reason}")
            
            # Could add additional logic here like:
            # - Notifying the node
            # - Updating network state
            # - Triggering job reassignment
            
            return True
            
        except Exception as e:
            logging.error(f"Error suspending node {node_id}: {str(e)}")
            return False
        
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

    async def _handle_collusion(self, node_ids: List[str]):
        """Handle detected collusion between nodes"""
        try:
            logging.warning(f"Handling potential collusion between {len(node_ids)} nodes")
            
            # 1. Suspend all colluding nodes
            for node_id in node_ids:
                await self.suspend_node(node_id, reason="Suspected collusion in verification")
                
            # 2. Invalidate recent verifications from these nodes
            # This would connect to your verification system to invalidate results
            await self._invalidate_verifications_from_nodes(node_ids)
            
            # 3. Add to blacklist for heightened scrutiny
            for node_id in node_ids:
                if node_id in self.sybil_protection.nodes:
                    # Add identifying information to blacklist
                    node = self.sybil_protection.nodes[node_id]
                    
                    # Example: Blacklist IP subnet
                    if 'network_stats' in node.network_stats and 'ip_address' in node.network_stats:
                        ip = node.network_stats['ip_address']
                        subnet = '.'.join(ip.split('.')[:3])  # /24 subnet for IPv4
                        self.sybil_protection.blacklisted_addresses.add(subnet)
            
        except Exception as e:
            logging.error(f"Error handling collusion: {str(e)}")
            
    async def _invalidate_verifications_from_nodes(self, node_ids: List[str]):
        """Invalidate recent verifications from specified nodes"""
        try:
            # This would connect to your verification system
            # to invalidate recent verifications from these nodes
            
            # Example implementation (would need to be adapted to your system):
            invalidation_count = await self.verification_system.invalidate_verifications_from_nodes(
                node_ids,
                reason="Sybil attack mitigation"
            )
            
            logging.info(f"Invalidated {invalidation_count} verifications from suspicious nodes")
            
        except Exception as e:
            logging.error(f"Error invalidating verifications: {str(e)}")

    async def _notify_node_suspension(self, node_id: str):
        """Notify relevant systems about node suspension"""
        try:
            # This would notify other components that need to know about node suspension
            # Example: Job system, verification system, etc.
            
            # Placeholder implementation
            pass
            
        except Exception as e:
            logging.error(f"Error notifying about node suspension: {str(e)}")
            
    async def _notify_suspicious_pattern(self, pattern_analysis: Dict):
        """Notify monitoring systems about suspicious patterns"""
        try:
            # This would send alerts/notifications to monitoring systems
            # In a real implementation, might send to dashboard, alert system, etc.
            
            # Placeholder implementation
            pass
            
        except Exception as e:
            logging.error(f"Error notifying about suspicious pattern: {str(e)}")
            
    async def _notify_geographic_clustering(self, geo_analysis: Dict):
        """Notify monitoring systems about geographic clustering"""
        try:
            # This would send alerts/notifications about geographic clustering
            
            # Placeholder implementation
            pass
            
        except Exception as e:
            logging.error(f"Error notifying about geographic clustering: {str(e)}")

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