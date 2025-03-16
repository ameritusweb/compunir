from typing import Dict, Optional, List
import logging
import asyncio
from decimal import Decimal
import time

from ..core.node_manager import NodeManager
from ..verification.sybil_protection import SybilProtectionSystem
from ..payment.monero_wallet import MoneroWallet
from ..verification.verification_system import VerificationSystem

class NodeManagerWithSybil(NodeManager):
    """Enhanced NodeManager with Sybil attack protection"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.sybil_protection = SybilProtectionSystem(config)
        self.registration_challenges: Dict[str, str] = {}  # node_id -> pow_challenge
        self.monitoring_task = None
        
        logging.info("Initialized NodeManagerWithSybil with Sybil protection")
        
    async def start(self):
        """Start node manager with Sybil protection monitoring"""
        await super().start()
        
        # Start Sybil monitoring task
        self.monitoring_task = asyncio.create_task(self._monitor_suspicious_activity())
        logging.info("Started Sybil protection monitoring task")
        
    async def stop(self):
        """Stop node manager and monitoring tasks"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
                
        await super().stop()
        
    async def register_node(self, node_info: Dict) -> Dict:
        """Register node with Sybil protection"""
        try:
            node_id = node_info.get('node_id')
            logging.info(f"Processing node registration with Sybil protection for {node_id}")
            
            # 1. Verify basic node information
            if not self._validate_node_info(node_info):
                logging.warning(f"Invalid node information for {node_id}")
                return {
                    'status': 'error',
                    'error': 'Invalid node information'
                }
            
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
            
    async def _get_node_verification_stats(self, node_id: str) -> Dict:
        """Get verification statistics for a specific node"""
        try:
            # This would typically query your verification system
            # for the node's recent verification performance
            
            # Example implementation:
            verification_history = self.verification_system.get_node_verification_history(node_id)
            
            if not verification_history:
                return {
                    'success_rate': 0.5,  # Neutral starting point
                    'response_time': 1.0,  # Default response time
                    'compute_quality': 0.5  # Neutral compute quality
                }
            
            # Calculate statistics from recent history
            recent_verifications = verification_history[-10:]  # Last 10 verifications
            
            success_count = sum(1 for v in recent_verifications if v.get('success', False))
            success_rate = success_count / len(recent_verifications) if recent_verifications else 0.5
            
            response_times = [v.get('response_time', 1.0) for v in recent_verifications]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 1.0
            
            compute_qualities = [v.get('compute_quality', 0.5) for v in recent_verifications]
            avg_compute_quality = sum(compute_qualities) / len(compute_qualities) if compute_qualities else 0.5
            
            return {
                'success_rate': success_rate,
                'response_time': avg_response_time,
                'compute_quality': avg_compute_quality
            }
            
        except Exception as e:
            logging.error(f"Error getting verification stats for node {node_id}: {str(e)}")
            return {
                'success_rate': 0.5,
                'response_time': 1.0,
                'compute_quality': 0.5
            }
            
    async def _detect_suspicious_behavior(self, node_id: str) -> bool:
        """Detect suspicious node behavior"""
        try:
            if node_id not in self.nodes:
                return False
                
            # Check verification performance
            success_rate = await self._calculate_verification_success_rate(node_id)
            if success_rate < 0.5:  # Less than 50% success
                logging.warning(f"Node {node_id} has low verification success rate: {success_rate:.2f}")
                return True
                
            # Check network patterns
            if self._check_suspicious_network_patterns(node_id):
                logging.warning(f"Node {node_id} has suspicious network patterns")
                return True
                
            # Check computational consistency
            if await self._check_computational_consistency(node_id):
                logging.warning(f"Node {node_id} has inconsistent computation patterns")
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error detecting suspicious behavior for node {node_id}: {str(e)}")
            return False
            
    async def _handle_suspicious_node(self, node_id: str):
        """Handle detected suspicious node"""
        try:
            logging.warning(f"Handling suspicious node: {node_id}")
            
            # 1. Reduce reputation
            await self.sybil_protection.update_node_reputation(
                node_id,
                {'success_rate': 0.0, 'compute_quality': 0.0, 'response_time': 10.0}
            )
            
            # 2. Check if reputation is too low
            node_info = self.sybil_protection.nodes.get(node_id)
            if node_info and node_info.reputation_score < self.sybil_protection.MIN_REPUTATION:
                # 3. Suspend node
                await self.suspend_node(node_id, reason="Suspicious behavior detected")
                logging.warning(f"Node {node_id} suspended due to low reputation: {node_info.reputation_score}")
                
                # 4. Notify other components about the suspension
                await self._notify_node_suspension(node_id)
                
        except Exception as e:
            logging.error(f"Error handling suspicious node {node_id}: {str(e)}")
    
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
            
    async def _calculate_verification_success_rate(self, node_id: str) -> float:
        """Calculate verification success rate for a node"""
        try:
            # This would query your verification system
            # In a real implementation, this would access your verification tracking system
            verification_history = self.verification_system.get_node_verification_history(node_id)
            
            if not verification_history:
                return 0.5  # Default neutral value
                
            # Calculate success rate from recent history
            recent_history = verification_history[-20:]  # Last 20 verifications
            success_count = sum(1 for v in recent_history if v.get('success', False))
            
            return success_count / len(recent_history) if recent_history else 0.5
            
        except Exception as e:
            logging.error(f"Error calculating verification success rate for {node_id}: {str(e)}")
            return 0.5
            
    def _check_suspicious_network_patterns(self, node_id: str) -> bool:
        """Check for suspicious network patterns from a node"""
        try:
            # This function would check for network anomalies
            # In a real implementation, it would analyze network logs, connection patterns, etc.
            
            # For now, return a simple placeholder implementation
            return False
            
        except Exception as e:
            logging.error(f"Error checking network patterns for {node_id}: {str(e)}")
            return False
            
    async def _check_computational_consistency(self, node_id: str) -> bool:
        """Check for consistency in computational results"""
        try:
            # This would analyze the consistency of computation results
            # In a real implementation, you might compare against known benchmarks
            
            # Placeholder implementation
            return False
            
        except Exception as e:
            logging.error(f"Error checking computational consistency for {node_id}: {str(e)}")
            return False
            
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