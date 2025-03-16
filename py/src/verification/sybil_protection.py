import asyncio
import time
import hashlib
import json
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
import logging
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import math

@dataclass
class NodeIdentity:
    """Information about a node's identity and reputation"""
    node_id: str
    stake_amount: Decimal
    registration_time: float
    pow_difficulty: int
    reputation_score: float
    verified_computations: int
    last_verification: float
    geographic_data: Dict
    network_stats: Dict


class SybilProtectionSystem:
    """Core system for protecting against Sybil attacks in the network"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.nodes: Dict[str, NodeIdentity] = {}
        self.verified_stakes: Dict[str, Decimal] = {}  # node_id -> stake amount
        self.blacklisted_addresses: Set[str] = set()
        self.verification_history: Dict[str, List[Dict]] = {}
        self.suspicious_patterns: Dict[str, List[Dict]] = {}  # node_id -> suspicious patterns
        
        # Load constants from config
        sybil_config = config.get('sybil_protection', {})
        self.MIN_STAKE = Decimal(str(sybil_config.get('min_stake', '0.1')))  # Minimum XMR stake required
        self.BASE_POW_DIFFICULTY = sybil_config.get('base_pow_difficulty', 5)  # Base difficulty for PoW challenge
        self.MIN_REPUTATION = sybil_config.get('min_reputation', 0.3)  # Minimum reputation score to participate
        self.HISTORY_WINDOW = sybil_config.get('history_window', 7 * 24 * 3600)  # 7 days for historical analysis
        
        logging.info(f"Initialized SybilProtectionSystem with MIN_STAKE={self.MIN_STAKE}, " +
                    f"BASE_POW_DIFFICULTY={self.BASE_POW_DIFFICULTY}")
    
    async def verify_node_identity(self, 
                                 node_id: str,
                                 registration_data: Dict,
                                 proof_of_work: str) -> Dict:
        """Verify node identity and protect against Sybil attacks"""
        try:
            logging.info(f"Verifying identity for node {node_id}")
            
            # 1. Verify proof of work
            required_difficulty = self._get_pow_difficulty(node_id)
            if not self._verify_pow(proof_of_work, required_difficulty):
                logging.warning(f"Invalid proof of work for node {node_id}")
                return {
                    'verified': False,
                    'error': 'Invalid proof of work',
                    'required_difficulty': required_difficulty
                }

            # 2. Verify stake amount
            stake_amount = Decimal(str(registration_data.get('stake_amount', 0)))
            if stake_amount < self.MIN_STAKE:
                logging.warning(f"Insufficient stake ({stake_amount} < {self.MIN_STAKE}) for node {node_id}")
                return {
                    'verified': False,
                    'error': f'Insufficient stake. Minimum required: {self.MIN_STAKE} XMR',
                    'min_stake': str(self.MIN_STAKE)
                }

            # 3. Check geographic distribution
            geo_data = registration_data.get('geographic_data', {})
            if geo_data and self._detect_geographic_clustering(geo_data):
                logging.warning(f"Geographic clustering detected for node {node_id}")
                return {
                    'verified': False,
                    'error': 'Geographic clustering detected',
                    'cluster_threshold': self.config.get('sybil_protection', {}).get('max_geo_cluster_size', 5)
                }

            # 4. Analyze network behavior
            network_stats = registration_data.get('network_stats', {})
            if network_stats and self._detect_suspicious_networking(network_stats):
                logging.warning(f"Suspicious network behavior detected for node {node_id}")
                return {
                    'verified': False,
                    'error': 'Suspicious network behavior detected'
                }

            # 5. Calculate initial reputation score
            reputation_score = self._calculate_initial_reputation(
                stake_amount,
                registration_data
            )

            # 6. Create or update node identity
            node_identity = NodeIdentity(
                node_id=node_id,
                stake_amount=stake_amount,
                registration_time=time.time(),
                pow_difficulty=required_difficulty,
                reputation_score=reputation_score,
                verified_computations=0,
                last_verification=time.time(),
                geographic_data=geo_data,
                network_stats=network_stats
            )

            # Store identity
            self.nodes[node_id] = node_identity
            self.verified_stakes[node_id] = stake_amount
            
            logging.info(f"Node {node_id} verified with reputation score {reputation_score}")

            return {
                'verified': True,
                'reputation_score': reputation_score,
                'required_stake': str(self.MIN_STAKE),
                'pow_difficulty': required_difficulty
            }

        except Exception as e:
            logging.error(f"Error verifying node identity: {str(e)}")
            return {'verified': False, 'error': str(e)}

    def _verify_pow(self, proof: str, difficulty: int) -> bool:
        """Verify proof of work meets required difficulty"""
        try:
            # Check proof format
            if not isinstance(proof, str) or len(proof) != 64:
                return False

            # Verify hash meets difficulty requirement (has leading zeros)
            hash_int = int(proof, 16)
            return (hash_int >> (256 - difficulty)) == 0

        except Exception as e:
            logging.error(f"Error verifying PoW: {str(e)}")
            return False

    def _get_pow_difficulty(self, node_id: str) -> int:
        """Calculate required proof of work difficulty"""
        base_difficulty = self.BASE_POW_DIFFICULTY

        # Increase difficulty for nodes with suspicious behavior
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Lower reputation means higher difficulty
            reputation_factor = max(1, 2 - node.reputation_score)
            
            # Recent verifications reduce difficulty
            time_factor = min(1, (time.time() - node.last_verification) / 86400)
            
            adjusted_difficulty = int(base_difficulty * reputation_factor * time_factor)
            return max(base_difficulty, adjusted_difficulty)

        return base_difficulty

    def _detect_geographic_clustering(self, geo_data: Dict) -> bool:
        """Detect suspicious geographic clustering of nodes"""
        if not geo_data:
            return False

        try:
            # Get all nodes in same region
            region_nodes = [
                node for node in self.nodes.values()
                if self._are_nodes_proximate(node.geographic_data, geo_data)
            ]

            # Check if cluster size exceeds threshold
            max_cluster = self.config.get('sybil_protection', {}).get('max_geo_cluster_size', 5)
            if len(region_nodes) >= max_cluster:
                logging.warning(f"Geographic cluster size ({len(region_nodes)}) exceeds threshold ({max_cluster})")
                return True

            # Check for subnet similarity
            subnet_groups = {}
            for node in region_nodes:
                subnet = node.network_stats.get('subnet', '')
                subnet_groups[subnet] = subnet_groups.get(subnet, 0) + 1

            max_subnet = max(subnet_groups.values(), default=0)
            if max_subnet >= (max_cluster // 2):
                logging.warning(f"Subnet clustering detected. Max subnet size: {max_subnet}")
                return True

            return False

        except Exception as e:
            logging.error(f"Error detecting geographic clustering: {str(e)}")
            return False

    def _detect_suspicious_networking(self, network_stats: Dict) -> bool:
        """Detect suspicious network behavior patterns"""
        try:
            # Check for VPN/proxy usage if that's a concern
            if network_stats.get('is_proxy', False):
                logging.warning("Proxy usage detected in network stats")
                return True

            # Check connection patterns
            if self._analyze_connection_patterns(network_stats):
                return True

            # Check for IP range abuse
            if self._detect_ip_range_abuse(network_stats.get('ip_address')):
                return True

            return False

        except Exception as e:
            logging.error(f"Error detecting suspicious networking: {str(e)}")
            return True  # Fail closed on error

    def _calculate_initial_reputation(self, stake_amount: Decimal, registration_data: Dict) -> float:
        """Calculate initial reputation score for new node"""
        try:
            base_score = 0.5  # Start at 50%

            # Stake factor (more stake = higher initial reputation)
            stake_factor = min(1.0, float(stake_amount / (self.MIN_STAKE * 10)))
            
            # Hardware quality factor
            hw_factor = self._calculate_hardware_factor(registration_data.get('hardware_info', {}))
            
            # Network quality factor
            net_factor = self._calculate_network_factor(registration_data.get('network_stats', {}))

            # Combine factors
            reputation = base_score * (
                0.4 * stake_factor +
                0.3 * hw_factor +
                0.3 * net_factor
            )

            return max(0.1, min(1.0, reputation))  # Clamp between 0.1 and 1.0

        except Exception as e:
            logging.error(f"Error calculating reputation: {str(e)}")
            return 0.1  # Minimum score on error

    async def update_node_reputation(self, node_id: str, verification_data: Dict) -> float:
        """Update node reputation based on verification performance"""
        try:
            if node_id not in self.nodes:
                logging.warning(f"Attempted to update reputation for unknown node {node_id}")
                return 0.0

            node = self.nodes[node_id]
            current_reputation = node.reputation_score

            # Calculate performance factors
            success_rate = verification_data.get('success_rate', 0.0)
            response_time = verification_data.get('response_time', float('inf'))
            compute_quality = verification_data.get('compute_quality', 0.0)

            # Calculate reputation change
            # Higher success rate and compute quality increase reputation
            # Faster response time (lower value) increases reputation
            delta_reputation = (
                0.4 * (success_rate - 0.5) +  # Success rate impact
                0.3 * (1.0 - min(1.0, response_time / 10.0)) +  # Response time impact
                0.3 * (compute_quality - 0.5)  # Computation quality impact
            )

            # Apply change with dampening (0.1 factor) to avoid rapid fluctuations
            new_reputation = current_reputation + (delta_reputation * 0.1)
            new_reputation = max(0.0, min(1.0, new_reputation))  # Clamp between 0 and 1

            # Update node
            node.reputation_score = new_reputation
            node.verified_computations += 1
            node.last_verification = time.time()

            # Store verification history
            if node_id not in self.verification_history:
                self.verification_history[node_id] = []
                
            self.verification_history[node_id].append({
                'timestamp': time.time(),
                'old_reputation': current_reputation,
                'new_reputation': new_reputation,
                'delta': delta_reputation,
                'verification_data': verification_data
            })
            
            # Prune history if needed
            if len(self.verification_history[node_id]) > 100:  # Keep last 100 entries
                self.verification_history[node_id] = self.verification_history[node_id][-100:]

            logging.info(f"Updated reputation for node {node_id}: {current_reputation} -> {new_reputation}")
            return new_reputation

        except Exception as e:
            logging.error(f"Error updating reputation: {str(e)}")
            return 0.0

    def _analyze_connection_patterns(self, network_stats: Dict) -> bool:
        """Analyze network connection patterns for suspicious behavior"""
        try:
            # Check connection frequency
            connections_per_minute = network_stats.get('connections_per_minute', 0)
            max_connections = self.config.get('sybil_protection', {}).get('max_connections_per_minute', 100)
            
            if connections_per_minute > max_connections:
                logging.warning(f"High connection frequency detected: {connections_per_minute}/minute")
                return True

            # Check connection distribution
            peers = network_stats.get('peer_connections', {})
            if peers:
                # Calculate peer distribution entropy
                peer_counts = np.array(list(peers.values()))
                total_peers = peer_counts.sum()
                
                if total_peers > 0:
                    probabilities = peer_counts / total_peers
                    # Filter out zeros to avoid log(0)
                    probabilities = probabilities[probabilities > 0]
                    entropy = -np.sum(probabilities * np.log2(probabilities))
                    
                    min_entropy = self.config.get('sybil_protection', {}).get('min_peer_entropy', 2.0)
                    if entropy < min_entropy:
                        logging.warning(f"Low peer entropy detected: {entropy} < {min_entropy}")
                        return True

            return False

        except Exception as e:
            logging.error(f"Error analyzing connection patterns: {str(e)}")
            return True  # Fail closed on error

    def _detect_ip_range_abuse(self, ip_address: str) -> bool:
        """Detect if IP address is part of an abused range"""
        try:
            if not ip_address:
                return True  # No IP provided

            # Count nodes in same subnet (first 3 octets for IPv4 - /24 subnet)
            parts = ip_address.split('.')
            if len(parts) != 4:
                return True  # Invalid IP format
                
            subnet = '.'.join(parts[:3])  # /24 subnet
            
            subnet_count = sum(
                1 for node in self.nodes.values()
                if node.network_stats.get('ip_address', '').startswith(subnet)
            )
            
            max_subnet_nodes = self.config.get('sybil_protection', {}).get('max_nodes_per_subnet', 3)
            if subnet_count >= max_subnet_nodes:
                logging.warning(f"Subnet abuse detected. {subnet_count} nodes in subnet {subnet}")
                return True

            return False

        except Exception as e:
            logging.error(f"Error detecting IP range abuse: {str(e)}")
            return True  # Fail closed on error

    def _calculate_hardware_factor(self, hardware_info: Dict) -> float:
        """Calculate hardware quality factor"""
        try:
            if not hardware_info:
                return 0.5  # Default mid-range score

            # Check GPU capabilities
            gpu_score = min(1.0, hardware_info.get('gpu_tflops', 0) / 100.0)
            
            # Check memory
            memory_score = min(1.0, hardware_info.get('gpu_memory_gb', 0) / 24.0)
            
            # Combine scores
            return (gpu_score + memory_score) / 2

        except Exception as e:
            logging.error(f"Error calculating hardware factor: {str(e)}")
            return 0.5  # Default mid-range score on error

    def _calculate_network_factor(self, network_stats: Dict) -> float:
        """Calculate network quality factor"""
        try:
            if not network_stats:
                return 0.5  # Default mid-range score

            # Check bandwidth (higher is better)
            bandwidth_score = min(1.0, network_stats.get('bandwidth_mbps', 0) / 1000.0)
            
            # Check latency (lower is better)
            latency = network_stats.get('latency_ms', 1000)
            latency_score = max(0.0, min(1.0, 1 - (latency / 1000.0)))
            
            # Check stability (higher is better)
            stability = network_stats.get('stability', 0.5)

            # Combine scores
            return (bandwidth_score + latency_score + stability) / 3

        except Exception as e:
            logging.error(f"Error calculating network factor: {str(e)}")
            return 0.5  # Default mid-range score on error

    def _are_nodes_proximate(self, geo1: Dict, geo2: Dict) -> bool:
        """Check if two nodes are geographically proximate"""
        try:
            if not geo1 or not geo2:
                return False

            # Calculate distance between coordinates
            lat1, lon1 = geo1.get('latitude', 0), geo1.get('longitude', 0)
            lat2, lon2 = geo2.get('latitude', 0), geo2.get('longitude', 0)

            # Haversine formula for great-circle distance
            R = 6371  # Earth's radius in km

            lat1, lon1 = map(math.radians, [lat1, lon1])
            lat2, lon2 = map(math.radians, [lat2, lon2])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c

            proximity_threshold = self.config.get('sybil_protection', {}).get('proximity_threshold_km', 100)
            is_proximate = distance <= proximity_threshold
            
            if is_proximate:
                logging.info(f"Nodes are proximate: {distance} km (threshold: {proximity_threshold} km)")
                
            return is_proximate

        except Exception as e:
            logging.error(f"Error checking node proximity: {str(e)}")
            return False

    def get_node_stats(self) -> Dict:
        """Get statistics about nodes for monitoring"""
        try:
            total_nodes = len(self.nodes)
            if total_nodes == 0:
                return {
                    'total_nodes': 0,
                    'avg_reputation': 0,
                    'avg_stake': "0",
                    'total_stake': "0",
                    'suspicious_nodes': 0
                }
                
            # Calculate statistics
            reputations = [node.reputation_score for node in self.nodes.values()]
            avg_reputation = sum(reputations) / total_nodes
            
            stakes = [float(node.stake_amount) for node in self.nodes.values()]
            avg_stake = sum(stakes) / total_nodes
            total_stake = sum(stakes)
            
            # Count suspicious nodes (reputation below threshold)
            suspicious_nodes = sum(1 for rep in reputations if rep < self.MIN_REPUTATION)
            
            return {
                'total_nodes': total_nodes,
                'avg_reputation': avg_reputation,
                'avg_stake': f"{avg_stake:.4f}",
                'total_stake': f"{total_stake:.4f}",
                'suspicious_nodes': suspicious_nodes,
                'reputation_distribution': self._calculate_reputation_distribution()
            }
            
        except Exception as e:
            logging.error(f"Error getting node stats: {str(e)}")
            return {'error': str(e)}
            
    def _calculate_reputation_distribution(self) -> Dict:
        """Calculate distribution of node reputation scores"""
        try:
            if not self.nodes:
                return {}
                
            # Create distribution buckets
            buckets = {
                '0.0-0.2': 0,
                '0.2-0.4': 0,
                '0.4-0.6': 0,
                '0.6-0.8': 0,
                '0.8-1.0': 0
            }
            
            for node in self.nodes.values():
                rep = node.reputation_score
                if rep < 0.2:
                    buckets['0.0-0.2'] += 1
                elif rep < 0.4:
                    buckets['0.2-0.4'] += 1
                elif rep < 0.6:
                    buckets['0.4-0.6'] += 1
                elif rep < 0.8:
                    buckets['0.6-0.8'] += 1
                else:
                    buckets['0.8-1.0'] += 1
                    
            return buckets
            
        except Exception as e:
            logging.error(f"Error calculating reputation distribution: {str(e)}")
            return {}

    def get_suspicious_nodes(self) -> List[str]:
        """Get list of potentially suspicious node IDs"""
        return [
            node_id for node_id, node in self.nodes.items()
            if node.reputation_score < self.MIN_REPUTATION
        ]

    async def analyze_verification_patterns(self) -> Dict:
        """Analyze verification patterns across nodes for suspicious behavior"""
        try:
            if not self.verification_history:
                return {'patterns_detected': False}
                
            # Analyze verification result patterns
            result_patterns = self._analyze_verification_results()
            
            # Analyze verification timing patterns
            timing_patterns = self._analyze_verification_timing()
            
            # Combine results
            suspicious_nodes = set(result_patterns.get('suspicious_nodes', []))
            suspicious_nodes.update(timing_patterns.get('suspicious_nodes', []))
            
            return {
                'patterns_detected': len(suspicious_nodes) > 0,
                'suspicious_nodes': list(suspicious_nodes),
                'result_patterns': result_patterns.get('details', {}),
                'timing_patterns': timing_patterns.get('details', {})
            }
            
        except Exception as e:
            logging.error(f"Error analyzing verification patterns: {str(e)}")
            return {'patterns_detected': False, 'error': str(e)}
            
    def _analyze_verification_results(self) -> Dict:
        """Analyze patterns in verification results"""
        try:
            suspicious_nodes = []
            details = {}
            
            # Check each node's verification history
            for node_id, history in self.verification_history.items():
                if len(history) < 5:  # Need sufficient history
                    continue
                    
                # Look for patterns
                patterns = []
                
                # Check for uniform results (always same result)
                results = [entry.get('verification_data', {}).get('success', False) for entry in history]
                unique_results = set(results)
                if len(unique_results) == 1:
                    patterns.append('uniform_results')
                    
                # Check for exact alternating patterns
                if self._is_alternating_pattern(results):
                    patterns.append('alternating_results')
                
                # If patterns detected, mark node as suspicious
                if patterns:
                    suspicious_nodes.append(node_id)
                    details[node_id] = patterns
                    
            return {
                'suspicious_nodes': suspicious_nodes,
                'details': details
            }
            
        except Exception as e:
            logging.error(f"Error analyzing verification results: {str(e)}")
            return {'suspicious_nodes': []}
            
    def _analyze_verification_timing(self) -> Dict:
        """Analyze timing patterns in verification responses"""
        try:
            suspicious_nodes = []
            details = {}
            
            # Check each node's verification timing
            for node_id, history in self.verification_history.items():
                if len(history) < 5:  # Need sufficient history
                    continue
                    
                # Extract response times
                response_times = [
                    entry.get('verification_data', {}).get('response_time', 0)
                    for entry in history
                ]
                
                patterns = []
                
                # Check for impossibly fast responses
                avg_time = np.mean(response_times)
                min_time = self.config.get('sybil_protection', {}).get('min_verification_time', 0.1)
                if avg_time < min_time:
                    patterns.append('too_fast')
                
                # Check for exactly identical timings
                if len(set(response_times)) == 1 and len(response_times) > 3:
                    patterns.append('identical')
                
                # Check for mechanical patterns (e.g., exact linear increase)
                if self._is_mechanical_pattern(response_times):
                    patterns.append('mechanical')
                
                # If patterns detected, mark node as suspicious
                if patterns:
                    suspicious_nodes.append(node_id)
                    details[node_id] = patterns
            
            return {
                'suspicious_nodes': suspicious_nodes,
                'details': details
            }
            
        except Exception as e:
            logging.error(f"Error analyzing verification timing: {str(e)}")
            return {'suspicious_nodes': []}
    
    def _is_alternating_pattern(self, values: List) -> bool:
        """Check if a list has a simple alternating pattern"""
        if len(values) < 6:  # Need minimum length to detect pattern
            return False
            
        # Check for alternating True/False or similar
        for i in range(2, len(values)):
            if values[i] != values[i % 2]:
                return False
                
        return True
    
    def _is_mechanical_pattern(self, values: List[float]) -> bool:
        """Check if a list of values follows a mechanical pattern"""
        if len(values) < 5:
            return False
            
        # Check for linear pattern
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        avg_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        # If standard deviation is very low relative to average difference,
        # it suggests a mechanical pattern
        if avg_diff > 0 and std_diff/avg_diff < 0.05:
            return True
            
        return False