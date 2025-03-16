import logging
import asyncio
import time
from typing import Dict, List, Set, Tuple, Optional
from decimal import Decimal
from collections import defaultdict

class SybilResponseSystem:
    """System for responding to detected Sybil attacks"""
    
    def __init__(self, config: Dict, node_manager, verification_system, payment_processor):
        self.config = config
        self.node_manager = node_manager
        self.verification_system = verification_system
        self.payment_processor = payment_processor
        
        self.suspicious_nodes = {}  # node_id -> detection data
        self.response_history = []  # history of responses
        
        # Response settings
        sybil_config = config.get('sybil_protection', {})
        self.response_config = sybil_config.get('response', {})
        self.verification_invalidation_window = self.response_config.get('verification_invalidation_window', 86400)  # 24h
        
        logging.info("Initialized SybilResponseSystem")
        
    async def handle_sybil_detection(self, detection_data: Dict):
        """Handle a detected potential Sybil attack"""
        try:
            logging.warning("Handling potential Sybil attack detection")
            
            # Extract affected nodes
            affected_nodes = detection_data.get('affected_nodes', [])
            if not affected_nodes:
                logging.warning("Sybil detection with no affected nodes")
                return
                
            # Store detection information
            detection_time = time.time()
            for node_id in affected_nodes:
                self.suspicious_nodes[node_id] = {
                    'detection_time': detection_time,
                    'detection_patterns': detection_data.get('patterns', []),
                    'pending_response': True
                }
                
            # Calculate severity
            severity = self._calculate_severity(detection_data)
            
            # Execute response based on severity
            if severity >= 0.8:  # High severity
                await self._execute_high_severity_response(affected_nodes, detection_data)
            elif severity >= 0.5:  # Medium severity
                await self._execute_medium_severity_response(affected_nodes, detection_data)
            else:  # Low severity
                await self._execute_low_severity_response(affected_nodes, detection_data)
                
            # Record response
            self._record_response(detection_data, severity)
            
            # Notify network
            severity_label = 'high' if severity >= 0.8 else 'medium' if severity >= 0.5 else 'low'
            await self._notify_network(severity_label, affected_nodes)
            
            logging.info(f"Completed Sybil response with severity {severity:.2f}")
            
        except Exception as e:
            logging.error(f"Error handling Sybil detection: {str(e)}")
            
    async def _execute_high_severity_response(self, affected_nodes: List[str], detection_data: Dict):
        """Execute response for high-severity Sybil attack"""
        try:
            logging.warning(f"Executing high severity response for {len(affected_nodes)} nodes")
            
            # 1. Suspend all affected nodes
            for node_id in affected_nodes:
                await self.node_manager.suspend_node(
                    node_id=node_id,
                    reason="High-severity Sybil attack mitigation"
                )
                
            # 2. Seize stake if applicable
            if self.response_config.get('enable_stake_seizure', False):
                await self._seize_stake(affected_nodes)
                
            # 3. Invalidate verifications from these nodes
            num_invalidated = await self._invalidate_verifications(affected_nodes)
            
            # 4. Increase security measures
            await self._increase_security_measures()
            
            logging.info(f"Completed high-severity response. Invalidated {num_invalidated} verifications")
            
        except Exception as e:
            logging.error(f"Error executing high severity response: {str(e)}")
            
    async def _execute_medium_severity_response(self, affected_nodes: List[str], detection_data: Dict):
        """Execute response for medium-severity Sybil attack"""
        try:
            logging.warning(f"Executing medium severity response for {len(affected_nodes)} nodes")
            
            # 1. Temporarily suspend most suspicious nodes
            most_suspicious = self._identify_most_suspicious(affected_nodes)
            for node_id in most_suspicious:
                await self.node_manager.suspend_node(
                    node_id=node_id,
                    reason="Medium-severity Sybil attack mitigation"
                )
                
            # 2. Increase verification requirements for other affected nodes
            await self._increase_verification_requirements(
                [n for n in affected_nodes if n not in most_suspicious]
            )
            
            # 3. Invalidate suspicious verifications
            num_invalidated = await self._invalidate_suspicious_verifications(affected_nodes)
            
            # 4. Enhance monitoring
            await self._enhance_monitoring(affected_nodes)
            
            logging.info(f"Completed medium-severity response. Invalidated {num_invalidated} verifications")
            
        except Exception as e:
            logging.error(f"Error executing medium severity response: {str(e)}")
            
    async def _execute_low_severity_response(self, affected_nodes: List[str], detection_data: Dict):
        """Execute response for low-severity Sybil attack"""
        try:
            logging.info(f"Executing low severity response for {len(affected_nodes)} nodes")
            
            # 1. Apply reputation penalty
            for node_id in affected_nodes:
                await self.node_manager.sybil_protection.update_node_reputation(
                    node_id,
                    {'success_rate': 0.3, 'compute_quality': 0.3, 'response_time': 3.0}
                )
                
            # 2. Increase PoW difficulty for affected nodes
            for node_id in affected_nodes:
                if node_id in self.node_manager.sybil_protection.nodes:
                    node = self.node_manager.sybil_protection.nodes[node_id]
                    node.pow_difficulty += 1
                    
            # 3. Flag for enhanced monitoring
            await self._enhance_monitoring(affected_nodes)
            
            logging.info(f"Completed low-severity response")
            
        except Exception as e:
            logging.error(f"Error executing low severity response: {str(e)}")
            
    async def _seize_stake(self, node_ids: List[str]):
        """Seize stake from malicious nodes"""
        try:
            # This would connect to your payment system
            # to seize the stake of malicious nodes
            
            # Simplified example:
            seized_amounts = []
            for node_id in node_ids:
                if node_id in self.node_manager.sybil_protection.verified_stakes:
                    stake_amount = self.node_manager.sybil_protection.verified_stakes[node_id]
                    
                    # Attempt to seize stake
                    success = await self.payment_processor.seize_stake(
                        node_id=node_id,
                        amount=stake_amount,
                        reason="Sybil attack mitigation"
                    )
                    
                    if success:
                        seized_amounts.append(stake_amount)
            
            total_seized = sum(seized_amounts, Decimal('0'))
            logging.info(f"Seized {total_seized} XMR from {len(seized_amounts)} nodes")
            
        except Exception as e:
            logging.error(f"Error seizing stake: {str(e)}")
            
    async def _invalidate_verifications(self, node_ids: List[str]):
        """Invalidate recent verifications from suspicious nodes"""
        try:
            # Get timeframe for invalidation
            cutoff_time = time.time() - self.verification_invalidation_window
            
            # Get recent verifications
            recent_verifications = await self.verification_system.get_verifications_since(cutoff_time)
            
            # Find affected verifications
            affected_verifications = []
            for verification in recent_verifications:
                if any(node_id in verification['verifier_ids'] for node_id in node_ids):
                    affected_verifications.append(verification)
            
            # Invalidate affected verifications
            for verification in affected_verifications:
                await self.verification_system.invalidate_verification(
                    verification_id=verification['id'],
                    reason="Sybil attack mitigation",
                    affected_nodes=node_ids
                )
                
                # Trigger re-verification if needed
                if verification['status'] == 'completed':
                    await self._trigger_reverification(verification)
            
            return len(affected_verifications)
            
        except Exception as e:
            logging.error(f"Error invalidating verifications: {str(e)}")
            return 0
            
    async def _invalidate_suspicious_verifications(self, node_ids: List[str]):
        """Invalidate only suspicious verifications from the nodes"""
        try:
            # This would selectively invalidate verifications that look suspicious
            # For now, a simplified version that only invalidates verifications 
            # where multiple suspicious nodes participated together
            
            # Get timeframe for invalidation
            cutoff_time = time.time() - self.verification_invalidation_window
            
            # Get recent verifications
            recent_verifications = await self.verification_system.get_verifications_since(cutoff_time)
            
            # Find suspicious verifications (multiple suspicious nodes participated)
            suspicious_verifications = []
            for verification in recent_verifications:
                # Count number of suspicious nodes involved
                suspicious_verifiers = [
                    v_id for v_id in verification['verifier_ids']
                    if v_id in node_ids
                ]
                
                # If multiple suspicious nodes were involved
                if len(suspicious_verifiers) > 1:
                    suspicious_verifications.append(verification)
            
            # Invalidate suspicious verifications
            for verification in suspicious_verifications:
                await self.verification_system.invalidate_verification(
                    verification_id=verification['id'],
                    reason="Suspicious verification pattern",
                    affected_nodes=node_ids
                )
                
                # Trigger re-verification
                if verification['status'] == 'completed':
                    await self._trigger_reverification(verification)
            
            return len(suspicious_verifications)
            
        except Exception as e:
            logging.error(f"Error invalidating suspicious verifications: {str(e)}")
            return 0
            
    async def _increase_security_measures(self):
        """Increase network-wide security measures"""
        try:
            # Update PoW difficulty
            current_difficulty = self.config.get('sybil_protection', {}).get('base_pow_difficulty', 5)
            new_difficulty = min(current_difficulty * 2, self.config.get('sybil_protection', {}).get('max_pow_difficulty', 12))
            await self._update_network_difficulty(new_difficulty)
            
            # Increase minimum stake requirement
            current_min_stake = self.config.get('sybil_protection', {}).get('min_stake', Decimal('0.1'))
            new_min_stake = min(current_min_stake * Decimal('1.5'), 
                              self.config.get('sybil_protection', {}).get('max_min_stake', Decimal('1.0')))
            await self._update_minimum_stake(new_min_stake)
            
            # Increase verification requirements
            await self._update_verification_requirements({
                'min_verifiers': self.config.get('verification', {}).get('min_verifiers', 3) + 1,
                'verification_threshold': min(
                    self.config.get('verification', {}).get('verification_threshold', 0.67) + 0.1,
                    0.9
                ),
                'verification_timeout': self.config.get('verification', {}).get('verification_timeout', 300) * 0.8
            })
            
            # Update network configuration
            await self.node_manager.broadcast_config_update({
                'pow_difficulty': new_difficulty,
                'min_stake': str(new_min_stake),
                'security_level': 'enhanced'
            })
            
            logging.info(f"Increased security measures: difficulty={new_difficulty}, min_stake={new_min_stake}")
            
        except Exception as e:
            logging.error(f"Error increasing security measures: {str(e)}")
            
    async def _increase_verification_requirements(self, node_ids: List[str]):
        """Increase verification requirements for specific nodes"""
        try:
            # Increase PoW difficulty for these nodes
            for node_id in node_ids:
                if node_id in self.node_manager.sybil_protection.nodes:
                    node = self.node_manager.sybil_protection.nodes[node_id]
                    node.pow_difficulty += 2
                    
            # Mark nodes for enhanced verification
            for node_id in node_ids:
                # This would set a flag in your verification system to apply
                # more stringent checks for these nodes
                await self.verification_system.set_enhanced_verification(
                    node_id,
                    enabled=True
                )
                
            logging.info(f"Increased verification requirements for {len(node_ids)} nodes")
            
        except Exception as e:
            logging.error(f"Error increasing verification requirements: {str(e)}")
            
    async def _enhance_monitoring(self, node_ids: List[str]):
        """Enable enhanced monitoring for suspicious nodes"""
        try:
            # This would connect to your monitoring system to
            # increase scrutiny on these nodes
            
            # In a real implementation, you might:
            # - Decrease time between checks
            # - Store more detailed metrics
            # - Apply more verification test cases
            
            # For now, just log the action
            logging.info(f"Enhanced monitoring enabled for {len(node_ids)} nodes")
            
        except Exception as e:
            logging.error(f"Error enhancing monitoring: {str(e)}")
            
    async def _notify_network(self, severity: str, affected_nodes: List[str]):
        """Notify network about detected Sybil attack"""
        try:
            notification = {
                'type': 'sybil_alert',
                'severity': severity,
                'timestamp': time.time(),
                'details': {
                    'affected_nodes': affected_nodes,
                    'detection_info': self._get_detection_summary(affected_nodes),
                    'response_measures': self._get_response_measures(severity)
                }
            }
            
            # Broadcast to all nodes
            await self.node_manager.broadcast_message(notification)
            
            # Log alert
            self._log_sybil_alert(notification)
            
            logging.info(f"Sent network notification about Sybil attack (severity: {severity})")
            
        except Exception as e:
            logging.error(f"Error notifying network: {str(e)}")
            
    def _calculate_severity(self, detection_data: Dict) -> float:
        """Calculate severity score for detected Sybil attack"""
        try:
            affected_nodes = detection_data.get('affected_nodes', [])
            
            factors = {
                'num_nodes': len(affected_nodes) / 10,  # 0.1 per node, up to 1.0
                'stake_amount': float(detection_data.get('total_stake', 0)) / 100,  # Stake impact
                'verification_impact': len(detection_data.get('affected_verifications', [])) / 100,
                'pattern_strength': detection_data.get('pattern_confidence', 0.5),
                'geographical_factor': self._calculate_geo_severity(detection_data)
            }
            
            # Weight factors
            weights = {
                'num_nodes': 0.2,
                'stake_amount': 0.2,
                'verification_impact': 0.3,
                'pattern_strength': 0.2,
                'geographical_factor': 0.1
            }
            
            # Calculate weighted severity
            severity = sum(score * weights[factor] for factor, score in factors.items())
            
            # Normalize to 0-1
            return max(0.0, min(1.0, severity))
            
        except Exception as e:
            logging.error(f"Error calculating severity: {str(e)}")
            return 1.0  # Default to high severity on error
            
    def _calculate_geo_severity(self, detection_data: Dict) -> float:
        """Calculate severity factor based on geographic clustering"""
        try:
            # This would analyze geographic data to determine severity
            # For now, a simplified implementation
            
            # If geographic clustering is mentioned in patterns
            if 'geographic_clustering' in detection_data.get('patterns', []):
                return 0.8
                
            return 0.0
            
        except Exception as e:
            logging.error(f"Error calculating geo severity: {str(e)}")
            return 0.0
            
    async def _trigger_reverification(self, verification: Dict):
        """Trigger re-verification of affected computation"""
        try:
            # Get original computation details
            computation_id = verification['computation_id']
            computation = await self.verification_system.get_computation(computation_id)
            
            if not computation:
                logging.warning(f"Could not find original computation {computation_id}")
                return
            
            # Request new verification
            await self.verification_system.request_verification(
                computation_id=computation_id,
                computation_data=computation['data'],
                priority='high',
                exclude_nodes=verification['verifier_ids']  # Exclude original verifiers
            )
            
            logging.info(f"Triggered re-verification for computation {computation_id}")
            
        except Exception as e:
            logging.error(f"Error triggering re-verification: {str(e)}")
            
    async def _update_network_difficulty(self, new_difficulty: int):
        """Update network-wide PoW difficulty"""
        try:
            # Update local config
            self.config['sybil_protection']['base_pow_difficulty'] = new_difficulty
            
            # Calculate transition period
            transition_blocks = self.config.get('sybil_protection', {}).get('difficulty_transition_blocks', 10)
            
            # Create difficulty update
            difficulty_update = {
                'new_difficulty': new_difficulty,
                'transition_blocks': transition_blocks,
                'activation_time': time.time() + 3600  # 1 hour notice
            }
            
            # Broadcast update
            await self.node_manager.broadcast_config_update({
                'type': 'difficulty_update',
                'data': difficulty_update
            })
            
            logging.info(f"Updated network difficulty to {new_difficulty}")
            
        except Exception as e:
            logging.error(f"Error updating network difficulty: {str(e)}")
            
    async def _update_verification_requirements(self, requirements: Dict):
        """Update verification system requirements"""
        try:
            # Update verification system config
            self.verification_system.update_requirements(requirements)
            
            # Broadcast update
            await self.node_manager.broadcast_config_update({
                'type': 'verification_update',
                'data': requirements
            })
            
            logging.info(f"Updated verification requirements: {requirements}")
            
        except Exception as e:
            logging.error(f"Error updating verification requirements: {str(e)}")
            
    def _identify_most_suspicious(self, node_ids: List[str]) -> List[str]:
        """Identify the most suspicious nodes from a list"""
        try:
            suspicious_scores = []
            
            for node_id in node_ids:
                if node_id in self.node_manager.sybil_protection.nodes:
                    node = self.node_manager.sybil_protection.nodes[node_id]
                    
                    # Lower reputation means more suspicious
                    reputation_factor = 1.0 - node.reputation_score
                    
                    # More verification failures means more suspicious
                    verification_history = self.verification_system.get_node_verification_history(node_id)
                    failure_rate = 0.5  # Default
                    if verification_history:
                        failures = sum(1 for v in verification_history if not v.get('success', True))
                        failure_rate = failures / len(verification_history)
                    
                    # Higher stake might indicate more investment in attack
                    stake_factor = float(node.stake_amount) / 10.0  # Normalize
                    
                    # Calculate suspicion score
                    suspicion_score = (
                        0.5 * reputation_factor +
                        0.3 * failure_rate +
                        0.2 * stake_factor
                    )
                    
                    suspicious_scores.append((node_id, suspicion_score))
            
            # Sort by suspicion score (descending)
            suspicious_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 30% of nodes, or at least one
            count = max(1, int(len(node_ids) * 0.3))
            return [node_id for node_id, _ in suspicious_scores[:count]]
            
        except Exception as e:
            logging.error(f"Error identifying most suspicious nodes: {str(e)}")
            # Return a few random nodes as fallback
            import random
            return random.sample(node_ids, min(3, len(node_ids)))
            
    def _get_detection_summary(self, affected_nodes: List[str]) -> Dict:
        """Get summary of Sybil detection data"""
        try:
            # Collect detection patterns
            patterns = defaultdict(int)
            for node_id in affected_nodes:
                if node_id in self.suspicious_nodes:
                    node_data = self.suspicious_nodes[node_id]
                    for pattern in node_data.get('detection_patterns', []):
                        patterns[pattern] += 1
            
            # Find earliest detection time
            detection_times = [
                self.suspicious_nodes[node_id]['detection_time']
                for node_id in affected_nodes
                if node_id in self.suspicious_nodes
            ]
            first_detection = min(detection_times) if detection_times else time.time()
            
            return {
                'total_nodes': len(affected_nodes),
                'detection_patterns': dict(patterns),
                'first_detection': first_detection,
                'network_impact': self._calculate_network_impact(affected_nodes)
            }
            
        except Exception as e:
            logging.error(f"Error getting detection summary: {str(e)}")
            return {}
            
    def _get_response_measures(self, severity: str) -> List[str]:
        """Get list of applied response measures"""
        if severity == 'high':
            return [
                "Node suspension",
                "Stake seizure",
                "Verification invalidation",
                "Increased network security",
                "Network-wide alert"
            ]
        elif severity == 'medium':
            return [
                "Temporary suspension",
                "Increased verification requirements",
                "Enhanced monitoring",
                "Network notification"
            ]
        else:
            return [
                "Increased monitoring",
                "Reputation penalty",
                "Increased PoW difficulty"
            ]
            
    def _calculate_network_impact(self, affected_nodes: List[str]) -> Dict:
        """Calculate impact of Sybil attack on network"""
        try:
            total_nodes = len(self.node_manager.nodes)
            if total_nodes == 0:
                return {
                    'node_percentage': 0,
                    'stake_percentage': 0,
                    'network_risk_level': 'low'
                }
                
            # Calculate stake impact
            total_stake = sum(
                float(self.node_manager.sybil_protection.nodes[n].stake_amount)
                for n in self.node_manager.sybil_protection.nodes
            )
            
            affected_stake = sum(
                float(self.node_manager.sybil_protection.nodes[n].stake_amount)
                for n in affected_nodes
                if n in self.node_manager.sybil_protection.nodes
            )
            
            node_ratio = len(affected_nodes) / total_nodes
            stake_ratio = affected_stake / total_stake if total_stake > 0 else 0
            
            return {
                'node_percentage': node_ratio * 100,
                'stake_percentage': stake_ratio * 100,
                'network_risk_level': self._calculate_risk_level(node_ratio, stake_ratio)
            }
            
        except Exception as e:
            logging.error(f"Error calculating network impact: {str(e)}")
            return {
                'node_percentage': 0,
                'stake_percentage': 0,
                'network_risk_level': 'unknown'
            }
            
    def _calculate_risk_level(self, node_ratio: float, stake_ratio: float) -> str:
        """Calculate network risk level"""
        try:
            # Combined risk score
            risk_score = node_ratio * 0.4 + stake_ratio * 0.6
            
            if risk_score > 0.3:
                return 'critical'
            elif risk_score > 0.15:
                return 'high'
            elif risk_score > 0.05:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logging.error(f"Error calculating risk level: {str(e)}")
            return 'critical'  # Default to critical on error
            
    def _record_response(self, detection_data: Dict, severity: float):
        """Record Sybil attack response for analysis"""
        try:
            response_record = {
                'timestamp': time.time(),
                'severity': severity,
                'affected_nodes': detection_data.get('affected_nodes', []),
                'detection_patterns': detection_data.get('patterns', []),
                'response_measures': self._get_response_measures(
                    'high' if severity >= 0.8 else 'medium' if severity >= 0.5 else 'low'
                ),
                'network_impact': self._calculate_network_impact(
                    detection_data.get('affected_nodes', [])
                )
            }
            
            self.response_history.append(response_record)
            
            # Prune old history
            max_history = self.config.get('sybil_protection', {}).get('max_response_history', 1000)
            if len(self.response_history) > max_history:
                self.response_history = self.response_history[-max_history:]
                
        except Exception as e:
            logging.error(f"Error recording response: {str(e)}")
            
    def _log_sybil_alert(self, notification: Dict):
        """Log Sybil alert to secure log"""
        try:
            # In a real implementation, this would log to a secure audit log
            # For now, just log to the standard logger
            logging.warning(f"SYBIL ALERT: {notification['severity']} severity attack detected")
            logging.warning(f"  Affected nodes: {len(notification['details']['affected_nodes'])}")
            logging.warning(f"  Response measures: {notification['details']['response_measures']}")
            
        except Exception as e:
            logging.error(f"Error logging Sybil alert: {str(e)}")
            
    async def _update_minimum_stake(self, new_min_stake: Decimal):
        """Update minimum stake requirement"""
        try:
            # Update local config
            self.config['sybil_protection']['min_stake'] = new_min_stake
            
            # Create stake update
            stake_update = {
                'new_min_stake': str(new_min_stake),
                'activation_time': time.time() + 86400  # 24 hour notice
            }
            
            # Broadcast update
            await self.node_manager.broadcast_config_update({
                'type': 'stake_update',
                'data': stake_update
            })
            
            logging.info(f"Updated minimum stake to {new_min_stake}")
            
        except Exception as e:
            logging.error(f"Error updating minimum stake: {str(e)}")