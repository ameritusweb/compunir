import asyncio
import logging
import time
from typing import Dict, List, Optional, Set
from decimal import Decimal
from collections import defaultdict
import math

class SybilMetricsCollector:
    """Collects and processes metrics related to Sybil protection"""
    
    def __init__(self, config: Dict, sybil_protection, node_manager, verification_system):
        self.config = config
        self.sybil_protection = sybil_protection
        self.node_manager = node_manager
        self.verification_system = verification_system
        self.metrics_history = []
        self.alert_history = []
        self.collection_task = None
        
        # Initialize metrics cache
        self.current_metrics = {
            'timestamp': time.time(),
            'trustScore': 100,
            'activeNodes': 0,
            'suspiciousNodes': 0,
            'avgReputation': 0.0,
            'totalStake': "0",
            'avgStake': "0",
            'geographicClusters': [],
            'alerts': []
        }
        
        # Configuration
        metrics_config = config.get('sybil_protection', {}).get('metrics', {})
        self.collection_interval = metrics_config.get('collection_interval', 300)  # 5 minutes
        self.history_length = metrics_config.get('history_length', 288)  # 24 hours at 5 min intervals
        
        logging.info("Initialized SybilMetricsCollector")
        
    async def start_collection(self):
        """Start background metrics collection"""
        if self.collection_task is not None:
            return
            
        self.collection_task = asyncio.create_task(self._collection_loop())
        logging.info("Started Sybil metrics collection task")
        
    async def stop_collection(self):
        """Stop metrics collection"""
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
            self.collection_task = None
            
        logging.info("Stopped Sybil metrics collection")
        
    async def _collection_loop(self):
        """Background loop for periodic metrics collection"""
        try:
            while True:
                try:
                    # Collect metrics
                    metrics = await self.collect_metrics()
                    
                    # Store in history
                    self.metrics_history.append(metrics)
                    
                    # Prune history if needed
                    if len(self.metrics_history) > self.history_length:
                        self.metrics_history = self.metrics_history[-self.history_length:]
                        
                    # Check for alertable conditions
                    await self._check_alerts(metrics)
                    
                except Exception as e:
                    logging.error(f"Error in metrics collection cycle: {str(e)}")
                    
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
        except asyncio.CancelledError:
            # Normal cancellation
            pass
        except Exception as e:
            logging.error(f"Metrics collection loop failed: {str(e)}")
            
    async def collect_metrics(self) -> Dict:
        """Collect current Sybil protection metrics"""
        # Get node stats from Sybil protection system
        node_stats = self.sybil_protection.get_node_stats()
        
        # Calculate network trust score
        trust_score = await self._calculate_trust_score(node_stats)
        
        # Get geographic distribution
        geo_clusters = await self._analyze_geographic_clusters()
        
        # Generate alerts if needed
        alerts = await self._generate_alerts(
            node_stats.get('suspicious_nodes', 0),
            geo_clusters,
            trust_score
        )
        
        # Compile metrics
        metrics = {
            'timestamp': time.time(),
            'trustScore': int(trust_score * 100),  # As percentage
            'activeNodes': node_stats.get('total_nodes', 0),
            'suspiciousNodes': node_stats.get('suspicious_nodes', 0),
            'avgReputation': float(node_stats.get('avg_reputation', 0)),
            'totalStake': node_stats.get('total_stake', "0"),
            'avgStake': node_stats.get('avg_stake', "0"),
            'geographicClusters': geo_clusters,
            'alerts': alerts,
            'reputationDistribution': node_stats.get('reputation_distribution', {}),
            'history': {
                'reputationHistory': self._get_reputation_history(),
                'trustHistory': self._get_trust_history(),
                'nodeCountHistory': self._get_node_count_history()
            }
        }
        
        # Update current metrics
        self.current_metrics = metrics
        
        return metrics
            
    async def _calculate_trust_score(self, node_stats: Dict) -> float:
        """Calculate overall network trust score"""
        try:
            # If no nodes, return perfect trust (nothing to distrust)
            if node_stats.get('total_nodes', 0) == 0:
                return 1.0
                
            # Get factors that affect trust
            suspicious_node_ratio = node_stats.get('suspicious_nodes', 0) / node_stats.get('total_nodes', 1)
            avg_reputation = node_stats.get('avg_reputation', 0.5)
            
            # Get verification stats
            verification_stats = await self._get_verification_stats()
            verification_factor = verification_stats.get('success_rate', 0.5)
            
            # Calculate combined trust score
            trust_score = (
                (1.0 - suspicious_node_ratio) * 0.4 +  # Lower ratio of suspicious nodes is better
                avg_reputation * 0.4 +                 # Higher average reputation is better
                verification_factor * 0.2              # Higher verification success rate is better
            )
            
            # Ensure reasonable bounds
            return max(0.0, min(1.0, trust_score))
            
        except Exception as e:
            logging.error(f"Error calculating trust score: {str(e)}")
            return 0.5  # Default to medium trust on error
            
    async def _analyze_geographic_clusters(self) -> List[Dict]:
        """Analyze and return geographic clustering data"""
        try:
            # This would connect to your node geographic data
            # to identify clusters of nodes
            
            # Get all nodes with geographic data
            geo_clusters = defaultdict(list)
            region_coords = {}
            
            # Group nodes by region
            for node_id, node in self.sybil_protection.nodes.items():
                if not node.geographic_data:
                    continue
                    
                geo = node.geographic_data
                region = geo.get('country_code', 'unknown')
                
                geo_clusters[region].append(node_id)
                
                # Store one set of coordinates per region for display
                if region not in region_coords and 'latitude' in geo and 'longitude' in geo:
                    region_coords[region] = {
                        'latitude': geo['latitude'],
                        'longitude': geo['longitude']
                    }
            
            # Convert to list format for frontend
            result = []
            max_cluster_size = self.config.get('sybil_protection', {}).get('max_geo_cluster_size', 5)
            
            for region, nodes in geo_clusters.items():
                # Skip regions with very few nodes
                if len(nodes) < 2:
                    continue
                    
                result.append({
                    'region': region,
                    'nodeCount': len(nodes),
                    'coordinates': region_coords.get(region, {'latitude': 0, 'longitude': 0}),
                    'suspicious': len(nodes) > max_cluster_size
                })
                
            # Sort by node count, descending
            result.sort(key=lambda x: x['nodeCount'], reverse=True)
            
            return result
            
        except Exception as e:
            logging.error(f"Error analyzing geographic clusters: {str(e)}")
            return []
            
    async def _generate_alerts(self, suspicious_nodes: int, geo_clusters: List[Dict], trust_score: float) -> List[Dict]:
        """Generate alerts based on metrics"""
        try:
            alerts = []
            
            # Check suspicious clusters
            suspicious_clusters = [c for c in geo_clusters if c['suspicious']]
            if suspicious_clusters:
                cluster_msg = ', '.join(f"{c['region']} ({c['nodeCount']} nodes)" 
                                      for c in suspicious_clusters[:3])
                alerts.append({
                    'severity': 'warning',
                    'title': 'Geographic Clustering Detected',
                    'message': f'Suspicious clustering in regions: {cluster_msg}'
                })
            
            # Check network-wide patterns
            if suspicious_nodes > 5:
                alerts.append({
                    'severity': 'high',
                    'title': 'Multiple Suspicious Nodes Detected',
                    'message': f'Found {suspicious_nodes} nodes with suspicious behavior'
                })
            
            # Check trust score
            if trust_score < 0.6:
                alerts.append({
                    'severity': 'high',
                    'title': 'Low Network Trust Score',
                    'message': f'Network trust score is {int(trust_score * 100)}%, below safe threshold'
                })
            
            # Store alerts in history if they're new
            current_time = time.time()
            for alert in alerts:
                alert_key = f"{alert['title']}:{alert['message']}"
                
                # Check if this exact alert is already in history
                is_new = True
                for old_alert in self.alert_history:
                    if old_alert['title'] == alert['title'] and old_alert['message'] == alert['message']:
                        # If alert exists but is old (>1 hour), consider it new again
                        if current_time - old_alert['timestamp'] < 3600:
                            is_new = False
                            break
                
                if is_new:
                    self.alert_history.append({
                        **alert,
                        'timestamp': current_time
                    })
            
            # Prune alert history (keep last 100 alerts)
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
            return alerts
            
        except Exception as e:
            logging.error(f"Error generating alerts: {str(e)}")
            return []
            
    async def _get_verification_stats(self) -> Dict:
        """Get statistics about verifications in the network"""
        try:
            # This would query your verification system
            # In a real implementation, you'd access your verification tracking system
            
            # For now, return a placeholder implementation
            return {
                'total_verifications': 1000,
                'success_rate': 0.85,
                'average_time': 2.5,
                'disputed_verifications': 50
            }
            
        except Exception as e:
            logging.error(f"Error getting verification stats: {str(e)}")
            return {
                'success_rate': 0.5  # Default neutral value
            }
            
    def _get_reputation_history(self) -> List[Dict]:
        """Get historical reputation data for charting"""
        try:
            history = []
            
            for metrics in self.metrics_history:
                history.append({
                    'timestamp': metrics['timestamp'],
                    'avgReputation': metrics['avgReputation'],
                    'minReputation': min([0.1] + [
                        float(rep) for rep in self.sybil_protection.nodes.values()
                        if rep.reputation_score > 0
                    ]) if self.sybil_protection.nodes else 0.1
                })
                
            return history
            
        except Exception as e:
            logging.error(f"Error getting reputation history: {str(e)}")
            return []
            
    def _get_trust_history(self) -> List[Dict]:
        """Get historical trust score data for charting"""
        try:
            return [
                {'timestamp': m['timestamp'], 'trustScore': m['trustScore']}
                for m in self.metrics_history
            ]
            
        except Exception as e:
            logging.error(f"Error getting trust history: {str(e)}")
            return []
            
    def _get_node_count_history(self) -> List[Dict]:
        """Get historical node count data for charting"""
        try:
            return [
                {
                    'timestamp': m['timestamp'],
                    'activeNodes': m['activeNodes'],
                    'suspiciousNodes': m['suspiciousNodes']
                }
                for m in self.metrics_history
            ]
            
        except Exception as e:
            logging.error(f"Error getting node count history: {str(e)}")
            return []
            
    def get_current_metrics(self) -> Dict:
        """Get most recently collected metrics"""
        return self.current_metrics
        
    async def get_alert_history(self, limit: int = 10) -> List[Dict]:
        """Get recent alert history"""
        # Sort by timestamp (newest first) and limit
        return sorted(
            self.alert_history,
            key=lambda a: a['timestamp'],
            reverse=True
        )[:limit]