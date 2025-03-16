import asyncio
import logging
from typing import Dict, List
import json
import aiohttp
from datetime import datetime

class SybilNotificationSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.notification_endpoints = config.get('notification_endpoints', {})
        self.alert_history = []
        self.max_history = config.get('max_alert_history', 1000)

    async def notify_node_suspension(self, node_id: str, reason: str):
        """Notify about node suspension"""
        try:
            alert = {
                'type': 'node_suspension',
                'severity': 'high',
                'node_id': node_id,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat(),
                'action_taken': 'suspension'
            }

            # Send to monitoring system
            await self._send_to_monitoring(alert)

            # Update network state
            await self._notify_network_update({
                'type': 'node_status_change',
                'node_id': node_id,
                'new_status': 'suspended',
                'reason': reason
            })

            # Store in history
            self._store_alert(alert)

        except Exception as e:
            logging.error(f"Error in node suspension notification: {str(e)}")

    async def notify_suspicious_pattern(self, pattern_analysis: Dict):
        """Notify about suspicious patterns"""
        try:
            affected_nodes = pattern_analysis.get('suspicious_nodes', [])
            patterns = pattern_analysis.get('result_patterns', {})

            alert = {
                'type': 'suspicious_pattern',
                'severity': 'warning',
                'affected_nodes': affected_nodes,
                'pattern_details': patterns,
                'timestamp': datetime.utcnow().isoformat(),
                'recommendations': self._generate_pattern_recommendations(patterns)
            }

            # Send to monitoring system
            await self._send_to_monitoring(alert)

            # Send to security analysis system
            await self._send_to_security_analysis(pattern_analysis)

            # Store in history
            self._store_alert(alert)

        except Exception as e:
            logging.error(f"Error in suspicious pattern notification: {str(e)}")

    async def notify_geographic_clustering(self, geo_analysis: Dict):
        """Notify about geographic clustering"""
        try:
            clusters = geo_analysis.get('clusters', {})
            suspicious_regions = geo_analysis.get('suspicious_regions', [])

            alert = {
                'type': 'geographic_clustering',
                'severity': 'medium',
                'affected_regions': suspicious_regions,
                'cluster_sizes': {region: len(nodes) for region, nodes in clusters.items()},
                'timestamp': datetime.utcnow().isoformat(),
                'recommendations': self._generate_clustering_recommendations(geo_analysis)
            }

            # Send to monitoring system
            await self._send_to_monitoring(alert)

            # Update geographic analysis system
            await self._update_geographic_analysis(geo_analysis)

            # Store in history
            self._store_alert(alert)

        except Exception as e:
            logging.error(f"Error in geographic clustering notification: {str(e)}")

    async def _send_to_monitoring(self, alert: Dict):
        """Send alert to monitoring system"""
        try:
            monitoring_endpoint = self.notification_endpoints.get('monitoring')
            if not monitoring_endpoint:
                return

            async with aiohttp.ClientSession() as session:
                async with session.post(monitoring_endpoint, json=alert) as response:
                    if response.status != 200:
                        logging.error(f"Failed to send alert to monitoring: {response.status}")

        except Exception as e:
            logging.error(f"Error sending to monitoring: {str(e)}")

    async def _notify_network_update(self, update: Dict):
        """Notify network about state changes"""
        try:
            network_endpoint = self.notification_endpoints.get('network')
            if not network_endpoint:
                return

            async with aiohttp.ClientSession() as session:
                async with session.post(network_endpoint, json=update) as response:
                    if response.status != 200:
                        logging.error(f"Failed to send network update: {response.status}")

        except Exception as e:
            logging.error(f"Error sending network update: {str(e)}")

    async def _send_to_security_analysis(self, pattern_analysis: Dict):
        """Send data to security analysis system"""
        try:
            security_endpoint = self.notification_endpoints.get('security')
            if not security_endpoint:
                return

            analysis_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'pattern_analysis': pattern_analysis,
                'historical_context': self._get_historical_context(pattern_analysis),
                'risk_assessment': self._assess_security_risk(pattern_analysis)
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(security_endpoint, json=analysis_data) as response:
                    if response.status != 200:
                        logging.error(f"Failed to send security analysis: {response.status}")

        except Exception as e:
            logging.error(f"Error sending security analysis: {str(e)}")

    async def _update_geographic_analysis(self, geo_analysis: Dict):
        """Update geographic analysis system"""
        try:
            geo_endpoint = self.notification_endpoints.get('geographic')
            if not geo_endpoint:
                return

            analysis_update = {
                'timestamp': datetime.utcnow().isoformat(),
                'geo_analysis': geo_analysis,
                'historical_clusters': self._get_historical_clusters(),
                'risk_zones': self._identify_risk_zones(geo_analysis)
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(geo_endpoint, json=analysis_update) as response:
                    if response.status != 200:
                        logging.error(f"Failed to update geographic analysis: {response.status}")

        except Exception as e:
            logging.error(f"Error updating geographic analysis: {str(e)}")

    def _generate_pattern_recommendations(self, patterns: Dict) -> List[str]:
        """Generate recommendations based on detected patterns"""
        recommendations = []

        if any('uniform_results' in p for p in patterns.values()):
            recommendations.append("Increase verification diversity requirements")
            recommendations.append("Implement additional verification challenges")

        if any('timing_pattern' in p for p in patterns.values()):
            recommendations.append("Add random delays to verification requests")
            recommendations.append("Implement timing entropy requirements")

        if any('collusion' in p for p in patterns.values()):
            recommendations.append("Increase minimum stake requirements")
            recommendations.append("Implement stricter node selection criteria")

        return recommendations

    def _generate_clustering_recommendations(self, geo_analysis: Dict) -> List[str]:
        """Generate recommendations for geographic clustering"""
        recommendations = []

        cluster_sizes = [len(nodes) for nodes in geo_analysis.get('clusters', {}).values()]
        max_cluster = max(cluster_sizes) if cluster_sizes else 0

        if max_cluster > self.config.get('max_cluster_size', 5):
            recommendations.append(f"Reduce maximum nodes per region from {max_cluster}")
            recommendations.append("Implement geographic distribution requirements")

        if len(geo_analysis.get('suspicious_regions', [])) > 2:
            recommendations.append("Increase geographic diversity requirements")
            recommendations.append("Adjust node selection to favor geographic distribution")

        return recommendations

    def _get_historical_context(self, pattern_analysis: Dict) -> Dict:
        """Get historical context for pattern analysis"""
        relevant_history = []
        pattern_types = set()

        # Extract pattern types from current analysis
        for patterns in pattern_analysis.get('result_patterns', {}).values():
            pattern_types.update(patterns)

        # Find similar patterns in history
        for alert in self.alert_history:
            if alert['type'] == 'suspicious_pattern':
                if any(pattern in str(alert.get('pattern_details', {})) for pattern in pattern_types):
                    relevant_history.append(alert)

        return {
            'similar_incidents': len(relevant_history),
            'pattern_frequency': self._calculate_pattern_frequency(relevant_history),
            'first_occurrence': min(h['timestamp'] for h in relevant_history) if relevant_history else None
        }

    def _get_historical_clusters(self) -> Dict:
        """Get historical clustering data"""
        cluster_history = []

        for alert in self.alert_history:
            if alert['type'] == 'geographic_clustering':
                cluster_history.append({
                    'timestamp': alert['timestamp'],
                    'regions': alert.get('affected_regions', []),
                    'sizes': alert.get('cluster_sizes', {})
                })

        return {
            'total_incidents': len(cluster_history),
            'recurring_regions': self._find_recurring_regions(cluster_history),
            'trend_analysis': self._analyze_clustering_trends(cluster_history)
        }

    def _identify_risk_zones(self, geo_analysis: Dict) -> Dict:
        """Identify geographic risk zones"""
        risk_zones = {}
        clusters = geo_analysis.get('clusters', {})
        suspicious_regions = set(geo_analysis.get('suspicious_regions', []))

        for region, nodes in clusters.items():
            risk_level = 'high' if region in suspicious_regions else 'medium' if len(nodes) > 3 else 'low'
            risk_zones[region] = {
                'risk_level': risk_level,
                'node_count': len(nodes),
                'historical_incidents': self._count_historical_incidents(region),
                'recommended_actions': self._get_risk_zone_actions(risk_level)
            }

        return risk_zones

    def _store_alert(self, alert: Dict):
        """Store alert in history"""
        self.alert_history.append(alert)
        
        # Prune history if needed
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

    def _calculate_pattern_frequency(self, history: List[Dict]) -> Dict:
        """Calculate frequency of different patterns"""
        pattern_counts = {}
        
        for incident in history:
            patterns = incident.get('pattern_details', {})
            for pattern_type in patterns:
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1

        return pattern_counts

    def _find_recurring_regions(self, cluster_history: List[Dict]) -> Dict:
        """Find regions that frequently appear in clusters"""
        region_frequency = {}
        
        for incident in cluster_history:
            for region in incident.get('regions', []):
                region_frequency[region] = region_frequency.get(region, 0) + 1

        return {
            region: count for region, count in region_frequency.items()
            if count > len(cluster_history) * 0.3  # Appears in >30% of incidents
        }

    def _analyze_clustering_trends(self, cluster_history: List[Dict]) -> Dict:
        """Analyze trends in clustering behavior"""
        if not cluster_history:
            return {}

        # Sort by timestamp
        sorted_history = sorted(cluster_history, key=lambda x: x['timestamp'])
        
        # Calculate trend metrics
        cluster_sizes = [sum(h.get('sizes', {}).values()) for h in sorted_history]
        size_trend = 'increasing' if cluster_sizes[-1] > cluster_sizes[0] else 'decreasing'

        return {
            'size_trend': size_trend,
            'avg_cluster_size': sum(cluster_sizes) / len(cluster_sizes),
            'max_cluster_size': max(cluster_sizes),
            'trend_duration': (sorted_history[-1]['timestamp'] - sorted_history[0]['timestamp']).total_seconds()
        }

    def _count_historical_incidents(self, region: str) -> int:
        """Count historical incidents for a region"""
        return sum(
            1 for alert in self.alert_history
            if alert['type'] == 'geographic_clustering' and
            region in alert.get('affected_regions', [])
        )

    def _get_risk_zone_actions(self, risk_level: str) -> List[str]:
        """Get recommended actions for risk zone"""
        if risk_level == 'high':
            return [
                "Implement strict node limits",
                "Require additional verification",
                "Increase stake requirements"
            ]
        elif risk_level == 'medium':
            return [
                "Monitor node growth",
                "Enhance verification frequency"
            ]
        else:
            return ["Regular monitoring"]