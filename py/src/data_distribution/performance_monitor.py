class DistributionPerformanceMonitor:
    """Monitor performance of data distribution system"""
    
    def __init__(self, config: Dict, distribution_manager: DataDistributionManager):
        self.config = config
        self.distribution_manager = distribution_manager
        self.metrics_history = {}  # timestamp -> metrics
        self.max_history = config.get('metrics_history_size', 1000)
        
    async def collect_metrics(self) -> Dict:
        """Collect current performance metrics"""
        try:
            current_time = time.time()
            
            # Collect node metrics
            node_metrics = await self._collect_node_metrics()
            
            # Collect distribution metrics
            distribution_metrics = await self._collect_distribution_metrics()
            
            # Collect network metrics
            network_metrics = await self._collect_network_metrics()
            
            # Combine metrics
            metrics = {
                'timestamp': current_time,
                'node_metrics': node_metrics,
                'distribution_metrics': distribution_metrics,
                'network_metrics': network_metrics
            }
            
            # Store history
            self.metrics_history[current_time] = metrics
            
            # Prune old metrics
            self._prune_history()
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error collecting performance metrics: {str(e)}")
            return {
                'error': str(e),
                'timestamp': time.time()
            }
            
    async def _collect_node_metrics(self) -> Dict:
        """Collect node-related metrics"""
        try:
            # Get node information
            nodes = self.distribution_manager.nodes
            
            # Calculate stats
            return {
                'active_nodes': len(nodes),
                'total_capacity': sum(node.capacity for node in nodes.values()),
                'avg_utilization': self._calculate_node_utilization(nodes),
                'node_distribution': self._calculate_node_distribution(nodes)
            }
            
        except Exception as e:
            logging.error(f"Error collecting node metrics: {str(e)}")
            return {}
            
    async def _collect_distribution_metrics(self) -> Dict:
        """Collect shard distribution metrics"""
        try:
            # Get shard statuses
            statuses = self.distribution_manager.shard_status
            
            # Calculate transfer stats
            successful_transfers = 0
            failed_transfers = 0
            pending_transfers = 0
            
            for status in statuses.values():
                for _, transfer_status in status.transfer_status.items():
                    if transfer_status == "completed":
                        successful_transfers += 1
                    elif transfer_status == "failed":
                        failed_transfers += 1
                    else:
                        pending_transfers += 1
                        
            # Calculate replication stats
            dataset_replication = {}
            for status in statuses.values():
                dataset_id = status.dataset_id
                if dataset_id not in dataset_replication:
                    dataset_replication[dataset_id] = {
                        'total_shards': 0,
                        'replicated_shards': 0
                    }
                    
                dataset_replication[dataset_id]['total_shards'] += 1
                
                # Count successful replications
                replications = sum(1 for _, transfer_status in status.transfer_status.items() 
                                 if transfer_status == "completed")
                                 
                if replications >= status.replication_factor:
                    dataset_replication[dataset_id]['replicated_shards'] += 1
                    
            return {
                'total_shards': len(statuses),
                'transfers': {
                    'successful': successful_transfers,
                    'failed': failed_transfers,
                    'pending': pending_transfers
                },
                'replication': {
                    'datasets': dataset_replication,
                    'overall_health': self._calculate_replication_health(statuses)
                }
            }
            
        except Exception as e:
            logging.error(f"Error collecting distribution metrics: {str(e)}")
            return {}
            
    async def _collect_network_metrics(self) -> Dict:
        """Collect network performance metrics"""
        try:
            # Use system tools to get network stats
            # This is platform-dependent
            
            # For Linux, could use /proc/net/dev
            # For cross-platform, could use psutil
            
            # Example with psutil:
            import psutil
            
            # Get current network stats
            net_io = psutil.net_io_counters()
            
            # Get historical values for comparison
            prev_metrics = self._get_previous_metrics()
            prev_net_io = prev_metrics.get('network_metrics', {}).get('raw_counters', {})
            
            # Calculate rates
            current_time = time.time()
            prev_time = prev_metrics.get('timestamp', current_time - 60)
            time_diff = max(0.1, current_time - prev_time)
            
            bytes_sent_rate = (net_io.bytes_sent - prev_net_io.get('bytes_sent', 0)) / time_diff
            bytes_recv_rate = (net_io.bytes_recv - prev_net_io.get('bytes_recv', 0)) / time_diff
            
            return {
                'throughput': {
                    'send_rate_bytes': bytes_sent_rate,
                    'recv_rate_bytes': bytes_recv_rate,
                    'total_rate_bytes': bytes_sent_rate + bytes_recv_rate
                },
                'raw_counters': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errin': net_io.errin,
                    'errout': net_io.errout,
                    'dropin': net_io.dropin,
                    'dropout': net_io.dropout
                }
            }
            
        except Exception as e:
            logging.error(f"Error collecting network metrics: {str(e)}")
            return {}
            
    def _get_previous_metrics(self) -> Dict:
        """Get previous metrics for comparison"""
        if not self.metrics_history:
            return {}
            
        # Get latest metrics
        latest_time = max(self.metrics_history.keys())
        return self.metrics_history[latest_time]
        
    def _prune_history(self):
        """Prune old metrics history"""
        if len(self.metrics_history) <= self.max_history:
            return
            
        # Sort by timestamp
        sorted_times = sorted(self.metrics_history.keys())
        
        # Remove oldest
        for time_key in sorted_times[:-self.max_history]:
            del self.metrics_history[time_key]
            
    def _calculate_node_utilization(self, nodes: Dict) -> float:
        """Calculate average node utilization"""
        if not nodes:
            return 0.0
            
        utilization = []
        for node in nodes.values():
            # Calculate utilization as assignments / capacity
            util = len(node.shard_assignments) / max(1.0, node.capacity)
            utilization.append(util)
            
        return sum(utilization) / len(utilization) if utilization else 0.0
        
    def _calculate_node_distribution(self, nodes: Dict) -> Dict:
        """Calculate distribution of shards across nodes"""
        result = {
            'min_shards': float('inf'),
            'max_shards': 0,
            'avg_shards': 0.0,
            'nodes_by_shard_count': {}
        }
        
        if not nodes:
            result['min_shards'] = 0
            return result
            
        # Count shards per node
        shard_counts = {}
        for node_id, node in nodes.items():
            count = len(node.shard_assignments)
            shard_counts[node_id] = count
            
            result['min_shards'] = min(result['min_shards'], count)
            result['max_shards'] = max(result['max_shards'], count)
            
            # Group by count
            count_key = str(count)
            if count_key not in result['nodes_by_shard_count']:
                result['nodes_by_shard_count'][count_key] = 0
                
            result['nodes_by_shard_count'][count_key] += 1
            
        # Calculate average
        result['avg_shards'] = sum(shard_counts.values()) / len(shard_counts)
        
        return result
        
    def _calculate_replication_health(self, statuses: Dict) -> float:
        """Calculate overall replication health (0.0-1.0)"""
        if not statuses:
            return 1.0  # Nothing to replicate
            
        # Count total required and successful replications
        total_required = 0
        total_successful = 0
        
        for status in statuses.values():
            # Each shard should have replication_factor copies
            total_required += status.replication_factor
            
            # Count successful replications
            successful = sum(1 for _, transfer_status in status.transfer_status.items() 
                           if transfer_status == "completed")
                           
            total_successful += successful
            
        # Calculate health
        return total_successful / total_required if total_required > 0 else 1.0