import psutil
import torch
import time
import asyncio
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
import resource

@dataclass
class ScalingMetrics:
    """Metrics for scaling analysis"""
    cpu_usage: float
    memory_usage: float
    gpu_memory: Optional[float]
    network_throughput: float
    verification_time: float
    batch_size: int
    num_verifiers: int

class PerformanceAnalyzer:
    """Analyze system performance and scaling"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.baseline_metrics: Optional[ScalingMetrics] = None
        self.metrics_history: List[ScalingMetrics] = []
        
    async def analyze_cpu_scaling(self, verification_system: Any) -> Dict:
        """Analyze CPU usage scaling"""
        try:
            results = {
                'scaling_factor': [],
                'cpu_usage': [],
                'verification_time': []
            }
            
            # Test different batch sizes
            batch_sizes = [1, 2, 4, 8, 16, 32]
            for batch_size in batch_sizes:
                # Run verification with batch
                metrics = await self._measure_batch_performance(
                    verification_system,
                    batch_size
                )
                
                # Record metrics
                results['scaling_factor'].append(batch_size)
                results['cpu_usage'].append(metrics.cpu_usage)
                results['verification_time'].append(metrics.verification_time)
                
            # Calculate scaling characteristics
            cpu_scaling = self._calculate_scaling_characteristics(
                results['scaling_factor'],
                results['cpu_usage']
            )
            
            time_scaling = self._calculate_scaling_characteristics(
                results['scaling_factor'],
                results['verification_time']
            )
            
            return {
                'raw_metrics': results,
                'cpu_scaling': cpu_scaling,
                'time_scaling': time_scaling,
                'efficiency_score': self._calculate_efficiency_score(
                    cpu_scaling,
                    time_scaling
                )
            }
            
        except Exception as e:
            logging.error(f"Error analyzing CPU scaling: {str(e)}")
            raise
            
    async def analyze_memory_scaling(self, verification_system: Any) -> Dict:
        """Analyze memory usage scaling"""
        try:
            results = {
                'num_verifiers': [],
                'memory_usage': [],
                'gpu_memory': []
            }
            
            # Test different numbers of verifiers
            verifier_counts = [2, 4, 8, 16, 32]
            for num_verifiers in verifier_counts:
                # Run verification with verifiers
                metrics = await self._measure_verifier_performance(
                    verification_system,
                    num_verifiers
                )
                
                # Record metrics
                results['num_verifiers'].append(num_verifiers)
                results['memory_usage'].append(metrics.memory_usage)
                if metrics.gpu_memory is not None:
                    results['gpu_memory'].append(metrics.gpu_memory)
                    
            # Calculate scaling characteristics
            memory_scaling = self._calculate_scaling_characteristics(
                results['num_verifiers'],
                results['memory_usage']
            )
            
            gpu_scaling = None
            if len(results['gpu_memory']) > 0:
                gpu_scaling = self._calculate_scaling_characteristics(
                    results['num_verifiers'],
                    results['gpu_memory']
                )
                
            return {
                'raw_metrics': results,
                'memory_scaling': memory_scaling,
                'gpu_scaling': gpu_scaling,
                'memory_efficiency': self._calculate_memory_efficiency(
                    memory_scaling,
                    gpu_scaling
                )
            }
            
        except Exception as e:
            logging.error(f"Error analyzing memory scaling: {str(e)}")
            raise
            
    async def analyze_network_scaling(self, verification_system: Any) -> Dict:
        """Analyze network throughput scaling"""
        try:
            results = {
                'concurrent_tasks': [],
                'network_throughput': [],
                'response_time': []
            }
            
            # Test different concurrency levels
            concurrency_levels = [1, 2, 4, 8, 16]
            for num_tasks in concurrency_levels:
                # Run concurrent verifications
                metrics = await self._measure_network_performance(
                    verification_system,
                    num_tasks
                )
                
                # Record metrics
                results['concurrent_tasks'].append(num_tasks)
                results['network_throughput'].append(metrics.network_throughput)
                results['response_time'].append(metrics.verification_time)
                
            # Calculate scaling characteristics
            throughput_scaling = self._calculate_scaling_characteristics(
                results['concurrent_tasks'],
                results['network_throughput']
            )
            
            latency_scaling = self._calculate_scaling_characteristics(
                results['concurrent_tasks'],
                results['response_time']
            )
            
            return {
                'raw_metrics': results,
                'throughput_scaling': throughput_scaling,
                'latency_scaling': latency_scaling,
                'network_efficiency': self._calculate_network_efficiency(
                    throughput_scaling,
                    latency_scaling
                )
            }
            
        except Exception as e:
            logging.error(f"Error analyzing network scaling: {str(e)}")
            raise
            
    async def _measure_batch_performance(self,
                                       verification_system: Any,
                                       batch_size: int) -> ScalingMetrics:
        """Measure performance metrics for batch verification"""
        try:
            # Set up monitoring
            cpu_monitor = psutil.Process()
            start_cpu = cpu_monitor.cpu_percent()
            start_memory = self._get_memory_usage()
            start_time = time.time()
            
            # Run batch verification
            verification_tasks = []
            for _ in range(batch_size):
                task = verification_system.verify_batch()
                verification_tasks.append(task)
                
            await asyncio.gather(*verification_tasks)
            
            # Collect metrics
            end_time = time.time()
            end_cpu = cpu_monitor.cpu_percent()
            end_memory = self._get_memory_usage()
            
            return ScalingMetrics(
                cpu_usage=(end_cpu - start_cpu) / batch_size,
                memory_usage=end_memory - start_memory,
                gpu_memory=self._get_gpu_memory_usage(),
                network_throughput=0.0,  # Not measured for CPU scaling
                verification_time=end_time - start_time,
                batch_size=batch_size,
                num_verifiers=1
            )
            
        except Exception as e:
            logging.error(f"Error measuring batch performance: {str(e)}")
            raise
            
    async def _measure_verifier_performance(self,
                                          verification_system: Any,
                                          num_verifiers: int) -> ScalingMetrics:
        """Measure performance metrics for multiple verifiers"""
        try:
            # Set up monitoring
            start_memory = self._get_memory_usage()
            start_gpu = self._get_gpu_memory_usage()
            start_time = time.time()
            
            # Run verifications
            verification_tasks = []
            for _ in range(num_verifiers):
                task = verification_system.verify_with_verifier()
                verification_tasks.append(task)
                
            await asyncio.gather(*verification_tasks)
            
            # Collect metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_gpu = self._get_gpu_memory_usage()
            
            return ScalingMetrics(
                cpu_usage=0.0,  # Not measured for memory scaling
                memory_usage=end_memory - start_memory,
                gpu_memory=end_gpu - start_gpu if end_gpu is not None else None,
                network_throughput=0.0,  # Not measured for memory scaling
                verification_time=end_time - start_time,
                batch_size=1,
                num_verifiers=num_verifiers
            )
            
        except Exception as e:
            logging.error(f"Error measuring verifier performance: {str(e)}")
            raise
            
    async def _measure_network_performance(self,
                                         verification_system: Any,
                                         num_tasks: int) -> ScalingMetrics:
        """Measure network performance metrics"""
        try:
            # Set up monitoring
            start_bytes = self._get_network_bytes()
            start_time = time.time()
            
            # Run concurrent tasks
            tasks = []
            for _ in range(num_tasks):
                task = verification_system.verify_remote()
                tasks.append(task)
                
            await asyncio.gather(*tasks)
            
            # Collect metrics
            end_time = time.time()
            end_bytes = self._get_network_bytes()
            elapsed_time = end_time - start_time
            
            return ScalingMetrics(
                cpu_usage=0.0,  # Not measured for network scaling
                memory_usage=0.0,  # Not measured for network scaling
                gpu_memory=None,
                network_throughput=(end_bytes - start_bytes) / elapsed_time,
                verification_time=elapsed_time,
                batch_size=1,
                num_verifiers=num_tasks
            )
            
        except Exception as e:
            logging.error(f"Error measuring network performance: {str(e)}")
            raise
            
    def _calculate_scaling_characteristics(self,
                                        x_values: List[float],
                                        y_values: List[float]) -> Dict:
        """Calculate scaling characteristics from measurements"""
        try:
            x = np.array(x_values)
            y = np.array(y_values)
            
            # Calculate scaling factor
            log_x = np.log(x)
            log_y = np.log(y)
            slope, intercept = np.polyfit(log_x, log_y, 1)
            
            # Calculate R-squared
            y_pred = np.exp(intercept + slope * log_x)
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            return {
                'scaling_factor': slope,
                'base_value': np.exp(intercept),
                'r_squared': r_squared,
                'is_linear': abs(slope - 1.0) < 0.1,
                'efficiency': 1.0 / abs(slope) if slope != 0 else float('inf')
            }
            
        except Exception as e:
            logging.error(f"Error calculating scaling characteristics: {str(e)}")
            raise
            
    def _calculate_efficiency_score(self,
                                  cpu_scaling: Dict,
                                  time_scaling: Dict) -> float:
        """Calculate overall efficiency score"""
        try:
            # Weight different factors
            weights = {
                'cpu_efficiency': 0.4,
                'time_efficiency': 0.4,
                'scaling_quality': 0.2
            }
            
            # Calculate component scores
            cpu_score = cpu_scaling['efficiency']
            time_score = time_scaling['efficiency']
            quality_score = (cpu_scaling['r_squared'] + time_scaling['r_squared']) / 2
            
            # Combine scores
            return (
                weights['cpu_efficiency'] * cpu_score +
                weights['time_efficiency'] * time_score +
                weights['scaling_quality'] * quality_score
            )
            
        except Exception as e:
            logging.error(f"Error calculating efficiency score: {str(e)}")
            raise
            
    def _calculate_memory_efficiency(self,
                                   memory_scaling: Dict,
                                   gpu_scaling: Optional[Dict]) -> float:
        """Calculate memory efficiency score"""
        try:
            # Base score from CPU memory scaling
            base_score = memory_scaling['efficiency']
            
            # Adjust for GPU memory if available
            if gpu_scaling is not None:
                gpu_score = gpu_scaling['efficiency']
                return (base_score + gpu_score) / 2
                
            return base_score
            
        except Exception as e:
            logging.error(f"Error calculating memory efficiency: {str(e)}")
            raise
            
    def _calculate_network_efficiency(self,
                                    throughput_scaling: Dict,
                                    latency_scaling: Dict) -> float:
        """Calculate network efficiency score"""
        try:
            # Weight different factors
            weights = {
                'throughput': 0.6,
                'latency': 0.4
            }
            
            # Calculate component scores
            throughput_score = throughput_scaling['efficiency']
            latency_score = 1.0 / latency_scaling['scaling_factor']
            
            # Combine scores
            return (
                weights['throughput'] * throughput_score +
                weights['latency'] * latency_score
            )
            
        except Exception as e:
            logging.error(f"Error calculating network efficiency: {str(e)}")
            raise
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
            
        except Exception as e:
            logging.error(f"Error getting memory usage: {str(e)}")
            return 0.0
            
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get GPU memory usage if available"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated()
            return None
            
        except Exception as e:
            logging.error(f"Error getting GPU memory usage: {str(e)}")
            return None
            
    def _get_network_bytes(self) -> int:
        """Get total network bytes transferred"""
        try:
            network_stats = psutil.net_io_counters()
            return network_stats.bytes_sent + network_stats.bytes_recv
            
        except Exception as e:
            logging.error(f"Error getting network bytes: {str(e)}")
            return 0