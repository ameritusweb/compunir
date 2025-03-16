from typing import Dict, Tuple
import logging
import torch
import numpy as np
import asyncio
import time
from ..utils.gpu_monitoring import GPUMonitor
from ..verification.integrated_verification import IntegratedVerificationManager
from ..payment.payment_processor import PaymentProcessor
from ..network.client import NodeNetworkClient

class JobExecutor:
    def __init__(self, config: Dict, 
                 network_client: NodeNetworkClient,
                 payment_processor: PaymentProcessor,
                 verification_manager: IntegratedVerificationManager):
        self.config = config
        self.network_client = network_client
        self.payment_processor = payment_processor
        self.verification_manager = verification_manager
        self.active_jobs: Dict[str, Dict] = {}

    async def execute_job(self, job_id: str, job_spec: Dict):
        """Execute training job with integrated verification"""
        try:
            # Initialize job context
            self.active_jobs[job_id] = {
                'status': 'initializing',
                'current_epoch': 0,
                'metrics': [],
                'verifications': []
            }

            # Setup training environment
            model, optimizer, dataloader = await self._setup_training(job_spec)

            # Process epochs
            num_epochs = job_spec['training_config']['epochs']
            for epoch in range(num_epochs):
                try:
                    # Update job status
                    self.active_jobs[job_id]['status'] = 'training'
                    self.active_jobs[job_id]['current_epoch'] = epoch
                    
                    # Train one epoch
                    epoch_metrics = await self._train_epoch(
                        model=model,
                        optimizer=optimizer,
                        dataloader=dataloader,
                        epoch=epoch,
                        job_spec=job_spec
                    )
                    
                    # Store metrics
                    self.active_jobs[job_id]['metrics'].append(epoch_metrics)
                    
                    # Check if verification is needed
                    if self._should_verify_checkpoint(epoch, job_spec):
                        verification_result = await self._verify_checkpoint(
                            job_id=job_id,
                            model=model,
                            epoch=epoch,
                            metrics=epoch_metrics
                        )
                        
                        # Handle verification result
                        await self._handle_verification_result(
                            job_id=job_id,
                            verification_result=verification_result,
                            epoch=epoch
                        )
                    
                    # Report progress
                    await self._report_progress(job_id, epoch, epoch_metrics)
                    
                except Exception as e:
                    logging.error(f"Error in epoch {epoch}: {str(e)}")
                    await self._handle_epoch_failure(job_id, epoch, str(e))
                    raise
            
            # Complete job
            await self._complete_job(job_id)
            
        except Exception as e:
            logging.error(f"Error executing job {job_id}: {str(e)}")
            await self._handle_job_failure(job_id, str(e))
            raise
        finally:
            # Cleanup
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

    async def _train_epoch(self,
                          model: torch.nn.Module,
                          optimizer: torch.optim.Optimizer,
                          dataloader: torch.utils.data.DataLoader,
                          epoch: int,
                          job_spec: Dict) -> Dict:
        """Train one epoch with metric collection"""
        try:
            epoch_metrics = {
                'loss': 0.0,
                'accuracy': 0.0,
                'batches': 0,
                'gpu_metrics': [],
                'verification_points': []
            }
            
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                # Move to GPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = self._calculate_loss(outputs, targets, job_spec)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Collect metrics
                batch_metrics = {
                    'loss': loss.item(),
                    'accuracy': self._calculate_accuracy(outputs, targets),
                    'gpu_utilization': self._get_gpu_utilization(),
                    'batch_time': time.time()
                }
                
                # Update epoch metrics
                epoch_metrics['loss'] += batch_metrics['loss']
                epoch_metrics['accuracy'] += batch_metrics['accuracy']
                epoch_metrics['batches'] += 1
                epoch_metrics['gpu_metrics'].append(batch_metrics)
                
                # Store verification point if needed
                if self._should_store_verification_point(batch_idx, job_spec):
                    verification_point = {
                        'batch_idx': batch_idx,
                        'inputs': inputs.detach(),
                        'outputs': outputs.detach(),
                        'loss': loss.item(),
                        'metrics': batch_metrics
                    }
                    epoch_metrics['verification_points'].append(verification_point)
            
            # Calculate epoch averages
            epoch_metrics['loss'] /= epoch_metrics['batches']
            epoch_metrics['accuracy'] /= epoch_metrics['batches']
            
            return epoch_metrics
            
        except Exception as e:
            logging.error(f"Error training epoch: {str(e)}")
            raise

    async def _verify_checkpoint(self,
                               job_id: str,
                               model: torch.nn.Module,
                               epoch: int,
                               metrics: Dict) -> Dict:
        """Verify training checkpoint"""
        try:
            # Get latest verification point
            if not metrics['verification_points']:
                raise ValueError("No verification points available")
            
            verification_point = metrics['verification_points'][-1]
            
            # Initialize verification
            verification_task_id = await self.verification_manager.initialize_verification(
                job_id=job_id,
                model=model,
                inputs=verification_point['inputs'],
                outputs=verification_point['outputs'],
                metrics=verification_point['metrics'],
                checkpoint_id=epoch
            )
            
            # Wait for verification completion
            while True:
                status = await self.verification_manager.get_verification_status(
                    verification_task_id
                )
                
                if status['status'] in ['completed', 'timeout', 'error']:
                    return status
                    
                await asyncio.sleep(1)
            
        except Exception as e:
            logging.error(f"Error verifying checkpoint: {str(e)}")
            raise

    async def _handle_verification_result(self,
                                        job_id: str,
                                        verification_result: Dict,
                                        epoch: int):
        """Handle verification result"""
        try:
            if verification_result['status'] == 'completed' and verification_result['is_valid']:
                # Process payments
                await self._process_success_payments(job_id, epoch)
                
                # Update job status
                self.active_jobs[job_id]['verifications'].append({
                    'epoch': epoch,
                    'result': verification_result,
                    'status': 'success'
                })
                
            elif verification_result['status'] == 'timeout':
                # Handle timeout
                await self._handle_verification_timeout(job_id, epoch)
                
            else:
                # Handle verification failure
                await self._handle_verification_failure(job_id, epoch, verification_result)
            
        except Exception as e:
            logging.error(f"Error handling verification result: {str(e)}")
            raise

    def _should_verify_checkpoint(self, epoch: int, job_spec: Dict) -> bool:
        """Determine if epoch should be verified"""
        verification_config = job_spec.get('verification_config', {})
        
        # Check verification frequency
        frequency = verification_config.get('frequency', 5)
        if epoch % frequency == 0:
            return True
            
        # Check for specific epochs
        required_epochs = verification_config.get('required_epochs', [])
        if epoch in required_epochs:
            return True
            
        return False

    def _should_store_verification_point(self, batch_idx: int, job_spec: Dict) -> bool:
        """Determine if batch should be stored for verification"""
        verification_config = job_spec.get('verification_config', {})
        
        # Store last batch by default
        points_per_epoch = verification_config.get('points_per_epoch', 1)
        batch_interval = max(1, len(self.dataloader) // points_per_epoch)
        
        return batch_idx % batch_interval == 0

    async def _setup_training(self, job_spec: Dict) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader]:
        """Setup training environment"""
        # Implementation would create model, optimizer, and dataloader
        pass

    async def _report_progress(self, job_id: str, epoch: int, metrics: Dict):
        """Report training progress"""
        try:
            progress = {
                'job_id': job_id,
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': time.time()
            }
            
            await self.network_client.report_progress(progress)
            
        except Exception as e:
            logging.error(f"Error reporting progress: {str(e)}")
            raise

    async def _process_success_payments(self, job_id: str, epoch: int):
        """Process payments for successful verification"""
        try:
            # Calculate payment amount
            payment_amount = self._calculate_epoch_payment(job_id, epoch)
            
            # Process payment
            await self.payment_processor.process_training_payment(
                job_id=job_id,
                epoch=epoch,
                amount=payment_amount
            )
            
        except Exception as e:
            logging.error(f"Error processing payments: {str(e)}")
            raise

    async def _handle_verification_timeout(self, job_id: str, epoch: int):
        """Handle verification timeout"""
        try:
            # Update job status
            self.active_jobs[job_id]['verifications'].append({
                'epoch': epoch,
                'status': 'timeout',
                'timestamp': time.time()
            })
            
            # Retry verification if configured
            retry_count = self.config.get('verification_retry_count', 2)
            if len(self._get_timeout_verifications(job_id, epoch)) < retry_count:
                await self._retry_verification(job_id, epoch)
            
        except Exception as e:
            logging.error(f"Error handling verification timeout: {str(e)}")
            raise

    async def _handle_verification_failure(self, job_id: str, epoch: int, result: Dict):
        """Handle verification failure with adaptive response."""
        try:
            self.active_jobs[job_id]['verifications'].append({
                'epoch': epoch,
                'status': 'failed',
                'result': result,
                'timestamp': time.time()
            })

            failure_threshold = self.config.get('max_verification_failures', 3)
            failed_attempts = sum(1 for v in self.active_jobs[job_id]['verifications'] if v['status'] == 'failed')

            if failed_attempts >= failure_threshold:
                logging.warning(f"Job {job_id} verification failed too many times. Penalizing node.")
                await self.network_client.report_verification_penalty(job_id)
            else:
                logging.info(f"Verification failed for epoch {epoch}, but job will continue.")
        
        except Exception as e:
            logging.error(f"Error handling verification failure: {str(e)}")


    def _calculate_epoch_payment(self, job_id: str, epoch: int) -> float:
        """Calculate payment based on training efficiency."""
        base_rate = self.config.get('base_payment_rate', 0.001)
        
        metrics = self.active_jobs[job_id]['metrics'][epoch]

        time_factor = self._calculate_time_factor(metrics)
        performance_factor = self._calculate_performance_factor(metrics)
        verification_bonus = 1.2 if self._is_strong_verification(metrics) else 1.0  # 20% bonus if high-quality verification

        return base_rate * time_factor * performance_factor * verification_bonus

    def _is_strong_verification(self, metrics: Dict) -> bool:
        """Determine if verification results were high-quality."""
        if 'verification_points' not in metrics or not metrics['verification_points']:
            return False

        avg_loss = np.mean([v['loss'] for v in metrics['verification_points']])
        return avg_loss < 0.05  # Example threshold

    def _calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor, job_spec: Dict) -> float:
        """Calculate accuracy for classification or regression tasks."""
        task_type = job_spec['training_config'].get('task_type', 'classification')

        if task_type == 'classification':
            with torch.no_grad():
                predicted = torch.argmax(outputs, dim=1)
                correct = (predicted == targets).sum().item()
                total = targets.size(0)
            return correct / total if total > 0 else 0.0

        elif task_type == 'regression':
            # Mean Squared Error-based accuracy (inverse of error)
            with torch.no_grad():
                error = torch.mean(torch.abs(outputs - targets))
            return max(0.0, 1.0 - error.item())  # Normalize to [0,1]

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _calculate_time_factor(self, metrics: Dict) -> float:
        """Calculate time-based payment factor"""
        try:
            # Calculate average batch time
            batch_times = [m['batch_time'] for m in metrics['gpu_metrics']]
            avg_batch_time = np.mean(np.diff(batch_times))
            
            # Compare with target time
            target_time = self.config.get('target_batch_time', 0.1)
            return min(1.0, target_time / avg_batch_time)
            
        except Exception as e:
            logging.error(f"Error calculating time factor: {str(e)}")
            return 1.0

    def _calculate_performance_factor(self, metrics: Dict) -> float:
        """Calculate performance-based payment factor"""
        try:
            # Calculate GPU utilization
            gpu_utils = [m['gpu_utilization'] for m in metrics['gpu_metrics']]
            avg_utilization = np.mean(gpu_utils)
            
            # Scale factor based on utilization
            return min(1.0, avg_utilization / 90.0)  # Target 90% utilization
            
        except Exception as e:
            logging.error(f"Error calculating performance factor: {str(e)}")
            return 1.0

    def _check_failure_threshold(self, job_id: str) -> bool:
        """Check if verification failures exceed threshold"""
        try:
            failures = [v for v in self.active_jobs[job_id]['verifications']
                       if v['status'] == 'failed']
            max_failures = self.config.get('max_verification_failures', 3)
            
            return len(failures) >= max_failures
            
        except Exception as e:
            logging.error(f"Error checking failure threshold: {str(e)}")
            return False
        
    def _get_gpu_utilization(self) -> float:
        """Fetch GPU utilization percentage."""
        gpu_stats = self.gpu_monitor.get_current_stats()
        if gpu_stats:
            return gpu_stats[0].utilization  # Assuming first GPU; update for multi-GPU
        return 0.0

    def _calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor, job_spec: Dict) -> torch.Tensor:
        """Calculate loss function based on job specification."""
        loss_type = job_spec['training_config'].get('loss_function', 'cross_entropy')

        if loss_type == 'cross_entropy':
            loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_type == 'mse':
            loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

        return loss_fn(outputs, targets)
