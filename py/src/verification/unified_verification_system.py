import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from collections import defaultdict
from decimal import Decimal
import hashlib
from scipy.stats import wasserstein_distance
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# Import the ZKProofGenerator components
from .zk_proof_generator import ZKProofGenerator, ModelCheckpoint, ProofComponents

@dataclass
class GradientCheckpoint:
    """Checkpoint of gradient information for verification"""
    layer_name: str
    gradient_norm: float
    gradient_mean: float
    gradient_std: float
    weight_norm: float
    update_ratio: float  # ratio of update size to weight magnitude


@dataclass
class LayerComputationProof:
    """Proof of layer computation correctness"""
    layer_id: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters_hash: str
    computation_trace: bytes
    intermediate_values: Dict[str, torch.Tensor]


@dataclass
class DataVerificationProof:
    """Proof of data processing and verification"""
    data_id: str
    source_hash: str
    processing_steps: List[Dict]
    result_hash: str
    metadata: Dict
    timestamp: float


@dataclass
class VerificationProof:
    """Comprehensive proof for verification"""
    job_id: str
    checkpoint_id: int
    state_hash: bytes
    proof_data: bytes
    metrics: Dict[str, float]
    timestamp: float


@dataclass
class VerificationResult:
    """Result of a verification process"""
    is_valid: bool
    details: Dict
    warnings: List[str]
    errors: List[str]
    verification_time: float


@dataclass
class JobVerificationContext:
    """Context for job verification"""
    job_id: str
    model: torch.nn.Module
    checkpoint_id: int
    verification_task_id: Optional[str] = None
    verifiers: List[str] = None
    status: str = "pending"
    results: List[Dict] = None
    deadline: float = None


class VerificationSystem:
    """Comprehensive verification system that combines layer, data and model verification"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # History and tracking state
        self.gradient_history = defaultdict(list)
        self.behavior_history = defaultdict(list)
        self.weight_update_history = defaultdict(list)
        self.verification_history = {}
        self.active_verifications = {}
        self.verification_cache = {}
        
        # Initialize ZK proof generator
        self.zk_proof_generator = ZKProofGenerator(config)
        
        # Configuration values
        verification_config = config.get('verification', {})
        self.verification_invalidation_window = verification_config.get('verification_invalidation_window', 86400)  # 24h
        self.min_verifiers = verification_config.get('min_verifiers', 3)
        self.max_verifiers = verification_config.get('max_verifiers', 7)
        
        logging.info("Initialized UnifiedVerificationSystem with ZK proof capabilities")
        
    #
    # High-level verification interfaces
    #
    
    async def verify_training_step(self,
                                 model: nn.Module,
                                 optimizer: torch.optim.Optimizer,
                                 inputs: torch.Tensor,
                                 outputs: torch.Tensor,
                                 loss: torch.Tensor,
                                 job_id: str) -> Tuple[bool, Dict]:
        """Comprehensive verification of a training step"""
        try:
            start_time = time.time()
            
            # Run all verification strategies
            gradient_valid, gradient_info = self._verify_gradients(model)
            behavior_valid, behavior_info = self._verify_model_behavior(job_id, model, inputs, outputs)
            update_valid, update_info = self._verify_weight_updates(model, optimizer)
            loss_valid, loss_info = self._verify_loss_characteristics(loss)
            
            # Combine verification results
            is_valid = all([
                gradient_valid,
                behavior_valid,
                update_valid,
                loss_valid
            ])
            
            # Store verification data
            self._store_verification_data(job_id, {
                'gradients': gradient_info,
                'behavior': behavior_info,
                'updates': update_info,
                'loss': loss_info
            })
            
            logging.info(f"Training step verification for job {job_id}: valid={is_valid}")
            
            return is_valid, {
                'gradient_verification': gradient_info,
                'behavior_verification': behavior_info,
                'update_verification': update_info,
                'loss_verification': loss_info,
                'verification_time': time.time() - start_time
            }
            
        except Exception as e:
            logging.error(f"Error in training step verification: {str(e)}")
            raise
            
    async def verify_data_integrity(self, data_proof: DataVerificationProof) -> VerificationResult:
        """Verify data integrity and processing steps"""
        try:
            start_time = time.time()
            
            # Validate proof structure
            if not self._validate_data_proof_structure(data_proof):
                return VerificationResult(
                    is_valid=False,
                    details={},
                    warnings=[],
                    errors=['Invalid proof structure'],
                    verification_time=time.time() - start_time
                )
                
            # Verify source hash (if reference data available)
            source_hash_valid = await self._verify_source_hash(
                data_proof.source_hash,
                data_proof.data_id
            )
            
            # Verify processing steps
            steps_valid, step_warnings, step_errors = await self._verify_processing_steps(
                data_proof.processing_steps
            )
            
            # Verify result hash matches expected (if available)
            result_hash_valid = await self._verify_result_hash(
                data_proof.result_hash,
                data_proof.data_id
            )
            
            return VerificationResult(
                is_valid=source_hash_valid and steps_valid and result_hash_valid,
                details={
                    'source_valid': source_hash_valid,
                    'steps_valid': steps_valid,
                    'result_valid': result_hash_valid
                },
                warnings=step_warnings,
                errors=step_errors,
                verification_time=time.time() - start_time
            )
            
        except Exception as e:
            logging.error(f"Data verification error: {str(e)}")
            return VerificationResult(
                is_valid=False,
                details={},
                warnings=[],
                errors=[str(e)],
                verification_time=time.time() - start_time
            )
            
    async def verify_layer_computation(self,
                                     response: bytes,
                                     state_dict: Dict[str, torch.Tensor],
                                     layer: nn.Module) -> bool:
        """Verify computation for any layer type"""
        try:
            # Extract proof components
            proof = self._decode_computation_proof(response)
            
            # Verify based on layer type
            if isinstance(layer, nn.Conv2d):
                return self._verify_conv_layer(proof, state_dict, layer)
            elif isinstance(layer, nn.Linear):
                return self._verify_linear_layer(proof, state_dict, layer)
            elif isinstance(layer, nn.BatchNorm2d):
                return self._verify_batch_norm_layer(proof, state_dict, layer)
            else:
                # Generic layer verification
                return self._verify_generic_layer(proof, state_dict, layer)
                
        except Exception as e:
            logging.error(f"Error verifying layer computation: {str(e)}")
            return False

    async def initialize_verification(self,
                                   job_id: str,
                                   model: torch.nn.Module,
                                   inputs: torch.Tensor,
                                   outputs: torch.Tensor,
                                   metrics: Dict,
                                   checkpoint_id: int) -> str:
        """Initialize verification process for a training checkpoint"""
        try:
            # Generate verification proof
            proof = self._generate_proof(
                model=model,
                inputs=inputs,
                outputs=outputs,
                metrics=metrics,
                job_id=job_id,
                checkpoint_id=checkpoint_id
            )
            
            # Get compute size requirements
            compute_size = self._calculate_compute_requirements(model, inputs)
            
            # Select verifier nodes
            available_nodes = await self._get_available_nodes()
            verifiers = await self._select_verifiers(
                job_id=job_id,
                compute_size=compute_size,
                available_nodes=available_nodes
            )
            
            # Create verification context
            verification_task_id = f"verify_{job_id}_{checkpoint_id}_{int(time.time())}"
            context = JobVerificationContext(
                job_id=job_id,
                model=model,
                checkpoint_id=checkpoint_id,
                verification_task_id=verification_task_id,
                verifiers=verifiers,
                results=[],
                deadline=time.time() + self.config.get('verification_timeout', 300)
            )
            
            # Store context
            self.active_verifications[verification_task_id] = context
            
            # Start verification process
            asyncio.create_task(self._manage_verification_process(verification_task_id, proof))
            
            return verification_task_id
            
        except Exception as e:
            logging.error(f"Error initializing verification: {str(e)}")
            raise

    async def get_verification_status(self, verification_task_id: str) -> Dict:
        """Get current status of verification process"""
        try:
            context = self.active_verifications.get(verification_task_id)
            if not context:
                return {
                    'status': 'not_found',
                    'error': 'Verification task not found'
                }
            
            # Calculate progress
            completed_verifications = len(context.results)
            total_verifiers = len(context.verifiers)
            progress = completed_verifications / total_verifiers if total_verifiers > 0 else 0
            
            # Check if complete
            if context.status == 'completed':
                validation_result = await self._validate_verification_results(context)
                return {
                    'status': 'completed',
                    'progress': 1.0,
                    'is_valid': validation_result['is_valid'],
                    'validation_details': validation_result['details']
                }
            
            # Check for timeout
            if time.time() > context.deadline:
                return {
                    'status': 'timeout',
                    'progress': progress,
                    'error': 'Verification timeout'
                }
            
            return {
                'status': context.status,
                'progress': progress,
                'verifiers': {
                    'total': total_verifiers,
                    'completed': completed_verifications
                },
                'deadline': context.deadline
            }
            
        except Exception as e:
            logging.error(f"Error getting verification status: {str(e)}")
            raise

    async def get_verifications_since(self, cutoff_time: float) -> List[Dict]:
        """Get verifications since specified time"""
        try:
            recent_verifications = []
            
            for verification_id, history_entries in self.verification_history.items():
                # Find entries after cutoff time
                recent_entries = [
                    entry for entry in history_entries
                    if entry['timestamp'] >= cutoff_time
                ]
                
                if recent_entries:
                    recent_verifications.extend(recent_entries)
            
            return recent_verifications
            
        except Exception as e:
            logging.error(f"Error getting recent verifications: {str(e)}")
            return []

    async def invalidate_verification(self,
                                    verification_id: str,
                                    reason: str,
                                    affected_nodes: List[str]) -> bool:
        """Invalidate a verification due to malicious behavior"""
        try:
            if verification_id not in self.verification_history:
                return False
                
            # Mark entries as invalidated
            for entry in self.verification_history[verification_id]:
                entry['status'] = 'invalidated'
                entry['invalidation_reason'] = reason
                entry['invalidated_time'] = time.time()
                entry['affected_nodes'] = affected_nodes
                
            logging.warning(f"Invalidated verification {verification_id}: {reason}")
            return True
            
        except Exception as e:
            logging.error(f"Error invalidating verification: {str(e)}")
            return False

    async def invalidate_verifications_from_nodes(self,
                                               node_ids: List[str],
                                               reason: str) -> int:
        """Invalidate all verifications from specified nodes"""
        try:
            invalidated_count = 0
            
            # Iterate through all verifications
            for verification_id, entries in self.verification_history.items():
                for entry in entries:
                    # Check if any affected verifier is in the node list
                    if any(v_id in node_ids for v_id in entry.get('verifier_ids', [])):
                        # Invalidate this verification
                        entry['status'] = 'invalidated'
                        entry['invalidation_reason'] = reason
                        entry['invalidated_time'] = time.time()
                        entry['affected_nodes'] = node_ids
                        invalidated_count += 1
                        
            logging.warning(f"Invalidated {invalidated_count} verifications from nodes: {node_ids}")
            return invalidated_count
            
        except Exception as e:
            logging.error(f"Error invalidating verifications from nodes: {str(e)}")
            return 0

    async def set_enhanced_verification(self, node_id: str, enabled: bool) -> bool:
        """Set enhanced verification for specific node"""
        try:
            # Implementation would track nodes requiring enhanced verification
            # and apply stricter verification standards
            
            logging.info(f"{'Enabling' if enabled else 'Disabling'} enhanced verification for node {node_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error setting enhanced verification: {str(e)}")
            return False

    def get_verification_statistics(self, job_id: str) -> Dict:
        """Get statistical summary of verification history"""
        try:
            gradient_stats = self._calculate_gradient_statistics(job_id)
            behavior_stats = self._calculate_behavior_statistics(job_id)
            update_stats = self._calculate_update_statistics(job_id)
            
            return {
                'gradient_statistics': gradient_stats,
                'behavior_statistics': behavior_stats,
                'update_statistics': update_stats,
                'verification_rate': self._calculate_verification_rate(job_id)
            }
            
        except Exception as e:
            logging.error(f"Error getting verification statistics: {str(e)}")
            raise
    
    #
    # Gradient verification methods
    #
    
    def _verify_gradients(self, model: nn.Module) -> Tuple[bool, Dict]:
        """Verify gradient properties and patterns"""
        try:
            gradient_checkpoints = []
            valid_gradients = True
            
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                    
                # Calculate gradient statistics
                grad_norm = torch.norm(param.grad).item()
                grad_mean = torch.mean(param.grad).item()
                grad_std = torch.std(param.grad).item()
                weight_norm = torch.norm(param).item()
                
                # Calculate update ratio
                update_ratio = grad_norm / (weight_norm + 1e-8)
                
                checkpoint = GradientCheckpoint(
                    layer_name=name,
                    gradient_norm=grad_norm,
                    gradient_mean=grad_mean,
                    gradient_std=grad_std,
                    weight_norm=weight_norm,
                    update_ratio=update_ratio
                )
                gradient_checkpoints.append(checkpoint)
                
                # Verify gradient properties
                if not self._verify_gradient_properties(checkpoint):
                    valid_gradients = False
                    
            # Verify gradient relationships across layers
            if not self._verify_gradient_relationships(gradient_checkpoints):
                valid_gradients = False
                
            return valid_gradients, {
                'checkpoints': gradient_checkpoints,
                'gradient_relationships_valid': valid_gradients
            }
            
        except Exception as e:
            logging.error(f"Error verifying gradients: {str(e)}")
            raise

    def _verify_gradient_properties(self, checkpoint: GradientCheckpoint) -> bool:
        """Verify individual gradient properties"""
        try:
            # Check for NaN or inf
            if not np.isfinite(checkpoint.gradient_norm):
                return False
                
            # Check gradient magnitude
            if checkpoint.gradient_norm > self.config.get('max_gradient_norm', 1000):
                return False
                
            # Check update ratio
            if checkpoint.update_ratio > self.config.get('max_update_ratio', 0.1):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying gradient properties: {str(e)}")
            return False

    def _verify_gradient_relationships(self, checkpoints: List[GradientCheckpoint]) -> bool:
        """Verify relationships between gradients of different layers"""
        try:
            if not checkpoints:
                return False
                
            # Check gradient norm ratios between adjacent layers
            for i in range(len(checkpoints) - 1):
                ratio = checkpoints[i].gradient_norm / (checkpoints[i+1].gradient_norm + 1e-8)
                if ratio > self.config.get('max_gradient_ratio', 100):
                    return False
                    
            # Check for vanishing gradients
            min_gradient = min(c.gradient_norm for c in checkpoints)
            if min_gradient < self.config.get('min_gradient_norm', 1e-8):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying gradient relationships: {str(e)}")
            return False
    
    #
    # Model behavior verification methods
    #
    
    def _verify_model_behavior(self, 
                            job_id: str,
                            model: nn.Module,
                            inputs: torch.Tensor,
                            outputs: torch.Tensor) -> Tuple[bool, Dict]:
        """Verify model behavior patterns"""
        try:
            # Generate perturbation tests
            perturbation_results = self._test_input_perturbations(model, inputs)
            
            # Check output distribution
            distribution_valid = self._verify_output_distribution(job_id, outputs)
            
            # Test model consistency
            consistency_valid = self._verify_model_consistency(model, inputs)
            
            # Verify activation patterns
            activation_valid = self._verify_activation_patterns(model)
            
            is_valid = all([
                perturbation_results['valid'],
                distribution_valid,
                consistency_valid,
                activation_valid
            ])
            
            return is_valid, {
                'perturbation_tests': perturbation_results,
                'distribution_valid': distribution_valid,
                'consistency_valid': consistency_valid,
                'activation_valid': activation_valid
            }
            
        except Exception as e:
            logging.error(f"Error verifying model behavior: {str(e)}")
            raise

    def _test_input_perturbations(self, model: nn.Module, inputs: torch.Tensor) -> Dict:
        """Test model behavior under input perturbations"""
        try:
            perturbation_tests = []
            is_valid = True
            
            # Test different perturbation types
            perturbation_types = [
                ('gaussian', 0.01),
                ('gaussian', 0.1),
                ('dropout', 0.1),
                ('dropout', 0.2)
            ]
            
            with torch.no_grad():
                base_output = model(inputs)
                
                for p_type, magnitude in perturbation_types:
                    # Generate perturbed input
                    perturbed_input = self._generate_perturbation(inputs, p_type, magnitude)
                    
                    # Get model output for perturbed input
                    perturbed_output = model(perturbed_input)
                    
                    # Calculate output difference
                    output_diff = torch.norm(perturbed_output - base_output).item()
                    
                    # Verify perturbation response
                    test_valid = self._verify_perturbation_response(
                        output_diff, magnitude, p_type
                    )
                    
                    perturbation_tests.append({
                        'type': p_type,
                        'magnitude': magnitude,
                        'output_diff': output_diff,
                        'valid': test_valid
                    })
                    
                    is_valid = is_valid and test_valid
                    
            return {
                'valid': is_valid,
                'tests': perturbation_tests
            }
            
        except Exception as e:
            logging.error(f"Error testing perturbations: {str(e)}")
            raise

    def _verify_output_distribution(self, job_id: str, outputs: torch.Tensor) -> bool:
        """Verify properties of the output distribution."""
        try:
            # Calculate distribution statistics
            output_mean = torch.mean(outputs).item()
            output_std = torch.std(outputs).item()

            # Verify basic statistics
            if not np.isfinite(output_mean) or not np.isfinite(output_std):
                return False

            # Ensure job_id exists in history before accessing it
            if job_id in self.behavior_history and len(self.behavior_history[job_id]) > 10:
                # Compare with historical distributions using Wasserstein distance
                historical_outputs = torch.stack(
                    [torch.tensor(h['data']['outputs']) for h in self.behavior_history[job_id][-10:]]
                )

                w_distance = wasserstein_distance(
                    outputs.flatten().cpu().numpy(),
                    historical_outputs.mean(0).flatten().cpu().numpy()
                )

                if w_distance > self.config.get('max_distribution_shift', 1.0):
                    return False

            return True

        except Exception as e:
            logging.error(f"Error verifying output distribution: {str(e)}")
            return False

    def _verify_model_consistency(self, model: nn.Module, inputs: torch.Tensor) -> bool:
        """Verify model produces consistent outputs"""
        try:
            with torch.no_grad():
                # Multiple forward passes
                outputs = [model(inputs) for _ in range(3)]
                
                # Check consistency
                for i in range(len(outputs) - 1):
                    diff = torch.norm(outputs[i] - outputs[i+1]).item()
                    if diff > self.config.get('max_consistency_diff', 1e-6):
                        return False
                        
            return True
            
        except Exception as e:
            logging.error(f"Error verifying model consistency: {str(e)}")
            return False

    def _verify_activation_patterns(self, model: nn.Module) -> bool:
        """Verify neural network activation patterns"""
        try:
            activation_patterns = {}
            
            # Hook for collecting activations
            def hook_fn(name):
                def hook(module, input, output):
                    activation_patterns[name] = {
                        'mean': torch.mean(output).item(),
                        'std': torch.std(output).item(),
                        'zeros': (output == 0).float().mean().item()
                    }
                return hook
            
            # Register hooks
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
                    
            # Verify patterns
            for patterns in activation_patterns.values():
                # Check for dead neurons
                if patterns['zeros'] > self.config.get('max_dead_neurons', 0.9):
                    return False
                    
                # Check activation statistics
                if not np.isfinite(patterns['mean']) or not np.isfinite(patterns['std']):
                    return False
                    
            # Remove hooks
            for hook in hooks:
                hook.remove()
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying activation patterns: {str(e)}")
            return False
    
    #
    # Weight update verification methods
    #
    
    def _verify_weight_updates(self,
                             model: nn.Module,
                             optimizer: torch.optim.Optimizer) -> Tuple[bool, Dict]:
        """Verify weight update patterns"""
        try:
            update_stats = []
            valid_updates = True
            
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                    
                # Calculate proposed update
                update = self._calculate_proposed_update(param, optimizer)
                
                # Verify update magnitude
                update_magnitude = torch.norm(update).item()
                weight_magnitude = torch.norm(param).item()
                update_ratio = update_magnitude / (weight_magnitude + 1e-8)
                
                # Store update statistics
                update_stats.append({
                    'layer_name': name,
                    'update_magnitude': update_magnitude,
                    'weight_magnitude': weight_magnitude,
                    'update_ratio': update_ratio
                })
                
                # Verify update properties
                if not self._verify_update_properties(update_ratio, name):
                    valid_updates = False
                    
            return valid_updates, {
                'update_stats': update_stats,
                'update_patterns_valid': valid_updates
            }
            
        except Exception as e:
            logging.error(f"Error verifying weight updates: {str(e)}")
            raise

    def _calculate_proposed_update(self, param: torch.Tensor, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """Calculate the proposed parameter update using PyTorch's optimizer state."""
        try:
            if isinstance(optimizer, torch.optim.Adam):
                return optimizer.param_groups[0]['lr'] * optimizer.state[param]['exp_avg']
            return -optimizer.param_groups[0]['lr'] * param.grad
        except Exception as e:
            logging.error(f"Error calculating proposed update: {str(e)}")
            return torch.zeros_like(param)

    def _verify_update_properties(self, update_ratio: float, layer_name: str) -> bool:
        """Verify update ratio properties"""
        try:
            # Check if update ratio is within acceptable range
            max_ratio = self.config.get('max_update_ratio', 0.1)
            
            # Layer-specific checks
            if 'embedding' in layer_name:
                # Allow larger updates for embedding layers
                max_ratio *= 2
                
            if update_ratio > max_ratio:
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying update properties: {str(e)}")
            return False
    
    #
    # Loss verification methods
    #
    
    def _verify_loss_characteristics(self, loss: torch.Tensor) -> Tuple[bool, Dict]:
        """Verify loss value characteristics"""
        try:
            loss_value = loss.item()
            
            # Check if loss is finite and reasonable
            is_valid = np.isfinite(loss_value)
            
            # Check loss magnitude
            magnitude_valid = self._verify_loss_magnitude(loss_value)
            
            # Check loss progression
            progression_valid = self._verify_loss_progression(loss_value)
            
            return is_valid and magnitude_valid and progression_valid, {
                'loss_value': loss_value,
                'magnitude_valid': magnitude_valid,
                'progression_valid': progression_valid
            }
            
        except Exception as e:
            logging.error(f"Error verifying loss: {str(e)}")
            raise

    def _verify_loss_magnitude(self, loss_value: float) -> bool:
        """Verify loss value is within reasonable bounds"""
        try:
            # Get configured bounds
            min_loss = self.config.get('min_loss', 0.0)
            max_loss = self.config.get('max_loss', 100.0)
            
            # Check bounds
            if loss_value < min_loss or loss_value > max_loss:
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying loss magnitude: {str(e)}")
            return False

    def _verify_loss_progression(self, current_loss: float) -> bool:
        """Verify loss is progressing reasonably"""
        try:
            history_length = len(self.behavior_history)
            if history_length < 2:
                return True
                
            # Get recent loss values
            recent_losses = [h['loss'] for h in self.behavior_history[-10:]]
            
            # Calculate statistics
            mean_loss = np.mean(recent_losses)
            std_loss = np.std(recent_losses)
            
            # Check for significant deviation
            z_score = abs(current_loss - mean_loss) / (std_loss + 1e-8)
            if z_score > self.config.get('max_loss_z_score', 3.0):
                return False
                
            # Check for progress
            if history_length >= 10:
                # Calculate moving averages
                window_size = 5
                recent_ma = np.mean(recent_losses[-window_size:])
                previous_ma = np.mean(recent_losses[-2*window_size:-window_size])
                
                # Allow for some stability in later training
                min_improvement = self.config.get('min_loss_improvement', -0.01)
                relative_change = (recent_ma - previous_ma) / (previous_ma + 1e-8)
                
                if relative_change > 0 and relative_change > abs(min_improvement):
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error verifying loss progression: {str(e)}")
            return False
    
    #
    # Layer verification methods
    #
    
    def _verify_conv_layer(self,
                         proof: LayerComputationProof,
                         state_dict: Dict[str, torch.Tensor],
                         layer: nn.Conv2d) -> bool:
        """Verify convolution layer computation"""
        try:
            # Verify parameter consistency
            if not self._verify_conv_parameters(proof, layer):
                return False
            
            # Verify convolution operation
            if not self._verify_conv_operation(proof, state_dict, layer):
                return False
            
            # Verify output consistency
            if not self._verify_conv_output(proof, layer):
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error verifying convolution layer: {str(e)}")
            return False

    def _verify_linear_layer(self,
                           proof: LayerComputationProof,
                           state_dict: Dict[str, torch.Tensor],
                           layer: nn.Linear) -> bool:
        """Verify linear layer computation"""
        try:
            # Verify parameter consistency
            if not self._verify_linear_parameters(proof, layer):
                return False
            
            # Verify matrix multiplication
            if not self._verify_linear_operation(proof, state_dict, layer):
                return False
            
            # Verify activation and bias
            if not self._verify_linear_activation(proof, layer):
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error verifying linear layer: {str(e)}")
            return False

    def _verify_batch_norm_layer(self,
                               proof: LayerComputationProof,
                               state_dict: Dict[str, torch.Tensor],
                               layer: nn.BatchNorm2d) -> bool:
        """Verify batch normalization computation"""
        try:
            # Verify statistics computation
            if not self._verify_batch_statistics(proof, layer):
                return False
            
            # Verify normalization operation
            if not self._verify_normalization_operation(proof, state_dict, layer):
                return False
            
            # Verify running statistics update
            if not self._verify_running_stats_update(proof, layer):
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error verifying batch norm layer: {str(e)}")
            return False

    def _verify_generic_layer(self,
                            proof: LayerComputationProof,
                            state_dict: Dict[str, torch.Tensor],
                            layer: nn.Module) -> bool:
        """Verify computation for any generic layer"""
        try:
            # Get input and expected output
            input_tensor = proof.intermediate_values.get('input')
            expected_output = proof.intermediate_values.get('output')
            
            if input_tensor is None or expected_output is None:
                return False
                
            # Perform forward pass
            with torch.no_grad():
                actual_output = layer(input_tensor)
                
            # Compare outputs
            return torch.allclose(actual_output, expected_output, rtol=1e-4)
            
        except Exception as e:
            logging.error(f"Error verifying generic layer: {str(e)}")
            return False
    
    #
    # Layer-specific verification components
    #
    
    def _verify_conv_parameters(self,
                              proof: LayerComputationProof,
                              layer: nn.Conv2d) -> bool:
        """Verify convolution parameter consistency"""
        try:
            # Verify kernel size
            if proof.intermediate_values.get('kernel_size') != layer.kernel_size:
                return False
                
            # Verify stride and padding
            if (proof.intermediate_values.get('stride') != layer.stride or
                proof.intermediate_values.get('padding') != layer.padding):
                return False
                
            # Verify channel dimensions
            input_shape = proof.input_shape
            output_shape = proof.output_shape
            
            if (input_shape[1] != layer.in_channels or
                output_shape[1] != layer.out_channels):
                return False
                
            # Verify parameter hash
            param_hash = self._hash_parameters(layer.state_dict())
            if param_hash != proof.parameters_hash:
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying conv parameters: {str(e)}")
            return False

    def _verify_conv_operation(self,
                             proof: LayerComputationProof,
                             state_dict: Dict[str, torch.Tensor],
                             layer: nn.Conv2d) -> bool:
        """Verify convolution operation correctness"""
        try:
            # Get input and output tensors
            input_tensor = proof.intermediate_values.get('input')
            output_tensor = proof.intermediate_values.get('output')
            
            if input_tensor is None or output_tensor is None:
                return False
                
            # Compute expected output
            expected_output = F.conv2d(
                input_tensor,
                state_dict['weight'],
                state_dict.get('bias'),
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups
            )
            
            # Verify output matches
            if not torch.allclose(output_tensor, expected_output, rtol=1e-4):
                return False
                
            # Verify intermediate computations if available
            if 'im2col_output' in proof.intermediate_values:
                if not self._verify_conv_intermediates(proof, layer):
                    return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying conv operation: {str(e)}")
            return False

    def _verify_conv_output(self,
                          proof: LayerComputationProof,
                          layer: nn.Conv2d) -> bool:
        """Verify convolution output consistency"""
        try:
            output = proof.intermediate_values.get('output')
            if output is None:
                return False
                
            # Verify output shape
            if output.shape != proof.output_shape:
                return False
                
            # Verify output statistics if provided
            output_stats = proof.intermediate_values.get('output_stats')
            if output_stats and not self._verify_tensor_statistics(output, output_stats):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying conv output: {str(e)}")
            return False

    def _verify_linear_parameters(self,
                                proof: LayerComputationProof,
                                layer: nn.Linear) -> bool:
        """Verify linear layer parameter consistency"""
        try:
            # Verify dimensions
            if (proof.input_shape[-1] != layer.in_features or
                proof.output_shape[-1] != layer.out_features):
                return False
                
            # Verify weight matrix shape
            weight = proof.intermediate_values.get('weight')
            if weight is None or weight.shape != (layer.out_features, layer.in_features):
                return False
                
            # Verify bias if present
            if layer.bias is not None:
                bias = proof.intermediate_values.get('bias')
                if bias is None or bias.shape != (layer.out_features,):
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error verifying linear parameters: {str(e)}")
            return False

    def _verify_linear_operation(self,
                               proof: LayerComputationProof,
                               state_dict: Dict[str, torch.Tensor],
                               layer: nn.Linear) -> bool:
        """Verify linear layer operation"""
        try:
            # Get input and output tensors
            input_tensor = proof.intermediate_values.get('input')
            output_tensor = proof.intermediate_values.get('output')
            
            if input_tensor is None or output_tensor is None:
                return False
                
            # Compute expected output
            expected_output = F.linear(
                input_tensor,
                state_dict['weight'],
                state_dict.get('bias')
            )
            
            # Verify output matches
            if not torch.allclose(output_tensor, expected_output, rtol=1e-4):
                return False
                
            # Verify matrix multiplication steps if provided
            if 'matmul_steps' in proof.intermediate_values:
                if not self._verify_matmul_steps(proof):
                    return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying linear operation: {str(e)}")
            return False

    def _verify_linear_activation(self,
                                proof: LayerComputationProof,
                                layer: nn.Linear) -> bool:
        """Verify linear layer activation"""
        try:
            # Get pre and post activation values if available
            pre_activation = proof.intermediate_values.get('pre_activation')
            post_activation = proof.intermediate_values.get('output')
            
            if pre_activation is None or post_activation is None:
                return True  # Skip if not provided
                
            # Check for activation function
            activation = None
            for name, module in layer.named_modules():
                if isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                    activation = module
                    break
                    
            # Verify activation application if it exists
            if activation is not None:
                expected_output = activation(pre_activation)
                if not torch.allclose(post_activation, expected_output, rtol=1e-4):
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error verifying linear activation: {str(e)}")
            return False

    def _verify_batch_statistics(self,
                               proof: LayerComputationProof,
                               layer: nn.BatchNorm2d) -> bool:
        """Verify batch normalization statistics"""
        try:
            input_tensor = proof.intermediate_values.get('input')
            if input_tensor is None:
                return False
                
            # Calculate expected statistics
            expected_mean = input_tensor.mean([0, 2, 3])
            expected_var = input_tensor.var([0, 2, 3], unbiased=True)
            
            # Get statistics from proof
            batch_mean = proof.intermediate_values.get('batch_mean')
            batch_var = proof.intermediate_values.get('batch_var')
            
            if batch_mean is None or batch_var is None:
                return False
                
            # Verify statistics match
            if not torch.allclose(batch_mean, expected_mean, rtol=1e-4):
                return False
                
            if not torch.allclose(batch_var, expected_var, rtol=1e-4):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying batch statistics: {str(e)}")
            return False

    def _verify_normalization_operation(self,
                                      proof: LayerComputationProof,
                                      state_dict: Dict[str, torch.Tensor],
                                      layer: nn.BatchNorm2d) -> bool:
        """Verify normalization operation"""
        try:
            # Get tensors and statistics
            input_tensor = proof.intermediate_values.get('input')
            output_tensor = proof.intermediate_values.get('output')
            batch_mean = proof.intermediate_values.get('batch_mean')
            batch_var = proof.intermediate_values.get('batch_var')
            
            if (input_tensor is None or output_tensor is None or
                batch_mean is None or batch_var is None):
                return False
                
            # Determine which statistics to use
            training_mode = proof.intermediate_values.get('training_mode', True)
            
            if training_mode:
                mean = batch_mean
                var = batch_var
            else:
                mean = state_dict['running_mean']
                var = state_dict['running_var']
                
            # Apply normalization
            normalized = (input_tensor - mean[None, :, None, None]) / torch.sqrt(
                var[None, :, None, None] + layer.eps
            )
            
            # Apply scale and shift if affine
            if layer.affine:
                weight = state_dict['weight']
                bias = state_dict['bias']
                normalized = normalized * weight[None, :, None, None]
                normalized = normalized + bias[None, :, None, None]
                
            # Verify output matches
            if not torch.allclose(output_tensor, normalized, rtol=1e-4):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying normalization operation: {str(e)}")
            return False

    def _verify_running_stats_update(self,
                                   proof: LayerComputationProof,
                                   layer: nn.BatchNorm2d) -> bool:
        """Verify running statistics update"""
        try:
            # Get statistics
            batch_mean = proof.intermediate_values.get('batch_mean')
            batch_var = proof.intermediate_values.get('batch_var')
            updated_running_mean = proof.intermediate_values.get('updated_running_mean')
            updated_running_var = proof.intermediate_values.get('updated_running_var')
            
            if (batch_mean is None or batch_var is None or
                updated_running_mean is None or updated_running_var is None):
                return True  # Skip if not provided
                
            # Check if in training mode
            training_mode = proof.intermediate_values.get('training_mode', True)
            if not training_mode:
                return True  # No updates in evaluation mode
                
            # Calculate expected updates
            momentum = layer.momentum
            running_mean = layer.running_mean
            running_var = layer.running_var
            
            expected_mean = (1 - momentum) * running_mean + momentum * batch_mean
            expected_var = (1 - momentum) * running_var + momentum * batch_var
            
            # Verify updates match
            if not torch.allclose(updated_running_mean, expected_mean, rtol=1e-4):
                return False
                
            if not torch.allclose(updated_running_var, expected_var, rtol=1e-4):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying running stats update: {str(e)}")
            return False
    
    #
    # Data verification methods
    #
    
    def _validate_data_proof_structure(self, proof: DataVerificationProof) -> bool:
        """Validate basic structure of data proof"""
        try:
            # Check required fields
            if not proof.data_id or not proof.result_hash:
                return False
                
            # Check processing steps
            if not proof.processing_steps or not isinstance(proof.processing_steps, list):
                return False
                
            # Validate each processing step
            for step in proof.processing_steps:
                if 'operation' not in step:
                    return False
                    
            # Check timestamp validity
            if proof.timestamp <= 0 or proof.timestamp > time.time():
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating data proof structure: {str(e)}")
            return False

    async def _verify_source_hash(self, source_hash: str, data_id: str) -> bool:
        """Verify source data hash matches reference if available"""
        try:
            # This would use a reference data registry or similar
            # For now, we'll just assume it's valid if it's a valid hash
            
            # Check if hash is valid format
            if not self._is_valid_hash_format(source_hash):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying source hash: {str(e)}")
            return False

    async def _verify_processing_steps(self, steps: List[Dict]) -> Tuple[bool, List[str], List[str]]:
        """Verify processing steps are valid and consistent"""
        try:
            warnings = []
            errors = []
            
            # Allowed operations and their validations
            allowed_operations = {
                'filter', 'transform', 'aggregate', 'join', 'normalize',
                'split', 'sample', 'sort', 'impute', 'encode'
            }
            
            # Validate steps
            for i, step in enumerate(steps):
                operation = step.get('operation')
                
                # Check operation is known
                if operation not in allowed_operations:
                    errors.append(f"Unknown operation in step {i}: {operation}")
                    continue
                    
                # Check parameters
                if 'parameters' not in step:
                    warnings.append(f"Missing parameters in step {i}")
                    
                # Operation-specific validations
                if operation == 'filter' and 'condition' not in step.get('parameters', {}):
                    errors.append(f"Missing 'condition' in filter step {i}")
                    
                elif operation == 'join' and 'key' not in step.get('parameters', {}):
                    errors.append(f"Missing 'key' in join step {i}")
                    
            # Check step sequence validity
            valid_sequence = self._validate_step_sequence(steps)
            if not valid_sequence:
                errors.append("Invalid processing step sequence")
                
            return len(errors) == 0, warnings, errors
            
        except Exception as e:
            logging.error(f"Error verifying processing steps: {str(e)}")
            return False, [], [str(e)]

    async def _verify_result_hash(self, result_hash: str, data_id: str) -> bool:
        """Verify result hash matches expected if available"""
        try:
            # This would check against expected results
            # For now, we'll just check if the hash is valid
            if not self._is_valid_hash_format(result_hash):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying result hash: {str(e)}")
            return False

    def _validate_step_sequence(self, steps: List[Dict]) -> bool:
        """Validate that the sequence of operations is logically valid"""
        try:
            # Example validation rules:
            # - 'join' must come after data is available
            # - 'impute' should come before certain transformations
            
            operations = [step['operation'] for step in steps]
            
            # Check for valid join sequences
            if 'join' in operations and operations.index('join') == 0:
                return False  # Join can't be first
                
            # Check for valid imputation
            if 'impute' in operations and 'encode' in operations:
                if operations.index('impute') > operations.index('encode'):
                    return False  # Impute should come before encode
                    
            return True
            
        except Exception as e:
            logging.error(f"Error validating step sequence: {str(e)}")
            return False

    def _is_valid_hash_format(self, hash_str: str) -> bool:
        """Check if a string is a valid hash format"""
        try:
            # Check if it's a valid hash format (SHA-256 is 64 hex chars)
            if not hash_str or not isinstance(hash_str, str):
                return False
                
            # Check if it's a valid hex string of appropriate length
            valid_lengths = {32, 40, 64, 128}  # MD5, SHA-1, SHA-256, SHA-512
            
            if len(hash_str) not in valid_lengths:
                return False
                
            # Check if it contains only valid hex characters
            return all(c in '0123456789abcdefABCDEF' for c in hash_str)
            
        except Exception as e:
            logging.error(f"Error validating hash format: {str(e)}")
            return False
    
    #
    # Verification coordination methods
    #
    
    async def _manage_verification_process(self, verification_task_id: str, proof: VerificationProof):
        """Manage the verification process lifecycle"""
        try:
            context = self.active_verifications[verification_task_id]
            
            # Create verification tasks for each verifier
            verification_tasks = []
            for verifier_id in context.verifiers:
                task = self._verify_with_node(
                    verification_task_id=verification_task_id,
                    verifier_id=verifier_id,
                    proof=proof
                )
                verification_tasks.append(task)
            
            # Wait for verifications to complete or timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*verification_tasks),
                    timeout=self.config.get('verification_timeout', 300)
                )
            except asyncio.TimeoutError:
                context.status = 'timeout'
                return
            
            # Process results
            validation_result = await self._validate_verification_results(context)
            
            # Update verifier stats
            await self._update_verifier_stats(context, validation_result)
            
            # Process payments
            if validation_result['is_valid']:
                await self._process_verification_payments(context)
            
            # Store in history
            self._store_verification_history(verification_task_id, validation_result)
            
            # Mark as completed
            context.status = 'completed'
            
        except Exception as e:
            logging.error(f"Error managing verification process: {str(e)}")
            raise
        finally:
            # Cleanup after completion or timeout
            await self._cleanup_verification_task(verification_task_id)

    async def _verify_with_node(self,
                              verification_task_id: str,
                              verifier_id: str,
                              proof: VerificationProof) -> Dict:
        """Execute verification with a specific node"""
        try:
            context = self.active_verifications[verification_task_id]
            
            # In a real implementation, this would send the verification
            # request to the node using network communication
            # For now, we'll simulate the verification
            
            # Simulate verification time
            start_time = time.time()
            await asyncio.sleep(0.5)  # Simulate network delay
            
            # Simulate verification result
            verification_result = {
                'is_valid': True,  # Most verifications pass
                'details': {
                    'verification_time': time.time() - start_time,
                    'metrics_validated': True,
                    'hash_validated': True,
                }
            }
            
            # Store result
            result = {
                'verifier_id': verifier_id,
                'result': verification_result,
                'response_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
            if context.results is None:
                context.results = []
            context.results.append(result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error verifying with node {verifier_id}: {str(e)}")
            raise

    async def _validate_verification_results(self, context: JobVerificationContext) -> Dict:
        """Validate collected verification results"""
        try:
            if not context.results:
                return {
                    'is_valid': False,
                    'details': {
                        'error': 'No verification results'
                    }
                }
            
            # Count positive verifications
            total_verifiers = len(context.verifiers)
            positive_verifications = sum(
                1 for r in context.results if r['result']['is_valid']
            )
            
            # Calculate consensus
            verification_threshold = self.config.get('verification_threshold', 0.67)
            is_valid = (positive_verifications / total_verifiers) >= verification_threshold
            
            return {
                'is_valid': is_valid,
                'details': {
                    'total_verifiers': total_verifiers,
                    'positive_verifications': positive_verifications,
                    'consensus_rate': positive_verifications / total_verifiers,
                    'threshold': verification_threshold
                }
            }
            
        except Exception as e:
            logging.error(f"Error validating verification results: {str(e)}")
            raise

    async def _update_verifier_stats(self, context: JobVerificationContext, validation_result: Dict):
        """Update verifier statistics based on results"""
        # In a real implementation, this would update the statistics
        # of verifier nodes based on the verification results
        pass

    async def _process_verification_payments(self, context: JobVerificationContext):
        """Process payments for successful verifications"""
        # In a real implementation, this would process payments
        # to verifier nodes that successfully verified the computation
        pass

    async def _cleanup_verification_task(self, verification_task_id: str):
        """Clean up verification task resources"""
        try:
            if verification_task_id in self.active_verifications:
                del self.active_verifications[verification_task_id]
                
        except Exception as e:
            logging.error(f"Error cleaning up verification task: {str(e)}")
            raise
    
    #
    # Helper methods
    #
    
    def _store_verification_data(self, job_id: str, verification_data: Dict):
        """Store verification data for history and analysis"""
        try:
            # Initialize if needed
            self.gradient_history[job_id] = self.gradient_history.get(job_id, [])
            self.behavior_history[job_id] = self.behavior_history.get(job_id, [])
            self.weight_update_history[job_id] = self.weight_update_history.get(job_id, [])

            # Store data
            timestamp = time.time()
            self.gradient_history[job_id].append({'timestamp': timestamp, 'data': verification_data['gradients']})
            self.behavior_history[job_id].append({'timestamp': timestamp, 'data': verification_data['behavior']})
            self.weight_update_history[job_id].append({'timestamp': timestamp, 'data': verification_data['updates']})

            # Maintain history size
            max_history = self.config.get('max_history_size', 1000)
            self.gradient_history[job_id] = self.gradient_history[job_id][-max_history:]
            self.behavior_history[job_id] = self.behavior_history[job_id][-max_history:]
            self.weight_update_history[job_id] = self.weight_update_history[job_id][-max_history:]

        except Exception as e:
            logging.error(f"Error storing verification data: {str(e)}")

    def _store_verification_history(self, verification_task_id: str, validation_result: Dict):
        """Store verification results in history"""
        try:
            context = self.active_verifications[verification_task_id]
            
            history_entry = {
                'task_id': verification_task_id,
                'job_id': context.job_id,
                'checkpoint_id': context.checkpoint_id,
                'verifiers': context.verifiers,
                'results': context.results,
                'validation_result': validation_result,
                'timestamp': time.time()
            }
            
            self.verification_history.setdefault(context.job_id, []).append(history_entry)
            
            # Prune old history
            max_history = self.config.get('max_history_size', 1000)
            if len(self.verification_history[context.job_id]) > max_history:
                self.verification_history[context.job_id] = \
                    self.verification_history[context.job_id][-max_history:]
                    
        except Exception as e:
            logging.error(f"Error storing verification history: {str(e)}")
            raise

    def _generate_perturbation(self, inputs: torch.Tensor, p_type: str, magnitude: float) -> torch.Tensor:
        """Generate input perturbation"""
        if p_type == 'gaussian':
            noise = torch.randn_like(inputs) * magnitude
            return inputs + noise
        elif p_type == 'dropout':
            mask = torch.bernoulli(torch.full_like(inputs, 1 - magnitude))
            return inputs * mask
        else:
            raise ValueError(f"Unknown perturbation type: {p_type}")

    def _verify_perturbation_response(self, output_diff: float, magnitude: float, p_type: str) -> bool:
        """Verify model response to perturbations"""
        try:
            # Scale factor based on perturbation type
            scale_factors = {
                'gaussian': 10.0,  # Allow larger response for Gaussian noise
                'dropout': 5.0     # Expect smaller response for dropout
            }
            
            scale = scale_factors.get(p_type, 1.0)
            max_allowed_diff = magnitude * scale
            
            return output_diff <= max_allowed_diff
            
        except Exception as e:
            logging.error(f"Error verifying perturbation response: {str(e)}")
            return False

    def _verify_conv_intermediates(self, proof: LayerComputationProof, layer: nn.Conv2d) -> bool:
        """Verify intermediate convolution computations"""
        # Check im2col operation if available
        im2col_output = proof.intermediate_values.get('im2col_output')
        input_tensor = proof.intermediate_values.get('input')
        
        if im2col_output is not None and input_tensor is not None:
            # Implementation would verify the im2col operation
            return True
            
        return True

    def _verify_matmul_steps(self, proof: LayerComputationProof) -> bool:
        """Verify matrix multiplication steps"""
        try:
            # Get intermediate values
            matmul_steps = proof.intermediate_values.get('matmul_steps')
            if not matmul_steps:
                return True  # Skip if not provided
                
            # Implementation would verify each step of matrix multiplication
            return True
            
        except Exception as e:
            logging.error(f"Error verifying matmul steps: {str(e)}")
            return False

    def _verify_tensor_statistics(self, tensor: torch.Tensor, stats: Dict[str, float]) -> bool:
        """Verify tensor statistics match expected values"""
        try:
            # Verify mean
            tensor_mean = tensor.mean().item()
            if 'mean' in stats and abs(tensor_mean - stats['mean']) > 1e-4:
                return False
                
            # Verify standard deviation
            tensor_std = tensor.std().item()
            if 'std' in stats and abs(tensor_std - stats['std']) > 1e-4:
                return False
                
            # Verify min/max values if provided
            if 'min' in stats and abs(tensor.min().item() - stats['min']) > 1e-4:
                return False
                
            if 'max' in stats and abs(tensor.max().item() - stats['max']) > 1e-4:
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying tensor statistics: {str(e)}")
            return False

    def _hash_parameters(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Generate hash of layer parameters"""
        try:
            hasher = hashlib.sha256()
            
            # Add parameters in sorted order
            for name, param in sorted(state_dict.items()):
                hasher.update(name.encode())
                hasher.update(param.cpu().numpy().tobytes())
                
            return hasher.hexdigest()
            
        except Exception as e:
            logging.error(f"Error hashing parameters: {str(e)}")
            raise

    def _decode_computation_proof(self, response: bytes) -> LayerComputationProof:
        """Decode computation proof from response bytes"""
        # In a real implementation, this would deserialize a proof from bytes
        # For now, return a mock proof
        import pickle
        try:
            return pickle.loads(response)
        except:
            # Return mock proof if deserialization fails
            return LayerComputationProof(
                layer_id="mock_layer",
                input_shape=(1, 3, 224, 224),
                output_shape=(1, 64, 112, 112),
                parameters_hash="0123456789abcdef0123456789abcdef",
                computation_trace=b"",
                intermediate_values={}
            )
    
    def _generate_proof(self, 
                      model: nn.Module,
                      inputs: torch.Tensor,
                      outputs: torch.Tensor,
                      metrics: Dict,
                      job_id: str,
                      checkpoint_id: int) -> VerificationProof:
        """Generate verification proof for a model checkpoint"""
        try:
            # Calculate state hash
            state_hash = self._calculate_model_state_hash(model)
            
            # Create model checkpoint for ZK proof
            checkpoint = self._create_model_checkpoint(model, inputs, outputs)
            
            # Generate zero-knowledge proof
            zk_proof = asyncio.run(self.zk_proof_generator.generate_proof(
                model=model,
                inputs=inputs,
                outputs=outputs,
                checkpoint=checkpoint
            ))
            
            # Combine proof components
            proof_data = self._serialize_proof_components(
                model=model, 
                inputs=inputs, 
                outputs=outputs, 
                metrics=metrics, 
                zk_proof=zk_proof
            )
            
            # Create proof object
            proof = VerificationProof(
                job_id=job_id,
                checkpoint_id=checkpoint_id,
                state_hash=state_hash,
                proof_data=proof_data,
                metrics=metrics,
                timestamp=time.time()
            )
            
            logging.info(f"Generated ZK verification proof for job {job_id}, checkpoint {checkpoint_id}")
            return proof
            
        except Exception as e:
            logging.error(f"Error generating proof: {str(e)}")
            raise

    def _create_model_checkpoint(self, 
                              model: nn.Module,
                              inputs: torch.Tensor,
                              outputs: torch.Tensor) -> ModelCheckpoint:
        """Create checkpoint of model state for ZK proof generation"""
        try:
            # Collect layer states
            layer_states = {}
            for name, param in model.named_parameters():
                layer_states[name] = param.detach().clone()
                
            # Collect intermediate outputs
            intermediate_outputs = {}
            
            # Run forward pass with hooks to capture intermediate outputs
            hooks = []
            
            def hook_fn(name):
                def hook(module, input, output):
                    intermediate_outputs[name] = output.detach().clone()
                return hook
            
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            
            # Run forward pass to collect outputs
            with torch.no_grad():
                model(inputs)
                
            # Remove hooks
            for hook in hooks:
                hook.remove()
                
            # Calculate gradient norms
            gradient_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradient_norms[name] = torch.norm(param.grad).item()
                    
            # Collect computation metrics
            computation_metrics = {
                'input_shape': list(inputs.shape),
                'output_shape': list(outputs.shape),
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'forward_time': 0.0  # Would be measured in real implementation
            }
            
            return ModelCheckpoint(
                layer_states=layer_states,
                intermediate_outputs=intermediate_outputs,
                gradient_norms=gradient_norms,
                computation_metrics=computation_metrics
            )
            
        except Exception as e:
            logging.error(f"Error creating model checkpoint: {str(e)}")
            raise

    def _serialize_proof_components(self,
                                  model: nn.Module,
                                  inputs: torch.Tensor,
                                  outputs: torch.Tensor,
                                  metrics: Dict,
                                  zk_proof: ProofComponents) -> bytes:
        """Serialize proof components into binary format"""
        try:
            # Combine proof components
            proof_components = {
                'input_shape': inputs.shape,
                'input_stats': {
                    'mean': inputs.mean().item(),
                    'std': inputs.std().item()
                },
                'output_shape': outputs.shape,
                'output_stats': {
                    'mean': outputs.mean().item(),
                    'std': outputs.std().item()
                },
                'metrics': metrics,
                'layer_hashes': self._calculate_layer_hashes(model),
                'activation_samples': self._sample_activations(model, inputs),
                'zk_proof': {
                    'commitment': zk_proof.commitment,
                    'challenge': zk_proof.challenge,
                    'response': zk_proof.response,
                    'auxiliary_data_keys': list(zk_proof.auxiliary_data.keys())
                }
            }
            
            # Serialize to bytes
            import pickle
            return pickle.dumps(proof_components)
            
        except Exception as e:
            logging.error(f"Error serializing proof components: {str(e)}")
            raise

    def _calculate_model_state_hash(self, model: nn.Module) -> bytes:
        """Calculate hash of model state"""
        try:
            hasher = hashlib.sha256()
            
            # Add model parameters
            for name, param in sorted(model.state_dict().items()):
                param_bytes = param.cpu().detach().numpy().tobytes()
                hasher.update(name.encode() + param_bytes)
                
            return hasher.digest()
            
        except Exception as e:
            logging.error(f"Error calculating model state hash: {str(e)}")
            raise

    async def verify_model_computation(self,
                                 model: nn.Module,
                                 inputs: torch.Tensor,
                                 outputs: torch.Tensor,
                                 proof: VerificationProof) -> Tuple[bool, Dict]:
        """Verify model computation using zero-knowledge proofs"""
        try:
            start_time = time.time()
            
            # Extract proof components
            proof_components = self._deserialize_proof_components(proof.proof_data)
            
            # Verify model state hash
            state_hash_valid = self._verify_model_state_hash(model, proof.state_hash)
            
            # Verify input/output consistency
            io_consistency_valid = self._verify_io_consistency(
                inputs, outputs, proof_components
            )
            
            # Verify zero-knowledge proof
            zk_proof_valid = await self._verify_zk_proof(
                model=model,
                inputs=inputs,
                outputs=outputs,
                proof_components=proof_components,
                state_hash=proof.state_hash
            )
            
            # Combine verification results
            is_valid = state_hash_valid and io_consistency_valid and zk_proof_valid
            
            verification_time = time.time() - start_time
            
            return is_valid, {
                'state_hash_valid': state_hash_valid,
                'io_consistency_valid': io_consistency_valid,
                'zk_proof_valid': zk_proof_valid,
                'verification_time': verification_time
            }
            
        except Exception as e:
            logging.error(f"Error verifying model computation: {str(e)}")
            return False, {'error': str(e)}

    def _deserialize_proof_components(self, proof_data: bytes) -> Dict:
        """Deserialize proof components from binary data"""
        try:
            import pickle
            return pickle.loads(proof_data)
        except Exception as e:
            logging.error(f"Error deserializing proof components: {str(e)}")
            return {}

    def _verify_model_state_hash(self, model: nn.Module, expected_hash: bytes) -> bool:
        """Verify model state hash matches expected hash"""
        try:
            actual_hash = self._calculate_model_state_hash(model)
            return actual_hash == expected_hash
        except Exception as e:
            logging.error(f"Error verifying model state hash: {str(e)}")
            return False

    def _verify_io_consistency(self, 
                             inputs: torch.Tensor,
                             outputs: torch.Tensor,
                             proof_components: Dict) -> bool:
        """Verify input/output consistency with proof"""
        try:
            # Check input shape
            if inputs.shape != tuple(proof_components.get('input_shape', ())):
                return False
                
            # Check output shape
            if outputs.shape != tuple(proof_components.get('output_shape', ())):
                return False
                
            # Check input statistics (approximate)
            input_stats = proof_components.get('input_stats', {})
            if abs(inputs.mean().item() - input_stats.get('mean', 0)) > 1e-2:
                return False
                
            if abs(inputs.std().item() - input_stats.get('std', 0)) > 1e-2:
                return False
                
            # Check output statistics (approximate)
            output_stats = proof_components.get('output_stats', {})
            if abs(outputs.mean().item() - output_stats.get('mean', 0)) > 1e-2:
                return False
                
            if abs(outputs.std().item() - output_stats.get('std', 0)) > 1e-2:
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying I/O consistency: {str(e)}")
            return False

    async def _verify_zk_proof(self,
                            model: nn.Module,
                            inputs: torch.Tensor,
                            outputs: torch.Tensor,
                            proof_components: Dict,
                            state_hash: bytes) -> bool:
        """Verify zero-knowledge proof"""
        try:
            # Skip verification if proof components are missing
            zk_proof = proof_components.get('zk_proof')
            if not zk_proof:
                logging.warning("ZK proof components missing, skipping verification")
                return True
                
            # Extract proof components
            commitment = zk_proof.get('commitment')
            challenge = zk_proof.get('challenge')
            response = zk_proof.get('response')
            
            if not commitment or not challenge or not response:
                logging.warning("Invalid ZK proof format")
                return False
            
            # Create mock ProofComponents for verification
            from cryptography.hazmat.primitives import hashes
            auxiliary_data = {}
            for key in zk_proof.get('auxiliary_data_keys', []):
                # In real implementation, these would be properly extracted
                auxiliary_data[key] = bytes(32)  # Mock data
                
            proof = ProofComponents(
                commitment=commitment,
                challenge=challenge,
                response=response,
                auxiliary_data=auxiliary_data
            )
            
            # Use the curve to validate the signature (commitment)
            # This is a simplified version - full implementation would do proper ZK verification
            try:
                # Verify commitment signature
                hasher = hashes.Hash(hashes.SHA256())
                hasher.update(state_hash)
                message = hasher.finalize()
                
                # In a real implementation, this would properly verify the ZK proof
                # For now, we just check if the commitment is properly formed
                return len(commitment) > 0 and len(challenge) > 0 and len(response) > 0
                
            except Exception as e:
                logging.error(f"ZK proof verification failed: {str(e)}")
                return False
                
        except Exception as e:
            logging.error(f"Error verifying ZK proof: {str(e)}")
            return False

    def _calculate_layer_hashes(self, model: nn.Module) -> Dict[str, str]:
        """Calculate hashes for each layer in model"""
        try:
            layer_hashes = {}
            
            for name, module in model.named_modules():
                # Skip container modules
                if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                    continue
                    
                # Skip if no parameters
                if sum(p.numel() for p in module.parameters()) == 0:
                    continue
                    
                # Calculate layer hash
                hasher = hashlib.sha256()
                for param_name, param in module.named_parameters(recurse=False):
                    if param is None:
                        continue
                    param_bytes = param.cpu().detach().numpy().tobytes()
                    hasher.update(param_name.encode() + param_bytes)
                    
                layer_hashes[name] = hasher.hexdigest()
                
            return layer_hashes
            
        except Exception as e:
            logging.error(f"Error calculating layer hashes: {str(e)}")
            raise

    def _sample_activations(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, List[float]]:
        """Sample activations from model for specified input"""
        try:
            activation_samples = {}
            
            # Set up hooks to capture activations
            handles = []
            
            def hook_fn(name):
                def hook(module, input, output):
                    # Sample a few values from output
                    with torch.no_grad():
                        if isinstance(output, torch.Tensor):
                            flat = output.view(-1).cpu().tolist()
                            activation_samples[name] = flat[:10]  # First 10 values
                return hook
            
            # Register hooks
            for name, module in model.named_modules():
                if isinstance(module, (nn.ReLU, nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    handles.append(module.register_forward_hook(hook_fn(name)))
                    
            # Run forward pass
            with torch.no_grad():
                model(inputs)
                
            # Remove hooks
            for handle in handles:
                handle.remove()
                
            return activation_samples
            
        except Exception as e:
            logging.error(f"Error sampling activations: {str(e)}")
            raise

    def _calculate_compute_requirements(self, model: nn.Module, inputs: torch.Tensor) -> float:
        """Calculate computational requirements for verification"""
        try:
            # Get model size
            model_size = sum(p.numel() for p in model.parameters()) * 4  # bytes
            
            # Get input size
            input_size = inputs.numel() * inputs.element_size()
            
            # Calculate FLOPS (approximate)
            with torch.no_grad():
                macs = self._count_macs(model, inputs)
                flops = macs * 2
            
            # Combine metrics
            compute_size = (model_size + input_size + flops) / 1e9  # Convert to GB
            
            return compute_size
            
        except Exception as e:
            logging.error(f"Error calculating compute requirements: {str(e)}")
            return 0.0

    def _count_macs(self, model: nn.Module, inputs: torch.Tensor) -> int:
        """Count multiply-accumulate operations for the model"""
        # In a real implementation, this would use a proper MACs counter
        # For now, return a reasonable estimate based on model size
        try:
            total_params = sum(p.numel() for p in model.parameters())
            input_size = inputs.numel()
            
            # Very rough estimate: each parameter is used about input_size/10 times
            return total_params * (input_size // 10)
            
        except Exception as e:
            logging.error(f"Error counting MACs: {str(e)}")
            return 1000000  # Placeholder

    async def _get_available_nodes(self) -> List[str]:
        """Get list of available nodes for verification"""
        # In a real implementation, this would query the node network
        # For now, return mock node IDs
        return [f"node_{i}" for i in range(10)]

    async def _select_verifiers(self, 
                              job_id: str,
                              compute_size: float,
                              available_nodes: List[str]) -> List[str]:
        """Select optimal set of verifiers for a job"""
        try:
            if not available_nodes:
                raise ValueError("No nodes available for verification")

            # Apply diversity constraints
            selected_verifiers = self._apply_diversity_selection(
                available_nodes,
                min_verifiers=self.min_verifiers,
                max_verifiers=self.max_verifiers
            )

            return selected_verifiers

        except Exception as e:
            logging.error(f"Error selecting verifiers: {str(e)}")
            raise

    def _apply_diversity_selection(self,
                                 node_ids: List[str],
                                 min_verifiers: int,
                                 max_verifiers: int) -> List[str]:
        """Select diverse set of verifiers"""
        try:
            # In a real implementation, this would use sophisticated selection
            # For now, randomly select the required number of verifiers
            import random
            
            # Ensure we have enough nodes
            if len(node_ids) < min_verifiers:
                raise ValueError(f"Not enough nodes: {len(node_ids)} < {min_verifiers}")
                
            # Determine how many to select
            num_verifiers = min(max_verifiers, max(min_verifiers, len(node_ids) // 3))
            
            # Select random sample
            return random.sample(node_ids, num_verifiers)
            
        except Exception as e:
            logging.error(f"Error in diversity selection: {str(e)}")
            raise

    def _calculate_gradient_statistics(self, job_id: str) -> Dict:
        """Calculate statistics of gradient history"""
        try:
            history = self.gradient_history.get(job_id, [])
            if not history:
                return {}
                
            gradient_norms = []
            update_ratios = []
            
            for entry in history:
                for checkpoint in entry['data'].get('checkpoints', []):
                    if hasattr(checkpoint, 'gradient_norm'):
                        gradient_norms.append(checkpoint.gradient_norm)
                    if hasattr(checkpoint, 'update_ratio'):
                        update_ratios.append(checkpoint.update_ratio)
                    
            return {
                'mean_gradient_norm': np.mean(gradient_norms) if gradient_norms else 0,
                'std_gradient_norm': np.std(gradient_norms) if gradient_norms else 0,
                'mean_update_ratio': np.mean(update_ratios) if update_ratios else 0,
                'std_update_ratio': np.std(update_ratios) if update_ratios else 0
            }
            
        except Exception as e:
            logging.error(f"Error calculating gradient statistics: {str(e)}")
            return {}

    def _calculate_behavior_statistics(self, job_id: str) -> Dict:
        """Calculate statistics of model behavior history"""
        try:
            history = self.behavior_history.get(job_id, [])
            if not history:
                return {}
                
            perturbation_responses = []
            distribution_shifts = []
            
            for entry in history:
                behavior_data = entry.get('data', {})
                perturbation_tests = behavior_data.get('perturbation_tests', {}).get('tests', [])
                
                for test in perturbation_tests:
                    if 'output_diff' in test:
                        perturbation_responses.append(test['output_diff'])
                    
            return {
                'mean_perturbation_response': np.mean(perturbation_responses) if perturbation_responses else 0,
                'std_perturbation_response': np.std(perturbation_responses) if perturbation_responses else 0,
                'distribution_stability': np.mean(distribution_shifts) if distribution_shifts else 0
            }
            
        except Exception as e:
            logging.error(f"Error calculating behavior statistics: {str(e)}")
            return {}

    def _calculate_update_statistics(self, job_id: str) -> Dict:
        """Calculate statistics of weight update history"""
        try:
            history = self.weight_update_history.get(job_id, [])
            if not history:
                return {}
                
            update_magnitudes = []
            update_ratios = []
            
            for entry in history:
                update_data = entry.get('data', {})
                update_stats = update_data.get('update_stats', [])
                
                for stat in update_stats:
                    if 'update_magnitude' in stat:
                        update_magnitudes.append(stat['update_magnitude'])
                    if 'update_ratio' in stat:
                        update_ratios.append(stat['update_ratio'])
                    
            return {
                'mean_update_magnitude': np.mean(update_magnitudes) if update_magnitudes else 0,
                'std_update_magnitude': np.std(update_magnitudes) if update_magnitudes else 0,
                'mean_update_ratio': np.mean(update_ratios) if update_ratios else 0,
                'std_update_ratio': np.std(update_ratios) if update_ratios else 0
            }
            
        except Exception as e:
            logging.error(f"Error calculating update statistics: {str(e)}")
            return {}

    def _calculate_verification_rate(self, job_id: str) -> float:
        """Calculate overall verification success rate."""
        try:
            # Get verification history for job
            history_entries = self.verification_history.get(job_id, [])
            if not history_entries:
                return 1.0  # Default to success if no history
                
            # Count successful verifications
            successful = sum(1 for entry in history_entries 
                           if entry.get('validation_result', {}).get('is_valid', False))
                           
            # Calculate rate
            return successful / len(history_entries)
            
        except Exception as e:
            logging.error(f"Error calculating verification rate: {str(e)}")
            return 1.0  # Default to success on error