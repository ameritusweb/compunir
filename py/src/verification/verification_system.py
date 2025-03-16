import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.stats import wasserstein_distance
import logging
from dataclasses import dataclass
import time
import hashlib
from collections import defaultdict

@dataclass
class GradientCheckpoint:
    layer_name: str
    gradient_norm: float
    gradient_mean: float
    gradient_std: float
    weight_norm: float
    update_ratio: float  # ratio of update size to weight magnitude

class AdvancedVerificationSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.gradient_history = defaultdict(list)
        self.behavior_history = defaultdict(list)
        self.weight_update_history = defaultdict(list)
        
    async def verify_training_step(self,
                                 model: nn.Module,
                                 optimizer: torch.optim.Optimizer,
                                 inputs: torch.Tensor,
                                 outputs: torch.Tensor,
                                 loss: torch.Tensor,
                                 job_id: str) -> Tuple[bool, Dict]:
        """Comprehensive verification of a training step"""
        try:
            # Run all verification strategies
            gradient_valid, gradient_info = self._verify_gradients(model)
            behavior_valid, behavior_info = self._verify_model_behavior(model, inputs, outputs)
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
            
            return is_valid, {
                'gradient_verification': gradient_info,
                'behavior_verification': behavior_info,
                'update_verification': update_info,
                'loss_verification': loss_info
            }
            
        except Exception as e:
            logging.error(f"Error in advanced verification: {str(e)}")
            raise

    async def verify_data_integrity(self, data_proof: Dict) -> Dict:
        """Verify data integrity and processing steps"""
        try:
            # Validate proof structure
            if not self._validate_data_proof_structure(data_proof):
                return {
                    'is_valid': False,
                    'error': 'Invalid proof structure'
                }
                
            # Verify source hash (if reference data available)
            source_hash_valid = await self._verify_source_hash(
                data_proof['source_hash'],
                data_proof['data_id']
            )
            
            # Verify processing steps
            steps_valid = await self._verify_processing_steps(
                data_proof['processing_steps']
            )
            
            # Verify result hash matches expected (if available)
            result_hash_valid = await self._verify_result_hash(
                data_proof['result_hash'],
                data_proof['data_id']
            )
            
            return {
                'is_valid': source_hash_valid and steps_valid and result_hash_valid,
                'source_valid': source_hash_valid,
                'steps_valid': steps_valid,
                'result_valid': result_hash_valid
            }
            
        except Exception as e:
            logging.error(f"Data verification error: {str(e)}")
            return {
                'is_valid': False,
                'error': str(e)
            }

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

    def _verify_model_behavior(self, 
                             model: nn.Module,
                             inputs: torch.Tensor,
                             outputs: torch.Tensor) -> Tuple[bool, Dict]:
        """Verify model behavior patterns"""
        try:
            # Generate perturbation tests
            perturbation_results = self._test_input_perturbations(model, inputs)
            
            # Check output distribution
            distribution_valid = self._verify_output_distribution(outputs)
            
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

    def _verify_output_distribution(self, outputs: torch.Tensor) -> bool:
        """Verify properties of the output distribution"""
        try:
            # Calculate distribution statistics
            output_mean = torch.mean(outputs).item()
            output_std = torch.std(outputs).item()
            
            # Verify basic statistics
            if not np.isfinite(output_mean) or not np.isfinite(output_std):
                return False
                
            # Calculate distribution metrics
            if len(self.behavior_history) > 0:
                # Compare with historical distributions using Wasserstein distance
                historical_outputs = torch.stack(list(self.behavior_history[-10:]))
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

    def _calculate_proposed_update(self, param: torch.Tensor, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """Calculate the proposed parameter update"""
        try:
            if isinstance(optimizer, torch.optim.Adam):
                # For Adam optimizer
                state = optimizer.state[param]
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step = state['step']
                
                bias_correction1 = 1 - optimizer.defaults['betas'][0] ** step
                bias_correction2 = 1 - optimizer.defaults['betas'][1] ** step
                
                denominator = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(optimizer.defaults['eps'])
                step_size = optimizer.defaults['lr'] / bias_correction1
                
                return -step_size * exp_avg / denominator
                
            else:
                # For SGD and other optimizers
                return -optimizer.defaults['lr'] * param.grad
                
        except Exception as e:
            logging.error(f"Error calculating proposed update: {str(e)}")
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

    def _store_verification_data(self, job_id: str, verification_data: Dict):
        """Store verification data for historical analysis"""
        try:
            # Store data in respective histories
            self.gradient_history[job_id].append({
                'timestamp': time.time(),
                'data': verification_data['gradients']
            })
            
            self.behavior_history[job_id].append({
                'timestamp': time.time(),
                'data': verification_data['behavior']
            })
            
            self.weight_update_history[job_id].append({
                'timestamp': time.time(),
                'data': verification_data['updates']
            })
            
            # Maintain history size
            max_history = self.config.get('max_history_size', 1000)
            if len(self.gradient_history[job_id]) > max_history:
                self.gradient_history[job_id] = self.gradient_history[job_id][-max_history:]
            if len(self.behavior_history[job_id]) > max_history:
                self.behavior_history[job_id] = self.behavior_history[job_id][-max_history:]
            if len(self.weight_update_history[job_id]) > max_history:
                self.weight_update_history[job_id] = self.weight_update_history[job_id][-max_history:]
                
        except Exception as e:
            logging.error(f"Error storing verification data: {str(e)}")
            raise

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

    def _calculate_gradient_statistics(self, job_id: str) -> Dict:
        """Calculate statistics of gradient history"""
        try:
            history = self.gradient_history[job_id]
            if not history:
                return {}
                
            gradient_norms = []
            update_ratios = []
            
            for entry in history:
                for checkpoint in entry['data']['checkpoints']:
                    gradient_norms.append(checkpoint.gradient_norm)
                    update_ratios.append(checkpoint.update_ratio)
                    
            return {
                'mean_gradient_norm': np.mean(gradient_norms),
                'std_gradient_norm': np.std(gradient_norms),
                'mean_update_ratio': np.mean(update_ratios),
                'std_update_ratio': np.std(update_ratios)
            }
            
        except Exception as e:
            logging.error(f"Error calculating gradient statistics: {str(e)}")
            return {}

    def _calculate_behavior_statistics(self, job_id: str) -> Dict:
        """Calculate statistics of model behavior history"""
        try:
            history = self.behavior_history[job_id]
            if not history:
                return {}
                
            perturbation_responses = []
            distribution_shifts = []
            
            for entry in history:
                for test in entry['data']['perturbation_tests']['tests']:
                    perturbation_responses.append(test['output_diff'])
                    
            return {
                'mean_perturbation_response': np.mean(perturbation_responses),
                'std_perturbation_response': np.std(perturbation_responses),
                'distribution_stability': np.mean(distribution_shifts) if distribution_shifts else 0
            }
            
        except Exception as e:
            logging.error(f"Error calculating behavior statistics: {str(e)}")
            return {}

    def _calculate_update_statistics(self, job_id: str) -> Dict:
        """Calculate statistics of weight update history"""
        try:
            history = self.weight_update_history[job_id]
            if not history:
                return {}
                
            update_magnitudes = []
            update_ratios = []
            
            for entry in history:
                for stat in entry['data']['update_stats']:
                    update_magnitudes.append(stat['update_magnitude'])
                    update_ratios.append(stat['update_ratio'])
                    
            return {
                'mean_update_magnitude': np.mean(update_magnitudes),
                'std_update_magnitude': np.std(update_magnitudes),
                'mean_update_ratio': np.mean(update_ratios),
                'std_update_ratio': np.std(update_ratios)
            }
            
        except Exception as e:
            logging.error(f"Error calculating update statistics: {str(e)}")
            return {}

    def _calculate_verification_rate(self, job_id: str) -> float:
        """Calculate the overall verification success rate"""
        try:
            total_verifications = 0
            successful_verifications = 0
            
            # Combine all verification histories
            all_histories = [
                self.gradient_history[job_id],
                self.behavior_history[job_id],
                self.weight_update_history[job_id]
            ]
            
            for history in all_histories:
                for entry in history:
                    total_verifications += 1
                    if entry.get('data', {}).get('valid', False):
                        successful_verifications += 1
                        
            return successful_verifications / total_verifications if total_verifications > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating verification rate: {str(e)}")
            return 0.0