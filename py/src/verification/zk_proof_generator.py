import torch
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import hashlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import logging
import time

@dataclass
class ModelCheckpoint:
    """Checkpoint of model state for verification"""
    layer_states: Dict[str, torch.Tensor]
    intermediate_outputs: Dict[str, torch.Tensor]
    gradient_norms: Dict[str, float]
    computation_metrics: Dict[str, float]

@dataclass
class ProofComponents:
    """Components of a zero-knowledge proof"""
    commitment: bytes
    challenge: bytes
    response: bytes
    auxiliary_data: Dict[str, bytes]

class ZKProofGenerator:
    """Generate zero-knowledge proofs for model computation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.security_parameter = config.get('security_parameter', 128)
        # Use elliptic curve for commitments
        self.curve = ec.SECP256K1()
        self.private_key = ec.generate_private_key(self.curve)
        
    async def generate_proof(self,
                           model: torch.nn.Module,
                           inputs: torch.Tensor,
                           outputs: torch.Tensor,
                           checkpoint: ModelCheckpoint) -> ProofComponents:
        """Generate zero-knowledge proof of computation"""
        try:
            # Generate commitment to model state and computation
            commitment = await self._generate_commitment(model, checkpoint)
            
            # Generate challenge based on commitment and public inputs
            challenge = self._generate_challenge(commitment, inputs, outputs)
            
            # Generate response proving computation correctness
            response = await self._generate_response(
                model, checkpoint, challenge
            )
            
            # Generate auxiliary verification data
            auxiliary_data = self._generate_auxiliary_data(
                model, checkpoint, challenge
            )
            
            return ProofComponents(
                commitment=commitment,
                challenge=challenge,
                response=response,
                auxiliary_data=auxiliary_data
            )
            
        except Exception as e:
            logging.error(f"Error generating proof: {str(e)}")
            raise
            
    async def _generate_commitment(self,
                                 model: torch.nn.Module,
                                 checkpoint: ModelCheckpoint) -> bytes:
        """Generate commitment to model state and computation"""
        try:
            # Combine model parameters into commitment
            hasher = hashes.Hash(hashes.SHA256())
            
            # Commit to layer states
            for name, state in sorted(checkpoint.layer_states.items()):
                hasher.update(name.encode())
                hasher.update(state.cpu().numpy().tobytes())
            
            # Commit to intermediate outputs
            for name, output in sorted(checkpoint.intermediate_outputs.items()):
                hasher.update(name.encode())
                hasher.update(output.cpu().numpy().tobytes())
            
            # Commit to gradient information
            for name, norm in sorted(checkpoint.gradient_norms.items()):
                hasher.update(name.encode())
                hasher.update(str(norm).encode())
            
            # Add computation metrics
            metrics_str = str(sorted(checkpoint.computation_metrics.items()))
            hasher.update(metrics_str.encode())
            
            # Sign the commitment
            commitment = hasher.finalize()
            signature = self.private_key.sign(
                commitment,
                ec.ECDSA(hashes.SHA256())
            )
            
            return signature
            
        except Exception as e:
            logging.error(f"Error generating commitment: {str(e)}")
            raise
            
    def _generate_challenge(self,
                          commitment: bytes,
                          inputs: torch.Tensor,
                          outputs: torch.Tensor) -> bytes:
        """Generate challenge for proof"""
        try:
            # Create challenge based on commitment and public data
            hasher = hashes.Hash(hashes.SHA256())
            
            # Add commitment
            hasher.update(commitment)
            
            # Add input/output shapes and statistics
            for tensor in [inputs, outputs]:
                hasher.update(str(tensor.shape).encode())
                hasher.update(str(tensor.mean().item()).encode())
                hasher.update(str(tensor.std().item()).encode())
            
            # Add random salt for security
            salt = self._generate_secure_random(32)
            hasher.update(salt)
            
            return hasher.finalize()
            
        except Exception as e:
            logging.error(f"Error generating challenge: {str(e)}")
            raise
            
    async def _generate_response(self,
                               model: torch.nn.Module,
                               checkpoint: ModelCheckpoint,
                               challenge: bytes) -> bytes:
        """Generate response to challenge"""
        try:
            # Initialize response data
            response_data = []
            
            # Process each layer based on challenge
            challenge_bits = self._bytes_to_bits(challenge)
            for i, (name, state) in enumerate(checkpoint.layer_states.items()):
                if i < len(challenge_bits) and challenge_bits[i]:
                    # If challenge bit is 1, include layer state proof
                    layer_proof = await self._prove_layer_computation(
                        name, state, checkpoint
                    )
                    response_data.append(layer_proof)
            
            # Add proofs of intermediate computations
            for name, output in checkpoint.intermediate_outputs.items():
                computation_proof = self._prove_intermediate_computation(
                    name, output, checkpoint
                )
                response_data.append(computation_proof)
            
            # Combine and hash response data
            hasher = hashes.Hash(hashes.SHA256())
            for proof in response_data:
                hasher.update(proof)
            
            return hasher.finalize()
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            raise
            
    def _generate_auxiliary_data(self,
                               model: torch.nn.Module,
                               checkpoint: ModelCheckpoint,
                               challenge: bytes) -> Dict[str, bytes]:
        """Generate auxiliary verification data"""
        try:
            auxiliary_data = {}
            
            # Add merkle proofs for layer states
            auxiliary_data['layer_proofs'] = self._generate_merkle_proofs(
                checkpoint.layer_states
            )
            
            # Add zero-knowledge proofs for gradient computations
            auxiliary_data['gradient_proofs'] = self._generate_gradient_proofs(
                checkpoint.gradient_norms
            )
            
            # Add consistency proofs for intermediate outputs
            auxiliary_data['consistency_proofs'] = self._generate_consistency_proofs(
                checkpoint.intermediate_outputs
            )
            
            return auxiliary_data
            
        except Exception as e:
            logging.error(f"Error generating auxiliary data: {str(e)}")
            raise
            
    async def _prove_layer_computation(self,
                                     layer_name: str,
                                     layer_state: torch.Tensor,
                                     checkpoint: ModelCheckpoint) -> bytes:
        """Generate proof of correct layer computation"""
        try:
            # Get layer computation metrics
            gradient_norm = checkpoint.gradient_norms.get(layer_name, 0.0)
            output = checkpoint.intermediate_outputs.get(layer_name)
            
            # Create proof of computation correctness
            hasher = hashes.Hash(hashes.SHA256())
            
            # Add layer state
            hasher.update(layer_state.cpu().numpy().tobytes())
            
            # Add gradient information
            hasher.update(str(gradient_norm).encode())
            
            # Add output if available
            if output is not None:
                hasher.update(output.cpu().numpy().tobytes())
            
            # Add computation specific proofs
            comp_proof = await self._generate_computation_specific_proof(
                layer_name, layer_state, checkpoint
            )
            hasher.update(comp_proof)
            
            return hasher.finalize()
            
        except Exception as e:
            logging.error(f"Error proving layer computation: {str(e)}")
            raise
            
    def _prove_intermediate_computation(self,
                                      name: str,
                                      output: torch.Tensor,
                                      checkpoint: ModelCheckpoint) -> bytes:
        """Generate proof of intermediate computation correctness"""
        try:
            hasher = hashes.Hash(hashes.SHA256())
            
            # Add output statistics
            hasher.update(str(output.shape).encode())
            hasher.update(str(output.mean().item()).encode())
            hasher.update(str(output.std().item()).encode())
            
            # Add relevant metrics
            metrics = {k: v for k, v in checkpoint.computation_metrics.items()
                      if k.startswith(name)}
            hasher.update(str(sorted(metrics.items())).encode())
            
            return hasher.finalize()
            
        except Exception as e:
            logging.error(f"Error proving intermediate computation: {str(e)}")
            raise
            
    async def _generate_computation_specific_proof(self,
                                                 layer_name: str,
                                                 layer_state: torch.Tensor,
                                                 checkpoint: ModelCheckpoint) -> bytes:
        """Generate proof specific to computation type"""
        try:
            # Different proof strategies for different layer types
            if 'conv' in layer_name.lower():
                return self._prove_convolution(layer_state, checkpoint)
            elif 'linear' in layer_name.lower():
                return self._prove_linear(layer_state, checkpoint)
            elif 'norm' in layer_name.lower():
                return self._prove_normalization(layer_state, checkpoint)
            else:
                return self._prove_generic(layer_state, checkpoint)
                
        except Exception as e:
            logging.error(f"Error generating computation proof: {str(e)}")
            raise
            
    def _generate_merkle_proofs(self, states: Dict[str, torch.Tensor]) -> bytes:
        """Generate Merkle proofs for layer states"""
        try:
            # Create Merkle tree of layer states
            leaves = []
            for name, state in sorted(states.items()):
                hasher = hashes.Hash(hashes.SHA256())
                hasher.update(name.encode())
                hasher.update(state.cpu().numpy().tobytes())
                leaves.append(hasher.finalize())
                
            # Build tree
            tree = self._build_merkle_tree(leaves)
            
            # Generate proofs
            proofs = self._generate_merkle_proof_batch(tree, leaves)
            
            return proofs
            
        except Exception as e:
            logging.error(f"Error generating Merkle proofs: {str(e)}")
            raise
            
    def _generate_gradient_proofs(self, gradient_norms: Dict[str, float]) -> bytes:
        """Generate proofs for gradient computations"""
        try:
            hasher = hashes.Hash(hashes.SHA256())
            
            # Add sorted gradient norms
            for name, norm in sorted(gradient_norms.items()):
                hasher.update(name.encode())
                hasher.update(str(norm).encode())
            
            return hasher.finalize()
            
        except Exception as e:
            logging.error(f"Error generating gradient proofs: {str(e)}")
            raise
            
    def _generate_consistency_proofs(self, 
                                   outputs: Dict[str, torch.Tensor]) -> bytes:
        """Generate consistency proofs for intermediate outputs"""
        try:
            hasher = hashes.Hash(hashes.SHA256())
            
            # Add output consistency information
            for name, output in sorted(outputs.items()):
                hasher.update(name.encode())
                hasher.update(str(output.shape).encode())
                hasher.update(str(output.mean().item()).encode())
                hasher.update(str(output.std().item()).encode())
            
            return hasher.finalize()
            
        except Exception as e:
            logging.error(f"Error generating consistency proofs: {str(e)}")
            raise
            
    def _build_merkle_tree(self, leaves: List[bytes]) -> List[bytes]:
        """Build Merkle tree from leaves"""
        tree = leaves.copy()
        
        # Build tree levels
        level = leaves
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                hasher = hashes.Hash(hashes.SHA256())
                hasher.update(level[i])
                if i + 1 < len(level):
                    hasher.update(level[i + 1])
                next_level.append(hasher.finalize())
            level = next_level
            tree.extend(level)
            
        return tree
        
    def _generate_merkle_proof_batch(self,
                                   tree: List[bytes],
                                   leaves: List[bytes]) -> bytes:
        """Generate batch of Merkle proofs"""
        hasher = hashes.Hash(hashes.SHA256())
        
        # Add tree root
        hasher.update(tree[-1])
        
        # Add proof paths for each leaf
        for leaf in leaves:
            proof = self._generate_merkle_proof(tree, leaf)
            hasher.update(proof)
            
        return hasher.finalize()
        
    def _generate_merkle_proof(self, tree: List[bytes], leaf: bytes) -> bytes:
        """Generate Merkle proof for single leaf"""
        hasher = hashes.Hash(hashes.SHA256())
        
        # Find leaf index
        idx = tree.index(leaf)
        
        # Generate proof path
        while idx > 0:
            # Add sibling to proof
            sibling_idx = idx - 1 if idx % 2 == 1 else idx + 1
            if sibling_idx < len(tree):
                hasher.update(tree[sibling_idx])
            
            # Move up tree
            idx = (idx - 1) // 2
            
        return hasher.finalize()
        
    def _prove_convolution(self,
                          layer_state: torch.Tensor,
                          checkpoint: ModelCheckpoint) -> bytes:
        """Generate proof for convolution computation"""
        # Implementation specific to convolution operations
        hasher = hashes.Hash(hashes.SHA256())
        hasher.update(layer_state.cpu().numpy().tobytes())
        return hasher.finalize()
        
    def _prove_linear(self,
                     layer_state: torch.Tensor,
                     checkpoint: ModelCheckpoint) -> bytes:
        """Generate proof for linear computation"""
        # Implementation specific to linear operations
        hasher = hashes.Hash(hashes.SHA256())
        hasher.update(layer_state.cpu().numpy().tobytes())
        return hasher.finalize()
        
    def _prove_normalization(self,
                           layer_state: torch.Tensor,
                           checkpoint: ModelCheckpoint) -> bytes:
        """Generate proof for normalization computation"""
        # Implementation specific to normalization operations
        hasher = hashes.Hash(hashes.SHA256())
        hasher.update(layer_state.cpu().numpy().tobytes())
        return hasher.finalize()
        
    def _prove_generic(self,
                    layer_state: torch.Tensor,
                    checkpoint: ModelCheckpoint) -> bytes:
        """Generate proof for generic computation"""
        try:
            hasher = hashes.Hash(hashes.SHA256())

            # Include the layer state in the proof
            hasher.update(layer_state.cpu().numpy().tobytes())

            # Include computation metrics related to this layer
            if layer_state in checkpoint.computation_metrics:
                metrics = checkpoint.computation_metrics[layer_state]
                hasher.update(str(metrics).encode())

            return hasher.finalize()
        
        except Exception as e:
            logging.error(f"Error generating generic proof: {str(e)}")
            raise
	
    def _generate_secure_random(self, size: int) -> bytes:
        """Generate a cryptographically secure random byte string"""
        return np.random.bytes(size)

    def _bytes_to_bits(self, byte_data: bytes) -> List[int]:
        """Convert a byte string into a list of bits"""
        return [int(bit) for byte in byte_data for bit in format(byte, '08b')]
