import pytest
import torch
import torch.nn as nn
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import time
import numpy as np

from verification_system import AdvancedVerificationSystem, VerificationProof
from verifier_selection import VerifierSelectionSystem
from integrated_verification import IntegratedVerificationManager

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

@pytest.fixture
def test_config():
    return {
        'min_verifiers': 3,
        'max_verifiers': 5,
        'verification_timeout': 30,
        'verification_threshold': 0.67,
        'min_verification_payment': 0.00001,
        'verification_base_rate': 0.0001,
        'target_response_time': 60
    }

@pytest.fixture
def simple_model():
    return SimpleModel()

@pytest.fixture
def test_inputs():
    return torch.randn(4, 10)

@pytest.fixture
def verification_system(test_config):
    return AdvancedVerificationSystem(test_config)

@pytest.fixture
def verifier_selector(test_config):
    return VerifierSelectionSystem(test_config)

@pytest.mark.asyncio
async def test_proof_generation(verification_system, simple_model, test_inputs):
    # Generate sample outputs
    outputs = simple_model(test_inputs)
    metrics = {
        'loss': 0.5,
        'accuracy': 0.85,
        'compute_time': 0.1,
        'gpu_utilization': 75.0,
        'memory_used': 1024
    }
    
    proof = verification_system.generate_proof(
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=metrics,
        job_id="test_job",
        checkpoint_id=1
    )
    
    assert isinstance(proof, VerificationProof)
    assert proof.job_id == "test_job"
    assert proof.checkpoint_id == 1
    assert proof.model_hash is not None
    assert proof.signature is not None

@pytest.mark.asyncio
async def test_proof_verification(verification_system, simple_model, test_inputs):
    # Generate and verify proof
    outputs = simple_model(test_inputs)
    metrics = {
        'loss': 0.5,
        'accuracy': 0.85,
        'compute_time': 0.1,
        'gpu_utilization': 75.0,
        'memory_used': 1024
    }
    
    proof = verification_system.generate_proof(
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=metrics,
        job_id="test_job",
        checkpoint_id=1
    )
    
    # Create challenge inputs
    challenge_inputs = torch.randn(4, 10)
    
    # Verify proof
    is_valid, details = verification_system.verify_proof(
        proof=proof,
        model=simple_model,
        challenge_inputs=challenge_inputs
    )
    
    assert is_valid
    assert 'challenge_input_hash' in details
    assert 'challenge_output_hash' in details

@pytest.mark.asyncio
async def test_verifier_selection(verifier_selector):
    # Mock available nodes
    available_nodes = [f"node_{i}" for i in range(10)]
    
    # Add some stats for nodes
    for node_id in available_nodes:
        await verifier_selector.update_verifier_stats(
            node_id=node_id,
            verification_result=True,
            response_time=0.5,
            metrics={
                'gpu_capacity': 8.0,
                'network_latency': 50.0
            }
        )
    
    # Select verifiers
    selected = await verifier_selector.select_verifiers(
        job_id="test_job",
        compute_size=2.0,
        available_nodes=available_nodes
    )
    
    assert len(selected) >= verifier_selector.min_verifiers
    assert len(selected) <= verifier_selector.max_verifiers
    assert len(set(selected)) == len(selected)  # No duplicates

@pytest.mark.asyncio
async def test_verification_timeout(verification_system, simple_model, test_inputs):
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = asyncio.TimeoutError()
        
        outputs = simple_model(test_inputs)
        metrics = {
            'loss': 0.5,
            'accuracy': 0.85,
            'compute_time': 0.1,
            'gpu_utilization': 75.0,
            'memory_used': 1024
        }
        
        proof = verification_system.generate_proof(
            model=simple_model,
            inputs=test_inputs,
            outputs=outputs,
            metrics=metrics,
            job_id="test_job",
            checkpoint_id=1
        )
        
        with pytest.raises(asyncio.TimeoutError):
            await verification_system.verify_proof(
                proof=proof,
                model=simple_model,
                challenge_inputs=torch.randn(4, 10)
            )

@pytest.mark.asyncio
async def test_verification_metrics(verification_system, simple_model, test_inputs):
    outputs = simple_model(test_inputs)
    
    # Test with invalid metrics
    invalid_metrics = {
        'loss': float('nan'),
        'accuracy': 150.0,  # Invalid accuracy
        'compute_time': -1.0,  # Invalid time
        'gpu_utilization': 120.0,  # Invalid utilization
        'memory_used': -100  # Invalid memory
    }
    
    with pytest.raises(ValueError):
        verification_system.generate_proof(
            model=simple_model,
            inputs=test_inputs,
            outputs=outputs,
            metrics=invalid_metrics,
            job_id="test_job",
            checkpoint_id=1
        )
    
    # Test with valid metrics
    valid_metrics = {
        'loss': 0.5,
        'accuracy': 85.0,
        'compute_time': 0.1,
        'gpu_utilization': 75.0,
        'memory_used': 1024
    }
    
    proof = verification_system.generate_proof(
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=valid_metrics,
        job_id="test_job",
        checkpoint_id=1
    )
    
    assert proof is not None

@pytest.mark.asyncio
async def test_verifier_stats_update(verifier_selector):
    node_id = "test_node"
    
    # Test successful verification
    await verifier_selector.update_verifier_stats(
        node_id=node_id,
        verification_result=True,
        response_time=0.5,
        metrics={
            'gpu_capacity': 8.0,
            'network_latency': 50.0
        }
    )
    
    stats = verifier_selector.verifier_stats[node_id]
    assert stats.successful_verifications == 1
    assert stats.total_verifications == 1
    assert stats.average_response_time == 0.5
    
    # Test failed verification
    await verifier_selector.update_verifier_stats(
        node_id=node_id,
        verification_result=False,
        response_time=1.0,
        metrics={
            'gpu_capacity': 8.0,
            'network_latency': 50.0
        }
    )
    
    stats = verifier_selector.verifier_stats[node_id]
    assert stats.successful_verifications == 1
    assert stats.total_verifications == 2
    assert stats.average_response_time == 0.75

@pytest.mark.asyncio
async def test_integrated_verification(test_config, simple_model, test_inputs):
    # Mock dependencies
    mock_network_client = AsyncMock()
    mock_payment_processor = AsyncMock()
    
    manager = IntegratedVerificationManager(
        config=test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    metrics = {
        'loss': 0.5,
        'accuracy': 0.85,
        'compute_time': 0.1,
        'gpu_utilization': 75.0,
        'memory_used': 1024
    }
    
    # Initialize verification
    verification_task_id = await manager.initialize_verification(
        job_id="test_job",
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=metrics,
        checkpoint_id=1
    )
    
    assert verification_task_id in manager.active_verifications
    
    # Check status
    status = await manager.get_verification_status(verification_task_id)
    assert status['status'] in ['pending', 'in_progress', 'completed']
    
    # Cleanup
    await manager._cleanup_verification_task(verification_task_id)
    assert verification_task_id not in manager.active_verifications