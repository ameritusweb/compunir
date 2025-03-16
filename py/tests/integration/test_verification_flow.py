import pytest
import torch
import asyncio
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from verification_system import AdvancedVerificationSystem
from payment_system import PaymentProcessor
from integrated_verification import IntegratedVerificationManager
from node_network.client import NodeNetworkClient

# Base test fixtures
@pytest.fixture
def integrated_test_config():
    return {
        'verification': {
            'min_verifiers': 3,
            'max_verifiers': 5,
            'verification_timeout': 30,
            'verification_threshold': 0.67
        },
        'payment': {
            'min_payment': Decimal('0.01'),
            'base_rate': Decimal('0.001'),
            'payment_timeout': 60
        },
        'network': {
            'retry_count': 3,
            'retry_delay': 1
        }
    }

@pytest.fixture
def mock_network_client():
    client = AsyncMock(spec=NodeNetworkClient)
    client.get_available_nodes.return_value = [f"node_{i}" for i in range(5)]
    return client

@pytest.fixture
def mock_payment_processor():
    processor = AsyncMock(spec=PaymentProcessor)
    processor.create_escrow.return_value = {
        'escrow_address': 'test_escrow',
        'transaction_id': 'test_tx',
        'total_amount': Decimal('1.0')
    }
    return processor

@pytest.fixture
def test_metrics():
    return {
        'loss': 0.5,
        'accuracy': 0.85,
        'compute_time': 0.1,
        'gpu_utilization': 75.0,
        'memory_used': 1024
    }

# Helper functions for tests
async def wait_for_verification_status(manager, verification_task_id, target_status=None):
    """Helper function to wait for verification completion"""
    while True:
        status = await manager.get_verification_status(verification_task_id)
        if target_status:
            if status['status'] == target_status:
                return status
        elif status['status'] in ['completed', 'timeout', 'error']:
            return status
        await asyncio.sleep(0.1)

# Basic verification flow tests
@pytest.mark.asyncio
async def test_basic_verification_flow(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test basic verification flow with successful completion"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    # Generate model outputs
    outputs = simple_model(test_inputs)
    
    # Mock successful verification
    mock_network_client.verify_proof.return_value = {
        'is_valid': True,
        'metrics': {'response_time': 0.5, 'gpu_utilization': 80.0}
    }
    
    # Initialize verification
    verification_task_id = await manager.initialize_verification(
        job_id="test_job",
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=test_metrics,
        checkpoint_id=1
    )
    
    # Wait for completion
    status = await wait_for_verification_status(manager, verification_task_id)
    
    # Verify results
    assert status['status'] == 'completed'
    assert status['is_valid']
    mock_payment_processor.process_verification_payment.assert_called_once()

@pytest.mark.asyncio
async def test_verification_with_timeout(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test verification timeout and recovery"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    # Set short timeout
    integrated_test_config['verification']['verification_timeout'] = 1
    
    # Mock timeout behavior
    async def verify_with_timeout(*args, **kwargs):
        await asyncio.sleep(2)
        return {'is_valid': True, 'metrics': {'response_time': 0.5}}
    
    mock_network_client.verify_proof.side_effect = verify_with_timeout
    
    # Initialize verification
    outputs = simple_model(test_inputs)
    verification_task_id = await manager.initialize_verification(
        job_id="test_job",
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=test_metrics,
        checkpoint_id=1
    )
    
    # Wait for timeout
    status = await wait_for_verification_status(manager, verification_task_id)
    assert status['status'] == 'timeout'
    
    # Reset timeout and retry
    integrated_test_config['verification']['verification_timeout'] = 10
    mock_network_client.verify_proof.side_effect = lambda *args, **kwargs: {
        'is_valid': True,
        'metrics': {'response_time': 0.5}
    }
    
    # Retry verification
    new_verification_task_id = await manager.initialize_verification(
        job_id="test_job",
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=test_metrics,
        checkpoint_id=1
    )
    
    # Wait for successful completion
    status = await wait_for_verification_status(manager, new_verification_task_id)
    assert status['status'] == 'completed'
    assert status['is_valid']