import pytest
import torch
import asyncio
from unittest.mock import patch

@pytest.mark.asyncio
async def test_invalid_metrics_handling(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs
):
    """Test handling of invalid metrics"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    invalid_metrics = {
        'loss': float('nan'),  # Invalid loss
        'accuracy': 120.0,     # Invalid accuracy
        'compute_time': -1.0,  # Invalid time
        'gpu_utilization': 150.0,  # Invalid utilization
        'memory_used': -100    # Invalid memory usage
    }
    
    # Test each invalid metric separately
    for key, invalid_value in invalid_metrics.items():
        metrics = {
            'loss': 0.5,
            'accuracy': 85.0,
            'compute_time': 0.1,
            'gpu_utilization': 75.0,
            'memory_used': 1024
        }
        metrics[key] = invalid_value
        
        with pytest.raises(ValueError, match=f"Invalid {key}"):
            await manager.initialize_verification(
                job_id="test_job",
                model=simple_model,
                inputs=test_inputs,
                outputs=outputs,
                metrics=metrics,
                checkpoint_id=1
            )

@pytest.mark.asyncio
async def test_model_mutation_detection(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test detection of model mutations during verification"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Start verification
    verification_task_id = await manager.initialize_verification(
        job_id="test_job",
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=test_metrics,
        checkpoint_id=1
    )
    
    # Modify model weights during verification
    with torch.no_grad():
        for param in simple_model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    mock_network_client.verify_proof.return_value = {
        'is_valid': False,
        'error': 'Model state mismatch'
    }
    
    # Wait for completion
    status = await wait_for_verification_status(manager, verification_task_id)
    
    assert status['status'] == 'completed'
    assert not status['is_valid']
    assert 'model_state_mismatch' in status.get('errors', [])
    mock_payment_processor.process_verification_payment.assert_not_called()

@pytest.mark.asyncio
async def test_verification_retry_mechanism(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test verification retry mechanism after failures"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Configure retries
    integrated_test_config['verification']['max_retries'] = 2
    integrated_test_config['verification']['retry_delay'] = 0.1
    
    # Track retry attempts
    retry_count = 0
    
    async def fail_then_succeed(*args, **kwargs):
        nonlocal retry_count
        retry_count += 1
        if retry_count <= 2:
            raise Exception("Temporary failure")
        return {'is_valid': True, 'metrics': {'response_time': 0.5}}
    
    mock_network_client.verify_proof.side_effect = fail_then_succeed
    
    # Start verification
    verification_task_id = await manager.initialize_verification(
        job_id="test_job",
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=test_metrics,
        checkpoint_id=1
    )
    
    # Wait for successful completion
    status = await wait_for_verification_status(manager, verification_task_id)
    
    assert status['status'] == 'completed'
    assert status['is_valid']
    assert retry_count == 3  # Two failures + one success
    assert mock_payment_processor.process_verification_payment.call_count == 1

@pytest.mark.asyncio
async def test_verification_state_recovery(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test recovery of verification state after system interruption"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Start verification
    verification_task_id = await manager.initialize_verification(
        job_id="test_job",
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=test_metrics,
        checkpoint_id=1
    )
    
    # Simulate system interruption
    await manager._save_verification_state()
    old_verifications = manager.active_verifications
    manager.active_verifications = {}
    
    # Restore state
    await manager._restore_verification_state()
    
    # Verify state was recovered
    assert verification_task_id in manager.active_verifications
    assert manager.active_verifications[verification_task_id].job_id == "test_job"
    
    # Complete verification
    mock_network_client.verify_proof.return_value = {
        'is_valid': True,
        'metrics': {'response_time': 0.5}
    }
    
    status = await wait_for_verification_status(manager, verification_task_id)
    assert status['status'] == 'completed'
    assert status['is_valid']