import pytest
import asyncio
from typing import List

@pytest.mark.asyncio
async def test_multiple_concurrent_verifications(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test handling multiple simultaneous verification requests"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Configure mock responses with different timings
    response_times = [0.5, 0.3, 0.7]
    mock_responses = [
        {'is_valid': True, 'metrics': {'response_time': rt}}
        for rt in response_times
    ]
    mock_network_client.verify_proof.side_effect = mock_responses
    
    # Start multiple verification tasks
    verification_tasks = []
    for i in range(3):
        task_id = await manager.initialize_verification(
            job_id=f"test_job_{i}",
            model=simple_model,
            inputs=test_inputs,
            outputs=outputs,
            metrics=test_metrics,
            checkpoint_id=i
        )
        verification_tasks.append(task_id)
    
    # Wait for all tasks to complete
    statuses = await asyncio.gather(*[
        wait_for_verification_status(manager, task_id)
        for task_id in verification_tasks
    ])
    
    # Verify results
    for status in statuses:
        assert status['status'] == 'completed'
        assert status['is_valid']
    
    assert mock_payment_processor.process_verification_payment.call_count == len(verification_tasks)

@pytest.mark.asyncio
async def test_resource_limited_concurrent_verifications(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test concurrent verifications with resource limits"""
    
    # Set resource limits
    integrated_test_config['verification']['max_concurrent_tasks'] = 2
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Track task execution order
    execution_order: List[str] = []
    
    async def mock_verify(*args, **kwargs):
        task_id = args[0] if args else kwargs.get('task_id', 'unknown')
        execution_order.append(task_id)
        await asyncio.sleep(0.5)  # Simulate work
        return {'is_valid': True, 'metrics': {'response_time': 0.5}}
    
    mock_network_client.verify_proof.side_effect = mock_verify
    
    # Start more tasks than allowed concurrently
    verification_tasks = []
    for i in range(4):
        task_id = await manager.initialize_verification(
            job_id=f"test_job_{i}",
            model=simple_model,
            inputs=test_inputs,
            outputs=outputs,
            metrics=test_metrics,
            checkpoint_id=i
        )
        verification_tasks.append(task_id)
    
    # Wait for completion
    statuses = await asyncio.gather(*[
        wait_for_verification_status(manager, task_id)
        for task_id in verification_tasks
    ])
    
    # Verify resource limits were respected
    assert len(set(execution_order[:2])) == 2  # First two tasks ran concurrently
    assert len(execution_order) == 4  # All tasks completed
    
    for status in statuses:
        assert status['status'] == 'completed'
        assert status['is_valid']

@pytest.mark.asyncio
async def test_concurrent_verifications_with_failures(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test concurrent verifications with mixed success/failure"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Configure mixed success/failure responses
    mock_responses = [
        {'is_valid': True, 'metrics': {'response_time': 0.5}},
        {'is_valid': False, 'error': 'Verification failed'},
        {'is_valid': True, 'metrics': {'response_time': 0.3}},
        Exception("Network error")
    ]
    response_iter = iter(mock_responses)
    
    async def mixed_responses(*args, **kwargs):
        try:
            response = next(response_iter)
            if isinstance(response, Exception):
                raise response
            return response
        except StopIteration:
            return {'is_valid': False, 'error': 'Unexpected verification'}
    
    mock_network_client.verify_proof.side_effect = mixed_responses
    
    # Start concurrent verifications
    verification_tasks = []
    for i in range(4):
        task_id = await manager.initialize_verification(
            job_id=f"test_job_{i}",
            model=simple_model,
            inputs=test_inputs,
            outputs=outputs,
            metrics=test_metrics,
            checkpoint_id=i
        )
        verification_tasks.append(task_id)
    
    # Wait for all tasks to complete
    statuses = await asyncio.gather(*[
        wait_for_verification_status(manager, task_id)
        for task_id in verification_tasks
    ], return_exceptions=True)
    
    # Count successful verifications
    successful = sum(1 for s in statuses 
                    if not isinstance(s, Exception) and 
                    s['status'] == 'completed' and 
                    s.get('is_valid', False))
    
    assert successful == 2  # Two tasks should succeed
    assert mock_payment_processor.process_verification_payment.call_count == successful