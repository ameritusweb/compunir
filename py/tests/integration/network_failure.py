import pytest
import asyncio
from unittest.mock import patch
from aiohttp import ClientError, ServerDisconnectedError, ClientTimeout

@pytest.mark.asyncio
async def test_intermittent_network_failures(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test handling of intermittent network failures during verification"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Configure network failure pattern
    failure_count = 0
    async def intermittent_failure(*args, **kwargs):
        nonlocal failure_count
        failure_count += 1
        if failure_count % 2 == 1:  # Fail on odd attempts
            raise ClientError("Connection reset by peer")
        return {'is_valid': True, 'metrics': {'response_time': 0.5}}
    
    mock_network_client.verify_proof.side_effect = intermittent_failure
    
    # Start verification
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
    
    assert status['status'] == 'completed'
    assert status['is_valid']
    assert failure_count > 1  # Verify multiple attempts were made

@pytest.mark.asyncio
async def test_network_latency_handling(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test handling of high network latency"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Simulate high latency responses
    async def high_latency_response(*args, **kwargs):
        await asyncio.sleep(2)  # Simulate network delay
        return {'is_valid': True, 'metrics': {'response_time': 2.0}}
    
    mock_network_client.verify_proof.side_effect = high_latency_response
    
    # Set aggressive timeout
    integrated_test_config['network']['request_timeout'] = 1
    
    # Start verification
    verification_task_id = await manager.initialize_verification(
        job_id="test_job",
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=test_metrics,
        checkpoint_id=1
    )
    
    # Verify timeout handling
    status = await wait_for_verification_status(manager, verification_task_id)
    assert 'network_timeout' in status.get('errors', [])

@pytest.mark.asyncio
async def test_verifier_disconnection(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test handling of verifier node disconnections"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Track verifier status
    connected_verifiers = set(range(5))
    
    async def simulate_disconnection(*args, **kwargs):
        verifier_id = kwargs.get('verifier_id', 0)
        if verifier_id not in connected_verifiers:
            raise ServerDisconnectedError()
        if len(connected_verifiers) > 2:  # Simulate gradual disconnections
            connected_verifiers.remove(verifier_id)
        return {'is_valid': True, 'metrics': {'response_time': 0.5}}
    
    mock_network_client.verify_proof.side_effect = simulate_disconnection
    
    # Start verification
    verification_task_id = await manager.initialize_verification(
        job_id="test_job",
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=test_metrics,
        checkpoint_id=1
    )
    
    # Wait for completion or failure
    status = await wait_for_verification_status(manager, verification_task_id)
    
    # Verify handling of reduced verifier set
    assert len(connected_verifiers) == 2  # Some verifiers remained
    assert status['status'] in ['completed', 'partial_success']

@pytest.mark.asyncio
async def test_network_partition_recovery(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test recovery from network partitions"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Simulate network partition
    partition_active = True
    async def partition_behavior(*args, **kwargs):
        if partition_active:
            raise ClientError("Network partition")
        return {'is_valid': True, 'metrics': {'response_time': 0.5}}
    
    mock_network_client.verify_proof.side_effect = partition_behavior
    
    # Start verification
    verification_task_id = await manager.initialize_verification(
        job_id="test_job",
        model=simple_model,
        inputs=test_inputs,
        outputs=outputs,
        metrics=test_metrics,
        checkpoint_id=1
    )
    
    # Wait for partition handling
    await asyncio.sleep(1)
    
    # Resolve partition
    partition_active = False
    
    # Wait for recovery
    status = await wait_for_verification_status(manager, verification_task_id)
    
    assert status['status'] == 'completed'
    assert status['is_valid']
    assert 'recovered_from_partition' in status.get('events', [])

@pytest.mark.asyncio
async def test_partial_network_failure(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test handling of partial network failures affecting some verifiers"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Simulate failures for specific verifiers
    failed_verifiers = {1, 3}  # Verifiers that will experience failures
    
    async def partial_failure(*args, **kwargs):
        verifier_id = int(kwargs.get('verifier_id', '0').split('_')[1])
        if verifier_id in failed_verifiers:
            raise ClientError("Connection failed")
        return {'is_valid': True, 'metrics': {'response_time': 0.5}}
    
    mock_network_client.verify_proof.side_effect = partial_failure
    
    # Start verification
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
    
    # Verify that we succeeded with partial verifier set
    assert status['status'] == 'completed'
    assert status['is_valid']
    assert len(status.get('failed_verifiers', [])) == len(failed_verifiers)