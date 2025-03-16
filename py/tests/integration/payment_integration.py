import pytest
import asyncio
from decimal import Decimal
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_successful_verification_payment(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test successful payment processing after verification"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Configure successful verification
    mock_network_client.verify_proof.return_value = {
        'is_valid': True,
        'metrics': {
            'response_time': 0.5,
            'gpu_utilization': 80.0,
            'compute_time': 100
        }
    }
    
    # Configure payment response
    mock_payment_processor.process_verification_payment.return_value = {
        'amount': Decimal('0.1'),
        'transaction_id': 'test_tx_123',
        'timestamp': 1234567890
    }
    
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
    
    # Verify payment was processed correctly
    mock_payment_processor.process_verification_payment.assert_called_once()
    payment_args = mock_payment_processor.process_verification_payment.call_args[1]
    assert payment_args['job_id'] == "test_job"
    assert isinstance(payment_args['amount'], Decimal)

@pytest.mark.asyncio
async def test_payment_calculation_based_on_metrics(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test payment calculation based on performance metrics"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Test different performance scenarios
    scenarios = [
        {
            'metrics': {'response_time': 0.1, 'gpu_utilization': 95.0, 'compute_time': 100},
            'expected_multiplier': 1.0  # High performance
        },
        {
            'metrics': {'response_time': 1.0, 'gpu_utilization': 50.0, 'compute_time': 100},
            'expected_multiplier': 0.7  # Medium performance
        },
        {
            'metrics': {'response_time': 2.0, 'gpu_utilization': 30.0, 'compute_time': 100},
            'expected_multiplier': 0.5  # Low performance
        }
    ]
    
    payment_amounts = []
    
    for scenario in scenarios:
        mock_network_client.verify_proof.return_value = {
            'is_valid': True,
            'metrics': scenario['metrics']
        }
        
        verification_task_id = await manager.initialize_verification(
            job_id=f"test_job_{len(payment_amounts)}",
            model=simple_model,
            inputs=test_inputs,
            outputs=outputs,
            metrics=test_metrics,
            checkpoint_id=1
        )
        
        await wait_for_verification_status(manager, verification_task_id)
        
        # Extract payment amount
        payment_args = mock_payment_processor.process_verification_payment.call_args[1]
        payment_amounts.append(payment_args['amount'])
    
    # Verify payments scale with performance
    assert payment_amounts[0] > payment_amounts[1] > payment_amounts[2]

@pytest.mark.asyncio
async def test_payment_failure_handling(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test handling of payment processing failures"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Configure successful verification
    mock_network_client.verify_proof.return_value = {
        'is_valid': True,
        'metrics': {'response_time': 0.5}
    }
    
    # Configure payment failure then success
    payment_attempts = 0
    async def payment_failure_recovery(*args, **kwargs):
        nonlocal payment_attempts
        payment_attempts += 1
        if payment_attempts == 1:
            raise Exception("Payment processing failed")
        return {
            'amount': Decimal('0.1'),
            'transaction_id': 'test_tx_123',
            'timestamp': 1234567890
        }
    
    mock_payment_processor.process_verification_payment.side_effect = payment_failure_recovery
    
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
    assert payment_attempts == 2  # Verify payment was retried
    assert mock_payment_processor.process_verification_payment.call_count == 2

@pytest.mark.asyncio
async def test_insufficient_funds_handling(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test handling of insufficient funds for payment"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Configure successful verification
    mock_network_client.verify_proof.return_value = {
        'is_valid': True,
        'metrics': {'response_time': 0.5}
    }
    
    # Configure insufficient funds error
    mock_payment_processor.process_verification_payment.side_effect = \
        ValueError("Insufficient funds in escrow")
    
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
    assert 'payment_error' in status
    assert 'insufficient_funds' in status['payment_error']

@pytest.mark.asyncio
async def test_payment_timeout_handling(
    integrated_test_config,
    mock_network_client,
    mock_payment_processor,
    simple_model,
    test_inputs,
    test_metrics
):
    """Test handling of payment processing timeouts"""
    
    manager = IntegratedVerificationManager(
        config=integrated_test_config,
        network_client=mock_network_client,
        payment_processor=mock_payment_processor
    )
    
    outputs = simple_model(test_inputs)
    
    # Configure successful verification
    mock_network_client.verify_proof.return_value = {
        'is_valid': True,
        'metrics': {'response_time': 0.5}
    }
    
    # Configure payment timeout
    async def payment_timeout(*args, **kwargs):
        await asyncio.sleep(2)  # Delay longer than timeout
        return {'amount': Decimal('0.1'), 'transaction_id': 'test_tx'}
    
    mock_payment_processor.process_verification_payment.side_effect = payment_timeout
    
    # Set short payment timeout
    integrated_test_config['payment']['payment_timeout'] = 1
    
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
    assert 'payment_error' in status
    assert 'timeout' in status['payment_error']