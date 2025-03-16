import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
import time
import json

from payment_system import PaymentProcessor, MoneroWallet, MoneroPaymentProcessor

@pytest.fixture
def wallet_config():
    return {
        'rpc_url': 'http://localhost:18082',
        'rpc_username': 'test_user',
        'rpc_password': 'test_pass',
        'min_payment': Decimal('0.01'),
        'base_rate': Decimal('0.001')
    }

@pytest.fixture
def monero_wallet(wallet_config):
    return MoneroWallet(wallet_config)

@pytest.fixture
def payment_processor(wallet_config):
    return PaymentProcessor(wallet_config)

@pytest.mark.asyncio
async def test_monero_wallet_create_address(monero_wallet):
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(
            return_value={
                'result': {
                    'address': 'test_address_123'
                }
            }
        )
        
        async with monero_wallet:
            address = await monero_wallet.create_address()
            
        assert address == 'test_address_123'
        mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_monero_wallet_get_balance(monero_wallet):
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(
            return_value={
                'result': {
                    'balance': 1000000000000  # 1 XMR in atomic units
                }
            }
        )
        
        async with monero_wallet:
            balance = await monero_wallet.get_balance()
            
        assert balance == Decimal('1.0')
        mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_monero_wallet_transfer(monero_wallet):
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(
            return_value={
                'result': {
                    'tx_hash': 'test_tx_hash',
                    'tx_key': 'test_tx_key',
                    'fee': 1000000000  # 0.001 XMR in atomic units
                }
            }
        )
        
        async with monero_wallet:
            result = await monero_wallet.transfer(
                dest_address='test_dest_address',
                amount=Decimal('1.0')
            )
            
        assert result['tx_id'] == 'test_tx_hash'
        assert result['amount'] == Decimal('1.0')
        assert result['fee'] == Decimal('0.001')
        mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_payment_processor_create_escrow(payment_processor):
    amount = Decimal('1.0')
    job_id = 'test_job'
    
    # Test escrow creation
    escrow_details = await payment_processor.create_escrow(amount, job_id)
    
    assert job_id in payment_processor.payment_records
    assert escrow_details['total_amount'] > amount  # Should include buffer
    assert 'escrow_address' in escrow_details
    assert 'transaction_id' in escrow_details

@pytest.mark.asyncio
async def test_payment_processor_process_verification(payment_processor):
    job_id = 'test_job'
    proof_data = {
        'compute_time': 100,
        'gpu_utilization': 80,
        'success_rate': 95
    }
    node_id = 'test_node'
    
    # Create initial escrow
    await payment_processor.create_escrow(Decimal('1.0'), job_id)
    
    # Process verification payment
    payment = await payment_processor.process_verification_payment(
        job_id=job_id,
        proof_data=proof_data,
        node_id=node_id
    )
    
    assert payment['amount'] > 0
    assert payment['transaction_id'] is not None
    assert payment['timestamp'] is not None

@pytest.mark.asyncio
async def test_payment_calculation(payment_processor):
    proof_data = {
        'compute_time': 100,  # seconds
        'gpu_utilization': 80,  # percentage
        'success_rate': 95  # percentage
    }
    
    payment = payment_processor._calculate_payment_amount(proof_data)
    
    assert payment >= payment_processor.min_payment
    assert isinstance(payment, Decimal)

@pytest.mark.asyncio
async def test_insufficient_escrow_funds(payment_processor):
    job_id = 'test_job'
    small_amount = Decimal('0.001')
    
    # Create escrow with small amount
    await payment_processor.create_escrow(small_amount, job_id)
    
    # Try to process a large payment
    proof_data = {
        'compute_time': 1000,
        'gpu_utilization': 100,
        'success_rate': 100
    }
    
    with pytest.raises(ValueError, match="Insufficient funds in escrow"):
        await payment_processor.process_verification_payment(
            job_id=job_id,
            proof_data=proof_data,
            node_id='test_node'
        )

@pytest.mark.asyncio
async def test_payment_record_tracking(payment_processor):
    job_id = 'test_job'
    initial_amount = Decimal('1.0')
    
    # Create escrow
    await payment_processor.create_escrow(initial_amount, job_id)
    
    # Process multiple payments
    payments = []
    for i in range(3):
        proof_data = {
            'compute_time': 100,
            'gpu_utilization': 80,
            'success_rate': 95
        }
        
        payment = await payment_processor.process_verification_payment(
            job_id=job_id,
            proof_data=proof_data,
            node_id=f'test_node_{i}'
        )
        payments.append(payment)
    
    # Check payment record
    record = payment_processor.payment_records[job_id]
    assert record['total_amount'] == initial_amount
    assert len(record.get('transfers', [])) == 3
    assert sum(Decimal(p['amount']) for p in record['transfers']) <= initial_amount

@pytest.mark.asyncio
async def test_failed_transaction_handling(payment_processor):
    with patch.object(MoneroWallet, 'transfer', side_effect=Exception("Transaction failed")):
        job_id = 'test_job'
        await payment_processor.create_escrow(Decimal('1.0'), job_id)
        
        proof_data = {
            'compute_time': 100,
            'gpu_utilization': 80,
            'success_rate': 95
        }
        
        with pytest.raises(Exception, match="Transaction failed"):
            await payment_processor.process_verification_payment(
                job_id=job_id,
                proof_data=proof_data,
                node_id='test_node'
            )
        
        # Check that payment record wasn't updated
        assert len(payment_processor.payment_records[job_id].get('transfers', [])) == 0

@pytest.mark.asyncio
async def test_escrow_address_generation(payment_processor):
    job_id = 'test_job'
    
    # Generate multiple escrow addresses
    addresses = []
    for _ in range(3):
        escrow_details = await payment_processor.create_escrow(Decimal('1.0'), f'{job_id}_{_}')
        addresses.append(escrow_details['escrow_address'])
    
    # Verify addresses are unique
    assert len(set(addresses)) == len(addresses)

@pytest.mark.asyncio
async def test_payment_verification(payment_processor):
    job_id = 'test_job'
    initial_amount = Decimal('1.0')
    
    # Create escrow
    escrow_details = await payment_processor.create_escrow(initial_amount, job_id)
    
    # Verify payment received
    verification = await payment_processor.verify_payment(
        job_id=job_id,
        expected_amount=initial_amount
    )
    
    assert verification['verified']
    assert verification['amount'] >= initial_amount
    assert verification['transaction_id'] == escrow_details['transaction_id']

@pytest.mark.asyncio
async def test_payment_batching(payment_processor):
    job_id = 'test_job'
    initial_amount = Decimal('1.0')
    await payment_processor.create_escrow(initial_amount, job_id)
    
    # Create multiple small payments
    small_payments = []
    for i in range(5):
        proof_data = {
            'compute_time': 20,
            'gpu_utilization': 80,
            'success_rate': 95
        }
        
        payment = await payment_processor._calculate_payment_amount(proof_data)
        small_payments.append({
            'node_id': f'test_node_{i}',
            'amount': payment
        })
    
    # Process batch payment
    batch_result = await payment_processor._process_batch_payment(
        job_id=job_id,
        payments=small_payments
    )
    
    assert len(batch_result['transactions']) == 1
    assert batch_result['total_amount'] == sum(p['amount'] for p in small_payments)

@pytest.mark.asyncio
async def test_network_fee_estimation(payment_processor):
    # Test fee estimation under different network conditions
    with patch.object(MoneroWallet, '_rpc_call', new_callable=AsyncMock) as mock_rpc:
        # Simulate normal network conditions
        mock_rpc.return_value = {
            'fee_estimate': 1000000000  # 0.001 XMR in atomic units
        }
        
        fee = await payment_processor._estimate_network_fees()
        assert fee == Decimal('0.001')
        
        # Simulate high network load
        mock_rpc.return_value = {
            'fee_estimate': 2000000000  # 0.002 XMR in atomic units
        }
        
        fee = await payment_processor._estimate_network_fees()
        assert fee == Decimal('0.002')

@pytest.mark.asyncio
async def test_payment_timeout_handling(payment_processor):
    job_id = 'test_job'
    
    # Set short timeout
    payment_processor.config['payment_timeout'] = 1
    
    with patch.object(MoneroWallet, 'transfer', new_callable=AsyncMock) as mock_transfer:
        # Simulate slow transaction
        async def slow_transfer(*args, **kwargs):
            await asyncio.sleep(2)
            return {'tx_id': 'test_tx', 'amount': Decimal('1.0')}
        
        mock_transfer.side_effect = slow_transfer
        
        await payment_processor.create_escrow(Decimal('1.0'), job_id)
        
        with pytest.raises(asyncio.TimeoutError):
            await payment_processor.process_verification_payment(
                job_id=job_id,
                proof_data={
                    'compute_time': 100,
                    'gpu_utilization': 80,
                    'success_rate': 95
                },
                node_id='test_node'
            )

@pytest.mark.asyncio
async def test_concurrent_payments(payment_processor):
    job_id = 'test_job'
    initial_amount = Decimal('2.0')
    await payment_processor.create_escrow(initial_amount, job_id)
    
    # Create multiple concurrent payment tasks
    async def make_payment(node_id):
        return await payment_processor.process_verification_payment(
            job_id=job_id,
            proof_data={
                'compute_time': 100,
                'gpu_utilization': 80,
                'success_rate': 95
            },
            node_id=node_id
        )
    
    payment_tasks = [
        make_payment(f'test_node_{i}')
        for i in range(5)
    ]
    
    # Process payments concurrently
    results = await asyncio.gather(*payment_tasks, return_exceptions=True)
    
    successful_payments = [r for r in results if not isinstance(r, Exception)]
    assert len(successful_payments) > 0
    assert sum(Decimal(p['amount']) for p in successful_payments) <= initial_amount

@pytest.mark.asyncio
async def test_payment_record_persistence(payment_processor):
    job_id = 'test_job'
    
    # Create payment record
    await payment_processor.create_escrow(Decimal('1.0'), job_id)
    
    # Simulate saving to persistent storage
    saved_records = payment_processor._save_payment_records()
    
    # Clear in-memory records
    payment_processor.payment_records.clear()
    
    # Restore from saved records
    payment_processor._load_payment_records(saved_records)
    
    assert job_id in payment_processor.payment_records
    assert payment_processor.payment_records[job_id]['status'] == 'active'