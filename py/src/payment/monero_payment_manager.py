import asyncio
import logging
from decimal import Decimal
from typing import Dict, Optional, List
import time
from dataclasses import dataclass

from .monero_wallet import MoneroWallet, MoneroPaymentProcessor
from ..verification import VerificationSystem
from ..core.node_manager import NodeManager

@dataclass
class PaymentRecord:
    payment_id: str
    amount: Decimal
    recipient_id: str
    recipient_address: str
    verification_id: Optional[str]
    job_id: Optional[str]
    tx_id: Optional[str]
    status: str  # 'pending', 'processing', 'completed', 'failed'
    timestamp: float
    error: Optional[str] = None

class MoneroPaymentManager:
    """Manages Monero payments and integrates with verification system"""
    
    def __init__(self, config: Dict, 
                 wallet: MoneroWallet,
                 verification_system: VerificationSystem,
                 node_manager: NodeManager):
        self.config = config
        self.wallet = wallet
        self.verification_system = verification_system
        self.node_manager = node_manager
        
        # Initialize payment processor
        self.payment_processor = MoneroPaymentProcessor(config.get('wallet', {}))
        
        # Payment tracking
        self.payment_records: Dict[str, PaymentRecord] = {}
        self.pending_transactions: Dict[str, str] = {}  # tx_id -> payment_id
        self.payment_queue = asyncio.Queue()
        
        # Configure payment settings
        payment_config = config.get('payment', {})
        self.min_payment = Decimal(str(payment_config.get('min_payment', '0.01')))
        self.base_rate = Decimal(str(payment_config.get('base_rate', '0.001')))
        self.max_payment = Decimal(str(payment_config.get('max_payment', '1.0')))
        self.processing_task = None
        
        logging.info("Initialized MoneroPaymentManager")

    async def start(self):
        """Start payment manager"""
        self.processing_task = asyncio.create_task(self._process_payment_queue())
        asyncio.create_task(self._monitor_transactions())
        logging.info("Started MoneroPaymentManager")

    async def stop(self):
        """Stop payment manager"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

    async def process_verification_payment(self, 
                                        verification_id: str,
                                        node_id: str,
                                        amount: Decimal,
                                        verification_result: Dict) -> Dict:
        """Process payment for successful verification"""
        try:
            # Get node's wallet address
            node_info = await self.node_manager.get_node_info(node_id)
            if not node_info or 'wallet_address' not in node_info:
                raise ValueError(f"No wallet address found for node {node_id}")

            # Create payment record
            payment_id = f"pay_{verification_id}_{int(time.time())}"
            payment_record = PaymentRecord(
                payment_id=payment_id,
                amount=amount,
                recipient_id=node_id,
                recipient_address=node_info['wallet_address'],
                verification_id=verification_id,
                job_id=verification_result.get('job_id'),
                tx_id=None,
                status='pending',
                timestamp=time.time()
            )

            # Store record
            self.payment_records[payment_id] = payment_record

            # Queue payment
            await self.payment_queue.put(payment_record)

            return {
                'payment_id': payment_id,
                'status': 'pending',
                'amount': str(amount),
                'recipient': node_id
            }

        except Exception as e:
            logging.error(f"Error processing verification payment: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def check_payment_status(self, payment_id: str) -> Dict:
        """Check status of a specific payment"""
        try:
            if payment_id not in self.payment_records:
                return {'status': 'not_found'}

            record = self.payment_records[payment_id]
            
            # If completed or failed, return final status
            if record.status in ['completed', 'failed']:
                return {
                    'payment_id': payment_id,
                    'status': record.status,
                    'amount': str(record.amount),
                    'recipient': record.recipient_id,
                    'tx_id': record.tx_id,
                    'error': record.error
                }

            # If transaction exists, check its status
            if record.tx_id:
                tx_status = await self.wallet.get_transfer_by_txid(record.tx_id)
                if tx_status:
                    return {
                        'payment_id': payment_id,
                        'status': 'processing',
                        'tx_status': tx_status['status'],
                        'confirmations': tx_status.get('confirmations', 0),
                        'amount': str(record.amount),
                        'recipient': record.recipient_id
                    }

            return {
                'payment_id': payment_id,
                'status': record.status,
                'amount': str(record.amount),
                'recipient': record.recipient_id
            }

        except Exception as e:
            logging.error(f"Error checking payment status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def get_node_earnings(self, node_id: str) -> Dict:
        """Get earnings information for a node"""
        try:
            # Find all payments for node
            node_payments = [
                record for record in self.payment_records.values()
                if record.recipient_id == node_id
            ]

            # Calculate totals
            total_earned = sum(
                record.amount for record in node_payments 
                if record.status == 'completed'
            )
            pending_amount = sum(
                record.amount for record in node_payments 
                if record.status in ['pending', 'processing']
            )

            # Get recent payments
            recent_payments = sorted(
                [record for record in node_payments if record.status == 'completed'],
                key=lambda r: r.timestamp,
                reverse=True
            )[:10]

            return {
                'node_id': node_id,
                'total_earned': str(total_earned),
                'pending_amount': str(pending_amount),
                'total_payments': len(node_payments),
                'recent_payments': [
                    {
                        'payment_id': p.payment_id,
                        'amount': str(p.amount),
                        'timestamp': p.timestamp,
                        'tx_id': p.tx_id
                    }
                    for p in recent_payments
                ]
            }

        except Exception as e:
            logging.error(f"Error getting node earnings: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _process_payment_queue(self):
        """Process queued payments"""
        try:
            while True:
                try:
                    # Get next payment to process
                    payment_record = await self.payment_queue.get()
                    payment_id = payment_record.payment_id

                    try:
                        # Update status
                        payment_record.status = 'processing'

                        # Process payment through wallet
                        transfer_result = await self.wallet.transfer(
                            dest_address=payment_record.recipient_address,
                            amount=payment_record.amount
                        )

                        # Update record with transaction ID
                        payment_record.tx_id = transfer_result['tx_id']
                        self.pending_transactions[transfer_result['tx_id']] = payment_id

                        # Wait for confirmation if configured
                        if self.config.get('payment', {}).get('wait_for_confirmation', True):
                            confirmation = await self._wait_for_confirmation(transfer_result['tx_id'])
                            if confirmation['confirmed']:
                                await self._complete_payment(payment_id)
                            else:
                                await self._fail_payment(payment_id, "Transaction failed to confirm")

                    except Exception as e:
                        logging.error(f"Error processing payment {payment_id}: {str(e)}")
                        await self._fail_payment(payment_id, str(e))

                    finally:
                        self.payment_queue.task_done()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.error(f"Error in payment queue processing: {str(e)}")
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logging.info("Payment queue processor stopped")
        except Exception as e:
            logging.error(f"Fatal error in payment processor: {str(e)}")

    async def _monitor_transactions(self):
        """Monitor pending transactions for status updates"""
        try:
            check_interval = self.config.get('payment', {}).get('tx_check_interval', 60)

            while True:
                try:
                    # Check all pending transactions
                    for tx_id, payment_id in list(self.pending_transactions.items()):
                        try:
                            tx_status = await self.wallet.get_transfer_by_txid(tx_id)
                            if tx_status:
                                if tx_status['status'] == 'completed':
                                    await self._complete_payment(payment_id)
                                    del self.pending_transactions[tx_id]
                                elif tx_status['status'] == 'failed':
                                    await self._fail_payment(payment_id, "Transaction failed")
                                    del self.pending_transactions[tx_id]

                        except Exception as e:
                            logging.error(f"Error checking transaction {tx_id}: {str(e)}")

                    await asyncio.sleep(check_interval)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.error(f"Error in transaction monitoring: {str(e)}")
                    await asyncio.sleep(check_interval)

        except asyncio.CancelledError:
            logging.info("Transaction monitor stopped")
        except Exception as e:
            logging.error(f"Fatal error in transaction monitor: {str(e)}")

    async def _wait_for_confirmation(self, tx_id: str) -> Dict:
        """Wait for transaction confirmation"""
        try:
            timeout = self.config.get('payment', {}).get('confirmation_timeout', 1800)  # 30 minutes
            check_interval = 10  # seconds
            start_time = time.time()

            while time.time() - start_time < timeout:
                tx_status = await self.wallet.get_transfer_by_txid(tx_id)
                if tx_status:
                    if tx_status['confirmations'] >= self.config.get('payment', {}).get('min_confirmations', 10):
                        return {'confirmed': True, 'confirmations': tx_status['confirmations']}
                    elif tx_status['status'] == 'failed':
                        return {'confirmed': False, 'error': 'Transaction failed'}

                await asyncio.sleep(check_interval)

            return {'confirmed': False, 'error': 'Confirmation timeout'}

        except Exception as e:
            logging.error(f"Error waiting for confirmation: {str(e)}")
            return {'confirmed': False, 'error': str(e)}

    async def _complete_payment(self, payment_id: str):
        """Mark payment as completed"""
        try:
            if payment_id in self.payment_records:
                record = self.payment_records[payment_id]
                record.status = 'completed'

                # Update verification system if this was a verification payment
                if record.verification_id:
                    await self.verification_system.update_verification_status(
                        verification_id=record.verification_id,
                        status='payment_completed',
                        payment_details={
                            'payment_id': payment_id,
                            'amount': str(record.amount),
                            'tx_id': record.tx_id
                        }
                    )

        except Exception as e:
            logging.error(f"Error completing payment {payment_id}: {str(e)}")

    async def _fail_payment(self, payment_id: str, error: str):
        """Mark payment as failed"""
        try:
            if payment_id in self.payment_records:
                record = self.payment_records[payment_id]
                record.status = 'failed'
                record.error = error

                # Update verification system if this was a verification payment
                if record.verification_id:
                    await self.verification_system.update_verification_status(
                        verification_id=record.verification_id,
                        status='payment_failed',
                        error=error
                    )

        except Exception as e:
            logging.error(f"Error marking payment {payment_id} as failed: {str(e)}")