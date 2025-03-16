import asyncio
from decimal import Decimal
import logging
from typing import Dict, Optional, List
import time
from dataclasses import dataclass

from .monero_wallet import MoneroWallet

@dataclass
class EscrowState:
    escrow_id: str
    job_id: str
    total_amount: Decimal
    released_amount: Decimal
    escrow_address: str
    transaction_id: str
    status: str  # 'pending', 'active', 'completed', 'failed'
    creation_time: float
    expiration_time: float
    participant_node_ids: List[str]

class EscrowManager:
    def __init__(self, config: Dict, wallet: MoneroWallet):
        self.config = config
        self.wallet = wallet
        self.escrows: Dict[str, EscrowState] = {}
        self.logger = logging.getLogger(__name__)
        self._cleanup_task = None

    async def start(self):
        """Start escrow manager and monitoring"""
        self._cleanup_task = asyncio.create_task(self._monitor_escrows())
        self.logger.info("Escrow manager started")

    async def stop(self):
        """Stop escrow manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await self._cleanup_task
        self.logger.info("Escrow manager stopped")

    async def create_escrow(self, 
                          job_id: str, 
                          amount: Decimal,
                          participant_nodes: List[str]) -> Dict:
        """Create new escrow for a job"""
        try:
            # Generate unique escrow address
            escrow_address = await self.wallet.create_address()

            # Create escrow transaction
            transaction = await self.wallet.transfer(
                dest_address=escrow_address,
                amount=amount
            )

            # Create escrow state
            escrow_id = f"escrow_{job_id}_{int(time.time())}"
            expiration = time.time() + self.config.get('escrow_timeout', 86400)  # 24 hours

            escrow = EscrowState(
                escrow_id=escrow_id,
                job_id=job_id,
                total_amount=amount,
                released_amount=Decimal('0'),
                escrow_address=escrow_address,
                transaction_id=transaction['tx_id'],
                status='pending',
                creation_time=time.time(),
                expiration_time=expiration,
                participant_node_ids=participant_nodes
            )

            self.escrows[escrow_id] = escrow

            # Wait for transaction confirmation
            confirmation = await self._wait_for_confirmation(transaction['tx_id'])
            if confirmation['status'] == 'confirmed':
                escrow.status = 'active'
                self.logger.info(f"Escrow {escrow_id} created successfully")
            else:
                escrow.status = 'failed'
                self.logger.error(f"Failed to create escrow {escrow_id}")

            return {
                'escrow_id': escrow_id,
                'address': escrow_address,
                'transaction_id': transaction['tx_id'],
                'status': escrow.status
            }

        except Exception as e:
            self.logger.error(f"Error creating escrow: {str(e)}")
            raise

    async def release_payment(self,
                            escrow_id: str,
                            node_id: str,
                            amount: Decimal) -> Dict:
        """Release payment from escrow to node"""
        try:
            escrow = self._get_escrow(escrow_id)
            
            # Verify node is participant
            if node_id not in escrow.participant_node_ids:
                raise ValueError(f"Node {node_id} not authorized for escrow {escrow_id}")

            # Verify sufficient funds
            if escrow.released_amount + amount > escrow.total_amount:
                raise ValueError("Insufficient funds in escrow")

            # Get node's wallet address
            node_address = await self._get_node_address(node_id)

            # Process transfer
            transaction = await self.wallet.transfer(
                dest_address=node_address,
                amount=amount,
                from_address=escrow.escrow_address
            )

            # Update escrow state
            escrow.released_amount += amount
            if escrow.released_amount >= escrow.total_amount:
                escrow.status = 'completed'

            return {
                'transaction_id': transaction['tx_id'],
                'amount': amount,
                'remaining': escrow.total_amount - escrow.released_amount
            }

        except Exception as e:
            self.logger.error(f"Error releasing payment: {str(e)}")
            raise

    async def get_escrow_status(self, escrow_id: str) -> Dict:
        """Get current escrow status"""
        try:
            escrow = self._get_escrow(escrow_id)
            
            # Get current balance
            balance = await self.wallet.get_balance(escrow.escrow_address)

            return {
                'escrow_id': escrow.escrow_id,
                'status': escrow.status,
                'total_amount': escrow.total_amount,
                'released_amount': escrow.released_amount,
                'current_balance': balance,
                'creation_time': escrow.creation_time,
                'expiration_time': escrow.expiration_time
            }

        except Exception as e:
            self.logger.error(f"Error getting escrow status: {str(e)}")
            raise

    async def refund_escrow(self, escrow_id: str) -> Dict:
        """Refund remaining escrow amount"""
        try:
            escrow = self._get_escrow(escrow_id)
            
            # Calculate refund amount
            current_balance = await self.wallet.get_balance(escrow.escrow_address)
            if current_balance == 0:
                return {'status': 'no_funds'}

            # Process refund
            transaction = await self.wallet.transfer(
                dest_address=self.config['refund_address'],
                amount=current_balance,
                from_address=escrow.escrow_address
            )

            escrow.status = 'refunded'
            return {
                'transaction_id': transaction['tx_id'],
                'amount': current_balance,
                'status': 'refunded'
            }

        except Exception as e:
            self.logger.error(f"Error refunding escrow: {str(e)}")
            raise

    async def _monitor_escrows(self):
        """Monitor and cleanup expired escrows"""
        while True:
            try:
                current_time = time.time()
                
                # Find expired escrows
                expired = [
                    escrow_id for escrow_id, escrow in self.escrows.items()
                    if escrow.expiration_time < current_time
                    and escrow.status not in ['completed', 'refunded']
                ]

                # Process expired escrows
                for escrow_id in expired:
                    await self.refund_escrow(escrow_id)

                # Cleanup old completed escrows
                cleanup_threshold = current_time - self.config.get('cleanup_age', 86400)
                completed = [
                    escrow_id for escrow_id, escrow in self.escrows.items()
                    if escrow.creation_time < cleanup_threshold
                    and escrow.status in ['completed', 'refunded']
                ]
                
                for escrow_id in completed:
                    del self.escrows[escrow_id]

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in escrow monitoring: {str(e)}")
                await asyncio.sleep(300)

    async def _wait_for_confirmation(self, tx_id: str, timeout: int = 1800) -> Dict:
        """Wait for transaction confirmation"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                tx_info = await self.wallet.get_transfer_by_txid(tx_id)
                if tx_info['confirmations'] >= self.config.get('min_confirmations', 10):
                    return {'status': 'confirmed', 'confirmations': tx_info['confirmations']}
                await asyncio.sleep(10)
            except Exception as e:
                self.logger.error(f"Error checking transaction: {str(e)}")
                await asyncio.sleep(10)
        
        return {'status': 'timeout'}

    def _get_escrow(self, escrow_id: str) -> EscrowState:
        """Get escrow by ID with validation"""
        escrow = self.escrows.get(escrow_id)
        if not escrow:
            raise ValueError(f"Escrow {escrow_id} not found")
        if escrow.status == 'failed':
            raise ValueError(f"Escrow {escrow_id} failed")
        return escrow

    async def _get_node_address(self, node_id: str) -> str:
        """Get node's wallet address"""
        # Implementation would get address from node registry
        # This is a placeholder
        return f"node_wallet_address_{node_id}"