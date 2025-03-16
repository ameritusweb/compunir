import asyncio
from decimal import Decimal
import logging
from typing import Dict
import time
from .monero_wallet import MoneroWallet

class MoneroPaymentProcessor:
    def __init__(self, wallet_config: Dict):
        self.wallet_config = wallet_config
        self.escrow_records: Dict[str, Dict] = {}
        self.payment_records: Dict[str, Dict] = {}
        self.pending_transactions: Dict[str, str] = {}  # tx_id -> payment_id
        self.payment_queue = asyncio.Queue()
        self.logger = logging.getLogger(__name__)
        self._cleanup_task = None
        self._transaction_monitor_task = None

    async def start(self):
        """Start periodic escrow and transaction monitoring."""
        self._cleanup_task = asyncio.create_task(self._monitor_escrows())
        self._transaction_monitor_task = asyncio.create_task(self._monitor_transactions())
        self.logger.info("Monero Payment Processor started.")

    async def stop(self):
        """Stop monitoring tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await self._cleanup_task
        if self._transaction_monitor_task:
            self._transaction_monitor_task.cancel()
            await self._transaction_monitor_task
        self.logger.info("Monero Payment Processor stopped.")

    async def process_verification_payment(self, job_id: str, node_id: str, amount: Decimal) -> Dict:
        """Process payment for verification results."""
        try:
            payment_id = f"pay_{job_id}_{int(time.time())}"
            recipient_wallet = await self._get_node_address(node_id)

            self.payment_records[payment_id] = {
                "payment_id": payment_id,
                "job_id": job_id,
                "node_id": node_id,
                "recipient_wallet": recipient_wallet,
                "amount": amount,
                "status": "pending",
                "timestamp": time.time(),
                "tx_id": None
            }

            await self.payment_queue.put(payment_id)
            return {"payment_id": payment_id, "status": "queued"}
        except Exception as e:
            logging.error(f"Error processing verification payment: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_node_earnings(self, node_id: str) -> Dict:
        """Retrieve earnings of a node."""
        total_earned = sum(
            record["amount"] for record in self.payment_records.values()
            if record["node_id"] == node_id and record["status"] == "completed"
        )
        pending_amount = sum(
            record["amount"] for record in self.payment_records.values()
            if record["node_id"] == node_id and record["status"] in ["pending", "processing"]
        )

        return {
            "node_id": node_id,
            "total_earned": str(total_earned),
            "pending_amount": str(pending_amount),
            "total_payments": len([p for p in self.payment_records.values() if p["node_id"] == node_id]),
        }

    async def _monitor_escrows(self):
        """Monitor expired escrows and process refunds."""
        while True:
            try:
                current_time = time.time()
                expired_jobs = [
                    job_id for job_id, record in self.escrow_records.items()
                    if record["expiration_time"] < current_time and record["status"] not in ["completed", "refunded"]
                ]
                for job_id in expired_jobs:
                    await self.refund_escrow(job_id)
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in escrow monitoring: {str(e)}")

    async def generate_escrow_address(self, job_id: str) -> str:
        """Generate unique escrow address for job"""
        async with MoneroWallet(self.wallet_config) as wallet:
            address = await wallet.create_address()
            
            self.escrow_records[job_id] = {
                "address": address,
                "created_at": time.time(),
                "status": "pending"
            }
            
            return address

    async def verify_payment(self, job_id: str, expected_amount: Decimal) -> bool:
        """Verify if payment for a job has been received at the escrow subaddress."""
        record = self.escrow_records.get(job_id)
        if not record:
            raise ValueError(f"No escrow record found for job {job_id}")

        # Check if the escrow has expired
        if time.time() > record.get("expires_at", float("inf")):
            logging.warning(f"Escrow for job {job_id} expired. Removing record.")
            del self.escrow_records[job_id]
            return False

        async with MoneroWallet(self.wallet_config) as wallet:
            try:
                # Get incoming transfers
                transfers = await wallet._rpc_call("get_transfers", {"in": True})

                total_received = Decimal("0.0")

                for tx in transfers.get("in", []):
                    if tx["address"] == record["address"] and tx["confirmations"] > 0:
                        total_received += Decimal(str(tx["amount"])) / Decimal("1e12")  # Convert atomic units

                # Check if payment meets or exceeds expected amount
                if total_received >= expected_amount:
                    record["status"] = "funded"
                    record["received_amount"] = total_received
                    return True

            except Exception as e:
                logging.error(f"Error verifying payment for job {job_id}: {str(e)}")

        return False

    async def release_payment(self, job_id: str, node_id: str, amount: Decimal) -> Dict:
        """Securely release escrow funds, verifying on-chain balance and authorized nodes."""
        record = self.escrow_records.get(job_id)
        if not record:
            raise ValueError(f"No escrow record found for job {job_id}")

        if node_id not in record["participant_nodes"]:
            raise ValueError(f"Node {node_id} is not authorized for escrow {job_id}")

        async with MoneroWallet(self.wallet_config) as wallet:
            # Fetch live blockchain data
            transfers = await wallet._rpc_call("get_transfers", {"in": True})
            total_received = Decimal("0.0")

            for tx in transfers.get("in", []):
                if tx["address"] == record["address"] and tx["confirmations"] > 0:
                    total_received += Decimal(str(tx["amount"])) / Decimal("1e12")

            if total_received < amount:
                raise ValueError(f"Insufficient escrow balance: {total_received} XMR < {amount} XMR")

            node_wallet = await self._get_node_address(node_id)
            transfer = await wallet.transfer(node_wallet, amount)

            record.setdefault("transfers", []).append({
                "tx_id": transfer["tx_id"],
                "amount": amount,
                "timestamp": time.time()
            })

            return transfer

    async def check_transaction_status(self, tx_id: str) -> Dict:
        """Check status of a transaction"""
        async with MoneroWallet(self.wallet_config) as wallet:
            return await wallet.get_transfer_by_txid(tx_id)

    async def _wait_for_confirmation(self, tx_id: str) -> Dict:
        """Wait for transaction confirmation."""
        timeout = 1800  # 30 min
        start_time = time.time()

        while time.time() - start_time < timeout:
            async with MoneroWallet(self.wallet_config) as wallet:
                tx_status = await wallet.get_transfer_by_txid(tx_id)
                if tx_status and tx_status["confirmations"] >= 10:
                    return {"confirmed": True}
                elif tx_status and tx_status["status"] == "failed":
                    return {"confirmed": False, "error": "Transaction failed"}
            await asyncio.sleep(10)

        return {"confirmed": False, "error": "Confirmation timeout"}
    
    async def _monitor_transactions(self):
        """Monitor pending transactions for status updates."""
        while True:
            try:
                for tx_id, payment_id in list(self.pending_transactions.items()):
                    async with MoneroWallet(self.wallet_config) as wallet:
                        tx_status = await wallet.get_transfer_by_txid(tx_id)
                        if tx_status and tx_status["status"] == "completed":
                            self.payment_records[payment_id]["status"] = "completed"
                            del self.pending_transactions[tx_id]
                        elif tx_status and tx_status["status"] == "failed":
                            self.payment_records[payment_id]["status"] = "failed"
                            del self.pending_transactions[tx_id]
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in transaction monitoring: {str(e)}")

    async def _process_payment_queue(self):
        """Process queued payments."""
        while True:
            try:
                payment_id = await self.payment_queue.get()
                record = self.payment_records.get(payment_id)
                if not record:
                    continue

                async with MoneroWallet(self.wallet_config) as wallet:
                    transfer_result = await wallet.transfer(
                        record["recipient_wallet"], record["amount"]
                    )
                    record["tx_id"] = transfer_result["tx_id"]
                    self.pending_transactions[transfer_result["tx_id"]] = payment_id

                    confirmation = await self._wait_for_confirmation(transfer_result["tx_id"])
                    if confirmation["confirmed"]:
                        record["status"] = "completed"
                    else:
                        record["status"] = "failed"

                self.payment_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error processing payment queue: {str(e)}")

    async def _get_node_address(self, node_id: str) -> str:
        """Retrieve the actual wallet address for a node."""
        node_info = await self.node_manager.get_node_info(node_id)
        if not node_info or 'wallet_address' not in node_info:
            raise ValueError(f"No wallet address found for node {node_id}")
        return node_info['wallet_address']
