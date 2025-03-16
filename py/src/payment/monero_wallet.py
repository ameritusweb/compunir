import aiohttp
import json
from decimal import Decimal
import logging
from typing import Dict, Optional
import time
import uuid

class MoneroWallet:
    def __init__(self, config: Dict):
        self.rpc_url = config['rpc_url']
        self.auth = aiohttp.BasicAuth(
            login=config['rpc_username'],
            password=config['rpc_password']
        )
        self.session = None

    async def __aenter__(self):
        """Set up aiohttp session"""
        self.session = aiohttp.ClientSession(auth=self.auth)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up aiohttp session"""
        if self.session:
            await self.session.close()

    async def _rpc_call(self, method: str, params: Dict = None) -> Dict:
        """Make RPC call to Monero wallet"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context")

        try:
            payload = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": method
            }
            if params:
                payload["params"] = params

            async with self.session.post(self.rpc_url, json=payload) as response:
                if response.status != 200:
                    raise ValueError(f"RPC call failed with status {response.status}")
                
                result = await response.json()
                if "error" in result:
                    raise ValueError(f"RPC error: {result['error']}")
                
                return result["result"]
        except Exception as e:
            logging.error(f"Monero RPC call failed: {str(e)}")
            raise

    async def create_address(self) -> str:
        """Create new subaddress for escrow"""
        result = await self._rpc_call("create_address", {
            "account_index": 0,
            "label": f"escrow_{int(time.time())}"
        })
        return result["address"]

    async def get_balance(self, address: Optional[str] = None) -> Decimal:
        """Get balance for address or entire wallet"""
        params = {}
        if address:
            params["address"] = address

        result = await self._rpc_call("get_balance", params)
        return Decimal(str(result["balance"])) / Decimal("1e12")  # Convert atomic units to XMR

    async def transfer(self, dest_address: str, amount: Decimal, priority: str = "normal") -> Dict:
        """Send XMR to destination address"""
        # Convert XMR to atomic units
        atomic_amount = int(amount * Decimal("1e12"))

        params = {
            "destinations": [{
                "address": dest_address,
                "amount": atomic_amount
            }],
            "priority": priority,
            "ring_size": 11,  # Standard ring size
            "get_tx_key": True
        }

        result = await self._rpc_call("transfer", params)
        return {
            "tx_id": result["tx_hash"],
            "tx_key": result["tx_key"],
            "amount": amount,
            "fee": Decimal(str(result["fee"])) / Decimal("1e12")
        }

    async def get_transfer_by_txid(self, txid: str) -> Dict:
        """Get transfer details by transaction ID"""
        result = await self._rpc_call("get_transfer_by_txid", {
            "txid": txid
        })
        return {
            "tx_id": txid,
            "amount": Decimal(str(result["transfer"]["amount"])) / Decimal("1e12"),
            "confirmations": result["transfer"]["confirmations"],
            "timestamp": result["transfer"]["timestamp"],
            "status": "completed" if result["transfer"]["confirmations"] > 0 else "pending"
        }

class MoneroPaymentProcessor:
    def __init__(self, wallet_config: Dict):
        self.wallet_config = wallet_config
        self.escrow_records: Dict[str, Dict] = {}

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
        """Verify payment received in escrow address"""
        record = self.escrow_records.get(job_id)
        if not record:
            raise ValueError(f"No escrow record found for job {job_id}")

        async with MoneroWallet(self.wallet_config) as wallet:
            balance = await wallet.get_balance(record["address"])
            return balance >= expected_amount

    async def release_payment(self, job_id: str, dest_address: str, amount: Decimal) -> Dict:
        """Release payment from escrow to recipient"""
        record = self.escrow_records.get(job_id)
        if not record:
            raise ValueError(f"No escrow record found for job {job_id}")

        async with MoneroWallet(self.wallet_config) as wallet:
            # Verify sufficient balance
            balance = await wallet.get_balance(record["address"])
            if balance < amount:
                raise ValueError(f"Insufficient balance in escrow: {balance} XMR < {amount} XMR")

            # Send payment
            transfer = await wallet.transfer(dest_address, amount)
            
            # Update record
            record["transfers"] = record.get("transfers", [])
            record["transfers"].append({
                "tx_id": transfer["tx_id"],
                "amount": amount,
                "timestamp": time.time()
            })

            return transfer

    async def check_transaction_status(self, tx_id: str) -> Dict:
        """Check status of a transaction"""
        async with MoneroWallet(self.wallet_config) as wallet:
            return await wallet.get_transfer_by_txid(tx_id)

# Update PaymentProcessor implementation to use MoneroPaymentProcessor

class PaymentProcessor:
    def __init__(self, wallet_config: Dict):
        self.monero = MoneroPaymentProcessor(wallet_config)
        self.payment_records: Dict[str, Dict] = {}
        self.base_rate = Decimal('0.001')  # XMR per GPU hour
        self.min_payment = Decimal('0.01')  # Minimum XMR for payout

    async def _generate_escrow_address(self, job_id: str) -> str:
        """Generate unique escrow address"""
        return await self.monero.generate_escrow_address(job_id)

    async def _submit_transaction(self, tx_params: Dict) -> Dict:
        """Submit transaction to Monero network"""
        if 'from_address' in tx_params:
            # Release from escrow
            return await self.monero.release_payment(
                job_id=tx_params.get('job_id'),
                dest_address=tx_params['destination'],
                amount=Decimal(tx_params['amount'])
            )
        else:
            # Direct transfer
            async with MoneroWallet(self.wallet_config) as wallet:
                return await wallet.transfer(
                    dest_address=tx_params['destination'],
                    amount=Decimal(tx_params['amount']),
                    priority=tx_params.get('priority', 'normal')
                )

    def _estimate_network_fees(self) -> Decimal:
        """Estimate network fees for transactions"""
        # Monero fees are typically very low and fairly stable
        # Return a conservative estimate
        return Decimal('0.0001')  # 0.0001 XMR per transaction