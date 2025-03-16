import aiohttp
import asyncio
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

    async def get_balance(self, account_index: int = 0) -> Decimal:
        """Get balance for a given Monero account index."""
        result = await self._rpc_call("get_balance", {"account_index": account_index})
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

