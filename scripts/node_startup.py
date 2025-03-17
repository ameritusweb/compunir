#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
from typing import Dict, Any

# Add project root to path
# If scripts/ is directly under the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import configuration manager
from ..py.config.config_manager import ConfigManager

# Import core components
from ..py.src.core.unified_node_manager import NodeManager
from ..py.src.core.job_executor import JobExecutor
from ..py.src.core.network_interface import NetworkInterface

# Import verification system
from ..py.src.verification.unified_verification_system import VerificationSystem
from ..py.src.verification.verifier_selection import VerifierSelectionSystem
from ..py.src.verification.sybil_protection import SybilProtectionSystem
from ..py.src.verification.sybil_metrics import SybilMetricsCollector
from ..py.src.verification.sybil_notification_system import SybilNotificationSystem
from ..py.src.verification.neutral_verifier_selector import NeutralVerifierSelector
from ..py.src.verification.data_verification import DataVerificationSystem

# Import data distribution system
from ..py.src.data_distribution.distribution_manager import DataDistributionManager
from ..py.src.data_distribution.shard_storage import ShardStorage
from ..py.src.data_distribution.transfer_server import ShardTransferServer
from ..py.src.data_distribution.verified_distribution import VerifiedDataDistribution
from ..py.src.data_distribution.performance_monitor import DistributionPerformanceMonitor

# Import payment system
from ..py.src.payment.monero_wallet import MoneroWallet
from ..py.src.payment.monero_payment_processor import MoneroPaymentProcessor


class NodeApplication:
    """Main application for the decentralized GPU node."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = None
        self.logger = None
        
        # Core components
        self.node_manager = None
        self.verification_system = None
        self.distribution_manager = None
        self.payment_processor = None
        
    async def initialize(self):
        """Initialize the node application."""
        # Load configuration
        self.config = self.config_manager.load_configuration()
        self.logger = logging.getLogger('decentralized_gpu')
        self.logger.info("Node application initializing...")
        
        # Initialize storage components
        await self._init_storage()
        
        # Initialize payment components
        await self._init_payment_system()
        
        # Initialize verification system
        await self._init_verification_system()
        
        # Initialize data distribution system
        await self._init_distribution_system()
        
        # Initialize core node manager
        self.node_manager = NodeManager(
            config=self.config,
            network_client=self._create_network_client(),
            payment_processor=self.payment_processor,
            verification_manager=self.verification_system
        )
        
        self.logger.info("Node application initialized successfully")
        
    async def _init_storage(self):
        """Initialize storage components."""
        # Create required directories
        os.makedirs("data/shards", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("work", exist_ok=True)
        
        self.logger.info("Storage directories initialized")
        
    async def _init_payment_system(self):
        """Initialize payment system."""
        try:
            # Initialize wallet connection
            wallet_config = self.config_manager.get_wallet_config()
            self.wallet = MoneroWallet(wallet_config)
            
            # Initialize payment processor
            self.payment_processor = MoneroPaymentProcessor(
                config=wallet_config,
                wallet=self.wallet
            )
            
            self.logger.info("Payment system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize payment system: {str(e)}")
            raise
        
    async def _init_verification_system(self):
        """Initialize verification system components."""
        try:
            # Initialize Sybil protection
            self.sybil_protection = SybilProtectionSystem(self.config)
            
            # Initialize metrics collector
            self.sybil_metrics = SybilMetricsCollector(
                config=self.config,
                sybil_protection=self.sybil_protection,
                node_manager=None,  # Will set this later
                verification_system=None  # Will set this later
            )
            
            # Initialize verification components
            verifier_selection = VerifierSelectionSystem(self.config)
            neutral_selector = NeutralVerifierSelector(self.config)
            
            # Initialize notification system
            self.sybil_notification = SybilNotificationSystem(self.config)
            
            # Initialize main verification system
            self.verification_system = VerificationSystem(self.config)
            
            # Initialize data verification system
            self.data_verification = DataVerificationSystem(
                config=self.config,
                verification_system=self.verification_system
            )
            
            # Complete circular references
            self.sybil_metrics.verification_system = self.verification_system
            
            self.logger.info("Verification system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize verification system: {str(e)}")
            raise
        
    async def _init_distribution_system(self):
        """Initialize data distribution system."""
        try:
            # Get distribution configuration
            dist_config = self.config_manager.get_distribution_config()
            
            # Initialize shard storage
            self.shard_storage = ShardStorage(dist_config['storage'])
            
            # Initialize transfer server
            self.transfer_server = ShardTransferServer(
                config=dist_config,
                shard_storage=self.shard_storage
            )
            
            # Initialize distribution manager
            self.distribution_manager = DataDistributionManager(dist_config)
            
            # Initialize performance monitor
            self.distribution_monitor = DistributionPerformanceMonitor(
                config=dist_config,
                distribution_manager=self.distribution_manager
            )
            
            # Initialize verified distribution
            self.verified_distribution = VerifiedDataDistribution(
                config=dist_config,
                verification_system=self.data_verification,
                distribution_manager=self.distribution_manager
            )
            
            # Start distribution components
            server_host = dist_config['network']['server_host']
            server_port = dist_config['network']['server_port']
            await self.transfer_server.start(server_host, server_port)
            await self.distribution_manager.start()
            await self.distribution_monitor.start_collection()
            
            self.logger.info(f"Data distribution system initialized on {server_host}:{server_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distribution system: {str(e)}")
            raise
        
    def _create_network_client(self):
        """Create network client for external communication."""
        # This would be your implementation of the network client
        # For now, just return a placeholder
        return None
        
    async def start(self):
        """Start the node application."""
        try:
            # Initialize all components
            await self.initialize()
            
            # Start the node manager
            await self.node_manager.start()
            
            self.logger.info("Node started successfully")
            
            # Keep application running
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                self.logger.info("Node shutdown requested")
                
        except Exception as e:
            self.logger.error(f"Error starting node: {str(e)}")
            raise
        finally:
            await self.shutdown()
        
    async def shutdown(self):
        """Shutdown the node application."""
        self.logger.info("Shutting down node application...")
        
        # Shutdown components in reverse initialization order
        if self.node_manager:
            await self.node_manager.stop()
            
        if self.distribution_manager:
            await self.distribution_manager.stop()
            
        if self.distribution_monitor:
            await self.distribution_monitor.stop_collection()
            
        if self.transfer_server:
            await self.transfer_server.stop()
            
        # Close wallet connection
        if hasattr(self, 'wallet') and self.wallet:
            await self.wallet.close()
            
        self.logger.info("Node application shutdown complete")


def run_node():
    """Run the node application."""
    node = NodeApplication()
    
    try:
        asyncio.run(node.start())
    except KeyboardInterrupt:
        print("\nNode shutdown requested via keyboard interrupt")
    except Exception as e:
        print(f"Error running node: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run_node()