import asyncio
import grpc
import logging
from typing import Optional, List
from datetime import datetime

from .generated import node_service_pb2 as pb2
from .generated import node_service_pb2_grpc as pb2_grpc

class NodeNetworkClient:
    def __init__(self, address: str):
        self.address = address
        self.node_id: Optional[str] = None
        self.channel = None
        self.stub = None
        self.is_connected = False
        
    async def connect(self):
        """Establish connection to the network"""
        try:
            # Create channel
            self.channel = grpc.aio.insecure_channel(self.address)
            self.stub = pb2_grpc.NodeServiceStub(self.channel)
            
            # Register node
            response = await self.register_node()
            self.node_id = response.assigned_node_id
            self.is_connected = True
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat_loop())
            
            logging.info(f"Connected to network with node_id: {self.node_id}")
        except Exception as e:
            logging.error(f"Failed to connect: {str(e)}")
            raise

    async def register_node(self) -> pb2.RegistrationResponse:
        """Register this node with the network"""
        try:
            node_info = self._create_node_info()
            response = await self.stub.RegisterNode(node_info)
            return response
        except Exception as e:
            logging.error(f"Registration failed: {str(e)}")
            raise

    async def submit_job(self, job_spec: dict) -> str:
        """Submit a new job to the network"""
        try:
            # Convert job spec to protobuf
            job_request = self._create_job_specification(job_spec)
            
            # Submit job
            response = await self.stub.SubmitJob(job_request)
            
            return response.job_id
        except Exception as e:
            logging.error(f"Job submission failed: {str(e)}")
            raise

    async def get_job_status(self, job_id: str) -> pb2.JobStatus:
        """Get status of a specific job"""
        try:
            request = pb2.JobId(job_id=job_id)
            response = await self.stub.GetJobStatus(request)
            return response
        except Exception as e:
            logging.error(f"Failed to get job status: {str(e)}")
            raise

    async def stream_metrics(self):
        """Stream metrics from the node"""
        try:
            request = pb2.google.protobuf.Empty()
            async for metrics in self.stub.StreamMetrics(request):
                yield metrics
        except Exception as e:
            logging.error(f"Metrics streaming failed: {str(e)}")
            raise

    async def submit_verification(self, proof: pb2.VerificationProof) -> bool:
        """Submit verification proof"""
        try:
            response = await self.stub.SubmitVerificationProof(proof)
            return response.valid
        except Exception as e:
            logging.error(f"Verification submission failed: {str(e)}")
            raise

    async def _heartbeat_loop(self):
        """Maintain connection with periodic heartbeats"""
        while self.is_connected:
            try:
                status = self._create_node_status()
                response = await self.stub.Heartbeat(status)
                
                # Handle required actions
                if response.actions_required:
                    await self._handle_required_actions(response.actions_required)
                    
                await asyncio.sleep(30)  # Configurable interval
            except Exception as e:
                logging.error(f"Heartbeat failed: {str(e)}")
                await self._handle_connection_failure()

    def _create_node_info(self) -> pb2.NodeInfo:
        """Create node information message"""
        return pb2.NodeInfo(
            version="1.0.0",
            gpu_info=self._get_gpu_info(),
            network_capabilities=self._get_network_capabilities(),
            supported_frameworks=["pytorch"],
            wallet_address="your_monero_address_here"
        )

    def _create_node_status(self) -> pb2.NodeStatus:
        """Create current node status message"""
        return pb2.NodeStatus(
            node_id=self.node_id,
            timestamp=datetime.now().timestamp(),
            gpu_metrics=self._get_current_gpu_metrics(),
            active_jobs=self._get_active_jobs(),
            network_metrics=self._get_network_metrics()
        )

    def _create_job_specification(self, job_spec: dict) -> pb2.JobSpecification:
        """Convert job specification dict to protobuf message"""
        return pb2.JobSpecification(
            job_id=job_spec.get('job_id', f"job_{datetime.now().timestamp()}"),
            framework=job_spec.get('framework', 'pytorch'),
            model=self._create_model_definition(job_spec.get('model', {})),
            training_config=self._create_training_config(job_spec.get('training', {})),
            resource_requirements=self._create_resource_requirements(job_spec.get('resources', {})),
            privacy_settings=self._create_privacy_settings(job_spec.get('privacy', {})),
            verification_config=self._create_verification_config(job_spec.get('verification', {})),
            payment_details=self._create_payment_details(job_spec.get('payment', {}))
        )

    async def _handle_required_actions(self, actions: List[str]):
        """Handle actions required by the network"""
        for action in actions:
            if action == "UPDATE_SOFTWARE":
                await self._handle_software_update()
            elif action == "REDUCE_LOAD":
                await self._handle_load_reduction()
            elif action == "VERIFICATION_NEEDED":
                await self._handle_verification_request()
            else:
                logging.warning(f"Unknown required action: {action}")

    async def _handle_connection_failure(self):
        """Handle connection failures"""
        self.is_connected = False
        await asyncio.sleep(60)  # Backoff before retry
        try:
            await self.connect()
        except Exception as e:
            logging.error(f"Reconnection failed: {str(e)}")

    def _get_gpu_info(self) -> pb2.GPUInfo:
        """Get GPU information"""
        # Implementation would use nvidia-ml-py3
        pass

    def _get_network_capabilities(self) -> pb2.NetworkCapabilities:
        """Get network capabilities"""
        # Implementation would check system capabilities
        pass

    def _get_current_gpu_metrics(self) -> pb2.GPUMetrics:
        """Get current GPU metrics"""
        # Implementation would use nvidia-ml-py3
        pass

    def _get_active_jobs(self) -> List[pb2.JobStatus]:
        """Get list of active jobs"""
        # Implementation would track running jobs
        pass

    def _get_network_metrics(self) -> pb2.NetworkMetrics:
        """Get current network metrics"""
        # Implementation would track network usage
        pass