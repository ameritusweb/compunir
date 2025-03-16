import asyncio
import grpc
import psutil
import pynvml
import subprocess
import logging
from typing import Optional, List
from datetime import datetime
from ..core.job_executor import JobExecutor

from .generated import node_service_pb2 as pb2
from .generated import node_service_pb2_grpc as pb2_grpc

class NodeNetworkClient:
    def __init__(self, address: str):
        self.address = address
        self.node_id: Optional[str] = None
        self.job_executor = JobExecutor()
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
        """Retrieve real-time CPU, RAM, GPU, and network stats."""
        return pb2.NodeStatus(
            node_id=self.node_id,
            timestamp=datetime.now().timestamp(),
            gpu_metrics=self._get_current_gpu_metrics(),
            active_jobs=self._get_active_jobs(),
            network_metrics=self._get_network_metrics(),
            cpu_usage=psutil.cpu_percent(interval=1),  # % CPU load
            memory_usage=psutil.virtual_memory().percent  # % RAM usage
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
        """Handle network failures with exponential backoff."""
        self.is_connected = False
        delay = 5  # Start with 5 seconds

        for attempt in range(5):  # Retry 5 times before giving up
            logging.warning(f"Retrying connection in {delay} seconds... (Attempt {attempt + 1})")
            await asyncio.sleep(delay)

            try:
                await self.connect()
                return  # Successfully reconnected
            except Exception as e:
                logging.error(f"Reconnection attempt {attempt + 1} failed: {str(e)}")
                delay *= 2  # Exponential backoff

        logging.critical("Failed to reconnect after 5 attempts.")

    def _get_gpu_info(self) -> pb2.GPUInfo:
        """Retrieve real GPU information using NVIDIA's NVML."""
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
            gpu_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            cuda_cores = 10496  # Placeholder (needs better retrieval)
            
            return pb2.GPUInfo(
                gpu_model=gpu_name,
                vram_total=mem_info.total // (1024 * 1024),  # Convert to MB
                vram_available=mem_info.free // (1024 * 1024),  # Convert to MB
                cuda_cores=cuda_cores,  # Static, update dynamically later
                compute_capability="8.6"  # Static, update dynamically later
            )
        except Exception as e:
            logging.error(f"Failed to fetch GPU info: {str(e)}")
            return pb2.GPUInfo()  # Return empty object to prevent crash

    def _get_network_capabilities(self) -> pb2.NetworkCapabilities:
        """Retrieve network capabilities using psutil."""
        return pb2.NetworkCapabilities(
            bandwidth_upload=psutil.net_if_stats()["eth0"].speed,  # Assuming `eth0`
            bandwidth_download=psutil.net_if_stats()["eth0"].speed,
            latency_ms=5  # Placeholder value
        )

    def _get_current_gpu_metrics(self) -> pb2.GPUMetrics:
        """Retrieve real-time GPU usage stats."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert mW → W
            
            return pb2.GPUMetrics(
                memory_used=mem_info.used // (1024 * 1024),  # Convert to MB
                memory_total=mem_info.total // (1024 * 1024),  # Convert to MB
                utilization=utilization.gpu,  # % GPU utilization
                temperature=temp,  # °C
                power_usage=power  # Watts
            )
        except Exception as e:
            logging.error(f"Failed to fetch real-time GPU metrics: {str(e)}")
            return pb2.GPUMetrics()

    def _get_active_jobs(self) -> List[pb2.JobStatus]:
        """Fetch currently running jobs from the job executor."""
        job_status_list = []
        for job_id, job_info in self.job_executor.active_jobs.items():
            job_status_list.append(pb2.JobStatus(
                job_id=job_id,
                status=job_info["status"],
                progress=job_info["progress"],
                last_update=pb2.Timestamp(seconds=int(job_info["last_update"])),
                active_nodes=job_info.get("active_nodes", [])
            ))
        return job_status_list


    def _get_network_metrics(self) -> pb2.NetworkMetrics:
        """Retrieve real-time network usage stats."""
        try:
            net_io = psutil.net_io_counters()
            return pb2.NetworkMetrics(
                bandwidth_usage=net_io.bytes_sent + net_io.bytes_recv,  # Total bandwidth
                active_connections=len(psutil.net_connections()),  # Open sockets
                latency_ms=10,  # Placeholder (need real ping test)
                bytes_transferred=net_io.bytes_sent + net_io.bytes_recv
            )
        except Exception as e:
            logging.error(f"Failed to fetch network metrics: {str(e)}")
            return pb2.NetworkMetrics()

    async def _handle_software_update(self):
        """Handle software update request (pull latest changes and restart)."""
        logging.info("Checking for software updates...")

        try:
            # Pull latest updates from Git
            process = await asyncio.create_subprocess_shell(
                "git pull origin main",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logging.info(f"Software update completed:\n{stdout.decode()}")
                
                # Optionally restart service if needed
                logging.info("Restarting service...")
                subprocess.Popen(["systemctl", "restart", "node_service"])
            else:
                logging.error(f"Software update failed:\n{stderr.decode()}")

        except Exception as e:
            logging.error(f"Error during software update: {str(e)}")

    async def _handle_load_reduction(self):
        """Reduce GPU/CPU load dynamically."""
        logging.info("Reducing workload due to network request.")

        # Reduce active training batch size
        for job_id, job in self.job_executor.active_jobs.items():
            if job["status"] == "running":
                old_batch_size = job["training_config"]["batch_size"]
                new_batch_size = max(1, old_batch_size // 2)  # Reduce by half
                job["training_config"]["batch_size"] = new_batch_size
                logging.info(f"Reduced batch size for job {job_id}: {old_batch_size} → {new_batch_size}")

        # Pause non-essential tasks
        if hasattr(self, "background_tasks"):
            for task in self.background_tasks:
                task.cancel()
                logging.info("Paused non-essential background task.")

        logging.info("Load reduction complete.")


    async def _handle_verification_request(self):
        """Process verification request (fetch and validate a verification task)."""
        logging.info("Processing verification request...")

        try:
            # Fetch verification task
            verification_task = await self.node_manager.get_pending_verification_task()

            if not verification_task:
                logging.info("No verification tasks available.")
                return

            logging.info(f"Fetched verification task: {verification_task['job_id']}")

            # Perform verification logic (e.g., check model outputs)
            verification_result = await self._run_verification(verification_task)

            # Submit verification result
            await self.verification_system.submit_verification_result(verification_task["job_id"], verification_result)

            logging.info(f"Verification completed for job {verification_task['job_id']}.")

        except Exception as e:
            logging.error(f"Error during verification process: {str(e)}")


    async def _run_verification(self, verification_task):
        """Run the verification check (dummy implementation)."""
        job_id = verification_task["job_id"]
        model_outputs = verification_task["model_outputs"]
        
        # Compare against expected outputs (simplified logic)
        verified = sum(model_outputs) % 2 == 0  # Fake validation (replace with real logic)
        
        return {"job_id": job_id, "verified": verified}
