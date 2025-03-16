"""
gRPC service implementation for node network communication.
"""

import grpc
from concurrent import futures
import time
from typing import Dict, Optional, List
import asyncio
import logging
from google.protobuf import empty_pb2

from .generated import node_service_pb2 as pb2
from .generated import node_service_pb2_grpc as pb2_grpc
from ..core.node_manager import NodeManager
from ..core.job_executor import JobExecutor
from ..verification import VerificationSystem
from ..payment import PaymentProcessor

class NodeService(pb2_grpc.NodeServiceServicer):
    """Complete implementation of NodeService gRPC service"""
    
    def __init__(self, 
                 node_manager: NodeManager,
                 job_executor: JobExecutor,
                 verification_system: VerificationSystem,
                 payment_processor: PaymentProcessor):
        self.node_manager = node_manager
        self.job_executor = job_executor
        self.verification_system = verification_system
        self.payment_processor = payment_processor
        
        # Active node tracking
        self.active_nodes: Dict[str, float] = {}  # node_id -> last_heartbeat
        self.verifier_pool: List[str] = []
        self.active_streams: Dict[int, bool] = {}
        
        logging.info("Initialized NodeService")

    async def RegisterNode(self, request: pb2.NodeInfo, 
                         context: grpc.aio.ServicerContext) -> pb2.RegistrationResponse:
        """Handle new node registration"""
        try:
            # Verify node identity
            verification_result = await self.node_manager.verify_node_identity(
                node_id=request.node_id,
                registration_data={
                    'gpu_info': self._convert_gpu_info(request.gpu_info),
                    'network_capabilities': self._convert_network_capabilities(request.network_capabilities),
                    'geographic_data': self._convert_geographic_data(request.geographic_data),
                    'wallet_address': request.wallet_address,
                    'pow_proof': request.pow_proof,
                    'stake_transaction_id': request.stake_transaction_id
                }
            )

            if not verification_result['verified']:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, 
                            verification_result.get('error', 'Verification failed'))

            # Register node
            node_id = self._generate_node_id(request)
            self.active_nodes[node_id] = time.time()

            registration = await self.node_manager.register_node({
                'node_id': node_id,
                'version': request.version,
                'gpu_info': self._convert_gpu_info(request.gpu_info),
                'network_capabilities': self._convert_network_capabilities(request.network_capabilities),
                'supported_frameworks': list(request.supported_frameworks),
                'wallet_address': request.wallet_address
            })

            # Create response
            return pb2.RegistrationResponse(
                assigned_node_id=node_id,
                bootstrap_nodes=self._get_bootstrap_nodes(),
                network_config=self._create_network_config(),
                reputation_score=verification_result['reputation_score'],
                pow_difficulty=verification_result['pow_difficulty'],
                required_stake=str(verification_result['required_stake'])
            )

        except Exception as e:
            logging.error(f"Error in RegisterNode: {str(e)}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Heartbeat(self, request: pb2.NodeStatus, 
                       context: grpc.aio.ServicerContext) -> pb2.HeartbeatResponse:
        """Process node heartbeat"""
        try:
            node_id = request.node_id
            if node_id not in self.active_nodes:
                context.abort(grpc.StatusCode.NOT_FOUND, "Node not registered")

            # Update last heartbeat time
            self.active_nodes[node_id] = time.time()

            # Update node status
            status_update = await self.node_manager.update_node_status(
                node_id=node_id,
                status={
                    'gpu_metrics': self._convert_gpu_metrics(request.gpu_metrics),
                    'active_jobs': [self._convert_job_status(j) for j in request.active_jobs],
                    'network_metrics': self._convert_network_metrics(request.network_metrics)
                }
            )

            # Check for required actions
            actions = self._get_required_actions(node_id)

            return pb2.HeartbeatResponse(
                accepted=True,
                actions_required=actions,
                updated_config=self._create_network_config(status_update.get('config_updates'))
            )

        except Exception as e:
            logging.error(f"Error in Heartbeat: {str(e)}")
            return pb2.HeartbeatResponse(accepted=False)

    async def SubmitJob(self, request: pb2.JobSpecification, 
                       context: grpc.aio.ServicerContext) -> pb2.JobSubmissionResponse:
        """Handle job submission"""
        try:
            # Validate job specification
            if not self._validate_job_spec(request):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid job specification")

            # Convert job specification
            job_spec = {
                'job_id': request.job_id,
                'framework': request.framework,
                'model': self._convert_model_definition(request.model),
                'training_config': self._convert_training_config(request.training_config),
                'resource_requirements': self._convert_resource_requirements(request.resource_requirements),
                'privacy_settings': self._convert_privacy_settings(request.privacy_settings),
                'verification_config': self._convert_verification_config(request.verification_config),
                'payment_details': self._convert_payment_details(request.payment_details)
            }

            # Find suitable nodes
            assigned_nodes = await self._assign_nodes(request)
            if not assigned_nodes:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "No suitable nodes available")

            # Process payment
            payment_confirmation = await self.payment_processor.process_job_payment(
                job_id=request.job_id,
                payment_details=self._convert_payment_details(request.payment_details)
            )

            # Distribute job
            await self._distribute_job(job_spec, assigned_nodes)

            # Submit job
            submission_result = await self.node_manager.submit_job(job_spec)

            return pb2.JobSubmissionResponse(
                job_id=submission_result['job_id'],
                assigned_nodes=assigned_nodes,
                estimated_start_time=self._estimate_start_time(request),
                payment_confirmation=self._create_payment_confirmation(payment_confirmation)
            )

        except Exception as e:
            logging.error(f"Error in SubmitJob: {str(e)}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetJobStatus(self, request: pb2.JobId, 
                          context: grpc.aio.ServicerContext) -> pb2.JobStatus:
        """Get current job status"""
        try:
            status = await self.job_executor.get_job_status(request.job_id)
            if not status:
                context.abort(grpc.StatusCode.NOT_FOUND, "Job not found")
            return status
        except Exception as e:
            logging.error(f"Get job status failed: {str(e)}")
            context.abort(grpc.StatusCode.INTERNAL, "Status retrieval failed")

    async def StreamMetrics(self, request: empty_pb2.Empty, 
                          context: grpc.aio.ServicerContext) -> pb2.MetricsReport:
        """Stream real-time metrics"""
        try:
            # Register stream
            stream_id = id(context)
            self.active_streams[stream_id] = True

            while self.active_streams[stream_id]:
                # Get current metrics
                metrics = await self.node_manager.get_current_metrics()

                # Create metrics report
                yield pb2.MetricsReport(
                    node_id=self.node_manager.node_id,
                    timestamp=metrics['timestamp'],
                    gpu_metrics=self._create_gpu_metrics(metrics['gpu']),
                    network_metrics=self._create_network_metrics(metrics['network']),
                    job_metrics=[self._create_job_metrics(j) for j in metrics['jobs']]
                )

                await asyncio.sleep(1)

        except Exception as e:
            logging.error(f"Error in StreamMetrics: {str(e)}")
            self.active_streams.pop(stream_id, None)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def SubmitVerificationProof(self, request: pb2.VerificationProof, 
                                    context: grpc.aio.ServicerContext) -> pb2.ProofValidation:
        """Handle verification proof submission"""
        try:
            # Select verifier nodes
            verifiers = self._select_verifier_nodes(request.job_id)

            # Create verification tasks
            verification_tasks = []
            for verifier_id in verifiers:
                task = self._verify_proof_with_node(verifier_id, request)
                verification_tasks.append(task)

            # Wait for verification results
            results = await asyncio.gather(*verification_tasks)

            # Aggregate results
            validation_result = self._aggregate_verification_results(results)

            # Update node reputation if valid
            if validation_result.valid:
                await self._update_node_reputation(request.node_id, True)

            # Process payment if valid
            payment_update = None
            if validation_result.valid:
                payment_update = await self.payment_processor.process_verification_payment(
                    job_id=request.job_id,
                    node_id=context.peer(),
                    amount=self._calculate_verification_payment(validation_result)
                )

            return pb2.ProofValidation(
                valid=validation_result.valid,
                validation_id=validation_result.validation_id,
                validator_signatures=validation_result.validator_signatures,
                payment_update=self._create_payment_update(payment_update) if payment_update else None
            )

        except Exception as e:
            logging.error(f"Error in SubmitVerificationProof: {str(e)}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    # Helper methods from first implementation
    def _generate_node_id(self, node_info: pb2.NodeInfo) -> str:
        """Generate unique node ID"""
        import hashlib
        import uuid
        
        # Combine node info with UUID for uniqueness
        node_data = f"{node_info.gpu_model}:{node_info.wallet_address}:{uuid.uuid4()}"
        return hashlib.sha256(node_data.encode()).hexdigest()[:16]

    def _get_bootstrap_nodes(self) -> List[str]:
        """Get list of bootstrap nodes"""
        return [node_id for node_id, last_heartbeat in self.active_nodes.items()
                if time.time() - last_heartbeat < 300]  # Active in last 5 minutes

    def _get_required_actions(self, node_id: str) -> List[str]:
        """Get list of required actions for node"""
        actions = []
        
        # Check if verification is needed
        if node_id in self.verifier_pool:
            actions.append('VERIFICATION_NEEDED')
            
        # Add other action checks as needed
        return actions

    async def _verify_proof_with_node(self, verifier_id: str, proof: pb2.VerificationProof):
        """Send proof to verifier node for validation"""
        try:
            # Get verifier node connection
            verifier_channel = await self._get_node_channel(verifier_id)
            
            # Create verification request
            verification_request = pb2.VerificationRequest(
                job_id=proof.job_id,
                checkpoint_id=proof.checkpoint_id,
                challenge_data=proof.proof_data
            )
            
            # Send to verifier
            stub = pb2_grpc.NodeServiceStub(verifier_channel)
            response = await stub.RequestVerification(verification_request)
            
            return response
        except Exception as e:
            logging.error(f"Verification with node {verifier_id} failed: {str(e)}")
            raise

    def _select_verifier_nodes(self, job_id: str) -> list:
        """Select nodes for proof verification"""
        verifiers = []
        for node_id, last_heartbeat in self.active_nodes.items():
            if self._is_node_eligible_verifier(node_id, job_id):
                verifiers.append(node_id)
            if len(verifiers) >= 3:  # Use configurable number of verifiers
                break
        return verifiers

    async def _distribute_job(self, job_spec: Dict, assigned_nodes: List[str]):
        """Distribute job to assigned nodes"""
        try:
            # Create job distribution tasks
            distribution_tasks = []
            for node_id in assigned_nodes:
                task = self._send_job_to_node(node_id, job_spec)
                distribution_tasks.append(task)
            
            # Wait for all distributions to complete
            await asyncio.gather(*distribution_tasks)
            
            # Initialize job tracking
            await self._initialize_job_tracking(job_spec['job_id'], assigned_nodes)
        except Exception as e:
            logging.error(f"Job distribution failed: {str(e)}")
            # Cleanup and revert assignments
            await self._cleanup_failed_distribution(job_spec['job_id'], assigned_nodes)
            raise

    def _convert_model_definition(model: pb2.ModelDefinition) -> Dict:
        """Convert protobuf ModelDefinition to internal format"""
        return {
            'model_format': model.model_format,
            'model_data': bytes(model.model_data),
            'hyperparameters': dict(model.hyperparameters),
            'required_packages': list(model.required_packages)
        }

    def _convert_training_config(config: pb2.TrainingConfig) -> Dict:
        """Convert protobuf TrainingConfig to internal format"""
        return {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'optimizer': config.optimizer,
            'optimizer_config': dict(config.optimizer_config),
            'metrics': list(config.metrics),
            'checkpoint_config': _convert_checkpoint_config(config.checkpoint_config)
        }

    def _convert_checkpoint_config(config: pb2.CheckpointConfig) -> Dict:
        """Convert protobuf CheckpointConfig to internal format"""
        return {
            'save_interval': config.save_interval,
            'max_checkpoints': config.max_checkpoints,
            'save_path': config.save_path
        }

    def _convert_resource_requirements(requirements: pb2.ResourceRequirements) -> Dict:
        """Convert protobuf ResourceRequirements to internal format"""
        return {
            'min_memory': requirements.min_memory,
            'min_compute_capability': requirements.min_compute_capability,
            'max_batch_size': requirements.max_batch_size,
            'expected_duration_seconds': requirements.expected_duration_seconds
        }

    def _convert_privacy_settings(settings: pb2.PrivacySettings) -> Dict:
        """Convert protobuf PrivacySettings to internal format"""
        return {
            'require_secure_enclave': settings.require_secure_enclave,
            'enable_federated_learning': settings.enable_federated_learning,
            'encryption_method': settings.encryption_method,
            'encryption_key': bytes(settings.encryption_key)
        }

    def _convert_verification_config(config: pb2.VerificationConfig) -> Dict:
        """Convert protobuf VerificationConfig to internal format"""
        return {
            'verification_interval': config.verification_interval,
            'required_verifiers': config.required_verifiers,
            'verification_timeout': config.verification_timeout,
            'verification_reward': str(config.verification_reward)
        }

    def _convert_payment_details(details: pb2.PaymentDetails) -> Dict:
        """Convert protobuf PaymentDetails to internal format"""
        return {
            'payment_id': details.payment_id,
            'currency': details.currency,
            'amount': str(details.amount),
            'recipient_address': details.recipient_address,
            'conditions': _convert_payment_conditions(details.conditions)
        }

    def _convert_payment_conditions(conditions: pb2.PaymentConditions) -> Dict:
        """Convert protobuf PaymentConditions to internal format"""
        return {
            'min_uptime': conditions.min_uptime,
            'min_verification_rate': conditions.min_verification_rate,
            'payment_interval_seconds': conditions.payment_interval_seconds,
            'performance_multiplier': conditions.performance_multiplier
        }

    def _convert_job_status(status: pb2.JobStatus) -> Dict:
        """Convert protobuf JobStatus to internal format"""
        return {
            'job_id': status.job_id,
            'status': status.status,
            'progress': status.progress,
            'current_metrics': dict(status.current_metrics),
            'last_update': status.last_update.timestamp(),
            'active_nodes': list(status.active_nodes)
        }

    def _create_gpu_metrics(metrics: Dict) -> pb2.GPUMetrics:
        """Create protobuf GPUMetrics from internal format"""
        return pb2.GPUMetrics(
            memory_used=metrics.get('memory_used', 0),
            memory_total=metrics.get('memory_total', 0),
            utilization=metrics.get('utilization', 0),
            temperature=metrics.get('temperature', 0),
            power_usage=metrics.get('power_usage', 0),
            custom_metrics=metrics.get('custom_metrics', {})
        )

    def _create_network_metrics(metrics: Dict) -> pb2.NetworkMetrics:
        """Create protobuf NetworkMetrics from internal format"""
        return pb2.NetworkMetrics(
            bandwidth_usage=metrics.get('bandwidth_usage', 0),
            active_connections=metrics.get('active_connections', 0),
            latency_ms=metrics.get('latency_ms', 0),
            bytes_transferred=metrics.get('bytes_transferred', 0)
        )

    def _create_job_metrics(metrics: Dict) -> pb2.JobMetrics:
        """Create protobuf JobMetrics from internal format"""
        return pb2.JobMetrics(
            job_id=metrics.get('job_id', ''),
            training_metrics=metrics.get('training_metrics', {}),
            resource_usage=_create_resource_usage(metrics.get('resource_usage', {}))
        )

    def _create_resource_usage(usage: Dict) -> pb2.ResourceUsage:
        """Create protobuf ResourceUsage from internal format"""
        return pb2.ResourceUsage(
            gpu_memory_percent=usage.get('gpu_memory_percent', 0),
            gpu_utilization=usage.get('gpu_utilization', 0),
            network_usage=usage.get('network_usage', 0),
            data_processed=usage.get('data_processed', 0)
        )

    def _create_payment_confirmation(confirmation: Dict) -> pb2.PaymentConfirmation:
        """Create protobuf PaymentConfirmation from internal format"""
        return pb2.PaymentConfirmation(
            transaction_id=confirmation.get('transaction_id', ''),
            escrow_address=confirmation.get('escrow_address', ''),
            amount_locked=float(confirmation.get('amount_locked', 0))
        )

    def _create_payment_update(update: Dict) -> pb2.PaymentUpdate:
        """Create protobuf PaymentUpdate from internal format"""
        return pb2.PaymentUpdate(
            payment_id=update.get('payment_id', ''),
            amount_released=float(update.get('amount_released', 0)),
            transaction_id=update.get('transaction_id', ''),
            reputation_change=update.get('reputation_change', 0)
        )

    def _validate_job_spec(spec: pb2.JobSpecification) -> bool:
        """Validate job specification completeness and correctness"""
        try:
            # Check required fields
            if not spec.job_id or not spec.framework:
                return False
                
            # Validate model definition
            if not spec.model.model_data:
                return False
                
            # Validate training config
            if spec.training_config.epochs <= 0 or spec.training_config.batch_size <= 0:
                return False
                
            # Validate resource requirements
            if spec.resource_requirements.min_memory <= 0:
                return False
                
            # All validations passed
            return True
            
        except Exception as e:
            logging.error(f"Error validating job spec: {str(e)}")
            return False