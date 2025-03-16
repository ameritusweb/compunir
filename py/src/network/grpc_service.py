import grpc
from concurrent import futures
import time
from typing import Dict, Optional
import asyncio
from google.protobuf import empty_pb2
import logging

from .generated import node_service_pb2 as pb2
from .generated import node_service_pb2_grpc as pb2_grpc

class NodeServicer(pb2_grpc.NodeServiceServicer):
    def __init__(self, node_manager, job_executor):
        self.node_manager = node_manager
        self.job_executor = job_executor
        self.active_nodes: Dict[str, float] = {}  # node_id -> last_heartbeat
        self.verifier_pool = []
        
    async def RegisterNode(self, request: pb2.NodeInfo, context) -> pb2.RegistrationResponse:
        """Handle new node registration"""
        try:
            # Validate node information
            if not self._validate_node_info(request):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid node information")
                
            # Generate node ID and register
            node_id = self._generate_node_id(request)
            self.active_nodes[node_id] = time.time()
            
            # Get bootstrap nodes
            bootstrap_nodes = self._get_bootstrap_nodes()
            
            return pb2.RegistrationResponse(
                assigned_node_id=node_id,
                bootstrap_nodes=bootstrap_nodes,
                network_config=self._get_network_config()
            )
        except Exception as e:
            logging.error(f"Node registration failed: {str(e)}")
            context.abort(grpc.StatusCode.INTERNAL, "Registration failed")

    async def Heartbeat(self, request: pb2.NodeStatus, context) -> pb2.HeartbeatResponse:
        """Process node heartbeat and status update"""
        try:
            node_id = request.node_id
            if node_id not in self.active_nodes:
                context.abort(grpc.StatusCode.NOT_FOUND, "Node not registered")
                
            # Update last heartbeat time
            self.active_nodes[node_id] = time.time()
            
            # Process status update
            self._process_status_update(request)
            
            # Check for required actions
            actions = self._get_required_actions(node_id)
            
            return pb2.HeartbeatResponse(
                accepted=True,
                actions_required=actions,
                updated_config=self._get_network_config()
            )
        except Exception as e:
            logging.error(f"Heartbeat processing failed: {str(e)}")
            return pb2.HeartbeatResponse(accepted=False)

    async def SubmitJob(self, request: pb2.JobSpecification, context) -> pb2.JobSubmissionResponse:
        """Handle new job submission"""
        try:
            # Validate job specification
            if not self._validate_job_spec(request):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid job specification")
                
            # Find suitable nodes for the job
            assigned_nodes = await self._assign_nodes(request)
            if not assigned_nodes:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "No suitable nodes available")
                
            # Process payment details
            payment_confirmation = await self._process_payment(request.payment_details)
            
            # Distribute job to assigned nodes
            await self._distribute_job(request, assigned_nodes)
            
            return pb2.JobSubmissionResponse(
                job_id=request.job_id,
                assigned_nodes=assigned_nodes,
                estimated_start_time=self._estimate_start_time(request),
                payment_confirmation=payment_confirmation
            )
        except Exception as e:
            logging.error(f"Job submission failed: {str(e)}")
            context.abort(grpc.StatusCode.INTERNAL, "Job submission failed")

    async def GetJobStatus(self, request: pb2.JobId, context) -> pb2.JobStatus:
        """Get current job status"""
        try:
            status = await self.job_executor.get_job_status(request.job_id)
            if not status:
                context.abort(grpc.StatusCode.NOT_FOUND, "Job not found")
            return status
        except Exception as e:
            logging.error(f"Get job status failed: {str(e)}")
            context.abort(grpc.StatusCode.INTERNAL, "Status retrieval failed")

    async def StreamMetrics(self, request: empty_pb2.Empty, context) -> pb2.MetricsReport:
        """Stream real-time metrics"""
        try:
            while True:
                metrics = await self.node_manager.get_current_metrics()
                yield pb2.MetricsReport(
                    node_id=self.node_manager.node_id,
                    timestamp=metrics.timestamp,
                    gpu_metrics=metrics.gpu_metrics,
                    network_metrics=metrics.network_metrics,
                    job_metrics=metrics.job_metrics
                )
                await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"Metrics streaming failed: {str(e)}")
            context.abort(grpc.StatusCode.INTERNAL, "Metrics streaming failed")

    async def SubmitVerificationProof(self, request: pb2.VerificationProof, context) -> pb2.ProofValidation:
        """Handle verification proof submission"""
        try:
            # Validate the proof
            validation_result = await self._validate_proof(request)
            
            # Update node reputation if validation successful
            if validation_result.valid:
                await self._update_node_reputation(request.node_id, True)
                
            # Process payment update
            payment_update = await self._process_verification_payment(request)
            
            return pb2.ProofValidation(
                valid=validation_result.valid,
                validation_id=validation_result.validation_id,
                validator_signatures=validation_result.signatures,
                payment_update=payment_update
            )
        except Exception as e:
            logging.error(f"Proof validation failed: {str(e)}")
            context.abort(grpc.StatusCode.INTERNAL, "Proof validation failed")

    def _validate_node_info(self, node_info: pb2.NodeInfo) -> bool:
        """Validate node information"""
        # Implementation would check GPU capabilities, network requirements, etc.
        pass

    def _generate_node_id(self, node_info: pb2.NodeInfo) -> str:
        """Generate unique node ID"""
        # Implementation would create a unique identifier
        pass

    def _get_bootstrap_nodes(self) -> list:
        """Get list of bootstrap nodes"""
        # Implementation would return active nodes for bootstrapping
        pass

    def _get_network_config(self) -> pb2.NetworkConfiguration:
        """Get current network configuration"""
        # Implementation would return network settings
        pass

    async def _assign_nodes(self, job_spec: pb2.JobSpecification) -> list:
        """Find suitable nodes for job execution"""
        # Implementation would match job requirements with available nodes
        pass

    async def _process_payment(self, payment_details: pb2.PaymentDetails) -> pb2.PaymentConfirmation:
        """Process payment for job submission"""
        try:
            # Verify payment amount matches resource requirements
            if not self._verify_payment_amount(payment_details):
                raise ValueError("Insufficient payment amount")
                
            # Create escrow transaction
            escrow_address = self._create_escrow_address()
            transaction_id = await self._create_escrow_transaction(
                payment_details.amount,
                escrow_address,
                payment_details.currency
            )
            
            return pb2.PaymentConfirmation(
                transaction_id=transaction_id,
                escrow_address=escrow_address,
                amount_locked=payment_details.amount
            )
        except Exception as e:
            logging.error(f"Payment processing failed: {str(e)}")
            raise

    async def _distribute_job(self, job_spec: pb2.JobSpecification, assigned_nodes: list):
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
            await self._initialize_job_tracking(job_spec.job_id, assigned_nodes)
        except Exception as e:
            logging.error(f"Job distribution failed: {str(e)}")
            # Cleanup and revert assignments
            await self._cleanup_failed_distribution(job_spec.job_id, assigned_nodes)
            raise

    async def _send_job_to_node(self, node_id: str, job_spec: pb2.JobSpecification):
        """Send job to specific node"""
        try:
            # Get node connection
            node_channel = await self._get_node_channel(node_id)
            
            # Create job submission request
            job_request = self._prepare_job_request(job_spec, node_id)
            
            # Send job to node
            stub = pb2_grpc.NodeServiceStub(node_channel)
            response = await stub.SubmitJob(job_request)
            
            # Verify response
            if not response.accepted:
                raise Exception(f"Node {node_id} rejected job")
                
            return response
        except Exception as e:
            logging.error(f"Failed to send job to node {node_id}: {str(e)}")
            raise

    async def _validate_proof(self, proof: pb2.VerificationProof):
        """Validate computation proof"""
        try:
            # Select verifier nodes
            verifiers = self._select_verifier_nodes(proof.job_id)
            
            # Create verification tasks
            verification_tasks = []
            for verifier_id in verifiers:
                task = self._verify_proof_with_node(verifier_id, proof)
                verification_tasks.append(task)
            
            # Wait for verification results
            results = await asyncio.gather(*verification_tasks)
            
            # Aggregate verification results
            validation_result = self._aggregate_verification_results(results)
            
            return validation_result
        except Exception as e:
            logging.error(f"Proof validation failed: {str(e)}")
            raise

    def _select_verifier_nodes(self, job_id: str) -> list:
        """Select nodes for proof verification"""
        # Implementation of verifier selection strategy
        # Could consider node reputation, availability, and past performance
        verifiers = []
        for node_id, last_heartbeat in self.active_nodes.items():
            if self._is_node_eligible_verifier(node_id, job_id):
                verifiers.append(node_id)
            if len(verifiers) >= 3:  # Use configurable number of verifiers
                break
        return verifiers

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

    def _aggregate_verification_results(self, results: list) -> pb2.ProofValidation:
        """Aggregate verification results from multiple verifiers"""
        # Count positive validations
        valid_count = sum(1 for result in results if result.verified)
        
        # Decision based on majority
        is_valid = valid_count > len(results) / 2
        
        # Collect signatures from validators
        validator_signatures = []
        for result in results:
            if result.verified:
                validator_signatures.extend(result.verifier_nodes)
        
        return pb2.ProofValidation(
            valid=is_valid,
            validation_id=f"val_{time.time()}",
            validator_signatures=validator_signatures
        )

    async def _update_node_reputation(self, node_id: str, success: bool):
        """Update node reputation based on verification result"""
        try:
            # Get current reputation
            current_reputation = await self._get_node_reputation(node_id)
            
            # Update based on success/failure
            new_reputation = self._calculate_new_reputation(
                current_reputation,
                success
            )
            
            # Store updated reputation
            await self._store_node_reputation(node_id, new_reputation)
            
            # Update node rankings
            await self._update_node_rankings()
        except Exception as e:
            logging.error(f"Reputation update failed: {str(e)}")
            raise

    async def _process_verification_payment(self, proof: pb2.VerificationProof) -> pb2.PaymentUpdate:
        """Process payment update based on verification"""
        try:
            # Calculate payment amount
            payment_amount = self._calculate_payment_amount(proof)
            
            # Release payment from escrow
            transaction_id = await self._release_escrow_payment(
                proof.job_id,
                payment_amount
            )
            
            # Update payment tracking
            await self._update_payment_tracking(proof.job_id, payment_amount)
            
            return pb2.PaymentUpdate(
                payment_id=f"pay_{time.time()}",
                amount_released=payment_amount,
                transaction_id=transaction_id,
                reputation_change=0.1  # Would be calculated based on performance
            )
        except Exception as e:
            logging.error(f"Payment processing failed: {str(e)}")
            raise

    def _calculate_payment_amount(self, proof: pb2.VerificationProof) -> float:
        """Calculate payment amount based on work performed"""
        # Implementation would consider:
        # - Resource usage
        # - Time spent
        # - Quality of results
        # - Node reputation
        pass