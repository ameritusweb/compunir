import asyncio
import logging
from typing import Dict, Optional, List
from decimal import Decimal
import time

from ..verification.verification_system import AdvancedVerificationSystem, VerificationProof
from ..payment.monero_payment_processor import MoneroPaymentProcessor
from ..core.unified_node_manager import NodeManager

class VerificationPaymentIntegrator:
    """Integrates verification results with payment processing"""
    
    def __init__(self, 
                 verification_system: AdvancedVerificationSystem,
                 payment_processor: MoneroPaymentProcessor,
                 node_manager: NodeManager,
                 config: Dict):
        self.verification_system = verification_system
        self.payment_processor = payment_processor
        self.node_manager = node_manager
        self.config = config
        
        self.pending_payments: Dict[str, Dict] = {}  # verification_id -> payment info
        self.verification_queue = asyncio.Queue()
        self.processing_task = None
        
        # Payment settings
        payment_config = config.get('payment', {})
        self.base_rate = Decimal(str(payment_config.get('base_rate', '0.001')))  # XMR per verification
        self.min_payment = Decimal(str(payment_config.get('min_payment', '0.0001')))
        self.quality_multiplier = payment_config.get('quality_multiplier', 1.5)
        
        logging.info("Initialized VerificationPaymentIntegrator")

    async def start(self):
        """Start the integrator"""
        self.processing_task = asyncio.create_task(self._process_verification_queue())
        logging.info("Started verification payment processing")

    async def stop(self):
        """Stop the integrator"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None

    async def handle_verification_result(self, 
                                      verification_id: str,
                                      verifier_node_id: str,
                                      result: Dict) -> Dict:
        """Handle a new verification result and trigger payment processing"""
        try:
            # Queue verification for payment processing
            payment_info = {
                'verification_id': verification_id,
                'verifier_node_id': verifier_node_id,
                'result': result,
                'timestamp': time.time()
            }
            
            await self.verification_queue.put(payment_info)
            self.pending_payments[verification_id] = payment_info
            
            # Return initial response
            return {
                'status': 'queued',
                'verification_id': verification_id,
                'estimated_payment': self._estimate_payment(result)
            }
            
        except Exception as e:
            logging.error(f"Error handling verification result: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def check_payment_status(self, verification_id: str) -> Dict:
        """Check status of payment for a verification"""
        try:
            if verification_id in self.pending_payments:
                payment_info = self.pending_payments[verification_id]
                return {
                    'status': 'pending',
                    'queued_at': payment_info['timestamp'],
                    'estimated_payment': self._estimate_payment(payment_info['result'])
                }
            
            # Check completed payments
            payment_result = await self.payment_processor.get_verification_payment(verification_id)
            if payment_result:
                return {
                    'status': 'completed',
                    'payment_details': payment_result
                }
            
            return {'status': 'not_found'}
            
        except Exception as e:
            logging.error(f"Error checking payment status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _process_verification_queue(self):
        """Process queued verifications and trigger payments"""
        try:
            while True:
                try:
                    # Get next verification from queue
                    payment_info = await self.verification_queue.get()
                    verification_id = payment_info['verification_id']
                    
                    try:
                        # Process verification result
                        processed_result = await self._process_verification_result(payment_info)
                        
                        if processed_result['should_pay']:
                            # Calculate payment amount
                            payment_amount = self._calculate_payment_amount(
                                payment_info['result'],
                                processed_result.get('quality_score', 1.0)
                            )
                            
                            # Process payment
                            payment_result = await self._process_payment(
                                verification_id=verification_id,
                                node_id=payment_info['verifier_node_id'],
                                amount=payment_amount,
                                result=payment_info['result']
                            )
                            
                            # Update verification status
                            await self.verification_system.update_verification_status(
                                verification_id=verification_id,
                                status='completed',
                                payment_details=payment_result
                            )
                            
                        else:
                            # Mark verification as rejected
                            await self.verification_system.update_verification_status(
                                verification_id=verification_id,
                                status='rejected',
                                reason=processed_result.get('rejection_reason')
                            )
                            
                    except Exception as e:
                        logging.error(f"Error processing verification payment: {str(e)}")
                        await self.verification_system.update_verification_status(
                            verification_id=verification_id,
                            status='error',
                            error=str(e)
                        )
                    
                    finally:
                        # Remove from pending payments
                        self.pending_payments.pop(verification_id, None)
                        self.verification_queue.task_done()
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.error(f"Error in verification queue processing: {str(e)}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logging.info("Verification queue processor stopped")
        except Exception as e:
            logging.error(f"Fatal error in verification queue processor: {str(e)}")

    async def _process_verification_result(self, payment_info: Dict) -> Dict:
        """Process verification result and determine if payment should be made"""
        try:
            result = payment_info['result']
            
            # Check basic validity
            if not result.get('is_valid', False):
                return {
                    'should_pay': False,
                    'rejection_reason': 'Invalid verification result'
                }
            
            # Calculate quality score
            quality_score = await self._calculate_quality_score(
                payment_info['verifier_node_id'],
                result
            )
            
            # Check minimum quality threshold
            min_quality = self.config.get('payment', {}).get('min_quality_score', 0.5)
            if quality_score < min_quality:
                return {
                    'should_pay': False,
                    'rejection_reason': f'Quality score {quality_score} below minimum {min_quality}',
                    'quality_score': quality_score
                }
            
            return {
                'should_pay': True,
                'quality_score': quality_score
            }
            
        except Exception as e:
            logging.error(f"Error processing verification result: {str(e)}")
            raise

    async def _calculate_quality_score(self, node_id: str, result: Dict) -> float:
        """Calculate quality score for verification result"""
        try:
            # Get node reputation
            node_info = await self.node_manager.get_node_info(node_id)
            reputation_score = node_info.get('reputation_score', 0.5)
            
            # Calculate metrics-based score
            metrics_score = self._calculate_metrics_score(result.get('metrics', {}))
            
            # Combine scores (weighted average)
            quality_score = (reputation_score * 0.4 + metrics_score * 0.6)
            
            return max(0.0, min(1.0, quality_score))  # Clamp between 0 and 1
            
        except Exception as e:
            logging.error(f"Error calculating quality score: {str(e)}")
            return 0.5  # Default to neutral score

    def _calculate_metrics_score(self, metrics: Dict) -> float:
        """Calculate score based on verification metrics"""
        try:
            scores = []
            
            # Response time score (lower is better)
            if 'response_time' in metrics:
                target_time = self.config.get('verification', {}).get('target_response_time', 1.0)
                response_score = max(0.0, min(1.0, target_time / metrics['response_time']))
                scores.append(response_score)
            
            # Compute quality score
            if 'compute_quality' in metrics:
                scores.append(metrics['compute_quality'])
            
            # Resource efficiency score
            if 'resource_efficiency' in metrics:
                scores.append(metrics['resource_efficiency'])
            
            return sum(scores) / len(scores) if scores else 0.5
            
        except Exception as e:
            logging.error(f"Error calculating metrics score: {str(e)}")
            return 0.5

    def _calculate_payment_amount(self, result: Dict, quality_score: float) -> Decimal:
        """Calculate payment amount based on verification result and quality"""
        try:
            # Start with base rate
            payment = self.base_rate
            
            # Apply quality multiplier
            quality_multiplier = 1.0 + ((quality_score - 0.5) * (self.quality_multiplier - 1.0))
            payment *= Decimal(str(quality_multiplier))
            
            # Apply resource usage factor
            if 'resource_usage' in result:
                resource_factor = min(2.0, max(0.5, float(result['resource_usage'])))
                payment *= Decimal(str(resource_factor))
            
            # Ensure minimum payment
            return max(self.min_payment, payment)
            
        except Exception as e:
            logging.error(f"Error calculating payment amount: {str(e)}")
            return self.min_payment

    async def _process_payment(self, 
                             verification_id: str,
                             node_id: str,
                             amount: Decimal,
                             result: Dict) -> Dict:
        """Process payment for successful verification"""
        try:
            # Create payment record
            payment_record = {
                'verification_id': verification_id,
                'node_id': node_id,
                'amount': amount,
                'quality_score': result.get('quality_score', 1.0),
                'timestamp': time.time()
            }
            
            # Process payment through payment processor
            payment_result = await self.payment_processor.process_verification_payment(
                node_id=node_id,
                amount=amount,
                verification_data=result,
                payment_record=payment_record
            )
            
            return {
                'status': 'completed',
                'payment_id': payment_result['payment_id'],
                'amount': str(amount),
                'transaction_id': payment_result['transaction_id']
            }
            
        except Exception as e:
            logging.error(f"Error processing payment: {str(e)}")
            raise

    def _estimate_payment(self, result: Dict) -> str:
        """Estimate payment amount for a verification result"""
        try:
            # Calculate basic quality score
            metrics_score = self._calculate_metrics_score(result.get('metrics', {}))
            
            # Calculate estimated payment
            estimated_amount = self._calculate_payment_amount(result, metrics_score)
            
            return str(estimated_amount)
            
        except Exception as e:
            logging.error(f"Error estimating payment: {str(e)}")
            return str(self.min_payment)