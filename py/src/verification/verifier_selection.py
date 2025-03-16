from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import time
import logging
from collections import defaultdict
import heapq
import random
from scipy.stats import beta

@dataclass
class VerifierStats:
    node_id: str
    successful_verifications: int
    total_verifications: int
    average_response_time: float
    last_active: float
    gpu_capacity: float
    network_latency: float
    verification_accuracy: float
    reputation_score: float

class VerifierSelectionSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.verifier_stats: Dict[str, VerifierStats] = {}
        self.verification_history = defaultdict(list)
        self.node_pairs_history = defaultdict(int)
        self.min_verifiers = config.get('min_verifiers', 3)
        self.max_verifiers = config.get('max_verifiers', 7)
        self.reputation_decay = config.get('reputation_decay', 0.99)
        self.network_weight = config.get('network_weight', 0.3)
        self.capacity_weight = config.get('capacity_weight', 0.3)
        self.reputation_weight = config.get('reputation_weight', 0.4)

    async def select_verifiers(self, 
                             job_id: str,
                             compute_size: float,
                             available_nodes: List[str],
                             excluded_nodes: Optional[List[str]] = None) -> List[str]:
        """Select optimal set of verifiers for a job"""
        try:
            if not available_nodes:
                raise ValueError("No nodes available for verification")

            # Remove excluded nodes
            if excluded_nodes:
                available_nodes = [n for n in available_nodes if n not in excluded_nodes]

            # Calculate selection scores
            node_scores = await self._calculate_node_scores(available_nodes, compute_size)

            # Apply diversity constraints
            selected_verifiers = self._apply_diversity_selection(
                node_scores,
                min_verifiers=self.min_verifiers,
                max_verifiers=self.max_verifiers
            )

            # Update selection history
            self._update_selection_history(job_id, selected_verifiers)

            return selected_verifiers

        except Exception as e:
            logging.error(f"Error selecting verifiers: {str(e)}")
            raise

    async def update_verifier_stats(self,
                                  node_id: str,
                                  verification_result: bool,
                                  response_time: float,
                                  metrics: Dict):
        """Update verifier statistics based on performance"""
        try:
            stats = self.verifier_stats.get(node_id)
            if not stats:
                stats = VerifierStats(
                    node_id=node_id,
                    successful_verifications=0,
                    total_verifications=0,
                    average_response_time=0,
                    last_active=0,
                    gpu_capacity=0,
                    network_latency=0,
                    verification_accuracy=0,
                    reputation_score=0.5
                )

            # Update verification counts
            stats.total_verifications += 1
            if verification_result:
                stats.successful_verifications += 1

            # Update response time average
            stats.average_response_time = (
                stats.average_response_time * (stats.total_verifications - 1) +
                response_time
            ) / stats.total_verifications

            # Update last active timestamp
            stats.last_active = time.time()

            # Update GPU and network metrics
            stats.gpu_capacity = metrics.get('gpu_capacity', stats.gpu_capacity)
            stats.network_latency = metrics.get('network_latency', stats.network_latency)

            # Update verification accuracy
            stats.verification_accuracy = (
                stats.successful_verifications / stats.total_verifications
            )

            # Update reputation score using beta distribution
            stats.reputation_score = self._calculate_reputation_score(stats)

            # Apply reputation decay
            self._apply_reputation_decay(stats)

            # Store updated stats
            self.verifier_stats[node_id] = stats

        except Exception as e:
            logging.error(f"Error updating verifier stats: {str(e)}")
            raise

    async def _calculate_node_scores(self, 
                                   nodes: List[str],
                                   compute_size: float) -> Dict[str, float]:
        """Calculate selection scores for available nodes"""
        try:
            node_scores = {}
            for node_id in nodes:
                stats = self.verifier_stats.get(node_id)
                if not stats:
                    continue

                # Calculate base score components
                reputation_score = stats.reputation_score
                network_score = self._calculate_network_score(stats)
                capacity_score = self._calculate_capacity_score(stats, compute_size)

                # Combine scores with weights
                combined_score = (
                    self.reputation_weight * reputation_score +
                    self.network_weight * network_score +
                    self.capacity_weight * capacity_score
                )

                # Apply recent activity bonus/penalty
                activity_modifier = self._calculate_activity_modifier(stats)
                final_score = combined_score * activity_modifier

                node_scores[node_id] = final_score

            return node_scores

        except Exception as e:
            logging.error(f"Error calculating node scores: {str(e)}")
            raise

    def _apply_diversity_selection(self,
                                 node_scores: Dict[str, float],
                                 min_verifiers: int,
                                 max_verifiers: int) -> List[str]:
        """Select diverse set of verifiers using score-based sampling"""
        try:
            selected_verifiers = []
            remaining_nodes = list(node_scores.keys())

            # Ensure minimum number of verifiers
            while len(selected_verifiers) < min_verifiers and remaining_nodes:
                # Convert scores to probabilities
                total_score = sum(node_scores[n] for n in remaining_nodes)
                probabilities = [node_scores[n] / total_score for n in remaining_nodes]

                # Sample node based on scores
                selected_idx = np.random.choice(len(remaining_nodes), p=probabilities)
                selected_node = remaining_nodes.pop(selected_idx)
                selected_verifiers.append(selected_node)

                # Update node pair history
                self._update_node_pairs(selected_verifiers)

            # Add additional verifiers if beneficial
            while len(selected_verifiers) < max_verifiers and remaining_nodes:
                # Calculate marginal benefit of adding another verifier
                current_diversity = self._calculate_verifier_diversity(selected_verifiers)
                best_marginal_improvement = 0
                best_additional_node = None

                for node in remaining_nodes:
                    potential_verifiers = selected_verifiers + [node]
                    new_diversity = self._calculate_verifier_diversity(potential_verifiers)
                    improvement = new_diversity - current_diversity

                    if improvement > best_marginal_improvement:
                        best_marginal_improvement = improvement
                        best_additional_node = node

                # Add node if improvement is significant
                if best_additional_node and best_marginal_improvement > self.config.get('min_diversity_improvement', 0.1):
                    selected_verifiers.append(best_additional_node)
                    remaining_nodes.remove(best_additional_node)
                else:
                    break

            return selected_verifiers

        except Exception as e:
            logging.error(f"Error in diversity selection: {str(e)}")
            raise

    def _calculate_reputation_score(self, stats: VerifierStats) -> float:
        """Calculate reputation score using beta distribution"""
        try:
            # Add small constant to avoid division by zero
            alpha = stats.successful_verifications + 1
            beta_param = (stats.total_verifications - stats.successful_verifications) + 1

            # Use beta distribution mean for reputation score
            reputation = beta.mean(alpha, beta_param)

            # Adjust for response time
            time_penalty = min(1.0, stats.average_response_time / self.config.get('max_response_time', 60))
            reputation *= (1 - time_penalty * 0.5)

            return reputation

        except Exception as e:
            logging.error(f"Error calculating reputation score: {str(e)}")
            return 0.5

    def _calculate_network_score(self, stats: VerifierStats) -> float:
        """Calculate network suitability score"""
        try:
            # Convert latency to score (lower is better)
            max_latency = self.config.get('max_acceptable_latency', 1000)
            latency_score = 1 - min(1.0, stats.network_latency / max_latency)

            # Consider recent activity
            time_since_active = time.time() - stats.last_active
            activity_score = np.exp(-time_since_active / self.config.get('activity_decay', 3600))

            return (latency_score + activity_score) / 2

        except Exception as e:
            logging.error(f"Error calculating network score: {str(e)}")
            return 0.0

    def _calculate_capacity_score(self, stats: VerifierStats, compute_size: float) -> float:
        """Calculate capacity suitability score"""
        try:
            # Check if node has sufficient capacity
            if stats.gpu_capacity < compute_size:
                return 0.0

            # Calculate utilization efficiency
            utilization_ratio = compute_size / stats.gpu_capacity
            efficiency_score = 1 - abs(0.7 - utilization_ratio)  # Optimal at 70% utilization

            return max(0.0, efficiency_score)

        except Exception as e:
            logging.error(f"Error calculating capacity score: {str(e)}")
            return 0.0

    def _calculate_activity_modifier(self, stats: VerifierStats) -> float:
        """Calculate activity-based score modifier"""
        try:
            time_since_active = time.time() - stats.last_active
            recent_activity_bonus = np.exp(-time_since_active / self.config.get('activity_decay', 3600))

            # Penalize very frequent selection
            if time_since_active < self.config.get('min_selection_interval', 300):
                return 0.5 * recent_activity_bonus

            return recent_activity_bonus

        except Exception as e:
            logging.error(f"Error calculating activity modifier: {str(e)}")
            return 1.0

    def _calculate_verifier_diversity(self, verifiers: List[str]) -> float:
        """Calculate diversity score for a set of verifiers"""
        try:
            if len(verifiers) < 2:
                return 0.0

            # Calculate pairwise diversity scores
            diversity_scores = []
            for i in range(len(verifiers)):
                for j in range(i + 1, len(verifiers)):
                    pair_key = tuple(sorted([verifiers[i], verifiers[j]]))
                    joint_selections = self.node_pairs_history[pair_key]
                    
                    # Lower score for frequently paired nodes
                    pair_diversity = 1.0 / (1.0 + np.log1p(joint_selections))
                    diversity_scores.append(pair_diversity)

            return np.mean(diversity_scores)

        except Exception as e:
            logging.error(f"Error calculating verifier diversity: {str(e)}")
            return 0.0

    def _update_node_pairs(self, verifiers: List[str]):
        """Update history of node pair selections"""
        try:
            for i in range(len(verifiers)):
                for j in range(i + 1, len(verifiers)):
                    pair_key = tuple(sorted([verifiers[i], verifiers[j]]))
                    self.node_pairs_history[pair_key] += 1

        except Exception as e:
            logging.error(f"Error updating node pairs: {str(e)}")

    def _apply_reputation_decay(self, stats: VerifierStats):
        """Apply time-based decay to reputation score"""
        try:
            time_since_active = time.time() - stats.last_active
            decay_factor = self.reputation_decay ** (time_since_active / 86400)  # Daily decay
            stats.reputation_score *= decay_factor

        except Exception as e:
            logging.error(f"Error applying reputation decay: {str(e)}")

    def _update_selection_history(self, job_id: str, selected_verifiers: List[str]):
        """Update verifier selection history"""
        try:
            self.verification_history[job_id].append({
                'verifiers': selected_verifiers,
                'timestamp': time.time()
            })

            # Prune old history
            max_history = self.config.get('max_history_size', 1000)
            if len(self.verification_history[job_id]) > max_history:
                self.verification_history[job_id] = self.verification_history[job_id][-max_history:]

        except Exception as e:
            logging.error(f"Error updating selection history: {str(e)}")
            raise