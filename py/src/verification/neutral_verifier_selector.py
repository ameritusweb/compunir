from typing import Dict, List, Set, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import time
import logging
import asyncio
from collections import defaultdict

@dataclass
class VerifierMetrics:
    """Metrics for verifier selection"""
    node_id: str
    total_verifications: int
    successful_verifications: int
    dispute_participations: int
    average_response_time: float
    reputation_score: float
    last_selection_time: float
    stake_amount: float
    network_score: float

class NeutralVerifierSelector:
    """Select neutral verifiers for dispute resolution"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.verifier_metrics: Dict[str, VerifierMetrics] = {}
        self.dispute_history: Dict[str, List[Dict]] = {}
        self.selection_history: Dict[str, List[str]] = {}
        self.stake_weights = config.get('stake_weights', {
            'reputation': 0.3,
            'stake': 0.2,
            'response_time': 0.2,
            'network': 0.15,
            'diversity': 0.15
        })
        
    async def select_neutral_verifiers(self,
                                     required_count: int,
                                     excluded_verifiers: Set[str],
                                     dispute_id: str) -> List[str]:
        """Select neutral verifiers for dispute resolution"""
        try:
            available_verifiers = self._get_available_verifiers(excluded_verifiers)
            if not available_verifiers:
                raise ValueError("No available verifiers")
                
            # Calculate selection scores
            verifier_scores = await self._calculate_verifier_scores(
                available_verifiers,
                dispute_id
            )
            
            # Apply diversity bonuses
            diversity_scores = self._calculate_diversity_scores(available_verifiers)
            
            # Combine scores
            final_scores = self._combine_scores(verifier_scores, diversity_scores)
            
            # Select verifiers
            selected_verifiers = self._select_top_verifiers(
                final_scores,
                required_count
            )
            
            # Update selection history
            self._update_selection_history(dispute_id, selected_verifiers)
            
            return selected_verifiers
            
        except Exception as e:
            logging.error(f"Error selecting neutral verifiers: {str(e)}")
            raise
            
    def _get_available_verifiers(self, excluded_verifiers: Set[str]) -> List[str]:
        """Get list of available verifiers excluding specified ones"""
        return [
            v_id for v_id in self.verifier_metrics.keys()
            if v_id not in excluded_verifiers
            and self._is_verifier_available(v_id)
        ]
        
    async def _calculate_verifier_scores(self,
                                       verifiers: List[str],
                                       dispute_id: str) -> Dict[str, float]:
        """Calculate selection scores for verifiers"""
        try:
            scores = {}
            for verifier_id in verifiers:
                metrics = self.verifier_metrics[verifier_id]
                
                # Calculate component scores
                reputation_score = self._calculate_reputation_score(metrics)
                stake_score = self._calculate_stake_score(metrics)
                response_score = self._calculate_response_score(metrics)
                network_score = metrics.network_score
                
                # Combine weighted scores
                scores[verifier_id] = (
                    reputation_score * self.stake_weights['reputation'] +
                    stake_score * self.stake_weights['stake'] +
                    response_score * self.stake_weights['response_time'] +
                    network_score * self.stake_weights['network']
                )
                
            return scores
            
        except Exception as e:
            logging.error(f"Error calculating verifier scores: {str(e)}")
            raise
            
    def _calculate_diversity_scores(self, verifiers: List[str]) -> Dict[str, float]:
        """Calculate diversity scores based on historical selections"""
        try:
            diversity_scores = {}
            recent_selections = self._get_recent_selections()
            
            for verifier_id in verifiers:
                # Calculate selection frequency
                selection_count = sum(
                    1 for selections in recent_selections
                    if verifier_id in selections
                )
                
                # Calculate co-selection patterns
                co_selection_penalty = self._calculate_co_selection_penalty(
                    verifier_id,
                    recent_selections
                )
                
                # Combined diversity score
                diversity_scores[verifier_id] = 1.0 - (
                    (selection_count / max(1, len(recent_selections))) +
                    co_selection_penalty
                )
                
            return diversity_scores
            
        except Exception as e:
            logging.error(f"Error calculating diversity scores: {str(e)}")
            raise
            
    def _combine_scores(self,
                       verifier_scores: Dict[str, float],
                       diversity_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine verifier and diversity scores"""
        try:
            final_scores = {}
            for verifier_id in verifier_scores:
                base_score = verifier_scores[verifier_id]
                diversity_score = diversity_scores[verifier_id]
                
                final_scores[verifier_id] = (
                    base_score * (1.0 - self.stake_weights['diversity']) +
                    diversity_score * self.stake_weights['diversity']
                )
                
            return final_scores
            
        except Exception as e:
            logging.error(f"Error combining scores: {str(e)}")
            raise
            
    def _select_top_verifiers(self,
                            scores: Dict[str, float],
                            count: int) -> List[str]:
        """Select top verifiers based on scores"""
        try:
            # Sort verifiers by score
            sorted_verifiers = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select top verifiers
            selected = [v[0] for v in sorted_verifiers[:count]]
            
            # Update selection time
            current_time = time.time()
            for verifier_id in selected:
                if verifier_id in self.verifier_metrics:
                    self.verifier_metrics[verifier_id].last_selection_time = current_time
                    
            return selected
            
        except Exception as e:
            logging.error(f"Error selecting top verifiers: {str(e)}")
            raise
            
    def _calculate_reputation_score(self, metrics: VerifierMetrics) -> float:
        """Calculate reputation-based score"""
        try:
            if metrics.total_verifications == 0:
                return 0.5  # Default score for new verifiers
                
            # Base reputation from verification success rate
            base_reputation = metrics.successful_verifications / metrics.total_verifications
            
            # Adjust for dispute participation
            dispute_ratio = metrics.dispute_participations / metrics.total_verifications
            dispute_penalty = max(0, dispute_ratio - 0.1)  # Penalize high dispute rates
            
            # Apply time decay to reputation
            time_since_last = time.time() - metrics.last_selection_time
            time_decay = np.exp(-time_since_last / self.config.get('reputation_decay_time', 86400))
            
            return base_reputation * (1.0 - dispute_penalty) * time_decay
            
        except Exception as e:
            logging.error(f"Error calculating reputation score: {str(e)}")
            return 0.0
            
    def _calculate_stake_score(self, metrics: VerifierMetrics) -> float:
        """Calculate stake-based score"""
        try:
            # Get stake thresholds
            min_stake = self.config.get('min_stake', 100)
            max_stake = self.config.get('max_stake', 10000)
            
            # Normalize stake amount
            normalized_stake = (metrics.stake_amount - min_stake) / (max_stake - min_stake)
            return max(0.0, min(1.0, normalized_stake))
            
        except Exception as e:
            logging.error(f"Error calculating stake score: {str(e)}")
            return 0.0
            
    def _calculate_response_score(self, metrics: VerifierMetrics) -> float:
        """Calculate response time-based score"""
        try:
            # Get response time thresholds
            target_time = self.config.get('target_response_time', 1.0)
            max_time = self.config.get('max_response_time', 5.0)
            
            # Calculate score based on average response time
            if metrics.average_response_time <= target_time:
                return 1.0
            elif metrics.average_response_time >= max_time:
                return 0.0
            else:
                return 1.0 - (
                    (metrics.average_response_time - target_time) /
                    (max_time - target_time)
                )
                
        except Exception as e:
            logging.error(f"Error calculating response score: {str(e)}")
            return 0.0
            
    def _calculate_co_selection_penalty(self,
                                      verifier_id: str,
                                      recent_selections: List[List[str]]) -> float:
        """Calculate penalty for frequent co-selection"""
        try:
            if not recent_selections:
                return 0.0
                
            # Count co-selections with other verifiers
            co_selections = defaultdict(int)
            for selection in recent_selections:
                if verifier_id in selection:
                    for other_id in selection:
                        if other_id != verifier_id:
                            co_selections[other_id] += 1
                            
            # Calculate penalty based on maximum co-selection frequency
            max_co_selections = max(co_selections.values()) if co_selections else 0
            return max_co_selections / len(recent_selections)
            
        except Exception as e:
            logging.error(f"Error calculating co-selection penalty: {str(e)}")
            return 0.0
            
    def _get_recent_selections(self) -> List[List[str]]:
        """Get recent verifier selections"""
        try:
            # Get all recent selections
            all_selections = []
            max_history = self.config.get('selection_history_length', 100)
            
            for dispute_selections in self.selection_history.values():
                all_selections.extend(dispute_selections[-max_history:])
                
            return all_selections[-max_history:]
            
        except Exception as e:
            logging.error(f"Error getting recent selections: {str(e)}")
            return []
            
    def _is_verifier_available(self, verifier_id: str) -> bool:
        """Check if verifier is available for selection"""
        try:
            metrics = self.verifier_metrics[verifier_id]
            
            # Check minimum stake requirement
            if metrics.stake_amount < self.config.get('min_stake', 100):
                return False
                
            # Check cooldown period
            cooldown_period = self.config.get('selection_cooldown', 3600)
            if time.time() - metrics.last_selection_time < cooldown_period:
                return False
                
            # Check minimum reputation
            if metrics.reputation_score < self.config.get('min_reputation', 0.5):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error checking verifier availability: {str(e)}")
            return False
            
    def _update_selection_history(self, dispute_id: str, selected_verifiers: List[str]):
        """Update verifier selection history"""
        try:
            if dispute_id not in self.selection_history:
                self.selection_history[dispute_id] = []
                
            self.selection_history[dispute_id].append(selected_verifiers)
            
            # Prune old history
            max_history = self.config.get('max_dispute_history', 1000)
            if len(self.selection_history[dispute_id]) > max_history:
                self.selection_history[dispute_id] = (
                    self.selection_history[dispute_id][-max_history:]
                )
                
        except Exception as e:
            logging.error(f"Error updating selection history: {str(e)}")
            raise