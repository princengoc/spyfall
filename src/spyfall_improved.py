import numpy as np
import torch
import pyro
import pyro.distributions as dist
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print(matplotlib.get_backend())
from tqdm import tqdm
from collections import Counter

# Utility functions for the game
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate KL divergence D_KL(q||p)."""
    # Add small epsilon to avoid log(0)
    p = p + 1e-10
    q = q + 1e-10
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # KL divergence
    return np.sum(q * np.log(q / p))

# Player base class
class Player:
    def __init__(self, player_id: int, num_players: int, num_locations: int, 
                 is_spy: bool, knows_location: Optional[int] = None):
        self.player_id = player_id
        self.num_players = num_players
        self.num_locations = num_locations
        self.is_spy = is_spy
        
        # Initialize beliefs
        if is_spy:
            # Spy doesn't know the location
            self.location_belief = np.ones(num_locations) / num_locations
        else:
            # Non-spy knows the location
            self.location_belief = np.zeros(num_locations)
            self.location_belief[knows_location] = 1.0
        
        # Initialize spy belief (everyone starts with uniform)
        self.spy_belief = np.ones(num_players) / num_players
        # everyone claims that they are NOT the spy, including the spy themselves
        self.spy_belief[player_id] = 0.0
        self.spy_belief = self.spy_belief / np.sum(self.spy_belief)
        
        # Game history
        self.game_history = []
    
    def update_beliefs(self, speaker_id: int, claim: np.ndarray, accusation: np.ndarray,
                      public_location_belief: np.ndarray, 
                      public_spy_belief: np.ndarray):
        """Update beliefs based on a new claim and accusation."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def generate_claim(self, public_location_belief: np.ndarray,
                      public_spy_belief: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a claim based on the player's strategy and provide accusation.
        
        Returns:
            Tuple of (location_claim, spy_accusation)
        """
        raise NotImplementedError("Subclasses must implement this method.")


# Bayesian Player using Flat Hide-Reveal strategy
class BayesianFlatHideRevealPlayer(Player):
    def __init__(self, player_id: int, num_players: int, num_locations: int, 
                 is_spy: bool, knows_location: Optional[int] = None,
                 beta: float = 5.0, lambda_val: float = 0.7):
        super().__init__(player_id, num_players, num_locations, is_spy, knows_location)
        self.beta = beta  # Concentration parameter
        self.lambda_val = lambda_val  # Truthfulness parameter
    
    def update_beliefs(self, speaker_id: int, claim: np.ndarray, accusation: np.ndarray,
                      public_location_belief: np.ndarray, 
                      public_spy_belief: np.ndarray):
        """Update beliefs using Bayesian update with both claim and accusation."""
        # Don't update our own beliefs based on our own claims
        if speaker_id == self.player_id:
            return
        
        if self.is_spy:
            # Spy updates location belief
            # P(N=l|G_t) ∝ P(C_t|N=l,j_t) P(N=l|G_{t-1})
            
            # Simple implementation: update proportionally to the claim
            new_belief = self.location_belief * (claim + 0.2)  # Adding noise to prevent convergence too fast
            self.location_belief = new_belief / np.sum(new_belief)
        
        # Everyone updates spy belief
        # Trust accusation based on how much we trust the speaker
        # If speaker's spy probability is high, trust less
        speaker_trustworthiness = 1.0 - self.spy_belief[speaker_id]
        trust_factor = 0.5 * speaker_trustworthiness  # Scale trust by suspicion
        
        # Blend our current spy belief with the accusation
        new_spy_belief = (1 - trust_factor) * self.spy_belief + trust_factor * accusation
        
        # everybody claims that they are NOT the spy, even the spy themselves
        new_spy_belief[self.player_id] = 0.0

        # normalizes
        self.spy_belief = new_spy_belief / np.sum(new_spy_belief)
        
        # Add to history (now includes accusation)
        self.game_history.append((speaker_id, claim, accusation))
    
    def generate_claim(self, public_location_belief: np.ndarray,
                      public_spy_belief: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a claim using Flat Hide-Reveal strategy and return accusation."""
        # γ = β((1-λ)1 + λP) - This remains the same
        gamma = self.beta * ((1 - self.lambda_val) * np.ones(self.num_locations) + 
                            self.lambda_val * self.location_belief)
        
        # Draw from Dirichlet distribution for location claim
        gamma_tensor = torch.tensor(gamma, dtype=torch.float32) + 1e-6 # make all entries > 0
        claim = dist.Dirichlet(gamma_tensor).sample().numpy()
        
        # Simply return the player's current spy belief as the accusation
        return claim, self.spy_belief


# Bayesian Player using Dynamic Hide-Reveal strategy
class BayesianDynamicHideRevealPlayer(Player):
    def __init__(self, player_id: int, num_players: int, num_locations: int, 
                 is_spy: bool, knows_location: Optional[int] = None,
                 initial_beta: float = 5.0, initial_lambda: float = 0.7):
        super().__init__(player_id, num_players, num_locations, is_spy, knows_location)
        self.beta = initial_beta  # Initial concentration parameter
        self.lambda_val = initial_lambda  # Initial truthfulness parameter
    
    def update_beliefs(self, speaker_id: int, claim: np.ndarray, accusation: np.ndarray,
                      public_location_belief: np.ndarray, 
                      public_spy_belief: np.ndarray):
        """Update beliefs using Bayesian update with both claim and accusation."""
        # Don't update our own beliefs based on our own claims
        if speaker_id == self.player_id:
            return
        
        if self.is_spy:
            # Spy updates location belief
            # Simple implementation: update proportionally to the claim
            new_belief = self.location_belief * (claim + 0.2)  # Adding noise to prevent convergence too fast
            self.location_belief = new_belief / np.sum(new_belief)
        
        # Everyone updates spy belief based on direct accusation
        # Trust accusation based on how much we trust the speaker
        speaker_trustworthiness = 1.0 - self.spy_belief[speaker_id]
        trust_factor = 0.5 * speaker_trustworthiness  # Scale trust by suspicion
        
        # Blend our current spy belief with the accusation
        new_spy_belief = (1 - trust_factor) * self.spy_belief + trust_factor * accusation
        
        if not self.is_spy:
            # Non-spy knows they're not the spy
            new_spy_belief[self.player_id] = 0.0
        
        if np.sum(new_spy_belief) > 0:
            new_spy_belief = new_spy_belief / np.sum(new_spy_belief)
        
        self.spy_belief = new_spy_belief
        
        # Add to history (now includes accusation)
        self.game_history.append((speaker_id, claim, accusation))
    
    def generate_claim(self, public_location_belief: np.ndarray,
                      public_spy_belief: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a claim using Dynamic Hide-Reveal strategy and return accusation."""
        # Update λ based on public spy belief
        # If player is top suspect, increase λ to be more truthful
        if np.argmax(public_spy_belief) == self.player_id:
            self.lambda_val = min(1.0, self.lambda_val + 0.2)
        
        # Update β based on public location belief
        # If public belief is concentrated near true location, lower β to obscure
        if not self.is_spy:
            entropy = -np.sum(public_location_belief * np.log(public_location_belief + 1e-10))
            max_entropy = np.log(self.num_locations)
            normalized_entropy = entropy / max_entropy
            
            # Lower beta when public belief is close to truth (low entropy)
            self.beta = max(1.0, self.beta * (0.5 + 0.5 * normalized_entropy))
        
        # γ = β((1-λ)1 + λP)
        gamma = self.beta * ((1 - self.lambda_val) * np.ones(self.num_locations) + 
                            self.lambda_val * self.location_belief)
        
        # Draw from Dirichlet distribution for location claim
        gamma_tensor = torch.tensor(gamma, dtype=torch.float32) + 1e-6 # make all entries > 0
        claim = dist.Dirichlet(gamma_tensor).sample().numpy()
        
        # Return the player's current spy belief as accusation
        return claim, self.spy_belief


# Game class
class SpyfallGameImproved:
    def __init__(self, num_players: int, num_locations: int, 
                 player_types: List[str] = None, 
                 max_turns: int = None,
                 location_names: List[str] = None):
        """
        Initialize a Spyfall game.
        
        Args:
            num_players: Number of players (s)
            num_locations: Number of possible locations (n)
            player_types: List of player types for each player
            max_turns: Maximum number of turns before automatic end-game
            location_names: Optional list of location names for visualization
        """
        self.s = num_players
        self.n = num_locations
        self.max_turns = max_turns if max_turns is not None else 5 * num_locations
        
        # Initialize player types
        if player_types is None:
            # Default: all players use Dynamic Hide-Reveal
            self.player_types = ["dynamic_hide_reveal"] * self.s
        else:
            assert len(player_types) == self.s
            self.player_types = player_types
            
        # Location names for visualization
        if location_names is None:
            self.location_names = [f"Location {i}" for i in range(num_locations)]
        else:
            assert len(location_names) == num_locations
            self.location_names = location_names
            
        # Game state variables
        self.true_location = None  # N
        self.spy = None  # S
        self.game_state = []  # G_t: history of (j_t, C_t, A_t)
        self.current_turn = 0
        
        # Player objects
        self.players = []
        
        # Public beliefs
        self.public_location_belief = None  # P_t
        self.public_spy_belief = None  # S_t
        
        # Marginal public beliefs (excluding each player's contributions)
        self.marginal_location_beliefs = None  # List of P_t for each excluded player
        self.marginal_spy_beliefs = None  # List of S_t for each excluded player
        
        # Game ongoing flag
        self.game_over = False
        self.winner = None
        
        # Statistics
        self.turn_statistics = []  # List of dictionaries with statistics for each turn
        self.player_statistics = {}  # Dictionary with statistics for each player
    
    def setup_game(self):
        """Set up a new game by drawing location, spy, and initializing players."""
        # Draw true location and spy uniformly
        self.true_location = np.random.randint(0, self.n)  # N ~ Unif(1, ..., n)
        self.spy = np.random.randint(0, self.s)  # S ~ Unif(1, ..., s)
        
        # Reset game state
        self.game_state = []
        self.current_turn = 0
        self.game_over = False
        self.winner = None
        
        # Initialize public beliefs
        self.public_location_belief = np.ones(self.n) / self.n
        self.public_spy_belief = np.ones(self.s) / self.s
        
        # Initialize marginal public beliefs (excluding each player)
        self.marginal_location_beliefs = [np.ones(self.n) / self.n for _ in range(self.s)]
        self.marginal_spy_beliefs = [np.ones(self.s) / self.s for _ in range(self.s)]
        
        # Initialize players
        self.players = []
        
        for i in range(self.s):
            is_spy = (i == self.spy)
            knows_location = self.true_location if not is_spy else None
            
            if self.player_types[i] == "flat_hide_reveal":
                player = BayesianFlatHideRevealPlayer(
                    player_id=i,
                    num_players=self.s,
                    num_locations=self.n,
                    is_spy=is_spy,
                    knows_location=knows_location,
                    beta=5.0,
                    lambda_val=0.7
                )
            elif self.player_types[i] == "dynamic_hide_reveal":
                player = BayesianDynamicHideRevealPlayer(
                    player_id=i,
                    num_players=self.s,
                    num_locations=self.n,
                    is_spy=is_spy,
                    knows_location=knows_location,
                    initial_beta=5.0,
                    initial_lambda=0.7
                )
            else:
                raise ValueError(f"Unknown player type: {self.player_types[i]}")
            
            self.players.append(player)
            
        # Reset statistics
        self.turn_statistics = []
        self.player_statistics = {i: {
            "turn_gains": [], 
            "aggregate_gains": [], 
            "location_gain_contribution": [],
            "spy_gain_contribution": [],
            "accusation_accuracy": []  # New statistic
        } for i in range(self.s)}
    
    def update_public_beliefs(self, speaker_id: int, claim: np.ndarray, accusation: np.ndarray):
        """
        Update public beliefs based on a player's claim and accusation.
        
        Args:
            speaker_id: Player who made the claim
            claim: The probability distribution claim over locations
            accusation: The probability distribution over players being the spy
        """
        # ----- Update location public beliefs -----
        # For each location l, compute P(C_t|N=l,j_t)
        loc_likelihoods = np.zeros(self.n)
        for loc in range(self.n):
            # Consider possibilities (player is spy or not)
            p_player_is_spy = self.public_spy_belief[speaker_id]
            
            if p_player_is_spy < 1.0:  # If there's some chance player is not spy
                expected_claim_if_not_spy = np.zeros(self.n)
                expected_claim_if_not_spy[loc] = 1.0  # Non-spy knows location
                kl_div = kl_divergence(expected_claim_if_not_spy, claim)
                likelihood_if_not_spy = np.exp(-kl_div)
            else:
                likelihood_if_not_spy = 0.0
                
            # If player is spy, claim is more uniform or random
            likelihood_if_spy = 1.0  # Simplistic model
            
            # Combine likelihoods weighted by spy/not-spy probability
            loc_likelihoods[loc] = p_player_is_spy * likelihood_if_spy + (1 - p_player_is_spy) * likelihood_if_not_spy
        
        # Normalize likelihoods
        loc_likelihoods = loc_likelihoods + 1e-10
        loc_likelihoods = loc_likelihoods / np.sum(loc_likelihoods)
        
        # Bayes update for location belief
        new_loc_belief = self.public_location_belief * loc_likelihoods
        self.public_location_belief = new_loc_belief / np.sum(new_loc_belief)
        
        # ----- Update spy public beliefs -----
        # Improved spy belief update logic
        
        # 1. Behavioral analysis: How suspicious is this claim?
        spy_likelihoods = np.ones(self.s)  # Default neutral for non-speakers
        
        # Calculate how much speaker's claim diverges from public expectation
        divergence = kl_divergence(self.public_location_belief, claim)
        
        # Convert divergence to suspiciousness score
        suspiciousness = 1.0 / (1.0 + np.exp(-divergence + 2.0))  # Offset for better scaling
        
        # Only update suspicion for the current speaker
        spy_likelihoods[speaker_id] = suspiciousness
        
        # Bayes update based on behavior
        behavior_belief = self.public_spy_belief * spy_likelihoods
        if np.sum(behavior_belief) > 0:
            behavior_belief = behavior_belief / np.sum(behavior_belief)
        
        # 2. Incorporate direct accusation
        accusation_weight = 0.3  # How much to weigh the accusation vs behavioral analysis
        
        # Combine behavior-based belief with direct accusation
        combined_belief = (1 - accusation_weight) * behavior_belief + accusation_weight * accusation
        
        # Normalize final belief
        if np.sum(combined_belief) > 0:
            self.public_spy_belief = combined_belief / np.sum(combined_belief)
        
        # ----- Update marginal public beliefs -----
        # Update marginal beliefs for each player
        for excluded_player in range(self.s):
            # Skip updating the marginal belief that excludes the current speaker
            if excluded_player == speaker_id:
                continue
                
            # Update marginal location belief
            new_loc_belief = self.marginal_location_beliefs[excluded_player] * loc_likelihoods
            self.marginal_location_beliefs[excluded_player] = new_loc_belief / np.sum(new_loc_belief)
            
            # Update marginal spy belief - similar logic as above but for marginal beliefs
            marg_spy_likelihoods = np.ones(self.s)
            marg_spy_likelihoods[speaker_id] = suspiciousness
            
            marg_behavior_belief = self.marginal_spy_beliefs[excluded_player] * marg_spy_likelihoods
            if np.sum(marg_behavior_belief) > 0:
                marg_behavior_belief = marg_behavior_belief / np.sum(marg_behavior_belief)
            
            marg_combined_belief = (1 - accusation_weight) * marg_behavior_belief + accusation_weight * accusation
            
            if np.sum(marg_combined_belief) > 0:
                self.marginal_spy_beliefs[excluded_player] = marg_combined_belief / np.sum(marg_combined_belief)
    
    def play_turn(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """Play a single turn of the game."""
        if self.game_over:
            return -1, None, None
        
        # Save the previous beliefs for information gain calculation
        prev_public_location_belief = self.public_location_belief.copy()
        prev_public_spy_belief = self.public_spy_belief.copy()
        prev_marginal_location_beliefs = [b.copy() for b in self.marginal_location_beliefs]
        prev_marginal_spy_beliefs = [b.copy() for b in self.marginal_spy_beliefs]
        
        # Select current player (round-robin)
        player_idx = self.current_turn % self.s
        
        # Generate claim and accusation from player
        claim, accusation = self.players[player_idx].generate_claim(
            self.public_location_belief, 
            self.public_spy_belief
        )
        
        # Update game state
        self.game_state.append((player_idx, claim, accusation))
        
        # Update public beliefs
        self.update_public_beliefs(player_idx, claim, accusation)
        
        # Update all players' private beliefs
        for i, player in enumerate(self.players):
            if i != player_idx:  # Skip the player who made the claim
                player.update_beliefs(
                    player_idx, 
                    claim, 
                    accusation,
                    self.public_location_belief, 
                    self.public_spy_belief
                )
        
        # Collect turn statistics including accusation metrics
        self.collect_turn_statistics(
            player_idx, 
            claim,
            accusation,
            prev_public_location_belief, 
            prev_public_spy_belief,
            prev_marginal_location_beliefs,
            prev_marginal_spy_beliefs
        )
        
        # Increment turn counter
        self.current_turn += 1
        
        # Check for end-game condition
        self.check_end_game()
        
        return player_idx, claim, accusation
    
    def check_end_game(self) -> bool:
        """
        Check if any player has reached end-game condition.
        """
        max_turns_reached = self.current_turn >= self.max_turns
        
        # Spy is confident about the location
        spy_found_location = np.max(self.players[self.spy].location_belief) > 0.7
        
        # critical mass is reached: everybody's top suspect is the same guy except for one
        # assume that the spy would never vote for themselves, this is the same as polling everyone else's votes
        guesses = [np.argmax(p.spy_belief) for idx, p in enumerate(self.players) if idx != self.spy]
        voted_spy, vote_count = Counter(guesses).most_common(1)[0]
        all_same_vote = vote_count == len(self.players) - 1
        

        if max_turns_reached or spy_found_location or all_same_vote:   
            # Spy guesses the location
            spy_guess = np.argmax(self.players[self.spy].location_belief)
            if voted_spy == self.spy: 
                if spy_guess == self.true_location:
                    self.winner = "tie"
                else: 
                    self.winner = "non-spy"
            else:
                self.winner = "spy"
            
            self.game_over = True
            return True
        
        return False
    
    def collect_turn_statistics(self, player_idx: int, claim: np.ndarray, accusation: np.ndarray,
                             prev_public_location_belief: np.ndarray,
                             prev_public_spy_belief: np.ndarray,
                             prev_marginal_location_beliefs: List[np.ndarray],
                             prev_marginal_spy_beliefs: List[np.ndarray]):
        """
        Collect statistics for this turn, including accusation metrics.
        
        Args:
            player_idx: The index of the player who made the claim
            claim: The location claim made
            accusation: The spy accusation made
            prev_public_location_belief: Public location belief before the claim
            prev_public_spy_belief: Public spy belief before the claim
            prev_marginal_location_beliefs: Marginal location beliefs before the claim
            prev_marginal_spy_beliefs: Marginal spy beliefs before the claim
        """
        # Calculate turn-wise information gain for location
        location_gain = self.calculate_information_gain(
            prev_public_location_belief, 
            self.public_location_belief
        )
        
        # Calculate turn-wise information gain for spy
        spy_gain = self.calculate_information_gain(
            prev_public_spy_belief,
            self.public_spy_belief
        )
        
        # Calculate accusation accuracy (how close to 1.0 at true spy position)
        # Higher is better - perfect accusation would be 1.0 at spy position
        accusation_accuracy = accusation[self.spy]
        self.player_statistics[player_idx]["accusation_accuracy"].append(accusation_accuracy)
        
        # Calculate contribution of each player to the public belief
        for i in range(self.s):
            # The contribution is the difference between the public belief and the marginal belief excluding player i
            loc_contribution = self.calculate_information_gain(
                self.marginal_location_beliefs[i],
                self.public_location_belief
            )
            
            spy_contribution = self.calculate_information_gain(
                self.marginal_spy_beliefs[i],
                self.public_spy_belief
            )
            
            # Update player statistics with their current contribution
            self.player_statistics[i]["location_gain_contribution"].append(loc_contribution)
            self.player_statistics[i]["spy_gain_contribution"].append(spy_contribution)
            self.player_statistics[i]["aggregate_gains"].append(loc_contribution + spy_contribution)
        
        # Store turn-wise gain for the active player
        total_gain = location_gain + spy_gain
        self.player_statistics[player_idx]["turn_gains"].append(total_gain)
        
        # Store statistics
        turn_stats = {
            "turn": self.current_turn,
            "player": player_idx,
            "location_gain": location_gain,
            "spy_gain": spy_gain,
            "total_gain": total_gain,
            "accusation_accuracy": accusation_accuracy,
            "accusation_entropy": -np.sum(accusation * np.log(accusation + 1e-10)),
            "player_contributions": {
                i: self.player_statistics[i]["aggregate_gains"][-1] for i in range(self.s)
            }
        }
        
        self.turn_statistics.append(turn_stats)
    
    def calculate_aggregate_gains(self):
        """
        Calculate aggregate information gain for each player.
        This compares public beliefs with vs. without each player's contributions.
        Updated to include accusation data.
        """
        # For each player, recalculate public beliefs excluding their claims
        for player_idx in range(self.s):
            # Start with uniform prior
            location_belief_without_player = np.ones(self.n) / self.n
            spy_belief_without_player = np.ones(self.s) / self.s
            
            # Replay the game without this player's claims and accusations
            for t, (j, c, a) in enumerate(self.game_state):
                if j != player_idx:  # Skip this player's claims
                    # Basic public belief update (simplified)
                    if j != self.spy:  # If speaker is not spy
                        # Non-spy's claim should be concentrated on true location
                        location_belief_without_player = location_belief_without_player * c
                        if np.sum(location_belief_without_player) > 0:
                            location_belief_without_player = location_belief_without_player / np.sum(location_belief_without_player)
                    
                    # Update spy belief using accusation data
                    accusation_weight = 0.3
                    spy_belief_without_player = (1 - accusation_weight) * spy_belief_without_player + accusation_weight * a
                    spy_belief_without_player = spy_belief_without_player / np.sum(spy_belief_without_player)
            
            # Calculate aggregate gain
            location_gain = self.calculate_information_gain(
                location_belief_without_player,
                self.public_location_belief
            )
            
            spy_gain = self.calculate_information_gain(
                spy_belief_without_player,
                self.public_spy_belief
            )
            
            self.player_statistics[player_idx]["aggregate_gain"] = location_gain + spy_gain
    
    def play_game(self) -> str:
        """Play an entire game and return the winner."""
        self.setup_game()
        
        while not self.game_over:
            self.play_turn()
        
        # Calculate aggregate gains at the end of the game
        self.calculate_aggregate_gains()
        
        return self.winner
    
    def calculate_information_gain(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence between two distributions."""
        return kl_divergence(p, q)
    
    def get_turn_wise_gains(self, player_idx: int) -> List[float]:
        """Calculate turn-wise information gains for a player."""
        gains = []
        player = self.players[player_idx]
        
        for t, (j, c, _) in enumerate(player.game_history):
            if j == player_idx:
                # Compare player's belief before and after their claim
                prev_belief = player.game_history[t-1][1] if t > 0 else np.ones(self.n) / self.n
                gain = self.calculate_information_gain(prev_belief, c)
                gains.append(gain)
        
        return gains
    
    def plot_beliefs(self):
        """Plot the current beliefs of players and public."""
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot location beliefs
        ax = axs[0]
        ax.set_title("Location Beliefs")
        
        # Plot public belief
        ax.bar(range(self.n), self.public_location_belief, alpha=0.3, label="Public")
        
        # Plot spy's belief
        ax.bar(range(self.n), self.players[self.spy].location_belief, alpha=0.5, label=f"Spy (Player {self.spy})")
        
        # Mark true location
        ax.axvline(x=self.true_location, color='r', linestyle='--', label="True Location")
        
        ax.set_xlabel("Location")
        ax.set_ylabel("Probability")
        ax.legend()
        
        # Plot spy beliefs
        ax = axs[1]
        ax.set_title("Spy Beliefs & Accusations")
        
        # Plot public spy belief
        ax.bar(range(self.s), self.public_spy_belief, alpha=0.3, label="Public Spy Belief")
        
        # Plot individual player accusations
        for i, player in enumerate(self.players):
            if i != self.spy:  # Only show non-spy accusations
                if len(self.game_state) > 0:
                    # Find the most recent accusation by this player
                    recent_accusation = None
                    for j, _, a in reversed(self.game_state):
                        if j == i:
                            recent_accusation = a
                            break
                    
                    if recent_accusation is not None:
                        ax.plot(range(self.s), recent_accusation, 'o-', alpha=0.5, 
                                label=f"Player {i} Accusation")
        
        # Mark true spy
        ax.axvline(x=self.spy, color='r', linestyle='--', label="True Spy")
        
        ax.set_xlabel("Player Number")
        ax.set_ylabel("Probability / Accusation Strength")
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def run_simulation(self, num_games: int = 100) -> Dict:
        """Run multiple games and collect statistics."""
        stats = {
            "spy_wins": 0,
            "non_spy_wins": 0,
            "ties": 0,
            "avg_turns": 0,
            "avg_turnwise_ig_by_player": np.zeros(self.s),
            "avg_aggregate_ig_by_player": np.zeros(self.s),
            "avg_loc_contribution_by_player": np.zeros(self.s),
            "avg_spy_contribution_by_player": np.zeros(self.s),
            "avg_accusation_accuracy": np.zeros(self.s)  # New statistic
        }
        
        for _ in tqdm(range(num_games), desc="Running games"):
            winner = self.play_game()
            
            if winner == "spy":
                stats["spy_wins"] += 1
            elif winner == "non-spy":
                stats["non_spy_wins"] += 1
            else: 
                stats["ties"] += 1
            
            stats["avg_turns"] += self.current_turn
            
            # Collect information gain statistics
            for i in range(self.s):
                # Add turn-wise gains (average of each player's turns)
                if self.player_statistics[i]["turn_gains"]:
                    stats["avg_turnwise_ig_by_player"][i] += np.mean(self.player_statistics[i]["turn_gains"])
                
                # Add final aggregate gain (information contribution)
                if self.player_statistics[i]["aggregate_gains"]:
                    stats["avg_aggregate_ig_by_player"][i] += self.player_statistics[i]["aggregate_gains"][-1]
                
                # Add final location & spy contributions
                if self.player_statistics[i]["location_gain_contribution"]:
                    stats["avg_loc_contribution_by_player"][i] += self.player_statistics[i]["location_gain_contribution"][-1]
                
                if self.player_statistics[i]["spy_gain_contribution"]:
                    stats["avg_spy_contribution_by_player"][i] += self.player_statistics[i]["spy_gain_contribution"][-1]
                
                # Add accusation accuracy
                if self.player_statistics[i]["accusation_accuracy"]:
                    stats["avg_accusation_accuracy"][i] += np.mean(self.player_statistics[i]["accusation_accuracy"])
        
        # Calculate averages
        stats["spy_win_rate"] = stats["spy_wins"] / num_games
        stats["non_spy_win_rate"] = stats["non_spy_wins"] / num_games
        stats["tie_rate"] = stats["ties"] / num_games
        stats["avg_turns"] /= num_games
        stats["avg_turnwise_ig_by_player"] /= num_games
        stats["avg_aggregate_ig_by_player"] /= num_games
        stats["avg_loc_contribution_by_player"] /= num_games
        stats["avg_spy_contribution_by_player"] /= num_games
        stats["avg_accusation_accuracy"] /= num_games
        
        return stats


# Run a comparative analysis
def run_strategy_comparison(num_players: int, num_locations: int, num_games: int = 100):
    """Compare different player strategies."""
    # Test different strategy combinations
    strategy_combinations = [
        ["flat_hide_reveal"] * num_players,
        ["dynamic_hide_reveal"] * num_players,
        ["flat_hide_reveal"] * (num_players // 2) + ["dynamic_hide_reveal"] * (num_players - num_players // 2)
    ]
    
    results = []
    
    for strategies in strategy_combinations:
        game = SpyfallGameImproved(num_players, num_locations, strategies)
        stats = game.run_simulation(num_games)
        
        strategy_name = "All Flat" if all(s == "flat_hide_reveal" for s in strategies) else \
                        "All Dynamic" if all(s == "dynamic_hide_reveal" for s in strategies) else \
                        "Mixed Strategies"
        
        results.append({
            "strategy": strategy_name,
            "spy_win_rate": stats["spy_win_rate"],
            "non_spy_win_rate": stats["non_spy_win_rate"],
            "tie_rate": stats["tie_rate"],
            "avg_turns": stats["avg_turns"],
            "avg_ig_by_player": stats["avg_aggregate_ig_by_player"].tolist(),
            "avg_accusation_accuracy": stats["avg_accusation_accuracy"].tolist()  # New metric
        })
    
    return results


# Example usage
if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    pyro.set_rng_seed(42)
    
    # Create a game with 5 players and 10 locations
    game = SpyfallGameImproved(num_players=5, num_locations=10)
    
    # Play a single game
    game.setup_game()
    print(f"True location: {game.true_location}, Spy: Player {game.spy}")
    
    # Play at most 10 turns
    for _ in range(10):
        player, claim, accusation = game.play_turn()
        if game.game_over:
            break
        print(f"Turn {game.current_turn-1}: Player {player} claimed {np.round(claim, 2)}")
        print(f"                     and accused {np.round(accusation, 2)}")
    
    print(f"Game over! Winner: {game.winner}")
    
    # Plot final beliefs
    fig = game.plot_beliefs()
    plt.savefig("plot_beliefs.png")
    
    # Run comparative analysis
    results = run_strategy_comparison(5, 10, 20)
    print("Strategy comparison results:")
    for result in results:
        print(f"{result['strategy']}:")
        print(f"  Spy win rate: {result['spy_win_rate']:.2f}")
        print(f"  Non-spy win rate: {result['non_spy_win_rate']:.2f}")
        print(f"  Tie rate: {result['tie_rate']:.2f}")
        print(f"  Average turns: {result['avg_turns']:.2f}")
        print(f"  Average information gain: {np.mean(result['avg_ig_by_player']):.4f}")
        print(f"  Average accusation accuracy: {np.mean(result['avg_accusation_accuracy']):.4f}")
        print()