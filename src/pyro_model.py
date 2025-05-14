import numpy as np
import torch
import pyro
import pyro.distributions as dist
from emissions import get_claim_distribution, get_accusation_distribution
from typing import List, Tuple


def model(game_history: List[Tuple[int, np.ndarray, int]],
          num_locations: int,
          num_players: int,
          beta: float,
          lambda_val: float):
    """
    Probabilistic model for Spyfall game history.

    Args:
        game_history: list of triples (speaker_idx, claim_vec, acc_idx)
        num_locations: total number of possible locations
        num_players: total number of players
        beta: concentration parameter (>0)
        lambda_val: truthfulness weight in [0,1]
    """
    # Prior over true location and spy identity
    loc_probs = torch.ones(num_locations) / num_locations
    spy_probs = torch.ones(num_players) / num_players
    true_loc = pyro.sample("true_loc", dist.Categorical(loc_probs))
    true_spy = pyro.sample("true_spy", dist.Categorical(spy_probs))

    # Track public belief over locations (uniform initial)
    public_loc_belief = torch.ones(num_locations) / num_locations

    for t, (speaker_idx, claim_vec, acc_idx) in enumerate(game_history):
        # Determine speaker's internal location belief
        if speaker_idx == true_spy:
            speaker_loc_belief = public_loc_belief
        else:
            # honest player knows true_loc
            speaker_loc_belief = torch.nn.functional.one_hot(true_loc, num_classes=num_locations).float()

        # Claim generation
        claim_dist = get_claim_distribution(speaker_loc_belief.numpy(),
                                            beta=beta,
                                            lambda_val=lambda_val)
        pyro.sample(f"claim_{t}", claim_dist, obs=torch.tensor(claim_vec, dtype=torch.float32))

        # Accusation generation
        # For simplicity, placeholder uses uniform; will replace with proper posterior
        acc_dist = get_accusation_distribution(np.ones(num_players)/num_players)
        pyro.sample(f"acc_{t}", acc_dist, obs=torch.tensor(acc_idx, dtype=torch.long))

        # PUBLIC belief update to be handled via separate inference wrapper (not in model)
        # TODO: remove manual updates and rely on PPL inference outputs

def guide(game_history: List[Tuple[int, np.ndarray, int]],
          num_locations: int,
          num_players: int,
          beta: float,
          lambda_val: float):
    """
    Mean-field guide for Spyfall latent variables.

    Defines independent variational Categorical distributions over true_loc and true_spy.
    """
    q_loc = pyro.param("q_loc", torch.ones(num_locations) / num_locations,
                       constraint=dist.constraints.simplex)
    q_spy = pyro.param("q_spy", torch.ones(num_players) / num_players,
                       constraint=dist.constraints.simplex)
    pyro.sample("true_loc", dist.Categorical(q_loc))
    pyro.sample("true_spy", dist.Categorical(q_spy))

