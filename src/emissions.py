import numpy as np
import torch
import pyro.distributions as dist
from typing import Sequence

def get_dirichlet_params(location_belief: np.ndarray, beta: float, lambda_val: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute concentration parameters \gamma for a Dirichlet over locations:
        \gamma = beta * ((1 - lambda) * 1 + lambda * location_belief)

    Args:
        location_belief: numpy array of shape [n] representing prior over locations
        beta: concentration parameter (>0)
        lambda_val: truthfulness weight in [0,1]
        eps: small constant to ensure positivity

    Returns:
        Torch tensor of shape [n] with \gamma_i > 0
    """
    gamma = beta * ((1 - lambda_val) * np.ones_like(location_belief) + lambda_val * location_belief)
    return torch.tensor(gamma, dtype=torch.float32) + eps


def get_claim_distribution(location_belief: np.ndarray, beta: float, lambda_val: float, eps: float = 1e-6) -> dist.Dirichlet:
    """
    Construct a Dirichlet distribution for generating a probabilistic location claim.

    Args:
        location_belief: numpy array [n]
        beta: concentration parameter
        lambda_val: truthfulness weight
        eps: small constant to ensure all parameters > 0

    Returns:
        A Pyro Dirichlet distribution over the locations
    """
    gamma = get_dirichlet_params(location_belief, beta, lambda_val, eps)
    return dist.Dirichlet(gamma)


def get_accusation_distribution(spy_belief: np.ndarray) -> dist.Categorical:
    """
    Construct a Categorical distribution for generating a probabilistic spy accusation.

    Args:
        spy_belief: numpy array [s] representing belief over who the spy is

    Returns:
        A Pyro Categorical distribution over players
    """
    probs = torch.tensor(spy_belief, dtype=torch.float32)
    probs = probs / probs.sum()  # normalize
    return dist.Categorical(probs)

