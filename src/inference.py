import numpy as np
import torch
import pyro
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro_model import model, guide
from typing import Tuple


def infer_public(game_history: list,
                 num_locations: int,
                 num_players: int,
                 beta: float,
                 lambda_val: float,
                 num_samples: int = 1000,
                 svi_steps: int = 200,
                 lr: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute public posterior over location and spy via SVI + sampling.

    Returns:
        P_loc: numpy array [num_locations]
        P_spy: numpy array [num_players]
    """
    pyro.clear_param_store()
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    # Fit variational parameters
    for _ in range(svi_steps):
        svi.step(game_history,
                 num_locations,
                 num_players,
                 beta,
                 lambda_val)

    # Generate posterior samples
    predictive = Predictive(model,
                             guide=guide,
                             num_samples=num_samples,
                             return_sites=["true_loc","true_spy"])
    samples = predictive(game_history,
                         num_locations,
                         num_players,
                         beta,
                         lambda_val)

    loc_counts = torch.bincount(samples["true_loc"], minlength=num_locations)
    P_loc = (loc_counts.float() / loc_counts.sum()).numpy()
    spy_counts = torch.bincount(samples["true_spy"], minlength=num_players)
    P_spy = (spy_counts.float() / spy_counts.sum()).numpy()
    return P_loc, P_spy


def infer_leave_one_out(game_history: list,
                         num_locations: int,
                         num_players: int,
                         beta: float,
                         lambda_val: float,
                         exclude_player: int,
                         num_samples: int = 1000,
                         svi_steps: int = 200,
                         lr: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute public posterior excluding a specific player's observations.
    """
    filtered_history = [obs for obs in game_history if obs[0] != exclude_player]
    return infer_public(filtered_history,
                        num_locations,
                        num_players,
                        beta,
                        lambda_val,
                        num_samples,
                        svi_steps,
                        lr)

