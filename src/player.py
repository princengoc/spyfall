# player.py
import abc
from typing import List, Tuple
import numpy as np
import torch
import pyro
from pyro.infer import Predictive
from emissions import get_claim_distribution, get_accusation_distribution
from pyro_model import model, guide


class Player(abc.ABC):
    """
    Abstract player interface:
      - holds private info (is_spy, true_location)
      - stores beliefs (location_belief, spy_belief)
      - defines abstract methods for emissions and private inference
    """

    def __init__(
        self,
        index: int,
        is_spy: bool,
        true_location: int,
        num_locations: int,
        num_players: int,
    ):
        self.index = index
        self.is_spy = is_spy
        self.true_location = true_location
        self.num_locations = num_locations
        self.num_players = num_players

        # initialize uniform priors
        self.location_belief = np.ones(num_locations) / num_locations
        self.spy_belief = np.ones(num_players) / num_players

    @abc.abstractmethod
    def generate_claim(self) -> np.ndarray:
        """Sample a location claim distribution."""
        ...

    @abc.abstractmethod
    def generate_accusation(self) -> int:
        """Sample an accusation (player index)."""
        ...

    @abc.abstractmethod
    def infer_private_beliefs(
        self,
        history: List[Tuple[int, np.ndarray, int]],
        num_samples: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Bayesian inference given this player's private information.
        Returns updated (location_belief, spy_belief).
        """
        ...


class EmissionPlayer(Player):
    """
    Concrete player using Dirichlet/Categorical emissions.
    Implements private inference by conditioning the PPL model.
    """

    def __init__(
        self,
        index: int,
        is_spy: bool,
        true_location: int,
        beta: float,
        lambda_val: float,
        num_locations: int,
        num_players: int,
    ):
        super().__init__(index, is_spy, true_location, num_locations, num_players)
        self.beta = beta
        self.lambda_val = lambda_val

    def generate_claim(self) -> np.ndarray:
        """Sample claim via shared emission routine."""
        dist = get_claim_distribution(
            self.location_belief,
            beta=self.beta,
            lambda_val=self.lambda_val,
        )
        return dist.sample().numpy()

    def generate_accusation(self) -> int:
        """Sample accusation via shared emission routine."""
        # FIXME: accusation should be an np.ndarray, ie return the catgories' probabilities rather than a single sample
        # should just return self.spy_belief if not a spy, otherwise return a public accusation belief conditioned on 
        # you are not a spy. 
        dist = get_accusation_distribution(self.spy_belief)
        return dist.sample().item()

    def infer_private_beliefs(
        self,
        history: List[Tuple[int, np.ndarray, int]],
        num_samples: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infer posterior beliefs conditioned on this player's private info:
          - If spy: condition on true_spy == self.index
          - Else:    condition on true_location == self.true_location
        """
        # Build conditioning dict
        condition_data = {}
        if self.is_spy:
            condition_data["true_spy"] = torch.tensor(
                self.index, dtype=torch.int64
            )
        else:
            condition_data["true_location"] = torch.tensor(
                self.true_location, dtype=torch.int64
            )

        # FIXME: if spy, should marginalize out own's claims and accusations when updating beliefs

        # Condition the global model
        conditioned_model = pyro.poutine.condition(model, data=condition_data)

        # Run predictive sampling
        predictive = Predictive(
            conditioned_model,
            guide=guide,
            num_samples=num_samples,
            return_sites=["true_location", "true_spy"],
        )
        samples = predictive(history)

        # Compute empirical distributions
        loc_samples = samples["true_location"].numpy()
        spy_samples = samples["true_spy"].numpy()

        loc_bel = np.bincount(loc_samples, minlength=self.num_locations) / num_samples
        spy_bel = np.bincount(spy_samples, minlength=self.num_players) / num_samples

        # Update internal beliefs
        self.location_belief = loc_bel
        self.spy_belief = spy_bel

        return loc_bel, spy_bel
