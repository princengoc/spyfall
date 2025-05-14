# spyfall_improved.py

import numpy as np
from player import EmissionPlayer
from inference import infer_public, infer_leave_one_out


class SpyfallGameImproved:
    def __init__(
        self,
        players: list[EmissionPlayer],
        num_samples: int = 1000,
    ):
        self.players = players
        self.num_players = len(players)
        self.num_locations = players[0].num_locations
        self.history: list[tuple[int, np.ndarray, int]] = []
        # initialize uniform public priors
        self.public_location_belief = np.ones(self.num_locations) / self.num_locations
        self.public_spy_belief = np.ones(self.num_players) / self.num_players
        self.num_samples = num_samples

    def play_turn(self, speaker: int):
        """Simulate one turn: speaker makes a claim & an accusation, then update beliefs."""
        claim = self.players[speaker].generate_claim()
        accusation = self.players[speaker].generate_accusation()
        # record the observation
        self.history.append((speaker, claim, accusation))

        # update public beliefs via Pyro
        loc_pub, spy_pub = infer_public(
            self.history, num_samples=self.num_samples
        )
        self.public_location_belief = loc_pub
        self.public_spy_belief = spy_pub

        # each player updates their private beliefs
        for player in self.players:
            player.infer_private_beliefs(
                self.history, num_samples=self.num_samples
            )

        return speaker, claim, accusation

    def compute_leave_one_out(self, exclude_index: int):
        """
        Public posterior leaving out all observations
        by player `exclude_index`.
        """
        loc_loo, spy_loo = infer_leave_one_out(
            self.history, exclude_index, num_samples=self.num_samples
        )
        return loc_loo, spy_loo


def make_game(
    num_players: int,
    num_locations: int,
    beta: float,
    lambda_val: float,
    true_location: int,
    spy_index: int,
    num_samples: int = 1000,
) -> SpyfallGameImproved:
    """Helper to construct a game with EmissionPlayer instances."""
    players = [
        EmissionPlayer(
            index=i,
            is_spy=(i == spy_index),
            true_location=true_location,
            beta=beta,
            lambda_val=lambda_val,
            num_locations=num_locations,
            num_players=num_players,
        )
        for i in range(num_players)
    ]
    return SpyfallGameImproved(players, num_samples=num_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate Spyfall with Pyro-based Bayesian inference"
    )
    parser.add_argument("--num_players", type=int, default=5)
    parser.add_argument("--num_locations", type=int, default=10)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda_val", type=float, default=0.5)
    parser.add_argument("--true_location", type=int, default=0)
    parser.add_argument("--spy_index", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--turns", type=int, default=10)
    args = parser.parse_args()

    game = make_game(
        num_players=args.num_players,
        num_locations=args.num_locations,
        beta=args.beta,
        lambda_val=args.lambda_val,
        true_location=args.true_location,
        spy_index=args.spy_index,
        num_samples=args.num_samples,
    )

    for t in range(args.turns):
        speaker_idx, claim, accusation = game.play_turn(t % args.num_players)
        print(
            f"Turn {t+1:2d} | Speaker {speaker_idx} | "
            f"Claim {np.round(claim, 3)} | Accuse {accusation}"
        )
        print("  Public loc belief:", np.round(game.public_location_belief, 3))
        print("  Public spy belief:", np.round(game.public_spy_belief, 3))
        print()
