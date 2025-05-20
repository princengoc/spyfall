import torch
import pyro
import pyro.distributions as dist
from pyro.ops.indexing import Vindex

def spyfall_dbn_model(C_obs, player_index, n_players=6):
    T, L = C_obs.shape
    
    # latent sites: true location and true spy index
    N_true = pyro.sample("loc_true",
                        dist.Categorical(torch.ones(L)/L), infer={'enumerate': 'parallel'})
    S_true = pyro.sample("spy_true",
                        dist.Categorical(torch.ones(n_players)/n_players), infer={'enumerate': 'parallel'})

    # global "honesty" parameters for agents that governs p(claim | N_t)
    beta = pyro.param("beta", torch.tensor(1.0),
                      constraint=dist.constraints.positive)
    lam  = pyro.param("lambda", torch.tensor(0.5),
                      constraint=dist.constraints.unit_interval)

    one_hot_loc = torch.nn.functional.one_hot(N_true, L).float()
    # concentration parameter for non-spy
    loc_alpha_true = beta * ((1.0 - lam) + lam * one_hot_loc)  # [L]

    # spy's location prior distribution, will be updated sequentially
    pi    = torch.ones(n_players, L) / L
    eye_p = torch.eye(n_players)

    for t in pyro.markov(range(T)):
        claim = C_obs[t]  # shape [L]

        # — build a per‐hypothesis speaker mask: [enum_S, 1]
        mask = (torch.arange(n_players) == player_index[t]).unsqueeze(-1)  # [enum_S,1]

        # — Bayes update of spy belief (batching over P via mask)
        alphas0 = beta * ((1.0 - lam) + lam * torch.eye(L))    # [L,L]
        C_rep   = claim.unsqueeze(0).expand(L, L)              # [L,L]
        ls      = torch.exp(dist.Dirichlet(alphas0).log_prob(C_rep))  # [L]
        ls_b    = ls.unsqueeze(0).expand(n_players, L)        # [enum_S,L]

        pi_bayes = pi * ls_b
        pi_bayes = pi_bayes / pi_bayes.sum(dim=-1, keepdim=True)      # [enum_S,L]

        # — mix “no‐update if spy spoke” vs full Bayes update
        pi = torch.where(mask, pi, pi_bayes)                    # [enum_S,L]

        # non‐spy and spy concentrations [enum_S,L]
        alpha_ns = beta * ((1 - lam) + lam * one_hot_loc)
        alpha_ns = alpha_ns.unsqueeze(0).expand(n_players, L)

        alpha_sp = beta * ((1 - lam) + lam * pi)       # [enum_S,L]

        # choose per-hypothesis
        alpha_mix = torch.where(mask, alpha_sp, alpha_ns)  # [enum_S,L]

        # — observe claims as an L‐vector event
        pyro.sample(f"claims_{t}",
                    dist.Dirichlet(alpha_mix).to_event(1),
                    obs=claim)

    return N_true, S_true


if __name__ == "__main__":
    # Example placeholders
    T, n_locations, n_players = 4, 10, 3
    raw = torch.randn(T, n_locations).abs()
    C_obs  = raw / raw.sum(dim=-1, keepdim=True)
    player_index = torch.tensor([t % n_players for t in range(T)])

    g = pyro.render_model(
        spyfall_dbn_model,
        model_args=(C_obs, player_index, n_players),
        render_distributions=True,
        render_params=True
    )
    g.render(filename="dbn", format="png", cleanup=True)
