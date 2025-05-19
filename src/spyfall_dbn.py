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
    loc_alpha = beta * ((1.0 - lam) + lam * one_hot_loc)  # [enum_N, L]

    # spy's location prior distribution, will be updated sequentially
    pi = torch.ones(n_players, n_locations) / n_locations   # shape [P, L]
    eye_p = torch.eye(n_players)                       # [P,P]

    for t in pyro.markov(range(T)):
        claim = C_obs[t]                                # [L]

        # 1) spy‐mask via Vindex
        one_hot_spy = Vindex(eye_p)[S_true]             # [enum_S, 1, P]
        mask        = one_hot_spy[..., player_index[t]] # [enum_S, 1]

        # 2) Bayes‐update spy’s belief pi over locations
        alphas0 = beta * ((1.0 - lam) + lam * torch.eye(L))
        C_rep   = claim.unsqueeze(0).expand(L, -1)      # [L, L]
        # p(C|N_t=i), for i in [L]
        ls      = torch.exp(dist.Dirichlet(alphas0).log_prob(C_rep))  # [L]
        # bring into players’ enum dim:
        ls_b    = ls.unsqueeze(0).expand(n_players, L)  # [enum_S, L]
        # update pi based on mask 
        pi_bayes = pi * ls_b
        pi_bayes = pi_bayes / pi_bayes.sum(dim=-1, keepdim=True)   # [P, L]
        pi = mask * pi + (1.0 - mask) * pi_bayes   # [P, L]

        # 3) mix spy vs. non‐spy concentration
        alpha_ns = loc_alpha.unsqueeze(0)\
                     .expand(n_players, L, L)          # [enum_S, L, L]
        alpha_sp = (beta * ((1.0 - lam) + lam * pi))\
                     .unsqueeze(1)\
                     .expand(n_players, L, L)          # [enum_S, L, L]
        alpha_mx = mask.unsqueeze(-1) * alpha_sp \
                   + (1.0 - mask).unsqueeze(-1) * alpha_ns   # [enum_S, L, L]

        # 4) observe / sample claims
        pyro.sample(f"claims_{t}",
                    dist.Dirichlet(alpha_mx).to_event(2),
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
