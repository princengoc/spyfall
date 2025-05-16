import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SMCFilter, EmpiricalMarginal


def spyfall_dbn_model(C_obs=None, n_players = 6):
    """
    Dynamic Bayesian network for Spyfall:
    - C_obs: optional tensor of shape [T, n_locations]
    """
    T = C_obs.size(0)
    j_ts = torch.arange(0, T)
    n_locations = C_obs.size(1)

    # Priors
    probs_N = torch.ones(n_locations) / float(n_locations)
    probs_S = torch.ones(n_players)   / float(n_players)
    N = pyro.sample("N", dist.Categorical(probs_N))
    S = pyro.sample("S", dist.Categorical(probs_S))

    # Learnable Dirichlet concentration parameters
    alpha_loc_nonspy = pyro.param("alpha_loc_nonspy", torch.ones(n_locations), constraint=dist.constraints.positive)
    alpha_loc_spy    = pyro.param("alpha_loc_spy",    torch.ones(n_locations), constraint=dist.constraints.positive)

    # Plate over time steps
    with pyro.plate("time", T) as t:
        j = j_ts[t]
        # indicator: True if spy speaks this turn
        is_spy = (j == S)

        # Choose Dirichlet parameters based on speaker
        loc_alpha = torch.where(
            is_spy.unsqueeze(-1),
            alpha_loc_spy,
            alpha_loc_nonspy * torch.nn.functional.one_hot(N, n_locations).float()
        )

        # Observe or sample actions
        pyro.sample(
            "C_{}".format(t),
            dist.Dirichlet(loc_alpha).to_event(1),
            obs=(C_obs[t] if C_obs is not None else None)
        )

    return N, S


# ---- Inference example using SMC ----
if __name__ == "__main__":
    # Example placeholders
    T = 5
    n_locations = 10
    n_players   = 5
    C_obs = torch.randn(T, n_locations).abs()  # replace with valid probability vectors

    g = pyro.render_model(spyfall_dbn_model, model_args=(C_obs, n_players), render_distributions=True, render_params=True)
    g.render(filename="dbn", format="png", cleanup=True)


    # # Initialize SMC filter
    # smc = SMCFilter(
    #     model=spyfall_dbn_model,
    #     num_particles=200
    # )

    # # Run inference
    # posterior = smc.run(
    #     j_ts,
    #     C_obs=C_obs,
    #     A_obs=A_obs
    # )

    # # Extract marginals
    # marginal_N = EmpiricalMarginal(posterior, sites="N")
    # marginal_S = EmpiricalMarginal(posterior, sites="S")

    # print("Posterior over N:", marginal_N.mean)
    # print("Posterior over S:", marginal_S.mean)
