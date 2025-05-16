import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SMCFilter, EmpiricalMarginal

def spyfall_dbn_model(C_obs=None, n_players=6):
    T, n_locations = C_obs.shape
    # Priors for game location N and spy index S
    probs_N = torch.ones(n_locations) / n_locations
    probs_S = torch.ones(n_players)   / n_players
    N = pyro.sample("N", dist.Categorical(probs_N))
    S = pyro.sample("S", dist.Categorical(probs_S))

    # Hyperparameters β and λ
    beta   = pyro.param("beta",   torch.tensor(1.0),
                        constraint=dist.constraints.positive)
    lambda_= pyro.param("lambda", torch.tensor(0.5),
                        constraint=dist.constraints.unit_interval)

    # initialize spy’s belief P_spy = uniform
    P_spy = torch.ones(n_locations) / n_locations

    # one-hot for the true location
    one_hot_loc = torch.nn.functional.one_hot(N, n_locations).float()

    # sequential (non-vectorized) loop
    for t in pyro.markov(range(T)):
        # predicate: is current speaker the spy?
        is_spy = (t % n_players == S)

        # choose P_{j_t,t}
        P = P_spy       if is_spy else one_hot_loc

        # compute γ_{j_t,t} = β((1−λ)1 + λ P)
        loc_alpha = beta * ((1 - lambda_) + lambda_ * P)

        # sample or observe the message C_t
        C_t = pyro.sample(f"C_{t}",
                          dist.Dirichlet(loc_alpha),
                          obs=(C_obs[t] if C_obs is not None else None))

        # ---- spy belief update (Bayesian) ----
        # posterior ∝ prior * likelihood(C_t | N=i) for each i
        # likelihood under true-speaker model = Dirichlet(β((1−λ)+λ e_i))
        # evaluate log likelihood for each i:
        all_alphas = beta * ((1 - lambda_) + lambda_ * torch.eye(n_locations))
        log_likes = torch.stack([
            dist.Dirichlet(all_alphas[i]).log_prob(C_t)
            for i in range(n_locations)
        ])
        # Bayes rule (in the generative model the spy “hears” C_t and updates)
        P_spy = P_spy * torch.exp(log_likes)
        P_spy = P_spy / P_spy.sum()

    return N, S


# ---- Inference example using SMC ----
if __name__ == "__main__":
    # Example placeholders
    T = 5
    n_locations = 10
    n_players   = 5
    raw = torch.randn(T, n_locations).abs() 
    C_obs = raw / raw.sum(dim=-1, keepdim=True)
    

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
