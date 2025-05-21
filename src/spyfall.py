import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.infer.predictive import Predictive
from pyro.optim import Adam
from pyro import poutine
import pprint
from scipy.special import expit  # inverse of logit
from typing import Dict
import numpy as np
from tqdm import tqdm


def spyfall_model(T, L, P, i_seq, C_obs=None, beta_ini = None, lam_ini = None):
    """
    Dynamic Bayes net for Spyfall-inspired game with unknown beta and lambda.

    Args:
        T       : int, number of turns
        L       : int, number of locations
        P       : int, number of players
        i_seq   : list[int], length-T speaker indices in [0..P-1]
        C_obs   : torch.Tensor of shape (T, L) or None
                  Observations for conditioning; if None, samples from prior.

    pyro.params (to be learned): 
        beta    : positive real, concentration scaler, one number for each player
        lam     : [0,1] weight between uniform and true/hyp belief, one number for each player

    Sample sites:
        N       : enumerated location index
        S       : enumerated spy index
        C_t     : Dirichlet utterance at turn t
    """
    # Priors over model hyperparameters
    if lam_ini is None: 
        lam_ini = torch.ones(P)*0.5
    if beta_ini is None:
        beta_ini = torch.ones(P)

    # beta = pyro.param("beta", beta_ini, constraint=constraints.positive)
    # lam = pyro.param("lam", lam_ini, constraint=constraints.unit_interval)
    beta = torch.tensor(1.0)
    lam = torch.tensor(0.5)

    # Enumerate discrete latents
    N = pyro.sample("N",
                    dist.Categorical(torch.ones(L) / L),
                    infer={"enumerate": "parallel"})
    S = pyro.sample("S",
                    dist.Categorical(torch.ones(P) / P),
                    infer={"enumerate": "parallel"})

    # Initial spy belief
    pi = torch.ones(L) / L

    # beta = beta.unsqueeze(-1) # [L,1]
    # lam = lam.unsqueeze(-1) # [L,1]

    for t in range(T):
        speaker = i_seq[t]
        # beta_s = beta[speaker]
        # lam_s = lam[speaker]        
        beta_s = beta
        lam_s = lam
        # Non-spy concentration
        ns_alpha = beta_s * ((1 - lam_s) * torch.ones_like(pi) + lam_s * F.one_hot(N, num_classes=L).float())
        # Spy concentration
        spy_alpha = beta_s * ((1 - lam_s) * torch.ones_like(pi) + lam_s * pi)
        # Mixture per speaker
        mask_spy = (speaker == S).unsqueeze(-1)
        alpha = torch.where(mask_spy, spy_alpha, ns_alpha)

        # Emit or condition
        obs = None if C_obs is None else C_obs[t]
        C_t = pyro.sample(f"C_{t}", dist.Dirichlet(alpha), obs=obs)

        # Spy belief update: compute ell(k) = p(C_t | N = k), then weight those secenarios by pi
        ell = torch.zeros_like(pi)
        for j in range(L):
            a_j = beta_s * ((1 - lam_s) * torch.ones_like(pi) + lam_s * F.one_hot(torch.tensor(j), num_classes=L).float())
            ell_j = torch.exp(dist.Dirichlet(a_j).log_prob(C_t))
            ell[..., j] = ell_j
        # Lock spy's own component
        ell = ell.masked_fill(F.one_hot(S, num_classes=L).bool(), 1.0)
        pi = (pi * ell) / (pi * ell).sum(dim=-1, keepdim=True)


def sample_prior(T, L, P, i_seq, num_samples=100):
    """
    Generate prior samples of latent and utterance trajectories.
    """
    predictive = Predictive(
        spyfall_model,
        num_samples=num_samples,
        return_sites=["N","S"] + [f"C_{t}" for t in range(T)]
    )
    return predictive(T, L, P, i_seq, None)


def extract_observations(samples, idx):
    """
    Convert prior samples into observation tensor plus true labels.
    """
    T = len([k for k in samples if k.startswith('C_')])
    L = samples['C_0'].shape[-1]
    # Stack C_t
    C_obs = torch.stack([samples[f'C_{t}'][idx] for t in range(T)], dim=0)
    # beta_true = pyro.param("beta").detach().numpy()
    # lam_true  = pyro.param("lam").detach().numpy()
    beta_true = 1
    lam_true = 0.5
    true_N = int(samples['N'][idx])
    true_S = int(samples['S'][idx])
    return C_obs, beta_true, lam_true, true_N, true_S


def infer_posterior(T, L, P, i_seq, C_obs, num_steps=2000, lr=1e-2, N = None, S = None, verbose=False):
    """
    Infer posterior over (beta, lam, N, S) via SVI with enumeration.

    Non-spy inference: known N, find S
    spy inference: known S, find N
    """
    data={f"C_{t}": C_obs[t] for t in range(T)}
    if N is not None: 
        assert S is None, 'cannot infer if both N and S are supplied'
        data['N'] = N
    if S is not None: 
        assert N is None, 'cannot infer if both N and S are supplied'
        data['S'] = S

    conditioned = pyro.condition(
        spyfall_model,
        data=data        
    )

    # empty guide since we don't have continuous variational parameters
    # beta and lam are just pyro.param, will get learned directly
    def guide(T, L, P, i_seq, C_obs=None, init_beta=None, init_lam=None):
        pass    

    optim = Adam({"lr": lr})
    svi = SVI(conditioned, guide, optim, loss=TraceEnum_ELBO())
    for step in range(num_steps):
        if N is not None:
            loss = svi.step(T, L, P, i_seq, C_obs, N)
        elif S is not None: 
            loss = svi.step(T, L, P, i_seq, C_obs, S)
        else: 
            loss = svi.step(T, L, P, i_seq, C_obs)
        if verbose and step % (num_steps // 5) == 0:
            print(f"Step {step:4d} \tLoss = {loss:.3f}")

    # beta_post = pyro.param("beta").detach().numpy()
    # lam_post  = pyro.param("lam").detach().numpy()
    beta_post = 1
    lam_post = 0.5

    # Exact discrete marginals
    marginals = TraceEnum_ELBO().compute_marginals(
        conditioned, guide, T, L, P, i_seq, C_obs
    )
    # convert logits to probabilities
    if N is None:
        qN = expit(marginals['N'].logits.detach().numpy()).round(4)
    else: 
        qN = np.ones(L) * np.nan
    if S is None:
        qS = expit(marginals['S'].logits.detach().numpy()).round(4)
    else: 
        qS = np.ones(P) * np.nan
    return beta_post, lam_post, qN, qS

def argmax_unique(arr: np.ndarray, default_val = -1): 
    """Returns -1 if multiple argmax found"""
    max_val = np.max(arr)
    argmax_idx = np.where(arr == max_val)[0]
    if len(argmax_idx) == 1:
        return argmax_idx[0]
    else: 
        return -1 # multiple argmax

def get_outcomes(result: Dict): 
    """Get game outcomes assuming that this is the last turn"""
    if not np.all(np.isnan(result['qS'])): 
        top_spy_suspect = argmax_unique(result['qS'])
    else: 
        top_spy_suspect = -1
    if not np.all(np.isnan(result['qN'])): 
        top_location_suspect = argmax_unique(result['qN'])
    else: 
        top_location_suspect = -1
    outcomes = {}
    outcomes['spy_found'] = top_spy_suspect == result['true_S']
    outcomes['loc_found'] = top_location_suspect == result['true_N']
    outcomes['spy_max_p'] = np.max(result['qS'])
    outcomes['loc_max_p'] = np.max(result['qN'])
    return outcomes

def run_experiment(
    T: int,
    L: int,
    P: int,
    i_seq: list,
    num_samples: int = 50,
    num_steps: int = 1000,
    lr: float = 1e-2,
    mode = 'public', 
    verbose = False
) -> list[dict]:
    """
    1. Generate `num_samples` trajectories under the prior.
    2. For each trajectory, extract observations and true (beta, lam, N, S).
    3. Infer posteriors via SVI.

    Returns:
        results: list of dicts with keys:
            'beta_true', 'lam_true', 'beta_post', 'lam_post',
            'true_N', 'qN', 'true_S', 'qS'
    """
    samples = sample_prior(T, L, P, i_seq, num_samples)
    results = []
    outcomes = []

    for idx in tqdm(range(num_samples)):
        C_obs, beta_true, lam_true, true_N, true_S = extract_observations(samples, idx)
        if mode == 'spy':
            beta_post, lam_post, qN, qS = infer_posterior(
                T, L, P, i_seq, C_obs, num_steps=num_steps, lr=lr, S = torch.tensor(true_S), verbose=verbose
            )
        elif mode == 'nonspy': 
            beta_post, lam_post, qN, qS = infer_posterior(
                T, L, P, i_seq, C_obs, num_steps=num_steps, lr=lr, N = torch.tensor(true_N), verbose=verbose
            )            
        else: 
            # public mode, both N and S are unknown
            beta_post, lam_post, qN, qS = infer_posterior(
                T, L, P, i_seq, C_obs, num_steps=num_steps, lr=lr, verbose=verbose
            )            

        result = {
            "beta_true": beta_true,
            "lam_true": lam_true,
            "beta_post": beta_post,
            "lam_post": lam_post,
            "true_N": true_N,
            "qN": qN,
            "true_S": true_S,
            "qS": qS,
        }
        results.append(result)
        outcome = get_outcomes(result)
        # record some meta data
        outcome['T'] = T
        outcome['P'] = P
        outcome['L'] = L
        outcome['true_N'] = true_N
        outcome['true_S'] = true_S
        outcomes.append(outcome)

    return results, outcomes



# experiment functions
## spy inference: 


if __name__ == '__main__':
    # T = 4
    P = 3
    L = 10

    for T in [1, 3, 6]: 
        i_seq = [t % P for t in range(T)]
        for mode in ['public', 'spy', 'nonspy']: 
            print(f"Gathering data for T={T}, {mode}")
            results, outcomes = run_experiment(
                T=T, L=L, P=P,
                i_seq=i_seq,
                num_samples=500,
                num_steps=2,
                lr=1e-2, 
                mode=mode, 
                verbose=False
            )
            # pprint.pprint(outcomes, indent=2, depth=3, compact=False)
            import pickle
            with open(f'T{T}P{P}L{L}_{mode}.pk', 'wb') as f:
                pickle.dump(outcomes, f)

            
