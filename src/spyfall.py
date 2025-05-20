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



def spyfall_model(T, L, P, i_seq, C_obs=None):
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
    beta = pyro.param("beta", torch.tensor(1.0), constraint=constraints.positive)
    lam = pyro.param("lam", torch.tensor(0.5), constraint=constraints.unit_interval)

    # Enumerate discrete latents
    N = pyro.sample("N",
                    dist.Categorical(torch.ones(L) / L),
                    infer={"enumerate": "parallel"})
    S = pyro.sample("S",
                    dist.Categorical(torch.ones(P) / P),
                    infer={"enumerate": "parallel"})

    # Initial spy belief
    pi = torch.ones(L) / L

    for t in range(T):
        speaker = i_seq[t]
        # Non-spy concentration
        ns_alpha = beta * ((1 - lam) * torch.ones_like(pi) + lam * F.one_hot(N, num_classes=L).float())
        # Spy concentration
        spy_alpha = beta * ((1 - lam) * torch.ones_like(pi) + lam * pi)
        # Mixture per speaker
        mask_spy = (speaker == S).unsqueeze(-1)
        alpha = torch.where(mask_spy, spy_alpha, ns_alpha)

        # Emit or condition
        obs = None if C_obs is None else C_obs[t]
        C_t = pyro.sample(f"C_{t}", dist.Dirichlet(alpha), obs=obs)

        # Spy belief update: compute ell(k) = p(C_t | N = k), then weight those secenarios by pi
        ell = torch.zeros_like(pi)
        for j in range(L):
            a_j = beta * ((1 - lam) * torch.ones_like(pi) + lam * F.one_hot(torch.tensor(j), num_classes=L).float())
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
    beta_true = pyro.param("beta").item()
    lam_true  = pyro.param("lam").item()
    true_N = int(samples['N'][idx])
    true_S = int(samples['S'][idx])
    return C_obs, beta_true, lam_true, true_N, true_S


def infer_posterior(T, L, P, i_seq, C_obs, num_steps=2000, lr=1e-2):
    """
    Infer posterior over (beta, lam, N, S) via SVI with enumeration.
    """
    conditioned = pyro.condition(
        spyfall_model,
        data={f"C_{t}": C_obs[t] for t in range(T)}
    )

    # empty guide since we don't have continuous variational parameters
    # beta and lam are just pyro.param, will get learned directly
    def guide(T, L, P, i_seq, C_obs=None, init_beta=None, init_lam=None):
        pass    

    optim = Adam({"lr": lr})
    svi = SVI(conditioned, guide, optim, loss=TraceEnum_ELBO())
    for step in range(num_steps):
        loss = svi.step(T, L, P, i_seq, C_obs)
        if step % (num_steps // 5) == 0:
            print(f"Step {step:4d} \tLoss = {loss:.3f}")

    beta_post = pyro.param("beta").item()
    lam_post  = pyro.param("lam").item()

    # Exact discrete marginals
    marginals = TraceEnum_ELBO().compute_marginals(
        conditioned, guide, T, L, P, i_seq, C_obs
    )
    # convert logits to probabilities
    qN = expit(marginals['N'].logits.detach().numpy()).round(4)
    qS = expit(marginals['S'].logits.detach().numpy()).round(4)
    return beta_post, lam_post, qN, qS

def run_experiment(
    T: int,
    L: int,
    P: int,
    i_seq: list,
    num_samples: int = 50,
    num_steps: int = 1000,
    lr: float = 1e-2,
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

    for idx in range(num_samples):
        C_obs, beta_true, lam_true, true_N, true_S = extract_observations(samples, idx)
        beta_post, lam_post, qN, qS = infer_posterior(
            T, L, P, i_seq, C_obs, num_steps=num_steps, lr=lr
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
    return results



if __name__ == '__main__':
    T = 4
    P = 3
    L = 10
    i_seq = [t % P for t in range(T)]

    results = run_experiment(
        T=T, L=L, P=P,
        i_seq=i_seq,
        num_samples=1,
        num_steps=300,
        lr=1e-2
    )

    pprint.pprint(results, indent=2, depth=3, compact=False)
