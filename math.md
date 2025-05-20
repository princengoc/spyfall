# Spyfall in Pyro: Model and implementation notes

## Game rules

The game has $s$ players with $n$ possible locations. A location and a player is chosen at random

$$N \sim Unif(1, \dots, n), \quad S \sim Unif(1, \dots, s).$$

Players who are not the spy knows the location $N$ but not $S$, player who is the spy knows $S$ but not $N$. The spy wants to find the location $N$, the non-spy players want to find the spy $S$. 

Each turn $t = 1, 2, \dots$, a designated player $j_t \in [s]$ publicly announces a claim $C_t$ that is a probability vector over locations

$$ C_t \in \Delta_{n-1}. $$

At any turn, any player can trigger a game-ending move. 
* The spy can name the location. If correct, then the spy wins. Otherwise, the non-spy team wins. 
* A non-spy can trigger a collective (majority vote) to name the spy. If correct, AND if the spy cannot name the location (this serves as "conclusive proof" that they are indeed the spy), then the non-spy team wins. Otherwise, the the spy wins (though in the codes we record this as a "tie" when we test out strategies). 

# Spyfall with PPL

For a first implementation, we make some simplifying assumptions. These can be relaxed to add more human-like dynamics later.  

1. **Direct communication with no errors.** Players communicate over $\Delta_n$ directly, and if a player intends to send out $C_t$, then everyone else receives $C_t$ and not a noise-corrupted version of it. In reality, people communicate with words that describe the location, which then adds two things: (i) a translation (NLP) to (probability) layer, and (ii) an encode/decode layer, which potentially makes the received message different from the sent message. 
2. **Automatic end-game trigger.** Game ends when either:
  - spy is very confident (> 90%) of the location, OR
  - everybody's except the spy top suspect is the same person (which could be wrong)

Then a vote is triggered, and players vote for the most-likely candidate (spy or location) based on their private beliefs.   

## The generative model

The model is the key specification of a probabilistic program. We want a "global model" that **all players** will use, where making claims is drawing a sample, and "processing others claims" is running inference, and we want to use ``pyro``. 

Our model is this. Let $N,S$ be the latent globals random variables, 
$$
  N\in\{1,\dots,n\},\quad S\in\{1,\dots,s\}.
$$
Over $T$ turns, we observe
$$
  \{(j_t,C_t)\}_{t=1}^T,
  \quad j_t\in\{1,\dots,s\},\;
  C_t\in\Delta_{n-1},
$$
where $j_t$ is the index of the speaker of that round, and $C_t$ is his claim. At round $t$, let $\pi_t \in \Delta_{n-1}$ be the initial spy's prior over the locations. We model the claim distribution emitted at by speaker $j_t$ as a Dirichlet, with concentration parameter

$$
\alpha(j_t) = 
\begin{cases}
  \beta \big((1 - \lambda) \mathbf{1} + \lambda \mathbf{e}_N\big) & \text{if } j_t \neq S \\
  \beta \big((1 - \lambda) \mathbf{1} + \lambda \pi_t\big) & \text{if } j_t = S
\end{cases} \quad  \in \Delta_{n-1}.
$$

After turn $t$, if the spy didn't speak that turn, the spy does a posterior update of his belief with Bayes' rule
$$
\pi_{t+1}(k) = \begin{cases} 
\pi_t(k) & \text{if } j_t = S \\
\propto \mathbb{P}(C_t | N =k)\pi_t(k)
\end{cases},
$$
where $P(C_t | N = k)$ is the Dirichlet with concentration $\alpha = \beta \big((1 - \lambda) \mathbf{1} + \lambda \mathbf{e}_k\big)$ above. 

### Interpretations

Here, $\lambda$ is how much we information to reveal, and $\beta$ is the Dirichlet concentration parameter, which controls how sharp we want the message to be (close to our intended point, more like uniform, or more on the edge of the simplex). 

* $\lambda \approx 1$, high $\beta$: Dirichlet is sharply concentrated near $P_{j_t,t}$. Player reveals their true private belief. 
* $\lambda \approx 1$, low $\beta$: Dirichlet is more concentrated on the vertices of the simplex, but biased towards the highest coordinate of $P_{j_t,t}$. This has the behavior that the player is revealing a small subset of locations that contain their  true private belief. 
* $\lambda \approx 0$, high $\beta$: Player is revealing nothing: makes vague statements that say along the lines of "all locations are equally likely".
* $\lambda \approx 0$, low $\beta$: Player is bluffing. 

These parameters are constant to the player, suggesting "personality" (cautious at reveal information vs eager to show others that they are not the spy). They are fixed and can be learned by ``pyro``. 

### Strategy parameters

The spy strategy here is Bayes update on the public information (the claims). The only "strategy parameters" are $\beta$ and $\lambda$, that controls the "player personality".

# Implementation 

I use [pyro](https://pyro.ai/), a Python package for probabilsitic programming. I find the [tutorial](https://pyro.ai/examples/intro_long.html) to be quite good, though details are lacking once we want to dig deeper on specific functions. For example, I struggled to get parallel[enumeration](https://pyro.ai/examples/enumeration.html) to work `pyro.markov` without having to do a lot of ``.unsqueeze(1).expand(...)`` gymnastics. In the end, I opted for a simple `for` loop. Probably not the most efficient, but considering that $n, T, s \sim 1e1$ for my case, it really doesn't matter. Using ``pyro`` is bit of an overkill, as our inference is exact, but it's a good excuse to learn `pyro`. For fixed $\beta, \lambda$, an online update will yield the posterior of $N$ and $S$ in $O(T \cdot (s+ n))$. I'm not 100% sure on `pyro` `TraceEnum_ELBO` implementation, but I suspect that it's $O(T (s^2+n^2))$, vectorized over $s$ and $n$ in the parallel enumeration step. 

## Pyro on Spyfall

The [pyro tutorial](https://pyro.ai/examples/intro_long.html) does a good job explaining PPL. Here are some short notes specific to our Spyfall implementation. 

### TraceEnum_ELBO

We use ``TraceEnum_ELBO`` since our latent variables $N, S$ are discrete and can be exhaustively enumerated. For our discrete latent $Z = (N,S)$, the posterior distribution $q(z|x)$ is represented by a point in $\Delta_{n-1} \times \Delta_{s-1}$. Reparametrize via the `logit` map, this means points $q(z|x)$ is represented by points $\mathbb{R}^{n+s-2}. The  evidence lower bound (ELBO) is exactly
$$ \mathbb{E}_{q(z|x)} (\log(\frac{p(x,z)}{q(z|x)})) = \sum_z q(z|x)\log(p(x,z) - \log(q(z|x))). $$

So `pyro` enumerates all possiblities of $z$ in parallel and finds $q(z|x)$ (ie the distributions of $N$ and $S$) by optimizing the ELBO over $\mathbb{R}^{n+s-2}$, which can be solved exactly in one step. For our game, this means `pyro` keep track of $ns$ many cases of "site is $i$, spy is $j$", and for each case, compute its likelihood given the data. 

### Effect of the strategy parameters 
One nice thing about ``pyro`` is that we can use SVI to learn the hyper parmeters like $\beta, \lambda$ with constraints just by declaring them as ``pyro.param``. We found that the exact values of $\beta, \lambda$ doesn't really move the loss all that much when all players share the same param. We had a variational implementation as well, which can introduce a correlation between $\beta$ and $\lambda$. In particular, let's assume that $logit(\lambda)$ and $\log(\beta)$ are correlated normal. If this correlation $\rho > 0$, this means that the player tends to be "honest and direct" when they want to convince others that they are not the spy. If $\rho < 0$, then the player tends to give "vague hint" in such situations as opposed to direct hints. This is quite interesting, so it would be interesting to learn a human's $\beta, \lambda$ with such a variational approach in a human-facing version of the game. (FUTURE work).

# Findings



<!-- Player strategies will manifest as different assumptions on the joint probability of the model. This makes it easy and interesting to test for effects such as
* a player changes their claim strategy from being "vague" to "sharp" 
* what happens to over-optimized players who have mistaken assumptions on how others behave
* win rates of naive vs calculating player
* what is the most robust strategy for a spy
* what is the most robust strategy for non-spy -->
