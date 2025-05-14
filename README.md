# Spyfall with Pyro

We implement bots ("AI") to play a Spyfall-inspired game using Pyro, a probabilistic programming language (PPL) in Python. 

## Game rules

The game has $s$ players with $n$ possible locations. A location and a player is chosen at random
$$N \sim Unif(1, \dots, n), \quad S \sim Unif(1, \dots, s).$$

Players who are not the spy knows the location $N$ but not $S$, player who is the spy knows $S$ but not $N$. The spy wants to find the location $N$, the non-spy players want to find the spy $S$. 

Each turn $t = 1, 2, \dots$, a designated player $j_t \in [s]$ publicly announces a claim $C_t$ that is a probability vector over locations
$$ C_t \in \Delta_n. $$

At any turn, any player can trigger a game-ending move. 
* The spy can name the location. If correct, then the spy wins. Otherwise, the non-spy team wins. 
* A non-spy can trigger a collective (majority vote) to name the spy. If correct, AND if the spy cannot name the location (this serves as "conclusive proof" that they are indeed the spy), then the non-spy team wins. Otherwise, the spy wins. 

# Spyfall with PPL

For a first implementation, we make some simplifying assumptions. These can be relaxed to add more human-like dynamics later.  

1. **Direct communication with no errors.** Players communicate over $\Delta_n$ directly, and if a player intends to send out $C_t$, then everyone else receives $C_t$ and not a noise-corrupted version of it. In reality, people communicate with words that describe the location, which then adds two things: (i) a translation (NLP) to (probability) layer, and (ii) an encode/decode layer, which potentially makes the received message different from the sent message. 
2. **Automatic end-game trigger.** Game ends after $5n$ rounds (each person share their info 5 times) and people do MLE votes, ie, vote for the most-likely candidate (spy or location) based on their private beliefs.   


## State variables

* **Latent globals**: 

  $$
    N\in\{1,\dots,n\},\quad S\in\{1,\dots,s\}.
  $$
* **Observed**

  $$
    \{(j_t,C_t)\}_{t=1}^T,
    \quad j_t\in\{1,\dots,s\},\;
          C_t\in\Delta_n.
  $$
* **Public game state** up to $t$: (who said what): $\mathcal G_t = \bigl\{(j_u,C_u)\bigr\}_{u=1}^t.$
* **Private info** of player $i$:

  $$
    \mathrm{priv}_i =
    \begin{cases}
      N & i\neq S,\\
      \varnothing & i = S.
    \end{cases}
  $$

## Beliefs, Prior, Posterior

* **Priors.** Uniform priors.

* **Player-$i$ beliefs** at start of turn $t$:

  * Location belief: a point in $\Delta_n$

    $$
      P_{i,t}(\ell)
      =P\bigl(N=\ell\mid \mathcal G_{t-1},\,\mathrm{priv}_i\bigr)
      \;\in\;\Delta_n.
    $$
  * Spy belief: a point in $\Delta_s$

    $$
      \widehat S_{i,t}(k)
      =P\bigl(S=k\mid \mathcal G_{t-1},\,\mathrm{priv}_i\bigr)
      \;\in\;\Delta_s.
    $$

* Players start with uniform priors, then do posterior updates of their beliefs as the game progress. The players have their own update strategies that need not be Bayesian. 

## Public belief and Information gain

It is useful to introduce the notation of a Public belief. These are computed without any private information by a watching public observer. The public beliefs and associated metrics (eg information gains, partial beliefs) are computed as a "service", ie as decoration or transformation of the public game state. Players can use this Public belief to decide their emission strategies below.  

  * **Public Location Belief**: a location belief, ie, a point in $\Delta_n$, computed without private information. 

    $$
      P_{t}(\ell)
      =P\bigl(N=\ell\mid \mathcal G_{t-1}\bigr)
      \;\in\;\Delta_n.
    $$

  * **Public Spy Belief**: a spy belief, ie, a point in $\Delta_S$, computed without private information. 

    $$
      \widehat S_{t}(k)
      =P\bigl(S=k\mid \mathcal G_{t-1}\bigr)
      \;\in\;\Delta_s.
    $$  

  * **Public marginal belief**: these are versions of public or location belief but marginalizing out some information (eg, all messages of a particular player). 

  * **Public belief update**: public believes (full or marginal) are computed with Bayes update. 


## Player strategy

A player strategy has two parts

* **Private belief update strategy**: this is a function $g_j$ that provides a posterior update to the private belief, given all public information so far (public game state, public beliefs) and past private information (prior private belief, private information). 

Belief update can be simple Bayesian, OR it can be more sophisticated strategies, OR it can be "random" (ie a "joker" player) for baseline, etc. 

* **Emission strategy**: an emission strategy of each player $j_t$ is a function that takes (private belief, private information) and outputs $C_t$, the message that the player wants to broadcast at their turn. For simplicity, we will assume that it has this form: first compute a latent state $\gamma_{j_t,t}$, and then the message is drawn as a Dirichlet with this parameter. That is, the emission strategy is given by the function

$$
  \Gamma_{j_t}:
    \bigl(P_{j_t,t},\,\widehat S_{j_t,t},\,\mathrm{priv}_{j_t}\bigr)
    \;\longrightarrow\;
    \gamma_{j_t,t}\in\mathbb R_{>0}^n.
$$

After a draw of $\Gamma_{j_t}$, the public claim is drawn as

$$
  C_t \;\sim\;\mathrm{Dirichlet}\bigl(\gamma_{j_t,t}\bigr).
$$

# Belief update strategies to implement

We will implement these belief updates for players: 
* Bayes update
* (more later: based on information gain)

## Bayes update

Do Bayesian updates on the private beliefs. This is the same as the logic for public beliefs updates, except that we also have private information of the player. 

# Emission strategies to implement

Recall that each speaker $j_t$ computes Dirichlet parameters by a strategy map

$$
  \Gamma_{j_t}:
    \bigl(P_{j_t,t},\,\widehat S_{j_t,t},\,\mathrm{priv}_{j_t}\bigr)
    \;\longrightarrow\;
    \gamma_{j_t,t}\in\mathbb R_{>0}^n.
$$

We will implement these strategies

* Flat Hide-Reveal
* Dynamic Hide-Reveal
* (more later: utility trade-off)

## Flat Hide-Reveal

Here,
$$
  \gamma_{j_t,t}
  =\beta ((1-\lambda) \mathbf 1
  \;+\; \lambda \,P_{j_t,t}),
  \qquad
  \beta > 0, \lambda \in [0,1]. 
$$

Here, $\lambda$ is how much we information to reveal, and $\beta$ is the Dirichlet concentration parameter, which controls how sharp we want the message to be (close to our intended point, more like uniform, or more on the edge of the simplex). 

* $\lambda \approx 1$, high $\beta$: Dirichlet is sharply concentrated near $P_{j_t,t}$. Player reveals their true private belief. 
* $\lambda \approx 1$, low $\beta$: Dirichlet is more concentrated on the vertices of the simplex, but biased towards the highest coordinate of $P_{j_t,t}$. This has the behavior that the player is revealing a small subset of locations that contain their  true private belief. 
* $\lambda \approx 0$, high $\beta$: Player is revealing nothing.
* $\lambda \approx 0$, low $\beta$: Player is bluffing. 


These parameters are constant to the player, suggesting "personality" (cautious at reveal information vs eager to show others that they are not the spy). 

## Dynamic Hide-Reveal

Same as previous, but $\beta$ and $\lambda$ depends on $t$. 

$$
  \gamma_{j_t,t}
  =\beta_t ((1-\lambda_t) \mathbf 1
  \;+\; \lambda_t \,P_{j_t,t}),
  \qquad
  \beta_t > 0, \lambda_t \in [0,1]. 
$$

$\lambda_t$ is updated based on the **public** spy belief $\hat{S}_{t}$. If the player is the top public suspect, then they try to avert the suspicion by increasing $\lambda_t$ (more truthful). Meanwhile, $\beta_t$ is updated based on the **public** location belief $\hat{P}_{t}$. If the public location belief is concentrated near the true location, then the player will lower $\beta_t$ to obscure their message. 

## Statistics to gather

These are statistics we want to compute based on public information, so that we can later form strategies that use these statistics. 

### Information gain

To quantify the “value” of a new claim, define for any two distributions $p,q\in\Delta_m$

$$
  \mathrm{IG}(p,q)
  =D_{\mathrm{KL}}\bigl(q\;\big\|\;p\bigr)
  =\sum_{i=1}^m q_i\log\frac{q_i}{p_i}\;\ge0.
$$

* **Turn-wise gain** for player $i$ on locations:
  $\mathrm{IG}\bigl(P_{i,t-1},\,P_{i,t}\bigr).$
* **Aggregate gain** from all of $j$’s messages up to $T$:
  $\mathrm{IG}\bigl(P_{i,0},\,P_{i,T}\bigr).$


# Code up the game with Pyro: TODOs

* Setup game: draw $N, S$, initialize players (choose their strategy types and initialize their prior parameters)
* For each turn: 
    * The speaker computes their emission, announces $C_t$. 
    * The public beliefs are updated. 
    * Each player updates their private belief. 
    * Check if the end-game threshold for any player has been reached. If so, reaches the endgame. Otherwise, pick next player.  

Our goal is to write codes to
* Simulate games
* Collect statistics to try out different strategies against each other and fine-tune parameters
* Write robust codes so that our agents (bot players) can be used as baselines for benchmarks like RL agents in a Gym environment. 


