# The Math

## Game rules

The game has $s$ players with $n$ possible locations. A location and a player is chosen at random

$$N \sim Unif(1, \dots, n), \quad S \sim Unif(1, \dots, s).$$

Players who are not the spy knows the location $N$ but not $S$, player who is the spy knows $S$ but not $N$. The spy wants to find the location $N$, the non-spy players want to find the spy $S$. 

Each turn $t = 1, 2, \dots$, a designated player $j_t \in [s]$ publicly announces a claim $C_t$ that is a probability vector over locations

$$ C_t \in \Delta_n, $$

and an accusation of spy
$$ A_{t} \in \Delta_s $$
of what the probability of each person being a spy is, from their viewpoint. 

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


## State variables

* **Latent globals**: 

  $$
    N\in\{1,\dots,n\},\quad S\in\{1,\dots,s\}.
  $$
* **Observed**

  $$
    \{(j_t,C_t, A_t)\}_{t=1}^T,
    \quad j_t\in\{1,\dots,s\},\;
          C_t\in\Delta_n, \quad A_t \in \Delta_s
  $$
* **Public game state** up to $t$: (who said what): $\mathcal G_t = \{(j_u,C_u, A_u)\}_{u=1}^t.$
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
      =P(N=\ell\mid \mathcal G_{t-1},\,\mathrm{priv}_i)
      \;\in\;\Delta_n.
    $$
  * Spy belief: a point in $\Delta_s$

    $$
      \widehat S_{i,t}(k)
      =P(S=k\mid \mathcal G_{t-1},\,\mathrm{priv}_i)
      \;\in\;\Delta_s.
    $$

* Players start with uniform priors, then do posterior updates of their beliefs as the game progress. The players have their own update strategies that need not be Bayesian. 

## Public belief and Information gain

It is useful to introduce the notation of a Public belief. These are computed without any private information by a watching public observer. The public beliefs and associated metrics (eg information gains, partial beliefs) are computed as a "service", ie as decoration or transformation of the public game state. Players can use this Public belief to decide their emission strategies below.  

  * **Public Location Belief**: a location belief, ie, a point in $\Delta_n$, computed without private information. 

    $$
      P_{t}(\ell)
      =P(N=\ell\mid \mathcal G_{t-1})
      \;\in\;\Delta_n.
    $$

  * **Public Spy Belief**: a spy belief, ie, a point in $\Delta_S$, computed without private information. 

    $$
      \widehat S_{t}(k)
      =P(S=k\mid \mathcal G_{t-1})
      \;\in\;\Delta_s.
    $$  

  * **Public marginal belief**: these are versions of public or location belief but marginalizing out some information (eg, all messages of a particular player). 

  * **Public belief update**: public believes (full or marginal) are computed with Bayes update. 


# Public belief update

Public location and spy beliefs can be updated sequentially. 

## Public spy belief update

Let $\hat{S}_{t-1}$ be the previous round's public spy belief. In round $t$, player $j$ is active and they reveal an accusation vector $\alpha_t \in \Delta_s$. If they are the spy (happens with prior probability $\hat{S}_{t-1}(j)$), then our correct update should be $e_j$. If they are not the spy, then they are revealing their true information, so our update should be $\alpha_t$. Therefore, our Bayesian spy belief update equation is
$$ \hat{S}_t = p_{spy} e_j + (1-p_{spy}) \alpha_t $$
where $p_{spy}$ = probability that $j$ is a spy = $\hat{S}_{t-1}(j)$. 


## Public belief update

Fix $\beta=1$ and $\lambda=0.5$.  Define for each candidate $\ell$:

$$
\gamma^\mathrm{honest}_\ell
= (1-\lambda)\mathbf1 + \lambda\,e_\ell
= 0.5\,\mathbf1 + 0.5\,e_\ell,
\qquad
\gamma^\mathrm{spy}
= (1-\lambda)\mathbf1 + \lambda\,P_{t-1}
= 0.5\,\mathbf1 + 0.5\,P_{t-1}.
$$

Since $S_{t-1}(j_t)$ is the probability that the speaker $j = j_t$ is a spy, the mixture-likelihood of the observed claim $C_t\in\Delta_n$ is

$$
p(C_t\mid N=\ell)
=(1-S_{t-1}(j_t))\,\mathrm{Dir}(C_t;\gamma^\mathrm{honest}_\ell)
\;+\;
S_{t-1}(j_t)\,\mathrm{Dir}(C_t;\gamma^\mathrm{spy}).
$$

Finally the Bayes update is

$$
P_t(\ell)
=\frac{P_{t-1}(\ell)\;p(C_t\mid N=\ell)}
{\sum_{\ell'=1}^n P_{t-1}(\ell')\;p(C_t\mid N=\ell')}\,.
$$

# Player belief updates

Speaker does not update their belief.

## Accusation belief updates

Spy is pretending to be a non-spy, so they have the same update rule as non-spy. Furthermore, any player will deny that they are the spy, but is receptive to other suggestions. This means when player $j = j_t$ spoke in round $t$, their accusation updates player $i$'s belief $\hat{S}_{i,t}$ as follows
$$ \hat{S}_{i,t} = \hat{S}_{i,t-1}(j) e_j + (1-\hat{S}_{i,t-1}(j)) \bar{\alpha}_t, $$
where $\bar{\alpha}_t$ is the re-normalized version of $\alpha_t$ (the original accusation) but with the $i$-th coordinate zeroed-out. 

# Player strategies

## Accusation strategy

For both spy and non-spy, they will just reveal their private $\hat{S}_{i,t}$. 


### Claim strategy

The claim strategy has two parts

* **Private belief update strategy**: this is a function $g_j$ that provides a posterior update to the private belief, given all public information so far (public game state, public beliefs) and past private information (prior private belief, private information). 

Belief update can be simple Bayesian, OR it can be more sophisticated strategies, OR it can be "random" (ie a "joker" player) for baseline, etc. 

* **Emission strategy**: an emission strategy of each player $j_t$ is a function that takes (private belief, private information) and outputs $C_t$, the message that the player wants to broadcast at their turn. For simplicity, we will assume that it has this form: first compute a latent state $\gamma_{j_t,t}$, and then the message is drawn as a Dirichlet with this parameter. That is, the emission strategy is given by the function

$$
  \Gamma_{j_t}:
    (P_{j_t,t},\,\widehat S_{j_t,t},\,\mathrm{priv}_{j_t})
    \;\longrightarrow\;
    \gamma_{j_t,t}\in\mathbb R_{>0}^n.
$$

After a draw of $\Gamma_{j_t}$, the public claim is drawn as

$$
  C_t \;\sim\;\mathrm{Dirichlet}(\gamma_{j_t,t}).
$$

The accusation part: for now, we just keep it simple. 
- All players (including the spy) will always reveal their private $S_{i,t}$ belief, so this gives an appropriate update for the public belief $S_t$.
- All player's (including the spy) initial prior belief $S_{i,0}$ is that other people are spy, and not themselves. 




# Player belief update strategies to implement

We will implement these belief updates for players: 
* Bayes update
* (more later: based on information gain)

## Bayes update

Do Bayesian updates on the private beliefs. This is the same as the logic for public beliefs updates, except that we also have private information of the player. 

# Emission strategies to implement

Recall that each speaker $j_t$ computes Dirichlet parameters by a strategy map

$$
  \Gamma_{j_t}:
    (P_{j_t,t},\,\widehat S_{j_t,t},\,\mathrm{priv}_{j_t})
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
  =D_{\mathrm{KL}}(q\;\big\|\;p)
  =\sum_{i=1}^m q_i\log\frac{q_i}{p_i}\;\ge0.
$$

* **Turn-wise gain**: public information gain at step $t$ (thanks to player $i_t$ sharing information at that step): 
  $\mathrm{IG}(P_{i,t-1},\,P_{i,t}).$
* **Aggregate gain of player $i$**: need to compute public information WITHOUT person $i$'s messages, and then compute the information gain of those two distributions. Compute this with running Bayesian updates.  

# Code up the game with Pyro: TODOs

* Setup game: draw $N, S$, initialize players (choose their strategy types and initialize their prior parameters)
* For each turn: 
    * The speaker computes their emission, announces $C_t$, set accusation $A_t = S_{i,t}$.
    * The public beliefs are updated. 
    * Each player updates their private belief. 
    * Check if the end-game threshold for any player has been reached. If so, reaches the endgame. Otherwise, pick next player.  



