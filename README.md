# Spyfall with Pyro


We implement bots ("AI") to play a Spyfall-inspired game using Pyro, a probabilistic programming language (PPL) in Python. 

Our goal is to write codes to
* Simulate multiple games
* Collect statistics to try out different strategies against each other and fine-tune parameters
* Write robust codes so that our agents (bot players) can be used as baselines for benchmarks like RL agents in a Gym environment. 

For game rules and the math, see ``math.md``

## Code details

```
source ../shared-uv-venv/bin/activate
```

Packages
* `pyro-ppl`
* `matplotlib`
* `seaborn`

## TODOs

* [ ] Replace public spy belief with player directly share their spy beliefs. Do more sophisticated spy belief updates. 
* [ ] Ensure that in mixed strategy simulations, spy consistently uses the same strategy
* [ ] Check details of dynamic belief updates. 
* [ ] Add ``analysis`` codes for strat comparisons and param tuning. 
* [ ] Check if information gain is a good stat. 
* [ ] Automate strategy search
* [ ] Add NLP layer


## Major updates

* May 2025: initial implementation