# basis-aware-truncation
Code for the paper

## Setup
Clone and navigate to this repository, then
```
pip install .
pip install -r requirements.txt
```

## BAT sampling implementation

BAT itself is fairly simple to implement.
```python
import numpy as np, cvxpy as cp

def sample(
        embed: np.array,  # Model embedding matrix
        threshold: float,  # Truncation threshold
        probs: np.array,  # Model output probabilities
    ):
    for token_id in np.random.choice(len(probs), size=len(probs), p=probs, replace=False)
        if probs[token_id] >= threshold: # Program will be infeasible, no need to run it
            return token_id

        # Construct and attempt to solve the linear program
        exp_delta = 1 / max(1 - threshold, 1e-10)
        p = cp.Variable(probs.shape)
        constraints = [
            embed.T @ p == embed.T @ hat_p,
            cp.sum(p) == 1,
            p[token_id] == 0,
            p <= hat_p * exp_delta,
        ]
        problem = cp.Problem(cp.Minimize(0), constraints)
        try:
            problem.solve(**kwargs)
            if problem.status == "infeasible":
                return token_id
        except cp.error.SolverError: # Numerical instability suggests infeasible
            return token_id 
    return np.argmax(probs) # Fall back to greedy
```
Our implementation can be found in `src/parallel.py`

## Experiments

Use `scripts/generate.py` to generate completions to prefixes from `data/webtext/`.

```
$ python scripts/generate.py -h
usage: generate.py [-h] [--data DATA] [--out OUT] [--verbose] [--seed SEED]
                   [--examples EXAMPLES] [--max-length MAX_LENGTH]
                   [--prompt-length PROMPT_LENGTH] [--start START]
                   [--model MODEL] [--constraint-size CONSTRAINT_SIZE]
                   [--solver SOLVER] [--dpp] [--processes PROCESSES]
                   [--threads THREADS] [--cpu] [--parallel]
                   [--generate GENERATE] [--adverse-rate ADVERSE_RATE]
                   [--retry-limit RETRY_LIMIT] [--strat STRAT]
                   [--adversary-strat ADVERSARY_STRAT] [--param PARAM]
                   [--adversary-param ADVERSARY_PARAM] [--bottleneck-aware]
                   [--adversary-bottleneck-aware] [--wandb]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           webtext data
  --out OUT             results output file
  --verbose, -v
  --seed SEED           random seed
  --examples EXAMPLES   number of completions
  --max-length MAX_LENGTH
                        maximum completion length
  --prompt-length PROMPT_LENGTH
                        length of the prompt
  --start START         number of completions from the beginning to skip
  --model MODEL
  --constraint-size CONSTRAINT_SIZE
                        number of constraints in the linear program
  --solver SOLVER       solver backend
  --dpp                 use dpp compilation for the linear program. Too slow,
                        do not use.
  --processes PROCESSES
                        CPU processes to use
  --threads THREADS     threads per CPU process. Try 1.
  --cpu                 use if no GPU available
  --parallel            enables parallel decoding
  --generate GENERATE   generation setting. Only 'simple' used in the paper
  --adverse-rate ADVERSE_RATE
                        ignore
  --retry-limit RETRY_LIMIT
                        fall back to greedy after rejecting n tokens
  --strat STRAT         base truncation method, `eta` works best
  --adversary-strat ADVERSARY_STRAT
                        ignore
  --param PARAM         truncation parameter
  --adversary-param ADVERSARY_PARAM
                        ignore
  --bottleneck-aware    use BA constraints
  --adversary-bottleneck-aware
                        ignore
  --wandb               track GPU usage on W&B
```

Once the texts are generated, use `scripts/compute_mauve.py` to compute the MAUVE scores.
```
$ python scripts/compute_mauve.py --help
usage: compute_mauve.py [-h] [--device DEVICE] [--out OUT] [-n N]
                        [--q-file Q_FILE] [--p-file P_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --out OUT             output file
  -n N                  run MAUVE on first `n` examples
  --q-file Q_FILE, -q Q_FILE
                        generated text
  --p-file P_FILE, -p P_FILE
                        human text
```


## Figures

![`scripts/viz/fig1.py`](fig/fig1.png)
`scripts/viz/fig1.py`

![`scripts/viz/hrr.py`](fig/hrr.png)
`scripts/viz/hrr.py`

![`scripts/viz/relcon_by_cons.py`](fig/constraints.png)
`scripts/viz/relcon_by_cons.py`

![`scripts/viz/tern.py`](fig/tern.png)
`scripts/viz/tern.py`



## Contact

mattbnfin@gmail.com
