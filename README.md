# basis-aware-threshold
Code for the paper [Closing the Curious Case of Neural Text Degeneration](https://arxiv.org/abs/2310.01693)

## BAT sampling implementation

BAT itself is fairly simple to implement. The only requirements you need are `cvxpy` and `numpy`. You can install from `requirements.txt`.

```python
import numpy as np, cvxpy as cp

def sample(
        embed: np.array,  # Model embedding matrix (or SVD approximation)
        threshold: float,  # Truncation threshold
        probs: np.array,  # Model output probabilities
    ) -> int:
    for token_id in np.random.choice(len(probs), size=len(probs), p=probs, replace=False):
        if probs[token_id] >= threshold: # Program will be infeasible, no need to run it
            return token_id

        # Construct and attempt to solve the linear program
        exp_delta = 1 / max(1 - threshold, 1e-10)
        p = cp.Variable(probs.shape)
        constraints = [
            embed.T @ p == embed.T @ probs,
            cp.sum(p) == 1,
            p[token_id] == 0,
            p <= probs * exp_delta,
        ]
        problem = cp.Problem(cp.Minimize(0), constraints)
        try:
            problem.solve()
            if problem.status == "infeasible":
                return token_id
        except cp.error.SolverError: # Numerical instability suggests infeasible
            return token_id 
    return np.argmax(probs) # Fall back to greedy
```
Our implementation can be found in `src/parallel.py`

## Experiments

### Setup
To replicate experiments, clone and navigate to this repository, then
```
pip install .
```

### Parameter matching

To find the BAT parameter that rejects human tokens at the same rate as a threshold sampling parameter,
compute the per-token rejection thresholds with `scripts/param_matching.py`,
and analyze the results with `scripts/relative_conserve.py` (which prints the parameters for each method that reject 25% of human tokens).

### MAUVE
Use `scripts/generate.py` to generate completions to prefixes from `data/webtext/`.
Once the texts are generated, use `scripts/compute_mauve.py` to compute the MAUVE scores.

## Figures

![`scripts/viz/fig1.py`](fig/fig1.png)
Generate data with `scripts/unit_test.py`, then generate figures with `scripts/viz/fig1.py` and `scripts/unit_test.py`.

![`scripts/viz/hrr.py`](fig/hrr.png)
`scripts/viz/hrr.py`.

![`scripts/viz/relcon_by_cons.py`](fig/constraints.png)
`scripts/viz/relcon_by_cons.py`.

![`scripts/viz/tern.py`](fig/tern.png)
`scripts/viz/tern.py`.


## Cite

```bibtex
@misc{finlayson2023closing,
      title={Closing the Curious Case of Neural Text Degeneration}, 
      author={Matthew Finlayson and John Hewitt and Alexander Koller and Swabha Swayamdipta and Ashish Sabharwal},
      year={2023},
      eprint={2310.01693},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact

mattbnfin@gmail.com
