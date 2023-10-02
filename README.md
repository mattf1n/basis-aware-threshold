# basis-aware-truncation
Code for the paper

## Getting started
```
pip install .
pip install -r requirements.txt
```

## Experiments

```
python scripts/generate.py -h
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
