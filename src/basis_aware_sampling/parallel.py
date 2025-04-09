import argparse, json, sys, os, sys, operator, json, math, pickle, time
from functools import partial
from typing import Callable, Optional, Union, Literal
import torch, numpy as np, cvxpy as cp, torch.multiprocessing as mp
import src.lp as lp
from tqdm import tqdm

Device = Union[torch.device, str, None]
Source = Literal["prefix", "union", "primary", "adversary"]


formatters = dict(prefix=lp.BLACK, union=lp.BLACK, primary=lp.GREEN, adversary=lp.RED)


def get_token_source(
    get_threshold, get_adversary_threshold, logits, elapsed, next_token
) -> Source:
    threshold = get_threshold(logits)
    adversary_threshold = get_adversary_threshold(logits)
    probs = get_probs(logits)
    prob = probs[next_token]
    if prob >= adversary_threshold and (prob < threshold and elapsed is None):
        return "adversary"
    elif prob >= adversary_threshold:
        return "union"
    else:
        return "primary"


def our_generate(
    tokenizer,
    get_token_source: Callable[[torch.Tensor, int], str],
    forward: Callable[[torch.Tensor, Optional[tuple]], tuple[torch.Tensor, tuple]],
    get_threshold,
    sample: Callable[[torch.Tensor], tuple[int, Optional[float]]],
    sample_state,
    input_ids: torch.Tensor,
    max_length: int = 1024,
    parallel: bool = False,
    verbose: bool = False,
) -> list[int]:
    generation = input_ids[0].tolist()
    token_sources = ["prefix" for _ in generation]
    past_key_values = None
    position = None if not parallel else mp.current_process()._identity[0]
    idxs = range(input_ids.shape[-1], max_length)
    progress = tqdm(idxs, leave=False, position=position, disable=verbose)
    outfile = sys.stderr if verbose else open(os.devnull, "w")
    write = partial(print, end="", file=outfile, flush=True)
    if verbose:
        print(tokenizer.decode(generation), file=sys.stderr, end="", flush=True)
    for _ in progress:
        logits_, past_key_values = forward(input_ids, past_key_values)
        logits = logits_[0, -1, :].rename("vocab")
        threshold = get_threshold(logits)
        next_token, sample_state, elapsed = sample(logits, threshold, sample_state)
        if elapsed is not None:
            progress.set_description(f"{elapsed:.2f} seconds per program")  # type: ignore
        if next_token == tokenizer.eos_token_id:
            break
        generation.append(next_token)
        token_source = get_token_source(logits, elapsed, next_token)
        token_sources.append(token_source)
        write(formatters.get(token_source).format(tokenizer.decode(next_token)))
        input_ids = torch.tensor([[next_token]])
    print(sample_state, file=sys.stderr)
    return generation, token_sources


@torch.inference_mode()
def inference(
    model,
    input_ids: torch.Tensor,
    past_key_values: Optional[tuple],
    device: Device = None,
) -> tuple[torch.Tensor, tuple]:
    input_ids = input_ids.to(device)
    out = model(input_ids=input_ids, past_key_values=past_key_values)
    del input_ids
    del past_key_values
    return out.logits, out.past_key_values


def get_next_token(
    is_in_support: Callable[[torch.tensor, int], tuple[bool, float]],
    logits: torch.tensor,
    threshold: float,
    sample_state=None,
    device: Device = "cpu",
    max_retries: int = 10,
    **kwargs,
) -> tuple[int, Optional[float]]:
    p_hat = get_probs(logits)
    elapsed = None
    success = False
    size = min((max_retries, len(p_hat), sum(p_hat > 0)))
    for token_id in np.random.choice(len(p_hat), size=size, p=p_hat, replace=False):
        if p_hat[token_id] >= threshold:
            success = True
            break
        else:
            start_time = time.time()
            if is_in_support(logits, token_id, **kwargs):
                elapsed = time.time() - start_time
                success = True
                break
    if not success:
        token_id = p_hat.argmax()
    return int(token_id), sample_state, elapsed


def truncation_sample(
    logits,
    threshold,
    sample_state=None,
) -> tuple[int, None]:
    probs = get_probs(logits)
    truncation_probs = get_truncation_probs(probs, threshold)
    token_id = np.random.choice(len(probs), p=truncation_probs)
    elapsed = None
    return token_id, sample_state, elapsed


def get_truncation_probs(probs, threshold) -> np.array:
    trunc = probs * (probs >= threshold)
    remaining_prob = trunc.sum()
    return (
        trunc / remaining_prob
        if remaining_prob > 0
        else np.arange(len(probs)) == probs.argmax()
    )


def get_interval_probs(probs, lower_bound, upper_bound):
    trunc = probs * ((lower_bound <= probs) & (probs <= upper_bound))
    if trunc.sum() > 0:
        return trunc / trunc.sum()
    else:
        return None


def get_probs(logits):
    return logits.softmax(dim=-1).cpu().rename(None).numpy()


def adverse_sample(
    adverse_rate,
    get_adverse_probs,
    is_adverse_token,
    simple_sample,
    logits,
    _threshold,
    counter,
    retry_limit=None,
):
    elapsed = None
    is_adverse = np.random.random() < adverse_rate
    adverse_probs = get_adverse_probs(logits)
    if (is_adverse or counter > 0) and adverse_probs is not None:
        size = len(logits) if retry_limit is None else retry_limit
        for token_id in np.random.choice(
            len(logits),
            size=min(size, sum(adverse_probs > 0)),
            p=adverse_probs,
            replace=False,
        ):
            if is_adverse_token(logits, token_id):
                return int(token_id), counter - (not is_adverse), elapsed
    return simple_sample(logits, counter + is_adverse)


def is_in_support(embedding, get_threshold, logits, token_id, **kwargs):
    threshold = get_threshold(logits)
    logits = logits.refine_names("vocab")
    hat_p = get_probs(logits)
    if hat_p[token_id] >= threshold:
        return True
    sigma = 1 / max(1 - threshold, 1e-10)
    p = cp.Variable(hat_p.shape)
    constraints = [
        embedding.T @ p == embedding.T @ hat_p,
        cp.sum(p) == 1,
        p[token_id] == 0,
        p <= hat_p * sigma,
    ]
    problem = cp.Problem(cp.Minimize(0), constraints)
    try:
        problem.solve(**kwargs)
        return problem.status == "infeasible"
    except cp.error.SolverError:
        return True


def parallel_compose(reduce_func, *funcs):
    def func(*args, **kwargs):
        return reduce_func(f(*args, **kwargs) for f in funcs)

    return func


def not_func(func):
    def f(*args, **kwargs):
        return not func(*args, **kwargs)

    return f


def constant(value):
    def f(*args, **kwargs):
        return value

    return f
