from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse, json, sys, functools, os, sys, operator, json, math, pickle, time
from functools import partial
from typing import Callable, Optional, Union
import torch, numpy as np, cvxpy as cp, torch.multiprocessing as mp
import src.lp as lp, src.strats as strats
from src.parallel import (
    get_next_token,
    inference,
    is_in_support,
    adverse_sample,
    our_generate,
    truncation_sample,
    get_truncation_probs,
    get_interval_probs,
    get_token_source,
    not_func,
    constant,
    parallel_compose,
)
from tqdm import tqdm
import wandb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=argparse.FileType("r"), default=sys.stdin, help="webtext data"
    )
    parser.add_argument(
        "--out",
        type=argparse.FileType("a"),
        default=sys.stdout,
        help="results output file",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--examples", type=int, default=None, help="number of completions"
    )
    parser.add_argument(
        "--max-length", type=int, default=1024, help="maximum completion length"
    )
    parser.add_argument(
        "--prompt-length", type=int, default=35, help="length of the prompt"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="number of completions from the beginning to skip",
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--constraint-size",
        type=int,
        default=None,
        help="number of constraints in the linear program",
    )
    parser.add_argument("--solver", default="MOSEK", help="solver backend")
    parser.add_argument(
        "--dpp",
        action="store_true",
        help="use dpp compilation for the linear program. Too slow, do not use. ",
    )
    parser.add_argument(
        "--processes", type=int, default=None, help="CPU processes to use"
    )
    parser.add_argument(
        "--threads", type=int, default=16, help="threads per CPU process. Try 1."
    )
    parser.add_argument("--cpu", action="store_true", help="use if no GPU available")
    parser.add_argument(
        "--parallel", action="store_true", help="enables parallel decoding"
    )
    parser.add_argument(
        "--generate",
        type=str,
        default="simple",
        help="generation setting. Only 'simple' used in the paper",
    )
    parser.add_argument("--adverse-rate", type=float, default=0.1, help="ignore")
    parser.add_argument(
        "--retry-limit",
        type=int,
        default=10,
        help="fall back to greedy after rejecting n tokens",
    )
    parser.add_argument(
        "--strat",
        type=strats.strats_dict.get,
        default=strats.eta,
        help="base truncation method, `eta` works best",
    )
    parser.add_argument(
        "--adversary-strat",
        type=strats.strats_dict.get,
        default=strats.eta,
        help="ignore",
    )
    parser.add_argument("--param", type=float, default=1, help="truncation parameter")
    parser.add_argument("--adversary-param", type=float, default=1, help="ignore")
    parser.add_argument(
        "--bottleneck-aware", action="store_true", help="use BA constraints"
    )
    parser.add_argument(
        "--adversary-bottleneck-aware", action="store_true", help="ignore"
    )
    parser.add_argument("--wandb", action="store_true", help="track GPU usage on W&B")
    return parser.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.wandb:
        wandb.login()
        wandb.init()
    resume_file = "/results/generations.jsonl"
    if os.path.exists(resume_file):
        with open(resume_file) as file:
            already_processed = len(file.readlines())
    else:
        already_processed = 0
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    assert mp.get_start_method() == "spawn"
    mp.set_sharing_strategy("file_system")
    data = list(map(json.loads, args.data))[
        args.start : args.examples
        if args.examples is None
        else args.start + args.examples
    ][already_processed:]
    texts = map(operator.itemgetter("text"), data)
    model, tokenizer = lp.load_model(args.model)
    tokenize = partial(
        tokenizer,
        return_tensors="pt",
        truncation=True,
        max_length=args.prompt_length,
    )
    tokens = map(tokenize, texts)
    device = "cpu" if args.cpu else torch.device(0)
    param_device = "cpu" if args.cpu else torch.device(1)
    input_ids = (toks.input_ids for toks in tokens)
    model.to(device)
    model.share_memory()
    forward = partial(inference, model, device=device)
    kwargs = dict(solver="MOSEK", mosek_params={"MSK_IPAR_NUM_THREADS": args.threads})
    svd_embed = lp.svd(args.model, args.constraint_size).cpu().rename(None).numpy()
    get_threshold = partial(args.strat, args.param)
    get_adversary_threshold = partial(args.adversary_strat, args.adversary_param)
    is_in_adversary_support = partial(is_in_support, svd_embed, get_adversary_threshold)
    is_in_support_ = partial(is_in_support, svd_embed, get_threshold)
    simple_sample = partial(
        partial(get_next_token, is_in_support_)
        if args.bottleneck_aware
        else truncation_sample,
    )
    get_lower_bound = (
        partial(strats.epsilon, 0) if args.bottleneck_aware else get_threshold
    )
    is_adverse_token = parallel_compose(
        all,
        is_in_support_ if args.bottleneck_aware else constant(True),
        not_func(is_in_adversary_support)
        if args.adversary_bottleneck_aware
        else constant(True),
    )
    if args.generate == "adversarial":
        sample = partial(
            adverse_sample,
            args.adverse_rate,
            partial(get_interval_probs, get_lower_bound, get_adversary_threshold),
            is_adverse_token,
            simple_sample,
            retry_limit=args.retry_limit,
        )
    else:
        sample = simple_sample
    sample_state = 0 if args.generate == "adversarial" else None
    generate = partial(
        our_generate,
        tokenizer,
        partial(get_token_source, get_threshold, get_adversary_threshold),
        forward,
        (
            get_threshold
            if args.generate == "simple"
            else parallel_compose(min, get_threshold, get_adversary_threshold)
        ),
        sample,
        sample_state,
        max_length=args.max_length,
        parallel=args.parallel,
        verbose=args.verbose,
    )
    with mp.Pool(processes=args.processes, maxtasksperchild=50) as pool:
        map_func = pool.imap if args.parallel else map
        for generation, sources in tqdm(map_func(generate, input_ids), total=len(data)):
            json.dump(dict(generation=generation, sources=sources), args.out)
            print(file=args.out, flush=True)


if __name__ == "__main__":
    main()
