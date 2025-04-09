from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
import jax, cvxpy as cp, numpy as np
from scipy.special import logsumexp, softmax
from tqdm import tqdm
import json, operator, sys, os, itertools, functools, argparse, random, multiprocessing
from basis_aware_sampling.lp import svd
from dataclasses import dataclass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument(
        "--data", type=argparse.FileType("r"), default=open("data/webtext/valid.jsonl")
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--tokens", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--constraints", type=int, default=20)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


program_cache = dict()


@dataclass
class LinearProgram:
    problem: cp.Problem
    probs: cp.Parameter
    embed: cp.Parameter
    onehot: cp.Parameter


def get_program(embed, max_delta=0.9):
    vocab_size, embed_size = embed.shape
    true_probs = cp.Variable(vocab_size, nonneg=True)
    error = cp.Variable(nonneg=True)
    probs = cp.Parameter(vocab_size, nonneg=True)
    embed = cp.Parameter((vocab_size, embed_size))
    constraints = [
        true_probs[0] == 0,
        probs @ embed == true_probs @ embed,
        true_probs <= probs * (1 + error),
        error <= 1 / (1 - max_delta) - 1,
        cp.sum(true_probs) == 1,
    ]
    problem = cp.Problem(cp.Minimize(error), constraints)
    return LinearProgram(problem, probs, embed, None)


def get_param_thresholds(program: LinearProgram, embed, io_pair):
    logits, input_id = io_pair
    probs = softmax(logits)
    lse = logsumexp(logits)
    logprobs = logits - logsumexp(logits)
    square_log_entropy = np.square(np.log(probs @ -logprobs))
    sample = np.random.choice(len(logits), size=3, p=probs, replace=False)
    thresholds = list()
    for token_id in (input_id, *sample):
        prob = probs[token_id]
        param_thesholds = dict(
            epsilon=float(prob),
            eta=float(max(prob, (prob / square_log_entropy))),
            nucleus=float(probs[probs > prob].sum()),
            topk=int((probs >= prob).sum()),
        )
        program.probs.value = np.roll(probs, -token_id)
        program.embed.value = np.roll(embed, -token_id)
        try:
            program.problem.solve(
                solver="MOSEK",
                mosek_params=dict(MSK_IPAR_NUM_THREADS=1),
                ignore_dpp=True,
                warm_start=True,
            )
        except cp.error.SolverError:
            print("Error!", probs[token_id], file=sys.stderr)
        if program.problem.value is None:
            print("Infeasible", file=sys.stderr)
            thresholds.append(
                param_thesholds
                | dict(ba_epsilon=1.0, ba_eta=float("inf"), ba_nucleus=0.0, ba_topk=1)
            )
        else:
            delta = 1 - 1 / (1 + program.problem.value)
            nextmost_token_prob = probs[probs <= delta].max(initial=0.0)
            thresholds.append(
                param_thesholds
                | dict(
                    ba_epsilon=float(delta),
                    ba_eta=float(max(delta, (delta / square_log_entropy))),
                    ba_nucleus=float(probs[probs > delta].sum()),
                    ba_topk=int((probs >= nextmost_token_prob).sum()),
                )
            )
    return thresholds


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = jax.jit(FlaxAutoModelForCausalLM.from_pretrained(args.model))
    embed = svd(args.model, args.constraints).rename(None).cpu().numpy()
    program = get_program(embed)
    get_thresholds = functools.partial(get_param_thresholds, program, embed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    texts = list(json.loads(row)["text"] for row in args.data)
    shuffled_texts = random.sample(texts, len(texts))
    tokenized_batches = list(
        tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        for text in shuffled_texts
    )
    outputs = (model(**batch) for batch in tokenized_batches)
    io_pairs = (
        (np.asarray(logits), input_id)
        for output, tokens in zip(outputs, tokenized_batches)
        for logits_seq, input_ids, mask in zip(
            output.logits, tokens.input_ids, tokens.attention_mask
        )
        for logits, input_id in zip(logits_seq[mask == 1], input_ids[mask == 1][1:])
    )
    with multiprocessing.Pool(processes=args.processes) as pool:
        for i, thresholds in tqdm(
            enumerate(pool.imap(get_thresholds, io_pairs)), total=args.tokens
        ):
            if i > args.tokens:
                break
            json.dump(thresholds, args.out)
            print(file=args.out, flush=True)


if __name__ == "__main__":
    main()
