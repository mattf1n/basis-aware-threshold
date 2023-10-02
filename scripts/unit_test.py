import src.lp as lp
from src.strats import eta, nucleus, epsilon
from src.parallel import is_in_support
import argparse, sys, json
from functools import partial
from operator import getitem
import torch, pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-xl")
    parser.add_argument(
        "--file", type=argparse.FileType("r"), default=open("data/unit_tests.txt")
    )
    parser.add_argument("--unit", type=int, default=0)
    parser.add_argument("--ba-eta", type=float, default=0.015)
    parser.add_argument("--ba-nuc", type=float, default=0.35)
    parser.add_argument("--ba-eps", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=0.0009)
    parser.add_argument("--nucleus", type=float, default=0.92)
    parser.add_argument("--epsilon", type=float, default=0.0003)
    return parser.parse_args()


def binary_search(sorted_token_ids, is_lower_bound, min_value, max_value):
    center = (min_value + max_value) // 2
    token_id = sorted_token_ids[center]
    print(min_value, center, max_value, is_lower_bound(token_id), file=sys.stderr)
    if max_value - min_value < 3:
        return center
    elif is_lower_bound(token_id):
        return binary_search(sorted_token_ids, is_lower_bound, center, max_value)
    else:
        return binary_search(sorted_token_ids, is_lower_bound, min_value, center)


@torch.inference_mode()
def main():
    args = parse_args()
    model, tokenizer = lp.load_model(args.model)
    with open("config/relcon.json") as f:
        params = json.load(f)[3]["parameters"]
    svd = lp.svd(args.model, 20).cpu().rename(None).numpy()
    for i, text in enumerate(args.file.read().strip().split("\n")):
        print("-" * 45)
        print(text)
        inputs = tokenizer(text, return_tensors="pt")
        logits = model(**inputs).logits[0, -1, :].rename("vocab")
        probs = torch.softmax(logits, dim="vocab").rename(None)
        is_in_eta_support = partial(getitem, eta(params["eta"], logits) <= probs)
        is_in_nucleus_support = partial(
            getitem, nucleus(params["nucleus"], logits) <= probs
        )
        is_in_eps_support = partial(
            getitem, epsilon(params["epsilon"], logits) <= probs
        )
        is_in_ba_eta_support = partial(
            is_in_support,
            svd,
            partial(eta, params["ba_eta"]),
            logits,
            solver="MOSEK",
        )
        is_in_ba_nuc_support = partial(
            is_in_support,
            svd,
            partial(nucleus, params["ba_nucleus"]),
            logits,
            solver="MOSEK",
        )
        is_in_ba_eps_support = partial(
            is_in_support,
            svd,
            partial(epsilon, params["ba_epsilon"]),
            logits,
            solver="MOSEK",
        )
        sorted_token_ids = logits.rename(None).argsort(descending=True)
        sorted_probs = probs[sorted_token_ids]
        pad = 20
        boundary_idxs = [
            torch.arange(len(probs))[sorted_probs < epsilon][0]
            for epsilon in (
                eta(params["eta"], logits),
                # nucleus(params["nucleus"], logits),
                epsilon(params["epsilon"], logits),
            )
        ]
        ba_eta_boundary = binary_search(
            sorted_token_ids, is_in_ba_eta_support, 0, len(logits)
        )
        # ba_nuc_boundary = binary_search(
        #     sorted_token_ids, is_in_ba_nuc_support, 0, len(logits)
        # )
        # ba_eps_boundary = binary_search(
        #     sorted_token_ids, is_in_ba_eps_support, 0, len(logits)
        # )
        idxs = (0, ba_eta_boundary, *boundary_idxs)
        headers = (
            "Rank",
            "Prob",
            "Token",
            "BA Eta",
            "Eta",
            "Epsilon",
            "Nucleus",
        )
        print(("\n{:<8}{:<8}{:<20}" + "{:<4}" * 4).format(*headers))
        row_template = "{:<8}{:<8.4f}{:<20}" + "{:<4}" * 4
        rows = list()
        for j, token_id in enumerate(sorted_token_ids):
            if any(abs(j - idx) <= pad for idx in idxs):
                row = (
                    j,
                    sorted_probs[j].item(),
                    repr(tokenizer.decode(token_id)),
                    is_in_ba_eta_support(token_id),
                    is_in_eta_support(token_id).item(),
                    is_in_eps_support(token_id).item(),
                    is_in_nucleus_support(token_id).item(),
                )
                print(row_template.format(*row), flush=True)
                rows.append(row)
        print("-" * 45)
        pd.DataFrame(rows, columns=headers).to_csv(
            "results/unit/15_{}.csv".format(i), index=False
        )


if __name__ == "__main__":
    main()
