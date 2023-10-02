import pandas as pd
from glob import glob
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 8,
    }
)

with open(
    "data/param_matching/gpt2-xl_constraints_20_relative_conserve_nuc.jsonl"
) as f:
    data = list(map(json.loads, f))

for line in data:
    for sample in line:
        sample["nucleus"] = 1 - sample["nucleus"]
        sample["ba_nucleus"] = 1 - sample["ba_nucleus"]

gold = pd.DataFrame(data=[line[0] for line in data])

method_tuples = [
    ("eta", "ba_eta"),
    ("epsilon", "ba_epsilon"),
    ("nucleus", "ba_nucleus"),
    ("epsilon", "nucleus"),
    ("eta", "epsilon"),
    ("ba_eta", "ba_epsilon"),
]
# fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(5, 3))
# for i, (ax, method_tuple) in enumerate(zip(axs.flatten(), method_tuples)):
#     ordered = pd.DataFrame(
#         {
#             method: gold[method].sort_values().to_list()
#             for method in gold.columns
#             if method in method_tuple
#         }
#     )
#     contrib_rates = []
#     for idx, params in tqdm(list(ordered.iterrows())[:4500:100]):
#         contributions = {method: 0 for method in params.index}
#         for _, *samples in data:
#             done = set()
#             for sample in samples:
#                 contribs = {
#                     method: sample[method] >= param for method, param in params.items()
#                 }
#                 if sum(contribs.values()) == 1:
#                     for method, contrib in contribs.items():
#                         contributions[method] += contrib and method not in done
#                 done |= set(method for method, contrib in contribs.items() if contrib)
#         contrib_rates.append(
#             {method: 100 * c / len(data) for method, c in contributions.items()}
#         )

#     contrib_rates = pd.DataFrame(contrib_rates).rename(
#         columns=dict(
#             eta="Eta",
#             ba_eta="BA eta",
#             epsilon="Epsilon",
#             ba_epsilon="BA epsilon",
#             nucleus="Nucleus",
#             ba_nucleus="BA nucleus",
#         )
#     )
#     contrib_rates.plot(ax=ax)
# fig.supylabel("Divergence rate (%)")
# fig.supxlabel("Rejection rate (%)")
# plt.tight_layout()
# plt.savefig("paper/fig/relcons.pgf")
# plt.clf()

for fname in [
    "data/param_matching/gpt2-xl_constraints_30_relative_conserve_nuc.jsonl",
    "data/param_matching/gpt2_constraints_20_relative_conserve_nuc.jsonl",
    "data/param_matching/gpt2-medium_constraints_20_relative_conserve.jsonl",
    "data/param_matching/gpt2-large_constraints_20_relative_conserve.jsonl",
    "data/param_matching/gpt2_constraints_30_relative_conserve_nuc.jsonl",
    "data/param_matching/gpt2-xl_constraints_20_relative_conserve_nuc.jsonl",
]:
    print(fname)
    with open(fname) as f:
        data = list(map(json.loads, f))

    for line in data:
        for sample in line:
            sample["nucleus"] = 1 - sample["nucleus"]
            sample["ba_nucleus"] = 1 - sample["ba_nucleus"]

    gold = pd.DataFrame(data=[line[0] for line in data])
    ordered = (
        pd.DataFrame(
            {
                method: gold[method].sort_values().to_list()
                for method in gold.columns
                if "topk" not in method
            }
        )
        .assign(nucleus=lambda df: 1 - df.nucleus)
        .assign(ba_nucleus=lambda df: 1 - df.ba_nucleus)
        .rename(
            columns=dict(
                eta="Eta",
                ba_eta="BA eta",
                epsilon="Epsilon",
                ba_epsilon="BA epsilon",
                nucleus="Nucleus",
                ba_nucleus="BA nucleus",
            )
        )
    )

    ordered.iloc[:4500:100].reset_index(drop=True).plot(figsize=(4, 2))
    plt.xlabel("Rejection rate (%)")
    plt.ylabel("Hyperparameter")
    plt.legend(bbox_to_anchor=(1.04, 1))
    plt.tight_layout()
    plt.savefig("paper/fig/rel_params.pgf")

    print(ordered.iloc[2_500])
