import pandas as pd
import json, os, functools, re
from glob import glob


# results/mauve/scores/test_gpt2-large_20_eta_seed_0_mauve.json
# results/mauve/scores/test_gpt2-large_bottleneck-aware_mauve.json
# results/mauve/scores/test_gpt2-large_bottleneck-aware_seed_1_mauve.json


def main():
    data = list()
    methods = [
        "Bottleneck-aware",
        "Eta",
        "BA eta",
        "Epsilon",
        "BA epsilon",
        "Nucleus",
        "BA nucleus",
    ]
    sizes = ["Small", "Medium", "Large", "XL"]
    matched = [
        fname
        for fname in glob("results/mauve/scores/test_*_*_seed_*_mauve.json")
        if ("20" in fname or "30" in fname) and "relcon" not in fname
    ]
    for m in matched:
        print(m)
    for fname in matched:
        size = get_size(fname)
        method = "BA " + get_method(fname).lower()
        seed = 0 if "seed" not in fname else int(re.findall(r"seed_(\d)", fname)[0])
        constraints = 20 if "20" in fname else 30
        with open(fname) as file:
            mauve = json.load(file)["mauve"]
        if constraints == 20:
            data.append((size, method, mauve, seed, constraints))
    for fname in glob("results/mauve/scores/test_*_*_mauve.json"):
        if fname not in matched and "relcon" not in fname:
            size = get_size(fname)
            method = get_method(fname)
            seed = 0 if "seed" not in fname else int(re.findall(r"seed_(\d)", fname)[0])
            constraints = "-"
            with open(fname) as file:
                mauve = json.load(file)["mauve"]
            # if seed <= 1:
            # if seed <= 4 and "bottleneck-aware" not in fname:
            data.append((size, method, mauve, seed, constraints))
    df = pd.DataFrame(
        data, columns=("Size", "Method", "Mauve", "Seed", "Constraints")
    ).assign(Mauve=lambda x: x.Mauve * 100)
    plot = (
        df[["Size", "Method", "Mauve", "Constraints"]]
        .reset_index()
        .reset_index()
        .groupby(["Size", "Method", "Constraints"])
        .agg(Mean=("Mauve", "mean"), SEM=("Mauve", lambda x: x.sem() * 1.96))
        .reset_index()
        .sort_values("Size", key=lambda x: x.map(sizes.index))
    )
    print(plot)
    for method in methods:
        plot.loc[(plot.Method == method) & (plot.Constraints.isin((20, "-")))].to_csv(
            f"paper/data/test/{method}.csv", index=False
        )
    big_table = df.pivot(
        columns="Size", index=["Method", "Seed", "Constraints"], values="Mauve"
    )[["Small", "Medium", "Large", "XL"]]
    print(big_table)
    small_table = (
        big_table.groupby(["Method", "Constraints"])
        .agg(
            lambda x: "${:.1f}_{{{:.1f}}}$".format(x.mean(), 1.96 * x.sem()).replace(
                "nan", "-"
            )
        )
        .sort_index(key=lambda x: x.map(methods.index), level=0)
        .droplevel(1)
    )
    print(big_table)
    print(small_table)
    style = (
        small_table.style.format(na_rep="-")
        .highlight_max(axis="index", props="boldmath: ;")
        .to_latex(
            buf="paper/tab/test.tex",
            hrules=True,
            column_format="lrrrrr",
            multirow_align="t",
        )
    )


def get_size(fname: str):
    size: str = re.findall(r"gpt2-*(.*?)_", fname)[0]
    if size == "xl":
        return size.upper()
    elif size == "":
        return "Small"
    else:
        return size.capitalize()


def get_method(fname: str):
    method = re.findall(r"(epsilon|bottleneck-aware|eta|nucleus)", fname)[0]
    return method.capitalize() if "ba_" not in fname else "BA " + method


if __name__ == "__main__":
    main()
