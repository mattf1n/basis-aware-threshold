import pandas as pd, matplotlib.pyplot as plt
import json, os, functools, re
from glob import glob

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 8,
    }
)


def main():
    data = list()
    methods = [
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
        for fname in glob(
            "results/mauve/scores/test_gpt2-xl_*_seed_*_relcon_mauve.json"
        )
    ]
    for m in matched:
        print(m)
    for fname in matched:
        size = get_size(fname)
        method = get_method(fname)
        seed = int(re.findall(r"seed_(\d)", fname)[0])
        constraints = (
            20 if "20" in fname and "ba" in fname else 30 if "30" in fname else 0
        )
        with open(fname) as file:
            mauve = json.load(file)["mauve"]
        if method == "Eta":
            data.append((method, mauve, seed, constraints))
    df = pd.DataFrame(data, columns=("Method", "Mauve", "Seed", "Constraints")).assign(
        Mauve=lambda x: x.Mauve * 100
    )
    plot = (
        df[["Constraints", "Method", "Mauve"]]
        .groupby(["Constraints", "Method"])
        .agg(Mean=("Mauve", "mean"), SEM=("Mauve", lambda x: x.sem() * 1.96))
        .reset_index()
    )
    print(plot)
    print(df)
    big_table = df.pivot(
        columns="Constraints", index=["Method", "Seed"], values="Mauve"
    ).filter([20, 30, 0])
    print(big_table)
    small_table = (
        big_table.groupby(["Method"])
        .agg(
            lambda x: "${:.1f}_{{{:.1f}}}$".format(x.mean(), 1.96 * x.sem()).replace(
                "nan", "-"
            )
        )
        .sort_index(key=lambda x: x.map(methods.index), level=0)
    )
    print(big_table)
    print(small_table.T)
    a = big_table.groupby(["Method"])
    mean = a.mean().T.Eta
    std = a.std().T.Eta
    print(mean)
    plt.figure(figsize=(2.5, 2))
    plt.plot(mean, label="Mean", marker=".")
    plt.fill_between(
        mean.index, mean - std, mean + std, alpha=0.1, label="Standard deviation"
    )
    plt.xlabel("Constraints")
    plt.ylabel("MAUVE")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("paper/fig/constraints.pgf")
    style = small_table.style.format(na_rep="-").to_latex(
        buf="paper/tab/relcon_by_constraints.tex",
        hrules=True,
        column_format="lrrrrr",
        multirow_align="t",
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
    return method.capitalize()


if __name__ == "__main__":
    main()
