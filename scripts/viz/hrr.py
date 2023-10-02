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
        "BA eta",
        "Eta",
        "BA epsilon",
        "Epsilon",
        "BA nucleus",
        "Nucleus",
    ]
    sizes = ["Small", "Medium", "Large", "XL"]
    matched = [
        fname
        for fname in glob(
            "results/mauve/scores/test_gpt2-xl_*_seed_*_relcon_*mauve.json"
        )
        if "20" in fname and "incorrect" not in fname
    ]
    for m in matched:
        print(m)
    for fname in matched:
        size = get_size(fname)
        method = get_method(fname)
        seed = int(re.findall(r"seed_(\d)", fname)[0])
        if not re.findall(r"relcon_(\d+)", fname):
            hrr = 25
        else:
            hrr = int(re.findall(r"relcon_(\d+)", fname)[0])
        with open(fname) as file:
            mauve = json.load(file)["mauve"]
        if "eta" in fname:
            data.append((hrr, method, mauve, seed))
    df = pd.DataFrame(data, columns=("HRR", "Method", "Mauve", "Seed")).assign(
        Mauve=lambda x: x.Mauve * 100
    )
    plot = (
        df[["HRR", "Method", "Mauve"]]
        .groupby(["HRR", "Method"])
        .agg(Mean=("Mauve", "mean"), SEM=("Mauve", lambda x: x.std() * 1.96))
        .reset_index()
        # .sort_values("HRR", key=lambda x: x.map(sizes.index))
    )
    print(plot)
    print(df)
    big_table = df.pivot(columns="HRR", index=["Method", "Seed"], values="Mauve")
    print(big_table)
    a = big_table.groupby(["Method"]).agg(["mean", "std"]).stack().T
    print(a)
    plt.figure(figsize=(2.5, 2))
    for method in ["Eta", "BA eta"]:
        mean = a[(method, "mean")]
        std = a[(method, "std")]
        plt.plot(
            [0.0001, 0.0004, 0.001, 0.005, 0.02],
            mean,
            label=r"$\eta$" if method == "Eta" else r"BA-$\eta$",
            marker=".",
        )
        plt.fill_between(
            [0.0001, 0.0004, 0.001, 0.005, 0.02], mean - std, mean + std, alpha=0.3
        )
    plt.xscale("log")
    plt.xlabel(r"$\eta$")
    plt.ylabel("MAUVE")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("paper/fig/hrr.pgf")
    small_table = (
        big_table.groupby(["Method"])
        .agg(
            lambda x: "${:04.1f}_{{{:.1f}}}$".format(x.mean(), x.std()).replace(
                # lambda x: "${:04.1f}$".format(x.mean()).replace(
                "nan",
                "-",
            )
        )
        .sort_index(key=lambda x: x.map(methods.index), level=0)
    )
    print(big_table)
    print(small_table)
    style = (
        small_table.style.format(na_rep="-")
        .highlight_max(axis="index", props="boldmath: ;")
        .to_latex(
            buf="paper/tab/hrr.tex",
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
    if "ba_" not in fname:
        return method.capitalize()
    else:
        return "BA " + method


if __name__ == "__main__":
    main()
