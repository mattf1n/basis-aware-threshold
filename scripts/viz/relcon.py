import pandas as pd
import json, os, functools, re
from glob import glob


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
        for fname in glob("results/mauve/scores/test_*_*_seed_*_relcon_mauve.json")
        if "20" in fname
    ]
    for m in matched:
        print(m)
    for fname in matched:
        size = get_size(fname)
        method = get_method(fname)
        seed = int(re.findall(r"seed_(\d)", fname)[0])
        with open(fname) as file:
            mauve = json.load(file)["mauve"]
        if method in ["Eta", "BA eta"]:
            data.append((size, method, mauve, seed))
    df = pd.DataFrame(data, columns=("Size", "Method", "Mauve", "Seed")).assign(
        Mauve=lambda x: x.Mauve * 100
    )
    plot = (
        df[["Size", "Method", "Mauve"]]
        .groupby(["Size", "Method"])
        .agg(Mean=("Mauve", "mean"), SEM=("Mauve", lambda x: x.sem() * 1.96))
        .reset_index()
        .sort_values("Size", key=lambda x: x.map(sizes.index))
    )
    print(plot)
    print(df)
    big_table = df.pivot(columns="Size", index=["Method", "Seed"], values="Mauve")[
        ["Small", "Medium", "Large", "XL"]
    ]
    small_table = (
        big_table.groupby(["Method"])
        .agg(
            lambda x: "${:04.1f}_{{{:.1f}}}$".format(x.mean(), 1.96 * x.sem()).replace(
                "nan", "-"
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
            buf="paper/tab/relcon.tex",
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
