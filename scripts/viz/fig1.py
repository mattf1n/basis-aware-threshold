import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 8,
        "lines.linewidth": 1,
    }
)


df = pd.read_csv("results/unit/0.csv")
ba_eta = df["BA Eta"]

plt.figure(figsize=(5, 2))
for i, method in enumerate(("Nucleus", "Eta", "Epsilon")):
    plt.vlines(
        df[df[method]].index.max(),
        0,
        1,
        label=(
            r"$\epsilon$"
            if method == "Epsilon"
            else r"$\eta$"
            if method == "Eta"
            else method
        )
        + " threshold",
        colors=f"C{i + 2}",
        linestyles="dashed",
        zorder=0,
    )
plt.bar(df[ba_eta].index, df.Prob[ba_eta], label=r"BAT accept")
plt.bar(df[~ba_eta].index, df.Prob[~ba_eta], label=r"BAT reject")
for idx, offset in [
    (0, (3, 0)),
    # (35, (0, 6)),
    # (64, (-1, 2)),
    # (66, (0, 4)),
    # (68, (1, 2)),
    # (80, (0, 2)),
    # (84, (0, 4)),
    (48, (3, 4)),
    (55, (2, 2)),
    (61, (3, 2)),
    (15, (0, 3)),
    (30, (-2, 3)),
    (37, (-4, 5.5)),
    (39, (2, 8)),
]:
    height = df.Prob[idx]
    plt.annotate(
        "\\texttt{{{}}}".format(df.Token[idx][2:-1]),
        (idx, height),
        offset,
        textcoords="offset fontsize",
        arrowprops=dict(arrowstyle="-", linewidth=0.3),
        ha="center",
        color="black" if df["BA Eta"][idx] else "red",
    )
plt.yscale("log")
plt.xlabel("Token (ordered by probability)")
plt.ylabel("Log probability")
plt.title(r"\texttt{Taylor} \rule{1cm}{0.15mm}")
plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1))
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=7)
plt.tight_layout()
plt.savefig("paper/fig/fig1.pgf")
print(df)
