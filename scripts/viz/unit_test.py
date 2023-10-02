import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from textwrap import wrap

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 8,
        "lines.linewidth": 1,
    }
)

with open("data/unit_tests.txt") as f:
    prefixes = list(map(str.strip, f.readlines()))
print(prefixes)

fig = plt.figure(figsize=(10, 6), constrained_layout=True)
subfigs = fig.subfigures(nrows=2, ncols=3, squeeze=True)

i = 0
ax = subfigs[0, 0].subplots(1, 1)
fname = f"results/unit/{i + 1}.csv"
df = pd.read_csv(fname)[:10]
ba_eta = df["BA Eta"]
for j, method in enumerate(("Nucleus", "Eta", "Epsilon")):
    ax.vlines(
        df[df[method]].index.max() - 0.25 + 0.25 * j,
        0,
        2,
        colors=f"C{j + 2}",
        linestyles="dashed",
        zorder=0,
    )
print(df[ba_eta].Rank)
ax.bar(
    df[ba_eta].Rank,
    df.Prob[ba_eta],
)
ax.bar(
    df[~ba_eta].Rank,
    df.Prob[~ba_eta],
)
for idx, offset in [
    (0, (6, 0)),
    (3, (3, 3)),
    (4, (3, 3)),
    (6, (2, 3)),
]:
    height = df.Prob[idx]
    ax.annotate(
        "\\texttt{{{}}}".format(df.Token[idx][1:-1].strip()),
        (idx, height),
        offset,
        textcoords="offset fontsize",
        color="black" if df["BA Eta"][idx] else "red",
        arrowprops=dict(arrowstyle="-", linewidth=0.3),
    )
ax.set_title(
    "\n".join(map("\\texttt{{{}}}".format, wrap(prefixes[i + 1][13:], 40)))
    + r" \rule{1cm}{0.15mm}",
    fontfamily="monospace",
)
# ax.set_title("\\texttt{{}}".format(prefixes[i + 1]) + r"\rule{1cm}{0.15mm}")
ax.set_yscale("log")

i, subfig = 1, subfigs[0, 1]
axes = subfig.subplots(1, 2)
subfig.subplots_adjust(wspace=0, right=0.3)
fname = f"results/unit/{i + 1}.csv"
df = pd.read_csv(fname)
df1, df2 = df[df.Rank < 1000], df[df.Rank > 1000][df.Rank < 1190]
print(df1)
for ax, df in zip(axes.flatten(), (df1, df2)):
    ba_eta = df["BA Eta"]
    for j, method in enumerate(("Nucleus", "Eta", "Epsilon")):
        if not df[method].all() and df[method].any():
            ax.vlines(
                df[df[method]].Rank.max() - 0.25 + 0.25 * j,
                0,
                1e-2,
                colors=f"C{j + 2}",
                linestyles="dashed",
                zorder=0,
            )
    ax.bar(
        df[ba_eta].Rank,
        df.Prob[ba_eta],
    )
    ax.bar(
        df[~ba_eta].Rank,
        df.Prob[~ba_eta],
    )
    ax.set_yscale("log")
axes[0].vlines(
    -1,
    0,
    1e-2,
    colors=f"C4",
    linestyles="dashed",
    zorder=0,
)
axes[1].vlines(
    df2.Rank.min() - 1,
    0,
    1e-2,
    colors=f"C2",
    linestyles="dashed",
    zorder=0,
)
print(df2)
for idx, offset, df, ax in [
    (0, (6, 0), df1, axes[0]),
    (3, (3, 0), df1, axes[0]),
    (23, (0, 15), df2, axes[1]),
    (33, (-1, 3), df2, axes[1]),
    (38, (-3, 8), df2, axes[1]),
    (41, (-2, 6), df2, axes[1]),
    (47, (-2, 4), df2, axes[1]),
]:
    height = df.Prob[idx]
    ax.annotate(
        "\\texttt{{{}}}".format(df.Token[idx][1:-1].strip()),
        (df.Rank[idx], height),
        offset,
        color="black" if df["BA Eta"][idx] else "red",
        textcoords="offset fontsize",
        arrowprops=dict(arrowstyle="-", linewidth=0.3),
    )
axes[0].spines.right.set_visible(False)
axes[1].spines.left.set_visible(False)
axes[1].yaxis.set_visible(False)
d = 1  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(0.2, -d), (-0.2, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
axes[0].plot([1, 1], [1, 0], transform=axes[0].transAxes, **kwargs)
axes[1].plot([0, 0], [1, 0], transform=axes[1].transAxes, **kwargs)
subfig.suptitle(
    "\n".join(map("\\texttt{{{}}}".format, wrap(prefixes[i + 1][13:], 40)))
    + r" \rule{1cm}{0.15mm}",
    fontfamily="monospace",
)
# ax.set_title("\\texttt{{}}".format(prefixes[i + 1]) + r"\rule{1cm}{0.15mm}")


i = 2
ax = subfigs[0, 2].subplots(1, 1)
fname = f"results/unit/{i + 1}.csv"
df = pd.read_csv(fname)[:10]
ba_eta = df["BA Eta"]
for j, method in enumerate(("Nucleus", "Eta", "Epsilon")):
    ax.vlines(
        df[df[method]].index.max() - 0.25 + 0.25 * j,
        0,
        3,
        colors=f"C{j + 2}",
        linestyles="dashed",
        zorder=0,
    )
print(df[ba_eta].Rank)
ax.bar(
    df[ba_eta].Rank,
    df.Prob[ba_eta],
)
ax.bar(
    df[~ba_eta].Rank,
    df.Prob[~ba_eta],
)
for idx, offset in [
    (0, (6, 0)),
    (1, (3, 3)),
    (2, (3, 3)),
    (3, (3, 3)),
    (4, (3, 3)),
    (6, (3, 3)),
]:
    height = df.Prob[idx]
    ax.annotate(
        "\\texttt{{{}}}".format(df.Token[idx][1:-1].strip()),
        (idx, height),
        offset,
        color="black" if df["BA Eta"][idx] else "red",
        textcoords="offset fontsize",
        arrowprops=dict(arrowstyle="-", linewidth=0.3),
    )
print(prefixes[i + 1])
ax.set_title(
    "\n".join(map("\\texttt{{{}}}".format, wrap(prefixes[i + 1][13:], 40)))
    + r" \rule{1cm}{0.15mm}",
    fontfamily="monospace",
)
# ax.set_title("\\texttt{{}}".format(prefixes[i + 1]) + r"\rule{1cm}{0.15mm}")
ax.set_yscale("log")

i = 3
ax = subfigs[1, 0].subplots(1, 1)
fname = f"results/unit/{i + 1}.csv"
df = pd.read_csv(fname)[:10]
ba_eta = df["BA Eta"]
for j, method in enumerate(("Nucleus", "Eta", "Epsilon")):
    ax.vlines(
        df[df[method]].index.max() - 0.25 + 0.25 * j,
        0,
        3,
        colors=f"C{j + 2}",
        linestyles="dashed",
        zorder=0,
    )
print(df[ba_eta].Rank)
ax.bar(
    df[ba_eta].Rank,
    df.Prob[ba_eta],
)
ax.bar(
    df[~ba_eta].Rank,
    df.Prob[~ba_eta],
)
for idx, offset in [
    (0, (6, 0)),
    (1, (3, 3)),
    (2, (3, 3)),
    (4, (3, 3)),
    (6, (3, 3)),
    (7, (3, 3)),
]:
    height = df.Prob[idx]
    ax.annotate(
        "\\texttt{{{}}}".format(df.Token[idx][1:-1].strip()),
        (idx, height),
        offset,
        color="black" if df["BA Eta"][idx] else "red",
        textcoords="offset fontsize",
        arrowprops=dict(arrowstyle="-", linewidth=0.3),
    )
print(prefixes[i + 1])
ax.set_title(
    "\n".join(map("\\texttt{{{}}}".format, wrap(prefixes[i + 1][13:], 40)))
    + r" \rule{1cm}{0.15mm}",
    fontfamily="monospace",
)
# ax.set_title("\\texttt{{}}".format(prefixes[i + 1]) + r"\rule{1cm}{0.15mm}")
ax.set_yscale("log")


i, subfig = 4, subfigs[1, 1]
axes = subfig.subplots(1, 3)
subfig.subplots_adjust(wspace=0, right=0.3)
fname = f"results/unit/{i + 1}.csv"
df = pd.read_csv(fname)
df1, df2, df3 = (
    df[df.Rank < 10],
    df[df.Rank > 4351][df.Rank < 4361],
    df[df.Rank > 5000][df.Rank < 5210],
)
print(df1)
for ax, df in zip(axes.flatten(), (df1, df2, df3)):
    ba_eta = df["BA Eta"]
    for j, method in enumerate(("Nucleus", "Eta", "Epsilon")):
        if not df[method].all() and df[method].any():
            ax.vlines(
                df[df[method]].Rank.max() - 0.25 + 0.25 * j,
                0,
                1e-2,
                colors=f"C{j + 2}",
                linestyles="dashed",
                zorder=0,
            )
    ax.bar(
        df[ba_eta].Rank,
        df.Prob[ba_eta],
    )
    ax.bar(
        df[~ba_eta].Rank,
        df.Prob[~ba_eta],
    )
    ax.set_yscale("log")
axes[0].vlines(
    -1,
    0,
    1e-2,
    colors=f"C4",
    linestyles="dashed",
    zorder=0,
)
axes[1].vlines(
    df2.Rank.min() - 1,
    0,
    1e-2,
    colors=f"C2",
    linestyles="dashed",
    zorder=0,
)
axes[2].vlines(
    5200,
    0,
    1e-2,
    colors=[0, 0, 0, 0],
    linestyles="dashed",
    zorder=0,
)
print(df2)
print(df3)
for idx, offset, df, ax in [
    (0, (1, 0), df1, axes[0]),
    (3, (1, 0), df1, axes[0]),
    (36, (-0.5, 15), df2, axes[1]),
    (41, (-2, 4), df2, axes[1]),
    (63, (0, 3), df3, axes[2]),
    (71, (0, 2), df3, axes[2]),
]:
    height = df.Prob[idx]
    ax.annotate(
        "\\texttt{{{}}}".format(df.Token[idx][1:-1].strip()),
        (df.Rank[idx], height),
        offset,
        textcoords="offset fontsize",
        arrowprops=dict(arrowstyle="-", linewidth=0.3),
        backgroundcolor="white",
        color="black" if df["BA Eta"][idx] else "red",
    )
axes[0].spines.right.set_visible(False)
axes[1].spines.right.set_visible(False)
axes[1].spines.left.set_visible(False)
axes[2].spines.left.set_visible(False)
axes[1].yaxis.set_visible(False)
axes[2].yaxis.set_visible(False)
d = 1  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(0.2, -d), (-0.2, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
axes[0].plot([1, 1], [1, 0], transform=axes[0].transAxes, **kwargs)
axes[1].plot([0, 0, 1, 1], [1, 0, 1, 0], transform=axes[1].transAxes, **kwargs)
axes[2].plot([0, 0], [1, 0], transform=axes[2].transAxes, **kwargs)
subfig.suptitle(
    "\n".join(map("\\texttt{{{}}}".format, wrap(prefixes[i + 1][13:], 40)))
    + r" \rule{1cm}{0.15mm}",
    fontfamily="monospace",
)
# ax.set_title("\\texttt{{}}".format(prefixes[i + 1]) + r"\rule{1cm}{0.15mm}")


i = 5
ax = subfigs[1, 2].subplots(1, 1)
fname = f"results/unit/{i + 1}.csv"
df = pd.read_csv(fname)[:10]
ba_eta = df["BA Eta"]
for j, method in enumerate(("Nucleus", "Eta", "Epsilon")):
    ax.vlines(
        df[df[method]].index.max() - 0.25 + 0.25 * j,
        0,
        3,
        label=method + " threshold",
        colors=f"C{j + 2}",
        linestyles="dashed",
        zorder=0,
    )
print(df[ba_eta].Rank)
ax.bar(df[ba_eta].Rank, df.Prob[ba_eta], label=r"BA-$\eta$ accept")
ax.bar(df[~ba_eta].Rank, df.Prob[~ba_eta], label=r"BA-$\eta$ reject")
for idx, offset in [
    (0, (6, 0)),
    # (1, (3, 3)),
    (2, (3, 3)),
    # (3, (3, 3)),
    (4, (3, 3)),
    (5, (3, 3)),
    (6, (3, 3)),
]:
    height = df.Prob[idx]
    ax.annotate(
        "\\texttt{{{}}}".format(df.Token[idx][1:-1].strip()),
        (idx, height),
        offset,
        color="black" if df["BA Eta"][idx] else "red",
        textcoords="offset fontsize",
        arrowprops=dict(arrowstyle="-", linewidth=0.3),
    )
print(prefixes[i + 1])
ax.set_title(
    "\n".join(map("\\texttt{{{}}}".format, wrap(prefixes[i + 1][13:], 40)))
    + r" \rule{1cm}{0.15mm}",
    fontfamily="monospace",
)
# ax.set_title("\\texttt{{}}".format(prefixes[i + 1]) + r"\rule{1cm}{0.15mm}")
ax.set_yscale("log")


fig.supxlabel("Token (ordered by probability)")
fig.supylabel("Log probability")
plt.legend()
# plt.tight_layout()
# plt.show()
plt.savefig("paper/fig/unit_tests.pdf")

# for idx, offset in [
#     (0, (2, 2)),
#     (15, (0, 3)),
#     (30, (-2, 3)),
#     (37, (-6, 8)),
#     (39, (2, 8)),
#     (40, (3, 6)),
#     (48, (3, 4)),
#     (55, (3, 2)),
#     (61, (3, 2)),
# ]:
#     height = df.Prob[idx]
#     plt.annotate(
#         "\\texttt{{{}}}".format(df.Token[idx][2:-1]),
#         (idx, height),
#         offset,
#         textcoords="offset fontsize",
#         arrowprops=dict(arrowstyle="-", linewidth=0.3),
#     )
