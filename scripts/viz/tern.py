import matplotlib.pyplot as plt
import mpltern
from mpltern.datasets import get_triangular_grid
import numpy as np
from scipy.special import softmax

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 8,
    }
)

np.random.seed(3)

W = np.random.random((3, 1))

hs = np.stack((np.linspace(-100, 100, 500),))
h = np.random.random((1,)) * 5
hat_ps = softmax(W @ hs, axis=0)
hat_p = softmax(W @ h)

rhs = W.T @ hat_p
p0s = np.linspace(0, 1, 500)
p1s = (rhs - W[2] - (W[0] - W[2]) * p0s) / (W[1] - W[2])
p2s = 1 - p0s - p1s
ps = np.stack((p0s, p1s, p2s))

print(W)
print(h)
print(hat_p)
print(ps[:, 0])

uniform = np.ones(3) / 3

threshold = 1.9
max_p = threshold * hat_p
min_p = 1 - np.roll(max_p, 1) - np.roll(max_p, 2)
trunc = np.array(
    [
        [max_p[0], max_p[0], min_p[0]],
        [max_p[1], min_p[1], max_p[1]],
        [min_p[2], max_p[2], max_p[2]],
    ]
)

plt.figure(figsize=(2.25, 1.75))
ax = plt.subplot(projection="ternary")

# ax.scatter(*uniform, label=r"Uniform", s=8)
ax.plot(*hat_ps, label="Possible outputs", linewidth=1)
ax.scatter(*hat_p, label="Model output", s=8, zorder=10)
ax.plot(*ps, label="BA constraint", linewidth=1)
ax.fill(*trunc, alpha=0.2, label="Log error constraint")
ax.scatter(*(0, 0.7, 0.3), label="$p_1$ might be $0$", zorder=20, s=10)
ax.set_tlabel("$p_1$")
ax.set_llabel("$p_2$")
ax.set_rlabel("$p_3$")
# position = "tick1"
# ax.taxis.set_label_position(position)
# ax.laxis.set_label_position(position)
# ax.raxis.set_label_position(position)
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=7)
plt.tight_layout()
plt.savefig("paper/fig/tern.pgf")
