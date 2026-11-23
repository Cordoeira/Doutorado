"""Figura de tracos dos dois modelos multimodais (PK com flip-flop e cadeia de decaimento).

Uso: python figura_tracos.py <resultados_a0> <entrega>
Le <resultados_a0>/<modelo>/cadeias/*.npz e grava <entrega>/figuras/fig_tracos_multimodais.{png,pdf}.
Imprime, por modelo e metodo, a fracao das amostras (sem aquecimento) em cada modo, por cadeia.
"""
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R, E = Path(sys.argv[1]), Path(sys.argv[2])
F = E / "figuras"
F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight"})
CORES = ["#0072B2", "#E69F00", "#009E73"]  # Okabe-Ito, uma por cadeia
METODOS = [("referencia", "Referência"), ("metropolis", "Metropolis"), ("ensemble", "Ensemble"), ("mala", "MALA"), ("hmc", "HMC")]
PERMS = {p: i for i, p in enumerate(itertools.permutations(range(3)))}
# (pasta, titulo, indice do parametro mostrado, rotulo do eixo, numero de modos, funcao que rotula o modo de cada amostra)
MODELOS = [
    ("pk_flipflop", "PK com flip-flop", 0, r"$\log k_a$", 2, lambda X: (X[:, 0] > X[:, 1]).astype(int)),
    ("cadeia_simetrica", "Cadeia de decaimento", 2, r"$k_3$", 6,
     lambda X: np.array([PERMS[tuple(o)] for o in np.argsort(X, axis=1)])),
]


def virg(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


def dens(x, g):
    s = np.std(x)
    if s == 0 or len(x) < 2:
        return np.zeros_like(g)
    h = 1.06 * s * len(x) ** -0.2
    z = (g[:, None] - x[None, ::max(1, len(x) // 4000)]) / h
    return np.exp(-0.5 * z ** 2).mean(axis=1) / (h * np.sqrt(2 * np.pi))


fig = plt.figure(figsize=(9.4, 8.2))
gs = fig.add_gridspec(len(METODOS), 5, width_ratios=[3.2, 0.75, 0.42, 3.2, 0.75], wspace=0.04, hspace=0.42)
for col, (pasta, titulo, j, rot, K, modo) in enumerate(MODELOS):
    cad = {m: np.load(R / pasta / "cadeias" / f"{m}.npz") for m, _ in METODOS}
    ref = cad["referencia"]
    q = int(ref["aquecimento"]) // int(ref["desbaste"])
    rpos = ref["cadeia"][:, q:].reshape(-1, 3)
    lo, hi = np.quantile(rpos[:, j], [0.001, 0.999])
    lo, hi = lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo)
    g = np.linspace(lo, hi, 200)
    dref = dens(rpos[:, j], g)
    print(f"== {pasta} ({str(ref['nomes'][j])})")
    for lin, (m, nome) in enumerate(METODOS):
        c = cad[m]
        X, k = c["cadeia"], int(c["desbaste"])
        q = int(c["aquecimento"]) // k
        C, n = X.shape[0], X.shape[1]
        ax = fig.add_subplot(gs[lin, 3 * col])
        ad = fig.add_subplot(gs[lin, 3 * col + 1], sharey=ax)
        ad.fill_betweenx(g, 0, dref, color="0.80", lw=0)
        visitados = np.zeros(K, bool)
        fr = []
        passo = max(1, n // 1500)
        t = np.arange(n)[::passo] / n
        for cc in range(C):
            serie = X[cc, :, 0, j] if X.ndim == 4 else X[cc, :, j]  # ensemble: um caminhante por cadeia
            ax.plot(t, serie[::passo], lw=0.55, color=CORES[cc], alpha=0.9, rasterized=True)
            pos = X[cc, q:].reshape(-1, X.shape[-1])
            ad.plot(dens(pos[:, j], g), g, color=CORES[cc], lw=0.9)
            f = np.bincount(modo(pos), minlength=K) / len(pos)
            visitados |= f >= 0.01
            fr.append(" ".join(f"{v:.2f}" for v in f))
        print(f"  {m:10s} " + " | ".join(fr))
        ax.axvspan(0, q / n, color="0.93", lw=0, zorder=0)
        ax.set(xlim=(0, 1), ylim=(lo, hi))
        ad.set(xlim=(0, None), xticks=[])
        ad.tick_params(labelleft=False)
        ad.spines["bottom"].set_visible(False)
        ax.text(0.0, 1.03, nome, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")
        ax.text(1.0, 1.03, rf"$\hat R$ máx. {virg(float(np.max(c['rhat'])))}    modos visitados: {int(visitados.sum())} de {K}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8, color="#333333")
        if lin < len(METODOS) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("fração da execução")
        ax.set_ylabel(rot)
        if lin == 0:
            ax.text(0.0, 1.32, titulo, transform=ax.transAxes, ha="left", va="bottom", fontsize=11)
fig.savefig(F / "fig_tracos_multimodais.png")
fig.savefig(F / "fig_tracos_multimodais.pdf")
print("ok", F / "fig_tracos_multimodais.png")
