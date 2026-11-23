"""Figuras novas do relatorio (a partir dos CSVs de <entrega>/dados)."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E = Path(sys.argv[1])
D = E / "dados"
F = E / "figuras_extra"
F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 10.5,
                     "axes.labelsize": 10, "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight"})
# Okabe-Ito (seguro para daltonismo) + marcador distinto por metodo (codificacao secundaria)
COR = dict(laplace="#E69F00", metropolis="#0072B2", ensemble="#009E73", mala="#D55E00", hmc="#CC79A7", npe="#56B4E9", npe_pfn="#000000")
MK = dict(laplace="s", metropolis="o", ensemble="^", mala="D", hmc="v", npe="P", npe_pfn="X")
NOME = dict(laplace="Laplace", metropolis="Metropolis", ensemble="Ensemble", mala="MALA", hmc="HMC", npe="NPE", npe_pfn="NPE-PFN")
TXT, GRID = "#222222", "#DDDDDD"

# ------------------------------------------------------------ robustez
nu = pd.read_csv(D / "15_piso_de_ruido_c2st.csv")
mm = pd.read_csv(D / "03_resumo_modelo_metodo.csv")
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw=dict(width_ratios=[1.15, 1]))
a = ax[0]
a.axhline(0.5, color="#888888", lw=0.8, ls="--")
a.scatter(nu.d, nu.c2st_nulo_media, s=26, color=COR["metropolis"], edgecolor="white", linewidth=0.8, zorder=3)
top = nu.sort_values("c2st_nulo_media").tail(2)
for _, r in top.iterrows():
    a.annotate(r.rotulo, (r.d, r.c2st_nulo_media), xytext=(6, 3), textcoords="offset points", fontsize=8, color=TXT)
a.set_xlabel("dimensão do problema $d$")
a.set_ylabel("C2ST da referência contra ela mesma")
a.set_ylim(0.48, 0.54)
a.set_title("(a) Piso de ruído do C2ST (1000 + 1000 amostras)", loc="left")
a.grid(axis="y", color=GRID, lw=0.6)
b = ax[1]
ordem = ["laplace", "metropolis", "ensemble", "mala", "hmc", "npe"]
data = [mm[mm.metodo == m].c2st_dp.dropna().values for m in ordem]
bp = b.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
               medianprops=dict(color="black", lw=1.4), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
for p, m in zip(bp["boxes"], ordem):
    p.set_facecolor(COR[m])
    p.set_alpha(0.75)
    p.set_edgecolor("#333333")
b.set_yticks(range(1, 7))
b.set_yticklabels([NOME[m] for m in ordem])
b.invert_yaxis()
b.set_xlabel("desvio-padrão do C2ST entre as 3 sementes")
b.set_title("(b) Variação entre sementes, por modelo", loc="left")
b.grid(axis="x", color=GRID, lw=0.6)
fig.tight_layout()
fig.savefig(F / "fig_robustez.png")
fig.savefig(F / "fig_robustez.pdf")
plt.close(fig)

# ------------------------------------------------------------ NPE-PFN contra NPE
s0 = pd.read_csv(D / "08_npe_pfn_piloto_semente0.csv")
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0.5, 1.0], [0.5, 1.0], color="#888888", lw=0.8, ls="--")
cls = {"EDO": "o", "EDO difícil": "s", "EDP": "^", "EDP difícil": "D"}
for c, mk in cls.items():
    g = s0[s0.classe == c]
    ax.scatter(g.npe, g.npe_pfn, marker=mk, s=40, color=COR["npe_pfn"], facecolor="none" if c.endswith("difícil") else COR["npe_pfn"],
               linewidth=1.2, label=c, zorder=3)
for _, r in s0.iterrows():
    if abs(r.npe - r.npe_pfn) > 0.12:
        ax.annotate(r.rotulo, (r.npe, r.npe_pfn), xytext=(5, -9 if r.npe_pfn < r.npe else 4), textcoords="offset points", fontsize=7.5, color=TXT)
ax.set_xlabel("C2ST do NPE (semente 0)")
ax.set_ylabel("C2ST do NPE-PFN (semente 0)")
ax.set_xlim(0.5, 1.02)
ax.set_ylim(0.5, 1.02)
ax.text(0.99, 0.52, "abaixo da diagonal: NPE-PFN melhor", ha="right", fontsize=8, color="#555555")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(color=GRID, lw=0.6)
ax.set_title("NPE-PFN contra NPE em 18 modelos", loc="left")
fig.tight_layout()
fig.savefig(F / "fig_pfn_vs_npe.png")
fig.savefig(F / "fig_pfn_vs_npe.pdf")
plt.close(fig)

# ------------------------------------------------------------ distribuicao do C2ST por classe e metodo (pontos = modelos)
cl = ["EDO", "EDO difícil", "EDP", "EDP difícil", "PEtab", "PEtab grande"]
fig, axs = plt.subplots(1, 6, figsize=(10.2, 3.3), sharey=True)
rng = np.random.default_rng(1)
for ax, c in zip(axs, cl):
    g = mm[(mm.classe == c) & mm.metodo.isin(ordem)]
    ax.axhline(0.5, color="#888888", lw=0.7, ls="--")
    for i, m in enumerate(ordem):
        v = g[g.metodo == m].c2st_media.values
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=11, color=COR[m], marker=MK[m], alpha=0.85, linewidth=0, zorder=3)
        ax.hlines(np.median(v), i - 0.32, i + 0.32, color="black", lw=1.4, zorder=4)
    ax.set_title(f"{c} ({g.problema.nunique()})", fontsize=9)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["Lap", "MH", "Ens", "MALA", "HMC", "NPE"], rotation=60, fontsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.5)
axs[0].set_ylabel("C2ST (média das 3 sementes)")
axs[0].set_ylim(0.47, 1.03)
fig.tight_layout()
fig.savefig(F / "fig_c2st_pontos_classe.png")
fig.savefig(F / "fig_c2st_pontos_classe.pdf")
plt.close(fig)
print("figuras ok")
