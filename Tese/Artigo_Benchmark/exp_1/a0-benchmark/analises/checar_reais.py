import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a0bench import config, modelos, problemas

cfg = config.montar("completo")
fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for ax, spec in zip(axs.ravel(), modelos.REAIS):
    p = problemas.Sintetico(spec)
    m = p.mapa(cfg)
    x, C = m["x"], m["cov"]
    sd = np.sqrt(np.maximum(np.diag(C), 0))
    print(f"\n{p.nome}  d={p.d} n={len(p.x_obs)} t_f={1e6*p.t_f:.0f}us  -logL(MAP)={m['fval']:.2f}  sigma={np.unique(np.round(p.sigma,4))}")
    for j, n in enumerate(p.nomes):
        lo4, hi4 = x[j] - 4 * sd[j], x[j] + 4 * sd[j]
        flag = "  <-- 4sd fora da caixa" if (lo4 < p.lo[j] or hi4 > p.hi[j]) else ""
        print(f"   {n:>8s}: {x[j]:10.4g} +- {sd[j]:9.3g}   caixa [{p.lo[j]:g}, {p.hi[j]:g}]{flag}")
    res = (p.sim(x[None])[0] - p.x_obs) / p.sigma
    print(f"   residuos padronizados: media {res.mean():.2f}, dp {res.std():.2f}, max |r| {np.abs(res).max():.2f}")
    # frac do prior com verossimilhanca finita
    ll = p.logpost_lote(p.amostrar_prior(500, np.random.default_rng(0)))
    print(f"   prior: {np.isfinite(ll).mean():.3f} finitos")
    t = p.t_dados
    D = p.x_obs.reshape(p.forma_obs)
    Y = p.sim(x[None])[0].reshape(p.forma_obs)
    if spec["tipo"] == "EDP":
        for k in range(D.shape[0]):
            ax.plot(spec["x"][spec["idx"]], D[k], "o", ms=2.5, color=f"C{k}")
            ax.plot(spec["x"][spec["idx"]], Y[k], "-", color=f"C{k}", label=f"t={t[k]:.0f} h")
        ax.set_xlabel("x (um)")
    else:
        for j in range(D.shape[1]):
            ax.errorbar(t, D[:, j], yerr=p.sigma.reshape(p.forma_obs)[:, j], fmt="o", ms=3, color=f"C{j}", lw=0.6)
            ax.plot(t, Y[:, j], "-", color=f"C{j}", label=p.rotulos_obs[j])
        ax.set_xlabel("t")
    ax.set_title(p.rotulo, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=80)
print("ok")
