"""Graficos do benchmark, todos feitos a partir dos arquivos gravados (nenhum roda simulacao).

Gerais (uma figura para todos os modelos):
  mapa_calor, por_classe, custo, eficiencia, mapa_diagnosticos      desempenho dos metodos
  mapa_identificabilidade, dimensoes, espectro, predicao            identificabilidade
Por modelo:
  tracos, autocorrelacao, convergencia                              cadeias de Markov
  solucao, estados, campo                                           espaco de solucao (dados, fase, x-t)
  marginais, pares, correlacao                                      distribuicoes a posteriori
  contornos, superficie                                             contornos da posterior com o caminho das cadeias; fatia do log p
  painel_identificabilidade, perfis, caminhos                       identificabilidade por quatro metodos
relatorio_geral e relatorio_modelo chamam tudo e gravam em disco. Tipografia do LaTeX (Computer Modern) com
seaborn; sem LaTeX instalado, usa STIX.
"""
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

METODOS = {"laplace": "Laplace", "metropolis": "Metropolis", "ensemble": "Ensemble", "mala": "MALA", "hmc": "HMC",
           "npe": "NPE", "npe_pfn": "NPE-PFN"}
AMOSTRADORES = ["Metropolis", "Ensemble", "MALA", "HMC"]
CADEIAS = ["metropolis", "ensemble", "mala", "hmc"]
CLASSES_ID = {"identificavel": "I", "parcial": "P", "nao identificavel": "N"}
COR = dict(zip(METODOS, sns.color_palette("colorblind", 7)))
FORMATOS = ["pdf", "png"]
from .resultados import CLASSES as ORDEM_TIPO


PREAMBULO = r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{amsmath}"
_TEX_OK = None


def latex_funciona():
    """O LaTeX desta maquina desenha uma figura? Precisa de latex, dvipng e das fontes cm-super; se faltar
    alguma coisa, as figuras saem na fonte STIX em vez de falharem uma a uma."""
    global _TEX_OK
    if _TEX_OK is None:
        _TEX_OK = False
        if shutil.which("latex") and shutil.which("dvipng"):
            import io

            try:
                with mpl.rc_context({"text.usetex": True, "text.latex.preamble": PREAMBULO, "font.family": "serif"}):
                    fig = mpl.figure.Figure(figsize=(1.2, 0.6))
                    fig.text(0.1, 0.4, r"$\chi^2_{\mathrm{PL}}$ ação 95\%")
                    fig.savefig(io.BytesIO(), format="png", dpi=40)
                _TEX_OK = True
            except Exception:  # noqa: BLE001  (pacote de LaTeX ausente, por exemplo)
                _TEX_OK = False
    return _TEX_OK


def estilo(usar_tex=None):
    tex = latex_funciona() if usar_tex is None else usar_tex
    sns.set_theme(context="notebook", style="ticks", palette="colorblind")
    mpl.rcParams.update({
        "text.usetex": tex,
        "text.latex.preamble": PREAMBULO,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"] if tex else ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm" if tex else "stix",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
    return tex


def preparar(tab):
    t = tab.copy()
    t["Método"] = t["metodo"].map(METODOS)
    t["Classe"] = t["classe"]
    t["Modelo"] = t["rotulo"] + np.where(t["ref_ok"], "", r" $^\ast$")
    t["ordem"] = t["classe"].map({k: i for i, k in enumerate(ORDEM_TIPO)}) * 100 + t["d"]
    return t.sort_values(["ordem", "rotulo"])


def _salvar(fig, saida, nome, formatos=None):
    if saida is not None:
        Path(saida).mkdir(parents=True, exist_ok=True)
        for ext in (formatos or FORMATOS):
            fig.savefig(Path(saida) / f"{nome}.{ext}")
    return fig


# ------------------------------------------------------------------------------------ leitura
def _npz(arq):
    arq = Path(arq)
    if not arq.exists():
        return None
    with np.load(arq, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _json(arq):
    try:
        return json.loads(Path(arq).read_text())
    except (OSError, ValueError):
        return None


def _problema(saida_dir, problema):
    """Informacoes do modelo: rotulo, nomes, limites do prior e valor verdadeiro (se houver)."""
    pasta = Path(saida_dir) / problema
    info = _json(pasta / "problema.json") or {}
    ident = _json(pasta / "identificabilidade.json") or {}
    if "nomes" not in info and "nomes" in ident:
        info.update(nomes=ident["nomes"], lo=ident["limites"][0], hi=ident["limites"][1])
    info.setdefault("rotulo", problema)
    return info


def _ref(saida_dir, problema):
    r = _npz(Path(saida_dir) / problema / "referencia.npz")
    return None if r is None else r["amostras"]


def _amostras(saida_dir, problema, metodo):
    """Amostras do metodo, com as sementes juntas."""
    arqs = sorted((Path(saida_dir) / problema).glob(f"{metodo}_s*.npy"))
    return np.concatenate([np.load(a) for a in arqs]) if arqs else None


def _cadeia(saida_dir, problema, metodo):
    return _npz(Path(saida_dir) / problema / "cadeias" / f"{metodo}.npz")


def _selecao(d, maximo, ref=None, lo=None, hi=None):
    """Indices de ate `maximo` parametros: os mais e os menos contraidos quando nao cabem todos."""
    if d <= maximo:
        return list(range(d))
    if ref is None or lo is None:
        return list(range(maximo))
    contr = 1.0 - ref.var(axis=0) / ((np.asarray(hi) - np.asarray(lo)) ** 2 / 12.0)
    ordem = np.argsort(contr)
    return sorted(set(ordem[: maximo // 2].tolist() + ordem[-(maximo - maximo // 2):].tolist()))


def _dens(x, grade):
    """Densidade de uma amostra numa grade (nucleo gaussiano)."""
    from scipy.stats import gaussian_kde

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.ptp(x) == 0:
        return np.zeros_like(grade)
    try:
        return gaussian_kde(x[:: max(1, len(x) // 4000)])(grade)
    except Exception:  # noqa: BLE001
        return np.zeros_like(grade)


def _hpd2(x, y, lim_x, lim_y, caixas=36, massas=(0.5, 0.9)):
    """Densidade conjunta suavizada e os niveis que encerram as massas pedidas."""
    from scipy.ndimage import gaussian_filter

    H, xe, ye = np.histogram2d(x, y, bins=caixas, range=[lim_x, lim_y])
    H = gaussian_filter(H, 1.3)
    v = np.sort(H.ravel())[::-1]
    c = np.cumsum(v) / max(v.sum(), 1e-300)
    niveis = sorted({float(v[min(len(v) - 1, int(np.searchsorted(c, m)))]) for m in massas})
    return 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), H.T, niveis


def _pct():
    return "\\%" if mpl.rcParams["text.usetex"] else "%"


_PARA_TEX = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
             "$": r"\$", "^": r"\^{}", "~": r"\~{}", "≈": r"$\approx$", "·": r"$\cdot$", "−": r"$-$",
             "×": r"$\times$", "≤": r"$\leq$", "≥": r"$\geq$", "→": r"$\to$", "χ": r"$\chi$"}


def _tex(txt):
    """Texto comum pronto para o LaTeX: escapa os caracteres especiais e troca os simbolos que o LaTeX nao
    aceita direto (um caractere desconhecido derrubaria a figura inteira)."""
    if not mpl.rcParams["text.usetex"]:
        return txt.replace("$", r"\$")
    return "".join(_PARA_TEX.get(c, c if ord(c) < 256 or c in "–—‘’“”" else "?") for c in txt)


def mapa_calor(tab, saida=None):
    t = preparar(tab)
    m = t.pivot(index="Modelo", columns="Método", values="c2st")
    m = m.loc[t["Modelo"].drop_duplicates(), [v for v in METODOS.values() if v in m.columns]]
    fig, ax = plt.subplots(figsize=(2.2 + 1.05 * m.shape[1], 0.34 * len(m) + 1.2))
    sns.heatmap(m, annot=True, fmt=".2f", cmap="rocket_r", vmin=0.5, vmax=1.0, linewidths=0.6,
                cbar_kws={"label": r"C2ST contra a referência"}, ax=ax)
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    return _salvar(fig, saida, "c2st_mapa")


def custo(tab, saida=None):
    t = preparar(tab)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sns.scatterplot(t, x="esd", y="c2st", hue="Método", style="Classe", s=70, alpha=0.85, ax=ax,
                    hue_order=[v for v in METODOS.values() if v in set(t["Método"])])
    ax.axhline(0.5, color="0.4", lw=0.8, ls="--")
    ax.set(xscale="log", xlabel=r"Custo (ESD)", ylabel=r"C2ST contra a referência", ylim=(0.45, 1.02))
    sns.move_legend(ax, "center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    sns.despine(fig)
    return _salvar(fig, saida, "c2st_custo")


def por_classe(tab, saida=None):
    t = preparar(tab)
    ordem = [v for v in METODOS.values() if v in set(t["Método"])]
    classes = [c for c in ORDEM_TIPO if c in set(t["Classe"])]
    fig, eixos = plt.subplots(1, 2, figsize=(15, 4.2))
    for ax, y, rot in zip(eixos, ["c2st", "razao_dp"],
                          [r"C2ST contra a referência", r"Razão de desvios-padrão $\hat\sigma/\sigma_{\mathrm{ref}}$"]):
        sns.boxplot(t, x="Método", y=y, hue="Classe", order=ordem, hue_order=classes, fliersize=0,
                    boxprops={"alpha": 0.45}, ax=ax)
        sns.stripplot(t, x="Método", y=y, hue="Classe", order=ordem, hue_order=classes, dodge=True, size=4,
                      legend=False, ax=ax)
        ax.set(xlabel="", ylabel=rot)
    eixos[0].axhline(0.5, color="0.4", lw=0.8, ls="--")
    eixos[1].axhline(1.0, color="0.4", lw=0.8, ls="--")
    eixos[1].set_yscale("log")
    eixos[1].set_yticks([0.25, 0.5, 1, 2, 4], labels=["0,25", "0,5", "1", "2", "4"])
    eixos[1].minorticks_off()
    eixos[0].get_legend().remove()
    sns.move_legend(eixos[1], "center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="")
    sns.despine(fig)
    fig.tight_layout()
    return _salvar(fig, saida, "c2st_classe")


def eficiencia(tab, saida=None):
    t = preparar(tab)
    t = t[t["Método"].isin(AMOSTRADORES)]
    classes = [c for c in ORDEM_TIPO if c in set(t["Classe"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    sns.boxplot(t, x="Método", y="ess_por_mil_esd", hue="Classe", order=AMOSTRADORES, hue_order=classes,
                fliersize=0, boxprops={"alpha": 0.45}, ax=ax)
    sns.stripplot(t, x="Método", y="ess_por_mil_esd", hue="Classe", order=AMOSTRADORES, hue_order=classes,
                  dodge=True, size=4, legend=False, ax=ax)
    ax.set(yscale="log", xlabel="", ylabel="ESS mínimo por mil ESD")
    sns.move_legend(ax, "center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="")
    sns.despine(fig)
    return _salvar(fig, saida, "eficiencia")


def _ordenar_id(ident):
    t = ident.copy()
    t["Modelo"] = t["rotulo"] + np.where(t["ref_ok"], "", r" $^\ast$")
    t["ordem"] = t["classe"].map({k: i for i, k in enumerate(ORDEM_TIPO)}) * 100 + t["d"]
    return t.sort_values(["ordem", "rotulo", "k"])


def _ler_id(saida_dir, problema):
    import json

    return json.loads((Path(saida_dir) / problema / "identificabilidade.json").read_text())


GREGAS = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "kappa", "lambda", "mu", "nu",
          "rho", "sigma", "tau", "phi", "psi", "omega"}


def _nome(nome, maximo=18):
    """Nome do parametro pronto para o titulo: k_a vira k com indice, beta vira a letra grega."""
    import re

    log, base = nome.startswith("log "), nome[4:] if nome.startswith("log ") else nome
    m = re.fullmatch(r"([A-Za-z]+)(?:_([A-Za-z0-9]+))?", base)
    if m and (len(m.group(1)) == 1 or m.group(1) in GREGAS):
        s = ("\\" + m.group(1)) if m.group(1) in GREGAS else m.group(1)
        if m.group(2):
            s += "_{" + (m.group(2) if len(m.group(2)) == 1 or m.group(2).isdigit() else rf"\mathrm{{{m.group(2)}}}") + "}"
        return "$" + (r"\log " if log else "") + s + "$"
    txt = base if len(base) <= maximo else base[: maximo - 3] + "..."
    if mpl.rcParams["text.usetex"]:
        for c in "_%&#":
            txt = txt.replace(c, "\\" + c)
    return ("log " if log else "") + txt


def _sigla(q):
    """I, P ou N, com 'e' para nao identificabilidade estrutural e 'm' para conjunto de confianca desconexo."""
    extra = ("e" if q.get("natureza") == "estrutural" else "") + ("m" if q.get("multimodal") else "")
    return CLASSES_ID[q["classe_id"] if "classe_id" in q else q["classe"]] + (rf"$^{{\mathrm{{{extra}}}}}$" if extra else "")


def mapa_identificabilidade(ident, saida=None):
    t = _ordenar_id(ident)
    t["sigla"] = [_sigla(q) for q in t.to_dict("records")]
    modelos = t["Modelo"].drop_duplicates()
    c = t.pivot(index="Modelo", columns="k", values="contracao").loc[modelos].clip(lower=0)
    a = t.pivot(index="Modelo", columns="k", values="sigla").loc[modelos]
    fig, ax = plt.subplots(figsize=(1.8 + 0.5 * c.shape[1], 0.34 * len(c) + 1.2))
    sns.heatmap(c, annot=a.fillna(""), fmt="", cmap="mako", vmin=0, vmax=1, linewidths=0.6,
                cbar_kws={"label": r"Contração da posterior $1-\sigma^2_{\mathrm{post}}/\sigma^2_{\mathrm{prior}}$"},
                ax=ax)
    ax.set(xlabel=r"Parâmetro $k$", ylabel="")
    ax.tick_params(length=0)
    return _salvar(fig, saida, "identificabilidade")


def espectro(saida_dir, ident, saida=None):
    """Autovalores da informacao de Fisher no MAP e da matriz H a posteriori, na caixa e apos normalizar o prior."""
    import pandas as pd

    t = _ordenar_id(ident).drop_duplicates("problema")
    rot = ["informada pelos dados", "dominada pelo prior"]
    linhas = []
    for _, r in t.iterrows():
        j = _ler_id(saida_dir, r["problema"])
        linhas += [dict(Modelo=r["Modelo"], painel=0, autovalor=max(x, 1e-8), Direção=rot[x <= 12])
                   for x in j["autovalores"]]
        for painel, chave in ((1, "caixa"), (2, "normal")):
            e = (j.get("lis") or {}).get(chave)
            if e:
                linhas += [dict(Modelo=r["Modelo"], painel=painel, autovalor=max(x, 1e-8), Direção=rot[k >= e["dim"]["0.1"]])
                           for k, x in enumerate(e["autovalores"])]
    df = pd.DataFrame(linhas)
    fig, eixos = plt.subplots(1, 3, figsize=(13.5, 0.3 * t.shape[0] + 1.4), sharey=True)
    titulos = [r"Fisher no MAP (local)", r"$H$ a posteriori, caixa do prior", r"$H$ a posteriori, prior normalizado"]
    for ax, k in zip(eixos, range(3)):
        sns.stripplot(df[df["painel"] == k], x="autovalor", y="Modelo", hue="Direção", jitter=False, size=5,
                      alpha=0.85, ax=ax, hue_order=rot, legend=(k == 0))
        ax.set(xscale="log", ylabel="", xlabel="Autovalor", title=titulos[k])
        ax.grid(axis="y", lw=0.4, alpha=0.5)
    eixos[0].axvline(12, color="0.4", lw=0.8, ls="--")
    sns.move_legend(eixos[0], "lower left", bbox_to_anchor=(0.0, 1.08), ncol=2, frameon=False, title="")
    sns.despine(fig)
    fig.tight_layout()
    return _salvar(fig, saida, "espectros")


def perfis(saida_dir, problema, saida=None, por_linha=7, inteiro=False):
    """Perfis de verossimilhanca no formato da figura 3 de Raue et al. (2009): chi2 do perfil contra o
    parametro (em log10 quando ele e positivo), linha preta; estrela cinza no valor calibrado; parabola
    cinza da aproximacao quadratica (intervalo assintotico); tracejados nos limiares pontual (de baixo) e
    simultaneo (de cima). Como no artigo, cada painel mostra o trecho em que o perfil esta na janela dos
    limiares; inteiro=True mostra a caixa do prior toda."""
    from scipy.interpolate import PchipInterpolator

    r = _ler_id(saida_dir, problema)
    P = r["parametros"]
    if "chi2" not in P[0]:                      # resultado de versao anterior do pacote
        return None
    info = _problema(saida_dir, problema)
    c_min = r["chi2_min"]
    d1, dd = r["delta_chi2_pontual"], r["delta_chi2_simultaneo"]
    nc = min(por_linha, len(P))
    nl = int(np.ceil(len(P) / nc))
    alt = 2.5 * nl + 0.9
    fig, eixos = plt.subplots(nl, nc, figsize=(1.75 * nc + 0.9, alt), sharey=True, squeeze=False)
    for ax, q in zip(eixos.ravel(), P):
        u, c2 = np.array(q["u"]), np.array(q["chi2"])
        a = q.get("assintotico")
        fino = np.unique(np.concatenate([np.linspace(u[0], u[-1], 400), u,
                                         q["u_otimo"] + np.linspace(-1, 1, 201) * 6 * ((a or {}).get("dp") or 0.0)]))
        fino = fino[(fino >= u[0]) & (fino <= u[-1])]
        if a is not None:
            parab = c_min + ((fino - a["u_otimo"]) / a["dp"]) ** 2 if not a["infinito"] else np.full_like(fino, c_min)
            ax.plot(fino, parab, color="0.6", lw=1.0, zorder=1)
        if len(u) >= 3 and np.all(np.diff(u) > 0):
            ax.plot(fino, PchipInterpolator(u, c2)(fino), color="black", lw=1.8, zorder=3)
        else:
            ax.plot(u, c2, color="black", lw=1.8, zorder=3)
        ax.axhline(c_min + d1, color="black", lw=0.9, ls=(0, (5, 3)), zorder=2)
        ax.axhline(c_min + dd, color="black", lw=0.9, ls=(0, (5, 3)), zorder=2)
        ax.plot(q["u_otimo"], c_min, marker="*", color="0.55", ms=9, mec="0.4", mew=0.4, zorder=4, clip_on=False)
        topo = c_min + 1.22 * dd
        xa, xb = u[0], u[-1]
        vis = np.flatnonzero(c2 <= topo)
        if not inteiro and len(vis) and xb > xa:
            k0, k1 = int(vis[0]), int(vis[-1])
            if k0 > 0:                              # onde o perfil sai pelo topo do painel, de cada lado
                xa = u[k0] - (u[k0] - u[k0 - 1]) * (topo - c2[k0]) / max(c2[k0 - 1] - c2[k0], 1e-300)
            if k1 < len(u) - 1:
                xb = u[k1] + (u[k1 + 1] - u[k1]) * (topo - c2[k1]) / max(c2[k1 + 1] - c2[k1], 1e-300)
            folga = 0.12 * max(xb - xa, 1e-9 * (u[-1] - u[0]))
            xa, xb = max(xa - folga, u[0]), min(xb + folga, u[-1])
        ax.set(xlabel=_nome_coord(q), xlim=(xa, xb), ylim=(c_min - 0.12 * dd, topo))
        ax.set_title(_sigla(q), fontsize=8, color="0.25", loc="right", pad=2)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        plt.setp(ax.get_xticklabels(), fontsize=8)
    for ax in eixos.ravel()[len(P):]:
        ax.set_visible(False)
    for ax in eixos[:, 0]:
        ax.set_ylabel(r"$\chi^2_{\mathrm{PL}}$")
    nivel = 100 * r.get("nivel_confianca", 0.95)
    leg = ("preto: perfil; cinza: aproximação quadrática no ótimo (intervalo assintótico); estrela: valor calibrado;",
           f"tracejados: limiares pontual (de baixo) e simultâneo (de cima) de {nivel:.0f}{_pct()}")
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.62 if nc >= 5 else 0.8) / alt))
    fig.text(0.5, 1 - 0.08 / alt, f"{info['rotulo']}: perfis de verossimilhança", ha="center", va="top", fontsize=10)
    fig.text(0.5, 1 - 0.36 / alt, " ".join(leg) if nc >= 5 else "\n".join(leg), ha="center", va="top", fontsize=7.5,
             color="0.25")
    return _salvar(fig, saida, f"perfis_{problema}")


def _nome_coord(q):
    """Rotulo do eixo do perfil: log10 do parametro quando o perfil e mostrado em ordens de grandeza."""
    rot = q.get("rotulo_coord", q["nome"])
    for pre, tex in (("log10(", r"\log_{10}"), ("ln(", r"\ln")):
        if rot.startswith(pre):
            dentro = _nome(rot[len(pre):-1], 16)
            return rf"${tex}($" + dentro + "$)$" if not dentro.startswith("$") else rf"${tex}({dentro[1:-1]})$"
    return _nome(rot, 18)


def caminhos(saida_dir, problema, saida=None, maximo=4):
    """Trajetorias dos demais parametros ao longo do perfil de cada parametro nao identificavel."""
    r = _ler_id(saida_dir, problema)
    P = r["parametros"]
    alvos = [k for k, q in enumerate(P) if q["classe"] != "identificavel"][:maximo]
    if not alvos:
        return None
    lim1 = r.get("limiar_pontual", 1.92)
    fig, eixos = plt.subplots(1, len(alvos), figsize=(3.3 * len(alvos), 3.0), sharey=True, squeeze=False)
    paleta = sns.color_palette("colorblind", len(P))
    for ax, k in zip(eixos[0], alvos):
        q = P[k]
        v, X, dl = np.array(q["grade"]), np.array(q["caminho"]), np.array(q["delta"])
        lo, hi = np.array(r["limites"]) if "limites" in r else (X.min(axis=0), X.max(axis=0))
        U = (X - lo) / np.maximum(hi - lo, 1e-300)
        sel = dl <= lim1
        amp = [np.ptp(U[sel, j]) if sel.sum() > 1 else 0.0 for j in range(len(P))]
        for j in sorted((j for j in range(len(P)) if j != k), key=lambda j: -amp[j])[:4]:
            ax.plot(v, U[:, j], color=paleta[j % len(paleta)], lw=0.7, alpha=0.4)
            ax.plot(v[sel], U[sel, j], color=paleta[j % len(paleta)], lw=1.8, marker="o", ms=2.5,
                    label=_nome(P[j]["nome"], 12))
        ax.set(title=f"Perfil de {_nome(q['nome'])}: {_sigla(q)}", xlabel=_nome(q["nome"]), ylim=(-0.03, 1.03))
        ax.legend(frameon=False, fontsize=8)
    eixos[0, 0].set_ylabel(r"Demais parâmetros (posição no prior)")
    sns.despine(fig)
    fig.tight_layout()
    return _salvar(fig, saida, f"caminhos_{problema}")


def dimensoes(saida_dir, ident, saida=None):
    """Quantas direcoes os dados informam, segundo cada criterio."""
    import pandas as pd

    t = _ordenar_id(ident).drop_duplicates("problema")
    linhas = []
    for _, r in t.iterrows():
        j = _ler_id(saida_dir, r["problema"])
        e, l = j.get("estrutura") or {}, j.get("lis") or {}
        n_i = sum(q["classe"] == "identificavel" for q in j["parametros"])

        def dim(esp):
            return (esp or {}).get("dim", {}).get("0.1", np.nan)
        linhas.append({"Modelo": r["Modelo"], r"$d$": r["d"], "Posto genérico": e.get("posto_generico", np.nan),
                       "Posto no ótimo": e.get("posto", np.nan),
                       "Fisher no MAP": j["dim_informada"], r"$H$ na caixa": dim(l.get("caixa")),
                       r"$H$ normalizado": dim(l.get("normal")),
                       r"$H$ livre de dados": dim((l.get("livre_de_dados") or {}).get("normal")),
                       "Perfis identificáveis": n_i})
    df = pd.DataFrame(linhas).set_index("Modelo")
    fr = df.div(df[r"$d$"], axis=0)
    fig, ax = plt.subplots(figsize=(1.9 + 1.25 * df.shape[1], 0.34 * len(df) + 1.3))
    sns.heatmap(fr, annot=df, fmt=".0f", cmap="crest", vmin=0, vmax=1, linewidths=0.6,
                cbar_kws={"label": r"Fração de $d$"}, ax=ax)
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="left")
    return _salvar(fig, saida, "dimensoes")


def predicao(saida_dir, ident, saida=None):
    """Contribuicao do perfil de cada parametro para a banda de predicao."""
    t = _ordenar_id(ident)
    t["sigla"] = [_sigla(q) for q in t.to_dict("records")]
    contrib = {}
    for prob in t["problema"].unique():
        c = (_ler_id(saida_dir, prob).get("predicao") or {}).get("contribuicao")
        if c:
            contrib.update({(prob, k + 1): v for k, v in enumerate(c)})
    t["contrib"] = [contrib.get((r["problema"], r["k"]), np.nan) for r in t.to_dict("records")]
    modelos = t["Modelo"].drop_duplicates()
    c = t.pivot(index="Modelo", columns="k", values="contrib").loc[modelos]
    a = t.pivot(index="Modelo", columns="k", values="sigla").loc[modelos]
    fig, ax = plt.subplots(figsize=(1.8 + 0.5 * c.shape[1], 0.34 * len(c) + 1.2))
    sns.heatmap(c, annot=a.fillna(""), fmt="", cmap="flare", vmin=0, vmax=1, linewidths=0.6,
                cbar_kws={"label": r"Fração da banda de predição"}, ax=ax)
    ax.set(xlabel=r"Parâmetro $k$", ylabel="")
    ax.tick_params(length=0)
    return _salvar(fig, saida, "predicao")


# ==================================================================================== cadeias de Markov
def _series(c):
    """Da cadeia gravada: series (C, n, d) para o traco, amostras ja sem aquecimento, faixa entre caminhantes."""
    X, k = c["cadeia"], int(c["desbaste"])
    q = int(c["aquecimento"]) // k
    if X.ndim == 4:
        return X.mean(axis=2), X[:, q:].reshape(X.shape[0], -1, X.shape[-1]), np.quantile(X, [0.05, 0.95], axis=2), q, k
    return X, X[:, q:], None, q, k


def tracos(saida_dir, problema, metodo, saida=None, maximo=6, pontos=1500):
    """Traco de cada parametro (uma cor por cadeia, aquecimento sombreado), densidade de cada cadeia contra a
    referencia e traco do log da posterior com os diagnosticos."""
    c = _cadeia(saida_dir, problema, metodo)
    if c is None:
        return None
    info, ref = _problema(saida_dir, problema), _ref(saida_dir, problema)
    serie, pos, faixa, q, k = _series(c)
    C, n, d = serie.shape
    nomes = [str(x) for x in c["nomes"]]
    js = _selecao(d, maximo, ref, info.get("lo"), info.get("hi"))
    passo = max(1, n // pontos)
    t = np.arange(n)[::passo] * k
    cores = sns.color_palette("colorblind", max(C, 3))
    alt = 1.35 * (len(js) + 1) + 1.0
    fig, eixos = plt.subplots(len(js) + 1, 2, figsize=(9.6, alt), squeeze=False,
                              gridspec_kw={"width_ratios": [3.4, 1], "wspace": 0.03, "top": 1 - 0.42 / alt,
                                           "bottom": 0.55 / alt, "left": 0.09, "right": 0.985, "hspace": 0.18})
    for (ax, ad), j in zip(eixos, js):
        tudo = np.concatenate([pos[:, :, j].ravel(), ref[:, j] if ref is not None else []])
        lo, hi = np.quantile(tudo, [0.001, 0.999])
        lo, hi = lo - 0.15 * (hi - lo + 1e-300), hi + 0.15 * (hi - lo + 1e-300)
        g = np.linspace(lo, hi, 160)
        if ref is not None:
            ad.fill_betweenx(g, 0, _dens(ref[:, j], g), color="0.82", lw=0)
        for cc in range(C):
            if faixa is not None:
                ax.fill_between(t, faixa[0, cc, ::passo, j], faixa[1, cc, ::passo, j], color=cores[cc], alpha=0.18, lw=0)
            ax.plot(t, serie[cc, ::passo, j], lw=0.6, color=cores[cc], alpha=0.9, rasterized=True)
            ad.plot(_dens(pos[cc][:, j], g), g, color=cores[cc], lw=1.0)
        ax.axvspan(0, q * k, color="0.92", lw=0, zorder=0)
        ax.set(ylabel=_nome(nomes[j]), xlim=(0, n * k), ylim=(lo, hi), xticklabels=[])
        ad.set(ylim=(lo, hi), xticks=[], yticklabels=[], xlim=(0, None))
        sns.despine(ax=ad, bottom=True)
    ax, ad = eixos[-1]
    lp = c["logp"] if c["logp"].ndim == 2 else c["logp"].mean(axis=2)
    bons = lp[:, q:][np.isfinite(lp[:, q:])]
    for cc in range(C):
        ax.plot(t, lp[cc, ::passo], lw=0.6, color=cores[cc], alpha=0.9, rasterized=True, label=f"cadeia {cc + 1}")
    if len(bons):
        a, b = np.quantile(bons, [0.001, 1.0])
        ax.set_ylim(a - 0.3 * (b - a + 1e-9), b + 0.15 * (b - a + 1e-9))
    ax.axvspan(0, q * k, color="0.92", lw=0, zorder=0)
    ax.set(ylabel=r"$\log p$", xlabel="Passo", xlim=(0, n * k))
    ax.legend(frameon=False, fontsize=7, ncol=C, loc="lower right")
    ad.axis("off")
    pasta = Path(saida_dir) / problema
    regs = [x for x in (_json(a) for a in sorted(pasta.glob(f"{metodo}_s*.json"))) if x and x.get("ok")]
    e_tot = info.get("ref_ess") if metodo == "referencia" else (sum(x.get("ess_min", 0) for x in regs) or None)
    linhas = [rf"$\hat R$ máx. {float(np.max(c['rhat'])):.3f}"]
    if e_tot:
        linhas.append(f"ESS mín. {e_tot:.0f}")
    linhas += [f"aceitação {float(np.mean(c['aceitacao'])):.2f}", f"{int(c['passos'])} passos por cadeia",
               f"{float(c['esd']):.0f} ESD por cadeia"]
    for k_, txt in enumerate(linhas):
        ad.text(0.08, 0.86 - 0.19 * k_, txt, transform=ad.transAxes, va="center", fontsize=8)
    sns.despine(fig)
    for ax_, _ in eixos:
        sns.despine(ax=ax_)
    rot = METODOS.get(metodo, "Referência" if metodo == "referencia" else metodo)
    fig.suptitle(f"{info['rotulo']}: {rot}" + (" (média e faixa dos caminhantes)" if faixa is not None else ""),
                 y=1 - 0.08 / alt, fontsize=11)
    fig.align_ylabels(eixos[:, 0])
    return _salvar(fig, saida, f"cadeias_{metodo}")


def autocorrelacao(saida_dir, problema, saida=None, maximo=8):
    """Autocorrelacao de cada parametro, um painel por amostrador (media entre as cadeias)."""
    from .cadeias import autocorrelacao as fac

    presentes = [(m, _cadeia(saida_dir, problema, m)) for m in ["referencia", *CADEIAS]]
    presentes = [(m, c) for m, c in presentes if c is not None]
    if not presentes:
        return None
    info = _problema(saida_dir, problema)
    fig, eixos = plt.subplots(1, len(presentes), figsize=(3.0 * len(presentes) + 0.6, 3.0), sharey=True, squeeze=False)
    for ax, (m, c) in zip(eixos[0], presentes):
        serie, _, _, q, k = _series(c)
        x = serie[:, q:]
        C, n, d = x.shape
        nomes = [str(v) for v in c["nomes"]]
        js = _selecao(d, maximo)
        curvas = np.array([[fac(x[cc, :, j], min(n - 1, 400)) for cc in range(C)] for j in js]).mean(axis=1)
        abaixo = np.flatnonzero(np.all(curvas < 0.05, axis=0))
        fim = int(min(curvas.shape[1] - 1, max(10, 2 * (abaixo[0] if len(abaixo) else curvas.shape[1]))))
        paleta = sns.color_palette("colorblind", len(js))
        for cor, j, a in zip(paleta, js, curvas):
            ax.plot(np.arange(fim + 1) * k, a[: fim + 1], color=cor, lw=1.1, label=_nome(nomes[j], 10))
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set(title=METODOS.get(m, "Referência"), xlabel="Defasagem (passos)", ylim=(-0.15, 1.02))
    eixos[0, 0].set_ylabel("Autocorrelação")
    eixos[0, -1].legend(frameon=False, fontsize=7, loc="upper right")
    sns.despine(fig)
    fig.suptitle(info["rotulo"], y=1.0, fontsize=11)
    fig.tight_layout()
    return _salvar(fig, saida, "autocorrelacao")


def convergencia(saida_dir, problema, saida=None, maximo=6):
    """Media acumulada de cada parametro contra o custo, em desvios-padrao da referencia: mostra quanto
    custo cada amostrador precisa para acertar a media."""
    ref = _ref(saida_dir, problema)
    presentes = [(m, _cadeia(saida_dir, problema, m)) for m in CADEIAS]
    presentes = [(m, c) for m, c in presentes if c is not None]
    if ref is None or not presentes:
        return None
    info = _problema(saida_dir, problema)
    d = ref.shape[1]
    js = _selecao(d, maximo, ref, info.get("lo"), info.get("hi"))
    mu, sd = ref.mean(axis=0), np.maximum(ref.std(axis=0), 1e-300)
    nc = min(3, len(js))
    nl = int(np.ceil(len(js) / nc))
    fig, eixos = plt.subplots(nl, nc, figsize=(3.4 * nc, 2.5 * nl + 0.5), sharex=True, sharey=True, squeeze=False)
    for ax, j in zip(eixos.ravel(), js):
        for m, c in presentes:
            serie, _, _, q, k = _series(c)
            x = serie[:, q:, j]
            custo_passo = float(c["esd"]) / int(c["passos"]) * k
            esd = (q + 1 + np.arange(x.shape[1])) * custo_passo
            acum = (np.cumsum(x, axis=1) / (1 + np.arange(x.shape[1])) - mu[j]) / sd[j]
            passo = max(1, x.shape[1] // 600)
            for cc in range(x.shape[0]):
                ax.plot(esd[::passo], acum[cc, ::passo], color=COR[m], lw=0.9, alpha=0.85, rasterized=True,
                        label=METODOS[m] if cc == 0 else None)
        ax.axhspan(-0.1, 0.1, color="0.88", lw=0, zorder=0)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set(xscale="log", ylim=(-1.5, 1.5), title=_nome(info.get("nomes", [f"theta_{j + 1}"] * d)[j]))
    for ax in eixos.ravel()[len(js):]:
        ax.set_visible(False)
    for ax in eixos[-1]:
        ax.set_xlabel("Custo acumulado (ESD)")
    for ax in eixos[:, 0]:
        ax.set_ylabel(r"(média acumulada $-\ \mu_{\mathrm{ref}})/\sigma_{\mathrm{ref}}$")
    h, r = eixos[0, 0].get_legend_handles_labels()
    fig.legend(h, r, frameon=False, ncol=len(presentes), loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=9,
               title=f"{info['rotulo']}: erro da média contra o custo (uma linha por cadeia)")
    sns.despine(fig)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / (2.5 * nl + 0.5)))
    return _salvar(fig, saida, "convergencia")


# ==================================================================================== espaco de solucao
def _eixo_tempo(ax, t):
    t = np.asarray(t, float)
    if len(t) > 3 and t[0] > 0 and t[-1] / t[0] > 1e3:
        ax.set_xscale("log")


def solucao(saida_dir, problema, saida=None, max_obs=8, max_cond=3):
    """Ajuste aos dados: medidas com barra de erro, solucao no ponto central e faixa de 95% a posteriori."""
    s = _npz(Path(saida_dir) / problema / "solucao.npz")
    if s is None:
        return None
    info = _problema(saida_dir, problema)
    rot = [str(x) for x in s["rotulos_obs"]]
    nc_cond = min(int(s["n_cond"]), max_cond)
    cores = sns.color_palette("colorblind", max(nc_cond, 3))
    m = s["dados_0"].shape[1]
    if s["dados_0"].shape[0] == 1:            # um unico instante: as componentes viram o eixo horizontal
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        xs = np.arange(m)
        ax.fill_between(xs, s["q_0"][0, 0], s["q_0"][2, 0], color=cores[0], alpha=0.25, lw=0, label=f"faixa de 95{_pct()}")
        ax.plot(xs, s["centro_0"][0], color=cores[0], lw=1.4, label="ponto central")
        ax.errorbar(xs, s["dados_0"][0], yerr=s["sigma_0"][0], fmt="o", ms=4, color="0.15", lw=0.9, label="dados")
        ax.set_xticks(xs, [_nome(r, 10) for r in rot], rotation=45 if m > 6 else 0, fontsize=8)
        ax.set(ylabel="Saída observada", title=info["rotulo"])
        ax.legend(frameon=False, fontsize=8)
        sns.despine(fig)
        fig.tight_layout()
        return _salvar(fig, saida, "solucao")
    js = [j for j in range(m) if any(np.isfinite(s[f"dados_{c}"][:, j]).any() for c in range(nc_cond))][:max_obs]
    nc = min(4, len(js))
    nl = int(np.ceil(len(js) / nc))
    fig, eixos = plt.subplots(nl, nc, figsize=(3.1 * nc, 2.4 * nl + 0.6), squeeze=False, sharex=True)
    for ax, j in zip(eixos.ravel(), js):
        for c in range(nc_cond):
            t, D, S, Yc, Q = (s[f"{k}_{c}"] for k in ("t", "dados", "sigma", "centro", "q"))
            ax.fill_between(t, Q[0, :, j], Q[2, :, j], color=cores[c], alpha=0.25, lw=0)
            ax.plot(t, Yc[:, j], color=cores[c], lw=1.3, label=f"condição {c + 1}" if nc_cond > 1 else "ponto central")
            ok = np.isfinite(D[:, j])
            ax.errorbar(t[ok], D[ok, j], yerr=np.where(np.isfinite(S[ok, j]), S[ok, j], 0.0), fmt="o", ms=3,
                        color="0.15" if nc_cond == 1 else cores[c], lw=0.8, label="dados" if c == 0 else None)
        ax.set_title(_nome(rot[j], 22), fontsize=9)
        _eixo_tempo(ax, s["t_0"])
    for ax in eixos.ravel()[len(js):]:
        ax.set_visible(False)
    for ax in eixos[-1]:
        ax.set_xlabel("Tempo")
    eixos[0, 0].legend(frameon=False, fontsize=7)
    sns.despine(fig)
    fig.suptitle(f"{info['rotulo']}: dados, solução e faixa de 95{_pct()} a posteriori", y=1.0, fontsize=11)
    fig.tight_layout()
    return _salvar(fig, saida, "solucao")


def estados(saida_dir, problema, saida=None, max_est=6):
    """EDOs: todos os estados no tempo (faixa a posteriori, ponto central, verdade) e o espaco de fase."""
    s = _npz(Path(saida_dir) / problema / "solucao.npz")
    if s is None or "est_centro" not in s or str(s["tipo"]).startswith("EDP"):
        return None
    info = _problema(saida_dir, problema)
    t, Yc, Q, A = s["t_est"], s["est_centro"], s["est_q"], s["est_amostras"]
    n = Yc.shape[1]
    obs = set(np.asarray(s["idx"], int).tolist())
    js = list(range(min(n, max_est)))
    pares = [(0, 1)] if n == 2 else [(0, 1), (0, 2)] if n >= 3 else []
    nl = len(js) * max(len(pares), 1)
    fig = plt.figure(figsize=(10.0 if pares else 6.0, 1.25 * max(len(js), 2 * len(pares), 2) + 1.0))
    gs = fig.add_gridspec(nl, 2 if pares else 1, width_ratios=[1.35, 1] if pares else [1], wspace=0.28, hspace=0.35)
    cor = sns.color_palette("colorblind")[0]
    alt = nl // len(js)
    for k_, j in enumerate(js):
        ax = fig.add_subplot(gs[k_ * alt:(k_ + 1) * alt, 0])
        ax.fill_between(t, Q[0, :, j], Q[2, :, j], color=cor, alpha=0.25, lw=0)
        ax.plot(t, Yc[:, j], color=cor, lw=1.2)
        if "est_verd" in s:
            ax.plot(t, s["est_verd"][:, j], color="0.2", lw=0.8, ls="--")
        ax.set_ylabel(rf"$y_{{{j + 1}}}$" + (r"$^{\ast}$" if j in obs else ""))
        _eixo_tempo(ax, t)
        if k_ < len(js) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Tempo")
        sns.despine(ax=ax)
    alt = nl // max(len(pares), 1)
    for k_, (a, b) in enumerate(pares):
        ax = fig.add_subplot(gs[k_ * alt:(k_ + 1) * alt, 1])
        for y in A[:40]:
            ax.plot(y[:, a], y[:, b], color=cor, lw=0.4, alpha=0.25, rasterized=True)
        ax.plot(Yc[:, a], Yc[:, b], color=cor, lw=1.3, label="ponto central")
        if "est_verd" in s:
            ax.plot(s["est_verd"][:, a], s["est_verd"][:, b], color="0.2", lw=0.8, ls="--", label="verdade")
        ax.plot(Yc[0, a], Yc[0, b], "o", color="0.1", ms=4)
        ax.set(ylabel=rf"$y_{{{b + 1}}}$")
        if k_ == len(pares) - 1:
            ax.set_xlabel(rf"$y_{{{a + 1}}}$")
        else:
            ax.set_xticklabels([])
        if k_ == 0:
            ax.legend(frameon=False, fontsize=7)
        sns.despine(ax=ax)
    fig.suptitle(f"{info['rotulo']}: estados (* observado) e espaço de fase", y=0.995, fontsize=11)
    return _salvar(fig, saida, "estados")


def campo(saida_dir, problema, saida=None):
    """EDPs: mapa de calor da solucao no plano x-t, largura da faixa de 95% e pontos de medida. Nos modelos
    estacionarios, o perfil u(x) com a faixa e os dados."""
    s = _npz(Path(saida_dir) / problema / "solucao.npz")
    if s is None or "est_centro" not in s or not str(s["tipo"]).startswith("EDP"):
        return None
    info = _problema(saida_dir, problema)
    t, Yc, Q = s["t_est"], s["est_centro"], s["est_q"]
    x = s["x_malha"] if "x_malha" in s else np.arange(Yc.shape[1])
    idx = np.asarray(s["idx"], int)
    if bool(s["estatico"]):
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        cor = sns.color_palette("colorblind")[0]
        ax.fill_between(x, Q[0, 0], Q[2, 0], color=cor, alpha=0.25, lw=0, label=f"faixa de 95{_pct()}")
        ax.plot(x, Yc[0], color=cor, lw=1.4, label="ponto central")
        if "est_verd" in s:
            ax.plot(x, s["est_verd"][0], color="0.2", lw=0.8, ls="--", label="verdade")
        ax.errorbar(x[idx], s["dados_0"][0], yerr=s["sigma_0"][0], fmt="o", ms=4, color="0.15", lw=0.9, label="dados")
        ax.set(xlabel=r"$x$", ylabel=r"$u(x)$", title=info["rotulo"])
        ax.legend(frameon=False, fontsize=8)
        sns.despine(fig)
        fig.tight_layout()
        return _salvar(fig, saida, "campo")
    nf = int(s["campos"])
    pts = Yc.shape[1] // nf
    fig, eixos = plt.subplots(2, nf, figsize=(4.6 * nf + 0.6, 5.6), squeeze=False, sharex=True, sharey=True)
    for f_ in range(nf):
        U = Yc[:, f_ * pts:(f_ + 1) * pts].T
        Lg = (Q[2] - Q[0])[:, f_ * pts:(f_ + 1) * pts].T
        duplo = np.nanmin(U) < 0 < np.nanmax(U)
        lim = float(np.nanmax(np.abs(U)))
        for ax, Z, kw, rot in ((eixos[0, f_], U, dict(cmap="vlag", vmin=-lim, vmax=lim) if duplo else dict(cmap="mako"),
                                r"$u(x,t)$ no ponto central"),
                               (eixos[1, f_], Lg, dict(cmap="rocket_r"), f"largura da faixa de 95{_pct()}")):
            im = ax.pcolormesh(t, x[:pts], Z, shading="auto", rasterized=True, **kw)
            fig.colorbar(im, ax=ax, pad=0.02).set_label(rot, fontsize=8)
            daqui = idx[(idx >= f_ * pts) & (idx < (f_ + 1) * pts)] - f_ * pts
            tt, xx = np.meshgrid(s["t_0"], x[daqui])
            ax.plot(tt.ravel(), xx.ravel(), "o", ms=3, mfc="none", mec="white", mew=0.9)
            ax.plot(tt.ravel(), xx.ravel(), "o", ms=3, mfc="none", mec="0.1", mew=0.4)
        if nf > 1:
            eixos[0, f_].set_title(f"campo {f_ + 1}")
        eixos[1, f_].set_xlabel(r"$t$")
    for ax in eixos[:, 0]:
        ax.set_ylabel(r"$x$")
    fig.suptitle(info["rotulo"] + ": solução no plano $x$-$t$ (círculos: medidas)", y=0.995, fontsize=11)
    fig.tight_layout()
    return _salvar(fig, saida, "campo")


# ==================================================================================== posteriores
def _janela(ref, j, lo=None, hi=None, extra=()):
    v = np.concatenate([ref[:, j], *[np.asarray(e, float).ravel() for e in extra]])
    a, b = np.quantile(v, [0.001, 0.999])
    a, b = a - 0.2 * (b - a + 1e-300), b + 0.2 * (b - a + 1e-300)
    if lo is not None:
        a, b = max(a, lo[j]), min(b, hi[j])
    return (a, b) if b > a else (a - 1e-9, a + 1e-9)


def marginais(saida_dir, problema, saida=None, maximo=12):
    """Distribuicao marginal de cada parametro: referencia (area) e cada metodo (linha)."""
    ref = _ref(saida_dir, problema)
    if ref is None:
        return None
    info = _problema(saida_dir, problema)
    d = ref.shape[1]
    nomes = info.get("nomes", [f"theta_{j + 1}" for j in range(d)])
    lo, hi = info.get("lo"), info.get("hi")
    js = _selecao(d, maximo, ref, lo, hi)
    met = [(m, _amostras(saida_dir, problema, m)) for m in METODOS]
    met = [(m, a) for m, a in met if a is not None]
    sol = _npz(Path(saida_dir) / problema / "solucao.npz") or {}
    nc = min(4, len(js))
    nl = int(np.ceil(len(js) / nc))
    fig, eixos = plt.subplots(nl, nc, figsize=(2.9 * nc, 2.2 * nl + 0.8), squeeze=False)
    for ax, j in zip(eixos.ravel(), js):
        a, b = _janela(ref, j, lo, hi)
        g = np.linspace(a, b, 200)
        ax.fill_between(g, 0, _dens(ref[:, j], g), color="0.8", lw=0, label="Referência")
        for m, am in met:
            ax.plot(g, _dens(am[:, j], g), color=COR[m], lw=1.1, label=METODOS[m])
        if "x_centro" in sol:
            ax.axvline(sol["x_centro"][j], color="0.1", lw=0.7, ls=":")
        if info.get("verd") is not None:
            ax.axvline(info["verd"][j], color="0.1", lw=0.8, ls="--")
        ax.set(title=_nome(nomes[j]), yticks=[], xlim=(a, b), ylim=(0, None))
        sns.despine(ax=ax, left=True)
    for ax in eixos.ravel()[len(js):]:
        ax.set_visible(False)
    h, r = eixos[0, 0].get_legend_handles_labels()
    alt = 2.2 * nl + 0.8
    fig.legend(h, r, frameon=False, ncol=min(8, len(r)), loc="upper center", bbox_to_anchor=(0.5, 1 - 0.3 / alt),
               fontsize=8)
    fig.suptitle(info["rotulo"] + (" (tracejado: valor verdadeiro; pontilhado: ponto central)"
                                   if info.get("verd") is not None else " (pontilhado: ponto central)"),
                 y=1 - 0.02 / alt, va="top", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / alt))
    return _salvar(fig, saida, "marginais")


def pares(saida_dir, problema, saida=None, maximo=6):
    """Distribuicoes conjuntas de cada par: regioes de 50% e 90% da referencia (areas) e o contorno de 90% de
    cada amostrador (linhas). Na diagonal, as marginais."""
    ref = _ref(saida_dir, problema)
    if ref is None or ref.shape[1] < 2:
        return None
    info = _problema(saida_dir, problema)
    d = ref.shape[1]
    nomes = info.get("nomes", [f"theta_{j + 1}" for j in range(d)])
    lo, hi = info.get("lo"), info.get("hi")
    js = _selecao(d, maximo, ref, lo, hi)
    met = [(m, _amostras(saida_dir, problema, m)) for m in METODOS]
    met = [(m, a) for m, a in met if a is not None]
    lim = {j: _janela(ref, j, lo, hi) for j in js}
    k = len(js)
    fig, eixos = plt.subplots(k, k, figsize=(1.85 * k + 0.6, 1.85 * k + 0.6), squeeze=False)
    for a_, i in enumerate(js):
        for b_, j in enumerate(js):
            ax = eixos[a_, b_]
            if b_ > a_:
                ax.axis("off")
                continue
            if a_ == b_:
                g = np.linspace(*lim[i], 160)
                ax.fill_between(g, 0, _dens(ref[:, i], g), color="0.8", lw=0)
                for m, am in met:
                    ax.plot(g, _dens(am[:, i], g), color=COR[m], lw=0.9)
                ax.set(xlim=lim[i], ylim=(0, None), yticks=[])
            else:
                xc, yc, H, niv = _hpd2(ref[:, j], ref[:, i], lim[j], lim[i])
                ax.contourf(xc, yc, H, levels=[*niv, H.max() + 1e-12], colors=["0.84", "0.62"][-len(niv):] if len(niv) > 1
                            else ["0.7"])
                for m, am in met:
                    xm, ym, Hm, nm = _hpd2(am[:, j], am[:, i], lim[j], lim[i], massas=(0.9,))
                    if Hm.max() > nm[0] > 0:
                        ax.contour(xm, ym, Hm, levels=nm, colors=[COR[m]], linewidths=0.9)
                ax.set(xlim=lim[j], ylim=lim[i])
            if a_ == k - 1:
                ax.set_xlabel(_nome(nomes[j]))
                plt.setp(ax.get_xticklabels(), rotation=45, fontsize=7)
            else:
                ax.set_xticklabels([])
            if b_ == 0 and a_ > 0:
                ax.set_ylabel(_nome(nomes[i]))
                plt.setp(ax.get_yticklabels(), fontsize=7)
            elif a_ != b_:
                ax.set_yticklabels([])
            sns.despine(ax=ax, left=(a_ == b_))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(color="0.62", label=f"Referência (50{_pct()})"), Patch(color="0.84", label=f"Referência (90{_pct()})")]
    h += [Line2D([], [], color=COR[m], lw=1.2, label=METODOS[m]) for m, _ in met]
    fig.legend(handles=h, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.95), fontsize=8, title=info["rotulo"])
    fig.tight_layout()
    return _salvar(fig, saida, "pares")


def correlacao(saida_dir, problema, saida=None):
    """Mapa de calor da correlacao a posteriori entre os parametros (referencia)."""
    ref = _ref(saida_dir, problema)
    if ref is None or ref.shape[1] < 2 or ref.std(axis=0).min() == 0:
        return None
    info = _problema(saida_dir, problema)
    d = ref.shape[1]
    nomes = [_nome(n, 12) for n in info.get("nomes", [f"theta_{j + 1}" for j in range(d)])]
    R = np.corrcoef(ref.T)
    fig, ax = plt.subplots(figsize=(1.6 + 0.5 * d, 1.2 + 0.45 * d))
    sns.heatmap(R, mask=np.triu(np.ones_like(R, bool), 1), annot=d <= 10, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                linewidths=0.5, square=True, xticklabels=nomes, yticklabels=nomes, annot_kws={"fontsize": 7},
                cbar_kws={"label": "Correlação a posteriori", "shrink": 0.8}, ax=ax)
    ax.tick_params(length=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_title(info["rotulo"], fontsize=10)
    fig.tight_layout()
    return _salvar(fig, saida, "correlacao")


# ==================================================================================== contornos
DELTAS = [0.0, 0.69, 2.30, 4.61, 6.91, 11.5, 18.0, 30.0]     # quedas de log p: regioes de 50, 90, 99 e 99,9% e alem


def _pares_salvos(saida_dir, problema, d):
    s = _npz(Path(saida_dir) / problema / "superficie.npz")
    if s is not None:
        return [tuple(int(v) for v in par) for par in s["pares"]], s
    return ([(0, 1)] if d >= 2 else []), None


def contornos(saida_dir, problema, saida=None, passos=150):
    """Contornos da posterior conjunta de dois parametros (regioes de 50, 90 e 99% da referencia), com os
    primeiros passos de uma cadeia de cada amostrador: mostra como cada cadeia anda sobre a posterior."""
    ref = _ref(saida_dir, problema)
    if ref is None or ref.shape[1] < 2:
        return None
    info = _problema(saida_dir, problema)
    d = ref.shape[1]
    nomes = info.get("nomes", [f"theta_{j + 1}" for j in range(d)])
    lo, hi = info.get("lo"), info.get("hi")
    pares_, s = _pares_salvos(saida_dir, problema, d)
    cad = [(m, _cadeia(saida_dir, problema, m)) for m in CADEIAS]
    cad = [(m, c) for m, c in cad if c is not None] or [(None, None)]
    P = len(pares_)
    alt = 2.75 * P + 1.0
    fig, eixos = plt.subplots(P, len(cad), figsize=(3.0 * len(cad) + 0.6, alt), squeeze=False)
    tons = sns.color_palette("mako_r", 5)[:3]
    for l_, (i, j) in enumerate(pares_):
        li, lj = _janela(ref, i, lo, hi), _janela(ref, j, lo, hi)
        xc, yc, H, niv = _hpd2(ref[:, i], ref[:, j], li, lj, caixas=44, massas=(0.5, 0.9, 0.99))
        for ax, (m, c) in zip(eixos[l_], cad):
            if len(niv) and H.max() > niv[0]:
                ax.contourf(xc, yc, H, levels=[*niv, H.max() + 1e-12], colors=tons[-len(niv):])
            if c is not None:
                X = c["cadeia"][0]
                if X.ndim == 3:                       # ensemble: tres caminhantes
                    for w in range(min(3, X.shape[1])):
                        ax.plot(X[:passos, w, i], X[:passos, w, j], "-", lw=0.6, color=COR[m], alpha=0.95)
                        ax.plot(X[0, w, i], X[0, w, j], "o", ms=3.5, color=COR[m], mec="0.05", mew=0.5)
                else:
                    ax.plot(X[:passos, i], X[:passos, j], "-o", lw=0.6, ms=1.6, color=COR[m], alpha=0.95)
                    ax.plot(X[0, i], X[0, j], "o", ms=4.5, color=COR[m], mec="0.05", mew=0.7)
            if s is not None:
                ax.plot(s["x_centro"][i], s["x_centro"][j], "+", color="0.05", ms=8, mew=1.2, zorder=5)
            if info.get("verd") is not None:
                ax.plot(info["verd"][i], info["verd"][j], "x", color="0.05", ms=5, mew=1.0, zorder=5)
            ax.set(xlim=li, ylim=lj, xlabel=_nome(nomes[i]))
            if l_ == 0:
                ax.set_title(METODOS[m] if m else "Posterior de referência", fontsize=10)
            plt.setp(ax.get_xticklabels(), fontsize=7)
            plt.setp(ax.get_yticklabels(), fontsize=7)
            sns.despine(ax=ax)
        eixos[l_, 0].set_ylabel(_nome(nomes[j]))
        for ax in eixos[l_, 1:]:
            ax.set_yticklabels([])
    fig.suptitle(f"{info['rotulo']}: regiões de 50, 90 e 99{_pct()} da posterior (referência) e os primeiros {passos} "
                 f"passos de uma cadeia (+ ponto central, × verdade)", y=1 - 0.12 / alt, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / alt))
    return _salvar(fig, saida, "contornos")


def superficie(saida_dir, problema, saida=None):
    """Fatia exata do log da posterior em dois parametros, com os demais fixos no ponto central. As linhas
    tracejadas sao a regiao de 90% da posterior conjunta (referencia): onde a fatia e mais estreita que a
    conjunta, os outros parametros compensam o par."""
    s = _npz(Path(saida_dir) / problema / "superficie.npz")
    if s is None:
        return None
    info, ref = _problema(saida_dir, problema), _ref(saida_dir, problema)
    d = len(s["x_centro"])
    nomes = info.get("nomes", [f"theta_{j + 1}" for j in range(d)])
    P = len(s["pares"])
    fig, eixos = plt.subplots(1, P, figsize=(3.5 * P + 1.1, 3.6), squeeze=False)
    for l_, ((i, j), ax) in enumerate(zip(s["pares"], eixos[0])):
        gi, gj, Z = s["eixo_i"][l_], s["eixo_j"][l_], s["logp"][l_]
        fin = np.isfinite(Z)
        queda = np.where(fin, Z[fin].max() - Z, np.nan) if fin.any() else np.full_like(Z, np.nan)
        im = ax.contourf(gi, gj, queda.T, levels=DELTAS, cmap="mako")
        ax.contour(gi, gj, queda.T, levels=DELTAS[1:4], colors="white", linewidths=0.5)
        if ref is not None:
            xc, yc, H, niv = _hpd2(ref[:, i], ref[:, j], (gi[0], gi[-1]), (gj[0], gj[-1]), massas=(0.9,))
            if H.max() > niv[0] > 0:
                ax.contour(xc, yc, H, levels=niv, colors=[sns.color_palette("colorblind")[1]], linewidths=1.1,
                           linestyles="--")
        ax.plot(s["x_centro"][i], s["x_centro"][j], "+", color="white", ms=8, mew=1.2)
        ax.set(xlim=(gi[0], gi[-1]), ylim=(gj[0], gj[-1]), xlabel=_nome(nomes[i]), ylabel=_nome(nomes[j]))
        plt.setp(ax.get_xticklabels(), fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.suptitle(f"{info['rotulo']}: fatia do log da posterior no ponto central (tracejado: região de 90{_pct()} "
                 f"da posterior conjunta)", y=0.99, fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.9, 0.95))
    cb = fig.colorbar(im, cax=fig.add_axes([0.915, 0.2, 0.014, 0.62]))
    cb.set_label(r"Queda de $\log p$ em relação ao máximo", fontsize=8)
    return _salvar(fig, saida, "superficie")


# ==================================================================================== identificabilidade
def painel_identificabilidade(saida_dir, problema, saida=None):
    """Quatro leituras lado a lado: intervalos do perfil de verossimilhanca contra os da posterior; contracao
    da posterior; espectros de Fisher e do subespaco informado; composicao das direcoes informadas. O titulo
    traz o posto do jacobiano e as combinacoes de parametros."""
    r = _json(Path(saida_dir) / problema / "identificabilidade.json")
    if r is None or "parametros" not in r:
        return None
    info, ref = _problema(saida_dir, problema), _ref(saida_dir, problema)
    P = r["parametros"]
    d = len(P)
    lo, hi = np.array(r["limites"][0]), np.array(r["limites"][1])
    nomes = [_nome(q["nome"], 14) for q in P]
    cores = dict(zip(CLASSES_ID, sns.color_palette("colorblind", 3)))
    h1 = max(2.2, 0.36 * d)
    alt = h1 + 3.2 + 2.2
    fig = plt.figure(figsize=(11.5, alt))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[h1, 3.2], hspace=1.25 / (0.5 * (h1 + 3.2)),
                          wspace=0.3, top=1 - 0.75 / alt, bottom=0.6 / alt, left=0.1, right=0.96)
    ax = fig.add_subplot(gs[0, 0])
    ys = np.arange(d)[::-1]
    for y, q, j in zip(ys, P, range(d)):
        u = lambda v: (np.asarray(v, float) - lo[j]) / (hi[j] - lo[j])
        ax.plot([0, 1], [y, y], color="0.85", lw=1.0, zorder=0)
        for a, b in q.get("ic_conjunto", [q["ic"]]):
            ax.plot(u([a, b]), [y + 0.14, y + 0.14], color=cores[q["classe"]], lw=4.5, solid_capstyle="butt")
        if ref is not None:
            a, b = np.quantile(ref[:, j], [0.025, 0.975])
            ax.plot(u([a, b]), [y - 0.16, y - 0.16], color="0.15", lw=1.6, solid_capstyle="butt")
        ax.plot(u(r["x_otimo"][j]), y + 0.14, "|", color="0.05", ms=7, mew=1.0)
        ax.text(1.03, y, _sigla(q), va="center", fontsize=9)
    ax.set(yticks=ys, yticklabels=nomes, xlim=(-0.02, 1.12), ylim=(-0.7, d - 0.3), xlabel="Posição no prior (0 a 1)",
           title=f"Perfil de verossimilhança (cor) e posterior (preto), 95{_pct()}")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)
    ax = fig.add_subplot(gs[0, 1])
    ax.barh(ys, [max(q["contracao"], 0.0) for q in P], color=[cores[q["classe"]] for q in P], height=0.6)
    ax.set(yticks=ys, yticklabels=[], xlim=(0, 1), ylim=(-0.7, d - 0.3), title="Contração da posterior",
           xlabel=r"$1-\sigma^2_{\mathrm{post}}/\sigma^2_{\mathrm{prior}}$")
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)
    ax = fig.add_subplot(gs[1, 0])
    lis = r.get("lis") or {}
    k = np.arange(1, d + 1)
    piso = 1e-10
    ax.plot(k, np.maximum(r["autovalores"], piso), "o-", ms=4, lw=1.0, color="0.25", label="Fisher no ótimo")
    for chave, rot, cor in (("caixa", "$H$ a posteriori, caixa", sns.color_palette("colorblind")[3]),
                            ("normal", "$H$ a posteriori, normalizado", sns.color_palette("colorblind")[4])):
        if chave in lis:
            ax.plot(k, np.maximum(lis[chave]["autovalores"], piso), "s-", ms=3.5, lw=1.0, color=cor,
                    label=f"{rot} ($r={lis[chave]['dim']['0.1']}$)")
    ax.axhline(12, color="0.6", lw=0.7, ls="--")
    ax.set(yscale="log", xlabel="Direção (ordem decrescente)", ylabel="Autovalor", title="Espectros", xticks=k)
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)
    ax = fig.add_subplot(gs[1, 1])
    if "caixa" in lis:
        V = np.abs(np.array(lis["caixa"]["vetores"]))
        sns.heatmap(V, cmap="mako_r", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "peso do parâmetro na direção"},
                    xticklabels=k, yticklabels=nomes if d <= 20 else False, linewidths=0.3 if d <= 12 else 0)
        ax.set(xlabel="Direção de $H$ (caixa)", title="Quem compõe cada direção")
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
        plt.setp(ax.get_xticklabels(), fontsize=7)
        ax.tick_params(length=0)
    else:
        ax.axis("off")
    e = r.get("estrutura") or {}
    linha = info["rotulo"]
    if e:
        linha += f": posto do jacobiano {e.get('posto_generico', e.get('posto'))} de {d}"
        livres = [c["texto"] for c in e.get("nucleo", []) if c.get("texto")]
        if livres:
            linha += "; combinações livres: " + _tex("; ".join(livres[:4]))
    fig.suptitle(linha, y=1 - 0.1 / alt, fontsize=10)
    return _salvar(fig, saida, "identificabilidade")


def mapa_diagnosticos(tab, saida=None):
    """Mapas de calor por modelo e amostrador: R-hat maximo entre cadeias, ESS minimo por mil ESD e aceitacao."""
    from matplotlib.colors import LogNorm

    t = preparar(tab)
    t = t[t["Método"].isin(AMOSTRADORES)]
    if t.empty or "rhat_max" not in t or t["rhat_max"].isna().all():
        return None
    modelos = t["Modelo"].drop_duplicates()
    ordem = [m for m in AMOSTRADORES if m in set(t["Método"])]
    fig, eixos = plt.subplots(1, 3, figsize=(3.0 + 3 * 0.95 * len(ordem) + 3.2, 0.34 * len(modelos) + 1.4), sharey=True)
    for ax, col, rot, kw in (
            (eixos[0], "rhat_max", r"$\hat R$ máximo", dict(cmap="rocket_r", vmin=1.0, vmax=1.2, fmt=".2f")),
            (eixos[1], "ess_por_mil_esd", "ESS mínimo por mil ESD", dict(cmap="mako", fmt=".1f", norm=LogNorm(
                vmin=max(1e-2, float(np.nanmin(t["ess_por_mil_esd"]))), vmax=float(np.nanmax(t["ess_por_mil_esd"]))))),
            (eixos[2], "aceitacao", "Aceitação", dict(cmap="crest", vmin=0, vmax=1, fmt=".2f"))):
        m = t.pivot(index="Modelo", columns="Método", values=col).loc[modelos, ordem]
        sns.heatmap(m, annot=True, linewidths=0.5, cbar_kws={"label": rot}, annot_kws={"fontsize": 7}, ax=ax, **kw)
        ax.set(xlabel="", ylabel="")
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
    fig.tight_layout()
    return _salvar(fig, saida, "diagnosticos_cadeias")


# ==================================================================================== relatorios
POR_MODELO = [("solucao", solucao), ("estados", estados), ("campo", campo), ("marginais", marginais), ("pares", pares),
              ("correlacao", correlacao), ("contornos", contornos), ("superficie", superficie),
              ("autocorrelacao", autocorrelacao),
              ("convergencia", convergencia), ("identificabilidade", painel_identificabilidade)]


def relatorio_modelo(saida_dir, problema, destino, formatos=("png",), tex=None):
    """Gera todas as figuras de um modelo em destino/. Devolve (problema, figuras feitas, erros)."""
    mpl.use("Agg")
    estilo(tex)
    destino = Path(destino)
    feitas, erros = [], []

    def tentar(nome, f):
        try:
            if f() is not None:
                feitas.append(nome)
        except Exception as e:  # noqa: BLE001
            erros.append(f"{nome}: {type(e).__name__}: {str(e)[:160]}")
        plt.close("all")
    antigos = list(FORMATOS)
    FORMATOS[:] = list(formatos)
    try:
        for nome, f in POR_MODELO:
            tentar(nome, lambda f=f: f(saida_dir, problema, saida=destino))
        for m in ["referencia", *CADEIAS]:
            tentar(f"cadeias_{m}", lambda m=m: tracos(saida_dir, problema, m, saida=destino))
        if (Path(saida_dir) / problema / "identificabilidade.json").exists():
            tentar("perfis", lambda: _renomear(perfis(saida_dir, problema, destino), destino, problema, "perfis"))
            tentar("caminhos", lambda: _renomear(caminhos(saida_dir, problema, destino), destino, problema, "caminhos"))
    finally:
        FORMATOS[:] = antigos
    return problema, feitas, erros


def _renomear(fig, saida, problema, nome):
    """perfis e caminhos gravam com o nome do modelo; dentro da pasta do modelo o nome curto basta."""
    for a in Path(saida).glob(f"{nome}_{problema}.*"):
        a.replace(a.with_name(nome + a.suffix))
    return fig


def relatorio_geral(saida_dir, destino, tab, ident):
    """Figuras que juntam todos os modelos. Devolve (figuras feitas, erros)."""
    feitas, erros = [], []

    def tentar(nome, f, *args):
        try:
            if f(*args, destino) is not None:
                feitas.append(nome)
        except Exception as e:  # noqa: BLE001
            erros.append(f"{nome}: {type(e).__name__}: {str(e)[:160]}")
        plt.close("all")
    if not tab.empty:
        for nome, f in (("c2st_mapa", mapa_calor), ("c2st_classe", por_classe), ("eficiencia", eficiencia),
                        ("c2st_custo", custo), ("diagnosticos_cadeias", mapa_diagnosticos)):
            tentar(nome, f, tab)
    if not ident.empty:
        tentar("identificabilidade", mapa_identificabilidade, ident)
        for nome, f in (("espectros", espectro), ("dimensoes", dimensoes), ("predicao", predicao)):
            tentar(nome, f, saida_dir, ident)
    return feitas, erros
