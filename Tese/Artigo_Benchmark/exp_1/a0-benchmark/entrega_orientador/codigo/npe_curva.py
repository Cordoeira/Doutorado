"""Curva de orcamento do NPE: C2ST contra numero de simulacoes, reaproveitando o banco.npz de cada modelo.
Uso: python npe_curva.py <saida> [--processos N] [--tamanhos 1000,2000,5000] [--so-figura]
Grava <saida>/relatorio/npe_curva.csv e npe_curva.{png,pdf}; um json por modelo em <saida>/<modelo>/npe_curva.json (retomavel)."""
import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ORDEM = ["EDO", "EDO difícil", "EDP", "EDP difícil", "PEtab", "PEtab grande"]


def _classe(info):
    from a0bench.resultados import _classe
    return _classe(info)


def um_modelo(saida, nome, tamanhos):
    import torch
    torch.set_num_threads(int(os.environ.get("NPE_THREADS", "2")))
    from a0bench import config, inferencia, problemas
    pasta = Path(saida) / nome
    out = pasta / "npe_curva.json"
    feito = json.load(open(out)) if out.exists() else {}
    est = json.load(open(pasta / "estado.json"))
    cfg = config.montar("completo")
    grupo = est.get("grupo", "facil")
    origem = "petab" if grupo.startswith("petab") else "sintetico"
    p = problemas.construir(origem, nome, grupo, cfg, str(Path(saida) / "_amici_models"))
    ref = np.load(pasta / "referencia.npz")["amostras"]
    banco = dict(np.load(pasta / "banco.npz"))
    for n in tamanhos:
        k = str(n)
        if k in feito or n > len(banco["theta"]):
            continue
        sub = dict(theta=banco["theta"][:n], x=banco["x"][:n], n_pedido=n, falhas=0)
        t0 = time.time()
        try:
            am, _ = inferencia.amostrar_npe(p, sub, cfg["n_amostras"], 0, cfg["lote_treino"])
            m = inferencia.metricas(p, am, ref, cfg["n_c2st"])
            m.update(n_sim=n, segundos=time.time() - t0, ok=True)
        except Exception as e:  # noqa: BLE001
            m = dict(n_sim=n, segundos=time.time() - t0, ok=False, erro=str(e)[:200])
        feito[k] = m
        tmp = out.with_suffix(".tmp")
        json.dump(feito, open(tmp, "w"), indent=1)
        os.replace(tmp, out)
    return nome, feito


def juntar(saida):
    import pandas as pd
    linhas = []
    for pasta in sorted(Path(saida).iterdir()):
        est = pasta / "estado.json"
        if not est.exists():
            continue
        e = json.load(open(est))
        if e.get("situacao") != "concluido":
            continue
        cl, rot = _classe(e), e.get("rotulo", pasta.name)
        cur = pasta / "npe_curva.json"
        if cur.exists():
            for k, m in json.load(open(cur)).items():
                if m.get("ok"):
                    linhas.append(dict(modelo=pasta.name, rotulo=rot, classe=cl, n_sim=int(k), c2st=m["c2st"],
                                       erro_media=m["erro_media"], semente=0, fonte="curva"))
        # ponto de 10 mil ja calculado pelo benchmark (todas as sementes)
        for a in sorted(pasta.glob("npe_s*.json")):
            m = json.load(open(a))
            if m.get("c2st") is not None:
                linhas.append(dict(modelo=pasta.name, rotulo=rot, classe=cl, n_sim=int(round(m.get("esd", 10000))),
                                   c2st=m["c2st"], erro_media=m.get("erro_media"),
                                   semente=int(a.stem.rsplit("_s", 1)[1]), fonte="benchmark"))
    return pd.DataFrame(linhas)


def figura(df, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from a0bench import graficos
    graficos.estilo(False)
    classes = [c for c in ORDEM if c in set(df.classe)]
    cores = dict(zip(ORDEM, sns.color_palette("colorblind", len(ORDEM))))
    med = df.groupby(["modelo", "classe", "n_sim"], as_index=False).c2st.mean()  # media entre sementes
    ns = sorted(med.n_sim.unique())
    rot = [f"{n // 1000} mil" for n in ns]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=False, gridspec_kw=dict(wspace=0.25))
    ax = axs[0]
    for (m, cl), g in med.groupby(["modelo", "classe"]):
        g = g.sort_values("n_sim")
        ax.plot(g.n_sim, g.c2st, color=cores[cl], lw=0.9, alpha=0.35, marker="o", ms=3)
    for cl in classes:
        g = med[med.classe == cl].groupby("n_sim").c2st.median()
        ax.plot(g.index, g.values, color=cores[cl], lw=2.4, marker="o", ms=6, label=f"{cl} (mediana)")
    ax.axhline(0.5, color="0.5", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(rot)
    ax.set(xlabel="simulações no treino do NPE", ylabel="C2ST contra a referência", ylim=(0.45, 1.02),
           title="Cada modelo (fino) e mediana por classe (grosso)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axs[1]
    frac = med.assign(ok=med.c2st < 0.6).groupby(["classe", "n_sim"]).ok.mean().reset_index()
    for cl in classes:
        g = frac[frac.classe == cl].sort_values("n_sim")
        ax.plot(g.n_sim, g.ok, color=cores[cl], lw=2.4, marker="o", ms=6, label=cl)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(rot)
    ax.set(xlabel="simulações no treino do NPE", ylabel="fração de modelos",
           title="Fração de modelos com C2ST < 0,6", ylim=(-0.03, 1.03))
    sns.despine(fig)
    for ext in ("png", "pdf"):
        fig.savefig(f"{destino}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("saida")
    ap.add_argument("--processos", type=int, default=4)
    ap.add_argument("--tamanhos", default="1000,2000,5000")
    ap.add_argument("--so-figura", action="store_true")
    a = ap.parse_args()
    saida = Path(a.saida)
    tam = [int(t) for t in a.tamanhos.split(",")]
    if not a.so_figura:
        nomes = [p.name for p in sorted(saida.iterdir()) if (p / "estado.json").exists()
                 and json.load(open(p / "estado.json")).get("situacao") == "concluido" and (p / "banco.npz").exists()]
        print(f"{len(nomes)} modelos, tamanhos {tam}, {a.processos} processos", flush=True)
        with ProcessPoolExecutor(a.processos) as ex:
            futs = {ex.submit(um_modelo, str(saida), n, tam): n for n in nomes}
            for f in as_completed(futs):
                try:
                    nome, feito = f.result()
                    print(time.strftime("%H:%M:%S"), nome,
                          {k: round(v["c2st"], 3) for k, v in feito.items() if v.get("ok")}, flush=True)
                except Exception as e:  # noqa: BLE001
                    print(time.strftime("%H:%M:%S"), futs[f], "ERRO", e, flush=True)
    df = juntar(saida)
    rel = saida / "relatorio"
    rel.mkdir(exist_ok=True)
    df.to_csv(rel / "npe_curva.csv", index=False)
    piv = df.groupby(["classe", "n_sim"]).c2st.median().unstack()
    print("\nC2ST mediano por classe e simulações:\n", piv.round(3).to_string())
    figura(df, rel / "npe_curva")
    print("figura:", rel / "npe_curva.png")


if __name__ == "__main__":
    main()
