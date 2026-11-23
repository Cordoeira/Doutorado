"""Estimativa do tempo da trilha de dados reais, calibrada pelos tempos medidos na execucao completa anterior.

Numba (6 convertidos): cada modelo tem um analogo sintetico ja rodado. Custo por passo da referencia = custo
medido agora de logpost_lote em 12 pontos + sobrecarga do PT (tempo antigo por passo - custo medido do analogo).
Amostradores, solucoes e identificabilidade escalam pela razao do custo de uma avaliacao; NPE fica igual ao analogo.
PEtab: fator de carga L = (tempo por avaliacao na execucao antiga) / (t_f medido agora, maquina parada), calibrado
nos 4 PEtab pequenos antigos; referencia = passos x 12 x t_f x L; amostradores = 600 mil ESD x t_f x L; identif.
pela formula do `estimar` do pacote, corrigida pela razao medido/previsto dos PEtab antigos."""
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")
import logging
for n in ("pypesto", "amici", "petab"):
    logging.getLogger(n).setLevel(logging.CRITICAL)
import numpy as np
import pandas as pd

from a0bench import config, modelos, problemas

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
D = str(REPO / "entrega_orientador" / "dados") + "/"
tt = pd.read_csv(D + "09_tempos_por_etapa_registro.csv")
pr = pd.read_csv(D + "01_modelos_e_referencias.csv").set_index("problema")
tab = tt.groupby(["problema", "etapa"]).segundos.sum().unstack()
AMOS = ["metropolis", "ensemble", "mala", "hmc"]


def custo(p, k, reps):
    X = np.tile(np.asarray(p.spec["centro"] if getattr(p, "verd", None) is None and hasattr(p, "spec") else
                           (p.verd if getattr(p, "verd", None) is not None else np.clip(p.xn, p.lo, p.hi))), (k, 1))
    X = X + 1e-4 * (p.hi - p.lo) * np.random.default_rng(0).standard_normal(X.shape)
    X = np.clip(X, p.lo, p.hi)
    p.logpost_lote(X)
    t = time.perf_counter()
    for _ in range(reps):
        p.logpost_lote(X)
    return (time.perf_counter() - t) / reps


ANALOGO = {"lebre_lince": "lotka_volterra", "gripe_internato": "seir", "gripe_internato_inicio": "seir_inicio",
           "pk_teofilina": "pk_flipflop", "pk_indometacina": "pk2", "fisher_kpp_jin": "fisher_kpp"}
linhas = []
for novo, ana in ANALOGO.items():
    pn, pa = problemas.Sintetico(modelos.POR_NOME[novo]), problemas.Sintetico(modelos.POR_NOME[ana])
    s12n, s12a = custo(pn, 12, 300), custo(pa, 12, 300)
    s1n, s1a = custo(pn, 1, 600), custo(pa, 1, 600)
    o = tab.loc[ana]
    passo_antigo = o["referencia"] / pr.loc[ana, "ref_passos"]
    sobre = max(passo_antigo - s12a, 0.0)
    passo = s12n + sobre
    r1 = s1n / s1a
    am = sum(o[m] for m in AMOS) * r1
    outros = (o["map"] + o["solucoes"] + o["laplace"]) * r1 + o["identificabilidade"] * r1 * pn.d / pa.d
    npe = o["npe"] * (1.0 + 0.002 * max(len(pn.x_obs) - len(pa.x_obs), 0))   # entrada maior na rede: pequena correcao
    linhas.append(dict(modelo=novo, tipo="numba", d=pn.d, ref_50k=50000 * passo, ref_400k=400000 * passo,
                       amostradores=am, npe=npe, outros=outros, analogo=ana, passos_analogo=int(pr.loc[ana, "ref_passos"]),
                       us_aval_12=1e6 * s12n / 12))

# ---- PEtab: fator de carga pelos antigos
pasta = os.environ.get("A0_MODELOS_AMICI", os.path.expanduser("~/a0bench_runs/resultados_a0/_amici_models"))
L, Lam, Fid = [], [], []
for nome in ["Bertozzi_PNAS2020", "Boehm_JProteomeRes2014", "Rahman_MBS2016", "Crauste_CellSystems2017"]:
    p = problemas.Petab(nome, 1.0, pasta, config.PETAB[nome])
    o = tab.loc[nome]
    por_aval = o["referencia"] / (pr.loc[nome, "ref_passos"] * 12)
    L.append(por_aval / p.t_f)
    Lam.append(sum(o[m] for m in AMOS) / (600000 * p.t_f))
    prev = 3.3 * p.d * (30 + 9 + 8) * 12 * p.c_s * p.t_f
    Fid.append(o["identificabilidade"] / prev)
    print(f"{nome}: t_f agora {1e3*p.t_f:.2f} ms, por avaliacao na ref antiga {1e3*por_aval:.2f} ms -> L {L[-1]:.2f};"
          f" amostradores L' {Lam[-1]:.2f}; identif. medido/previsto(L=1) {Fid[-1]:.2f}")
L, Lam, Fid = np.median(L), np.median(Lam), np.median(Fid)
print(f"medianas: L {L:.2f}, L' {Lam:.2f}, F_id {Fid:.2f}")
npe_treino = np.median([tab.loc[n, "npe"] for n in ["Bertozzi_PNAS2020", "Boehm_JProteomeRes2014", "Rahman_MBS2016",
                                                     "Crauste_CellSystems2017", "Sneyd_PNAS2002", "Elowitz_Nature2000"]])
for l in open(REPO / "docs" / "petab_novos_custos.jsonl"):
    r = json.loads(l)
    if not r["ok"]:
        continue
    tf, cs, d = r["t_f_ms"] * 1e-3, r["c_s"], r["d"]
    linhas.append(dict(modelo=r["nome"], tipo="PEtab", d=d, ref_50k=50000 * 12 * tf * L, ref_400k=400000 * 12 * tf * L,
                       amostradores=600000 * tf * Lam, npe=10000 * tf * L + npe_treino,
                       outros=Fid * 3.3 * d * 47 * 12 * cs * tf + 200 * tf * L * 40,
                       analogo="", passos_analogo=0, us_aval_12=1e3 * r["t_f_ms"]))
df = pd.DataFrame(linhas)
df["total_50k"] = df.ref_50k + df.amostradores + df.npe + df.outros
df["total_400k"] = df.ref_400k + df.amostradores + df.npe + df.outros
pd.set_option("display.width", 250)
print((df.set_index("modelo")[["tipo", "d", "us_aval_12", "ref_50k", "ref_400k", "amostradores", "npe", "outros", "total_50k", "total_400k"]]
       .assign(**{c: lambda x, c=c: (x[c] / 60).round(1) for c in ["ref_50k", "ref_400k", "amostradores", "npe", "outros", "total_50k", "total_400k"]}))
      .to_string())
print("(minutos; us_aval_12 = microssegundos por avaliacao)")
df.to_csv(REPO / "docs" / "estimativa_tempo_reais.csv", index=False)
