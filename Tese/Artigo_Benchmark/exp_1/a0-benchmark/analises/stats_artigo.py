"""Numeros para o resumo preliminar do artigo, a partir de resultados_a0 (so le arquivos)."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from a0bench.resultados import carregar, resumo_identificabilidade  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
saida = Path(sys.argv[1])
r = carregar(str(saida))
t = r.tabela()
t = t[t.c2st.notna()].copy()
ordem = ["laplace", "metropolis", "ensemble", "mala", "hmc", "npe"]
print("=== modelos na tabela:", t.problema.nunique(), "| linhas:", len(t))

print("\n=== mediana e quartis de C2ST por metodo")
q = t.groupby("metodo").c2st.describe(percentiles=[.25, .5, .75])[["25%", "50%", "75%"]].reindex(ordem)
print(q.round(3).to_string())

print("\n=== fracao de modelos com C2ST < 0,6 por metodo")
print((t.assign(ok=t.c2st < 0.6).groupby("metodo").ok.mean().reindex(ordem) * 100).round(0).to_string())

print("\n=== mediana por classe x metodo: c2st / erro_media / ess_por_mil_esd")
for v in ("c2st", "erro_media", "ess_por_mil_esd"):
    print(f"-- {v}")
    print(t.pivot_table(index="classe", columns="metodo", values=v, aggfunc="median").reindex(columns=ordem).round(3).to_string())

print("\n=== vitorias: melhor C2ST por modelo (empate dentro de 0,01 conta para todos)")
best = t.groupby("problema").c2st.transform("min")
print(t[t.c2st <= best + 0.01].groupby("metodo").size().reindex(ordem).fillna(0).astype(int).to_string())

print("\n=== amostradores: R-hat max > 1.05 e aceitacao mediana")
cad = t[t.metodo.isin(["metropolis", "ensemble", "mala", "hmc"])]
print(cad.groupby("metodo").agg(rhat_ruim=("rhat_max", lambda s: int((s > 1.05).sum())), n=("rhat_max", "size"),
                                aceitacao=("aceitacao", "median")).round(3).to_string())

print("\n=== modelos em que todos os metodos falham (min C2ST > 0,65)")
piores = t.groupby(["problema", "rotulo", "classe"]).c2st.min().reset_index()
print(piores[piores.c2st > 0.65].sort_values("c2st", ascending=False).to_string(index=False))

print("\n=== PEtab: C2ST por metodo")
print(t[t.classe.str.startswith("PEtab")].pivot_table(index="rotulo", columns="metodo", values="c2st").reindex(columns=ordem).round(2).to_string())

print("\n=== identificabilidade: totais por classe")
ri = resumo_identificabilidade(str(saida))
cols = ["identificaveis", "parciais", "nao_identificaveis", "estruturais", "multimodais", "d"]
print(ri.groupby("classe")[cols].sum().to_string())
print("total parametros:", int(ri.d.sum()), "| identificaveis:", int(ri.identificaveis.sum()),
      "| parciais:", int(ri.parciais.sum()), "| nao:", int(ri.nao_identificaveis.sum()), "| estruturais:", int(ri.estruturais.sum()))

print("\n=== tempos de parede por etapa (mediana, s), a partir dos registros")
linhas = []
for m in r.modelos:
    reg = r.registro(m)
    if reg is None:
        continue
    for e in reg:
        if e.get("evento") == "fim" and "etapa" in e and e["segundos"] < 8000:  # exclui as suspensoes
            linhas.append(dict(modelo=m, etapa=e["etapa"], s=e["segundos"]))
tt = pd.DataFrame(linhas)
cl = t.drop_duplicates("problema").set_index("problema").classe
tt["classe"] = tt.modelo.map(cl)
print(tt.pivot_table(index="classe", columns="etapa", values="s", aggfunc="median").round(0).to_string())

print("\n=== curva NPE (mediana por classe)")
cv = saida / "relatorio" / "npe_curva.csv"
if cv.exists():
    c = pd.read_csv(cv)
    print(c.groupby(["classe", "n_sim"]).c2st.median().unstack().round(3).to_string())

print("\n=== referencias: passos usados")
probs = []
for m in r.modelos:
    p = json.load(open(saida / m / "problema.json"))
    if p.get("ref_passos"):
        probs.append(dict(modelo=m, passos=p["ref_passos"], ess=p["ref_ess"], rhat=p["ref_rhat"]))
pr = pd.DataFrame(probs)
print(pr.passos.value_counts().sort_index().to_string())
print("ESS min:", pr.ess.min().round(0), "| mediana:", pr.ess.median().round(0), "| R-hat max:", pr.rhat.max().round(4))
