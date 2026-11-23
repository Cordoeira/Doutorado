"""Resumo do NPE-PFN no subconjunto em que ele rodou: C2ST contra os outros metodos, nos mesmos modelos."""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
from a0bench.resultados import carregar  # noqa: E402

pd.set_option("display.width", 200)
saida = Path(sys.argv[1])
r = carregar(str(saida))
t = r.tabela()
if t.empty or "npe_pfn" not in set(t.metodo):
    print("Nenhuma semente do NPE-PFN foi concluida.")
    raise SystemExit(0)
mods = sorted(set(t[t.metodo == "npe_pfn"].problema))
sub = t[t.problema.isin(mods)]
ordem = [c for c in ["laplace", "metropolis", "ensemble", "mala", "hmc", "npe", "npe_pfn"] if c in set(sub.metodo)]
piv = sub.pivot_table(index=["classe", "rotulo", "d"], columns="metodo", values="c2st").reindex(columns=ordem).round(2)
print(f"NPE-PFN concluido em {len(mods)} de {t.problema.nunique()} modelos (uma semente cada).\n")
print("C2ST por modelo:\n", piv.to_string(), "\n")
print("Mediana de C2ST nesses modelos:\n", piv.median().round(3).to_string(), "\n")
print("Mediana por classe:\n", piv.groupby(level="classe").median().round(2).to_string(), "\n")
comp = piv.dropna(subset=["npe", "npe_pfn"])
print(f"NPE-PFN melhor que NPE em {int((comp.npe_pfn < comp.npe).sum())} de {len(comp)} modelos; "
      f"mediana da diferenca (npe - npe_pfn): {(comp.npe - comp.npe_pfn).median():.3f}\n")
tempos = []
for m in mods:
    for e in (r.registro(m) or []):
        if e.get("evento") == "fim" and e.get("etapa") == "npe_pfn":
            tempos.append(dict(modelo=m, d=int(sub[sub.problema == m].d.iloc[0]), horas=round(e["segundos"] / 3600, 2)))
if tempos:
    tt = pd.DataFrame(tempos)
    print("Tempo de parede da etapa NPE-PFN (1 semente, 1 nucleo, CPU):\n", tt.groupby("d").horas.describe()[["count", "min", "50%", "max"]].to_string())
