"""Analises adicionais e tabelas discretizadas do experimento A0 (so le arquivos de resultados_a0).

Uso: python analises_extras.py <resultados_a0> <pasta_de_saida>
Gera em <saida>/dados: CSVs; em <saida>/figuras_extra: figuras novas; <saida>/numeros.json: numeros usados no texto.
"""
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
R = Path(sys.argv[1])
OUT = Path(sys.argv[2])
D = OUT / "dados"
D.mkdir(parents=True, exist_ok=True)
REL = R / "relatorio"
METODOS = ["laplace", "metropolis", "ensemble", "mala", "hmc", "npe"]
CLASSES = ["EDO", "EDO difícil", "EDP", "EDP difícil", "PEtab", "PEtab grande"]
num = {}

ps = pd.read_csv(REL / "resultados_por_semente.csv")
pr = pd.read_csv(REL / "problemas.csv")
ps["classe"] = pd.Categorical(ps["classe"], CLASSES)

# ---------------------------------------------------------------- tabelas basicas
colunas_ps = ["problema", "rotulo", "classe", "tipo", "d", "n_obs", "metodo", "semente", "c2st", "erro_media", "razao_dp",
              "esd", "ess_min", "ess_por_mil_esd", "rhat_max", "aceitacao", "passos", "segundos", "falhas_simulacao"]
ps[[c for c in colunas_ps if c in ps.columns]].sort_values(["classe", "problema", "metodo", "semente"]).to_csv(
    D / "02_resultados_por_semente.csv", index=False, float_format="%.6g")
pr.to_csv(D / "01_modelos_e_referencias.csv", index=False, float_format="%.6g")

# por modelo x metodo (media sobre sementes)
mm = (ps.groupby(["classe", "rotulo", "problema", "d", "metodo"], observed=True)
        .agg(sementes=("semente", "nunique"), c2st_media=("c2st", "mean"), c2st_dp=("c2st", "std"),
             c2st_min=("c2st", "min"), c2st_max=("c2st", "max"), erro_media=("erro_media", "mean"),
             razao_dp=("razao_dp", "mean"), esd=("esd", "mean"), ess_por_mil_esd=("ess_por_mil_esd", "mean"),
             rhat_max=("rhat_max", "max"), aceitacao=("aceitacao", "mean"), segundos=("segundos", "mean"))
        .reset_index())
mm.to_csv(D / "03_resumo_modelo_metodo.csv", index=False, float_format="%.6g")
piv = mm.pivot_table(index=["classe", "rotulo", "d"], columns="metodo", values="c2st_media", observed=True)
piv = piv.reindex(columns=[m for m in METODOS + ["npe_pfn"] if m in piv.columns])
piv.round(3).to_csv(D / "04_c2st_modelo_x_metodo.csv")

# por classe x metodo (sobre os modelos; cada modelo = media das sementes)
lin = []
for (cl, me), g in mm[mm.metodo.isin(METODOS)].groupby(["classe", "metodo"], observed=True):
    v = g.c2st_media.values
    lin.append(dict(classe=cl, metodo=me, n_modelos=len(v), c2st_q1=np.quantile(v, .25), c2st_mediana=np.median(v),
                    c2st_q3=np.quantile(v, .75), frac_c2st_menor_0_6=(v < 0.6).mean(),
                    erro_media_mediana=g.erro_media.median(), razao_dp_mediana=g.razao_dp.median(),
                    ess_por_mil_esd_mediana=g.ess_por_mil_esd.median(), rhat_max_mediano=g.rhat_max.median()))
pd.DataFrame(lin).to_csv(D / "05_resumo_classe_metodo.csv", index=False, float_format="%.4g")
lin = []
for me, g in mm[mm.metodo.isin(METODOS)].groupby("metodo"):
    v = g.c2st_media.values
    lin.append(dict(metodo=me, n_modelos=len(v), c2st_q1=np.quantile(v, .25), c2st_mediana=np.median(v),
                    c2st_q3=np.quantile(v, .75), c2st_media=v.mean(), frac_c2st_menor_0_6=(v < 0.6).mean(),
                    frac_c2st_menor_0_55=(v < 0.55).mean(), erro_media_mediana=g.erro_media.median(),
                    ess_por_mil_esd_mediana=g.ess_por_mil_esd.median(),
                    cadeias_rhat_maior_1_05=int((ps[(ps.metodo == me)].rhat_max > 1.05).sum()) if me not in ("laplace", "npe") else np.nan,
                    cadeias_total=int((ps.metodo == me).sum())))
glob = pd.DataFrame(lin)
glob.to_csv(D / "06_resumo_global_metodo.csv", index=False, float_format="%.4g")
num["global"] = glob.set_index("metodo").to_dict("index")

# ---------------------------------------------------------------- variacao entre sementes
dpm = mm[mm.metodo.isin(METODOS)].groupby("metodo").c2st_dp.median()
num["dp_semente_mediano"] = dpm.round(4).to_dict()
num["dp_semente_q90"] = mm[mm.metodo.isin(METODOS)].groupby("metodo").c2st_dp.quantile(.9).round(4).to_dict()

# ---------------------------------------------------------------- testes pareados (modelo = unidade)
W = piv[METODOS].dropna()          # 41 modelos x 6 metodos
rk = W.rank(axis=1, method="average")
num["rank_medio"] = rk.mean().round(3).to_dict()
fr = stats.friedmanchisquare(*[W[m].values for m in METODOS])
num["friedman"] = dict(chi2=float(fr.statistic), p=float(fr.pvalue), n=int(len(W)), k=len(METODOS))
pares = []
for a, b in itertools.combinations(METODOS, 2):
    d = W[a] - W[b]
    try:
        w = stats.wilcoxon(W[a], W[b], zero_method="wilcox")
        p = float(w.pvalue)
    except ValueError:
        p = 1.0
    pares.append(dict(metodo_a=a, metodo_b=b, mediana_dif_c2st=float(np.median(d)), a_melhor=int((d < -0.005).sum()),
                      b_melhor=int((d > 0.005).sum()), empate=int((d.abs() <= 0.005).sum()), p_wilcoxon=p))
pp = pd.DataFrame(pares).sort_values("p_wilcoxon")
m = len(pp)
holm = []
run = 0.0
for i, p in enumerate(pp.p_wilcoxon.values):
    run = max(run, min(1.0, (m - i) * p))
    holm.append(run)
pp["p_holm"] = holm
pp.to_csv(D / "07_testes_pareados_metodos.csv", index=False, float_format="%.4g")
W2 = W.drop(index=[i for i in W.index if i[1].startswith("Elowitz")])
fr2 = stats.friedmanchisquare(*[W2[m].values for m in METODOS])
num["friedman_sem_elowitz"] = dict(chi2=float(fr2.statistic), p=float(fr2.pvalue), n=int(len(W2)))
# vitorias por modelo (empate +-0,01 conta para todos)
best = W.min(axis=1)
num["vitorias"] = {m: int((W[m] <= best + 0.01).sum()) for m in METODOS}

# ---------------------------------------------------------------- NPE-PFN piloto
if "npe_pfn" in piv.columns:
    pf = piv.dropna(subset=["npe_pfn"])
    # npe com a semente 0 apenas, para comparar semente a semente
    s0 = ps[(ps.semente == 0) & ps.metodo.isin(["npe", "npe_pfn", "metropolis"])].pivot_table(
        index=["classe", "rotulo", "d"], columns="metodo", values="c2st", observed=True).dropna(subset=["npe_pfn"])
    s0.round(3).to_csv(D / "08_npe_pfn_piloto_semente0.csv")
    dif = s0.npe - s0.npe_pfn
    try:
        wp = float(stats.wilcoxon(s0.npe, s0.npe_pfn).pvalue)
    except ValueError:
        wp = float("nan")
    num["pfn"] = dict(n=int(len(s0)), mediana_npe=float(s0.npe.median()), mediana_pfn=float(s0.npe_pfn.median()),
                      mediana_metropolis=float(s0.metropolis.median()), pfn_melhor=int((dif > 0.01).sum()),
                      npe_melhor=int((dif < -0.01).sum()), empate=int((dif.abs() <= 0.01).sum()),
                      mediana_dif=float(dif.median()), p_wilcoxon=wp)
    t = ps[ps.metodo == "npe_pfn"]
    num["pfn"]["segundos_mediano"] = float(t.segundos.median())
    num["pfn"]["segundos_max"] = float(t.segundos.max())

# ---------------------------------------------------------------- tempos por etapa (registros)
linhas = []
for mod in sorted(p.name for p in R.iterdir() if (p / "registro.jsonl").exists()):
    for ln in open(R / mod / "registro.jsonl", encoding="utf-8"):
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if e.get("evento") == "fim" and "etapa" in e and "segundos" in e:
            linhas.append(dict(problema=mod, etapa=e["etapa"], segundos=e["segundos"], quando=e.get("t") or e.get("hora") or ""))
tt = pd.DataFrame(linhas)
cl = pr.set_index("problema").classe
tt["classe"] = tt.problema.map(cl)
tt.to_csv(D / "09_tempos_por_etapa_registro.csv", index=False, float_format="%.4g")
tab = (tt[tt.segundos < 20000].pivot_table(index="classe", columns="etapa", values="segundos", aggfunc="median").round(0))
tab.to_csv(D / "10_tempo_mediano_etapa_classe.csv")

# ---------------------------------------------------------------- identificabilidade
for src, dst in [("identificabilidade.csv", "11_identificabilidade_parametros.csv"),
                 ("identificabilidade_por_modelo.csv", "12_identificabilidade_por_modelo.csv"),
                 ("intervalos_de_confianca.csv", "13_intervalos_de_confianca.csv"),
                 ("npe_curva.csv", "14_npe_curva_orcamento.csv")]:
    pd.read_csv(REL / src).to_csv(D / dst, index=False, float_format="%.6g")

# ---------------------------------------------------------------- piso de ruido do C2ST (referencia contra ela mesma)
import torch  # noqa: E402
from sbi.utils.metrics import c2st  # noqa: E402

torch.set_num_threads(8)
linhas = []
for mod in sorted(pr.problema):
    p = json.load(open(R / mod / "problema.json"))
    lo, hi = np.array(p["lo"], float), np.array(p["hi"], float)
    ref = np.load(R / mod / "referencia.npz")["amostras"]
    n = 1000
    vals = []
    for k in range(4):
        a = (ref[k * 2 * n: k * 2 * n + n] - lo) / (hi - lo)
        b = (ref[k * 2 * n + n: k * 2 * n + 2 * n] - lo) / (hi - lo)
        torch.manual_seed(k)
        vals.append(float(c2st(torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32))))
    linhas.append(dict(problema=mod, rotulo=p["rotulo"], classe=cl[mod], d=int(p["d"]), c2st_nulo_media=np.mean(vals),
                       c2st_nulo_max=np.max(vals), c2st_nulo_dp=np.std(vals, ddof=1), pares=len(vals)))
    print("piso", mod, round(np.mean(vals), 3), flush=True)
nu = pd.DataFrame(linhas)
nu.to_csv(D / "15_piso_de_ruido_c2st.csv", index=False, float_format="%.4g")
num["piso"] = dict(mediana=float(nu.c2st_nulo_media.median()), q90=float(nu.c2st_nulo_media.quantile(.9)),
                   maximo=float(nu.c2st_nulo_media.max()), maximo_modelo=nu.loc[nu.c2st_nulo_media.idxmax(), "rotulo"],
                   por_classe=nu.groupby("classe").c2st_nulo_media.median().round(3).to_dict())

json.dump(num, open(OUT / "numeros.json", "w"), indent=1, ensure_ascii=False, default=float)
print("ok")
