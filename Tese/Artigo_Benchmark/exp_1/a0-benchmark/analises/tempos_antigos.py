import pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
D = str(REPO / "entrega_orientador" / "dados") + "/"
tt = pd.read_csv(D + "09_tempos_por_etapa_registro.csv")
pr = pd.read_csv(D + "01_modelos_e_referencias.csv").set_index("problema")
# ultima ocorrencia de cada etapa por modelo (retomadas geram varias linhas: soma)
tab = tt.groupby(["problema", "etapa"]).segundos.sum().unstack()
cols = ["map", "referencia", "identificabilidade", "solucoes", "laplace", "metropolis", "ensemble", "mala", "hmc", "npe", "npe_pfn"]
tab = tab.reindex(columns=cols)
tab["t_f_ms"] = pr.t_f_ms
tab["c_s"] = pr.custo_grad
tab["d"] = pr.d
tab["ref_passos"] = pr.ref_passos
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
sel = ["lotka_volterra", "seir", "seir_inicio", "pk_flipflop", "pk2", "fisher_kpp", "fisher_frente", "Bertozzi_PNAS2020", "Boehm_JProteomeRes2014",
       "Rahman_MBS2016", "Crauste_CellSystems2017", "Bruno_JExpBot2016", "Sneyd_PNAS2002", "Elowitz_Nature2000", "Fujita_SciSignal2010"]
print(tab.loc[sel].round(1).to_string())
print()
# contagens de registros por etapa (retomadas)
print(tt.groupby(["problema"]).size().loc[sel].to_string())
