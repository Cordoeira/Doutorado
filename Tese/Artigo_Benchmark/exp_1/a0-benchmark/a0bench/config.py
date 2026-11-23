"""Configuracao, presets e selecao de modelos."""

import os
from pathlib import Path

# pasta com os dados reais, uma subpasta por modelo (CSV curado ou os arquivos PEtab com as medidas). Padrao: a
# pasta dados/ na raiz do repositorio; a variavel A0_DADOS aponta para outra copia
DIR_DADOS = Path(os.environ.get("A0_DADOS") or Path(__file__).resolve().parent.parent / "dados")

AMOSTRADORES = ["metropolis", "ensemble", "mala", "hmc"]
METODOS = ["laplace", *AMOSTRADORES, "npe", "npe_pfn"]
VERSAO_IDENT = 4   # versao do formato de identificabilidade.json; resultados mais antigos sao refeitos

# nome na colecao Benchmark-Models-PEtab -> grupo
PETAB = {
    "Bertozzi_PNAS2020": "petab",
    "Boehm_JProteomeRes2014": "petab",
    "Rahman_MBS2016": "petab",
    "Crauste_CellSystems2017": "petab",
    "Bruno_JExpBot2016": "petab_grande",
    "Sneyd_PNAS2002": "petab_grande",
    "Fujita_SciSignal2010": "petab_grande",
    "Elowitz_Nature2000": "petab_grande",
    # acrescentados na versao com dados reais (mesma colecao; grupo pelo numero de parametros, como os acima)
    "Perelson_Science1996": "petab",
    "SalazarCavazos_MBoC2020": "petab",
    "Blasi_CellSystems2016": "petab",
    "Liu_IFACPapersOnLine2025": "petab",
    "Oliveira_NatCommun2021": "petab",
    "Laske_PLOSComputBiol2019": "petab_grande",
    "Armistead_CellDeathDis2024": "petab_grande",
    "Okuonghae_ChaosSolitonsFractals2020": "petab_grande",
}
GRUPOS = ["facil", "dificil", "petab", "petab_grande"]

CFG = dict(
    grupos=["facil", "dificil", "petab", "petab_grande"],
    modelos=None,                 # lista de nomes; None = todos os dos grupos
    dados="reais",                # "reais" (medidas publicadas, em dados/<modelo>), "sinteticos" ou "todos"
    metodos=list(METODOS),
    identificabilidade=True,
    sementes=[0, 1, 2],
    largura=1.0,                  # meia-largura (log10) da caixa do prior nos problemas PEtab
    n_ref=50000,                  # passos iniciais da referencia (dobra ate convergir)
    n_ref_max=400000,
    n_cadeias_ref=2,
    betas=[1.0, 0.5, 0.25, 0.1, 0.03, 0.01],
    rhat_max=1.05,
    ess_min=400,
    orcamento_esd=50000,          # custo de cada cadeia dos quatro amostradores
    aquecimento_grad=0.5,         # fracao da cadeia usada para adaptar MALA e HMC
    hmc_passos_max=8,
    n_sim=10000,
    lote_treino=500,
    n_ctx_pfn=2000,               # simulacoes de contexto do NPE-PFN (na CPU, no maximo 1000)
    dispositivo_pfn="auto",       # "auto" usa a GPU se houver e volta para a CPU se ela falhar; "cpu"; "cuda"
    n_amostras=2000,
    n_c2st=1000,
    n_inicios=8,
    perfil_passos_max=30,         # passos por lado da varredura local do perfil
    n_perfil_global=9,            # pontos da grade regular da varredura global do perfil
    n_ref_perfil=2000,            # amostras da referencia usadas como inicios dos perfis
    n_pts_estrutura=16,           # pontos do prior para o posto do jacobiano e o subespaco livre de dados
    tol_invariancia=1e-3,
    n_lis=300,                    # amostras da referencia para o subespaco informado
    n_pred_ref=400,               # amostras da referencia para a banda de predicao
    perfil_polimento=True,        # tenta de novo cada ponto do perfil a partir do otimo e dos vizinhos
    perfil_inicios_extras=0,      # inicios sorteados do prior somados na varredura global e no polimento
    nivel_confianca=0.95,         # nivel dos intervalos de confianca (Raue et al. usam 1 sigma, 0,683, na figura 3)
    passos_ck=1000,               # a cadeia e o estado do amostrador vao para o disco a cada n passos
    desbaste_cadeias=1,           # guarda 1 a cada k passos das cadeias dos amostradores (1 = tudo)
    pontos_ref=20000,             # maximo de pontos guardados por cadeia do traco da referencia
    solucoes=True,                # grava solucoes e superficies para os graficos
    n_sol=200,                    # amostras da referencia para a faixa a posteriori das solucoes
    n_traj=40,                    # trajetorias guardadas para o espaco de fase
    n_grade=41,                   # pontos por eixo das superficies de contorno
    n_pares=3,                    # pares de parametros com superficie
)

PRESETS = {
    "completo": {},
    "reais": {},                  # igual ao completo (que ja usa so dados reais); mantido por compatibilidade
    # trilha de validacao, separada: so os modelos sinteticos, com verdade conhecida
    "sinteticos": dict(dados="sinteticos", grupos=["facil", "dificil"]),
    "padrao": dict(sementes=[0, 1], n_sim=5000, n_ref_max=200000),
    "rapido": dict(sementes=[0, 1], n_ref=4000, n_ref_max=4000, orcamento_esd=6000, n_sim=1500,
                   perfil_passos_max=12, n_perfil_global=5, perfil_polimento=False, n_lis=100, n_pred_ref=150,
                   n_sol=100, n_grade=25,
                   grupos=["facil", "dificil", "petab"]),
}


def montar(preset="completo", **extra):
    cfg = {**CFG, **PRESETS[preset]}
    cfg.update({k: v for k, v in extra.items() if v is not None})
    cfg["preset"] = preset
    return cfg


def etapas(cfg):
    e = ["map", "referencia"]
    if cfg["identificabilidade"]:
        e.append("identificabilidade")
    if cfg.get("solucoes", True):
        e.append("solucoes")
    return e + [m for m in METODOS if m in cfg["metodos"]]


DADOS = ["todos", "reais", "sinteticos"]


def dados_reais(nome):
    """True se o modelo usa medidas publicadas (os PEtab e os de modelos.reais())."""
    from . import modelos

    return nome in PETAB or modelos.POR_NOME.get(nome, {}).get("dados") is not None


def selecionar(cfg):
    """Lista de (origem, nome, grupo) na ordem de execucao (os mais caros primeiro). Com `modelos`, vale a lista;
    sem ela, os modelos dos `grupos`, filtrados por `dados` (reais, sinteticos ou todos)."""
    from . import modelos

    itens = [("petab", n, g) for n, g in PETAB.items()]
    itens += [("sint", m["nome"], m["grupo"]) for m in modelos.MODELOS]
    if cfg["modelos"]:
        pedidos = set(cfg["modelos"])
        faltam = pedidos - {n for _, n, _ in itens}
        if faltam:
            raise SystemExit(f"Modelos desconhecidos: {', '.join(sorted(faltam))}. Use 'listar'.")
        itens = [i for i in itens if i[1] in pedidos]
    else:
        itens = [i for i in itens if i[2] in cfg["grupos"]]
        if cfg.get("dados", "reais") != "todos":
            reais = set(PETAB) | {m["nome"] for m in modelos.MODELOS if m.get("dados") is not None}
            itens = [i for i in itens if (i[1] in reais) == (cfg["dados"] == "reais")]
    peso = {"petab_grande": 0, "petab": 1, "dificil": 2, "facil": 3}
    return sorted(itens, key=lambda i: peso[i[2]])
