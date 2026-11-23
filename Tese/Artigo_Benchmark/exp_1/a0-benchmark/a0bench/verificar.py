"""Verificacao da instalacao nesta maquina: roda, em poucos minutos, um pouco de cada parte do pacote.

  python -m a0bench verificar              tudo, menos o NPE-PFN
  python -m a0bench verificar --pfn        inclui o NPE-PFN (baixa os pesos do TabPFN na primeira vez)
  python -m a0bench verificar --sem-petab  pula a compilacao de um modelo PEtab pelo AMICI

Cada etapa termina em ok, aviso ou falha. O comando devolve codigo 1 se alguma etapa falhar.
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

NLL_BOEHM = 138.2220     # -log L do Boehm no ponto nominal da colecao de benchmarks


class _Pausa(Exception):
    pass


def _memoria_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def _sistema():
    import platform

    v = sys.version_info
    mem = _memoria_gb()
    txt = (f"Python {v.major}.{v.minor}.{v.micro}, {platform.system()} {platform.machine()}, "
           f"{os.cpu_count()} núcleos" + (f", {mem:.0f} GB de memória" if mem else ""))
    if v < (3, 12):
        return "falha", txt + ". O pacote pede Python 3.12 ou mais novo."
    if platform.system() == "Windows":
        return "aviso", txt + ". Não testado no Windows; use o WSL2."
    if mem and mem < 1.5 * (os.cpu_count() or 1):
        return "aviso", txt + (f". Reserve 1,5 GB por processo: com esta memória, rode com "
                               f"--processos {max(1, int(mem // 1.5))} ou menos.")
    return "ok", txt


def _pacotes():
    from importlib import metadata

    arq = Path(__file__).parent.parent / "requirements.txt"
    pedidos = {}
    if arq.exists():
        for linha in arq.read_text().splitlines():
            linha = linha.split("#")[0].strip()
            if "==" in linha:
                nome, versao = linha.split("==")
                pedidos[nome.split("[")[0].strip()] = versao.strip()
    from .util import PACOTES

    faltam, diferentes = [], []
    for nome in PACOTES:
        try:
            v = metadata.version(nome)
        except metadata.PackageNotFoundError:
            faltam.append(nome)
            continue
        if nome in pedidos and pedidos[nome] != v:
            diferentes.append(f"{nome} {v} (testado com {pedidos[nome]})")
    if faltam:
        obrig = [n for n in faltam if n != "tabpfn"]
        if obrig:
            return "falha", "faltam: " + ", ".join(faltam) + ". Rode pip install -r requirements.txt"
        return "aviso", "falta o tabpfn: o NPE-PFN não vai rodar (use --sem-pfn ou instale sbi[tabpfn])"
    if diferentes:
        return "aviso", "versões diferentes das testadas: " + "; ".join(diferentes)
    return "ok", f"{len(PACOTES)} pacotes nas versões do requirements.txt" if pedidos else f"{len(PACOTES)} pacotes instalados"


def _simuladores():
    from . import modelos

    t0 = time.perf_counter()
    partes = []
    for nome in ("lotka_volterra", "robertson", "calor"):
        m = modelos.POR_NOME[nome]
        y = m["sim"](np.tile(m["verd"], (50, 1)))
        if not np.all(np.isfinite(y)):
            return "falha", f"{nome}: a simulação devolveu valores não finitos"
        t = time.perf_counter()
        m["sim"](np.tile(m["verd"], (200, 1)))
        partes.append(f"{nome} {1e6 * (time.perf_counter() - t) / 200:.0f} µs")
    return "ok", f"compilados em {time.perf_counter() - t0:.0f} s; por solução: " + ", ".join(partes)


def _retomada(pasta):
    """Cada amostrador roda de uma vez e com uma pausa forcada no meio; as cadeias tem de sair identicas."""
    from . import cadeias, config, modelos, problemas

    p = problemas.Sintetico(modelos.POR_NOME["lotka_volterra"])
    cfg = config.montar("rapido", sementes=[0, 1])
    cfg.update(orcamento_esd=1600, passos_ck=37, n_amostras=200)
    mapa = p.mapa(dict(n_inicios=2))

    def parar(f, *_):
        if f >= 0.5:
            raise _Pausa

    for nome in config.AMOSTRADORES:
        cadeias.rodar_amostrador(nome, p, mapa, cfg, pasta / "direto")
        try:
            cadeias.rodar_amostrador(nome, p, mapa, cfg, pasta / "pausado", parar)
            return "falha", f"{nome}: a pausa forçada não aconteceu"
        except _Pausa:
            pass
        _, retomado = cadeias.rodar_amostrador(nome, p, mapa, cfg, pasta / "pausado")
        if not retomado:
            return "falha", f"{nome}: a cadeia recomeçou do zero em vez de continuar"
        a = np.load(pasta / "direto" / "cadeias" / f"{nome}.npz")
        b = np.load(pasta / "pausado" / "cadeias" / f"{nome}.npz")
        if not (np.array_equal(a["cadeia"], b["cadeia"]) and np.array_equal(a["logp"], b["logp"])):
            return "falha", f"{nome}: a cadeia retomada difere da cadeia rodada de uma vez"
    return "ok", "os quatro amostradores continuam do estado salvo e dão os mesmos números, bit a bit"


def _ponta_a_ponta(saida, processos):
    from . import config, executor
    from .util import ler_json

    nomes = ["oscilador", "calor"]
    cfg = config.montar("rapido", modelos=nomes, sementes=[0, 1])
    cfg["metodos"] = [m for m in config.METODOS if m != "npe_pfn"]
    cfg.update(n_ref=1500, n_ref_max=1500, orcamento_esd=1500, n_sim=300, n_amostras=400, n_c2st=200, n_sol=40,
               n_grade=15, perfil_passos_max=8, n_lis=40, n_pred_ref=60)
    t0 = time.time()
    print("  (orçamento mínimo, só para exercitar o código: os avisos de referência não convergida são esperados)")
    executor.rodar(cfg, saida, processos, None, monitor=False)
    problemas = []
    for n in nomes:
        est = ler_json(saida / n / "estado.json") or {}
        problemas += [f"{n}: {e}" for e in est.get("erros", [])]
        if est.get("situacao") != "concluido" and not est.get("erros"):
            problemas.append(f"{n}: situação {est.get('situacao')}")
        for arq in ("problema.json", "referencia.npz", "identificabilidade.json", "solucao.npz", "superficie.npz",
                    *[f"cadeias/{a}.npz" for a in config.AMOSTRADORES]):
            if not (saida / n / arq).exists():
                problemas.append(f"{n}: falta {arq}")
        for m in cfg["metodos"]:
            for s in cfg["sementes"]:
                if not (ler_json(saida / n / f"{m}_s{s}.json") or {}).get("ok"):
                    problemas.append(f"{n}: {m} (semente {s}) não terminou bem")
    if problemas:
        return "falha", "; ".join(problemas[:6])
    return "ok", (f"{len(nomes)} modelos (uma EDO e uma EDP), MAP, referência, identificabilidade, soluções e "
                  f"{len(cfg['metodos'])} métodos em {time.time() - t0:.0f} s")


def _petab(pasta_modelos):
    import logging

    for nome in ("pypesto", "amici", "petab"):
        logging.getLogger(nome).setLevel(logging.CRITICAL)
    from . import config, problemas

    t0 = time.time()
    p = problemas.construir("petab", "Boehm_JProteomeRes2014", "petab", config.montar("rapido"), pasta_modelos)
    f = float(p.obj(p.xn, sensi_orders=(0,)))
    ll, g = p.grad_lote(np.clip(p.xn, p.lo, p.hi)[None])
    txt = (f"Boehm compilado e simulado em {time.time() - t0:.0f} s; -log L no ponto nominal {f:.4f} "
           f"(esperado {NLL_BOEHM:.4f}); {1e3 * p.t_f:.1f} ms por solução")
    if abs(f - NLL_BOEHM) > 1e-2:
        return "falha", txt
    if not (np.isfinite(ll[0]) and np.all(np.isfinite(g[0]))):
        return "falha", txt + "; o gradiente não é finito"
    return "ok", txt


def _pfn():
    from . import inferencia, modelos, problemas

    from . import config

    cfg = config.montar("completo")
    p = problemas.Sintetico(modelos.POR_NOME["oscilador"])
    rng = np.random.default_rng(0)
    th = p.amostrar_prior(cfg["n_ctx_pfn"], rng)
    x, ok = p.simular_lote(th, rng)
    banco = dict(theta=th[ok], x=x[ok], n_pedido=len(th), falhas=0)
    t0 = time.time()
    s, n_ctx = inferencia.amostrar_npe_pfn(p, banco, cfg["n_amostras"], 0, cfg["n_ctx_pfn"])
    seg = time.time() - t0
    if s.shape != (cfg["n_amostras"], p.d) or not np.all(np.isfinite(s)):
        return "falha", f"amostras com forma {s.shape} ou valores não finitos"
    return "ok", (f"{len(s)} amostras com {n_ctx:.0f} simulações de contexto em {seg:.0f} s, num modelo de 3 parâmetros "
                  f"({100 * p.dentro(s).mean():.0f}% dentro do prior). É o tamanho usado no preset completo: conte "
                  f"esse tempo por semente e por modelo, e mais nos modelos com mais parâmetros")


def _figuras(saida):
    import warnings

    warnings.filterwarnings("ignore")
    import matplotlib

    matplotlib.use("Agg")
    from . import graficos, resultados

    tex = graficos.estilo(None)
    out = saida / "relatorio"
    out.mkdir(parents=True, exist_ok=True)
    tab, ident = resultados.tabela(saida), resultados.identificabilidade(saida)
    feitas, erros = graficos.relatorio_geral(saida, out, tab, ident)
    n = len(feitas)
    for nome in resultados.problemas(saida)["problema"]:
        _, f, e = graficos.relatorio_modelo(str(saida), nome, str(out / "modelos" / nome), ("png",), tex)
        n += len(f)
        erros += [f"{nome}: {x}" for x in e]
    fonte = "com LaTeX" if tex else "sem LaTeX (fonte STIX; para Computer Modern instale LaTeX, cm-super e dvipng)"
    if erros:
        return "falha", f"{len(erros)} figuras falharam ({fonte}): " + "; ".join(str(e)[:120] for e in erros[:3])
    return ("ok" if tex else "aviso"), f"{n} figuras, {fonte}"


def _arquivo(saida):
    from . import resultados

    destino, m = resultados.exportar(saida)
    dif = resultados.conferir(saida)
    if dif:
        return "falha", f"{len(dif)} arquivos diferem do manifesto"
    r = resultados.carregar(destino)
    if sorted(r.modelos) != ["calor", "oscilador"] or r.tabela().empty or r.cadeia("oscilador", "hmc") is None:
        return "falha", "o arquivo exportado não reabre com todos os resultados"
    return "ok", f"{m['n_arquivos']} arquivos exportados, conferidos pelo sha256 e reabertos com carregar()"


def rodar(pfn=False, sem_petab=False, processos=None, pasta_modelos=None, manter=None):
    from rich import box
    from rich.console import Console
    from rich.table import Table

    con = Console()
    base = Path(manter) if manter else Path(tempfile.mkdtemp(prefix="a0bench_verificar_"))
    base.mkdir(parents=True, exist_ok=True)
    saida = base / "execucao"
    etapas = [("Python e sistema", _sistema),
              ("Pacotes", _pacotes),
              ("Simuladores (numba)", _simuladores),
              ("Pausa e retomada das cadeias", lambda: _retomada(base / "retomada")),
              ("Execução de ponta a ponta", lambda: _ponta_a_ponta(saida, processos or min(2, os.cpu_count() or 1))),
              ("Figuras e tabelas", lambda: _figuras(saida)),
              ("Exportar, conferir e reabrir", lambda: _arquivo(saida))]
    if not sem_petab:
        etapas.append(("PEtab e AMICI", lambda: _petab(pasta_modelos or base / "amici_models")))
    if pfn:
        etapas.append(("NPE-PFN (TabPFN)", _pfn))
    linhas = []
    for nome, f in etapas:
        con.print(f"[cyan]▶ {nome}[/cyan]")
        t0 = time.time()
        try:
            sit, txt = f()
        except Exception as e:  # noqa: BLE001  (a verificacao relata a falha e segue para a etapa seguinte)
            sit, txt = "falha", f"{type(e).__name__}: {str(e)[:400]}"
        linhas.append((nome, sit, txt, time.time() - t0))
    tab = Table(title="Verificação da instalação", box=box.SIMPLE_HEAD, header_style="bold")
    tab.add_column("")
    tab.add_column("Etapa", no_wrap=True)
    tab.add_column("Resultado")
    tab.add_column("Tempo", justify="right", no_wrap=True)
    icone = {"ok": "[bold green]✔[/]", "aviso": "[bold yellow]⚠[/]", "falha": "[bold red]✖[/]"}
    for nome, sit, txt, seg in linhas:
        tab.add_row(icone[sit], nome, txt, f"{seg:.0f} s")
    con.print(tab)
    if not pfn:
        con.print("[grey62]O NPE-PFN não foi testado. Para testar: python -m a0bench verificar --pfn[/]")
    falhas = [n for n, s, _, _ in linhas if s == "falha"]
    if manter:
        con.print(f"[grey62]Arquivos da verificação em {base}[/]")
    else:
        shutil.rmtree(base, ignore_errors=True)
    if falhas:
        con.print(f"[bold red]{len(falhas)} etapa(s) com falha: {', '.join(falhas)}.[/]")
        return 1
    con.print("[bold green]Instalação pronta para rodar o benchmark.[/]")
    return 0
