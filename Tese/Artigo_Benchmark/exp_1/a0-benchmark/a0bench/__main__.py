"""Linha de comando do A0.

  python -m a0bench verificar   [--pfn] [--sem-petab]
  python -m a0bench listar
  python -m a0bench estimar     [--preset completo] [--processos 8]
  python -m a0bench rodar       [--preset completo] [--saida resultados] [--processos 8]
  python -m a0bench identificar [--saida resultados] [--processos 8]
  python -m a0bench monitor     [--saida resultados]
  python -m a0bench relatorio   [--saida resultados]
  python -m a0bench exportar    [--saida resultados]
  python -m a0bench conferir    [--saida resultados]
"""
import argparse
import os
import sys
import time
from pathlib import Path

from . import config

# problemas PEtab na maquina de referencia: tempo por solucao (ms) no ponto nominal e na media do prior,
# custo do gradiente em solucoes, e numero de parametros
T_PETAB = {"Bertozzi_PNAS2020": (0.70, 1.21, 3.3, 8), "Boehm_JProteomeRes2014": (1.08, 1.12, 9.1, 9),
           "Rahman_MBS2016": (0.88, 0.90, 8.6, 9), "Crauste_CellSystems2017": (0.96, 4.45, 11.6, 12),
           "Bruno_JExpBot2016": (2.03, 2.19, 4.2, 13), "Sneyd_PNAS2002": (6.82, 7.30, 17.0, 15),
           "Fujita_SciSignal2010": (5.60, 6.25, 52.8, 19), "Elowitz_Nature2000": (2.39, 2.81, 17.8, 21),
           # medidos em 16/11/2026 no laptop da execucao completa (mediana no nominal e em 8 pontos do prior)
           "Perelson_Science1996": (0.36, 0.36, 3.2, 3), "SalazarCavazos_MBoC2020": (19.0, 19.0, 106.1, 6),
           "Blasi_CellSystems2016": (0.80, 0.80, 17.0, 9), "Liu_IFACPapersOnLine2025": (0.75, 0.75, 6.7, 9),
           "Oliveira_NatCommun2021": (0.94, 0.94, 7.4, 12), "Laske_PLOSComputBiol2019": (10.0, 10.0, 23.7, 13),
           "Armistead_CellDeathDis2024": (0.87, 0.87, 7.9, 14),
           "Okuonghae_ChaosSolitonsFractals2020": (0.82, 0.82, 7.5, 16)}
T_REF_LV = 30e-6          # Lotka-Volterra na maquina de referencia (s por solucao)
S_NPE_POR_SIM = 0.007     # treino do NPE: segundos por simulacao do banco, por semente
S_PFN_POR_SEMENTE = 60.0  # NPE-PFN em GPU: suposicao, nao medido


def _lista(txt, tipo=str):
    return [tipo(x) for x in txt.split(",") if x.strip()] if txt else None


def _nivel(txt):
    """Nivel de confianca: fracao (0.95) ou numero de desvios de uma normal (1sigma = 0,6827, 2sigma = 0,9545)."""
    t = str(txt).strip().lower().replace(",", ".")
    if t.endswith("sigma"):
        from math import erf, sqrt

        return erf(float(t[:-5] or 1) / sqrt(2.0))
    return min(max(float(t), 0.01), 0.9999)


def _conhecidos(nomes, validos, o_que):
    ruins = [n for n in (nomes or []) if n not in validos]
    if ruins:
        raise SystemExit(f"{o_que} desconhecido(s): {', '.join(ruins)}. Opções: {', '.join(validos)}.")


def _cfg(a):
    _conhecidos(_lista(a.grupos), config.GRUPOS, "Grupo(s)")
    _conhecidos(_lista(a.metodos), config.METODOS, "Método(s)")
    extra = dict(grupos=_lista(a.grupos), modelos=_lista(a.modelos), sementes=_lista(a.sementes, int),
                 dados=getattr(a, "dados", None))
    cfg = config.montar(a.preset, **extra)
    if a.metodos:
        cfg["metodos"] = [m for m in config.METODOS if m in _lista(a.metodos)]
    if a.sem_pfn:
        cfg["metodos"] = [m for m in cfg["metodos"] if m != "npe_pfn"]
    if getattr(a, "nivel", None):
        cfg["nivel_confianca"] = _nivel(a.nivel)
    if getattr(a, "sem_identificabilidade", False):
        cfg["identificabilidade"] = False
    if getattr(a, "salvar_a_cada", None):
        cfg["passos_ck"] = max(1, a.salvar_a_cada)
    if getattr(a, "desbaste", None):
        cfg["desbaste_cadeias"] = max(1, a.desbaste)
    return cfg


def cmd_listar(a):
    from rich import box
    from rich.console import Console
    from rich.table import Table

    from . import modelos
    from .monitor import COR_GRUPO

    tab = Table(title="Modelos do A0", box=box.SIMPLE_HEAD, header_style="bold")
    for c in ("Nome", "Rótulo", "Grupo", "Tipo", "d", "Dados", "Desafio"):
        tab.add_column(c, justify="right" if c == "d" else "left")
    for n, g in config.PETAB.items():
        tab.add_row(n, n.split("_")[0] + " (PEtab)", f"[{COR_GRUPO[g]}]{g}[/]", "EDO", "", "reais", "")
    for m in modelos.MODELOS:
        tab.add_row(m["nome"], m["rotulo"], f"[{COR_GRUPO[m['grupo']]}]{m['grupo']}[/]", m["tipo"], str(len(m["lo"])),
                    "reais" if m.get("dados") is not None else "sintéticos", m.get("desafio", ""))
    Console().print(tab)
    Console().print(f"[grey62]{len(config.PETAB) + len(modelos.MODELOS)} modelos. Grupos: {', '.join(config.GRUPOS)}[/]")


def estimar(cfg, processos, medir=True):
    """Segundos previstos por modelo e tempo de parede com `processos` processos."""
    import numpy as np

    from . import modelos

    fator = 1.0
    if medir:
        m = modelos.POR_NOME["lotka_volterra"]
        m["sim"](m["verd"][None])
        t = time.perf_counter()
        m["sim"](np.tile(m["verd"], (2000, 1)))
        fator = (time.perf_counter() - t) / 2000 / T_REF_LV
    S, K, mref = len(cfg["sementes"]), len(cfg["betas"]), cfg["n_cadeias_ref"]
    n_am = sum(x in cfg["metodos"] for x in config.AMOSTRADORES)
    linhas = []
    for origem, nome, grupo in config.selecionar(cfg):
        if origem == "petab":
            tf, tp, cg, d = T_PETAB.get(nome, (3.0, 3.0, 15.0, 12))     # sem medida: valores tipicos
            tf, tp = tf * 1e-3 * fator, tp * 1e-3 * fator
            sobre = 0.0
        else:
            m = modelos.POR_NOME[nome]
            d = len(m["lo"])
            ponto = m["verd"] if m.get("verd") is not None else m["centro"]
            if medir:
                m["sim"](ponto[None])
                t = time.perf_counter()
                m["sim"](np.tile(ponto, (200, 1)))
                tf = (time.perf_counter() - t) / 200
            else:
                tf = 100e-6
            tp, cg = tf, d + 1.0
            sobre = (150e-6 * cfg["n_ref"] + n_am * 60e-6 * cfg["orcamento_esd"]) * fator
        ref = mref * K * cfg["n_ref"] * 0.5 * (tf + tp)
        outros = (n_am * S * cfg["orcamento_esd"] + cfg["n_inicios"] * 40 * cg) * tf
        if cfg["identificabilidade"]:   # fatores medidos em 37 modelos, com e sem o polimento dos perfis
            f_id = {(True, True): 3.3, (True, False): 1.2, (False, True): 1.7, (False, False): 0.65}[
                (origem == "petab", bool(cfg.get("perfil_polimento", True)))]
            outros += f_id * d * (cfg["perfil_passos_max"] + cfg["n_perfil_global"] + 8) * 12 * cg * tf
        n_met = sum(x in cfg["metodos"] for x in config.METODOS)
        amort = 0.0
        if "npe" in cfg["metodos"]:
            amort += cfg["n_sim"] * tp + S * cfg["n_sim"] * S_NPE_POR_SIM * fator
        pfn = S * S_PFN_POR_SEMENTE if "npe_pfn" in cfg["metodos"] else 0.0
        metr = n_met * S * 2.5 * fator
        base = ref + outros + amort + pfn + metr + sobre + 5.0
        linhas.append(dict(nome=nome, grupo=grupo, seg=base, seg_max=base + ref * (cfg["n_ref_max"] / cfg["n_ref"] - 1)))

    def parede(chave):
        cargas = [0.0] * processos
        for v in sorted((l[chave] for l in linhas), reverse=True):
            cargas[cargas.index(min(cargas))] += v
        return max(cargas) if cargas else 0.0
    return linhas, parede("seg"), parede("seg_max"), fator


def cmd_estimar(a):
    from rich import box
    from rich.console import Console
    from rich.table import Table

    from .monitor import COR_GRUPO, _tempo

    con = Console()
    cfg = _cfg(a)
    proc = a.processos or os.cpu_count() or 2
    with con.status("Medindo a velocidade dos simuladores nesta máquina..."):
        linhas, t_min, t_max, fator = estimar(cfg, proc)
    tab = Table(title=f"Estimativa de tempo (preset {a.preset}, {proc} processos)", box=box.SIMPLE_HEAD,
                header_style="bold")
    for c in ("Grupo", "Modelos", "Soma sequencial", "Se a referência precisar do máximo"):
        tab.add_column(c, justify="left" if c == "Grupo" else "right")
    for g in config.GRUPOS:
        sel = [l for l in linhas if l["grupo"] == g]
        if sel:
            tab.add_row(f"[{COR_GRUPO[g]}]{g}[/]", str(len(sel)), _tempo(sum(l["seg"] for l in sel)),
                        _tempo(sum(l["seg_max"] for l in sel)))
    tab.add_row("[bold]total[/]", f"[bold]{len(linhas)}[/]", f"[bold]{_tempo(sum(l['seg'] for l in linhas))}[/]",
                f"[bold]{_tempo(sum(l['seg_max'] for l in linhas))}[/]")
    con.print(tab)
    con.print(f"Tempo de parede com {proc} processos: [bold bright_cyan]{_tempo(t_min)}[/] "
              f"(até [bold]{_tempo(t_max)}[/] se todas as referências precisarem ser estendidas).")
    mais = max(linhas, key=lambda l: l["seg"])
    con.print(f"[grey62]Modelo mais caro: {mais['nome']} ({_tempo(mais['seg'])}); ele define o piso do tempo de parede.\n"
              f"Velocidade desta máquina: {1 / fator:.2f}x a de referência. Os tempos dos PEtab vêm de uma tabela "
              f"medida na máquina de referência; o NPE-PFN entra como {S_PFN_POR_SEMENTE:.0f} s por semente (suposição).[/]")


def cmd_rodar(a):
    from . import executor

    executor.rodar(_cfg(a), a.saida, a.processos, a.pasta_modelos, monitor=not a.sem_monitor)


def cmd_identificar(a):
    from . import executor

    cfg = _cfg(a)
    cfg.update(metodos=[], identificabilidade=True)
    executor.rodar(cfg, a.saida, a.processos, a.pasta_modelos, monitor=not a.sem_monitor)


def cmd_verificar(a):
    from . import verificar

    raise SystemExit(verificar.rodar(pfn=a.pfn, sem_petab=a.sem_petab, processos=a.processos,
                                     pasta_modelos=a.pasta_modelos, manter=a.manter))


def cmd_monitor(a):
    from . import monitor

    monitor.assistir(a.saida)


def cmd_relatorio(a):
    import warnings
    from concurrent.futures import ProcessPoolExecutor, as_completed

    warnings.filterwarnings("ignore")
    import matplotlib

    matplotlib.use("Agg")
    from rich.console import Console
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    from . import graficos, monitor, resultados

    con = Console()
    saida, out = Path(a.saida), Path(a.saida) / "relatorio"
    out.mkdir(parents=True, exist_ok=True)
    tab, ident = resultados.tabela(saida), resultados.identificabilidade(saida)
    probs = resultados.problemas(saida)
    if probs.empty:
        con.print("[yellow]Ainda não há resultados nesta pasta.[/]")
        return
    tex = graficos.estilo(False if a.sem_latex else None)
    probs.drop(columns=[c for c in ("nomes", "lo", "hi", "verd", "rotulos_obs") if c in probs]).to_csv(
        out / "problemas.csv", index=False)
    if not tab.empty:
        tab.to_csv(out / "resumo.csv", index=False)
        resultados.resultados(saida).drop(columns=["nomes", "lo", "hi", "verd", "rotulos_obs"], errors="ignore").to_csv(
            out / "resultados_por_semente.csv", index=False)
    if not ident.empty:
        ident.to_csv(out / "identificabilidade.csv", index=False)
        resultados.resumo_identificabilidade(saida).to_csv(out / "identificabilidade_por_modelo.csv", index=False)
        ic = resultados.intervalos(saida)
        if not ic.empty:
            ic.to_csv(out / "intervalos_de_confianca.csv", index=False)
    erros = []
    if a.so in (None, "geral"):
        with con.status("Figuras gerais..."):
            feitas, er = graficos.relatorio_geral(saida, out, tab, ident)
        erros += er
        con.print(f"[green]✔[/green] {len(feitas)} figuras gerais em {out}")
    if a.so in (None, "modelos"):
        nomes = [n for n in probs["problema"] if not a.modelos or n in a.modelos.split(",")]
        formatos = ("png", "pdf") if a.pdf else ("png",)
        proc = max(1, min(a.processos or 1, len(nomes)))
        barra = Progress(TextColumn("[bold]Figuras por modelo"), BarColumn(), MofNCompleteColumn(),
                         TimeElapsedColumn(), TextColumn("{task.fields[atual]}"), console=con)
        with barra, ProcessPoolExecutor(max_workers=proc) as ex:
            tarefa = barra.add_task("", total=len(nomes), atual="")
            fut = {ex.submit(graficos.relatorio_modelo, str(saida), n, str(out / "modelos" / n), formatos, tex): n
                   for n in nomes}
            total = 0
            for f in as_completed(fut):
                nome, feitas, er = f.result()
                total += len(feitas)
                erros += [f"{nome}: {e}" for e in er]
                barra.update(tarefa, advance=1, atual=nome)
        con.print(f"[green]✔[/green] {total} figuras de {len(nomes)} modelos em {out / 'modelos'}")
    for e in erros[:20]:
        con.print(f"[red]✖ {e}[/red]")
    monitor.resumo(saida, con)
    con.print(f"[green]Relatório em {out}[/] ({'LaTeX' if tex else 'fonte STIX; instale LaTeX para Computer Modern'})")


def cmd_exportar(a):
    from rich.console import Console
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    from . import resultados

    con = Console()
    if not (Path(a.saida) / "config.json").exists():
        con.print(f"[yellow]Não há execução em {a.saida}.[/]")
        return
    with Progress(TextColumn("[bold]Empacotando"), BarColumn(), MofNCompleteColumn(), console=con) as barra:
        t = barra.add_task("", total=None)
        destino, m = resultados.exportar(a.saida, a.destino, com_cadeias=not a.sem_cadeias,
                                         com_relatorio=not a.sem_relatorio, com_banco=not a.sem_banco,
                                         prog=lambda k, n: barra.update(t, completed=k, total=n))
    con.print(f"[green]✔[/green] {m['n_arquivos']} arquivos ({m['bytes'] / 1e6:.1f} MB) em [bold]{destino}[/bold] "
              f"({destino.stat().st_size / 1e6:.1f} MB). Manifesto com sha256 em {Path(a.saida) / 'manifesto.json'}.")
    con.print("Para reabrir: [cyan]from a0bench import carregar; r = carregar('" + str(destino) + "')[/cyan]")


def cmd_conferir(a):
    from rich.console import Console

    from . import resultados

    con = Console()
    prob = resultados.conferir(a.saida)
    if prob:
        for p in prob[:30]:
            con.print(f"[red]✖ {p}[/red]")
        con.print(f"[red]{len(prob)} diferenças em relação ao manifesto.[/red]")
        raise SystemExit(1)
    con.print("[green]✔ Todos os arquivos conferem com o manifesto.[/green]")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m a0bench", description="A0: benchmark de inferência bayesiana.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def comuns(s):
        s.add_argument("--preset", default="completo", choices=list(config.PRESETS))
        s.add_argument("--grupos", help=f"lista separada por vírgulas: {', '.join(config.GRUPOS)}")
        s.add_argument("--modelos", help="nomes separados por vírgulas (veja 'listar')")
        s.add_argument("--dados", choices=config.DADOS,
                       help="só os modelos com dados reais, só os sintéticos, ou todos (padrão do preset)")
        s.add_argument("--metodos", help=f"lista: {', '.join(config.METODOS)}")
        s.add_argument("--sementes", help="ex.: 0,1,2")
        s.add_argument("--sem-pfn", action="store_true", help="não roda o NPE-PFN (TabPFN)")
        s.add_argument("--processos", type=int, help="processos em paralelo (padrão: todos os núcleos)")
        s.add_argument("--salvar-a-cada", type=int, metavar="N",
                       help="grava a cadeia e o estado do amostrador a cada N passos (padrão: 1000)")
        s.add_argument("--nivel", metavar="ALFA",
                       help="nível dos intervalos de confiança: fração (0.95, o padrão) ou 1sigma, 2sigma")
        s.add_argument("--desbaste", type=int, metavar="K",
                       help="guarda 1 a cada K passos das cadeias completas (padrão: 1, tudo)")

    s = sub.add_parser("verificar", help="confere a instalação nesta máquina (poucos minutos)")
    s.add_argument("--pfn", action="store_true", help="testa também o NPE-PFN (baixa os pesos do TabPFN)")
    s.add_argument("--sem-petab", action="store_true", help="não compila um modelo PEtab pelo AMICI")
    s.add_argument("--processos", type=int, help="processos na execução de teste (padrão: 2)")
    s.add_argument("--pasta-modelos", help="cache dos modelos AMICI compilados")
    s.add_argument("--manter", metavar="PASTA", help="guarda os arquivos da verificação nesta pasta")
    s.set_defaults(f=cmd_verificar)
    sub.add_parser("listar", help="lista os modelos").set_defaults(f=cmd_listar)
    s = sub.add_parser("estimar", help="estima o tempo de execução nesta máquina")
    comuns(s)
    s.set_defaults(f=cmd_estimar)
    s = sub.add_parser("rodar", help="roda (ou retoma) o benchmark")
    comuns(s)
    s.add_argument("--sem-identificabilidade", action="store_true",
                   help="pula a análise de identificabilidade (ela pode ser feita depois com 'identificar')")
    s.add_argument("--saida", default="resultados_a0")
    s.add_argument("--pasta-modelos", help="cache dos modelos AMICI compilados")
    s.add_argument("--sem-monitor", action="store_true", help="só mensagens de log, sem painel")
    s.set_defaults(f=cmd_rodar)
    s = sub.add_parser("identificar", help="roda só o MAP, a referência e a análise de identificabilidade")
    comuns(s)
    s.add_argument("--saida", default="resultados_a0")
    s.add_argument("--pasta-modelos", help="cache dos modelos AMICI compilados")
    s.add_argument("--sem-monitor", action="store_true", help="só mensagens de log, sem painel")
    s.set_defaults(f=cmd_identificar)
    s = sub.add_parser("monitor", help="acompanha uma execução em andamento")
    s.add_argument("--saida", default="resultados_a0")
    s.set_defaults(f=cmd_monitor)
    s = sub.add_parser("relatorio", help="gera tabelas e figuras a partir dos arquivos gravados")
    s.add_argument("--saida", default="resultados_a0")
    s.add_argument("--sem-latex", action="store_true", help="fonte STIX em vez do LaTeX (bem mais rápido)")
    s.add_argument("--so", choices=["geral", "modelos"], help="só as figuras gerais ou só as de cada modelo")
    s.add_argument("--modelos", help="figuras só destes modelos (nomes separados por vírgulas)")
    s.add_argument("--pdf", action="store_true", help="grava também em PDF as figuras de cada modelo")
    s.add_argument("--processos", type=int, default=1, help="modelos desenhados em paralelo")
    s.set_defaults(f=cmd_relatorio)
    s = sub.add_parser("exportar", help="junta resultados, cadeias, registros e código num .zip com manifesto")
    s.add_argument("--saida", default="resultados_a0")
    s.add_argument("--destino", help="arquivo .zip de destino (padrão: <saida>.zip)")
    s.add_argument("--sem-cadeias", action="store_true", help="deixa de fora as cadeias completas")
    s.add_argument("--sem-relatorio", action="store_true", help="deixa de fora as figuras e tabelas")
    s.add_argument("--sem-banco", action="store_true", help="deixa de fora os bancos de simulações do NPE")
    s.set_defaults(f=cmd_exportar)
    s = sub.add_parser("conferir", help="confere os arquivos da pasta com o manifesto (sha256)")
    s.add_argument("--saida", default="resultados_a0")
    s.set_defaults(f=cmd_conferir)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
