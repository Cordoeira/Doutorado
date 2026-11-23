"""Executor: processa cada modelo num processo proprio, com retomada, trava e estado para o monitor."""
import os
import time
import traceback
from pathlib import Path

import numpy as np

from . import config
from .util import Estado, Trava, ambiente, copiar_codigo, gravar, gravar_json, ler_json, registrar


def _feito(arq):
    r = ler_json(arq)
    return bool(r and r.get("ok"))


def _pendentes(pasta, cfg):
    return {m: [s for s in cfg["sementes"] if not _feito(pasta / f"{m}_s{s}.json")] for m in cfg["metodos"]}


def completo(pasta, cfg):
    pasta = Path(pasta)
    if not (pasta / "referencia.npz").exists() or not (pasta / "problema.json").exists():
        return False
    if cfg["identificabilidade"] and (ler_json(pasta / "identificabilidade.json") or {}).get("versao", 1) < config.VERSAO_IDENT:
        return False
    if cfg.get("solucoes", True) and not (pasta / "solucao.npz").exists():
        return False
    return not any(_pendentes(pasta, cfg).values())


def processar(origem, nome, grupo, cfg, saida, pasta_modelos):
    """Roda (ou retoma) todas as etapas de um modelo. Chamado dentro de um processo trabalhador."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import logging
    import signal
    import warnings

    try:    # Ctrl+C e tratado so pelo processo principal, que encerra os trabalhadores
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
    warnings.filterwarnings("ignore")
    for nome_log in ("pypesto", "amici", "petab", "sbi"):
        logging.getLogger(nome_log).setLevel(logging.CRITICAL)
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    from . import cadeias, identificabilidade, inferencia, problemas

    pai = os.getppid()
    pasta = Path(saida) / nome
    pasta.mkdir(parents=True, exist_ok=True)
    trava = Trava(pasta)
    if not trava.adquirir():
        return nome, "ocupado"
    est = Estado(pasta, config.etapas(cfg), modelo=nome, grupo=grupo)
    try:
        est.iniciar("preparo", "compilando o modelo")
        p = problemas.construir(origem, nome, grupo, cfg, pasta_modelos)
        est.info(rotulo=p.rotulo, tipo=p.tipo, d=p.d, desafio=p.desafio, t_f_ms=1e3 * p.t_f)

        if getattr(p, "grad_ok", 1.0) < 0.8:
            est.aviso(f"o gradiente falha em {100 * (1 - p.grad_ok):.0f}% dos pontos de teste do prior, mesmo com a "
                      f"tolerância afrouxada: MALA, HMC e a otimização ficam prejudicados neste modelo")
        est.iniciar("map", "otimização multistart")
        mapa = inferencia.tarefa_map(p, pasta, cfg)
        if str(mapa.get("origem", "")).startswith("nominal"):
            est.aviso("a otimização com vários inícios não chegou ao ponto nominal da coleção; o MAP parte dele")
        if "Fisher" in str(mapa.get("origem", "")):
            est.aviso("as sensibilidades falham no ótimo: a covariância inicial dos amostradores vem "
                      + ("de um ponto vizinho" if "vizinho" in mapa["origem"] else "de 1/10 da largura do prior"))
        est.concluir("map")

        est.iniciar("referencia", "temperação paralela")
        ref, retomado = inferencia.tarefa_referencia(p, pasta, mapa, cfg, est.progresso)
        if retomado:
            est.retomada("referencia", retomado)
        ref_ok = bool(ref["ok"]) if "ok" in ref else True
        info = dict(problema=nome, rotulo=p.rotulo, tipo=p.tipo, grupo=grupo, desafio=p.desafio, d=p.d,
                    dados="real" if getattr(p, "dados_reais", False) else "sintetico", fonte=getattr(p, "fonte", ""),
                    n_obs=len(p.x_obs), t_f_ms=1e3 * p.t_f, custo_grad=float(p.custo_grad),
                    grad_ok=float(getattr(p, "grad_ok", 1.0)), map_origem=str(mapa.get("origem", "otimizacao")),
                    ref_rhat=float(ref["rhat_max"]), ref_ess=float(ref["ess_min"]),
                    ref_passos=int(ref["n_passos"]), ref_ok=ref_ok, nomes=list(p.nomes), lo=p.lo.tolist(),
                    hi=p.hi.tolist(), verd=None if getattr(p, "verd", None) is None else p.verd.tolist(),
                    rotulos_obs=list(p.rotulos_obs))
        gravar_json(pasta / "problema.json", info)
        est.info(ref_rhat=info["ref_rhat"], ref_ess=info["ref_ess"], ref_ok=ref_ok)
        if not ref_ok:
            est.aviso(f"referência não convergiu (R-hat {info['ref_rhat']:.2f}, ESS {info['ref_ess']:.0f})")
        est.concluir("referencia")

        if cfg["identificabilidade"]:
            est.iniciar("identificabilidade", "perfis de verossimilhança")
            try:
                ja_havia = (pasta / "identificabilidade.json").exists()    # resultado pronto de outra sessao
                ident = identificabilidade.tarefa_identificabilidade(p, pasta, mapa, ref, cfg, est.progresso)
                if "retomada" in ident and not ja_havia:
                    est.retomada("identificabilidade", ident["retomada"]["perfis_prontos"])
                for e in ident.get("erros", []):
                    est.aviso(f"identificabilidade: {e}")
                viz = ident.get("vizinhanca") or {}
                if viz.get("falhas", 0) > 0.05 * viz.get("n", 1):
                    est.aviso(f"a simulação falha em {viz['falhas']} de {viz['n']} perturbações de 0,2% do prior "
                              f"em torno do ótimo: há uma região inviável colada nele")
                if "otimo_refeito" in ident:
                    o = ident["otimo_refeito"]
                    est.aviso(f"os perfis acharam um ponto melhor que o ótimo (-log L {sum(o['melhoras']):.1f} menor) "
                              f"e foram refeitos a partir dele" + ("" if o["estavel"] else
                                                                   "; depois de 3 voltas ainda havia melhora: "
                                                                   "leia os perfis com cautela"))
                if "map_melhorado" in ident:
                    est.aviso(f"o MAP não era o ótimo global: a referência achou -log L "
                              f"{ident['map_melhorado']['melhora']:.1f} menor")
                cls = [q["classe"] for q in ident["parametros"]]
                est.info(ident=[cls.count("identificavel"), cls.count("parcial"), cls.count("nao identificavel")])
            except Exception as e:  # noqa: BLE001
                est.erro("identificabilidade", e)
            est.concluir("identificabilidade")

        if cfg.get("solucoes", True):
            est.iniciar("solucoes", "soluções e superfícies")
            try:
                inferencia.tarefa_solucao(p, pasta, mapa, ref, cfg)
                est.progresso(0.5, "superfícies de contorno")
                inferencia.tarefa_superficie(p, pasta, mapa, ref, cfg)
            except Exception as e:  # noqa: BLE001
                est.erro("solucoes", e)
            est.concluir("solucoes")

        pend = _pendentes(pasta, cfg)
        c2 = {}

        def registrar(metodo, s, am, esd, t0, **extra):
            reg = dict(problema=nome, metodo=metodo, semente=s, ok=True, esd=float(esd), **extra)
            reg.update(inferencia.metricas(p, am, ref["amostras"], cfg["n_c2st"]))
            reg["segundos"] = time.time() - t0
            gravar(pasta / f"{metodo}_s{s}.npy", lambda fh: np.save(fh, am))
            gravar_json(pasta / f"{metodo}_s{s}.json", reg)

        def falhou(metodo, s, e, t0):
            gravar_json(pasta / f"{metodo}_s{s}.json", dict(
                problema=nome, metodo=metodo, semente=s, ok=False, segundos=time.time() - t0,
                erro=f"{type(e).__name__}: {str(e)[:300]}"))

        def resumo(metodo):
            v = [r["c2st"] for s in cfg["sementes"] if (r := ler_json(pasta / f"{metodo}_s{s}.json")) and r.get("ok")]
            if v:
                c2[metodo] = float(np.median(v))
                est.info(c2st=dict(c2))
            est.concluir(metodo)

        if "laplace" in cfg["metodos"]:
            est.iniciar("laplace")
            for s in pend["laplace"]:
                t0 = time.time()
                am = cadeias.amostrar_laplace(p, mapa, cfg["n_amostras"], np.random.default_rng(s))
                registrar("laplace", s, am, mapa["esd"], t0)
            resumo("laplace")

        for metodo in config.AMOSTRADORES:
            if metodo not in cfg["metodos"]:
                continue
            n_cad = len(cfg["sementes"])
            est.iniciar(metodo, f"{n_cad} cadeia{'s' if n_cad > 1 else ''}")
            if pend[metodo]:
                t0 = time.time()
                try:
                    saidas, retomado = cadeias.rodar_amostrador(metodo, p, mapa, cfg, pasta, est.progresso)
                    if retomado:
                        est.retomada(metodo, retomado)
                    for s, (am, diag) in zip(cfg["sementes"], saidas):
                        esd = diag.pop("esd")
                        registrar(metodo, s, am, esd, t0, **diag)
                except Exception as e:  # noqa: BLE001
                    est.erro(metodo, e)
                    for s in cfg["sementes"]:
                        falhou(metodo, s, e, t0)
            resumo(metodo)

        banco = None
        marca_pfn = Path(saida) / "_tabpfn_indisponivel.txt"
        for metodo in ("npe", "npe_pfn"):
            if metodo not in cfg["metodos"]:
                continue
            est.iniciar(metodo)
            for k, s in enumerate(pend[metodo]):
                t0 = time.time()
                try:
                    if metodo == "npe_pfn" and marca_pfn.exists():
                        raise RuntimeError("TabPFN indisponível nesta execução: " + marca_pfn.read_text()[:150])
                    if banco is None:
                        banco = inferencia.tarefa_banco(p, pasta, cfg, est.progresso)
                    est.progresso(k / len(pend[metodo]), f"semente {s}")
                    if metodo == "npe":
                        am, esd = inferencia.amostrar_npe(p, banco, cfg["n_amostras"], s, cfg["lote_treino"])
                    else:
                        am, esd = inferencia.amostrar_npe_pfn(p, banco, cfg["n_amostras"], s, cfg["n_ctx_pfn"],
                                                              cfg.get("dispositivo_pfn", "auto"))
                    registrar(metodo, s, am, esd, t0, falhas_simulacao=int(banco["falhas"]))
                except Exception as e:  # noqa: BLE001
                    est.erro(f"{metodo} s{s}", e)
                    falhou(metodo, s, e, t0)
                    if metodo == "npe_pfn":
                        if isinstance(e, ImportError) or any(t in str(e).lower() for t in (
                                "download", "gated", "license", "connect", "token", "not installed", "huggingface")):
                            marca_pfn.write_text(f"{type(e).__name__}: {e}")
                            est.aviso("NPE-PFN indisponível (pacote tabpfn ou pesos do TabPFN): a mensagem completa "
                                      f"está em {marca_pfn}. Para rodar sem ele, use --sem-pfn")
                        for s2 in pend[metodo][k + 1:]:
                            falhou(metodo, s2, e, t0)
                        break
            resumo(metodo)
        est.fim()
        return nome, est.d["situacao"]
    except Exception as e:  # noqa: BLE001
        est.erro(est.d.get("etapa", "?"), e)
        gravar(pasta / "erro.txt", lambda fh: fh.write(traceback.format_exc().encode()))
        est.fim()
        return nome, "erro"
    finally:
        trava.liberar()
        if os.getppid() != pai:   # orfao: o processo principal morreu
            os._exit(1)


def _descrever(origem, nome, grupo):
    from . import modelos

    if origem == "petab":
        return dict(nome=nome, grupo=grupo, origem=origem, rotulo=nome.split("_")[0] + " (PEtab)",
                    desafio="dados reais")
    m = modelos.POR_NOME[nome]
    return dict(nome=nome, grupo=grupo, origem=origem, rotulo=m["rotulo"], d=len(m["lo"]), desafio=m.get("desafio", ""))


def rodar(cfg, saida, processos=None, pasta_modelos=None, monitor=True):
    """Dispara os modelos em paralelo e acompanha a execucao."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from . import monitor as mon

    import sys

    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    pasta_modelos = str(pasta_modelos or saida / "_amici_models")
    processos = max(1, processos or (os.cpu_count() or 2))
    itens = config.selecionar(cfg)
    antiga = ler_json(saida / "config.json") or {}
    mudou = sorted(k for k in cfg if k in antiga and antiga[k] != cfg[k]
                   and k not in ("modelos", "grupos", "metodos", "preset", "identificabilidade", "solucoes"))
    gravar_json(saida / "config.json", cfg)
    fila = [i for i in itens if not completo(saida / i[1], cfg)]
    for _, n, _ in fila:    # o estado gravado por uma sessao anterior nao vale para quem volta para a fila
        ant = ler_json(saida / n / "estado.json")
        if ant and ant.get("situacao") in ("concluido", "erro"):
            ant.update(situacao="fila", etapa="", fracao=0.0, feitas=[], msg="")
            gravar_json(saida / n / "estado.json", ant)
    # o arquivo do processo: ambiente, copia do codigo e uma entrada por sessao em execucao.json
    amb = ambiente()
    gravar_json(saida / "ambiente.json", amb)
    copiar_codigo(saida / "codigo")
    ex_ant = ler_json(saida / "execucao.json") or {}
    sessoes = list(ex_ant.get("sessoes", []))
    for s in sessoes:       # uma sessao anterior sem registro de fim foi encerrada a forca (kill -9, queda de energia)
        s.setdefault("situacao", "encerrada sem aviso")
    sessoes.append(dict(inicio=time.time(), pid=os.getpid(), argv=sys.argv[1:], processos=processos,
                        preset=cfg.get("preset", ""), codigo=amb["codigo"]["total"], maquina=amb["maquina"],
                        na_fila=[i[1] for i in fila]))
    conhecidos = {m["nome"]: m for m in ex_ant.get("modelos", [])}
    conhecidos.update({n: _descrever(o, n, g) for o, n, g in itens})
    execucao = dict(inicio=time.time(), processos=processos, preset=cfg.get("preset", ""), pid=os.getpid(),
                    modelos=[_descrever(o, n, g) for o, n, g in itens],
                    todos_os_modelos=list(conhecidos.values()), etapas=config.etapas(cfg), sessoes=sessoes)
    gravar_json(saida / "execucao.json", execucao)
    registrar(saida, f"sessão iniciada: {len(fila)} de {len(itens)} modelos na fila, {processos} processos, "
                     f"preset {cfg.get('preset', '')}, código {amb['codigo']['total']}")
    if mudou:
        registrar(saida, f"atenção: configuração diferente da sessão anterior em {', '.join(mudou)}")
    (saida / "_tabpfn_indisponivel.txt").unlink(missing_ok=True)
    painel = mon.Painel(saida, ao_vivo=monitor)
    if "npe_pfn" in cfg["metodos"] and fila:
        from importlib import util as importacao

        if importacao.find_spec("tabpfn") is None:
            painel.log("[yellow]⚠ o pacote tabpfn não está instalado: o NPE-PFN vai falhar em todos os modelos. "
                       "Instale com pip install 'sbi[tabpfn]' ou rode com --sem-pfn.[/yellow]")
    if mudou:
        painel.log(f"[yellow]⚠ configuração diferente da sessão anterior em: {', '.join(mudou)}. "
                   f"Etapas já concluídas não são refeitas.[/yellow]")

    def fechar_sessao(situacao):
        sessoes[-1].update(fim=time.time(), situacao=situacao)
        gravar_json(saida / "execucao.json", execucao)
        registrar(saida, f"sessão encerrada: {situacao}")
    if not fila:
        painel.encerrar("Tudo já estava calculado.")
        fechar_sessao("nada a fazer")
        return
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    # ao interromper, o multiprocessing avisa de um semaforo que ele mesmo limpa; o aviso so confunde
    os.environ["PYTHONWARNINGS"] = ",".join(filter(None, [os.environ.get("PYTHONWARNINGS"),
                                                          "ignore:resource_tracker:UserWarning"]))
    def _encerrar(*_):
        raise KeyboardInterrupt

    try:
        import signal

        signal.signal(signal.SIGTERM, _encerrar)   # kill no processo principal tambem encerra os trabalhadores
    except (ValueError, OSError):
        pass
    ex = ProcessPoolExecutor(max_workers=min(processos, len(fila)), mp_context=mp.get_context("spawn"))
    futuros = [ex.submit(processar, o, n, g, cfg, str(saida), pasta_modelos) for o, n, g in fila]
    try:
        painel.acompanhar(lambda: all(f.done() for f in futuros))
        for f, (_, n, _) in zip(futuros, fila):
            if f.exception() is not None:
                painel.log(f"[red]{n}: o processo terminou com erro: {f.exception()!r}[/red]")
            elif f.result()[1] == "ocupado":
                painel.log(f"[yellow]⚠ {n}: não rodou, porque outro processo está trabalhando nesta pasta. Se não "
                           f"houver outra execução, apague {saida / n / 'em_execucao.lock'} e rode de novo.[/yellow]")
        painel.encerrar()
        fechar_sessao("concluída")
    except KeyboardInterrupt:
        for f in futuros:
            f.cancel()
        for proc in list(getattr(ex, "_processes", {}).values()):
            proc.terminate()
        painel.encerrar("Interrompido. Rode o mesmo comando para continuar de onde parou.")
        fechar_sessao("interrompida")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
