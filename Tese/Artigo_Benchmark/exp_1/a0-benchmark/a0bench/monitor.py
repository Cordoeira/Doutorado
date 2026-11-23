"""Monitor de terminal (rich): painel ao vivo com o andamento de cada modelo."""
import time
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from .util import ler_json, pid_vivo, registrar

NOMES = {"preparo": "Preparo", "map": "MAP", "referencia": "Referência", "identificabilidade": "Identificab.",
         "solucoes": "Soluções",
         "laplace": "Laplace", "metropolis": "Metropolis", "ensemble": "Ensemble", "mala": "MALA", "hmc": "HMC",
         "npe": "NPE", "npe_pfn": "NPE-PFN"}
PESO_ETAPA = {"map": 1, "referencia": 8, "identificabilidade": 2, "solucoes": 0.3, "laplace": 0.2, "metropolis": 3, "ensemble": 3,
              "mala": 3, "hmc": 3, "npe": 4, "npe_pfn": 2}
PESO_GRUPO = {"petab_grande": 14, "petab": 6, "dificil": 1.6, "facil": 1}
GIRO = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
COR_GRUPO = {"facil": "green", "dificil": "magenta", "petab": "blue", "petab_grande": "bright_blue"}


def _tempo(s):
    s = int(max(0, s))
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}" if h else f"{m}min{s % 60:02d}s" if m else f"{s}s"


def _cor_c2st(v):
    return "green" if v <= 0.6 else "yellow" if v <= 0.8 else "red"


def ler(saida):
    """Estado de todos os modelos da execucao."""
    saida = Path(saida)
    ex = ler_json(saida / "execucao.json", {}) or {}
    linhas = []
    for m in ex.get("modelos", []):
        e = ler_json(saida / m["nome"] / "estado.json") or {}
        etapas = e.get("etapas") or ex.get("etapas", [])
        sit = e.get("situacao", "fila")
        if sit == "rodando" and not pid_vivo(e.get("pid")):
            sit = "interrompido"
        feitas = [x for x in e.get("feitas", []) if x in etapas]
        total = sum(PESO_ETAPA.get(x, 1) for x in etapas) or 1
        feito = sum(PESO_ETAPA.get(x, 1) for x in feitas)
        if sit == "rodando" and e.get("etapa") in etapas and e.get("etapa") not in feitas:
            feito += PESO_ETAPA.get(e["etapa"], 1) * e.get("fracao", 0.0)
        frac = 1.0 if sit in ("concluido", "erro") else min(1.0, feito / total)
        linhas.append({**m, **e, "situacao": sit, "etapas": etapas, "feitas": feitas, "frac": frac,
                       "peso": PESO_GRUPO.get(m["grupo"], 1)})
    return ex, linhas


def _etapas(l):
    t = Text()
    for x in l["etapas"]:
        if x in l["feitas"] or l["situacao"] == "concluido":
            t.append("●", style="green")
        elif l["situacao"] == "rodando" and x == l.get("etapa"):
            t.append("◉", style="bold cyan")
        else:
            t.append("○", style="grey37")
    return t


def _linha(l, agora):
    sit = l["situacao"]
    icone = {"concluido": Text("✔", style="bold green"), "erro": Text("✖", style="bold red"),
             "fila": Text("·", style="grey50"), "interrompido": Text("‖", style="yellow"),
             "rodando": Text(GIRO[int(agora * 8) % len(GIRO)], style="bold cyan")}[sit]
    if sit == "concluido" and l.get("avisos"):
        icone = Text("⚠", style="bold yellow")
    nome = Text(l.get("rotulo") or l["nome"].replace("_", " "), style="bold" if sit == "rodando" else "")
    if sit == "fila":
        nome.stylize("grey50")
    classe = Text(l["grupo"].replace("_", " "), style=COR_GRUPO.get(l["grupo"], "white"))
    if sit == "rodando":
        et = Text(NOMES.get(l.get("etapa"), l.get("etapa", "")), style="cyan")
        if l.get("msg"):
            et.append(f"  {l['msg']}", style="grey62")
    elif sit == "erro":
        et = Text((l.get("erros") or ["erro"])[-1][:46], style="red")
    elif l.get("avisos"):
        et = Text(l["avisos"][-1][:46], style="yellow")
    elif sit == "interrompido":
        et = Text("interrompido; rode de novo para continuar", style="yellow")
    else:
        et = Text(l.get("desafio", "") if sit == "fila" else "", style="grey50")
    barra = Table.grid(padding=(0, 1))
    cor = {"concluido": "green", "erro": "red", "rodando": "cyan"}.get(sit, "grey37")
    barra.add_row(ProgressBar(total=1.0, completed=l["frac"], width=14, complete_style=cor, finished_style=cor),
                  Text(f"{100 * l['frac']:3.0f}%", style=cor))
    rh = l.get("ref_rhat")
    ref = Text("")
    if rh is not None:
        ref = Text(f"{rh:.2f}", style="green" if l.get("ref_ok", True) else "bold yellow")
        ref.append(f" {l.get('ref_ess', 0):>5.0f}", style="grey62")
    idn = Text("")
    if l.get("ident"):
        i, pp, n = l["ident"]
        idn = Text.assemble((f"{i}", "green"), ("/", "grey50"), (f"{pp}", "yellow"), ("/", "grey50"), (f"{n}", "red"))
    c2 = Text("")
    for k, v in (l.get("c2st") or {}).items():
        c2.append("■", style=_cor_c2st(v))
    seg = l.get("segundos", 0.0)
    if sit == "rodando" and l.get("atualizado"):
        seg += agora - l["atualizado"]
    return [icone, nome, classe, str(l.get("d", "")), _etapas(l), et, barra, ref, idn, c2,
            Text(_tempo(seg) if seg else "", style="grey62")]


class Painel:
    def __init__(self, saida, ao_vivo=True):
        self.saida = Path(saida)
        self.console = Console()
        if not self.console.is_terminal:
            self.console = Console(width=170)
        self.ao_vivo = ao_vivo and self.console.is_terminal
        self.t0 = time.time()
        self.f0 = None
        self.msgs = []
        self._visto = {}

    def log(self, texto):
        self.msgs.append(f"[grey50]{time.strftime('%H:%M:%S')}[/grey50] {texto}")
        if not self.ao_vivo:
            self.console.print(self.msgs[-1])
        try:
            registrar(self.saida, Text.from_markup(texto).plain)
        except OSError:
            pass

    def _novidades(self, linhas):
        for l in linhas:
            chave = (l["situacao"], l.get("etapa"), len(l.get("feitas", [])))
            ant = self._visto.get(l["nome"])
            if ant == chave:
                continue
            self._visto[l["nome"]] = chave
            rot = l.get("rotulo") or l["nome"]
            if l["situacao"] == "concluido" and ant is None:
                continue    # ja estava pronto quando a sessao comecou
            if l["situacao"] == "concluido" and ant[0] != "concluido":
                c2, idn = l.get("c2st") or {}, l.get("ident")
                extra = f" (melhor C2ST {min(c2.values()):.2f})" if c2 else ""
                if idn and not c2:
                    extra = f" (identificáveis/parciais/não: {idn[0]}/{idn[1]}/{idn[2]})"
                self.log(f"[green]✔[/green] [bold]{rot}[/bold] concluído em {_tempo(l.get('segundos', 0))}{extra}")
            elif l["situacao"] == "erro" and (ant is None or ant[0] != "erro"):
                self.log(f"[red]✖ {rot}: {(l.get('erros') or ['erro'])[-1][:90]}[/red]")
            elif l["situacao"] == "rodando" and not self.ao_vivo and l.get("etapa"):
                self.log(f"[cyan]{rot}[/cyan]: {NOMES.get(l['etapa'], l['etapa'])}")

    def montar(self):
        agora = time.time()
        ex, linhas = ler(self.saida)
        self._novidades(linhas)
        cont = {s: sum(l["situacao"] == s for l in linhas) for s in ("concluido", "rodando", "fila", "erro",
                                                                     "interrompido")}
        avisos = sum(bool(l.get("avisos")) for l in linhas)
        peso = sum(l["peso"] for l in linhas) or 1
        F = sum(l["peso"] * l["frac"] for l in linhas) / peso
        if self.f0 is None:
            self.f0 = F
        dec = agora - self.t0
        eta = dec * (1 - F) / (F - self.f0) if F - self.f0 > 0.01 else None

        cab = Table.grid(expand=True)
        cab.add_column(ratio=1)
        cab.add_column(justify="right")
        cab.add_row(
            Text.assemble(("A0", "bold bright_cyan"), ("  benchmark de inferência bayesiana em EDOs e EDPs", "bold")),
            Text.assemble(("preset ", "grey62"), (str(ex.get("preset", "")), "bold"), ("   processos ", "grey62"),
                          (str(ex.get("processos", "")), "bold")))
        est = Text.assemble(
            ("✔ ", "green"), (f"{cont['concluido']} concluídos   ", ""), ("● ", "cyan"), (f"{cont['rodando']} rodando   ", ""),
            ("· ", "grey50"), (f"{cont['fila']} na fila   ", ""), ("⚠ ", "yellow"), (f"{avisos} avisos   ", ""),
            ("✖ ", "red"), (f"{cont['erro']} erros", ""))
        if cont["interrompido"]:
            est.append(f"   ‖ {cont['interrompido']} interrompidos", style="yellow")
        tempo = Text.assemble(("decorrido ", "grey62"), (_tempo(dec), "bold"), ("   restante ", "grey62"),
                              (("~" + _tempo(eta)) if eta is not None else "calculando", "bold"))
        cab.add_row(est, tempo)
        geral = Table.grid(padding=(0, 1))
        geral.add_row(ProgressBar(total=1.0, completed=F, width=max(20, self.console.size.width - 12),
                                  complete_style="bright_cyan", finished_style="green"),
                      Text(f"{100 * F:3.0f}%", style="bold bright_cyan"))

        tab = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False, header_style="bold grey70")
        for nome, kw in [("", dict(width=1)), ("Modelo", dict(no_wrap=True, ratio=3)), ("Classe", dict(no_wrap=True)),
                         ("d", dict(justify="right")), ("Etapas", dict(no_wrap=True)),
                         ("Agora", dict(no_wrap=True, ratio=4)), ("Progresso", dict(no_wrap=True)),
                         ("R̂  ESS ref.", dict(no_wrap=True)), ("I/P/N", dict(no_wrap=True)),
                         ("C2ST", dict(no_wrap=True)), ("Tempo", dict(justify="right", no_wrap=True))]:
            tab.add_column(nome, **kw)
        ordem = {"rodando": 0, "erro": 1, "interrompido": 2, "concluido": 3, "fila": 4}
        linhas.sort(key=lambda l: (ordem[l["situacao"]], -l.get("atualizado", 0) if l["situacao"] == "concluido" else 0))
        cabem = max(6, self.console.size.height - 13 - min(4, len(self.msgs)))
        for l in linhas[:cabem]:
            tab.add_row(*_linha(l, agora))
        if len(linhas) > cabem:
            resto = linhas[cabem:]
            tab.add_row("", Text(f"… e mais {len(resto)} ({sum(x['situacao'] == 'fila' for x in resto)} na fila)",
                                 style="grey50"), *[""] * 9)
        leg = Text.assemble(("Etapas: ", "grey62"), ("● feita ", "green"), ("◉ atual ", "cyan"), ("○ pendente", "grey37"),
                            ("    I/P/N: ", "grey62"), ("identificáveis", "green"), ("/", "grey50"), ("parciais", "yellow"),
                            ("/", "grey50"), ("não identificáveis", "red"), ("    C2ST por método: ", "grey62"),
                            ("■ ≤0,6 ", "green"), ("■ ≤0,8 ", "yellow"), ("■ >0,8", "red"))
        corpo = [cab, geral, tab, leg]
        if self.msgs:
            corpo.append(Text.from_markup("\n".join(self.msgs[-4:])))
        return Panel(Group(*corpo), border_style="bright_cyan", box=box.ROUNDED, padding=(0, 1),
                     subtitle=Text(f"{self.saida}  ·  Ctrl+C interrompe; o mesmo comando retoma", style="grey50"))

    def acompanhar(self, terminou):
        if not self.ao_vivo:
            while not terminou():
                self.montar()
                time.sleep(2.0)
            self.montar()
            return
        with Live(self.montar(), console=self.console, refresh_per_second=4, screen=False) as live:
            while not terminou():
                time.sleep(0.4)
                live.update(self.montar())
            live.update(self.montar())

    def encerrar(self, msg=None):
        if msg:
            self.console.print(f"[bold yellow]{msg}[/bold yellow]")
        resumo(self.saida, self.console)


def assistir(saida):
    """Acompanha, de outro terminal, uma execucao em andamento."""
    p = Painel(saida)

    def terminou():
        ex, linhas = ler(saida)
        return not pid_vivo(ex.get("pid")) and not any(l["situacao"] == "rodando" for l in linhas)
    try:
        p.acompanhar(terminou)
    except KeyboardInterrupt:
        pass


def resumo(saida, console=None):
    """Tabela final: C2ST mediano por modelo e metodo."""
    console = console or Console()
    ex, linhas = ler(saida)
    if not linhas:
        console.print("[yellow]Nenhuma execução encontrada nesta pasta.[/yellow]")
        return
    metodos = [m for m in ex.get("etapas", []) if m in PESO_ETAPA and m not in ("map", "referencia", "identificabilidade", "solucoes")]
    titulo = "C2ST contra a referência (0,5 é o ideal)" if metodos else "Identificabilidade (I/P/N por modelo)"
    tab = Table(title=titulo, box=box.SIMPLE_HEAD, header_style="bold")
    tab.add_column("Modelo", no_wrap=True)
    tab.add_column("Classe", no_wrap=True)
    tab.add_column("d", justify="right")
    tab.add_column("R̂ ref.", justify="right")
    tab.add_column("I/P/N", justify="center")
    for m in metodos:
        tab.add_column(NOMES[m], justify="right")
    for l in linhas:
        if l["situacao"] == "fila":
            continue
        c2 = l.get("c2st") or {}
        rh = l.get("ref_rhat")
        ref = Text("" if rh is None else f"{rh:.2f}", style="green" if l.get("ref_ok", True) else "bold yellow")
        idn = "/".join(str(x) for x in l["ident"]) if l.get("ident") else ""
        cel = [Text(f"{c2[m]:.2f}", style=_cor_c2st(c2[m])) if m in c2 else Text("–", style="grey50") for m in metodos]
        tab.add_row(l.get("rotulo") or l["nome"], Text(l["grupo"].replace("_", " "), style=COR_GRUPO.get(l["grupo"], "")),
                    str(l.get("d", "")), ref, idn, *cel)
    console.print(tab)
    erros = [(l.get("rotulo") or l["nome"], e) for l in linhas for e in l.get("erros", [])]
    for rot, e in erros[:12]:
        console.print(f"[red]✖ {rot}: {e}[/red]")
    for l in linhas:
        for a in l.get("avisos", []):
            console.print(f"[yellow]⚠ {l.get('rotulo') or l['nome']}: {a}[/yellow]")
    console.print(f"[grey62]Resultados em {saida}. Figuras e tabelas: python -m a0bench relatorio --saida {saida}[/grey62]")
