"""Utilidades: escrita atomica, estado e registro de eventos de cada modelo, trava e descricao do ambiente."""
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np


def gravar(caminho, escreve):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(f"{caminho.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        escreve(fh)
    os.replace(tmp, caminho)


def gravar_json(caminho, obj):
    gravar(caminho, lambda fh: fh.write(json.dumps(obj, indent=1).encode()))


def ler_json(caminho, padrao=None):
    try:
        return json.loads(Path(caminho).read_text())
    except (OSError, ValueError):
        return padrao


def gravar_pickle(caminho, obj):
    gravar(caminho, lambda fh: pickle.dump(obj, fh))


def ler_pickle(caminho):
    with open(caminho, "rb") as fh:
        return pickle.load(fh)


def cov_de_hessiana(h, lo, hi):
    val, vec = np.linalg.eigh(0.5 * (h + h.T))
    piso = 1.0 / np.max(hi - lo) ** 2
    return (vec / np.maximum(val, piso)) @ vec.T


def pid_vivo(pid):
    """O processo existe? Nunca envia sinal que encerre o processo (no Windows, os.kill(pid, 0) encerraria)."""
    try:
        pid = int(pid)
    except (ValueError, TypeError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        nucleo = ctypes.windll.kernel32
        h = nucleo.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        codigo = ctypes.c_ulong()
        ok = nucleo.GetExitCodeProcess(h, ctypes.byref(codigo))
        nucleo.CloseHandle(h)
        return bool(ok) and codigo.value == 259             # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except PermissionError:                                 # existe, mas e de outro usuario
        return True
    except OSError:
        return False
    return True


class Trava:
    """Impede que dois processos trabalhem na mesma pasta de modelo."""

    def __init__(self, pasta):
        self.arq = Path(pasta) / "em_execucao.lock"

    def adquirir(self):
        if self.arq.exists():
            dono = ler_json(self.arq, {}).get("pid")
            if dono != os.getpid() and pid_vivo(dono):
                return False
        gravar_json(self.arq, dict(pid=os.getpid(), inicio=time.time()))
        return True

    def liberar(self):
        self.arq.unlink(missing_ok=True)


class Estado:
    """Arquivo estado.json de um modelo, lido pelo monitor."""

    def __init__(self, pasta, etapas, **info):
        self.arq = Path(pasta) / "estado.json"
        antigo = ler_json(self.arq, {})
        self.d = dict(etapas=list(etapas), feitas=[], etapa="", fracao=0.0, situacao="rodando", msg="",
                      inicio=time.time(), segundos_antes=float(antigo.get("segundos", 0.0)), segundos=0.0,
                      pid=os.getpid(), avisos=[], erros=[])
        self.d.update(info)
        self._t = 0.0
        self._pai = os.getppid()
        self._t_etapa = time.time()
        self.gravar()
        self.evento("sessao", pid=os.getpid(), etapas=list(etapas), feitas_antes=antigo.get("feitas", []))

    def evento(self, tipo, **kw):
        """Acrescenta uma linha ao registro do modelo (registro.jsonl): o historico completo do processo."""
        with open(self.arq.with_name("registro.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(t=round(time.time(), 3), evento=tipo, **kw), ensure_ascii=False) + "\n")

    def gravar(self):
        if os.getppid() != self._pai:   # o processo principal morreu: nao continuar como orfao
            raise SystemExit("processo principal encerrado")
        self.d["atualizado"] = time.time()
        self.d["segundos"] = self.d["segundos_antes"] + time.time() - self.d["inicio"]
        gravar_json(self.arq, self.d)
        self._t = time.time()

    def info(self, **kw):
        self.d.update(kw)
        self.gravar()

    def iniciar(self, etapa, msg=""):
        self.d.update(etapa=etapa, fracao=0.0, msg=msg)
        self._t_etapa = time.time()
        self.gravar()
        self.evento("inicio", etapa=etapa)

    def progresso(self, fracao, msg=None):
        self.d["fracao"] = float(min(1.0, max(0.0, fracao)))
        mudou = msg is not None and msg != self.d["msg"]
        if msg is not None:
            self.d["msg"] = msg
        if mudou or time.time() - self._t > 0.7:
            self.gravar()

    def concluir(self, etapa):
        if etapa not in self.d["feitas"]:
            self.d["feitas"].append(etapa)
        self.d["fracao"] = 1.0
        self.gravar()
        self.evento("fim", etapa=etapa, segundos=round(time.time() - self._t_etapa, 2))

    def aviso(self, texto):
        self.d["avisos"].append(texto)
        self.gravar()
        self.evento("aviso", texto=texto)

    def retomada(self, etapa, passo):
        """Registra que a cadeia desta etapa continuou de um estado salvo."""
        self.d.setdefault("retomadas", []).append(dict(etapa=etapa, passo=int(passo)))
        self.gravar()
        self.evento("retomada", etapa=etapa, passo=int(passo))

    def erro(self, etapa, exc):
        self.d["erros"].append(f"{etapa}: {type(exc).__name__}: {str(exc)[:200]}")
        self.gravar()
        self.evento("erro", etapa=etapa, texto=f"{type(exc).__name__}: {str(exc)[:500]}")

    def fim(self):
        self.d.update(situacao="erro" if self.d["erros"] else "concluido", etapa="", fracao=1.0)
        self.gravar()
        self.evento("encerrado", situacao=self.d["situacao"], segundos=round(self.d["segundos"], 1))


# ------------------------------------------------------------------------------------ arquivo do processo
PACOTES = ("numpy", "scipy", "pandas", "matplotlib", "seaborn", "numba", "rich", "torch", "sbi", "tabpfn", "amici",
           "petab", "pypesto", "benchmark_models_petab")


def sha256(caminho):
    import hashlib

    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def codigo():
    """Hash de cada arquivo do pacote e um hash do conjunto."""
    import hashlib

    raiz = Path(__file__).parent
    arqs = {a.name: sha256(a) for a in sorted(raiz.glob("*.py"))}
    return dict(arquivos=arqs, total=hashlib.sha256("".join(arqs.values()).encode()).hexdigest()[:16])


def copiar_codigo(destino):
    """Copia os arquivos do pacote para a pasta de resultados, para que a execucao possa ser refeita."""
    import shutil

    destino = Path(destino) / "a0bench"
    destino.mkdir(parents=True, exist_ok=True)
    for a in Path(__file__).parent.glob("*.py"):
        shutil.copy2(a, destino / a.name)
    for nome in ("requirements.txt", "requirements-lock.txt", "README.md", "pyproject.toml"):
        a = Path(__file__).parent.parent / nome
        if a.exists():
            shutil.copy2(a, destino.parent / nome)


def ambiente():
    """Descricao da maquina e das versoes dos pacotes."""
    import platform
    import sys
    from importlib import metadata

    pac = {}
    for n in PACOTES:
        try:
            pac[n] = metadata.version(n)
        except Exception:  # noqa: BLE001  (pacote opcional ausente)
            pac[n] = None
    return dict(python=sys.version.split()[0], plataforma=platform.platform(), maquina=platform.node(),
                processador=platform.processor() or platform.machine(), nucleos=os.cpu_count(), pacotes=pac,
                codigo=codigo())


def registrar(saida, texto):
    """Acrescenta uma linha com data e hora ao registro geral da execucao (registro.log)."""
    with open(Path(saida) / "registro.log", "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {texto}\n")
