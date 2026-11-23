"""Cadeias de Markov: amostradores, gravacao em disco a cada n passos, retomada e diagnosticos.

Amostradores (todos trabalham com C cadeias independentes em lote, uma por semente):
  amostrar_pt        Metropolis adaptativo; com mais de uma temperatura vira temperacao paralela (referencia)
  amostrar_mala      MALA pre-condicionado
  amostrar_hmc       HMC com numero de passos sorteado
  amostrar_ensemble  ensemble invariante afim (movimento de esticamento)
  amostrar_laplace   aproximacao gaussiana no MAP (nao e cadeia; serve de linha de base e de ponto de partida)

Pausa e retomada: o Gravador escreve a cadeia num arquivo .npy mapeado em memoria e, a cada `passos_ck`
passos, descarrega as linhas novas e troca de forma atomica o estado do amostrador (posicoes, adaptacao e
gerador aleatorio). Ao rodar de novo, a cadeia continua do ultimo estado salvo e produz exatamente os mesmos
numeros que produziria sem a pausa.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .util import gravar, gravar_pickle, ler_pickle


# ------------------------------------------------------------------------------------ diagnosticos
def autocorrelacao(x, max_def=None):
    """Funcao de autocorrelacao de uma serie, pelas defasagens 0..max_def."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 or np.var(x) == 0:
        return np.ones(1)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (np.arange(n, 0, -1) * np.var(x))
    return acf if max_def is None else acf[: max_def + 1]


def ess(x):
    """Tamanho efetivo de amostra (sequencia positiva inicial de Geyer)."""
    acf = autocorrelacao(x)
    n = len(np.asarray(x))
    if len(acf) == 1:
        return float(n)
    soma = 0.0
    for k in range(1, n - 1, 2):
        par = acf[k] + acf[k + 1]
        if par < 0:
            break
        soma += par
    return float(n / max(1.0 + 2.0 * soma, 1e-12))


def rhat(cadeias):
    """R-hat com cadeias divididas ao meio; cadeias tem forma (m, n, d)."""
    m, n, d = cadeias.shape
    metades = np.concatenate([cadeias[:, : n // 2], cadeias[:, n // 2: 2 * (n // 2)]], axis=0)
    w = metades.var(axis=1, ddof=1).mean(axis=0)
    b = metades.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(1.0 + b / np.maximum(w, 1e-300))


def diagnosticos(cadeias):
    """R-hat e ESS por parametro de cadeias (m, n, d) ja sem o aquecimento."""
    m, n, d = cadeias.shape
    e = np.array([sum(ess(cadeias[c, :, j]) for c in range(m)) for j in range(d)])
    return dict(rhat=rhat(cadeias), ess=e)


# ------------------------------------------------------------------------------------ gravacao
def _assinatura(p, nome, *partes):
    """Identifica a cadeia: um estado salvo so e retomado se o problema e os ajustes forem os mesmos."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(p.x_obs, float).tobytes())
    h.update(np.ascontiguousarray(np.concatenate([p.lo, p.hi]), float).tobytes())
    h.update(json.dumps([p.nome, nome, *partes], default=float).encode())
    return h.hexdigest()[:16]


class Gravador:
    """Cadeia em disco (uma linha por passo) mais o estado do amostrador, salvos a cada `passos_ck` passos."""

    def __init__(self, base, n, forma, passos_ck, assinatura, criar):
        base = str(base)
        self.arq, self.arq_x, self.arq_lp = Path(base + ".ck"), Path(base + ".ck.cadeia.npy"), Path(base + ".ck.logp.npy")
        self.passos_ck = max(1, int(passos_ck))
        e = None
        if self.arq.exists() and self.arq_x.exists() and self.arq_lp.exists():
            try:
                e = ler_pickle(self.arq)
            except Exception:  # noqa: BLE001  (arquivo truncado por uma queda de energia, por exemplo)
                e = None
            if e is not None and e.get("assinatura") != assinatura:
                e = None
        self.descartado = e is None and self.arq.exists()
        if e is None:
            self.apagar()
            self.arq.parent.mkdir(parents=True, exist_ok=True)
            e = criar()
            e.update(i=0, assinatura=assinatura)
            self.x = open_memmap(self.arq_x, mode="w+", dtype=np.float64, shape=(n, *forma))
            self.lp = open_memmap(self.arq_lp, mode="w+", dtype=np.float64, shape=(n, *forma[:-1]))
            self.retomado = None
        else:
            self.x, self.lp = open_memmap(self.arq_x, mode="r+"), open_memmap(self.arq_lp, mode="r+")
            self.retomado = int(e["i"])
            if self.x.shape[0] < n or self.lp.shape[0] < n:
                self._crescer(n)
        self.e = e
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = e["rng"]

    def _crescer(self, n):
        for nome in ("x", "lp"):
            velho = getattr(self, nome)
            if velho.shape[0] >= n:     # uma pausa no meio do crescimento pode deixar so um dos dois maior
                continue
            arq = self.arq_x if nome == "x" else self.arq_lp
            tmp = arq.with_name(arq.name + f".{os.getpid()}.tmp")
            novo = open_memmap(tmp, mode="w+", dtype=np.float64, shape=(n, *velho.shape[1:]))
            novo[: velho.shape[0]] = velho
            novo.flush()
            del velho, novo
            setattr(self, nome, None)
            os.replace(tmp, arq)
            setattr(self, nome, open_memmap(arq, mode="r+"))

    def registrar(self, x, lp):
        """Grava o passo atual e avanca o contador."""
        i = self.e["i"]
        self.x[i], self.lp[i] = x, lp
        self.e["i"] = i + 1

    def salvar(self):
        self.x.flush()
        self.lp.flush()
        self.e["rng"] = self.rng.bit_generator.state
        gravar_pickle(self.arq, self.e)

    def talvez_salvar(self, n):
        if self.e["i"] % self.passos_ck == 0 and self.e["i"] < n:
            self.salvar()

    def concluir(self, n, manter=False):
        """Devolve a cadeia e o log da densidade com a cadeia no primeiro eixo: (C, n, ...)."""
        if manter:
            self.salvar()
        x, lp = np.array(self.x[:n]), np.array(self.lp[:n])
        if not manter:
            self.apagar()
        return np.moveaxis(x, 0, 1), np.moveaxis(lp, 0, 1)

    def apagar(self):
        self.x = self.lp = None
        for a in (self.arq, self.arq_x, self.arq_lp):
            a.unlink(missing_ok=True)


def apagar_ck(base):
    for suf in (".ck", ".ck.cadeia.npy", ".ck.logp.npy"):
        Path(str(base) + suf).unlink(missing_ok=True)


# ------------------------------------------------------------------------------------ amostradores
def amostrar_laplace(p, mapa, n, rng):
    L = np.linalg.cholesky(mapa["cov"] + 1e-12 * np.eye(p.d))
    aceitas, total = [], 0
    for _ in range(200):
        z = mapa["x"] + rng.standard_normal((4 * n, p.d)) @ L.T
        aceitas.append(z[p.dentro(z)])
        total += len(aceitas[-1])
        if total >= n:
            break
    z = np.concatenate(aceitas)[:n]
    if len(z) < n:
        extra = np.clip(mapa["x"] + rng.standard_normal((n - len(z), p.d)) @ L.T, p.lo, p.hi)
        z = np.concatenate([z, extra])
    return z


def amostrar_pt(p, X0, cov0, n, semente, base, passos_ck, betas=(1.0,), prog=None, manter=False):
    """C execucoes independentes de Metropolis adaptativo com temperacao paralela (K temperaturas).

    Devolve um dicionario com as cadeias frias (C, n, d), o log da posterior (C, n), a aceitacao (C,) e a
    taxa de troca. Com manter=True o estado final fica em disco, e uma chamada posterior com n maior
    continua a mesma cadeia.
    """
    C, d, K = len(X0), p.d, len(betas)
    betas = np.asarray(betas, float)
    escala, eye, cs = 2.38 ** 2 / d, 1e-10 * np.eye(d), np.arange(C)

    def criar():
        X = np.repeat(np.asarray(X0, float)[:, None, :], K, axis=1)
        covs = np.tile(np.asarray(cov0, float), (C, K, 1, 1))
        return dict(X=X, LL=p.logpost_lote(X.reshape(-1, d)).reshape(C, K), medias=X.copy(), covs=covs,
                    Ls=np.linalg.cholesky(escala * covs + eye), log_s=np.zeros((C, K)), aceitos=np.zeros(C),
                    trocas=0, rng=np.random.default_rng(semente).bit_generator.state)
    g = Gravador(base, n, (C, d), passos_ck, _assinatura(p, "pt", C, betas.tolist(), semente), criar)
    e, rng = g.e, g.rng
    while e["i"] < n:
        i = e["i"]
        Z = rng.standard_normal((C, K, d))
        prop = e["X"] + np.exp(0.5 * e["log_s"])[..., None] * np.einsum("ckij,ckj->cki", e["Ls"], Z)
        ll = p.logpost_lote(prop.reshape(-1, d)).reshape(C, K)
        with np.errstate(invalid="ignore"):
            ac = np.log(rng.random((C, K))) < betas * (ll - e["LL"])
        e["X"] = np.where(ac[..., None], prop, e["X"])
        e["LL"] = np.where(ac, ll, e["LL"])
        e["aceitos"] += ac[:, 0]
        e["log_s"] += (ac - 0.234) / (i + 100.0) ** 0.6
        gam = 1.0 / (i + 100.0)
        dif = e["X"] - e["medias"]
        e["medias"] += gam * dif
        e["covs"] = (1 - gam) * e["covs"] + gam * dif[..., :, None] * dif[..., None, :]
        if K > 1:
            k = rng.integers(K - 1, size=C)
            with np.errstate(invalid="ignore"):
                tr = np.log(rng.random(C)) < (betas[k] - betas[k + 1]) * (e["LL"][cs, k + 1] - e["LL"][cs, k])
            for c in np.flatnonzero(tr):
                e["X"][c, [k[c], k[c] + 1]] = e["X"][c, [k[c] + 1, k[c]]]
                e["LL"][c, [k[c], k[c] + 1]] = e["LL"][c, [k[c] + 1, k[c]]]
            e["trocas"] += int(tr.sum())
        g.registrar(e["X"][:, 0], e["LL"][:, 0])
        if e["i"] % 100 == 0:
            e["Ls"] = np.linalg.cholesky(escala * e["covs"] + eye)
            if prog:
                prog(e["i"] / n)
        g.talvez_salvar(n)
    cadeia, logp = g.concluir(n, manter)
    return dict(cadeia=cadeia, logp=logp, aceitacao=e["aceitos"] / n, troca=e["trocas"] / (n * C),
                retomado=g.retomado)


def _adaptar(e, i, queima, ac, alvo):
    if i < queima:
        e["log_e"] += (ac - alvo) / (i + 10.0) ** 0.6
        gam = 1.0 / (i + 50.0)
        dif = e["Z"] - e["media"]
        e["media"] += gam * dif
        e["cov"] = (1 - gam) * e["cov"] + gam * dif[:, :, None] * dif[:, None, :]


def _de_z(p, Z, LP):
    """Cadeia e log da densidade do espaco sem restricoes para o dos parametros."""
    forma = Z.shape
    X, s = p.de_z(Z.reshape(-1, forma[-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = LP.reshape(-1) - np.sum(np.log(s) + np.log1p(-s), axis=1)
    return X.reshape(forma), ll.reshape(forma[:-1])


def amostrar_mala(p, Z0, cov0, n, semente, base, passos_ck, aquec=0.5, prog=None):
    """MALA pre-condicionado no espaco sem restricoes; passo e covariancia adaptados so no aquecimento."""
    C, d = Z0.shape
    queima, eye = int(aquec * n), 1e-10 * np.eye(d)

    def criar():
        lp, gr = p.alvo_grad_z(Z0)
        cov = np.tile(cov0, (C, 1, 1))
        return dict(Z=Z0.copy(), LP=lp, G=gr, log_e=np.zeros(C), media=Z0.copy(), cov=cov,
                    L=np.linalg.cholesky(cov + eye), aceitos=np.zeros(C),
                    rng=np.random.default_rng(semente).bit_generator.state)
    g = Gravador(base, n, (C, d), passos_ck, _assinatura(p, "mala", C, n, aquec, semente), criar)
    e, rng = g.e, g.rng

    def logq(a, media, eps):
        v = np.linalg.solve(e["L"], (a - media)[..., None])[..., 0]
        return -0.5 * np.sum(v ** 2, axis=1) / eps ** 2
    while e["i"] < n:
        i = e["i"]
        eps = np.exp(e["log_e"])
        m1 = e["Z"] + 0.5 * eps[:, None] ** 2 * np.einsum("cij,cj->ci", e["cov"], e["G"])
        Zp = m1 + eps[:, None] * np.einsum("cij,cj->ci", e["L"], rng.standard_normal((C, d)))
        lpp, gp = p.alvo_grad_z(Zp)
        m2 = Zp + 0.5 * eps[:, None] ** 2 * np.einsum("cij,cj->ci", e["cov"], gp)
        with np.errstate(invalid="ignore"):
            ac = np.log(rng.random(C)) < lpp - e["LP"] + logq(e["Z"], m2, eps) - logq(Zp, m1, eps)
        e["Z"] = np.where(ac[:, None], Zp, e["Z"])
        e["G"] = np.where(ac[:, None], gp, e["G"])
        e["LP"] = np.where(ac, lpp, e["LP"])
        _adaptar(e, i, queima, ac, 0.574)
        if i >= queima:
            e["aceitos"] += ac
        g.registrar(e["Z"], e["LP"])
        if e["i"] % 50 == 0:
            if prog:
                prog(e["i"] / n)
            if e["i"] <= queima:
                e["L"] = np.linalg.cholesky(e["cov"] + eye)
        g.talvez_salvar(n)
    Z, LP = g.concluir(n)
    cadeia, logp = _de_z(p, Z, LP)
    return dict(cadeia=cadeia, logp=logp, aceitacao=e["aceitos"] / max(1, n - queima), aquecimento=queima,
                esd=n * p.custo_grad, passo=np.exp(e["log_e"]), retomado=g.retomado)


def amostrar_hmc(p, Z0, cov0, n, semente, base, passos_ck, passos_max, aquec=0.5, prog=None):
    """HMC com massa igual a inversa da covariancia e numero de passos sorteado em {1..passos_max}."""
    C, d = Z0.shape
    queima, eye = int(aquec * n), 1e-10 * np.eye(d)

    def criar():
        lp, gr = p.alvo_grad_z(Z0)
        cov = np.tile(cov0, (C, 1, 1))
        return dict(Z=Z0.copy(), LP=lp, G=gr, log_e=np.full(C, np.log(0.5)), media=Z0.copy(), cov=cov,
                    L=np.linalg.cholesky(cov + eye), aceitos=np.zeros(C), n_grad=0,
                    rng=np.random.default_rng(semente).bit_generator.state)
    g = Gravador(base, n, (C, d), passos_ck, _assinatura(p, "hmc", C, n, aquec, passos_max, semente), criar)
    e, rng = g.e, g.rng
    while e["i"] < n:
        i = e["i"]
        L = e["L"]
        eps = np.exp(e["log_e"])[:, None]
        r0 = rng.standard_normal((C, d))
        nl = int(rng.integers(1, passos_max + 1))
        z, gr, r = e["Z"].copy(), e["G"].copy(), r0.copy()
        lp = e["LP"].copy()
        for _ in range(nl):
            r = r + 0.5 * eps * np.einsum("cji,cj->ci", L, gr)
            z = z + eps * np.einsum("cij,cj->ci", L, r)
            lp, gr = p.alvo_grad_z(z)
            r = r + 0.5 * eps * np.einsum("cji,cj->ci", L, gr)
        e["n_grad"] += nl
        with np.errstate(invalid="ignore"):
            dh = lp - 0.5 * np.sum(r ** 2, axis=1) - e["LP"] + 0.5 * np.sum(r0 ** 2, axis=1)
            ac = np.log(rng.random(C)) < dh
        e["Z"] = np.where(ac[:, None], z, e["Z"])
        e["G"] = np.where(ac[:, None], gr, e["G"])
        e["LP"] = np.where(ac, lp, e["LP"])
        _adaptar(e, i, queima, ac, 0.75)
        if i >= queima:
            e["aceitos"] += ac
        g.registrar(e["Z"], e["LP"])
        if e["i"] % 20 == 0:
            if prog:
                prog(e["i"] / n)
            if e["i"] <= queima:
                e["L"] = np.linalg.cholesky(e["cov"] + eye)
        g.talvez_salvar(n)
    Z, LP = g.concluir(n)
    cadeia, logp = _de_z(p, Z, LP)
    return dict(cadeia=cadeia, logp=logp, aceitacao=e["aceitos"] / max(1, n - queima), aquecimento=queima,
                esd=e["n_grad"] * p.custo_grad, passo=np.exp(e["log_e"]), retomado=g.retomado)


def amostrar_ensemble(p, Z0, n, semente, base, passos_ck, a=2.0, prog=None):
    """Amostrador de ensemble invariante afim (movimento de esticamento); Z0 tem forma (C, W, d)."""
    C, W, d = Z0.shape

    def criar():
        return dict(Z=Z0.copy(), LP=p.alvo_z(Z0.reshape(-1, d)).reshape(C, W), aceitos=np.zeros(C),
                    rng=np.random.default_rng(semente).bit_generator.state)
    g = Gravador(base, n, (C, W, d), passos_ck, _assinatura(p, "ensemble", C, W, n, a, semente), criar)
    e, rng = g.e, g.rng
    metade, cs = W // 2, np.arange(C)[:, None]
    grupos = [(np.arange(metade), np.arange(metade, W)), (np.arange(metade, W), np.arange(metade))]
    while e["i"] < n:
        for S, R in grupos:
            par = R[rng.integers(len(R), size=(C, len(S)))]
            zz = ((a - 1.0) * rng.random((C, len(S))) + 1.0) ** 2 / a
            Zr = e["Z"][cs, par]
            prop = Zr + zz[..., None] * (e["Z"][:, S] - Zr)
            lpp = p.alvo_z(prop.reshape(-1, d)).reshape(C, len(S))
            with np.errstate(invalid="ignore"):
                ac = np.log(rng.random((C, len(S)))) < (d - 1) * np.log(zz) + lpp - e["LP"][:, S]
            e["Z"][:, S] = np.where(ac[..., None], prop, e["Z"][:, S])
            e["LP"][:, S] = np.where(ac, lpp, e["LP"][:, S])
            e["aceitos"] += ac.sum(axis=1)
        g.registrar(e["Z"], e["LP"])
        if e["i"] % 20 == 0 and prog:
            prog(e["i"] / n)
        g.talvez_salvar(n)
    Z, LP = g.concluir(n)
    cadeia, logp = _de_z(p, Z, LP)
    return dict(cadeia=cadeia, logp=logp, aceitacao=e["aceitos"] / (n * W), aquecimento=n // 4, esd=n * W,
                retomado=g.retomado)


# ------------------------------------------------------------------------------------ orquestracao
def passos_do_orcamento(nome, p, cfg):
    """Numero de passos de cada amostrador para o mesmo orcamento de ESD por cadeia."""
    orc, d = cfg["orcamento_esd"], p.d
    if nome == "metropolis":
        return orc
    if nome == "mala":
        return max(200, int(orc / p.custo_grad))
    if nome == "hmc":
        return max(200, int(orc / (p.custo_grad * (cfg["hmc_passos_max"] + 1) / 2)))
    if nome == "ensemble":
        return max(40, orc // max(8, 4 * d))
    raise ValueError(nome)


def rodar_amostrador(nome, p, mapa, cfg, pasta, prog=None):
    """Roda um amostrador para todas as sementes com o mesmo orcamento de ESD.

    Grava as cadeias completas em cadeias/<nome>.npz e devolve, por semente, as amostras ja sem o
    aquecimento (em theta) e o diagnostico. O terceiro valor e o passo em que a cadeia foi retomada (ou None).
    """
    S, d = cfg["sementes"], p.d
    C, base, ck = len(S), pasta / nome, cfg["passos_ck"]
    lap = amostrar_laplace(p, mapa, 500, np.random.default_rng(99))
    covz = np.cov(p.para_z(lap).T).reshape(d, d) + 1e-8 * np.eye(d)
    z_map = p.para_z(mapa["x"][None])
    n, W = passos_do_orcamento(nome, p, cfg), 1
    if nome == "metropolis":
        r = amostrar_pt(p, np.tile(mapa["x"], (C, 1)), mapa["cov"], n, 2000, base, ck, prog=prog)
        r.update(aquecimento=n // 4, esd=float(n))
    elif nome == "mala":
        r = amostrar_mala(p, np.tile(z_map, (C, 1)), covz, n, 2001, base, ck, cfg["aquecimento_grad"], prog)
    elif nome == "hmc":
        r = amostrar_hmc(p, np.tile(z_map, (C, 1)), covz, n, 2002, base, ck, cfg["hmc_passos_max"],
                         cfg["aquecimento_grad"], prog)
    elif nome == "ensemble":
        W = max(8, 4 * d)
        Z0 = p.para_z(amostrar_laplace(p, mapa, C * W, np.random.default_rng(2003))).reshape(C, W, d)
        r = amostrar_ensemble(p, Z0, n, 2003, base, ck, prog=prog)
    else:
        raise ValueError(nome)
    q, cad = r["aquecimento"], r["cadeia"]
    pos = cad[:, q:]
    series = pos.mean(axis=2) if nome == "ensemble" else pos
    rh = rhat(pos.reshape(C, -1, d))      # no ensemble, cada conjunto de caminhantes conta como uma cadeia
    k = max(1, int(cfg["desbaste_cadeias"]))
    arq = pasta / "cadeias" / f"{nome}.npz"
    gravar(arq, lambda fh: np.savez(
        fh, cadeia=cad[:, ::k], logp=r["logp"][:, ::k], aquecimento=q, desbaste=k, passos=n, caminhantes=W,
        aceitacao=np.asarray(r["aceitacao"], float), esd=float(r["esd"]), sementes=np.array(S),
        rhat=rh, nomes=np.array(p.nomes)))
    saida = []
    for c in range(C):
        x = pos[c].reshape(-1, d)
        e = min(ess(series[c][:, j]) for j in range(d)) * W
        e = min(e, len(x))
        passo = np.linspace(0, len(x) - 1, min(cfg["n_amostras"], len(x))).astype(int)
        saida.append((x[passo], dict(aceitacao=float(r["aceitacao"][c]), ess_min=float(e),
                                     esd=float(mapa["esd"] + r["esd"]), ess_por_mil_esd=float(1e3 * e / r["esd"]),
                                     rhat_max=float(rh.max()), passos=int(n))))
    return saida, r["retomado"]
