"""Analise de identificabilidade em quatro camadas.

1. Perfil de verossimilhanca (Raue et al., 2009): varredura local com passo adaptativo, varredura global
   iniciada nas amostras da referencia (para achar outros modos), trajetorias dos demais parametros e
   distincao entre limite imposto pelos dados e pelo prior. Intervalos de confianca da secao 2.2 do artigo:
   assintoticos pela Hessiana (eq. 7) e pela verossimilhanca (eq. 8), pontuais e simultaneos.
2. Estrutura local: posto do jacobiano das saidas (Stigter e Molenaar, 2015) em varios pontos do prior e
   reparametrizacao por imagem invariante (Maclaren et al., 2025), que devolve as combinacoes de
   parametros identificaveis e as nao identificaveis.
3. Subespaco informado pela verossimilhanca: cota certificada do erro (Zahm et al., 2022) apos normalizar o
   prior (Cui, Tong e Zahm, 2022), nas versoes a posteriori e livre de dados (Cui e Zahm, 2021), e o mesmo
   espectro na caixa unitaria do prior, como diagnostico de direcoes lineares.
4. Intervalos de predicao perfil a perfil (Simpson e Maclaren, 2023), comparados a banda da referencia.
"""
import json
from fractions import Fraction
from math import lcm

import numpy as np

from .config import VERSAO_IDENT
from .util import gravar_json, gravar_pickle, ler_pickle

VERSAO = VERSAO_IDENT
LIM1 = 1.9207   # limiar pontual em -log L: chi2(nivel; 1 g.l.) / 2. Redefinido em tarefa_identificabilidade


# ---------------------------------------------------------------------------------------- perfil
def _ativos(p, x, i):
    larg = p.hi - p.lo
    perto = (x - p.lo < 1e-7 * larg) | (p.hi - x < 1e-7 * larg)
    perto[i] = False
    return frozenset(np.flatnonzero(perto).tolist())


def _presos(p, x, candidatos):
    """Dos parametros encostados num limite, quais estao de fato presos por ele, e com que forca: quanto
    -log L piora, por largura do prior, ao afastar o parametro do limite (so os acima de 1). Um parametro sem
    efeito na verossimilhanca tambem para num limite, mas nao esta preso e nao limita o perfil de ninguem."""
    if not candidatos:
        return {}
    larg, f0, presos = p.hi - p.lo, p.nll(x), {}
    for j in candidatos:
        for h in (1e-3, 1e-4):   # se o passo cai onde a simulacao falha, tenta um menor; se falhar, nao conta
            xj = np.array(x, float)
            xj[j] += h * larg[j] * (1.0 if x[j] - p.lo[j] < p.hi[j] - x[j] else -1.0)
            fj = p.nll(xj)
            if np.isfinite(fj):
                if (fj - f0) / h > 1.0:
                    presos[j] = float((fj - f0) / h)
                break
    return presos


def perfilar(p, i, mapa, nll_map, ref, ll_ref, dp, cfg, limd):
    """Perfil de -log L do parametro i. Devolve (v, nll, X) ordenados por v."""
    lo, hi = float(p.lo[i]), float(p.hi[i])
    L = hi - lo
    pts = {}

    def guardar(v, f, x):
        k = round(float(v), 12)
        if k not in pts or f < pts[k][0]:
            pts[k] = (float(f), np.array(x, float))
        return pts[k]

    def avaliar(v, inicios):
        melhor = None
        for x0 in inicios:
            f, x = p.perfil_ponto(i, float(v), x0)
            if np.isfinite(f) and (melhor is None or f < melhor[0]):
                melhor = (f, x)
        return guardar(v, *melhor) if melhor else None

    x_map, v0 = np.array(mapa["x"], float), float(mapa["x"][i])
    guardar(v0, nll_map, x_map)
    parar = 1.15 * max(limd, 2 * LIM1)

    # 1) varredura local com passo adaptativo (Raue et al., 2009)
    #    o passo e aceito se -log L sobe no maximo 3 alvos; se sobe mais, o passo encolhe e e refeito
    alvo, h_min = LIM1 / 4, 1e-6 * L
    dp_local = min(float(dp[i]), float(np.sqrt(max(mapa["cov"][i, i], 0.0))) or float(dp[i]))
    for sentido in (1.0, -1.0):
        v, x, f, fmin = v0, x_map.copy(), nll_map, nll_map
        x_ant, v_ant = None, None
        h = float(np.clip(0.3 * dp_local, 1e-4 * L, L / 20))
        passos = recuos = falhas = 0
        while passos < cfg["perfil_passos_max"]:
            v_novo = float(np.clip(v + sentido * h, lo, hi))
            if abs(v_novo - v) < 1e-12 * max(1.0, L):
                break
            chute = x.copy()
            if x_ant is not None and v != v_ant:
                chute = np.clip(x + (x - x_ant) * (v_novo - v) / (v - v_ant), p.lo, p.hi)
            r = avaliar(v_novo, [chute, x])
            if r is None:   # a simulacao falhou: aproxima-se da borda da regiao viavel com passos menores
                if falhas < 3 and h > h_min:
                    h, falhas = max(h / 4, h_min), falhas + 1
                    continue
                break
            inc = r[0] - f
            if inc > 3 * alvo and f - fmin <= LIM1 and h > h_min and recuos < 10:
                h, recuos = max(h / 4, h_min), recuos + 1
                continue
            passos, recuos = passos + 1, 0
            x_ant, v_ant, v, x, f = x, v, v_novo, r[1], r[0]
            fmin = min(fmin, f)
            if f - fmin > parar or v_novo in (lo, hi):
                break
            if inc > 2 * alvo:
                h = max(h / 2, h_min)
            elif inc < alvo / 2:
                h = min(2 * h, L / 12)

    # 2) varredura global: grade regular e quantis da referencia, iniciada nas melhores amostras
    grade = list(np.linspace(lo, hi, cfg["n_perfil_global"]))
    grade += list(np.quantile(ref[:, i], [0.01, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 0.99]))
    meia = L / (2 * (cfg["n_perfil_global"] - 1))
    extras = p.amostrar_prior(int(cfg.get("perfil_inicios_extras", 0)), np.random.default_rng(900 + i))
    for v in sorted(grade, key=lambda g: abs(g - v0)):   # de dentro para fora: cada ponto parte do vizinho
        if min(abs(v - k) for k in pts) < (1e-12 if v in (lo, hi) else 0.01) * L:   # os limites entram sempre
            continue
        inicios = []
        viz = np.abs(ref[:, i] - v) < meia
        if viz.any():
            inicios.append(ref[int(np.argmax(np.where(viz, ll_ref, -np.inf)))])
        inicios.append(pts[min(pts, key=lambda k: abs(k - v))][1])
        avaliar(v, inicios + list(extras))

    # 3) reparo: um maximo local pode ser so um otimo local ruim; tenta de novo a partir dos vizinhos
    for _ in range(2):
        vs, mudou = sorted(pts), False
        for a, b, c in zip(vs[:-2], vs[1:-1], vs[2:]):
            if pts[b][0] > max(pts[a][0], pts[c][0]) + 0.05:
                antes = pts[b][0]
                avaliar(b, [pts[a][1], pts[c][1]])
                mudou |= pts[b][0] < antes - 1e-6
        if not mudou:
            break

    # 3b) polimento: dentro de 1,2 x o limiar simultaneo (a parte que decide os intervalos), cada ponto e
    #     tentado de novo a partir do otimo e dos dois vizinhos (e dos inicios extras); fica o menor valor.
    #     Depois da primeira passada, so os vizinhos de quem melhorou sao revistos.
    if cfg.get("perfil_polimento", True):
        k0, rever = round(v0, 12), None
        for _ in range(4):
            vs = sorted(pts)
            fmin = min(f for f, _ in pts.values())
            melhorou = set()
            for k, b in enumerate(vs):
                if b == k0 or pts[b][0] - fmin > 1.2 * limd or (rever is not None and b not in rever):
                    continue
                antes = pts[b][0]
                viz = [vs[j] for j in (k - 1, k + 1) if 0 <= j < len(vs)]
                avaliar(b, [x_map] + [pts[j][1] for j in viz] + list(extras))
                if pts[b][0] < antes - 1e-3:
                    melhorou |= set(viz)
            if not melhorou:
                break
            rever = melhorou

    # 4) refino dos cruzamentos do limiar pontual, ate 2% do prior ou 1/4 da largura do componente
    for _ in range(3):
        vs = sorted(pts)
        fmin = min(f for f, _ in pts.values())
        comp = _intervalos(np.array(vs), np.array([pts[k][0] - fmin for k in vs]), LIM1)
        for a, b in zip(vs[:-1], vs[1:]):
            da, db = pts[a][0] - fmin - LIM1, pts[b][0] - fmin - LIM1
            if da * db >= 0:
                continue
            larg_c = min((c[1] - c[0] for c in comp if a <= c[0] <= b or a <= c[1] <= b), default=L)
            if b - a > max(min(0.02 * L, 0.25 * larg_c), 1e-5 * L):
                w = float(np.clip(abs(da) / (abs(da) + abs(db)), 0.25, 0.75))
                avaliar(a + (b - a) * w, [pts[a][1], pts[b][1]])

    vs = sorted(pts)
    return np.array(vs), np.array([pts[k][0] for k in vs]), np.array([pts[k][1] for k in vs])


def _intervalos(v, delta, lim):
    """Uniao de intervalos em que delta <= lim, com interpolacao linear nos cruzamentos."""
    saida, ini = [], None
    for k in range(len(v)):
        dentro = delta[k] <= lim
        if dentro and ini is None:
            ini = v[k] if k == 0 else v[k - 1] + (lim - delta[k - 1]) / (delta[k] - delta[k - 1]) * (v[k] - v[k - 1])
        if not dentro and ini is not None:
            fim = v[k - 1] + (lim - delta[k - 1]) / (delta[k] - delta[k - 1]) * (v[k] - v[k - 1])
            saida.append([float(ini), float(fim)])
            ini = None
    if ini is not None:
        saida.append([float(ini), float(v[-1])])
    return saida


def analisar_perfil(p, i, v, nll, X, nll_min, limd):
    """Classifica o parametro a partir do perfil. Cada lado do conjunto de confianca e: dados (o perfil cruza
    o limiar), aberto (chega ao limite do prior), prior (outro parametro ficou preso no limite dele) ou falha
    (o perfil acaba antes do limite do prior porque a simulacao deixa de existir)."""
    delta = nll - nll_min
    k0 = int(np.argmin(delta))
    presos = [_presos(p, X[k], _ativos(p, X[k], i)) for k in range(len(v))]
    base = set(presos[k0])
    comp = _intervalos(v, delta, LIM1)
    dentro = np.flatnonzero(delta <= LIM1)
    if len(dentro) == 0:    # o minimo deste perfil ficou acima do limiar (otimo instavel): usa so o ponto minimo
        dentro, comp = np.array([k0]), [[float(v[k0]), float(v[k0])]]
    larg_i = float(p.hi[i] - p.lo[i])

    # cada extremo do conjunto de confianca (a uniao dos componentes, para que modos simetricos recebam a
    # mesma classe): aberto se chega ao limite do prior; falha se o perfil acaba antes; senao, dados ou prior.
    # E prior quando outro parametro ficou preso num limite que nao prendia no minimo vizinho e, em primeira
    # ordem, alargar esse limite em 10% deslocaria o extremo em mais de 5% da largura do componente.
    lados, bloqueio = {}, set()
    for nome, k_in, passo, c in (("inf", int(dentro[0]), -1, comp[0]), ("sup", int(dentro[-1]), 1, comp[-1])):
        k_fora = k_in + passo
        if k_fora < 0 or k_fora >= len(v):
            borda = p.lo[i] if passo < 0 else p.hi[i]
            lados[nome] = "aberto" if abs(v[k_in] - borda) < 1e-9 * larg_i else "falha"
            continue
        k_min = k_in
        while 0 <= k_min - passo < len(v) and delta[k_min - passo] <= delta[k_min]:
            k_min -= passo
        novos = {j: f for j, f in presos[k_fora].items() if j not in presos[k_min]}
        subida = max((delta[k_fora] - delta[k_in]) / max(abs(v[k_fora] - v[k_in]), 1e-300), 1e-300)
        desloc = 0.1 * sum(novos.values()) / subida
        if novos and desloc > 0.05 * max(c[1] - c[0], 1e-3 * larg_i):
            lados[nome] = "prior"
            bloqueio |= set(novos)
        else:
            lados[nome] = "dados"
    n_dados = sum(e == "dados" for e in lados.values())
    principal = next((c for c in comp if c[0] - 1e-12 <= v[k0] <= c[1] + 1e-12), [float(v[k0])] * 2)
    simult = _intervalos(v, delta, limd)

    # trecho em que nenhum outro parametro fica preso num limite novo: o perfil ali reflete so os dados
    livre = np.array([set(presos[k]) <= base for k in range(len(v))])
    larg = p.hi - p.lo
    plano = bool(livre.sum() >= 3 and np.max(delta[livre]) < 0.02
                 and np.ptp(v[livre]) >= 0.2 * larg[i])

    # trajetorias dos demais parametros ao longo do perfil (relacoes funcionais de Raue)
    compensam = []
    sel = livre & (delta <= LIM1)
    if n_dados < 2 and sel.sum() >= 4 and np.ptp(v[sel]) > 0:
        for j in range(p.d):
            if j == i:
                continue
            amp = float(np.ptp(X[sel, j]) / larg[j])
            if amp >= 0.05:
                c = float(np.corrcoef(v[sel], X[sel, j])[0, 1])
                compensam.append(dict(nome=p.nomes[j], amplitude=amp, correlacao=c,
                                      inclinacao=float(np.polyfit(v[sel], X[sel, j], 1)[0])))
        compensam.sort(key=lambda c: -c["amplitude"])
    return dict(
        classe=["nao identificavel", "parcial", "identificavel"][n_dados], lados=lados,
        bloqueio=[p.nomes[j] for j in sorted(bloqueio)], multimodal=len(comp) > 1, n_componentes=len(comp),
        ic=[float(principal[0]), float(principal[1])], ic_conjunto=comp,
        ic_simultaneo=[float(simult[0][0]), float(simult[-1][1])] if simult else [float(v[k0])] * 2,
        plano=plano, compensam=compensam[:4], grade=v.tolist(), delta=delta.tolist(),
        caminho=np.round(X, 8).tolist())


# ---------------------------------------------------------------------------------------- estrutura
def _base_inteira(N, den=6, tol=5e-3):
    """Base de vetores inteiros pequenos para o espaco gerado pelas colunas de N, ou None.
    Eliminacao de Gauss-Jordan com pivo total (o maior elemento restante), para o ruido numerico nao virar pivo."""
    A = np.array(N.T, float)
    n_l, n_c = A.shape
    usadas = []
    for r in range(n_l):
        sub = np.abs(A[r:])
        sub[:, usadas] = 0.0
        k, c = np.unravel_index(int(np.argmax(sub)), sub.shape)
        if sub[k, c] < 1e-3:
            return None
        k += r
        A[[r, k]] = A[[k, r]]
        A[r] /= A[r, c]
        for l in range(n_l):
            if l != r:
                A[l] -= A[l, c] * A[r]
        usadas.append(int(c))
    base = []
    for linha in A:
        fr = [Fraction(float(x)).limit_denominator(den) for x in linha]
        m = lcm(*[f.denominator for f in fr])
        inteiros = np.array([int(f * m) for f in fr])
        if np.max(np.abs(inteiros)) > 12 or np.linalg.norm(linha - inteiros / m) > tol * max(1.0, np.linalg.norm(linha)):
            return None
        base.append(inteiros.tolist())
    return base


def _nome_combinacao(vetor, nomes, coord_log):
    """Texto da combinacao: monomio quando as coordenadas sao logaritmicas, soma caso contrario."""
    termos = [(a, n, e) for a, n, e in zip(vetor, nomes, coord_log) if a != 0]
    if all(e for _, _, e in termos):
        num = [(a, n) for a, n, _ in termos if a > 0]
        den = [(-a, n) for a, n, _ in termos if a < 0]

        def prod(ts):
            return "·".join((n[4:] if n.startswith("log ") else n) + (f"^{a}" if a != 1 else "") for a, n in ts)
        if num and den:
            return f"{prod(num)} / {prod(den)}" if len(den) == 1 else f"{prod(num)} / ({prod(den)})"
        return prod(num) if num else f"1 / ({prod(den)})"
    txt = ""
    for k, (a, n, _) in enumerate(termos):
        sinal = "-" if a < 0 else ("+" if k else "")
        txt += f" {sinal} " if k else sinal
        txt += (f"{abs(a)}·" if abs(a) != 1 else "") + n
    return txt.strip()


def _nome_aproximado(vetor, nomes, coord_log):
    """Direcao sem forma inteira: os termos com peso de pelo menos 10% do maior, com duas casas."""
    v = np.array(vetor, float)
    v = v / v[int(np.argmax(np.abs(v)))]
    ordem = [j for j in np.argsort(-np.abs(v)) if abs(v[j]) >= 0.1]
    partes = []
    for k, j in enumerate(ordem):
        n = ("log " + nomes[j]) if coord_log[j] and not nomes[j].startswith("log") else nomes[j]
        coef = "" if abs(abs(v[j]) - 1) < 5e-3 else f"{abs(v[j]):.2f}·"
        partes.append((" - " if v[j] < 0 else " + ") * (k > 0) + ("-" if v[j] < 0 and k == 0 else "") + coef + n)
    return "≈ " + "".join(partes)


def estrutura_local(p, mapa, cfg, rng):
    """Posto do jacobiano das saidas e reparametrizacao por imagem invariante."""
    d, larg = p.d, p.hi - p.lo
    em_log = np.array([e == "log" for e in p.escalas])

    def jac_t(X):
        """Jacobiano em relacao as coordenadas transformadas (log das taxas positivas)."""
        return p.jac_lote(X) * np.where(em_log, X, 1.0)[:, None, :]

    def posto(s):
        return int(np.sum(s > p.tol_posto * s[0])) if s[0] > 0 else 0

    x0 = np.array(mapa["x"], float)
    J0 = jac_t(x0[None])[0]
    _, s0, Vt = np.linalg.svd(J0, full_matrices=True)
    r0 = posto(s0)
    V0 = Vt[r0:].T

    # o posto se repete em pontos sorteados do prior?
    Xp = p.amostrar_prior(cfg["n_pts_estrutura"], rng)
    J_nat = p.jac_lote(Xp)                                  # coordenadas naturais: usado no subespaco informado
    Jp = J_nat * np.where(em_log, Xp, 1.0)[:, None, :]
    postos, angulos = [], []
    for J in Jp:
        if not np.all(np.isfinite(J)):
            continue
        _, s, Vt_k = np.linalg.svd(J, full_matrices=True)
        rk = posto(s)
        postos.append(rk)
        if V0.shape[1] and rk < d:
            Vk = Vt_k[rk:].T
            angulos.append(float(np.linalg.norm(V0 - Vk @ (Vk.T @ V0), 2)))

    # invariancia do nucleo: derivada do jacobiano ao longo de cada coordenada, aplicada ao nucleo
    N, sem_derivada = np.zeros((d, 0)), []
    if V0.shape[1]:
        blocos = []
        passo_t = 1e-3
        for i in range(d):
            xa, xb = x0.copy(), x0.copy()
            if em_log[i]:
                amp = np.log(p.hi[i] / p.lo[i])
                xa[i], xb[i] = x0[i] * np.exp(passo_t * amp), x0[i] * np.exp(-passo_t * amp)
            else:
                amp = larg[i]
                xa[i], xb[i] = x0[i] + passo_t * amp, x0[i] - passo_t * amp
            Ja, Jb = jac_t(np.array([xa, xb]))
            fa, fb = bool(np.all(np.isfinite(Ja))), bool(np.all(np.isfinite(Jb)))
            if fa and fb:
                blocos.append((Ja - Jb) @ V0 / (2 * passo_t))
            elif fa or fb:   # a simulacao falha de um dos lados do otimo: diferenca unilateral
                blocos.append(((Ja - J0) if fa else (J0 - Jb)) @ V0 / passo_t)
            else:
                sem_derivada.append(p.nomes[i])
        if blocos:
            _, sm, Vm = np.linalg.svd(np.vstack(blocos), full_matrices=True)
            sm = np.concatenate([sm, np.zeros(V0.shape[1] - len(sm))])
            N = V0 @ Vm[sm <= cfg["tol_invariancia"] * s0[0]].T
    U, _, _ = np.linalg.svd(N, full_matrices=True) if N.shape[1] else (np.eye(d), None, None)
    C = U[:, N.shape[1]:]

    def descrever(B):
        if B.shape[1] == 0:
            return []
        base = _base_inteira(B)
        if base is None:
            return [dict(vetor=np.round(B[:, k], 3).tolist(), exata=False,
                         texto=_nome_aproximado(B[:, k], p.nomes, p.coord_log)) for k in range(B.shape[1])]
        return [dict(vetor=v, exata=True, texto=_nome_combinacao(v, p.nomes, p.coord_log)) for v in base]

    # posto generico: o maior entre o otimo e os pontos sorteados do prior. Se ele e menor que d, as d - posto
    # direcoes menos sensiveis no otimo nao sao vistas em lugar nenhum: nao identificabilidade estrutural (ate a
    # tolerancia). Perda de posto so no otimo (posto local menor que o generico) e pratica.
    r_gen = max(postos + [r0])
    V_gen = Vt[r_gen:].T
    participa = np.max(np.abs(V_gen), axis=1) > 0.05 if V_gen.shape[1] else np.zeros(d, bool)
    return dict(
        posto=r0, posto_generico=r_gen, d=d, n_saidas=int(J0.shape[0]), valores_singulares=(s0 / s0[0]).tolist(),
        postos_prior=postos, fracao_mesmo_posto=float(np.mean(np.array(postos) == r0)) if postos else None,
        angulo_max=max(angulos) if angulos else None, dim_nucleo=int(V0.shape[1]),
        dim_nucleo_invariante=int(N.shape[1]), invariancia_nao_testada=sem_derivada, nucleo=descrever(N),
        combinacoes_identificaveis=descrever(C) if N.shape[1] else [],
        deficiente_em_todo_o_prior=bool(r_gen < d), participa=participa.tolist()), J_nat, Xp


# ---------------------------------------------------------------------------------------- subespaco
def _fator_normal(p, X):
    """dtheta/dz da transformacao que leva o prior uniforme na normal padrao: (hi - lo) * phi(z)."""
    from scipy.stats import norm

    u = np.clip((X - p.lo) / (p.hi - p.lo), 1e-9, 1 - 1e-9)
    return (p.hi - p.lo) * norm.pdf(norm.ppf(u))


def _espectro(H, const, certificada):
    """Autovalores, autovetores e a dimensao r(eps): const * soma dos autovalores descartados <= eps."""
    lam, V = np.linalg.eigh(0.5 * (H + H.T))
    lam, V = np.maximum(lam[::-1], 0.0), V[:, ::-1]
    cota = const * np.cumsum(lam[::-1])[::-1]
    return dict(autovalores=lam.tolist(), vetores=np.round(V, 4).tolist(), constante=float(const),
                certificada=certificada, cota_kl=cota.tolist(),
                dim={str(e): int(np.sum(cota > e)) for e in (1.0, 0.1, 0.01)})


def subespaco_informado(p, ref, Jp, Xp, cfg):
    """Matriz H = E[grad log L grad log L^T] a posteriori e sua versao livre de dados (Fisher medio no prior).
    Jp sao os jacobianos das saidas, em coordenadas naturais, nos pontos Xp sorteados do prior.

    Duas coordenadas:
    - normal: o prior uniforme e levado a normal padrao, coordenada a coordenada (normalizacao do prior, Cui,
      Tong e Zahm, 2022). Vale a cota certificada KL <= 1/2 * soma dos autovalores descartados (Zahm et al.,
      2022), e na versao livre de dados a mesma cota em media sobre os dados (Cui e Zahm, 2021).
    - caixa: a caixa unitaria do prior. Os autovetores sao direcoes lineares nos parametros, o que revela
      cristas retas que a normalizacao entorta. A constante e d / 16 NAO vem dos artigos (que nao cobrem prior
      de suporte compacto): sai de escrever a uniforme no cubo como exp(-V - Psi), V = |u - c|^2 / (2t),
      Psi = -V, e aplicar Bakry-Emery e Holley-Stroock com t = d / 8. Leia como diagnostico, nao como certificado.
    """
    d, larg = p.d, p.hi - p.lo
    c_caixa, c_normal = np.e * d / 16.0, 0.5
    X = ref[: cfg["n_lis"]]
    _, g = p.grad_lote(X)
    ok = np.all(np.isfinite(g), axis=1) & np.any(g != 0, axis=1)
    gu, gz = g[ok] * larg, g[ok] * _fator_normal(p, X[ok])
    out = dict(n_amostras=int(ok.sum()),
               caixa=_espectro(gu.T @ gu / max(1, ok.sum()), c_caixa, False),
               normal=_espectro(gz.T @ gz / max(1, ok.sum()), c_normal, True))
    okp = np.array([np.all(np.isfinite(J)) for J in Jp])
    if okp.any():
        Ju, Jz = Jp[okp] * larg, Jp[okp] * _fator_normal(p, Xp[okp])[:, None, :]
        out["livre_de_dados"] = dict(
            n_pontos=int(okp.sum()),
            caixa=_espectro(np.einsum("bni,bnj->ij", Ju, Ju) / okp.sum(), c_caixa, False),
            normal=_espectro(np.einsum("bni,bnj->ij", Jz, Jz) / okp.sum(), c_normal, True))
    return out


# ---------------------------------------------------------------------------------------- predicao
def _caminho_no_ic(q, lim):
    """Pontos do caminho do perfil com delta <= lim, incluindo os cruzamentos interpolados."""
    v, dl, X = np.array(q["grade"]), np.array(q["delta"]), np.array(q["caminho"])
    pts = [X[k] for k in range(len(v)) if dl[k] <= lim]
    for a in range(len(v) - 1):
        if (dl[a] - lim) * (dl[a + 1] - lim) < 0:
            w = (lim - dl[a]) / (dl[a + 1] - dl[a])
            pts.append(X[a] + w * (X[a + 1] - X[a]))
    return np.array(pts)


def predicao_perfil(p, params, mapa, ref, ll_ref, cfg, limd):
    """Intervalos de predicao perfil a perfil: envelope das predicoes ao longo de cada perfil, dentro do
    limiar simultaneo. Comparados a banda de 95% a posteriori da referencia e ao envelope da regiao de
    verossimilhanca completa (amostras da referencia acima do mesmo limiar)."""
    n = cfg["n_pred_ref"]
    yr = p.pred_lote(ref[:n])
    if yr is None:
        raise RuntimeError("nenhuma predição pôde ser simulada")
    bom = np.all(np.isfinite(yr), axis=1)
    lo_r, hi_r = np.quantile(yr[bom], [0.025, 0.975], axis=0)
    w_ref = hi_r - lo_r
    dentro = bom & (ll_ref[:n] >= np.max(ll_ref) - limd)
    w_comp = yr[dentro].max(axis=0) - yr[dentro].min(axis=0) if dentro.sum() >= 5 else None
    y_map = p.pred_lote(np.array(mapa["x"], float)[None])[0]
    lo_u, hi_u = y_map.copy(), y_map.copy()
    larguras = []
    for q in params:
        X = _caminho_no_ic(q, limd)
        y = p.pred_lote(X) if len(X) else None
        y = y_map[None] if y is None else y[np.all(np.isfinite(y), axis=1)]
        if not len(y):
            larguras.append(np.zeros_like(y_map))
            continue
        lo_u, hi_u = np.minimum(lo_u, y.min(axis=0)), np.maximum(hi_u, y.max(axis=0))
        larguras.append(y.max(axis=0) - y.min(axis=0))
    w_u = hi_u - lo_u
    escala = np.maximum(np.abs(yr[bom]).max(axis=0), 1e-300)
    util = (w_ref > 1e-6 * escala) & (w_u > 0)
    if not util.any():
        return dict(n_saidas=int(len(y_map)), razao_posterior=None, contribuicao=[0.0] * len(params))
    return dict(n_saidas=int(len(y_map)), razao_posterior=float(np.median(w_u[util] / w_ref[util])),
                razao_completa=float(np.median(w_u[util] / w_comp[util])) if w_comp is not None else None,
                largura_relativa_ref=float(np.median(w_ref[util] / escala[util])),
                contribuicao=[float(np.median(w[util] / w_u[util])) for w in larguras])


# ---------------------------------------------------------------------------------------- tarefa
def coordenadas(p):
    """Coordenada em que cada perfil e mostrado, como em Raue et al. (2009): log10 do parametro quando ele e
    positivo. Devolve, por parametro, ("log10" ou "lin", rotulo). O perfil nao depende da coordenada; a
    aproximacao quadratica (intervalo assintotico) depende."""
    esc = getattr(p, "escala_petab", None)
    out = []
    for i, n in enumerate(p.nomes):
        if p.escalas[i] == "log":
            out.append(("log10", f"log10({n})"))
        elif esc and esc[i] == "log10":
            out.append(("lin", f"log10({n})"))
        elif esc and esc[i] == "log":
            out.append(("lin", f"ln({n})"))
        else:
            out.append(("lin", n))
    return out


def _para_u(coord, v):
    v = np.asarray(v, float)
    return np.log10(v) if coord == "log10" else v


def _de_u(coord, u):
    u = np.asarray(u, float)
    return 10.0 ** u if coord == "log10" else u


def hessiana(p, x, coords, F):
    """Hessiana de -log L na coordenada do perfil, por diferencas centrais do gradiente (2 d gradientes).
    O passo de cada coordenada e 5% do desvio condicional de Gauss-Newton, limitado a [1e-6, 1e-3] da largura
    do prior; junto a um limite do prior a diferenca fica unilateral. Devolve None se algum gradiente falhar."""
    log = np.array([c == "log10" for c, _ in coords])
    x = np.asarray(x, float)

    def para_u(v):
        return np.where(log, np.log10(np.where(log, v, 1.0)), v)

    def grad(u):
        xx = np.clip(np.where(log, 10.0 ** u, u), p.lo, p.hi)
        ll, g = p.grad_lote(xx[None])
        if not np.isfinite(ll[0]) or not np.all(np.isfinite(g[0])):
            return None
        return -g[0] * np.where(log, xx * np.log(10.0), 1.0)

    u, ulo, uhi = para_u(x), para_u(p.lo), para_u(p.hi)
    larg = uhi - ulo
    h = np.clip(0.05 / np.sqrt(np.maximum(np.diag(F), 1e-300)), 1e-6 * larg, 1e-3 * larg)
    H = np.zeros((p.d, p.d))
    for j in range(p.d):
        ua, ub = u.copy(), u.copy()
        ua[j], ub[j] = min(u[j] + h[j], uhi[j]), max(u[j] - h[j], ulo[j])
        ga, gb = grad(ua), grad(ub)
        if ga is None or gb is None or ua[j] - ub[j] < 0.5 * h[j]:
            return None
        H[j] = (ga - gb) / (ua[j] - ub[j])
    return 0.5 * (H + H.T)


def _variancias(M, esc, base=None, tol=1e-10):
    """Diagonal da inversa generalizada de M (simetrica), com M escalonada por esc e, se base for dada,
    restrita ao espaco gerado por ela. Um parametro com componente numa direcao sem curvatura positiva
    recebe variancia infinita. Devolve (variancias, autovetores, autovalores, bom)."""
    Ms = M / np.outer(esc, esc)
    if base is None:
        lam, V = np.linalg.eigh(Ms)
    else:
        lam, W = np.linalg.eigh(base.T @ Ms @ base)
        V = base @ W
    bom = lam > tol * max(float(lam.max()), 1e-300)
    var = np.sum(V[:, bom] ** 2 / lam[bom], axis=1) / esc ** 2
    var[np.any(V[:, ~bom] ** 2 > 1e-8, axis=1)] = np.inf
    return var, V, lam, bom


def intervalos_assintoticos(p, x, coords, lim1, limd):
    """Intervalos de confianca assintoticos, equacao 7 de Raue et al. (2009):

        sigma_i = theta_i +- sqrt(chi2(alfa, gl) * C_ii),   C = 2 H^-1,   H = Hessiana de chi2 no otimo,

    com gl = 1 (pontual) e gl = numero de parametros (simultaneo), na coordenada do perfil. H e a Hessiana
    exata (diferencas do gradiente), como na equacao. A aproximacao de Gauss-Newton H ~ 2 J^T J (informacao de
    Fisher; J e o jacobiano das saidas dividido pelo desvio do ruido) e calculada tambem: decide quais direcoes
    nao tem sensibilidade nenhuma (C_ii infinito) e serve de comparacao, porque despreza o termo dos residuos
    vezes a curvatura do modelo e pode errar muito em direcoes frouxas. Com a informacao de Fisher singular
    numa direcao que envolve o parametro, o intervalo e infinito (a Hessiana por diferencas nao e confiavel
    ali: a verossimilhanca pode crescer so em quarta ordem). Um parametro cuja Hessiana nao tem curvatura
    positiva acima do ruido numerico (otimo no limite do prior, verossimilhanca plana) ou nao pode ser
    calculada recebe o intervalo de Gauss-Newton, com o motivo anotado. O intervalo ignora a caixa do prior.
    Devolve (lista por parametro, resumo)."""
    x = np.asarray(x, float)
    J = p.jac_lote(x[None])[0]
    if not np.all(np.isfinite(J)):
        return None, None
    J = J * np.array([x[i] * np.log(10.0) if c == "log10" else 1.0 for i, (c, _) in enumerate(coords)])
    F = J.T @ J                                             # = H / 2 em Gauss-Newton, logo C = F^-1
    dF = np.diag(F)
    escF = np.sqrt(np.where(dF > 1e-300, dF, 1.0))
    var_gn, VF, lamF, bomF = _variancias(F, escF)
    var_gn[dF <= 1e-300] = np.inf
    sem_sens = ~np.isfinite(var_gn)                         # sem sensibilidade nenhuma: nao identificavel local
    resumo = dict(autovalores_gauss_newton=np.linalg.eigvalsh(F).tolist())
    try:
        H = hessiana(p, x, coords, F)
    except Exception:  # noqa: BLE001
        H = None
    var_h = None
    if H is not None:
        dH = np.abs(np.diag(H))
        escH = np.sqrt(np.where(np.maximum(dH, dF) > 1e-300, np.maximum(dH, dF), 1.0))
        nulo = (VF[:, ~bomF] / escF[:, None]) * escH[:, None]    # direcoes sem sensibilidade, na escala de H
        base = np.linalg.qr(nulo, mode="complete")[0][:, nulo.shape[1]:] if nulo.shape[1] else np.eye(p.d)
        var_h = _variancias(H, escH, base=base, tol=1e-6)[0]    # abaixo disso e ruido das diferencas
        resumo["autovalores_hessiana"] = np.linalg.eigvalsh(H).tolist()
    larg = p.hi - p.lo
    no_limite = (x - p.lo < 1e-5 * larg) | (p.hi - x < 1e-5 * larg)
    resumo["otimo_no_limite"] = [p.nomes[i] for i in np.flatnonzero(no_limite)]
    out = []
    for i, (c, _) in enumerate(coords):
        u0 = float(_para_u(c, x[i]))
        gn = float(np.sqrt(var_gn[i])) if np.isfinite(var_gn[i]) else None
        hs = float(np.sqrt(var_h[i])) if var_h is not None and not sem_sens[i] and np.isfinite(var_h[i]) else None
        q = dict(u_otimo=u0, dp_hessiana=hs, dp_gauss_newton=gn, no_limite=bool(no_limite[i]), motivo=None,
                 metodo="hessiana" if hs is not None or (sem_sens[i] and H is not None) else "gauss-newton")
        if sem_sens[i]:
            q["motivo"] = "Fisher singular"
        elif hs is None:
            q["motivo"] = "Hessiana sem curvatura positiva" if H is not None else "Hessiana indisponível"
        dp = hs if hs is not None else gn
        if dp is None:
            q.update(infinito=True, dp=None, pontual=None, simultaneo=None)
        else:
            r1, rd = float(np.sqrt(2 * lim1) * dp), float(np.sqrt(2 * limd) * dp)
            q.update(infinito=False, dp=dp, pontual=[u0 - r1, u0 + r1], simultaneo=[u0 - rd, u0 + rd])
        out.append(q)
    usados = [q["metodo"] for q in out]
    resumo["metodo"] = "hessiana" if all(m == "hessiana" for m in usados) else (
        "gauss-newton" if all(m == "gauss-newton" for m in usados) else "misto")
    return out, resumo


def tarefa_identificabilidade(p, pasta, mapa, ref, cfg, prog=None):
    from scipy.stats import chi2

    global LIM1

    arq = pasta / "identificabilidade.json"
    if arq.exists():
        antigo = json.loads(arq.read_text())
        if antigo.get("versao", 1) >= VERSAO:
            return antigo
    amostras = ref["amostras"]
    larg = p.hi - p.lo
    n0 = p.n_aval
    contr = 1.0 - amostras.var(axis=0) / (larg ** 2 / 12.0)
    fim = 0.5 * (mapa["fim"] + mapa["fim"].T) * np.outer(larg, larg)
    val = np.linalg.eigvalsh(fim)[::-1]
    nivel = float(cfg.get("nivel_confianca", 0.95))
    LIM1 = float(chi2.ppf(nivel, 1) / 2)                    # limiares em -log L; em chi2 valem o dobro
    limd = float(chi2.ppf(nivel, p.d) / 2)
    out = dict(versao=VERSAO, problema=p.nome, nomes=list(p.nomes), limites=[p.lo.tolist(), p.hi.tolist()],
               autovalores=val.tolist(), dim_informada=int(np.sum(val > 12.0)), nivel_confianca=nivel,
               limiar_pontual=LIM1, limiar_simultaneo=limd, delta_chi2_pontual=2 * LIM1,
               delta_chi2_simultaneo=2 * limd, erros=[])
    rng = np.random.default_rng(4242)

    def passo(f, msg):
        if prog:
            prog(f, msg)

    # 0) o MAP da otimizacao e mesmo o otimo? A referencia pode ter achado um ponto melhor (modelos
    #    caoticos ou multimodais); nesse caso os perfis partem do ponto melhor, depois de polido.
    sub = amostras[: cfg["n_ref_perfil"]]
    ll_sub = p.logpost_lote(sub)
    dp = amostras.std(axis=0)
    nll_map = p.nll(mapa["x"])
    centro, nll_centro = mapa, nll_map
    k = int(np.argmax(ll_sub))
    if -ll_sub[k] < nll_map - 0.5:
        x, f = sub[k].copy(), float(-ll_sub[k])
        for i in range(min(2, p.d)):
            f2, x2 = p.perfil_ponto(i, float(x[i]), x)
            if f2 < f:
                f, x = f2, x2
        centro, nll_centro = {**mapa, "x": x}, f
        out["map_melhorado"] = dict(melhora=float(nll_map - f), x=x.tolist())

    # 1) perfis. O metodo supoe que eles partem do otimo global. Se um perfil acha um ponto melhor que o centro
    #    (a otimizacao tinha parado num otimo local), todos sao refeitos a partir desse ponto, ate duas vezes.
    #    Cada perfil pronto vai para o disco: numa pausa, a etapa continua do perfil seguinte.
    import hashlib

    ck = pasta / "identificabilidade.ck"
    h = hashlib.sha256()
    for v in (p.x_obs, p.lo, p.hi, mapa["x"], sub):
        h.update(np.ascontiguousarray(v, float).tobytes())
    h.update(json.dumps([p.nome, VERSAO, nivel] + [cfg.get(k) for k in (
        "perfil_passos_max", "n_perfil_global", "perfil_polimento", "perfil_inicios_extras")]).encode())
    assinatura = h.hexdigest()[:16]
    melhoras, volta0, prontos, esd_antes = [], 0, [], 0.0
    if ck.exists():
        try:
            salvo = ler_pickle(ck)
        except Exception:  # noqa: BLE001  (arquivo truncado)
            salvo = None
        if salvo and salvo.get("assinatura") == assinatura:
            melhoras, volta0, prontos, esd_antes = salvo["melhoras"], salvo["volta"], salvo["perfis"], salvo["esd"]
            centro, nll_centro = {**centro, "x": salvo["x"]}, salvo["nll"]
            out["retomada"] = dict(volta=volta0 + 1, perfis_prontos=len(prontos))
    for volta in range(volta0, 3):
        perfis = list(prontos) if volta == volta0 else []
        for i in range(len(perfis), p.d):
            passo(0.02 + 0.76 * i / p.d, f"perfil {i + 1} de {p.d}" + (f" (volta {volta + 1})" if volta else ""))
            perfis.append(perfilar(p, i, centro, nll_centro, sub, ll_sub, dp, cfg, limd))
            gravar_pickle(ck, dict(assinatura=assinatura, melhoras=melhoras, volta=volta, perfis=perfis,
                                   x=np.array(centro["x"], float), nll=nll_centro,
                                   esd=esd_antes + float(p.n_aval - n0)))
        k_melhor = int(np.argmin([float(f.min()) for _, f, _ in perfis]))
        f_melhor = float(perfis[k_melhor][1].min())
        if f_melhor >= nll_centro - 0.05 or volta == 2:
            break
        melhoras.append(float(nll_centro - f_melhor))
        x_novo = np.array(perfis[k_melhor][2][int(np.argmin(perfis[k_melhor][1]))], float)
        centro, nll_centro = {**centro, "x": x_novo}, f_melhor
    if melhoras:
        out["otimo_refeito"] = dict(voltas=len(melhoras), melhoras=melhoras,
                                    estavel=bool(f_melhor >= nll_centro - 0.05))

    # a simulacao falha perto do otimo? (explosao em tempo finito, integrador): perturba uma coordenada por vez
    xc, rng_v = np.array(centro["x"], float), np.random.default_rng(7)
    viz = np.tile(xc, (4 * p.d, 1))
    for k in range(len(viz)):
        viz[k, k % p.d] = np.clip(xc[k % p.d] + rng_v.normal() * 2e-3 * larg[k % p.d], p.lo[k % p.d], p.hi[k % p.d])
    out["vizinhanca"] = dict(n=len(viz), falhas=int(np.sum(~np.isfinite(p.logpost_lote(viz)))))

    # 2) estrutura local no centro final (barata; indica quem pode ser estruturalmente nao identificavel)
    passo(0.79, "posto e imagem invariante")
    Jp = Xp = None
    try:
        out["estrutura"], Jp, Xp = estrutura_local(p, centro, cfg, rng)
    except Exception as e:  # noqa: BLE001
        out["erros"].append(f"estrutura: {type(e).__name__}: {str(e)[:150]}")

    nll_min = min(nll_map, min(float(f.min()) for _, f, _ in perfis))
    participa = (out.get("estrutura") or {}).get("participa", [False] * p.d)
    params = []
    for i, (v, f, X) in enumerate(perfis):
        q = analisar_perfil(p, i, v, f, X, nll_min, limd)
        q.update(nome=p.nomes[i], contracao=float(contr[i]), natureza=None)
        if q["classe"] != "identificavel":
            q["natureza"] = "estrutural" if participa[i] else "pratica"
        params.append(q)
    # o ponto calibrado: o melhor encontrado, que pode ter saido de um perfil
    x_ot = np.array(centro["x"], float)
    k_melhor = int(np.argmin([float(f.min()) for _, f, _ in perfis]))
    if float(perfis[k_melhor][1].min()) < nll_centro - 1e-6:
        x_ot = np.array(perfis[k_melhor][2][int(np.argmin(perfis[k_melhor][1]))], float)
    # intervalos de confianca de Raue et al. (2009), secao 2.2: assintoticos (eq. 7) e pela verossimilhanca
    # (eq. 8), cada um pontual (1 grau de liberdade) e simultaneo (d graus de liberdade)
    coords = coordenadas(p)
    try:
        assint, out["curvatura"] = intervalos_assintoticos(p, x_ot, coords, LIM1, limd)
    except Exception as e:  # noqa: BLE001
        assint = None
        out["erros"].append(f"assintótico: {type(e).__name__}: {str(e)[:150]}")
    for i, (q, (v, f, X)) in enumerate(zip(params, perfis)):
        c, rot = coords[i]
        sim = _intervalos(v, f - nll_min, limd)
        a = assint[i] if assint else None
        if a and not a["infinito"]:
            a.update(pontual_theta=_de_u(c, a["pontual"]).tolist(), simultaneo_theta=_de_u(c, a["simultaneo"]).tolist())
        q.update(coord=c, rotulo_coord=rot, u=_para_u(c, v).tolist(), chi2=(2.0 * f).tolist(),
                 u_otimo=float(_para_u(c, x_ot[i])), assintotico=a,
                 ic_perfil=dict(pontual=q["ic_conjunto"], simultaneo=sim,
                                pontual_u=[_para_u(c, k).tolist() for k in q["ic_conjunto"]],
                                simultaneo_u=[_para_u(c, k).tolist() for k in sim],
                                simultaneo_chega_ao_limite=[bool(f[0] - nll_min <= limd), bool(f[-1] - nll_min <= limd)]))
    out.update(parametros=params, nll_min=nll_min, chi2_min=2.0 * nll_min,
               melhora_sobre_map=float(nll_map - nll_min), x_otimo=x_ot.tolist())

    # 3) subespaco informado pela verossimilhanca
    passo(0.82, "subespaço informado")
    try:
        if Jp is None:
            Xp = p.amostrar_prior(cfg["n_pts_estrutura"], rng)
            Jp = np.full((len(Xp), 1, p.d), np.nan)
        out["lis"] = subespaco_informado(p, amostras, Jp, Xp, cfg)
    except Exception as e:  # noqa: BLE001
        out["erros"].append(f"subespaço: {type(e).__name__}: {str(e)[:150]}")

    # 4) predicao perfil a perfil
    passo(0.92, "predição perfil a perfil")
    try:
        out["predicao"] = predicao_perfil(p, params, centro, sub, ll_sub, cfg, limd)
    except Exception as e:  # noqa: BLE001
        out["erros"].append(f"predição: {type(e).__name__}: {str(e)[:150]}")
    out["esd"] = esd_antes + float(p.n_aval - n0)
    gravar_json(arq, out)
    ck.unlink(missing_ok=True)
    return out
