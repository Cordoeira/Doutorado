"""Problemas de inferencia: modelos sinteticos (numba) e problemas PEtab (AMICI)."""
import time
import zlib
from pathlib import Path

import numpy as np

from .util import cov_de_hessiana


class Base:
    n_aval = 0

    def dentro(self, X):
        return np.all((X >= self.lo) & (X <= self.hi), axis=1)

    def amostrar_prior(self, n, rng):
        return self.lo + (self.hi - self.lo) * rng.random((n, self.d))

    def para_z(self, X):
        s = np.clip((X - self.lo) / (self.hi - self.lo), 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s))

    def de_z(self, Z):
        s = 1.0 / (1.0 + np.exp(-Z))
        return self.lo + (self.hi - self.lo) * s, s

    def alvo_z(self, Z):
        X, s = self.de_z(Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = self.logpost_lote(X) + np.sum(np.log(s) + np.log1p(-s), axis=1)
        lp[~np.isfinite(lp)] = -np.inf
        return lp

    def alvo_grad_z(self, Z):
        X, s = self.de_z(Z)
        ll, g = self.grad_lote(X)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = ll + np.sum(np.log(s) + np.log1p(-s), axis=1)
            gz = g * (self.hi - self.lo) * s * (1 - s) + (1 - 2 * s)
        ruim = ~np.isfinite(lp) | ~np.all(np.isfinite(gz), axis=1)
        lp[ruim], gz[ruim] = -np.inf, 0.0
        return lp, gz

    def nll(self, x):
        return -float(self.logpost_lote(np.asarray(x, float)[None])[0])


class Sintetico(Base):
    """Modelo escrito a mao (numba), prior uniforme, ruido gaussiano de desvio conhecido. Os dados sao sinteticos
    (simulados no valor verdadeiro com ruido de semente fixa) ou reais (spec["dados"]: medidas publicadas e o
    desvio calibrado por grupo de saidas, ver dados_reais/fontes.json). O desvio e um vetor, um valor por saida."""

    def __init__(self, spec):
        self.nome, self.rotulo, self.tipo = spec["nome"], spec["rotulo"], spec["tipo"]
        self.grupo, self.desafio = spec.get("grupo", "facil"), spec.get("desafio", "")
        self.lo, self.hi, self.sim, self.nomes = spec["lo"], spec["hi"], spec["sim"], list(spec["nomes"])
        self.d = len(self.lo)
        self.custo_grad = self.d + 1.0
        self.pred = spec.get("pred", self.sim)
        self.spec = spec
        dados = spec.get("dados")
        self.dados_reais = dados is not None
        self.fonte = dados["fonte"] if dados else ""
        self.verd = None if dados else np.array(spec["verd"], float)
        ponto = np.asarray(spec["centro"] if dados else spec["verd"], float)
        self.forma_obs = tuple(spec.get("forma_obs", (1, len(self.sim(ponto[None])[0]))))
        self.t_dados = np.asarray(spec.get("t_dados", np.zeros(self.forma_obs[0])), float)
        self.rotulos_obs = list(spec.get("rotulos_obs", [f"y_{j + 1}" for j in range(self.forma_obs[1])]))
        # coordenadas da reparametrizacao: log para taxas positivas, identidade para o que ja esta em log
        self.escalas = ["id" if (n.startswith("log") or lo <= 0) else "log" for n, lo in zip(self.nomes, self.lo)]
        self.coord_log = [n.startswith("log") or e == "log" for n, e in zip(self.nomes, self.escalas)]
        self.escala_petab = None
        self.tol_posto = 1e-6
        if dados:
            self.x_obs = np.asarray(dados["x_obs"], float)
            g = np.asarray(dados["grupo_sigma"], int)
            sg = dados.get("sigma")
            self.sigma = np.ones(len(self.x_obs)) if sg is None else np.asarray(sg, float)[g]
            if len(self.sigma) != len(self.x_obs) or len(self.sim(ponto[None])[0]) != len(self.x_obs):
                raise ValueError(f"{self.nome}: dados, sigma e saidas do simulador com tamanhos diferentes")
        else:
            y = self.sim(ponto[None])[0]
            self.sigma = np.full(y.shape, spec["ruido"] * float(np.std(y)))
            rng = np.random.default_rng(zlib.crc32(self.nome.encode()))
            self.x_obs = y + self.sigma * rng.standard_normal(y.shape)
        t = time.perf_counter()
        self.sim(np.tile(ponto, (200, 1)))
        self.t_f = (time.perf_counter() - t) / 200

    def logpost_lote(self, X):
        out = np.full(len(X), -np.inf)
        m = self.dentro(X)
        if m.any():
            self.n_aval += int(m.sum())
            ll = -0.5 * np.sum(((self.sim(X[m]) - self.x_obs) / self.sigma) ** 2, axis=1)
            ll[~np.isfinite(ll)] = -np.inf
            out[m] = ll
        return out

    def simular_lote(self, TH, rng):
        self.n_aval += len(TH)
        X = self.sim(TH)
        X = X + self.sigma * rng.standard_normal(X.shape)
        return X, np.all(np.isfinite(X), axis=1)

    def grad_lote(self, X):
        B, d = X.shape
        h = 1e-6 * (self.hi - self.lo)
        sg = np.where(X + h > self.hi, -1.0, 1.0)
        P = np.repeat(X[:, None, :], d + 1, axis=1)
        P[:, np.arange(1, d + 1), np.arange(d)] += sg * h
        self.n_aval += B * (d + 1)
        R = (self.sim(P.reshape(-1, d)).reshape(B, d + 1, -1) - self.x_obs) / self.sigma
        ll = -0.5 * np.sum(R[:, 0] ** 2, axis=1)
        g = -np.einsum("bn,bdn->bd", R[:, 0], R[:, 1:] - R[:, :1]) / (sg * h)
        ruim = ~np.isfinite(ll) | ~np.all(np.isfinite(g), axis=1)
        ll[ruim], g[ruim] = -np.inf, 0.0
        return ll, g

    def jac_lote(self, X):
        """Jacobiano das saidas branqueadas (dividido por sigma), por diferencas centrais: (B, n, d)."""
        B, d = X.shape
        h = 1e-5 * (self.hi - self.lo)
        k = np.arange(d)
        P = np.repeat(X[:, None, :], 2 * d, axis=1)
        P[:, k, k] += h
        P[:, d + k, k] -= h
        self.n_aval += 2 * B * d
        Y = self.sim(P.reshape(-1, d)).reshape(B, 2 * d, -1)
        return np.transpose((Y[:, :d] - Y[:, d:]) / (2 * h)[None, :, None], (0, 2, 1)) / self.sigma[None, :, None]

    def pred_lote(self, X):
        self.n_aval += len(X)
        return self.pred(X)

    # ---- solucoes para os graficos
    def solucao(self, X):
        """Saidas do modelo no formato dos dados: lista por condicao de (t, Y) com Y de forma (B, nt, m)."""
        self.n_aval += len(X)
        return [(self.t_dados, self.sim(np.atleast_2d(X)).reshape(-1, *self.forma_obs))]

    def dados(self):
        """Lista por condicao de (t, medidas, desvio do ruido), cada matriz de forma (nt, m)."""
        return [(self.t_dados, self.x_obs.reshape(self.forma_obs), self.sigma.reshape(self.forma_obs))]

    def estados(self, X):
        """Todos os estados: (t, Y) com Y de forma (B, nt, n). Nos modelos estacionarios, nt = 1."""
        X = np.atleast_2d(X)
        self.n_aval += len(X)
        if "traj" in self.spec:
            return self.spec["t_fino"], self.spec["traj"](X)
        return np.zeros(1), self.pred(X)[:, None, :]

    def _ajustar(self, x0, fixo=None):
        from scipy.optimize import least_squares

        livres = np.array([j for j in range(self.d) if j != fixo])
        h = 1e-6 * (self.hi - self.lo)

        def cheio(xl):
            x = np.array(x0, float)
            x[livres] = xl
            return x

        def res(xl):
            self.n_aval += 1
            return (self.sim(cheio(xl)[None])[0] - self.x_obs) / self.sigma

        def jac(xl):
            x = cheio(xl)
            sg = np.where(x + h > self.hi, -1.0, 1.0)
            P = np.vstack([x, x + np.diag(sg * h)[livres]])
            self.n_aval += len(P)
            Y = self.sim(P)
            return ((Y[1:] - Y[0]) / (sg * h)[livres, None]).T / self.sigma[:, None]

        if len(livres) == 0:
            return 0.5 * float(np.sum(res(np.empty(0)) ** 2)), np.array(x0, float), None
        r = least_squares(res, np.clip(np.asarray(x0, float)[livres], self.lo[livres], self.hi[livres]), jac=jac,
                          bounds=(self.lo[livres], self.hi[livres]), x_scale=(self.hi - self.lo)[livres],
                          max_nfev=200)
        return float(r.cost), cheio(r.x), r.jac

    def mapa(self, cfg):
        n0 = self.n_aval
        melhor = None
        inicios = list(self.amostrar_prior(cfg["n_inicios"], np.random.default_rng(0)))
        if self.dados_reais:        # o ajuste de calibracao (dados/fontes.json) tambem concorre, como o ponto
            inicios.append(np.clip(np.asarray(self.spec["centro"], float), self.lo, self.hi))   # nominal nos PEtab
        for x0 in inicios:
            r = self._ajustar(x0)
            if melhor is None or r[0] < melhor[0]:
                melhor = r
        fval, x, J = melhor
        fim = J.T @ J
        return dict(x=x, fval=fval, fim=fim, cov=cov_de_hessiana(fim, self.lo, self.hi),
                    esd=float(self.n_aval - n0))

    def perfil_ponto(self, i, valor, x0):
        x = np.array(x0, float)
        x[i] = valor
        fval, x, _ = self._ajustar(x, fixo=i)
        return fval, x


class Petab(Base):
    """Problema PEtab (AMICI) com prior uniforme numa caixa em torno do ajuste publicado."""

    tipo = "EDO (PEtab)"

    desafio = "dados reais"

    def __init__(self, nome, largura, pasta_modelos, grupo="petab"):
        import amici
        import benchmark_models_petab as bm
        import pypesto
        import pypesto.petab

        self.nome, self.rotulo = nome, nome.split("_")[0] + " (PEtab)"
        self.grupo = grupo
        self.dados_reais, self.fonte = True, f"Benchmark-Models-PEtab: {nome}"
        # as medidas vem de dados/<nome>/ (copia dos arquivos PEtab da colecao); sem a pasta, da propria colecao
        from .config import DIR_DADOS

        yaml_local = DIR_DADOS / nome / "problem.yaml"
        if yaml_local.exists():
            import petab

            pp = petab.Problem.from_yaml(str(yaml_local))
            self.fonte = f"dados/{nome} (Benchmark-Models-PEtab)"
        else:
            pp = bm.get_problem(nome)
        pasta = Path(pasta_modelos) / amici.__version__ / nome
        orig = pypesto.petab.PetabImporter(pp, output_folder=str(pasta), model_name=nome).create_problem(
            verbose=False)
        self.xn = np.asarray(pp.x_nominal_free_scaled, float)
        self.d = len(self.xn)
        self.nomes = list(pp.x_free_ids)
        escalas = pp.parameter_df.loc[list(pp.x_free_ids), "parameterScale"].to_numpy()
        meia = np.where(escalas == "lin", 0.5 * np.maximum(np.abs(self.xn), 1e-8), largura)
        self.lo = np.maximum(np.asarray(orig.lb, float), self.xn - meia)
        self.hi = np.minimum(np.asarray(orig.ub, float), self.xn + meia)
        self.livres = np.asarray(orig.x_free_indices, int)
        lb, ub = np.array(orig.lb_full, float), np.array(orig.ub_full, float)
        lb[self.livres], ub[self.livres] = self.lo, self.hi
        self.pypesto = pypesto.Problem(
            objective=orig.objective, lb=lb, ub=ub, x_fixed_indices=orig.x_fixed_indices,
            x_fixed_vals=orig.x_fixed_vals, x_names=orig.x_names, x_scales=orig.x_scales,
            x_guesses=[np.clip(self.xn, self.lo, self.hi)])
        self.obj = self.pypesto.objective

        ids = list(self.obj.amici_model.get_observable_ids())
        tr = pp.observable_df.get("observableTransformation")
        tr = {} if tr is None else tr.fillna("lin").to_dict()
        self.transf = [tr.get(i, "lin") for i in ids]
        self.rotulos_obs = list(ids)
        self.verd = None
        medidas = [np.asarray(ed.get_measurements(), float).reshape(-1, len(ids)) for ed in self.obj.edatas]
        self.mascara = [np.isfinite(y) for y in medidas]
        self.x_obs = self._vetor(medidas)

        self.n_recuos = 0
        self.t_f, self.c_s, self.grad_ok = self._custos()
        self.custo_grad = self.c_s
        self.escalas = ["log" if (e == "lin" and lo > 0) else "id" for e, lo in zip(escalas, self.lo)]
        self.coord_log = [e in ("log", "log10") or t == "log" for e, t in zip(escalas, self.escalas)]
        self.escala_petab = [str(e) for e in escalas]
        self.tol_posto = 1e-5
        self._obj_res = None

    ATOL_RECUO = (1e-12, 1e-10)     # tolerancias absolutas tentadas quando o gradiente falha com a do AMICI

    def _com_recuo(self, obj, avaliar, existe):
        """Sensibilidades com recuo de tolerancia. `avaliar()` devolve o resultado ou None se ele nao for finito.
        Se a solucao existe (`existe()`) mas as sensibilidades falham com a tolerancia padrao do AMICI (acontece
        no Fujita, ate no ponto nominal), repete com tolerancia absoluta mais frouxa. Devolve None se nada
        funciona."""
        try:
            r = avaliar()
        except Exception:  # noqa: BLE001
            r = None
        if r is not None:
            return r
        try:
            if not existe():                                 # a propria solucao falha: nao ha o que tentar
                return None
        except Exception:  # noqa: BLE001
            return None
        solver = obj.amici_solver
        padrao = solver.get_absolute_tolerance()
        try:
            for atol in self.ATOL_RECUO:
                solver.set_absolute_tolerance(atol)
                try:
                    r = avaliar()
                except Exception:  # noqa: BLE001
                    r = None
                if r is not None:
                    self.n_recuos += 1
                    return r
        finally:
            solver.set_absolute_tolerance(padrao)
        return None

    def _fg(self, x):
        """-log L e gradiente, ou (inf, None) quando nao saem."""
        def avaliar():
            f, g = self.obj(x, sensi_orders=(0, 1))
            return (float(f), np.asarray(g, float)) if np.isfinite(f) and np.all(np.isfinite(g)) else None
        r = self._com_recuo(self.obj, avaliar, lambda: np.isfinite(float(self.obj(x, sensi_orders=(0,)))))
        return r if r is not None else (np.inf, None)

    def _sres(self, o, x):
        """Jacobiano dos residuos (objetivo em modo de residuos), ou None quando nao sai."""
        def avaliar():
            s = np.asarray(o(x, sensi_orders=(1,), mode="mode_res"), float)
            return s if np.all(np.isfinite(s)) else None

        def existe():
            return bool(np.all(np.isfinite(np.asarray(o(x, sensi_orders=(0,), mode="mode_res"), float))))
        return self._com_recuo(o, avaliar, existe)

    def _custos(self, n_pontos=8):
        """Tempo de uma solucao (s), custo de um gradiente em solucoes e fracao dos pontos de teste em que o
        gradiente sai. Medianas no ponto nominal e em pontos sorteados do prior, so onde a solucao existe (a que
        falha termina cedo). Medir so no ponto nominal engana: no Fujita o gradiente falhava ali, parecia custar
        2,4 solucoes, e o orcamento de MALA e HMC, contado em solucoes, saia mais de dez vezes maior."""
        def menor_tempo(f, n=3):                             # o menor de n tempos resiste a carga da maquina
            tempos, r = [], None
            for _ in range(n):
                t = time.perf_counter()
                r = f()
                tempos.append(time.perf_counter() - t)
            return min(tempos), r

        x0 = np.clip(self.xn, self.lo, self.hi)
        self.obj(x0, sensi_orders=(0, 1))                    # a primeira chamada carrega o modelo
        tf, tg, tentativas = [], [], []
        for x in [x0, *self.amostrar_prior(n_pontos, np.random.default_rng(2024))]:
            try:
                if not np.isfinite(float(self.obj(x, sensi_orders=(0,)))):
                    continue
                a, _ = menor_tempo(lambda x=x: self.obj(x, sensi_orders=(0,)))
                b, (_, g) = menor_tempo(lambda x=x: self._fg(x))
            except Exception:  # noqa: BLE001
                continue
            tf.append(a)
            tentativas.append(b)
            if g is not None:
                tg.append(b)
        if not tf:
            raise RuntimeError("a simulação falha no ponto nominal e em todos os pontos de teste do prior")
        t_f = float(np.median(tf))
        return t_f, float(max(1.0, np.median(tg or tentativas) / t_f)), len(tg) / len(tf)

    def _transformar(self, y):
        y = np.array(y, float)
        for j, t in enumerate(self.transf):
            if t == "log10":
                y[:, j] = np.log10(np.maximum(y[:, j], 1e-300))
            elif t == "log":
                y[:, j] = np.log(np.maximum(y[:, j], 1e-300))
        return y

    def _vetor(self, ys):
        return np.concatenate([self._transformar(y)[m] for y, m in zip(ys, self.mascara)])

    def logpost_lote(self, X):
        out = np.full(len(X), -np.inf)
        for i in np.flatnonzero(self.dentro(X)):
            self.n_aval += 1
            try:
                v = -float(self.obj(X[i], sensi_orders=(0,)))
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(v):
                out[i] = v
        return out

    def grad_lote(self, X):
        ll, g = np.full(len(X), -np.inf), np.zeros_like(X)
        for i in range(len(X)):
            self.n_aval += self.c_s
            try:
                f, gr = self._fg(X[i])
            except Exception:  # noqa: BLE001
                continue
            if gr is not None:
                ll[i], g[i] = -f, -gr
        return ll, g

    def _residuos(self):
        """Objetivo em modo de residuos (com os residuos de sigma), ou None se o AMICI nao permitir."""
        if self._obj_res is None:
            import copy

            try:
                o = copy.deepcopy(self.obj)
                o.amici_model.set_add_sigma_residuals(True)
                o(self.xn, sensi_orders=(1,), mode="mode_res")
                self._obj_res = o
            except Exception:  # noqa: BLE001
                self._obj_res = False
        return self._obj_res or None

    def jac_lote(self, X):
        """Jacobiano dos residuos (saidas e sigmas branqueados) pelas sensibilidades do AMICI: (B, n, d)."""
        o = self._residuos()
        if o is None:
            raise RuntimeError("modo de resíduos indisponível neste problema")
        J = None
        for b in range(len(X)):
            self.n_aval += self.c_s
            s = self._sres(o, X[b])
            if s is None:
                continue
            if J is None:
                J = np.full((len(X),) + s.shape, np.nan)
            if s.shape == J.shape[1:]:
                J[b] = s
        if J is None:
            raise RuntimeError("nenhuma sensibilidade pôde ser calculada")
        return J

    # ---- solucoes para os graficos
    def _rdatas(self, x):
        try:
            return self.obj(np.asarray(x, float), sensi_orders=(0,), return_dict=True)["rdatas"]
        except Exception:  # noqa: BLE001
            return None

    def solucao(self, X):
        """Observaveis simulados, por condicao experimental: lista de (t, Y) com Y de forma (B, nt, ny)."""
        X = np.atleast_2d(X)
        ts = [np.asarray(ed.get_timepoints(), float) for ed in self.obj.edatas]
        ny = len(self.rotulos_obs)
        out = [np.full((len(X), len(t), ny), np.nan) for t in ts]
        for b, x in enumerate(X):
            self.n_aval += 1
            rds = self._rdatas(x)
            if rds is None or any(rd.status != 0 for rd in rds):
                continue
            for c, rd in enumerate(rds):
                out[c][b] = np.asarray(rd.y, float).reshape(len(ts[c]), ny)
        return list(zip(ts, out))

    def dados(self):
        rds = self._rdatas(self.xn)
        out = []
        for c, ed in enumerate(self.obj.edatas):
            t = np.asarray(ed.get_timepoints(), float)
            y = np.asarray(ed.get_measurements(), float).reshape(len(t), -1)
            try:
                sg = np.asarray(rds[c].sigmay, float).reshape(y.shape)
            except Exception:  # noqa: BLE001
                sg = np.full(y.shape, np.nan)
            out.append((t, y, sg))
        return out

    def estados(self, X):
        return None

    def pred_lote(self, X):
        """Estados do modelo em todos os instantes e condicoes (inclui os estados nao observados)."""
        out = None
        for b in range(len(X)):
            self.n_aval += 1
            try:
                r = self.obj(X[b], sensi_orders=(0,), return_dict=True)
                v = np.concatenate([np.asarray(rd.x, float).ravel() for rd in r["rdatas"]])
            except Exception:  # noqa: BLE001
                continue
            if out is None:
                out = np.full((len(X), len(v)), np.nan)
            if len(v) == out.shape[1]:
                out[b] = v
        return out

    def perfil_ponto(self, i, valor, x0):
        x = np.clip(np.array(x0, float), self.lo, self.hi)
        x[i] = valor
        return self._otimizar(x, fixo=i)

    def _otimizar(self, x0, fixo=None):
        """Minimiza -log L dentro da caixa a partir de x0, com o parametro `fixo` (se houver) parado."""
        from scipy.optimize import least_squares, minimize

        i = fixo
        livres = np.array([j for j in range(self.d) if j != i])
        x = np.array(x0, float)
        valor = None if i is None else x[i]
        o = self._residuos()
        if o is not None:
            def res(xl):
                x[livres] = xl
                self.n_aval += 1
                try:
                    r = np.asarray(o(x, sensi_orders=(0,), mode="mode_res"), float)
                except Exception:  # noqa: BLE001
                    return np.full(self._n_res, 1e6)
                self._n_res = len(r)
                return r if np.all(np.isfinite(r)) else np.full(len(r), 1e6)

            def jac(xl):
                x[livres] = xl
                self.n_aval += self.c_s
                s = self._sres(o, x)
                return np.zeros((self._n_res, len(livres))) if s is None else s[:, livres]

            try:
                self._n_res = len(res(x[livres]))
                r = least_squares(res, x[livres], jac=jac, bounds=(self.lo[livres], self.hi[livres]),
                                  x_scale="jac", max_nfev=60)
                x[livres] = r.x
                return self.nll(x), x.copy()
            except Exception:  # noqa: BLE001
                x = np.array(x0, float)
                if i is not None:
                    x[i] = valor

        def fun(xl):
            x[livres] = xl
            self.n_aval += self.c_s
            try:
                f, g = self._fg(x)
            except Exception:  # noqa: BLE001
                return 1e30, np.zeros(len(livres))
            if g is None:
                return 1e30, np.zeros(len(livres))
            return f, g[livres]

        r = minimize(fun, x[livres], jac=True, method="L-BFGS-B",
                     bounds=list(zip(self.lo[livres], self.hi[livres])), options=dict(maxiter=60))
        x[livres] = r.x
        return float(r.fun), x.copy()

    def simular_lote(self, TH, rng):
        X, ok = np.zeros((len(TH), len(self.x_obs))), np.zeros(len(TH), bool)
        for i, th in enumerate(TH):
            self.n_aval += 1
            try:
                r = self.obj(th, sensi_orders=(0,), return_dict=True)
            except Exception:  # noqa: BLE001
                continue
            if not np.isfinite(r["fval"]):
                continue
            ys = []
            for rd in r["rdatas"]:
                y = self._transformar(np.asarray(rd.y, float))
                ys.append(y + np.asarray(rd.sigmay, float) * rng.standard_normal(y.shape))
            v = np.concatenate([y[m] for y, m in zip(ys, self.mascara)])
            if np.all(np.isfinite(v)):
                X[i], ok[i] = v, True
        return X, ok

    def mapa(self, cfg):
        import logging

        import pypesto.optimize as opt

        np.random.seed(0)
        res = opt.minimize(self.pypesto, optimizer=opt.FidesOptimizer(verbose=logging.ERROR),
                           n_starts=cfg["n_inicios"], progress_bar=False)
        lista = [r for r in res.optimize_result.list if r["fval"] is not None and np.isfinite(r["fval"])]
        melhor = min(lista, key=lambda r: r["fval"]) if lista else None
        x = np.clip(np.asarray(melhor["x"], float)[self.livres], self.lo, self.hi) if melhor else None
        f = self.nll(x) if melhor else np.inf
        origem, n0 = "otimizacao", self.n_aval
        # o ponto nominal da colecao (o ajuste publicado, centro da caixa do prior) tambem concorre: a otimizacao
        # a partir dele pode falhar logo no primeiro gradiente (Fujita) e sobrar um otimo local muito pior
        xn = np.clip(self.xn, self.lo, self.hi)
        fn = self.nll(xn)
        if np.isfinite(fn) and fn < f - 0.01:
            x, f, origem = xn, fn, "nominal"
            try:
                f2, x2 = self._otimizar(xn)
                if np.isfinite(f2) and f2 < f:
                    x, f, origem = x2, f2, "nominal, polido"
            except Exception:  # noqa: BLE001
                pass
        if x is None:
            raise RuntimeError("a otimização não achou ponto com verossimilhança finita, nem no ponto nominal")
        def fisher(z):
            h = np.asarray(self.obj(z, sensi_orders=(2,)), float)
            return h if np.all(np.isfinite(h)) else None
        # a Fisher da a covariancia inicial dos amostradores. Se as sensibilidades falham bem no otimo, serve a
        # de um ponto vizinho (a 0,1% do prior); em ultimo caso, uma covariancia larga, de 1/10 do prior
        rng_h = np.random.default_rng(31)
        vizinhos = [x] + [np.clip(x + 1e-3 * (self.hi - self.lo) * rng_h.standard_normal(self.d), self.lo, self.hi)
                          for _ in range(5)]
        h = None
        for k, z in enumerate(vizinhos):
            h = self._com_recuo(self.obj, lambda z=z: fisher(z), lambda z=z: np.isfinite(self.nll(z)))
            if h is not None:
                origem += ", Fisher de um ponto vizinho" if k else ""
                break
        if h is None:
            h = np.diag(100.0 / np.maximum(self.hi - self.lo, 1e-12) ** 2)
            origem += ", sem Fisher"
        n_f = sum(int(r["n_fval"] or 0) for r in res.optimize_result.list)
        n_g = sum(int(r["n_grad"] or 0) for r in res.optimize_result.list)
        return dict(x=x, fval=float(f), fim=h, cov=cov_de_hessiana(h, self.lo, self.hi), origem=origem,
                    esd=float(n_f + self.c_s * n_g + self.c_s * self.d + (self.n_aval - n0)))


def construir(origem, nome, grupo, cfg, pasta_modelos):
    from . import modelos

    if origem == "petab":
        return Petab(nome, cfg["largura"], pasta_modelos, grupo)
    return Sintetico(modelos.POR_NOME[nome])
