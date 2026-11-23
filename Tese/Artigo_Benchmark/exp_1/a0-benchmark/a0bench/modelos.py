"""Modelos do benchmark: 33 sinteticos e 14 com dados reais, de EDO e EDP (metodo das linhas), compilados pelo
numba.

Cada modelo e um dicionario em MODELOS com: nome, rotulo, tipo (EDO ou EDP), grupo (facil ou dificil), nomes
dos parametros, limites do prior uniforme (lo, hi), valor verdadeiro (verd), ruido relativo e os simuladores:
  sim(TH)   saidas observadas, uma linha por vetor de parametros (e o que entra na verossimilhanca)
  pred(TH)  predicoes fora da janela de dados (usadas na identificabilidade)
  traj(TH)  todos os estados numa malha fina de tempo (usado nos graficos de solucao)
Para acrescentar um modelo: escreva f(y, th, dy) e a condicao inicial com @njit e registre com _m(...).
Os problemas PEtab (dados reais) ficam em config.PETAB e sao montados em problemas.py. No fim deste arquivo,
reais() registra os modelos com dados reais publicados (dados/<modelo>/ na raiz; fontes em dados/fontes.json).
"""
import numpy as np
from numba import njit

PI = np.pi
N = 32
DX_D, DX_P = 1.0 / (N + 1), 1.0 / N
X_D = (np.arange(N) + 1.0) * DX_D
X_P = np.arange(N) * DX_P
X_N = (np.arange(N) + 0.5) * DX_P
NB = 64
DX_B = 1.0 / NB
X_B = np.arange(NB) * DX_B
MD = 64
DX_K = 1.0 / MD


@njit
def _lote(f, ic, TH, tobs, nsub, idx, n):
    B, nt, m = TH.shape[0], tobs.shape[0], idx.shape[0]
    out = np.empty((B, nt * m))
    y, yt = np.empty(n), np.empty(n)
    k1, k2, k3, k4 = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    for b in range(B):
        th = TH[b]
        ic(th, y)
        t = 0.0
        for i in range(nt):
            h = (tobs[i] - t) / nsub
            for _ in range(nsub):
                f(y, th, k1)
                for j in range(n):
                    yt[j] = y[j] + 0.5 * h * k1[j]
                f(yt, th, k2)
                for j in range(n):
                    yt[j] = y[j] + 0.5 * h * k2[j]
                f(yt, th, k3)
                for j in range(n):
                    yt[j] = y[j] + h * k3[j]
                f(yt, th, k4)
                for j in range(n):
                    y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
            t = tobs[i]
            for j in range(m):
                out[b, i * m + j] = y[idx[j]]
    return out


@njit
def _resolver(A, b, n):
    for c in range(n):
        p = c
        for r in range(c + 1, n):
            if abs(A[r, c]) > abs(A[p, c]):
                p = r
        if p != c:
            for k in range(n):
                A[c, k], A[p, k] = A[p, k], A[c, k]
            b[c], b[p] = b[p], b[c]
        for r in range(c + 1, n):
            if A[c, c] == 0.0:
                A[c, c] = 1e-300
            w = A[r, c] / A[c, c]
            for k in range(c, n):
                A[r, k] -= w * A[c, k]
            b[r] -= w * b[c]
    for r in range(n - 1, -1, -1):
        v = b[r]
        for k in range(r + 1, n):
            v -= A[r, k] * b[k]
        b[r] = v / (A[r, r] if A[r, r] != 0.0 else 1e-300)


@njit
def _lote_ros2(f, ic, TH, tobs, nsub, idx, n):
    """Rosenbrock de 2a ordem, L-estavel (ROS2), com jacobiano por diferencas finitas."""
    B, nt, m = TH.shape[0], tobs.shape[0], idx.shape[0]
    g = 1.0 + 1.0 / np.sqrt(2.0)
    out = np.empty((B, nt * m))
    y, yt, f0, f1 = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    k1, k2 = np.empty(n), np.empty(n)
    J, A = np.empty((n, n)), np.empty((n, n))
    for b in range(B):
        th = TH[b]
        ic(th, y)
        t = 0.0
        for i in range(nt):
            h = (tobs[i] - t) / nsub
            for _ in range(nsub):
                f(y, th, f0)
                for j in range(n):
                    yt[:] = y
                    dj = 1e-7 * abs(y[j]) + 1e-12
                    yt[j] += dj
                    f(yt, th, f1)
                    for r in range(n):
                        J[r, j] = (f1[r] - f0[r]) / dj
                for r in range(n):
                    for c in range(n):
                        A[r, c] = -g * h * J[r, c]
                    A[r, r] += 1.0
                    k1[r] = f0[r]
                _resolver(A, k1, n)
                for r in range(n):
                    yt[r] = y[r] + h * k1[r]
                f(yt, th, f1)
                for r in range(n):
                    for c in range(n):
                        A[r, c] = -g * h * J[r, c]
                    A[r, r] += 1.0
                    k2[r] = f1[r] - 2.0 * k1[r]
                _resolver(A, k2, n)
                for r in range(n):
                    y[r] += 1.5 * h * k1[r] + 0.5 * h * k2[r]
            t = tobs[i]
            for j in range(m):
                out[b, i * m + j] = y[idx[j]]
    return out


def _ic_fixa(*v):
    v = np.array(v, float)

    @njit
    def ic(th, y):
        for j in range(v.shape[0]):
            y[j] = v[j]
    return ic


# ------------------------------------------------------------------ EDOs
@njit
def f_lv(y, th, dy):
    dy[0] = th[0] * y[0] - th[1] * y[0] * y[1]
    dy[1] = -th[2] * y[1] + th[3] * y[0] * y[1]


@njit
def f_seir(y, th, dy):
    inf = th[0] * y[0] * y[2]
    dy[0] = -inf
    dy[1] = inf - th[1] * y[1]
    dy[2] = th[1] * y[1] - th[2] * y[2]
    dy[3] = th[2] * y[2]


@njit
def f_fhn(y, th, dy):
    dy[0] = th[2] * (y[0] - y[0] ** 3 / 3.0 + y[1])
    dy[1] = -(y[0] - th[0] + th[1] * y[1]) / th[2]


@njit
def f_goodwin(y, th, dy):
    dy[0] = th[0] / (1.0 + y[2] ** 4) - th[1] * y[0]
    dy[1] = y[0] - th[2] * y[1]
    dy[2] = y[1] - th[3] * y[2]


@njit
def f_osc(y, th, dy):
    dy[0] = y[1]
    dy[1] = -2.0 * th[1] * th[0] * y[1] - th[0] ** 2 * y[0] + th[2] * np.cos(1.5 * y[2])
    dy[2] = 1.0


@njit
def f_lorenz(y, th, dy):
    dy[0] = th[0] * (y[1] - y[0])
    dy[1] = y[0] * (th[1] - y[2]) - y[1]
    dy[2] = y[0] * y[1] - th[2] * y[2]


@njit
def f_vdp(y, th, dy):
    dy[0] = y[1]
    dy[1] = th[0] * (1.0 - y[0] ** 2) * y[1] - th[1] ** 2 * y[0]


@njit
def f_pendulo(y, th, dy):
    dy[0] = y[1]
    dy[1] = -th[1] * y[1] - th[0] * np.sin(y[0])


@njit
def f_rm(y, th, dy):
    pred = th[2] * y[0] * y[1] / (1.0 + th[2] * th[3] * y[0])
    dy[0] = th[0] * y[0] * (1.0 - y[0] / th[1]) - pred
    dy[1] = 0.5 * pred - th[4] * y[1]


@njit
def f_pk2(y, th, dy):
    dy[0] = -th[0] * y[0]
    dy[1] = th[0] * y[0] - (th[1] + th[2]) * y[1] + th[3] * y[2]
    dy[2] = th[2] * y[1] - th[3] * y[2]


@njit
def f_brus(y, th, dy):
    dy[0] = th[0] - (th[1] + 1.0) * y[0] + y[0] ** 2 * y[1]
    dy[1] = th[1] * y[0] - y[0] ** 2 * y[1]


# ------------------------------------------------------------------ EDPs
@njit
def f_calor(y, th, dy):
    for j in range(N):
        e = y[j - 1] if j > 0 else 0.0
        d = y[j + 1] if j < N - 1 else 0.0
        dy[j] = th[0] * (e - 2.0 * y[j] + d) / DX_D ** 2 + th[1] * np.sin(PI * X_D[j])


@njit
def ic_calor(th, y):
    for j in range(N):
        y[j] = np.exp(-50.0 * (X_D[j] - 0.3) ** 2)


@njit
def f_fisher(y, th, dy):
    for j in range(N):
        e = y[j - 1] if j > 0 else y[0]
        d = y[j + 1] if j < N - 1 else y[N - 1]
        dy[j] = th[0] * (e - 2.0 * y[j] + d) / DX_P ** 2 + th[1] * y[j] * (1.0 - y[j])


@njit
def ic_fisher(th, y):
    for j in range(N):
        y[j] = 1.0 / (1.0 + np.exp((X_N[j] - 0.15) / 0.03))


@njit
def f_advdif(y, th, dy):
    for j in range(N):
        e, d = y[j - 1], y[(j + 1) % N]
        dy[j] = -th[0] * (d - e) / (2.0 * DX_P) + th[1] * (e - 2.0 * y[j] + d) / DX_P ** 2


@njit
def ic_gauss_p(th, y):
    for j in range(N):
        y[j] = np.exp(-100.0 * (X_P[j] - 0.3) ** 2)


@njit
def f_burgers(y, th, dy):
    for j in range(NB):
        e, d = y[j - 1], y[(j + 1) % NB]
        dy[j] = -y[j] * (d - e) / (2.0 * DX_B) + th[0] * (e - 2.0 * y[j] + d) / DX_B ** 2


@njit
def ic_burgers(th, y):
    for j in range(NB):
        y[j] = th[1] * np.sin(2.0 * PI * X_B[j])


@njit
def f_allen(y, th, dy):
    for j in range(N):
        e, d = y[j - 1], y[(j + 1) % N]
        dy[j] = th[0] * (e - 2.0 * y[j] + d) / DX_P ** 2 + th[1] * (y[j] - y[j] ** 3)


@njit
def ic_allen(th, y):
    for j in range(N):
        y[j] = 0.3 * np.cos(2.0 * PI * X_P[j]) + 0.2 * np.sin(6.0 * PI * X_P[j])


@njit
def f_onda(y, th, dy):
    for j in range(N):
        e = y[j - 1] if j > 0 else 0.0
        d = y[j + 1] if j < N - 1 else 0.0
        dy[j] = y[N + j]
        dy[N + j] = th[0] ** 2 * (e - 2.0 * y[j] + d) / DX_D ** 2 - th[1] * y[N + j]


@njit
def ic_onda(th, y):
    for j in range(N):
        y[j] = np.exp(-200.0 * (X_D[j] - 0.3) ** 2)
        y[N + j] = 0.0


@njit
def _darcy(TH, idx):
    B, d, m = TH.shape[0], TH.shape[1], idx.shape[0]
    out = np.empty((B, m))
    k = np.empty(MD)
    a, bb, c, r = np.empty(MD - 1), np.empty(MD - 1), np.empty(MD - 1), np.empty(MD - 1)
    for b in range(B):
        for i in range(MD):
            xm, s = (i + 0.5) * DX_K, 0.0
            for j in range(d):
                s += TH[b, j] * np.cos((j + 1) * PI * xm) / (j + 1)
            k[i] = np.exp(s)
        for i in range(MD - 1):
            a[i], c[i] = -k[i], -k[i + 1]
            bb[i] = k[i] + k[i + 1]
            r[i] = DX_K ** 2
        for i in range(1, MD - 1):
            w = a[i] / bb[i - 1]
            bb[i] -= w * c[i - 1]
            r[i] -= w * r[i - 1]
        r[MD - 2] /= bb[MD - 2]
        for i in range(MD - 3, -1, -1):
            r[i] = (r[i] - c[i] * r[i + 1]) / bb[i]
        for j in range(m):
            out[b, j] = r[idx[j]]
    return out


# ------------------------------------------------------------------ registro
def _m(nome, rotulo, tipo, f, ic, n, nomes, lo, hi, verd, T, nt, dt, idx, ruido=0.1, grupo="facil",
       desafio="", metodo="rk4", tobs=None, pos=None, t_dados=None, campos=1):
    """Registra um modelo: f e o lado direito, ic a condicao inicial, n o numero de estados, idx os estados
    observados nos nt instantes ate T. Nas EDPs, `campos` e o numero de campos empilhados no vetor de estado.
    Se `pos` transforma ou seleciona as saidas, `t_dados` da os instantes das linhas que sobram."""
    tobs = np.linspace(T / nt, T, nt) if tobs is None else np.asarray(tobs, float)
    nsub = max(1, int(np.ceil((T / nt) / dt))) if metodo == "rk4" else int(dt)
    idx = np.array(idx, np.int64)
    nucleo = _lote if metodo == "rk4" else _lote_ros2

    def sim(TH):
        Y = nucleo(f, ic, np.ascontiguousarray(TH, float), tobs, nsub, idx, n)
        return Y if pos is None else pos(Y)

    # predicao: todos os estados (ou os pontos observados, nas EDPs) ate 1,5 T, sem transformacao
    if metodo == "rk4":
        t_pred = np.linspace(1.5 * T / 40, 1.5 * T, 40)
        nsub_pred = max(1, int(np.ceil((1.5 * T / 40) / dt)))
        t_fino = np.linspace(T / 80, T, 80)
        nsub_fino = max(1, int(np.ceil((T / 80) / dt)))
    else:
        t_pred, nsub_pred = tobs, nsub
        t_fino, nsub_fino = tobs, nsub
    idx_pred = np.arange(n, dtype=np.int64) if n <= 8 else idx
    todos = np.arange(n, dtype=np.int64)

    def pred(TH):
        return nucleo(f, ic, np.ascontiguousarray(TH, float), t_pred, nsub_pred, idx_pred, n)

    def traj(TH):
        """Todos os estados numa malha fina de tempo: (B, len(t_fino), n)."""
        TH = np.ascontiguousarray(TH, float)
        return nucleo(f, ic, TH, t_fino, nsub_fino, todos, n).reshape(len(TH), len(t_fino), n)
    t_dados = tobs if t_dados is None else np.asarray(t_dados, float)
    pts = n // campos
    if tipo == "EDP":
        x = np.arange(pts) / pts
        rot_obs = [f"x = {x[j % pts]:.2f}" for j in idx]
    else:
        x = None
        rot_obs = [f"y_{j + 1}" for j in idx]
    return dict(nome=nome, rotulo=rotulo, tipo=tipo, nomes=nomes, lo=np.array(lo, float),
                hi=np.array(hi, float), verd=np.array(verd, float), sim=sim, pred=pred, ruido=ruido, grupo=grupo,
                desafio=desafio, traj=traj, t_fino=t_fino, t_dados=t_dados, forma_obs=(len(t_dados), len(idx)),
                idx=idx, n_estados=n, campos=campos, x=x, rotulos_obs=rot_obs, metodo=metodo)


_PTS = list(range(2, N, 4))
_PTS_B = list(range(4, NB, 8))
_IDX_K = np.arange(3, MD - 1, 4)
_IDX_TODOS = np.arange(MD - 1)

MODELOS = [
    _m("lotka_volterra", "Lotka–Volterra", "EDO", f_lv, _ic_fixa(2, 1), 2,
       ["a", "b", "c", "d"], [0.5, 0.2, 0.5, 0.1], [1.5, 0.8, 1.5, 0.5], [1.0, 0.5, 1.0, 0.3],
       15, 20, 0.02, [0, 1]),
    _m("seir", "SEIR", "EDO", f_seir, _ic_fixa(0.99, 0.01, 0, 0), 4,
       ["beta", "sigma", "gamma"], [0.3, 0.1, 0.1], [1.5, 0.6, 0.5], [0.9, 0.3, 0.25],
       60, 25, 0.1, [2]),
    _m("fitzhugh_nagumo", "FitzHugh–Nagumo", "EDO", f_fhn, _ic_fixa(-1, 1), 2,
       ["a", "b", "c"], [0.0, 0.0, 1.0], [1.0, 1.0, 5.0], [0.2, 0.2, 3.0],
       20, 40, 0.01, [0, 1]),
    _m("goodwin", "Goodwin", "EDO", f_goodwin, _ic_fixa(0.5, 0.5, 0.5), 3,
       ["a", "b", "c", "d"], [1.0, 0.2, 0.2, 0.2], [3.0, 1.0, 1.0, 1.0], [2.0, 0.5, 0.5, 0.5],
       30, 30, 0.02, [0, 2]),
    _m("oscilador", "Oscilador forçado", "EDO", f_osc, _ic_fixa(1, 0, 0), 3,
       ["omega", "zeta", "A"], [1.0, 0.02, 0.2], [3.0, 0.5, 2.0], [2.0, 0.15, 1.0],
       15, 30, 0.01, [0]),
    _m("lorenz63", "Lorenz-63", "EDO", f_lorenz, _ic_fixa(-5, -6, 22), 3,
       ["sigma", "rho", "beta"], [5.0, 20.0, 1.5], [15.0, 36.0, 4.0], [10.0, 28.0, 8.0 / 3.0],
       1.5, 15, 0.002, [0, 1, 2]),
    _m("van_der_pol", "Van der Pol", "EDO", f_vdp, _ic_fixa(2, 0), 2,
       ["mu", "omega"], [0.2, 0.5], [3.0, 1.5], [1.5, 1.0],
       20, 30, 0.01, [0]),
    _m("pendulo", "Pêndulo amortecido", "EDO", f_pendulo, _ic_fixa(2.5, 0), 2,
       ["k", "gamma"], [1.0, 0.02], [8.0, 0.6], [4.0, 0.2],
       15, 30, 0.01, [0]),
    _m("rosenzweig_macarthur", "Rosenzweig–MacArthur", "EDO", f_rm, _ic_fixa(1, 0.5), 2,
       ["r", "K", "a", "h", "m"], [0.5, 1.5, 0.8, 0.2, 0.2], [1.5, 5.0, 2.5, 1.0, 0.7],
       [1.0, 3.0, 1.5, 0.5, 0.4], 40, 30, 0.02, [0, 1]),
    _m("pk2", "PK de dois compartimentos", "EDO", f_pk2, _ic_fixa(1, 0, 0), 3,
       ["k_a", "k_10", "k_12", "k_21"], [0.5, 0.1, 0.1, 0.05], [3.0, 0.8, 1.2, 0.8],
       [1.5, 0.3, 0.5, 0.25], 24, 20, 0.02, [1]),
    _m("brusselator", "Brusselator", "EDO", f_brus, _ic_fixa(1, 1), 2,
       ["a", "b"], [0.5, 1.5], [1.5, 3.5], [1.0, 2.5],
       20, 30, 0.005, [0, 1]),
    _m("calor", "Calor com fonte", "EDP", f_calor, ic_calor, N,
       ["D", "s"], [0.005, 0.0], [0.05, 2.0], [0.02, 1.0],
       1.0, 5, 0.004, _PTS),
    _m("fisher_kpp", "Fisher–KPP", "EDP", f_fisher, ic_fisher, N,
       ["D", "r"], [0.0005, 2.0], [0.005, 10.0], [0.002, 6.0],
       1.5, 5, 0.01, _PTS),
    _m("adveccao_difusao", "Advecção–difusão", "EDP", f_advdif, ic_gauss_p, N,
       ["c", "D"], [0.1, 0.001], [1.0, 0.02], [0.5, 0.005],
       1.0, 5, 0.01, _PTS),
    _m("burgers", "Burgers viscoso", "EDP", f_burgers, ic_burgers, NB,
       ["nu", "A"], [0.015, 0.5], [0.05, 1.5], [0.02, 1.0],
       0.5, 5, 0.002, _PTS_B),
    _m("allen_cahn", "Allen–Cahn", "EDP", f_allen, ic_allen, N,
       ["kappa", "r"], [0.0002, 2.0], [0.005, 8.0], [0.001, 5.0],
       1.0, 5, 0.01, _PTS),
    _m("onda", "Onda amortecida", "EDP", f_onda, ic_onda, 2 * N,
       ["c", "gamma"], [0.5, 0.0], [1.5, 2.0], [1.0, 0.5],
       2.0, 5, 0.008, _PTS, campos=2),
    dict(nome="darcy", rotulo="Darcy (difusão estacionária)", tipo="EDP",
         nomes=[f"theta_{j}" for j in range(1, 7)], lo=-np.ones(6), hi=np.ones(6),
         verd=np.array([0.6, -0.4, 0.5, -0.3, 0.4, -0.2]), ruido=0.05, grupo="facil", desafio="",
         sim=lambda TH: _darcy(np.ascontiguousarray(TH, float), _IDX_K),
         pred=lambda TH: _darcy(np.ascontiguousarray(TH, float), _IDX_TODOS),
         estatico=True, idx=_IDX_K, x=(_IDX_TODOS + 1.0) * DX_K, n_estados=MD - 1, campos=1,
         forma_obs=(1, len(_IDX_K)), t_dados=np.zeros(1),
         rotulos_obs=[f"x = {(j + 1) * DX_K:.2f}" for j in _IDX_K]),
]


# ================================================================== modelos dificeis
def _log10(Y):
    return np.log10(np.maximum(Y, 1e-12))


@njit
def f_robertson(y, th, dy):
    k1, k2, k3 = 10.0 ** th[0], 10.0 ** th[1], 10.0 ** th[2]
    dy[0] = -k1 * y[0] + k3 * y[1] * y[2]
    dy[1] = k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] ** 2
    dy[2] = k2 * y[1] ** 2


@njit
def f_hires(y, th, dy):
    b, c, e, k = 10.0 ** th[0], 10.0 ** th[1], 10.0 ** th[2], 10.0 ** th[3]
    dy[0] = -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007
    dy[1] = 1.71 * y[0] - b * y[1]
    dy[2] = -c * y[2] + 0.43 * y[3] + 0.035 * y[4]
    dy[3] = 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3]
    dy[4] = -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6]
    dy[5] = -e * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6]
    dy[6] = e * y[5] * y[7] - k * y[6]
    dy[7] = -e * y[5] * y[7] + k * y[6]


NL = 8


@njit
def f_lorenz96(y, th, dy):
    for i in range(NL):
        dy[i] = (y[(i + 1) % NL] - y[i - 2]) * y[i - 1] - th[1] * y[i] + th[0]


@njit
def ic_lorenz96(th, y):
    for i in range(NL):
        y[i] = 8.0 + np.sin(1.7 * i)


@njit
def f_cadeia(y, th, dy):
    dy[0] = -th[0] * y[0]
    dy[1] = th[0] * y[0] - th[1] * y[1]
    dy[2] = th[1] * y[1] - th[2] * y[2]
    dy[3] = th[2] * y[2]


@njit
def f_flipflop(y, th, dy):
    ka, ke = 10.0 ** th[0], 10.0 ** th[1]
    dy[0] = -ka * y[0]
    dy[1] = ka * y[0] / 10.0 ** th[2] - ke * y[1]


@njit
def f_produto(y, th, dy):
    dy[0] = -th[2] * y[0]
    dy[1] = 0.0


@njit
def ic_produto(th, y):
    y[0] = 10.0 ** (th[0] + th[1])
    y[1] = 0.0


@njit
def f_goodwin_hill(y, th, dy):
    dy[0] = th[0] / (1.0 + max(y[2], 0.0) ** th[4]) - th[1] * y[0]
    dy[1] = y[0] - th[2] * y[1]
    dy[2] = y[1] - th[3] * y[2]


@njit
def f_repressilator(y, th, dy):
    for i in range(3):
        j = (i + 2) % 3
        dy[i] = -y[i] + th[0] / (1.0 + max(y[3 + j], 0.0) ** th[3]) + th[1]
        dy[3 + i] = -th[2] * (y[3 + i] - y[i])


@njit
def f_calor_inv(y, th, dy):
    for j in range(N):
        e = y[j - 1] if j > 0 else 0.0
        d = y[j + 1] if j < N - 1 else 0.0
        dy[j] = 0.01 * (e - 2.0 * y[j] + d) / DX_D ** 2


@njit
def ic_calor_inv(th, y):
    for j in range(N):
        v = 0.0
        for k in range(th.shape[0]):
            v += th[k] * np.sin((k + 1) * PI * X_D[j])
        y[j] = v


@njit
def ic_fisher_frente(th, y):
    for j in range(N):
        y[j] = 1.0 / (1.0 + np.exp((X_N[j] - th[2]) / 0.03))


@njit
def ic_allen_meta(th, y):
    for j in range(N):
        y[j] = th[2] * (0.3 * np.cos(2.0 * PI * X_P[j]) + 0.2 * np.sin(6.0 * PI * X_P[j]))


@njit
def f_fhn_rd(y, th, dy):
    for j in range(N):
        e = y[j - 1] if j > 0 else y[0]
        d = y[j + 1] if j < N - 1 else y[N - 1]
        u, v = y[j], y[N + j]
        dy[j] = th[0] * (e - 2.0 * u + d) / DX_P ** 2 + th[3] * (u - u ** 3 / 3.0 + v)
        dy[N + j] = -(u - th[1] + th[2] * v) / th[3]


@njit
def ic_fhn_rd(th, y):
    for j in range(N):
        y[j] = -1.2 + 2.4 * np.exp(-((X_N[j] - 0.1) / 0.06) ** 2)
        y[N + j] = 0.6


_T_ROB = np.logspace(-5, 4, 91)
_SEL_ROB = np.arange(42, 91, 2)
_HV = np.log10(np.array([8.75, 10.03, 280.0, 1.81]))
_T_HIRES = np.logspace(-3, np.log10(320.0), 45)
_SEL_HIRES = np.arange(5, 45, 2)
_IDX_K16 = np.arange(7, MD - 1, 8)

DIFICEIS = [
    _m("robertson", "Robertson", "EDO", f_robertson, _ic_fixa(1, 0, 0), 3,
       ["log k_1", "log k_2", "log k_3"], [-2.4, 6.5, 3.0], [-0.4, 8.5, 5.0],
       [np.log10(0.04), np.log10(3e7), 4.0], 1e4, 91, 4, [0, 1, 2], grupo="dificil", desafio="rígido",
       metodo="ros2", tobs=_T_ROB,
       pos=lambda Y: (Y.reshape(len(Y), 91, 3)[:, _SEL_ROB] * np.array([1.0, 2e4, 1.0])).reshape(len(Y), -1),
       t_dados=_T_ROB[_SEL_ROB]),
    _m("hires", "HIRES", "EDO", f_hires, _ic_fixa(1, 0, 0, 0, 0, 0, 0, 0.0057), 8,
       ["log k_2", "log k_3", "log k_5", "log k_7"], _HV - np.array([0.1, 0.1, 1.0, 1.0]), _HV + 1.0, _HV,
       320.0, 45, 4, [0, 4, 5, 7], grupo="dificil", desafio="rígido", metodo="ros2", tobs=_T_HIRES,
       pos=lambda Y: Y.reshape(len(Y), 45, 4)[:, _SEL_HIRES].reshape(len(Y), -1),
       t_dados=_T_HIRES[_SEL_HIRES]),
    _m("lorenz63_longo", "Lorenz-63 (janela longa)", "EDO", f_lorenz, _ic_fixa(-5, -6, 22), 3,
       ["sigma", "rho", "beta"], [5.0, 20.0, 1.5], [15.0, 36.0, 4.0], [10.0, 28.0, 8.0 / 3.0],
       6.0, 40, 0.002, [0, 1, 2], grupo="dificil", desafio="caótico"),
    _m("lorenz96", "Lorenz-96", "EDO", f_lorenz96, ic_lorenz96, NL,
       ["F", "a"], [4.0, 0.5], [12.0, 1.5], [8.0, 1.0],
       3.0, 15, 0.005, list(range(0, NL, 2)), grupo="dificil", desafio="caótico"),
    _m("cadeia_simetrica", "Cadeia de decaimento", "EDO", f_cadeia, _ic_fixa(1, 0, 0, 0), 4,
       ["k_1", "k_2", "k_3"], [0.1, 0.1, 0.1], [3.0, 3.0, 3.0], [0.4, 1.0, 2.2],
       12.0, 25, 0.01, [3], ruido=0.03, grupo="dificil", desafio="multimodal (6 modos por simetria)"),
    _m("pk_flipflop", "PK com flip-flop", "EDO", f_flipflop, _ic_fixa(1, 0), 2,
       ["log k_a", "log k_e", "log V"], [-1.5, -1.5, -2.0], [0.8, 0.8, 1.0],
       [np.log10(1.5), np.log10(0.15), 0.0], 24.0, 16, 0.04, [1], ruido=0.05, grupo="dificil",
       desafio="multimodal (2 modos)"),
    _m("produto", "Decaimento com escala", "EDO", f_produto, ic_produto, 2,
       ["log A", "log s", "k"], [-1.0, -1.0, 0.1], [1.0, 1.0, 2.0], [0.3, -0.2, 0.7],
       6.0, 15, 0.01, [0], grupo="dificil", desafio="não identificável (só o produto)"),
    _m("seir_inicio", "SEIR (só o início)", "EDO", f_seir, _ic_fixa(0.999, 0.001, 0, 0), 4,
       ["beta", "sigma", "gamma"], [0.3, 0.1, 0.1], [1.5, 0.6, 0.5], [0.9, 0.3, 0.25],
       15.0, 12, 0.1, [2], grupo="dificil", desafio="pouco informativo"),
    _m("goodwin_hill", "Goodwin (Hill livre)", "EDO", f_goodwin_hill, _ic_fixa(0.5, 0.5, 0.5), 3,
       ["a", "b", "c", "d", "n"], [1.0, 0.1, 0.1, 0.1, 2.0], [6.0, 1.0, 1.0, 1.0, 12.0],
       [3.0, 0.4, 0.4, 0.4, 8.0], 40.0, 14, 0.02, [0], grupo="dificil", desafio="multimodal, dados esparsos"),
    _m("repressilator", "Repressilador", "EDO", f_repressilator, _ic_fixa(1, 2, 3, 2, 1, 3), 6,
       ["alpha", "alpha_0", "beta", "n"], [10.0, 0.0, 0.5, 1.5], [100.0, 2.0, 5.0, 3.5], [50.0, 0.5, 2.0, 2.4],
       30.0, 10, 0.02, [0], ruido=0.2, grupo="dificil", desafio="oscilatório, dados esparsos"),
    _m("calor_inverso", "Calor inverso", "EDP", f_calor_inv, ic_calor_inv, N,
       [f"c_{k}" for k in range(1, 9)], -np.ones(8), np.ones(8),
       [0.8, -0.5, 0.6, 0.4, -0.5, 0.3, -0.4, 0.3], 2.0, 2, 0.004, _PTS, ruido=0.1, grupo="dificil",
       desafio="mal-posto (modos altos apagados)"),
    dict(nome="darcy16", rotulo="Darcy (16 coeficientes)", tipo="EDP",
         nomes=[f"theta_{j}" for j in range(1, 17)], lo=-np.ones(16), hi=np.ones(16),
         verd=np.array([0.6, -0.4, 0.5, -0.3, 0.4, -0.2, 0.5, -0.6, 0.3, 0.4, -0.5, 0.2, -0.3, 0.6, -0.4, 0.3]),
         ruido=0.05, grupo="dificil", desafio="d alto, dominado pelo prior",
         sim=lambda TH: _darcy(np.ascontiguousarray(TH, float), _IDX_K16),
         pred=lambda TH: _darcy(np.ascontiguousarray(TH, float), _IDX_TODOS),
         estatico=True, idx=_IDX_K16, x=(_IDX_TODOS + 1.0) * DX_K, n_estados=MD - 1, campos=1,
         forma_obs=(1, len(_IDX_K16)), t_dados=np.zeros(1),
         rotulos_obs=[f"x = {(j + 1) * DX_K:.2f}" for j in _IDX_K16]),
    _m("fisher_frente", "Fisher–KPP (um instante)", "EDP", f_fisher, ic_fisher_frente, N,
       ["D", "r", "x_0"], [0.0005, 2.0, 0.05], [0.005, 10.0, 0.35], [0.002, 6.0, 0.15],
       1.5, 1, 0.01, _PTS, grupo="dificil", desafio="não identificável (crista D-r-x0)"),
    _m("allen_cahn_meta", "Allen–Cahn metaestável", "EDP", f_allen, ic_allen_meta, N,
       ["kappa", "r", "A"], [0.0002, 2.0, 0.5], [0.005, 8.0, 1.5], [0.001, 5.0, 1.0],
       4.0, 2, 0.01, _PTS, grupo="dificil", desafio="pouco informativo (estado quase estacionário)"),
    _m("fhn_difusao", "FitzHugh–Nagumo com difusão", "EDP", f_fhn_rd, ic_fhn_rd, 2 * N,
       ["D", "a", "b", "c"], [0.0005, 0.0, 0.0, 1.5], [0.004, 0.6, 0.6, 4.5], [0.0015, 0.2, 0.2, 3.0],
       6.0, 3, 0.004, _PTS[::2], ruido=0.2, grupo="dificil", desafio="onda não linear, dados esparsos", campos=2),
]
MODELOS = MODELOS + DIFICEIS


# ================================================================== modelos com dados reais
# Mesma estrutura dos sinteticos, mas o vetor observado vem de medidas publicadas: dados/<modelo>/*.csv na raiz
# do repositorio (config.DIR_DADOS), com as fontes e os sha256 em dados/fontes.json. Nao ha valor verdadeiro: verd e None, e centro (o ajuste de maxima
# verossimilhanca) so serve para cronometrar o simulador. O desvio do ruido e fixo por grupo de saidas
# (regra em fontes.json) e e o mesmo para a referencia e para todos os metodos.
import csv as _csv  # noqa: E402
import json as _json  # noqa: E402

from .config import DIR_DADOS as _DIR_DADOS  # noqa: E402


def _tabela(modelo, nome):
    """Linhas de dados/<modelo>/<nome> (CSV com cabecalho)."""
    arq = _DIR_DADOS / modelo / nome
    if not arq.exists():
        raise FileNotFoundError(f"dados reais de {modelo} nao encontrados em {arq} (defina A0_DADOS ou rode na "
                                f"raiz do repositorio)")
    with open(arq, newline="", encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def _fontes():
    return _json.loads((_DIR_DADOS / "fontes.json").read_text(encoding="utf-8"))


@njit
def ic_lv_real(th, y):
    y[0] = th[4]
    y[1] = th[5]


N_GRIPE = 763.0


@njit
def f_gripe(y, th, dy):
    """S, I (infeccioso, ainda de pe), B (acamado), C (convalescente), como no modelo bsflu do pomp."""
    inf = th[0] * y[0] * y[1] / N_GRIPE
    dy[0] = -inf
    dy[1] = inf - th[1] * y[1]
    dy[2] = th[1] * y[1] - th[2] * y[2]
    dy[3] = th[2] * y[2] - th[3] * y[3]


@njit
def f_pk2_iv(y, th, dy):
    """Dois compartimentos com bolus intravenoso; y em concentracao do compartimento central."""
    k10, k12, k21 = 10.0 ** th[0], 10.0 ** th[1], 10.0 ** th[2]
    dy[0] = -(k10 + k12) * y[0] + k21 * y[1]
    dy[1] = k12 * y[0] - k21 * y[1]


@njit
def ic_pk2_iv(th, y):
    y[0] = 10.0 ** th[3]
    y[1] = 0.0


NJ = 76                 # celulas de 25 um em [0, 1900] um (duas por coluna de 50 um dos dados)
DX_J = 25.0
K_JIN = 1.7             # capacidade de suporte, em 1e-3 celulas/um^2 (Jin et al., 2016)


@njit
def f_jin(y, th, dy):
    for j in range(NJ):
        e = y[j - 1] if j > 0 else y[0]
        d = y[j + 1] if j < NJ - 1 else y[NJ - 1]
        dy[j] = th[0] * (e - 2.0 * y[j] + d) / DX_J ** 2 + th[1] * y[j] * (1.0 - y[j] / K_JIN)


@njit
def f_jin_k(y, th, dy):
    """Fisher-KPP com a capacidade de suporte livre (th[2])."""
    for j in range(NJ):
        e = y[j - 1] if j > 0 else y[0]
        d = y[j + 1] if j < NJ - 1 else y[NJ - 1]
        dy[j] = th[0] * (e - 2.0 * y[j] + d) / DX_J ** 2 + th[1] * y[j] * (1.0 - y[j] / th[2])


def _log(Y):
    return np.log(np.maximum(Y, 1e-12))


def _raiz(Y):
    return np.sqrt(np.maximum(Y, 0.0))


def _real(spec, x_obs, grupo_sigma, fonte, sigmas, rotulos, centro, sigma_de=None, inicios=64):
    """Completa o dicionario de um modelo com os dados reais e o sigma calibrado (None: sigma = 1, para calibrar).
    sigma_de: nome do modelo cujo sigma vale tambem aqui (mesmo processo de medida). inicios: numero de inicios
    do ajuste de calibracao (mais nos modelos com muitos otimos locais)."""
    cal = (sigmas or {}).get(sigma_de or spec["nome"], {})
    mle = (sigmas or {}).get(spec["nome"], {}).get("mle")
    spec.update(verd=None, centro=np.array(mle if mle is not None else centro, float), rotulos_obs=list(rotulos),
                dados=dict(x_obs=np.asarray(x_obs, float), grupo_sigma=np.asarray(grupo_sigma, int),
                           sigma=cal.get("sigma"), fonte=fonte, sigma_de=sigma_de, inicios=inicios))
    return spec


def reais(sigma=None):
    """Modelos com dados reais. sigma=None le o sigma calibrado de fontes.json; sigma=False devolve os modelos
    com sigma = 1 (para a calibracao em dados_reais/preparar.py)."""
    sig = _fontes().get("sigma", {}) if sigma is None else (sigma or {})
    out = []

    # Lotka-Volterra nas peles de lebre e lince (1900 = t 0); estima tambem as populacoes iniciais
    lin = _tabela("lebre_lince", "lebre_lince.csv")
    t = np.array([float(r["ano"]) - 1900.0 for r in lin])
    Y = np.array([[float(r["lebre_mil"]), float(r["lince_mil"])] for r in lin])
    c0 = [0.55, 0.028, 0.8, 0.024, 34.0, 6.0]
    s = _m("lebre_lince", "Lebre e lince (LV)", "EDO", f_lv, ic_lv_real, 2,
           ["alpha", "beta", "gamma", "delta", "H_0", "L_0"], [0.2, 0.005, 0.3, 0.005, 10.0, 1.0],
           [1.2, 0.08, 1.6, 0.08, 70.0, 15.0], c0, 20.0, 21, 0.02, [0, 1], tobs=t, pos=_log)
    out.append(_real(s, np.log(Y).ravel(), np.tile([0, 1], len(t)), "lebre_lince", sig,
                     ["log lebre", "log lince"], c0))

    # gripe no internato: acamados e convalescentes, raiz quadrada das contagens; I(0) = 1 no dia 0
    c0 = [2.0, 1.0, 0.5, 0.3]
    n_dias = len(_tabela("gripe_internato", "gripe_internato.csv"))
    for nome, rot, k, grupo, desafio in (("gripe_internato", "Gripe no internato", n_dias, "facil", ""),
                                         ("gripe_internato_inicio", "Gripe no internato (só o início)", 5,
                                          "dificil", "pouco informativo (só os 5 primeiros dias)")):
        lin = _tabela(nome, "gripe_internato.csv")[:k]      # cada modelo le a propria pasta
        dias = np.array([float(r["dia"]) for r in lin])
        Y = np.array([[float(r["acamados"]), float(r["convalescentes"])] for r in lin])
        s = _m(nome, rot, "EDO", f_gripe, _ic_fixa(N_GRIPE - 1.0, 1.0, 0.0, 0.0), 4,
               ["beta", "mu_I", "mu_B", "mu_C"], [0.5, 0.1, 0.1, 0.05], [5.0, 3.0, 2.0, 1.5], c0,
               float(dias[k - 1]), k, 0.02, [2, 3], tobs=dias[:k], pos=_raiz, grupo=grupo, desafio=desafio)
        out.append(_real(s, _raiz(Y[:k]).ravel(), np.tile([0, 1], k), "gripe_internato", sig,
                         ["raiz acamados", "raiz convalescentes"], c0,
                         sigma_de=None if nome == "gripe_internato" else "gripe_internato"))

    # teofilina, sujeito 1: um compartimento com absorcao oral (o mesmo f do pk_flipflop), log da concentracao
    lin = [r for r in _tabela("pk_teofilina", "teofilina.csv") if int(r["sujeito"]) == 1 and float(r["tempo_h"]) > 0]
    t = np.array([float(r["tempo_h"]) for r in lin])
    c = np.array([float(r["conc_mg_L"]) for r in lin])
    dose = float(lin[0]["dose_mg_kg"])
    c0 = [0.2, -1.3, -0.35]
    s = _m("pk_teofilina", "PK da teofilina (flip-flop)", "EDO", f_flipflop, _ic_fixa(dose, 0.0), 2,
           ["log k_a", "log k_e", "log V"], [-2.0, -2.0, -2.5], [1.0, 1.0, 0.5], c0,
           float(t[-1]), len(t), 0.01, [1], tobs=t, pos=_log, grupo="dificil",
           desafio="multimodal (2 modos: troca de k_a e k_e)")
    out.append(_real(s, np.log(c), np.zeros(len(t), int), "teofilina", sig, ["log concentração"], c0))

    # indometacina, sujeito 1: dois compartimentos com bolus intravenoso, log da concentracao
    lin = [r for r in _tabela("pk_indometacina", "indometacina.csv") if int(r["sujeito"]) == 1]
    t = np.array([float(r["tempo_h"]) for r in lin])
    c = np.array([float(r["conc_ug_mL"]) for r in lin])
    c0 = [0.0, 0.0, -0.5, 0.4]
    s = _m("pk_indometacina", "PK da indometacina", "EDO", f_pk2_iv, ic_pk2_iv, 2,
           ["log k_10", "log k_12", "log k_21", "log C_0"], [-1.5, -1.5, -1.5, -0.5], [1.2, 1.2, 1.2, 1.5],
           c0, float(t[-1]), len(t), 0.005, [0], tobs=t, pos=_log)
    out.append(_real(s, np.log(c), np.zeros(len(t), int), "indometacina", sig, ["log concentração"], c0))

    # Fisher-KPP nos perfis de Jin et al. (2016), 10 mil celulas por poco, media das 3 replicas
    lin = [r for r in _tabela("fisher_kpp_jin", "jin2016.csv") if int(r["celulas_por_poco"]) == 10000]
    xs = sorted({float(r["x_um"]) for r in lin})
    ts = sorted({float(r["t_h"]) for r in lin})
    U = np.full((len(ts), len(xs)), np.nan)
    for r in lin:
        U[ts.index(float(r["t_h"])), xs.index(float(r["x_um"]))] = 1e3 * float(r["u_media"])
    xs = np.array(xs)
    xf = (np.arange(NJ) + 0.5) * DX_J
    u0 = np.interp(xf, xs[1:], U[0, 1:])          # a coluna da borda esquerda (x = 25 um) fica fora
    ncol, nt = len(xs), len(ts) - 1

    def pos_jin(Y):
        return Y.reshape(len(Y), nt, ncol, 2).mean(axis=3)[:, :, 1:].reshape(len(Y), -1)

    c0 = [500.0, 0.05]
    s = _m("fisher_kpp_jin", "Fisher–KPP (Jin et al.)", "EDP", f_jin, _ic_fixa(*u0), NJ,
           ["D", "lambda"], [50.0, 0.005], [3000.0, 0.15], c0, float(ts[-1]), nt, 0.05, list(range(NJ)),
           pos=pos_jin)
    s.update(forma_obs=(nt, ncol - 1), x=xf, idx=np.arange(2, NJ, 2), t_dados=np.array(ts[1:]))
    out.append(_real(s, U[1:, 1:].ravel(), np.zeros(U[1:, 1:].size, int), "jin2016", sig,
                     [f"x = {x:.0f} µm" for x in xs[1:]], c0))

    # os mesmos dados, so o ultimo instante (48 h), com a capacidade de suporte livre: analogo real do
    # "Fisher-KPP (um instante)"; um perfil nao separa difusao, proliferacao e capacidade
    lin = [r for r in _tabela("fisher_kpp_jin_instante", "jin2016.csv") if int(r["celulas_por_poco"]) == 10000]
    U1 = np.full((len(ts), len(xs)), np.nan)
    for r in lin:
        U1[ts.index(float(r["t_h"])), list(xs).index(float(r["x_um"]))] = 1e3 * float(r["u_media"])
    u01 = np.interp(xf, xs[1:], U1[0, 1:])

    def pos_inst(Y):
        return Y.reshape(len(Y), 1, ncol, 2).mean(axis=3)[:, :, 1:].reshape(len(Y), -1)

    c0 = [300.0, 0.045, 1.7]
    s = _m("fisher_kpp_jin_instante", "Fisher–KPP (Jin et al., um instante)", "EDP", f_jin_k, _ic_fixa(*u01), NJ,
           ["D", "lambda", "K"], [50.0, 0.005, 1.0], [3000.0, 0.15, 5.0], c0, float(ts[-1]), 1, 0.05,
           list(range(NJ)), pos=pos_inst, grupo="dificil",
           desafio="pouco identificável (um instante; crista entre proliferação e capacidade)")
    s.update(forma_obs=(1, ncol - 1), x=xf, idx=np.arange(2, NJ, 2), t_dados=np.array(ts[-1:]))
    out.append(_real(s, U1[-1, 1:].ravel(), np.zeros(ncol - 1, int), "jin2016", sig,
                     [f"x = {x:.0f} µm" for x in xs[1:]], c0, sigma_de="fisher_kpp_jin"))
    return out + reais2(sig)


# ------------------------------------------------------------------ modelos reais acrescentados em 22/11/2026
# Entram tambem os de ajuste fraco ou de dado processado (Veilleux, Gause, rato): o usuario pediu o maximo de
# modelos com dados reais. A ressalva de cada um esta em `desafio` e em dados/fontes.json.
LN10 = np.log(10.0)


@njit
def f_rm_real(y, th, dy):
    """Rosenzweig-MacArthur com as taxas em log10: r, K, a, h, e, m."""
    N, P = max(y[0], 0.0), max(y[1], 0.0)
    r, K, a = np.exp(LN10 * th[0]), np.exp(LN10 * th[1]), np.exp(LN10 * th[2])     # exp e mais rapida que 10 **
    h, e, m = np.exp(LN10 * th[3]), np.exp(LN10 * th[4]), np.exp(LN10 * th[5])
    g = a * N / (1.0 + a * h * N)
    dy[0] = r * N * (1.0 - N / K) - g * P
    dy[1] = e * g * P - m * P


@njit
def ic_rm_real(th, y):
    y[0] = th[6]
    y[1] = th[7]


T_IMIG = np.array([2.0, 5.0, 8.0, 11.0, 14.0])      # t = dia - 1: imigracao nos dias 3, 6, 9, 12 e 15


@njit
def f_gause(y, th, dy):
    """Lotka-Volterra com imigracao em pulsos (gaussianas de desvio 0,05 dia, 0,15 dia depois da contagem), a
    mesma quantidade th[4] para as duas especies. y[2] e o tempo."""
    N, P, t = max(y[0], 0.0), max(y[1], 0.0), y[2]
    pulso = 0.0
    for k in range(T_IMIG.shape[0]):
        u = (t - T_IMIG[k] - 0.15) / 0.05
        if abs(u) < 6.0:
            pulso += np.exp(-0.5 * u * u) / (0.05 * 2.5066282746310002)
    dy[0] = th[0] * N - th[1] * N * P + th[4] * pulso
    dy[1] = th[2] * N * P - th[3] * P + th[4] * pulso
    dy[2] = 1.0


@njit
def ic_gause(th, y):
    y[0] = th[5]
    y[1] = 0.0
    y[2] = 0.0


@njit
def f_pineno(y, th, dy):
    """Isomerizacao do alfa-pineno (Box et al., 1973), cinco constantes de primeira ordem em log10."""
    k1, k2, k3, k4, k5 = 10.0 ** th[0], 10.0 ** th[1], 10.0 ** th[2], 10.0 ** th[3], 10.0 ** th[4]
    dy[0] = -(k1 + k2) * y[0]
    dy[1] = k1 * y[0]
    dy[2] = k2 * y[0] - (k3 + k4) * y[2] + k5 * y[4]
    dy[3] = k3 * y[2]
    dy[4] = k4 * y[2] - k5 * y[4]


@njit
def f_hes1(y, th, dy):
    """Hes1 (Silk et al., 2011): mRNA, proteina citoplasmatica e nuclear; degradacao fixa de 0,03 por min."""
    p0, h, k1, nu = th[0], th[1], 10.0 ** th[2], 10.0 ** th[3]
    dy[0] = -0.03 * y[0] + 1.0 / (1.0 + (max(y[2], 0.0) / p0) ** h)
    dy[1] = -0.03 * y[1] + nu * y[0] - k1 * y[1]
    dy[2] = -0.03 * y[2] + k1 * y[1]


@njit
def ic_pendulo_real(th, y):
    y[0] = th[2]
    y[1] = th[3]


@njit
def _repouso_fhn(a, b):
    """Ponto fixo de x - x^3/3 + (a - x)/b = 0 (unico para b < 1), por bissecao."""
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mid - mid ** 3 / 3.0 + (a - mid) / b > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit
def f_fhn_rato(y, th, dy):
    """FitzHugh (1961) com escala de tempo tau (ms, em log10); y[2] = A (x* - x) e a saida observada."""
    k = 1.0 / 10.0 ** th[3]
    dx = k * th[2] * (y[0] - y[0] ** 3 / 3.0 + y[1])
    dy[0] = dx
    dy[1] = -k * (y[0] - th[0] + th[1] * y[1]) / th[2]
    dy[2] = -th[4] * dx


@njit
def ic_fhn_rato(th, y):
    """Parte do repouso, com a variavel de recuperacao deslocada de th[5] (o estimulo)."""
    xs = _repouso_fhn(th[0], th[1])
    y[0] = xs
    y[1] = (th[0] - xs) / th[1] + th[5]
    y[2] = 0.0


@njit
def _transporte(TH, xs, ts, s, e, ds):
    """Adveccao-dispersao 1D com entrada medida: c(x, t) = c_0 + r int e(s) g(t - s; x) ds, com a funcao de
    transferencia g(tau; x) = x / raiz(4 pi D tau^3) exp(-(x - v tau)^2 / (4 D tau)). TH = (v, log10 D, r, c_0).
    Saida (B, len(ts) * len(xs)), o tempo no indice lento."""
    B, nx, nt, ns = TH.shape[0], xs.shape[0], ts.shape[0], s.shape[0]
    out = np.empty((B, nt * nx))
    for b in range(B):
        v, D, r, c0 = TH[b, 0], 10.0 ** TH[b, 1], TH[b, 2], TH[b, 3]
        for i in range(nt):
            for j in range(nx):
                acc = 0.0
                for k in range(ns):
                    tau = ts[i] - s[k]
                    if tau > 0.0:
                        q = (xs[j] - v * tau) ** 2 / (4.0 * D * tau)
                        if q < 60.0:
                            acc += e[k] * xs[j] / np.sqrt(4.0 * PI * D * tau ** 3) * np.exp(-q)
                out[b, i * nx + j] = c0 + r * acc * ds
    return out


def reais2(sig):
    """Os modelos de 22/11/2026. `sig` e o dicionario de sigmas calibrados (vazio na calibracao)."""
    out = []

    # Didinium e Paramecium (Veilleux, 1976): Rosenzweig-MacArthur, log das densidades, 71 instantes
    lin = _tabela("didinium_veilleux", "veilleux.csv")
    t = np.array([float(r["dia"]) for r in lin])
    Y = np.array([[float(r["presa_paramecium_por_ml"]), float(r["predador_didinium_por_ml"])] for r in lin])
    c0 = [0.70, 2.24, -0.27, -0.60, 0.40, 0.96, 17.9, 27.3]
    s = _m("didinium_veilleux", "Didinium e Paramecium (Veilleux)", "EDO", f_rm_real, ic_rm_real, 2,
           ["log r", "log K", "log a", "log h", "log e", "log m", "N_0", "P_0"],
           [-0.5, 1.7, -2.5, -2.5, -1.5, -1.0, 5.0, 1.0], [1.2, 3.5, 0.3, 0.0, 0.7, 1.2, 40.0, 40.0], c0,
           float(t[-1]), len(t), 0.02, [0, 1], tobs=t, pos=_log, grupo="dificil",
           desafio="ciclo-limite; modelo mal especificado (explica 30% a 45% da variância)")
    out.append(_real(s, np.log(Y).ravel(), np.tile([0, 1], len(t)), "veilleux", sig,
                     ["log presa", "log predador"], c0, inicios=400))

    # Didinium e Paramecium com imigracao (Gause, 1934): raiz das contagens, 17 dias
    lin = _tabela("didinium_gause", "gause_1934_f32.csv")
    t = np.array([float(r["dia"]) - 1.0 for r in lin])
    Y = np.array([[float(r["presa_paramecium"]), float(r["predador_didinium"])] for r in lin])
    imig = [float(r["dia"]) - 1.0 for r in lin if int(r["imigracao"]) == 1]
    if not np.allclose(imig, T_IMIG):
        raise ValueError("didinium_gause: os dias de imigracao do CSV nao sao os do modelo")
    c0 = [1.0, 0.1, 0.05, 1.0, 1.0, 3.0]
    s = _m("didinium_gause", "Didinium e Paramecium com imigração (Gause)", "EDO", f_gause, ic_gause, 3,
           ["r", "a", "b", "m", "I", "N_0"], [0.1, 0.005, 0.005, 0.1, 0.02, 0.5], [3.0, 1.0, 1.0, 4.0, 8.0, 8.0], c0,
           float(t[-1]), len(t), 0.01, [0, 1], tobs=t, pos=_raiz, grupo="dificil",
           desafio="imigração em pulsos; contagens nulas; dado digitalizado de figura")
    out.append(_real(s, _raiz(Y).ravel(), np.tile([0, 1], len(t)), "gause", sig,
                     ["raiz presa", "raiz predador"], c0, inicios=300))

    # isomerizacao do alfa-pineno: cinco respostas em 8 instantes, escala linear, um sigma por resposta
    lin = _tabela("alfa_pineno", "alfa_pineno.csv")
    t = np.array([float(r["tempo_min"]) for r in lin])
    cols = ["y1_alfa_pineno", "y2_dipenteno", "y3_alo_ocimeno", "y4_pironeno", "y5_dimero"]
    Y = np.array([[float(r[c]) for c in cols] for r in lin])
    c0 = [-4.23, -4.53, -4.69, -3.56, -4.40]
    s = _m("alfa_pineno", "Isomerização do α-pineno", "EDO", f_pineno, _ic_fixa(100.0, 0.0, 0.0, 0.0, 0.0), 5,
           ["log k_1", "log k_2", "log k_3", "log k_4", "log k_5"], [-6.0] * 5, [-2.5] * 5, c0,
           float(t[-1]), len(t), 60.0, [0, 1, 2, 3, 4], tobs=t, grupo="dificil",
           desafio="respostas quase colineares; 8 instantes")
    out.append(_real(s, Y.ravel(), np.tile(np.arange(5), len(t)), "alfa_pineno", sig,
                     ["α-pineno", "dipenteno", "alo-ocimeno", "pironeno", "dímero"], c0, inicios=150))

    # Hes1 (Silk et al., 2011): mRNA em 8 instantes; analogo real do Goodwin com Hill livre
    lin = _tabela("hes1", "hes1.csv")
    t = np.array([float(r["tempo_min"]) for r in lin])
    y = np.array([float(r["mrna_relativo"]) for r in lin])
    c0 = [1.5, 5.0, -1.5, -1.5]
    s = _m("hes1", "Hes1 (oscilador de Goodwin)", "EDO", f_hes1, _ic_fixa(2.0, 5.0, 3.0), 3,
           ["P_0", "h", "log k_1", "log nu"], [0.1, 1.0, -3.0, -3.0], [10.0, 20.0, 0.0, 0.0], c0,
           float(t[-1]), len(t), 0.5, [0], tobs=t, grupo="dificil",
           desafio="oscilatório, dados esparsos (8 medidas)")
    out.append(_real(s, y, np.zeros(len(t), int), "hes1", sig, ["mRNA de Hes1"], c0, inicios=300))

    # pendulo (Giessler, 2025): angulo por captura optica, 130 instantes; estado inicial livre
    lin = _tabela("pendulo_mocap", "pendulo.csv")
    t = np.array([float(r["tempo_s"]) for r in lin])
    y = np.array([float(r["angulo_rad"]) for r in lin])
    c0 = [26.7, 0.02, -1.03, 0.0]
    s = _m("pendulo_mocap", "Pêndulo (captura de movimento)", "EDO", f_pendulo, ic_pendulo_real, 2,
           ["k", "gamma", "theta_0", "omega_0"], [10.0, 0.0, -1.2, -1.0], [45.0, 1.0, -0.9, 1.0], c0,
           float(t[-1]), len(t), 0.001, [0], tobs=t)
    out.append(_real(s, y, np.zeros(len(t), int), "pendulo", sig, ["ângulo (rad)"], c0))

    # rato (Metcalfe, 2020): FitzHugh-Nagumo ajustado ao disparo medio, 50 instantes em 0,5 ms
    lin = _tabela("fhn_rato", "potencial_medio.csv")
    t = np.array([float(r["tempo_ms"]) for r in lin])
    y = np.array([float(r["media"]) for r in lin])
    c0 = [0.7, 0.8, 3.0, -1.6, 0.1, -0.6]
    s = _m("fhn_rato", "FitzHugh–Nagumo (disparo médio, rato)", "EDO", f_fhn_rato, ic_fhn_rato, 3,
           ["a", "b", "c", "log tau", "A", "Delta"], [0.0, 0.05, 1.0, -2.2, 0.01, -2.5],
           [1.5, 0.95, 5.0, -0.5, 0.6, -0.02], c0, float(t[-1]), len(t), 0.0005, [2], tobs=t, grupo="dificil",
           desafio="pulso excitável; dado é média de disparos; parâmetros pouco identificáveis")
    out.append(_real(s, y, np.zeros(len(t), int), "fhn_rato", sig, ["potencial médio"], c0, inicios=400))

    # tracador em esgoto (Rieckermann et al., 2005): adveccao-dispersao, a curva a montante e a entrada
    up = _tabela("tracador_esgoto", "montante.csv")
    dn = _tabela("tracador_esgoto", "jusante.csv")
    comp = float(_tabela("tracador_esgoto", "trecho.csv")[0]["comprimento_m"])
    tu = np.array([float(r["tempo_s"]) for r in up])
    eu = np.array([float(r["excesso"]) for r in up])
    ds = 3.0
    sg = np.arange(tu[0], tu[-1] + 1e-9, ds)
    eg = np.interp(sg, tu, eu)
    t_dn = np.array([float(r["tempo_s"]) for r in dn])
    y_dn = np.array([float(r["condutividade_mS_cm"]) for r in dn])
    x_obs_pos = np.array([comp])
    x_campo = comp * (np.arange(8) + 1.0) / 8.0
    t_fino = np.linspace(tu[0], t_dn[-1], 80)

    def sim_esg(TH):
        return _transporte(np.ascontiguousarray(TH, float), x_obs_pos, t_dn, sg, eg, ds)

    def campo_esg(TH):
        return _transporte(np.ascontiguousarray(TH, float), x_campo, t_fino, sg, eg, ds)

    def traj_esg(TH):
        return campo_esg(TH).reshape(len(TH), len(t_fino), len(x_campo))

    c0 = [0.38, -1.15, 1.0, 0.93]
    s = dict(nome="tracador_esgoto", rotulo="Advecção–dispersão (traçador em esgoto)", tipo="EDP",
             nomes=["v", "log D", "r", "c_0"], lo=np.array([0.2, -2.0, 0.5, 0.8]), hi=np.array([0.6, 0.5, 1.5, 1.05]),
             grupo="facil", desafio="", sim=sim_esg, pred=campo_esg, traj=traj_esg, t_fino=t_fino,
             idx=np.array([7]), x=x_campo, n_estados=8, campos=1, forma_obs=(len(t_dn), 1), t_dados=t_dn,
             metodo="convolucao")
    out.append(_real(s, y_dn, np.zeros(len(t_dn), int), "tracador_esgoto", sig, ["condutividade a jusante"], c0))
    return out


REAIS = reais()
MODELOS = MODELOS + REAIS
POR_NOME = {m["nome"]: m for m in MODELOS}
