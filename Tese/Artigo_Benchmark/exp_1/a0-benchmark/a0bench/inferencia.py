"""Tarefas por modelo: MAP, referencia, solucoes e superficies para os graficos, banco de simulacoes, NPE,
NPE-PFN e metricas."""
import contextlib
import io

import numpy as np

from .cadeias import amostrar_laplace, amostrar_pt, apagar_ck, ess, rhat
from .util import gravar, gravar_pickle, ler_pickle


def tarefa_map(p, pasta, cfg):
    arq = pasta / "map.pkl"
    if arq.exists():
        return ler_pickle(arq)
    out = p.mapa(cfg)
    gravar_pickle(arq, out)
    return out


def tarefa_referencia(p, pasta, mapa, cfg, prog=None):
    """Temperacao paralela; dobra o comprimento ate R-hat e ESS passarem, ou ate n_ref_max.

    Grava referencia.npz (amostras embaralhadas e diagnostico) e cadeias/referencia.npz (o traco das cadeias
    frias, desbastado para no maximo `pontos_ref` pontos por cadeia). Devolve tambem o passo de retomada.
    """
    arq = pasta / "referencia.npz"
    if arq.exists():
        return dict(np.load(arq)), None
    m, n, base = cfg["n_cadeias_ref"], cfg["n_ref"], pasta / "referencia"
    X0 = amostrar_laplace(p, mapa, m, np.random.default_rng(12345))
    X0[~np.isfinite(p.logpost_lote(X0))] = mapa["x"]
    retomado, primeira = None, True
    while True:
        res = amostrar_pt(p, X0, mapa["cov"], n, 1000, base, cfg["passos_ck"], cfg["betas"],
                          prog=(lambda f, n=n: prog(f, f"{n} passos")) if prog else None, manter=True)
        if primeira:    # as dobras seguintes continuam a propria cadeia; isso nao e uma retomada
            retomado, primeira = res["retomado"], False
        cad = res["cadeia"]
        pos = cad[:, n // 4:]
        r = float(rhat(pos).max())
        e = float(min(sum(ess(pos[c, :, j]) for c in range(m)) for j in range(p.d)))
        ok = r < cfg["rhat_max"] and e >= cfg["ess_min"]
        if ok or 2 * n > cfg["n_ref_max"]:
            break
        n *= 2
    passo = np.linspace(0, pos.shape[1] - 1, min(10000 // m, pos.shape[1])).astype(int)
    out = dict(amostras=np.random.default_rng(0).permutation(pos[:, passo].reshape(-1, p.d)),
               rhat_max=r, ess_min=e, troca=float(res["troca"]), n_passos=n, ok=bool(ok),
               esd=float(m * n * len(cfg["betas"])))
    k = max(1, int(np.ceil(n / cfg["pontos_ref"])))
    gravar(pasta / "cadeias" / "referencia.npz", lambda fh: np.savez(
        fh, cadeia=cad[:, ::k], logp=res["logp"][:, ::k], aquecimento=n // 4, desbaste=k, passos=n, caminhantes=1,
        aceitacao=np.asarray(res["aceitacao"], float), esd=float(n * len(cfg["betas"])), sementes=np.arange(m),
        rhat=rhat(pos), nomes=np.array(p.nomes)))
    gravar(arq, lambda fh: np.savez(fh, **out))
    apagar_ck(base)
    return out, retomado


def _centro(p, mapa, ref):
    """O melhor ponto conhecido: o MAP ou, se for melhor, a melhor amostra da referencia."""
    x, f = np.array(mapa["x"], float), p.nll(mapa["x"])
    sub = ref["amostras"][:400]
    ll = p.logpost_lote(sub)
    k = int(np.argmax(ll))
    return (sub[k].copy(), float(-ll[k])) if -ll[k] < f - 0.5 else (x, f)


def tarefa_solucao(p, pasta, mapa, ref, cfg):
    """Solucoes do modelo para os graficos: saidas no formato dos dados (ponto central e faixa a posteriori)
    e, nos modelos sinteticos, todos os estados numa malha fina (centro, quantis e algumas trajetorias)."""
    arq = pasta / "solucao.npz"
    if arq.exists():
        return
    xc, _ = _centro(p, mapa, ref)
    X = ref["amostras"][: cfg["n_sol"]]
    out = dict(x_centro=xc, rotulos_obs=np.array(p.rotulos_obs), tipo=str(p.tipo))
    sol_c, sol, dados = p.solucao(xc[None]), p.solucao(X), p.dados()
    out["n_cond"] = len(sol)
    for c, ((t, Yc), (_, Y), (_, D, S)) in enumerate(zip(sol_c, sol, dados)):
        bom = np.all(np.isfinite(Y.reshape(len(Y), -1)), axis=1)
        q = np.quantile(Y[bom], [0.025, 0.5, 0.975], axis=0) if bom.sum() >= 5 else np.full((3, *Y.shape[1:]), np.nan)
        out.update({f"t_{c}": t, f"dados_{c}": D, f"sigma_{c}": S, f"centro_{c}": Yc[0], f"q_{c}": q})
    est = p.estados(np.vstack([xc[None], X[: cfg["n_traj"]]]))
    if est is not None:
        t, Y = est
        bom = np.all(np.isfinite(Y[1:].reshape(len(Y) - 1, -1)), axis=1)
        out.update(t_est=t, est_centro=Y[0], est_amostras=Y[1:][bom].astype(np.float32),
                   est_q=np.quantile(Y[1:][bom], [0.025, 0.5, 0.975], axis=0) if bom.sum() >= 5
                   else np.full((3, *Y.shape[1:]), np.nan),
                   campos=int(p.spec.get("campos", 1)), idx=np.asarray(p.spec.get("idx", []), int),
                   estatico=bool(p.spec.get("estatico", False)))
        if p.spec.get("x") is not None:
            out["x_malha"] = np.asarray(p.spec["x"], float)
        if p.verd is not None:
            out["est_verd"] = p.estados(p.verd[None])[1][0]
    gravar(arq, lambda fh: np.savez(fh, **out))


def tarefa_superficie(p, pasta, mapa, ref, cfg):
    """Log da posterior em grades de dois parametros (os demais fixos no ponto central), para os contornos.

    Pares: o de maior correlacao a posteriori, o dos dois parametros menos contraidos e o primeiro par. A
    janela cobre o modo em torno do ponto central, limitada ao prior."""
    arq = pasta / "superficie.npz"
    if arq.exists() or p.d < 2:
        return
    am = ref["amostras"]
    xc, _ = _centro(p, mapa, ref)
    corr = np.corrcoef(am.T) if am.std(axis=0).min() > 0 else np.eye(p.d)
    np.fill_diagonal(corr, 0.0)
    contr = 1.0 - am.var(axis=0) / ((p.hi - p.lo) ** 2 / 12.0)
    cand = [tuple(sorted(np.unravel_index(int(np.argmax(np.abs(corr))), corr.shape))),
            tuple(sorted(np.argsort(contr)[:2].tolist())), (0, 1)]
    pares = []
    for par in cand:
        par = (int(par[0]), int(par[1]))
        if par[0] != par[1] and par not in pares:
            pares.append(par)
    pares = pares[: cfg["n_pares"]]
    # janela: o modo que contem o ponto central. A escala local vem da aproximacao de Laplace; as amostras
    # da referencia a menos de 10 escalas do centro (o mesmo modo) alargam a janela se forem mais espalhadas
    larg = p.hi - p.lo
    esc = np.clip(np.sqrt(np.maximum(np.diag(mapa["cov"]), 0.0)), 2e-3 * larg, larg)
    perto = am[np.all(np.abs(am - xc) <= 10 * esc, axis=1)]
    meia = 6 * esc
    if len(perto) >= 20:
        meia = np.maximum(meia, 1.6 * np.quantile(np.abs(perto - xc), 0.995, axis=0))
    lo, hi = np.maximum(p.lo, xc - meia), np.minimum(p.hi, xc + meia)
    G = cfg["n_grade"]
    eixo_i, eixo_j, logp = [], [], []
    for i, j in pares:
        gi, gj = np.linspace(lo[i], hi[i], G), np.linspace(lo[j], hi[j], G)
        X = np.tile(xc, (G * G, 1))
        X[:, i], X[:, j] = np.repeat(gi, G), np.tile(gj, G)
        eixo_i.append(gi)
        eixo_j.append(gj)
        logp.append(p.logpost_lote(X).reshape(G, G))
    gravar(arq, lambda fh: np.savez(fh, pares=np.array(pares), eixo_i=np.array(eixo_i), eixo_j=np.array(eixo_j),
                                    logp=np.array(logp), x_centro=xc))


def tarefa_banco(p, pasta, cfg, prog=None):
    arq, n = pasta / "banco.npz", cfg["n_sim"]
    if arq.exists():
        b = dict(np.load(arq))
        if int(b["n_pedido"]) >= n:
            return b
    else:
        b = dict(theta=np.empty((0, p.d)), x=np.empty((0, len(p.x_obs))), n_pedido=0, falhas=0)
    while int(b["n_pedido"]) < n:
        i0 = int(b["n_pedido"])
        lote = min(500, n - i0)
        rng = np.random.default_rng([777, i0])
        th = p.amostrar_prior(lote, rng)
        x, ok = p.simular_lote(th, rng)
        b = dict(theta=np.concatenate([b["theta"], th[ok]]), x=np.concatenate([b["x"], x[ok]]),
                 n_pedido=i0 + lote, falhas=int(b["falhas"]) + int((~ok).sum()))
        gravar(arq, lambda fh, b=b: np.savez(fh, **b))
        if prog:
            prog(b["n_pedido"] / n, "simulando")
    return b


class _SemRegistro:
    """Registro de treino que nao grava nada (por padrao o sbi cria a pasta sbi-logs onde o comando roda)."""

    log_dir = None

    def log_metric(self, *a, **k):
        pass

    log_metrics = log_params = add_figure = log_metric

    def flush(self):
        pass


def _torch_prior(p, dispositivo="cpu"):
    import torch
    from sbi.utils import BoxUniform

    return BoxUniform(torch.tensor(p.lo, dtype=torch.float32, device=dispositivo),
                      torch.tensor(p.hi, dtype=torch.float32, device=dispositivo), device=dispositivo)


def _limpar(banco):
    """Remove simulacoes com valores absurdos, que desestabilizam o treino."""
    x = banco["x"]
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) + 1e-12
    ok = np.all(np.abs(x - med) < 1e4 * mad, axis=1)
    return banco["theta"][ok], x[ok]


def amostrar_npe(p, banco, n, semente, lote=200):
    import torch
    from sbi.inference import NPE

    torch.manual_seed(semente)
    th, x = _limpar(banco)
    th, x = torch.tensor(th, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)
    inf = NPE(prior=_torch_prior(p), show_progress_bars=False, tracker=_SemRegistro())
    with contextlib.redirect_stdout(io.StringIO()):
        est = inf.append_simulations(th, x).train(training_batch_size=lote, show_train_summary=False)
    # amostra direto do fluxo e descarta o que cai fora do prior (sem laco de rejeicao sem fim)
    xo = torch.tensor(p.x_obs, dtype=torch.float32)[None]
    aceitas, total = [], 0
    with torch.no_grad():
        for _ in range(25):
            s = est.sample((2 * n,), condition=xo).reshape(-1, p.d).numpy().astype(float)
            total += len(s)
            aceitas.append(s[p.dentro(s) & np.all(np.isfinite(s), axis=1)])
            if sum(len(a) for a in aceitas) >= n:
                break
    s = np.concatenate(aceitas)[:n]
    if len(s) < n // 4:
        raise RuntimeError(f"NPE com vazamento: so {len(s)} de {total} amostras dentro do prior")
    return s, float(banco["n_pedido"])


def amostrar_npe_pfn(p, banco, n, semente, n_ctx, dispositivo="auto"):
    """NPE-PFN (TabPFN como estimador de densidade, sem treino). `dispositivo`: "cpu", "cuda" ou "auto" (usa a
    GPU se houver e volta para a CPU se ela falhar). Na CPU o contexto e limitado a 1000 simulacoes."""
    import torch
    from sbi.inference import NPE_PFN

    th, x = _limpar(banco)

    def rodar(disp):
        torch.manual_seed(semente)
        m = n_ctx if disp != "cpu" else min(n_ctx, 1000)
        idx = np.random.default_rng(semente).choice(len(th), size=min(m, len(th)), replace=False)
        inf = NPE_PFN(prior=_torch_prior(p, disp), device=disp, show_progress_bars=False, tracker=_SemRegistro())
        with contextlib.redirect_stdout(io.StringIO()):
            inf.append_simulations(torch.tensor(th[idx], dtype=torch.float32),
                                   torch.tensor(x[idx], dtype=torch.float32))
            s = inf.build_posterior().sample((n,), x=torch.tensor(p.x_obs, dtype=torch.float32, device=disp),
                                             show_progress_bars=False)
        return s.detach().cpu().numpy().astype(float), float(len(idx))

    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        if dispositivo == "cuda":
            try:
                return rodar("cuda")
            except Exception:  # noqa: BLE001  (falta de memoria na GPU, por exemplo): tenta na CPU
                torch.cuda.empty_cache()
                dispositivo = "cpu"
    return rodar(dispositivo)


def metricas(p, amostras, ref, n):
    import torch
    from sbi.utils.metrics import c2st

    n = min(len(amostras) // 2, len(ref), n)
    sub = amostras[np.linspace(0, len(amostras) - 1, n).astype(int)]
    a = torch.tensor((sub - p.lo) / (p.hi - p.lo), dtype=torch.float32)
    b = torch.tensor((ref[:n] - p.lo) / (p.hi - p.lo), dtype=torch.float32)
    dp = np.maximum(ref.std(axis=0), 1e-300)
    return dict(c2st=float(c2st(a, b)), n_c2st=int(n),
                erro_media=float(np.max(np.abs(amostras.mean(0) - ref.mean(0)) / dp)),
                razao_dp=float(np.exp(np.mean(np.log(np.maximum(amostras.std(axis=0), 1e-300) / dp)))))
