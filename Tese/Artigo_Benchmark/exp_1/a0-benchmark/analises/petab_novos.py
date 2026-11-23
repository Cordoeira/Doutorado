"""Importa os PEtab reais candidatos e mede d, n_obs, t_f, c_S e grad_ok (para a estimativa de tempo)."""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import logging
for n in ("pypesto", "amici", "petab"):
    logging.getLogger(n).setLevel(logging.CRITICAL)
from pathlib import Path  # noqa: E402
from a0bench import problemas  # noqa: E402

pasta = os.environ.get("A0_MODELOS_AMICI", os.path.expanduser("~/a0bench_runs/resultados_a0/_amici_models"))
saida = Path(__file__).resolve().parents[1] / "docs" / "petab_novos_custos.jsonl"
for nome in sys.argv[1:]:
    t0 = time.time()
    try:
        p = problemas.Petab(nome, 1.0, pasta, "petab")
        r = dict(nome=nome, ok=True, d=int(p.d), n_obs=int(len(p.x_obs)), t_f_ms=1e3 * p.t_f, c_s=float(p.c_s),
                 grad_ok=float(p.grad_ok), segundos_import=time.time() - t0, nomes=list(p.nomes),
                 n_cond=len(p.obj.edatas))
        ll = p.logpost_lote(p.amostrar_prior(20, __import__("numpy").random.default_rng(0)))
        r["frac_prior_finito"] = float(__import__("numpy").isfinite(ll).mean())
    except Exception as e:  # noqa: BLE001
        r = dict(nome=nome, ok=False, erro=f"{type(e).__name__}: {str(e)[:300]}", segundos_import=time.time() - t0)
    print(json.dumps(r), flush=True)
    with open(saida, "a") as f:
        f.write(json.dumps(r) + "\n")
