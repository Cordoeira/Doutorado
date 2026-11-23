"""Gera os CSVs curados de dados reais (dados/<modelo>/) a partir dos arquivos brutos, e calibra o desvio do ruido.

Uso (na raiz do repositorio):
    python -m a0bench.dados_reais.preparar                      # CSVs curados a partir de dados/_brutos
    python -m a0bench.dados_reais.preparar dados/_brutos --sigma   # tambem recalcula o sigma de cada modelo
    python -m a0bench.dados_reais.preparar --sigma hes1 fhn_rato   # so recalibra esses modelos (sem curar)

dados/_brutos tem uma subpasta por fonte, com os arquivos baixados das URLs listadas em dados/fontes.json:
    lebre_lince/hudson-bay-lynx-hare.csv   gripe/bsflu_data.txt   teofilina/Theoph.csv
    indometacina/Indometh.csv              jin/cell_density_profiles.npy
    veilleux/veilleux.rda                  gause/gause_1934_book_f32.rda   pinene/pinene.jl
    pendulo/angle.csv                      fhn_rato/1551.mat               esgoto/aaa_paper_std.zip
Os .rda do R sao lidos com o pacote rdata (opcional; procurado tambem em ~/r_libs ou em A0_RLIBS).

O arquivo de Jin et al. e um .npy com um dicionario em pickle. Ele e lido por um unpickler restrito, que so
aceita classes do numpy e tipos basicos (nada de codigo arbitrario).
"""
import csv
import hashlib
import json
import os
import pickle
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

from ..config import DIR_DADOS  # noqa: E402

FONTES = DIR_DADOS / "fontes.json"


def _sha(arq):
    return hashlib.sha256(Path(arq).read_bytes()).hexdigest()


def _gravar(nome, cab, linhas):
    """nome = <modelo>/<arquivo>.csv, dentro de dados/."""
    (DIR_DADOS / nome).parent.mkdir(parents=True, exist_ok=True)
    with open(DIR_DADOS / nome, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cab)
        w.writerows(linhas)
    print(f"{nome}: {len(linhas)} linhas")


class _Restrito(pickle.Unpickler):
    PERMITIDAS = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
                  ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
                  ("numpy", "ndarray"), ("numpy", "dtype"), ("builtins", "dict"), ("builtins", "list"),
                  ("_codecs", "encode")}

    def find_class(self, mod, nome):
        if (mod, nome) not in self.PERMITIDAS:
            raise pickle.UnpicklingError(f"classe proibida no pickle: {mod}.{nome}")
        if mod.startswith("numpy.core") and hasattr(np, "_core"):
            mod = mod.replace("numpy.core", "numpy._core")
        return super().find_class(mod, nome)


def _npy_pickle(arq):
    from numpy.lib import format as npf

    with open(arq, "rb") as f:
        v = npf.read_magic(f)
        _, _, dt = npf.read_array_header_1_0(f) if v == (1, 0) else npf.read_array_header_2_0(f)
        if not dt.hasobject:
            raise ValueError("esperava um .npy com objeto")
        obj = _Restrito(f, encoding="latin1").load()
    return obj.item() if isinstance(obj, np.ndarray) else obj


def _rda(arq):
    """Le um .rda do R com o pacote rdata (leitor em Python puro; nao executa codigo). Series temporais viram
    matrizes. Sem o pacote, devolve None (os CSVs curados ja estao em dados/; `pip install rdata` para refazer)."""
    extra = os.environ.get("A0_RLIBS") or str(Path.home() / "r_libs")
    if Path(extra).is_dir() and extra not in sys.path:
        sys.path.append(extra)                    # no fim: os pacotes do venv tem prioridade
    try:
        import rdata
    except ImportError:
        return None
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mapa = dict(rdata.conversion.DEFAULT_CLASS_MAP)
        mapa["ts"] = lambda o, a: np.asarray(o)
        mapa["mts"] = mapa["ts"]
        return rdata.conversion.convert(rdata.parser.parse_file(str(arq)), constructor_dict=mapa)


def curar2(b, fontes, hoje):
    """Conjuntos acrescentados em 22/11/2026: Veilleux, Gause, alfa-pineno, pendulo, rato e tracador em esgoto.
    (O Hes1 foi transcrito de artigo e nao tem arquivo bruto: dados/hes1/hes1.csv e a propria fonte.)"""
    def marca(chave, arq):
        fontes[chave].update(sha256_bruto=_sha(arq), obtido_em=fontes[chave].get("obtido_em") or hoje)

    # Veilleux (1976): Paramecium aurelia (presa) e Didinium nasutum (predador), a cada 12 h por 35 dias.
    # No pacote TSA os rotulos das colunas estao trocados: a serie 1 ("Didinium") antecede a serie 2 em cerca de
    # 1 dia (correlacao cruzada 0,82), logo e a presa. O CSV curado traz os nomes corrigidos.
    arq = b / "veilleux" / "veilleux.rda"
    o = _rda(arq)
    if o is None:
        print("pacote rdata ausente: veilleux e gause nao foram regenerados")
    else:
        V = np.asarray(o["veilleux"], float)
        _gravar("didinium_veilleux/veilleux.csv", ["dia", "presa_paramecium_por_ml", "predador_didinium_por_ml"],
                [[f"{0.5 * i:g}", f"{a:g}", f"{c:g}"] for i, (a, c) in enumerate(V)])
        marca("veilleux", arq)
        # Gause (1934), figura 32: Paramecium caudatum e Didinium nasutum com imigracao a cada tres dias
        arq = b / "gause" / "gause_1934_book_f32.rda"
        df = _rda(arq)["gause_1934_book_f32"]
        _gravar("didinium_gause/gause_1934_f32.csv", ["dia", "presa_paramecium", "predador_didinium", "imigracao"],
                [[int(r.Day), f"{float(r.Individuals_Prey):g}", f"{float(r.Individuals_Predator):g}",
                  1 if str(r.Immigration).strip() == "y" else 0] for r in df.itertuples()])
        marca("gause", arq)

    # alfa-pineno (Fuguitt e Hawkins, 1947; Box et al., 1973), valores do problema "pinene" da colecao COPS
    arq = b / "pinene" / "pinene.jl"
    txt = arq.read_text(encoding="utf-8")
    tau = [float(v) for v in re.search(r"\btau = \[(.*?)\]", txt, re.S).group(1).split(",")]
    z = [[float(v) for v in l.replace(";", " ").split()]
         for l in re.search(r"\bz = \[(.*?)\]", txt, re.S).group(1).strip().splitlines()]
    assert len(tau) == len(z) == 8 and all(len(l) == 5 for l in z)
    _gravar("alfa_pineno/alfa_pineno.csv", ["tempo_min", "y1_alfa_pineno", "y2_dipenteno", "y3_alo_ocimeno",
                                            "y4_pironeno", "y5_dimero"],
            [[f"{t:g}"] + [f"{v:g}" for v in l] for t, l in zip(tau, z)])
    marca("alfa_pineno", arq)

    # pendulo (Giessler, 2025): angulo pela captura optica de movimento (coluna MOCAP), 1 a cada 5 amostras
    arq = b / "pendulo" / "angle.csv"
    with open(arq, newline="") as f:
        r = list(csv.reader(f))
    assert r[0][0].strip() == "time" and r[0][1].strip() == "MOCAP"
    _gravar("pendulo_mocap/pendulo.csv", ["tempo_s", "angulo_rad"], [[l[0], l[1]] for l in r[1::5]])
    marca("pendulo", arq)

    # rato (Metcalfe, 2020): media dos disparos alinhados pelo pico, primeiro registro em repouso (1551), canal 1.
    # Disparo = maximo local (janela de 101 amostras) acima de 6 desvios robustos, a mais de 100 amostras do
    # anterior. Janela de 250 amostras (50 antes do pico), guardada 1 a cada 5.
    from scipy.io import loadmat

    arq = b / "fhn_rato" / "1551.mat"
    raw = loadmat(str(arq))["rawdata"]
    t, x = raw[:, 0], raw[:, 1] - np.median(raw[:, 1])
    dp = 1.4826 * np.median(np.abs(x))
    picos = []
    for i in range(100, len(x) - 250):
        if x[i] > 6 * dp and x[i] == x[i - 50:i + 51].max() and (not picos or i - picos[-1] > 100):
            picos.append(i)
    W = np.array([x[i - 50:i + 200] for i in picos])
    m, ep = W.mean(0), W.std(0, ddof=1) / np.sqrt(len(W))
    dt_ms = 1e3 * float(t[1] - t[0])
    _gravar("fhn_rato/potencial_medio.csv", ["tempo_ms", "media", "erro_padrao", "n_disparos"],
            [[f"{k * dt_ms:.4g}", f"{m[k]:.6g}", f"{ep[k]:.6g}", len(W)] for k in range(0, 250, 5)])
    marca("fhn_rato", arq)

    # tracador em esgoto (Rieckermann et al., 2005), experimento 2 (trecho RUE, 469 m, NaCl): curva a montante
    # (janela de 150 s antes a 450 s depois do pico, menos uma linha de base reta entre as medianas das duas
    # bordas) e curva a jusante, como medida
    import zipfile

    arq = b / "esgoto" / "aaa_paper_std.zip"
    zf = zipfile.ZipFile(arq)

    def dat(nome):
        lin = [l for l in zf.read(nome).decode("latin1").splitlines() if l and not l.startswith("#")]
        return np.array([[float(v) for v in l.split()] for l in lin[1:]])

    U, Dn = dat("aaa_paper_std/02/02_up.dat"), dat("aaa_paper_std/02/02_dn.dat")
    with open(b / "esgoto" / "dataset_parameters.csv", newline="") as f:
        par = next(l for l in csv.DictReader(f) if int(l["ID"]) == 2)
    tp = U[np.argmax(U[:, 1]), 0]
    antes = np.median(U[(U[:, 0] >= tp - 200) & (U[:, 0] < tp - 150), 1])
    depois = np.median(U[(U[:, 0] > tp + 450) & (U[:, 0] <= tp + 500), 1])
    jan = (U[:, 0] >= tp - 150) & (U[:, 0] <= tp + 450)
    base = antes + (depois - antes) * (U[jan, 0] - (tp - 150)) / 600.0
    _gravar("tracador_esgoto/montante.csv", ["tempo_s", "condutividade_mS_cm", "linha_de_base", "excesso"],
            [[f"{a:g}", f"{c:g}", f"{bb:.5g}", f"{c - bb:.5g}"] for (a, c), bb in zip(U[jan], base)])
    _gravar("tracador_esgoto/jusante.csv", ["tempo_s", "condutividade_mS_cm"], [[f"{a:g}", f"{c:g}"] for a, c in Dn])
    _gravar("tracador_esgoto/trecho.csv", ["experimento", "comprimento_m", "vazao_m3_s", "massa_kg"],
            [[2, par["L_m"], par["Q_m3s"], par["mass_kg"]]])
    marca("tracador_esgoto", arq)


def curar(brutos):
    b = Path(brutos)
    fontes = json.loads(FONTES.read_text(encoding="utf-8"))
    hoje = date.today().isoformat()

    # lebre e lince (Hudson's Bay Company, 1900-1920), milhares de peles
    arq = b / "lebre_lince" / "hudson-bay-lynx-hare.csv"
    lin = [l.split(",") for l in arq.read_text().splitlines() if l and not l.startswith("#")][1:]
    _gravar("lebre_lince/lebre_lince.csv", ["ano", "lebre_mil", "lince_mil"],
            [[int(a), float(h), float(l)] for a, l, h in lin])
    fontes["lebre_lince"].update(sha256_bruto=_sha(arq), obtido_em=fontes["lebre_lince"].get("obtido_em") or hoje)

    # gripe no internato (BMJ 1978), via King et al. (curso SBIED)
    arq = b / "gripe" / "bsflu_data.txt"
    lin = [l.split() for l in arq.read_text().splitlines() if l and not l.startswith("#")][1:]
    gripe = [[int(d), dt.strip('"'), int(bb), int(c)] for dt, bb, c, d in lin]
    _gravar("gripe_internato/gripe_internato.csv", ["dia", "data", "acamados", "convalescentes"], gripe)
    _gravar("gripe_internato_inicio/gripe_internato.csv", ["dia", "data", "acamados", "convalescentes"], gripe)
    fontes["gripe_internato"].update(sha256_bruto=_sha(arq), obtido_em=fontes["gripe_internato"].get("obtido_em") or hoje)

    # teofilina (R datasets::Theoph) e indometacina (R datasets::Indometh), via Rdatasets
    arq = b / "teofilina" / "Theoph.csv"
    with open(arq, newline="") as f:
        r = list(csv.DictReader(f))
    _gravar("pk_teofilina/teofilina.csv", ["sujeito", "peso_kg", "dose_mg_kg", "tempo_h", "conc_mg_L"],
            [[int(x["Subject"]), float(x["Wt"]), float(x["Dose"]), float(x["Time"]), float(x["conc"])] for x in r])
    fontes["teofilina"].update(sha256_bruto=_sha(arq), obtido_em=fontes["teofilina"].get("obtido_em") or hoje)
    arq = b / "indometacina" / "Indometh.csv"
    with open(arq, newline="") as f:
        r = list(csv.DictReader(f))
    _gravar("pk_indometacina/indometacina.csv", ["sujeito", "tempo_h", "conc_ug_mL"],
            [[int(x["Subject"]), float(x["time"]), float(x["conc"])] for x in r])
    fontes["indometacina"].update(sha256_bruto=_sha(arq), obtido_em=fontes["indometacina"].get("obtido_em") or hoje)

    # Jin et al. (2016): perfis de densidade de celulas PC-3, 6 densidades iniciais, 3 replicas
    arq = b / "jin" / "cell_density_profiles.npy"
    o = _npy_pickle(arq)
    linhas = []
    for k in sorted(o, key=lambda s: int(s.split("_")[1])):
        d = o[k]
        x, t = np.asarray(d["x"], float).ravel(), np.asarray(d["t"], float).ravel()
        for i, xi in enumerate(x):
            for j, tj in enumerate(t):
                linhas.append([int(k.split("_")[1]), xi, tj] + [f"{float(d[u][i, j]):.6g}" for u in ("U1", "U2", "U3", "U_mean")])
    cab = ["celulas_por_poco", "x_um", "t_h", "u_rep1", "u_rep2", "u_rep3", "u_media"]
    _gravar("fisher_kpp_jin/jin2016.csv", cab, linhas)
    _gravar("fisher_kpp_jin_instante/jin2016.csv", cab, linhas)
    fontes["jin2016"].update(sha256_bruto=_sha(arq), obtido_em=fontes["jin2016"].get("obtido_em") or hoje)
    curar2(b, fontes, hoje)
    FONTES.write_text(json.dumps(fontes, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def calibrar_sigma():
    """sigma de cada grupo de saidas = raiz(RSS / (n - p n_k / n)) no ajuste de maxima verossimilhanca com
    sigma = 1 (estimador plug-in, com correcao de graus de liberdade). O ajuste usa 64 inicios do prior (ou o
    numero dado em spec["dados"]["inicios"], nos modelos com muitos otimos locais)."""
    from .. import modelos, problemas

    fontes = json.loads(FONTES.read_text(encoding="utf-8"))
    sig = {}
    so = [a for a in sys.argv[sys.argv.index("--sigma") + 1:] if not a.startswith("-")] if "--sigma" in sys.argv else []
    for spec in modelos.reais(sigma=False):
        if so and spec["nome"] not in so:                 # --sigma modelo1 modelo2: recalibra so esses
            continue
        if spec["dados"].get("sigma_de"):                 # usa o sigma de outro modelo (mesmo processo de medida)
            continue
        p = problemas.Sintetico(spec)
        melhor = None
        for x0 in p.amostrar_prior(int(spec["dados"].get("inicios", 64)), np.random.default_rng(0)):
            r = p._ajustar(x0)
            if melhor is None or r[0] < melhor[0]:
                melhor = r
        x = melhor[1]
        res = p.sim(x[None])[0] - p.x_obs
        grupos = np.asarray(spec["dados"]["grupo_sigma"])
        n, d = len(res), p.d
        s = []
        for g in range(grupos.max() + 1):
            m = grupos == g
            s.append(float(np.sqrt(np.sum(res[m] ** 2) / (m.sum() - d * m.sum() / n))))
        sig[spec["nome"]] = dict(sigma=s, mle=x.tolist(), nll_sigma1=float(melhor[0]),
                                 mle_na_borda=bool(np.any(np.isclose(x, p.lo, rtol=0, atol=1e-3 * (p.hi - p.lo)) |
                                                          np.isclose(x, p.hi, rtol=0, atol=1e-3 * (p.hi - p.lo)))))
        print(spec["nome"], "sigma", np.round(s, 4), "mle", np.round(x, 4), "borda", sig[spec["nome"]]["mle_na_borda"])
    fontes["sigma"] = {**fontes.get("sigma", {}), **sig}
    FONTES.write_text(json.dumps(fontes, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--sigma":
        curar(sys.argv[1])
    elif "--sigma" not in sys.argv:
        curar(DIR_DADOS / "_brutos")
    if "--sigma" in sys.argv:
        calibrar_sigma()
