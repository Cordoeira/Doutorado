"""Leitura dos resultados gravados e tabelas-resumo."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["EDO", "EDP", "EDO difícil", "EDP difícil", "PEtab", "PEtab grande"]


def _classe(info):
    g = info.get("grupo", "facil")
    if g.startswith("petab"):
        return "PEtab grande" if g == "petab_grande" else "PEtab"
    return info.get("tipo", "EDO") + (" difícil" if g == "dificil" else "")


def problemas(saida):
    linhas = []
    for a in sorted(Path(saida).glob("*/problema.json")):
        r = json.loads(a.read_text())
        r["classe"] = _classe(r)
        # resultados gravados antes da versao com dados reais: so os PEtab tinham dados reais
        r.setdefault("dados", "real" if str(r.get("grupo", "")).startswith("petab") else "sintetico")
        linhas.append(r)
    return pd.DataFrame(linhas)


def resultados(saida):
    info = {r["problema"]: r for r in problemas(saida).to_dict("records")}
    regs = []
    for arq in sorted(Path(saida).glob("*/*_s*.json")):
        r = json.loads(arq.read_text())
        regs.append({**info.get(r["problema"], {}), **r})
    return pd.DataFrame(regs)


def tabela(saida):
    """Mediana entre sementes, por modelo e metodo."""
    df = resultados(saida)
    if df.empty or "ok" not in df or not df["ok"].any():
        return pd.DataFrame()       # nenhum metodo terminou bem (ou so ha identificabilidade)
    df = df[df["ok"].astype(bool)]
    for col in ("ess_por_mil_esd", "rhat_max", "aceitacao"):
        if col not in df:
            df = df.assign(**{col: np.nan})
    g = df.groupby(["classe", "rotulo", "problema", "d", "ref_ok", "metodo"], sort=False)
    return g.agg(c2st=("c2st", "median"), erro_media=("erro_media", "median"), razao_dp=("razao_dp", "median"),
                 esd=("esd", "median"), ess_por_mil_esd=("ess_por_mil_esd", "median"),
                 rhat_max=("rhat_max", "median"), aceitacao=("aceitacao", "median"),
                 sementes=("semente", "count")).round(3).reset_index()


def identificabilidade(saida):
    saida = Path(saida)
    info = {r["problema"]: r for r in problemas(saida).to_dict("records")}
    linhas = []
    for arq in sorted(saida.glob("*/identificabilidade.json")):
        r = json.loads(arq.read_text())
        i = info.get(r["problema"])
        if i is None:
            continue
        for k, q in enumerate(r["parametros"]):
            linhas.append(dict(problema=r["problema"], rotulo=i["rotulo"], classe=i["classe"], d=i["d"],
                               ref_ok=i["ref_ok"], k=k + 1, parametro=q["nome"], contracao=q["contracao"],
                               classe_id=q["classe"], natureza=q.get("natureza"),
                               multimodal=bool(q.get("multimodal", False)), n_componentes=q.get("n_componentes", 1),
                               lado_inf=(q.get("lados") or {}).get("inf"), lado_sup=(q.get("lados") or {}).get("sup"),
                               limitado_por=", ".join(q.get("bloqueio", [])),
                               ic_inf=q["ic"][0], ic_sup=q["ic"][1],
                               ic_conjunto="; ".join(f"[{a:.4g}, {b:.4g}]" for a, b in q.get("ic_conjunto", [q["ic"]])),
                               compensado_por=", ".join(c["nome"] for c in q.get("compensam", [])),
                               dim_informada=r["dim_informada"]))
    return pd.DataFrame(linhas)


def intervalos(saida):
    """Os quatro intervalos de confianca da secao 2.2 de Raue et al. (2009) para cada parametro, na
    coordenada do perfil: assintotico (pela Hessiana) e pela verossimilhanca (pelo perfil), cada um pontual
    e simultaneo. Um intervalo assintotico infinito sai vazio. No do perfil, `perfil_pontual_extremos` traz
    o rotulo de cada extremo (dados, aberto, prior, falha) e `perfil_simultaneo_chega_ao_limite` diz se o
    simultaneo encosta no limite do dominio (nao e limitado pelos dados daquele lado)."""
    info = problemas(saida).set_index("problema") if not problemas(saida).empty else None
    linhas = []
    for arq in sorted(Path(saida).glob("*/identificabilidade.json")):
        r = _ler(arq)
        if not r or "parametros" not in r or "chi2" not in r["parametros"][0]:
            continue
        for q in r["parametros"]:
            a, pf = q.get("assintotico") or {}, q["ic_perfil"]
            fin = bool(a) and not a.get("infinito", True)

            def txt(lista):
                return " U ".join(f"[{x:.4g}; {y:.4g}]" for x, y in lista)
            linhas.append(dict(
                problema=r["problema"], rotulo=info.loc[r["problema"], "rotulo"] if info is not None else r["problema"],
                parametro=q["nome"], coordenada=q["rotulo_coord"], nivel=r["nivel_confianca"],
                valor_calibrado=q["u_otimo"], otimo_no_limite=bool(a.get("no_limite", False)),
                metodo_assintotico=a.get("metodo", "gauss-newton") if a else "", observacao=a.get("motivo") or "",
                dp_assintotico=a.get("dp") if fin else None, dp_hessiana=a.get("dp_hessiana"),
                dp_gauss_newton=a.get("dp_gauss_newton"),
                assintotico_pontual=txt([a["pontual"]]) if fin else "", assintotico_simultaneo=txt([a["simultaneo"]]) if fin else "",
                assintotico_infinito=bool(a) and not fin,
                perfil_pontual=txt(pf["pontual_u"]), perfil_simultaneo=txt(pf["simultaneo_u"]),
                perfil_pontual_extremos="/".join(q["lados"][k] for k in ("inf", "sup")),
                perfil_simultaneo_chega_ao_limite="/".join(str(x) for x in pf["simultaneo_chega_ao_limite"]),
                classe_id=q["classe"]))
    return pd.DataFrame(linhas)


def resumo_identificabilidade(saida):
    """Uma linha por modelo: contagens, dimensoes informadas e combinacoes de parametros."""
    saida = Path(saida)
    info = {r["problema"]: r for r in problemas(saida).to_dict("records")}
    linhas = []
    for arq in sorted(saida.glob("*/identificabilidade.json")):
        r = json.loads(arq.read_text())
        i = info.get(r["problema"])
        if i is None:
            continue
        P = r["parametros"]
        e, l, pr = r.get("estrutura") or {}, r.get("lis") or {}, r.get("predicao") or {}

        def textos(chave):
            return "; ".join(c["texto"] or str(c["vetor"]) for c in e.get(chave, []))

        def dim(esp, eps="0.1"):
            return (esp or {}).get("dim", {}).get(eps)
        linhas.append(dict(
            problema=r["problema"], rotulo=i["rotulo"], classe=i["classe"], d=i["d"], ref_ok=i["ref_ok"],
            identificaveis=sum(q["classe"] == "identificavel" for q in P),
            parciais=sum(q["classe"] == "parcial" for q in P),
            nao_identificaveis=sum(q["classe"] == "nao identificavel" for q in P),
            estruturais=sum(q.get("natureza") == "estrutural" for q in P),
            multimodais=sum(bool(q.get("multimodal")) for q in P),
            n_saidas=e.get("n_saidas"), posto_no_otimo=e.get("posto"), posto_generico=e.get("posto_generico"),
            nucleo_invariante=e.get("dim_nucleo_invariante"),
            combinacoes_identificaveis=textos("combinacoes_identificaveis"), combinacoes_livres=textos("nucleo"),
            dim_fisher_map=r["dim_informada"],
            dim_h_normal=dim(l.get("normal")), dim_h_caixa=dim(l.get("caixa")),
            dim_h_livre_normal=dim((l.get("livre_de_dados") or {}).get("normal")),
            dim_h_livre_caixa=dim((l.get("livre_de_dados") or {}).get("caixa")),
            predicao_razao_posterior=pr.get("razao_posterior"), predicao_razao_completa=pr.get("razao_completa"),
            map_melhorado=(r.get("map_melhorado") or {}).get("melhora"),
            falhas_perto_do_otimo=(r.get("vizinhanca") or {}).get("falhas"), esd=r.get("esd")))
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------------------------ reuso dos resultados
FORA_DO_ARQUIVO = ("em_execucao.lock", ".tmp", ".ck", ".ck.cadeia.npy", ".ck.logp.npy")


def _npz(arq):
    if not Path(arq).exists():
        return None
    with np.load(arq, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _ler(arq, padrao=None):
    try:
        return json.loads(Path(arq).read_text())
    except (OSError, ValueError):
        return padrao


class Arquivo:
    """Acesso aos resultados gravados de uma execucao, sem rodar nada.

        r = carregar("resultados_a0")
        r.modelos                               nomes dos modelos com resultado
        r.tabela()                              mediana das metricas por modelo e metodo
        r.cadeia("lotka_volterra", "hmc")       cadeias completas: cadeia (C, n, d), logp, aquecimento, ...
        r.amostras("lotka_volterra", "hmc")     amostras ja sem aquecimento (todas as sementes)
        r.referencia("lotka_volterra")          amostras da posterior de referencia
        r.identificabilidade("lotka_volterra")  perfis, estrutura, subespaco informado, predicao
    """

    def __init__(self, saida):
        self.saida = Path(saida)
        if not self.saida.is_dir():
            raise FileNotFoundError(f"pasta de resultados não encontrada: {saida}")
        self.config = _ler(self.saida / "config.json", {})
        self.execucao = _ler(self.saida / "execucao.json", {})
        self.ambiente = _ler(self.saida / "ambiente.json", {})

    def __repr__(self):
        return f"Arquivo({str(self.saida)!r}: {len(self.modelos)} modelos, preset {self.config.get('preset', '?')})"

    @property
    def modelos(self):
        return sorted(a.parent.name for a in self.saida.glob("*/problema.json"))

    def problema(self, modelo):
        """Descricao do modelo: nomes dos parametros, limites do prior, diagnostico da referencia."""
        return _ler(self.saida / modelo / "problema.json", {})

    def metodos(self, modelo):
        return sorted({a.name.rsplit("_s", 1)[0] for a in (self.saida / modelo).glob("*_s*.json")})

    def referencia(self, modelo):
        r = _npz(self.saida / modelo / "referencia.npz")
        return None if r is None else r["amostras"]

    def amostras(self, modelo, metodo, semente=None):
        """Amostras da posterior segundo o metodo; sem `semente`, junta todas."""
        arqs = sorted((self.saida / modelo).glob(f"{metodo}_s{'*' if semente is None else semente}.npy"))
        return np.concatenate([np.load(a) for a in arqs]) if arqs else None

    def metricas(self, modelo, metodo):
        """Lista com um dicionario por semente: C2ST, erro da media, ESS, custo, tempo."""
        return [r for a in sorted((self.saida / modelo).glob(f"{metodo}_s*.json")) if (r := _ler(a))]

    def cadeia(self, modelo, metodo):
        """Cadeias completas do amostrador ("metropolis", "ensemble", "mala", "hmc") ou da "referencia"."""
        return _npz(self.saida / modelo / "cadeias" / f"{metodo}.npz")

    def identificabilidade(self, modelo):
        return _ler(self.saida / modelo / "identificabilidade.json")

    def solucao(self, modelo):
        return _npz(self.saida / modelo / "solucao.npz")

    def superficie(self, modelo):
        return _npz(self.saida / modelo / "superficie.npz")

    def registro(self, modelo=None):
        """Historico do processo: eventos de um modelo (lista de dicionarios) ou o registro geral (texto)."""
        if modelo is None:
            arq = self.saida / "registro.log"
            return arq.read_text(encoding="utf-8") if arq.exists() else ""
        arq = self.saida / modelo / "registro.jsonl"
        return [json.loads(l) for l in arq.read_text(encoding="utf-8").splitlines() if l.strip()] if arq.exists() else []

    def tabela(self):
        return tabela(self.saida)

    def tabela_identificabilidade(self, por_modelo=False):
        return resumo_identificabilidade(self.saida) if por_modelo else identificabilidade(self.saida)

    def para_arviz(self, modelo, metodo):
        """Cadeias no formato do ArviZ (se o pacote arviz estiver instalado), ja sem o aquecimento."""
        import arviz as az

        c = self.cadeia(modelo, metodo)
        if c is None:
            raise FileNotFoundError(f"sem cadeias de {metodo} para {modelo}")
        q = int(c["aquecimento"]) // int(c["desbaste"])
        X = c["cadeia"][:, q:]
        if X.ndim == 4:
            X = X.reshape(X.shape[0], -1, X.shape[-1])
        dados = {str(n): X[:, :, j] for j, n in enumerate(c["nomes"])}
        try:
            return az.from_dict({"posterior": dados})       # ArviZ 1.x
        except TypeError:
            return az.from_dict(posterior=dados)            # ArviZ 0.x


def carregar(saida, destino=None):
    """Abre uma pasta de resultados ou um .zip criado por `exportar` (extraido em `destino`)."""
    saida = Path(saida)
    if saida.suffix == ".zip":
        import zipfile

        # nunca extrai por cima da pasta original, que costuma estar ao lado do .zip
        destino = Path(destino) if destino else saida.with_name(saida.stem + "_extraido")
        with zipfile.ZipFile(saida) as z:
            z.extractall(destino)
        internas = [p for p in destino.iterdir() if p.is_dir() and (p / "config.json").exists()]
        saida = internas[0] if internas and not (destino / "config.json").exists() else destino
    return Arquivo(saida)


def _arquivos(saida, com_cadeias=True, com_relatorio=True, com_banco=True):
    saida = Path(saida)
    for a in sorted(saida.rglob("*")):
        rel = a.relative_to(saida)
        if not a.is_file() or rel.parts[0].startswith("_") or a.name == "manifesto.json":
            continue
        if any(a.name.endswith(s) or s + "." in a.name for s in FORA_DO_ARQUIVO):
            continue
        if not com_cadeias and "cadeias" in rel.parts:
            continue
        if not com_relatorio and rel.parts[0] == "relatorio":
            continue
        if not com_banco and a.name == "banco.npz":
            continue
        yield a, rel


def manifesto(saida, **filtros):
    """Lista de arquivos com tamanho e sha256, gravada em manifesto.json: permite conferir a integridade."""
    from .util import gravar_json, sha256

    saida = Path(saida)
    itens = [dict(arquivo=rel.as_posix(), bytes=a.stat().st_size, sha256=sha256(a)) for a, rel in _arquivos(saida, **filtros)]
    m = dict(criado=__import__("time").strftime("%Y-%m-%d %H:%M:%S"), n_arquivos=len(itens),
             bytes=int(sum(i["bytes"] for i in itens)), arquivos=itens)
    gravar_json(saida / "manifesto.json", m)
    return m


def conferir(saida):
    """Compara os arquivos com o manifesto. Devolve a lista de problemas (vazia se estiver tudo igual)."""
    from .util import sha256

    saida = Path(saida)
    m = _ler(saida / "manifesto.json")
    if m is None:
        return ["manifesto.json não encontrado; rode 'exportar' primeiro"]
    problemas_ = []
    for i in m["arquivos"]:
        a = saida / i["arquivo"]
        if not a.exists():
            problemas_.append(f"falta {i['arquivo']}")
        elif sha256(a) != i["sha256"]:
            problemas_.append(f"alterado {i['arquivo']}")
    return problemas_


def exportar(saida, destino=None, com_cadeias=True, com_relatorio=True, com_banco=True, prog=None):
    """Grava o manifesto e junta resultados, cadeias, registros, ambiente e copia do codigo num unico .zip."""
    import zipfile

    saida = Path(saida)
    filtros = dict(com_cadeias=com_cadeias, com_relatorio=com_relatorio, com_banco=com_banco)
    m = manifesto(saida, **filtros)
    destino = Path(destino) if destino else saida.with_name(saida.name + ".zip")
    tmp = destino.with_name(destino.name + ".tmp")
    arqs = list(_arquivos(saida, **filtros)) + [(saida / "manifesto.json", Path("manifesto.json"))]
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for k, (a, rel) in enumerate(arqs):
            z.write(a, (Path(saida.name) / rel).as_posix())
            if prog:
                prog(k + 1, len(arqs))
    tmp.replace(destino)
    return destino, m
