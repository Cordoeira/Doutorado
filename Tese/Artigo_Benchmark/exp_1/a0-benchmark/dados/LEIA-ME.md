# Dados reais

Uma pasta por modelo. O código lê daqui (`a0bench/config.py`, `DIR_DADOS`; a variável `A0_DADOS` aponta para outra cópia).

| Pasta | Conteúdo |
|---|---|
| `lebre_lince/`, `gripe_internato/`, `gripe_internato_inicio/`, `pk_teofilina/`, `pk_indometacina/`, `fisher_kpp_jin/`, `fisher_kpp_jin_instante/`, `didinium_veilleux/`, `didinium_gause/`, `alfa_pineno/`, `hes1/`, `pendulo_mocap/`, `fhn_rato/`, `tracador_esgoto/` | CSV curado com as medidas publicadas (14 modelos numba). O `hes1` foi transcrito de artigo e não tem arquivo bruto. |
| As 16 pastas com nome `Autor_RevistaAno/` | Arquivos PEtab da coleção Benchmark-Models-PEtab: `measurements.tsv` (as medidas), `observables.tsv`, `conditions.tsv`, `parameters.tsv`, `model.xml`, `problem.yaml`. |
| `fontes.json` | Fonte, referência, URL, data, sha256 do arquivo bruto e desvio do ruído calibrado de cada conjunto dos modelos numba. |
| `_petab_sha256.json` | sha256 de cada arquivo PEtab copiado e a versão da coleção. |
| `_brutos/` | Arquivos como foram baixados, sem alteração. |

Regenerar os CSVs: `python -m a0bench.dados_reais.preparar` (na raiz do repositório).

Os `.rda` do R (Veilleux, Gause) são lidos com o pacote `rdata`; sem ele, `preparar` pula esses dois e os CSVs curados continuam valendo.
