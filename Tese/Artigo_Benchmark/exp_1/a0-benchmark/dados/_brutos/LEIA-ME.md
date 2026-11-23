# Dados brutos

Arquivos como foram baixados em 16/11/2026, sem alteração. Fonte, URL, data e sha256 de cada um estão em `dados/fontes.json`; os CSVs curados que os modelos usam são gerados por:

```bash
python -m a0bench.dados_reais.preparar dados/_brutos         # CSVs curados e sha256
python -m a0bench.dados_reais.preparar dados/_brutos --sigma   # e recalibra o desvio do ruído
```

| Pasta | Arquivo | Origem |
|---|---|---|
| `lebre_lince/` | `hudson-bay-lynx-hare.csv` | stan-dev/example-models (Howard, 2009) |
| `gripe/` | `bsflu_data.txt` | curso SBIED de King et al. (BMJ, 1978) |
| `teofilina/` | `Theoph.csv` | Rdatasets (`datasets::Theoph` do R) |
| `indometacina/` | `Indometh.csv` | Rdatasets (`datasets::Indometh` do R) |
| `jin/` | `cell_density_profiles.npy` | repositório BINNs de Lagergren et al. (dados de Jin et al., 2016) |

O `.npy` de Jin et al. guarda um dicionário em pickle. `preparar.py` o lê com um unpickler restrito, que só aceita classes do numpy e tipos básicos; não o abra com `np.load(..., allow_pickle=True)`.

Os dados continuam sob as licenças e condições das fontes originais; cite os trabalhos listados em `fontes.json`.
