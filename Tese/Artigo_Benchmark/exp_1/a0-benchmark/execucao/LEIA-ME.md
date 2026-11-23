# Execução da trilha de dados reais

Tudo pronto para rodar o protocolo completo (`--preset reais`, sem o NPE-PFN) nos modelos com dados reais. Nada foi lançado ainda.

## Como rodar

O jeito mais simples, no Windows: dois cliques em `executar_reais.bat`, na raiz do repositório. Ele confere o ambiente, mostra o que vai rodar, pergunta e lança a execução completa (cenário D). Para outro cenário ou só conferir, num terminal aberto na raiz:

```bat
.\executar_reais.bat B          :: cenário B
.\executar_reais.bat /plano     :: só confere, não lança
.companhar_reais.bat          :: registro e painel da execução
```

O `.\` é necessário: neste computador o `cmd` não procura programas na pasta atual. Os `.bat` chamam os scripts abaixo.

No Windows (PowerShell, na raiz do repositório):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File execucao\lancar_reais.ps1 -Cenario B -Plano   # só confere o ambiente e mostra os comandos
powershell -NoProfile -ExecutionPolicy Bypass -File execucao\lancar_reais.ps1 -Cenario B          # lança
```

O lançador confere o plano, mantém o Windows acordado (`manter_acordado.ps1`, sem mudar configurações de energia) e roda `rodar_reais.sh` num `wsl.exe` desacoplado, que sobrevive ao fechamento do terminal. Dentro do WSL, o equivalente é `bash execucao/rodar_reais.sh B` (ou `--plano`).

Acompanhar: `python -m a0bench monitor --saida ~/a0bench_runs/resultados_reais` no venv. Registros em `~/a0bench_runs/reais_lancador.log` e `reais.log`; o fim é marcado por `~/a0bench_runs/REAIS.done`. Para retomar depois de uma interrupção, rode o mesmo comando: o que terminou é pulado.

## Cenários

Tempo de parede com 10 processos no laptop da execução anterior, calibrado pelos tempos medidos nela (ver `docs/estimativa_tempo_reais.md`):

| Cenário | Roda | Tempo |
|---|---|---|
| A | 14 modelos numba com dados reais | 1 a 2h30 |
| B | A + Perelson, Blasi, Liu, Oliveira, Armistead, Okuonghae | 2 a 7 h |
| C | B + Laske (em paralelo, 2 processos) | 12 a 22 h; até cerca de 2,7 dias |
| D (completo, padrão) | C + SalazarCavazos: os 30 modelos com dados reais | 1 a 2,4 dias; até cerca de 6 dias |

Com `A0_SEM_IDENT_GRANDES=1`, o Laske e o SalazarCavazos rodam sem identificabilidade: C cai para 8 a 14 h e D para 11 a 27 h. Some uns 30 min de relatório e exportação.

Os 8 PEtab já rodados na execução de 13 a 16/11 são copiados de `~/a0bench_runs/resultados_a0` e reaproveitados (custo zero). O Fujita não entra na lista de `rodar`: a identificabilidade dele recomeçaria, e levou mais de 25 h sem terminar. Seus resultados de amostragem entram no relatório.

## Variáveis

| Variável | Padrão | Uso |
|---|---|---|
| `A0_VENV` | `~/.venvs/a0bench` | venv do pacote |
| `A0_RUNS` | `~/a0bench_runs` | pasta das execuções |
| `A0_SAIDA` | `$A0_RUNS/resultados_reais` | saída desta execução |
| `A0_ANTIGOS` | `$A0_RUNS/resultados_a0` | execução anterior com os 8 PEtab |
| `A0_MODELOS_AMICI` | `$A0_ANTIGOS/_amici_models` | cache dos modelos compilados (os 8 PEtab novos já estão nele) |
| `A0_PROCESSOS` | 10 | processos em paralelo |
| `A0_SEM_IDENT_GRANDES` | 0 | 1 = Laske e SalazarCavazos sem identificabilidade |
| `A0_REFAZER_ANTIGOS` | 0 | 1 = sem a execução anterior, roda também os 7 PEtab antigos menores (mais 1 a 2 dias) |

## Fora destes números

- NPE-PFN: nos 6 modelos numba, 4 a 5,5 h a mais em CPU (3 sementes); nos PEtab, inviável sem GPU.
- Identificabilidade do Fujita (mais de 25 h, sem previsão) e a referência do Elowitz, que não convergiu em 400 mil passos.

`historico/` guarda os scripts da execução completa de 13 a 16/11 (fases, prazo das 7 h, instalação e certificado do proxy), com os caminhos trocados para relativos ao repositório.
