# A0 benchmark: resultados preliminares (preset completo, 41 modelos)

Estado em 15/11/2026, 20h. Os 41 modelos concluídos em todos os métodos exceto o NPE-PFN (em teste). Identificabilidade feita em 38 modelos; Sneyd, Elowitz e Fujita ficaram sem ela (relançados com `--sem-identificabilidade` por causa do custo do Fujita; pendente via `identificar`). Pacote exportado: `resultados_a0.zip` (786 MB, 2.796 arquivos, manifesto sha256). Arquivos citados abaixo estão em `resultados_a0/relatorio/`.

Ambiente: Ubuntu 24.04 em WSL2, Python 3.12.3, 12 núcleos, 17 GB, 10 processos, `requirements-lock.txt`. Preset completo: 3 cadeias por amostrador, 50 mil ESD por cadeia, referência de 50 mil passos dobrando até 400 mil, NPE com 10 mil simulações. Tempo de parede total cerca de 30 h de máquina acordada (fase 1: 4h37; PEtab grandes: o restante), fora 12 h de suspensões do notebook.

## 1. Referências

40 de 41 referências convergiram (R-hat < 1,05 e ESS >= 400). Passos: 50 mil em 35 modelos; 100 mil (Lorenz-63 longo); 200 mil (Cadeia de decaimento); 400 mil (Rahman, PK flip-flop, Sneyd, Elowitz). **Elowitz não convergiu nem com 400 mil passos (R-hat 1,04, ESS 99)**: seus C2ST são contra uma referência incerta e devem ser marcados no artigo. ESS mediano das referências: 4.875.

## 2. Amostradores com o mesmo orçamento (50 mil ESD por cadeia), 41 modelos

| Método | Q1 | Mediana | Q3 | Modelos com C2ST < 0,6 | Melhor por modelo (empate ±0,01) | Cadeias com R-hat > 1,05 |
|---|---|---|---|---|---|---|
| Metropolis adaptativo (PT) | 0,503 | 0,509 | 0,532 | 83% | 36 | 4 |
| Ensemble afim | 0,508 | 0,526 | 0,589 | 76% | 19 | 6 |
| MALA | 0,518 | 0,563 | 0,925 | 56% | 10 | 17 |
| HMC | 0,538 | 0,559 | 0,722 | 61% | 2 | 10 |
| Laplace (~660 ESD) | 0,513 | 0,588 | 0,867 | 59% | 12 | — |
| NPE (10 mil simulações) | 0,608 | 0,715 | 0,953 | 17% | 4 | — |

Mediana de C2ST por classe:

| Classe (n) | Laplace | Metropolis | Ensemble | MALA | HMC | NPE |
|---|---|---|---|---|---|---|
| EDO (11) | 0,54 | 0,51 | 0,51 | 0,52 | 0,55 | 0,75 |
| EDO difícil (10) | 0,70 | 0,51 | 0,58 | 0,73 | 0,69 | 0,88 |
| EDP (7) | 0,51 | 0,51 | 0,51 | 0,51 | 0,55 | 0,61 |
| EDP difícil (5) | 0,60 | 0,51 | 0,53 | 0,56 | 0,54 | 0,64 |
| PEtab (4) | 0,88 | 0,56 | 0,65 | 0,94 | 0,90 | 0,94 |
| PEtab grande (4) | 0,92 | 0,64 | 0,97 | 1,00 | 0,99 | 0,99 |

ESS mínimo por mil ESD (mediana): Metropolis 57 / 37 / 84 / 31 / 3,9 / 1,2 nas seis classes acima; MALA 35 / 0,5 / 77 / 2,9 / 0,12 / 0,06; HMC 12 / 1,4 / 18 / 2,5 / 0,19 / 0,06; Ensemble 13 / 4,0 / 20 / 5,2 / 3,3 / 4,0.

Leitura: com orçamento contado em soluções, o gradiente custa d+1 (sintéticos) ou c_S (PEtab) soluções e MALA e HMC ficam com poucos passos. Nos PEtab o ESS por mil ESD do Metropolis é 20 a 30 vezes o de MALA e HMC; nos PEtab grandes só o Metropolis fica abaixo de 0,7 (Bruno 0,51, Fujita 0,59, Sneyd 0,69, Elowitz 0,82). Figuras: `c2st_mapa.png`, `c2st_classe.png`, `c2st_custo.png`, `eficiencia.png`, `diagnosticos_cadeias.png`.

PEtab, C2ST por método:

| Modelo | d | Laplace | Metropolis | Ensemble | MALA | HMC | NPE |
|---|---|---|---|---|---|---|---|
| Boehm | 9 | 0,89 | 0,51 | 0,54 | 0,59 | 0,52 | 0,96 |
| Bertozzi | 8 | 0,85 | 0,53 | 0,57 | 0,92 | 0,79 | 0,92 |
| Rahman | 9 | 0,87 | 0,71 | 0,82 | 1,00 | 1,00 | 0,72 |
| Crauste | 12 | 0,95 | 0,59 | 0,72 | 0,96 | 1,00 | 0,99 |
| Bruno | 13 | 0,54 | 0,51 | 0,58 | 0,51 | 0,56 | 0,98 |
| Sneyd | 15 | 0,94 | 0,69 | 0,97 | 1,00 | 1,00 | 0,99 |
| Fujita | 19 | 0,90 | 0,59 | 0,97 | 1,00 | 0,99 | 1,00 |
| Elowitz (ref. não convergida) | 21 | 1,00 | 0,82 | 1,00 | 1,00 | 1,00 | 1,00 |

## 3. Casos em que todos os métodos falham (menor C2ST > 0,65)

| Modelo | Menor C2ST | Causa |
|---|---|---|
| Lorenz-96 | 1,00 | MAP em ótimo local; a referência achou −log L 1.937 menor. Amostradores partem do MAP. |
| Lorenz-63 (janela longa) | 0,97 | Idem, −log L 1.095 menor. |
| Cadeia de decaimento | 0,88 | Três modos por troca de rótulos; cada cadeia fica num modo. |
| Elowitz (PEtab) | 0,82 | Referência não convergida; 21 parâmetros. |
| PK com flip-flop | 0,71 | Dois modos (flip-flop). |
| Rahman (PEtab) | 0,71 | 8 de 9 parâmetros não identificáveis; referência precisou de 400 mil passos. |
| Sneyd (PEtab) | 0,69 | 15 parâmetros; referência de 400 mil passos. |

Os dois Lorenz devem ser relatados como falha de inicialização (limitação documentada do pacote para modelos caóticos), não de amostragem. Fujita: a otimização com vários inícios não chegou ao ponto nominal da coleção, o MAP parte dele; gradiente custa 50 a 80 soluções. Crauste: região inviável colada ao ótimo (20 de 48 perturbações de 0,2% falham).

## 4. Identificabilidade (perfil de verossimilhança, nível 0,95), 38 modelos

171 parâmetros: 99 identificáveis, 20 parciais, 52 não identificáveis (27 estruturais, detectados pelo posto do Fisher).

| Classe | Parâmetros | Identificáveis | Parciais | Não identificáveis | Estruturais |
|---|---|---|---|---|---|
| EDO | 35 | 35 | 0 | 0 | 0 |
| EDO difícil | 33 | 19 | 10 | 4 | 2 |
| EDP | 18 | 13 | 4 | 1 | 0 |
| EDP difícil | 34 | 5 | 3 | 26 | 16 |
| PEtab | 38 | 14 | 3 | 21 | 9 |
| PEtab grande (só Bruno) | 13 | 13 | 0 | 0 | 0 |

Darcy com 16 coeficientes: posto 7 com 7 saídas, 15 não identificáveis. Boehm: 7 de 9 identificáveis, 1 estrutural, coerente com a literatura. Arquivos: `identificabilidade.csv`, `identificabilidade_por_modelo.csv`, `intervalos_de_confianca.csv`, `identificabilidade.png`. Faltam Sneyd, Elowitz e Fujita.

## 5. NPE: curva de orçamento (análise adicional, `npe_curva.py`)

Treino com os primeiros 1, 2 e 5 mil pontos do `banco.npz` de cada modelo (semente 0) e o ponto de 10 mil do benchmark (média de 3 sementes). C2ST mediano:

| Classe | 1 mil | 2 mil | 5 mil | 10 mil |
|---|---|---|---|---|
| EDP | 0,76 | 0,71 | 0,60 | 0,61 |
| EDP difícil | 0,82 | 0,73 | 0,66 | 0,63 |
| EDO | 0,91 | 0,89 | 0,83 | 0,75 |
| EDO difícil | 0,96 | 0,94 | 0,93 | 0,88 |
| PEtab | 0,97 | 0,96 | 0,94 | 0,94 |
| PEtab grande | 1,00 | 0,99 | 0,99 | 0,99 |

Leitura: nas EDPs o NPE satura em 5 mil simulações (limite do estimador, não do orçamento); nas EDOs ainda cai em 10 mil (um ponto de 20 a 50 mil seria informativo e barato); nos PEtab está estagnado desde 1 mil (precisa de outra arquitetura ou estatísticas sumárias). Ressalva: uma semente nos pontos de 1, 2 e 5 mil. Figura: `npe_curva.png`; dados: `npe_curva.csv`.

## 6. Custo computacional (mediana de tempo de parede por etapa, segundos)

| Classe | Referência | Cada amostrador | Identificabilidade | NPE (treino + amostragem) |
|---|---|---|---|---|
| EDO | 104 | 38 a 51 | 3 | 571 |
| EDO difícil | 269 | 48 a 63 | 6 | 488 |
| EDP | 84 | 37 a 77 | 2 | 360 |
| EDP difícil | 66 | 35 a 49 | 12 | 496 |
| PEtab | 1.048 | 296 a 410 | 284 | 384 |
| PEtab grande | 2.462 (até 400 mil passos: horas) | 641 a 818 | 485 (Bruno) | 197 |

Custo por solução: 0,02 a 0,4 ms nos sintéticos (numba), 1,6 a 2,7 ms nos PEtab (AMICI). Nos sintéticos o NPE foi a etapa mais cara em tempo de parede, apesar de usar 15 vezes menos simulações que as três cadeias de um amostrador. Para a discussão sobre aceleração por rede, o eixo ESD precisa ser complementado por tempo de parede e por avaliação amortizada (várias observações por rede treinada).

## 7. Pendências antes da versão final

- NPE-PFN: `verificar --pfn` em execução; se passar, `rodar --metodos npe_pfn` nos 41 modelos e novo relatório/exportação (automático).
- Identificabilidade de Sneyd e Elowitz (`identificar --modelos Sneyd_PNAS2002,Elowitz_Nature2000`, barata) e do Fujita (horas a dias; em segundo plano).
- Curva do NPE com 3 sementes e ponto de 20 ou 50 mil simulações nas EDOs.
- Reiniciar Lorenz-63 longo e Lorenz-96 com o MAP tomado da referência.
- Elowitz: referência além de 400 mil passos ou relatar como não convergida.
