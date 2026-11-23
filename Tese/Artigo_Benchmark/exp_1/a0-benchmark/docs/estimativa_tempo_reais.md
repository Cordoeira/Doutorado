# Estimativa de tempo da trilha de dados reais

Medida em 16/11/2026, no laptop da execução completa (12 núcleos lógicos, WSL2), por `analises/estimar_reais.py`. Valores por modelo em `estimativa_tempo_reais.csv` (minutos); custos dos PEtab novos em `petab_novos_custos.jsonl`.

## Método

- **Modelos numba.** Cada modelo novo tem um análogo sintético já rodado: lebre e lince e Lotka-Volterra; gripe e SEIR; gripe só no início e SEIR só no início; teofilina e PK com flip-flop; indometacina e PK de dois compartimentos; Fisher-KPP de Jin e Fisher-KPP. O custo por passo da referência é o custo medido agora de 12 avaliações (2 cadeias × 6 temperaturas) mais a sobrecarga do PT medida no análogo. Amostradores, soluções e identificabilidade escalam pela razão do custo de uma avaliação; o NPE fica igual ao do análogo.
- **PEtab.** Na execução anterior, cada avaliação custou L ≈ 3,7 vezes o tempo de uma solução medido com a máquina parada (mediana de Bertozzi, Boehm, Rahman e Crauste, com 10 processos simultâneos). Referência = passos × 12 × t_f × L; amostradores = 600 mil ESD × t_f × 4,8; identificabilidade pela fórmula do `estimar` do pacote, multiplicada por 4,8 (razão medido/previsto nos mesmos 4 modelos).
- Cada modelo roda num processo; o tempo de parede é o do modelo mais lento. A faixa vai de 50 mil passos de referência (mínimo) a 400 mil (máximo). Quando só sobram os modelos grandes, a carga cai e eles andam mais rápido; daí o limite inferior dos cenários C e D.

## Por modelo (minutos, com a carga de 10 processos)

| Modelo | d | Referência (50 mil a 400 mil) | Amostradores | NPE | Outros | Total |
|---|---|---|---|---|---|---|
| lebre_lince | 6 | 1 a 10 | 2 | 16 | 0,4 | 20 a 29 |
| gripe_internato | 4 | 1 a 9 | 2 | 10 | 0,3 | 13 a 21 |
| gripe_internato_inicio | 4 | 0,6 a 5 | 1,5 | 3 | 0,2 | 5 a 9 |
| pk_teofilina | 3 | 6 a 45 | 14 | 11 | 1 | 32 a 71 |
| pk_indometacina | 4 | 3 a 23 | 8 | 8 | 1 | 20 a 40 |
| fisher_kpp_jin | 2 | 4 a 35 | 24 | 7 | 2 | 37 a 68 |
| Perelson | 3 | 13 a 105 | 17 | 4 | 1 | 35 a 127 |
| Blasi | 9 | 29 a 235 | 38 | 4 | 19 | 90 a 296 |
| Liu | 9 | 28 a 222 | 36 | 4 | 7 | 75 a 269 |
| Oliveira | 12 | 35 a 276 | 45 | 5 | 13 | 97 a 338 |
| Armistead | 14 | 32 a 257 | 41 | 5 | 15 | 93 a 317 |
| Okuonghae | 16 | 30 a 243 | 39 | 4 | 15 | 89 a 301 |
| Laske | 13 | 370 a 2 959 | 478 | 10 | 465 | 1 323 a 3 912 |
| SalazarCavazos | 6 | 702 a 5 613 | 907 | 16 | 1 811 | 3 434 a 8 345 |

## Cenários (tempo de parede, 10 processos; estimativa de 16/11, antes dos modelos de 22/11, ver a atualização no fim)

| Cenário | Modelos novos | Tempo |
|---|---|---|
| A | 6 numba | 40 min a 1h15 |
| B | A + 6 PEtab baratos | 1h40 a 6 h |
| C | B + Laske | 12 a 22 h; até cerca de 2,7 dias |
| D | C + SalazarCavazos | 1 a 2,4 dias; até cerca de 6 dias |

Incerteza: os fatores de carga variaram de 1,4 a 37 entre modelos na execução anterior (há etapas infladas por suspensões do laptop); use as faixas como ordem de grandeza. O `python -m a0bench estimar` do pacote previu 9 h para o cenário D; ele não tem o fator de carga e subestimou os PEtab de 2 a 3 vezes no teste do preset rápido.

## Atualização de 22/11/2026: 14 modelos numba

Entraram mais 8 modelos numba com dados reais. Custo medido de uma solução (máquina parada): `fisher_kpp_jin_instante` 0,98 ms; `didinium_veilleux` 0,37 ms; `pendulo_mocap` 0,25 ms; `alfa_pineno` 0,23 ms; `didinium_gause` 0,13 ms; `hes1` 0,11 ms; `fhn_rato` 0,10 ms; `tracador_esgoto` 0,07 ms. Pelos mesmos fatores acima, os dois primeiros levam de 40 min a cerca de 2 h cada, e os demais ficam abaixo de 1 h. Com 10 processos, o cenário A (14 modelos numba) passa a 1 a 2h30 e o cenário B (20 modelos) a 2 a 7 h; C e D não mudam, porque são limitados pelo Laske e pelo SalazarCavazos.

No teste com orçamento pequeno (preset `rapido`), os 8 modelos rodaram de ponta a ponta sem erro. A referência do `didinium_veilleux` ficou longe de convergir (R-hat 1,20 com 4 mil passos): a posterior de um ciclo-limite mal ajustado tem muitos ótimos locais, e é possível que ela não convirja nem com 400 mil passos. Nesse caso o modelo sai marcado com referência não convergida, como o Elowitz.
