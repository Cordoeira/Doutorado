# Entrega: resultados preliminares do benchmark A0 (versão 3.2, preset completo)

Data: 16/11/2026. Leia primeiro `Relatorio_Tecnico_A0.pdf` (60 páginas; o apêndice E traz posteriores e cadeias de 9 modelos; o apêndice F, as fontes dos dados reais e a justificativa dos dados simulados). Esta pasta contém os dados que sustentam cada número do relatório.

## Estado dos resultados (importante)

- 41 modelos, 6 métodos com 3 sementes cada: completo.
- **NPE-PFN: apenas piloto** (semente 0, 18 modelos de 2 e 3 parâmetros, contexto de 1000 simulações em CPU). Não há NPE-PFN nos 8 PEtab nem nos 15 sintéticos de dimensão maior.
- **Identificabilidade do Fujita não concluída** (1 de 19 perfis após cerca de 25 h). Os 40 demais modelos estão completos.
- **Referência do Elowitz não convergiu** (R-hat 1,042; ESS 99). Os C2ST desse modelo são contra uma referência incerta.
- **Ressalva metodológica:** a referência é gerada pelo mesmo núcleo Metropolis adaptativo avaliado (com 6 temperaturas e cadeias mais longas). A vantagem do Metropolis pode estar inflada. Ver seção 6 do relatório.

## Conteúdo

| Caminho | O que é |
|---|---|
| `Relatorio_Tecnico_A0.pdf` | Relatório técnico (desenho, resultados, discussão, ameaças à validade, próximos passos, apêndices). |
| `dados/` | Tabelas em CSV (UTF-8, separador vírgula, ponto decimal) e arquivos de configuração. |
| `figuras/` | Figuras gerais do benchmark (PNG e PDF), incluindo o mapa de diagnósticos das cadeias (`diagnosticos_cadeias`), e as figuras novas do relatório (`fig_*`, entre elas `fig_tracos_multimodais`). |
| `figuras/modelos_selecionados/` | Figuras por modelo para os 7 modelos de falha total, o Boehm e o Lotka–Volterra: traços das cadeias (`cadeias_referencia`, `cadeias_metropolis`, `cadeias_ensemble`, `cadeias_mala`, `cadeias_hmc`), `autocorrelacao`, `convergencia`, marginais, pares, solução, identificabilidade e perfis. |
| `codigo/` | Scripts das análises adicionais, das figuras novas e do PDF. O código do benchmark está dentro do pacote completo. |
| `zip_sha256.txt` | Soma sha256 do pacote completo `resultados_a0_final.zip` (789 MB; não copiado aqui). |

O pacote completo (cadeias, amostras de referência, bancos de simulação, registros, código e as 657 figuras por modelo) está em `\\wsl.localhost\Ubuntu-24.04\home\pedrocordoeira\a0bench_runs\resultados_a0_final.zip`. `python -m a0bench conferir` valida a integridade e `a0bench.carregar` o reabre sem reexecutar.

## Dicionário de dados (`dados/`)

Convenções: `c2st` é a acurácia de um classificador que distingue as amostras do método das da referência (0,5 = indistinguível; 1,0 = distinguível). `esd` é o custo em equivalentes de solução direta. Classes: EDO, EDO difícil, EDP, EDP difícil, PEtab, PEtab grande. Métodos: laplace, metropolis, ensemble, mala, hmc, npe, npe_pfn.

| Arquivo | Linhas | Conteúdo e colunas principais |
|---|---|---|
| `01_modelos_e_referencias.csv` | 41 | Um modelo por linha: `d` (parâmetros), `n_obs`, `t_f_ms` (tempo de uma solução), `custo_grad` (c_S, custo de um gradiente em soluções), `map_origem`, diagnóstico da referência (`ref_rhat`, `ref_ess`, `ref_passos`, `ref_ok`), `desafio`, `classe`. |
| `02_resultados_por_semente.csv` | 756 | Uma linha por modelo, método e semente: `c2st`, `erro_media` (erro máximo da média em desvios-padrão da referência), `razao_dp` (razão geométrica dos desvios-padrão), `esd`, `ess_min`, `ess_por_mil_esd`, `rhat_max`, `aceitacao`, `passos`, `segundos`. |
| `03_resumo_modelo_metodo.csv` | 264 | Média, desvio-padrão, mínimo e máximo do C2ST entre sementes, mais médias das demais métricas, por modelo e método. |
| `04_c2st_modelo_x_metodo.csv` | 41 | Matriz modelo × método do C2ST médio (base da Tabela B1 e da Figura 1). |
| `05_resumo_classe_metodo.csv` | 36 | Quartis e mediana do C2ST, fração com C2ST < 0,6, e medianas de erro, razão de desvios e ESS por mil ESD, por classe e método (modelo = unidade). |
| `06_resumo_global_metodo.csv` | 6 | Resumo sobre os 41 modelos por método (Tabela 1). |
| `07_testes_pareados_metodos.csv` | 15 | Wilcoxon pareado entre métodos (modelo = unidade), com `p_holm`; contagem de modelos em que cada método vence por mais de 0,005. O teste de Friedman está em `numeros.json`. |
| `08_npe_pfn_piloto_semente0.csv` | 18 | C2ST de Metropolis, NPE e NPE-PFN, semente 0, nos 18 modelos do piloto. |
| `09_tempos_por_etapa_registro.csv` | — | Tempo de parede de cada etapa de cada modelo, extraído dos registros. Inclui pausas por suspensão do computador. |
| `10_tempo_mediano_etapa_classe.csv` | 6 | Mediana por classe e etapa (s), excluindo etapas acima de 20 mil s. |
| `11_identificabilidade_parametros.csv` | 207 | Um parâmetro por linha: `classe_id` (identificavel, parcial, nao identificavel), `natureza` (pratica ou estrutural), `multimodal`, `contracao`, limites do perfil, `compensado_por`. |
| `12_identificabilidade_por_modelo.csv` | 40 | Contagens I/P/N, `estruturais`, `multimodais`, posto do jacobiano, dimensões do subespaço informado, razões de predição. Sem o Fujita. |
| `13_intervalos_de_confianca.csv` | 207 | Intervalos assintóticos e pelo perfil, pontuais e simultâneos (nível 0,95). |
| `14_npe_curva_orcamento.csv` | 245 | C2ST do NPE com 1, 2, 5 e 10 mil simulações de treino por modelo (semente 0 nos três primeiros pontos). |
| `15_piso_de_ruido_c2st.csv` | 41 | C2ST da referência contra ela mesma (4 pares disjuntos de 1000 amostras): `c2st_nulo_media`, `c2st_nulo_max`, `c2st_nulo_dp`. |
| `protocolo_efetivo.json` | — | Configuração do preset completo. O `config.json` da pasta de resultados foi sobrescrito pelas últimas sessões e não deve ser usado como protocolo. |
| `ambiente.json` | — | Máquina, sistema, versões dos pacotes e hash do código. |

`numeros.json` (na raiz desta pasta) traz os números das análises adicionais usados no texto (teste de Friedman, postos médios, piso de ruído, desvios entre sementes, resumo do NPE-PFN).

## Reproduzir as análises adicionais

```bash
source ~/.venvs/a0bench/bin/activate
python codigo/analises_extras.py resultados_a0 entrega      # tabelas 02 a 15 e numeros.json (usa o sbi para o piso de ruído)
python codigo/figuras_extra.py entrega                      # figuras fig_*
python codigo/figura_tracos.py resultados_a0 entrega        # fig_tracos_multimodais (lê as cadeias gravadas)
python codigo/build_pdf.py entrega                          # relatório em PDF (requer reportlab)
```
