# Análises

Scripts usados fora do pacote, na sessão de 13 a 16/11/2026. Rode no venv do WSL, na raiz do repositório.

| Script | Para quê |
|---|---|
| `analises_extras.py` | Tabelas discretizadas 02 a 15, testes de Friedman e Wilcoxon com Holm, piso de ruído do C2ST e `numeros.json` do relatório. `python analises/analises_extras.py <resultados_a0> <pasta_entrega>` |
| `figuras_extra.py` | As três figuras novas do relatório (robustez, NPE-PFN contra NPE, C2ST por classe). `python analises/figuras_extra.py <pasta_entrega>` |
| `build_pdf.py` | O PDF do relatório técnico, com o apêndice E (posteriores e cadeias de 9 modelos). Usa reportlab de `~/pdf_libs`. `python analises/build_pdf.py <pasta_entrega>` |
| `stats_artigo.py`, `pfn_resumo.py` | Estatísticas finais e resumo do piloto do NPE-PFN. |
| `npe_curva.py` | Curva de orçamento do NPE (1, 2, 5 e 10 mil simulações). |
| `checar_reais.py` | Ajuste de máxima verossimilhança, desvios de Laplace contra a caixa do prior e gráfico dos ajustes dos 6 modelos numba com dados reais (`docs/ajustes_dados_reais.png`). |
| `petab_novos.py` | Compila os PEtab novos e mede d, n, t_f, c_S e grad_ok (`docs/petab_novos_custos.jsonl`). |
| `estimar_reais.py`, `tempos_antigos.py` | Estimativa de tempo da trilha real, calibrada pelos tempos da execução anterior (`docs/estimativa_tempo_reais.*`). |

As análises de relatório supõem os 41 modelos da execução de 13 a 16/11; para a trilha de dados reais, filtre por `dados == "real"` em `relatorio/problemas.csv` antes de calcular estatísticas.
