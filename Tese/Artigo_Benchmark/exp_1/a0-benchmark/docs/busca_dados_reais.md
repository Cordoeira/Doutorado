# Busca de dados reais para os modelos sintéticos

Estado em 22/11/2026. Dos 33 modelos sintéticos, 15 têm agora um modelo equivalente com dados reais no repositório e 18 não têm. As referências completas estão em `dados/fontes.json` e no apêndice F do relatório (`entrega_orientador/Relatorio_Tecnico_A0.pdf`).

## Modelos com dados reais (14 numba, em `dados/<modelo>/`)

| Modelo real | Substitui o sintético | Dados | Conferência do ajuste |
|---|---|---|---|
| `lebre_lince` | Lotka-Volterra | Peles da Hudson's Bay Company, 1900 a 1920 | Reproduz as taxas do estudo de caso do Stan. |
| `gripe_internato`, `gripe_internato_inicio` | SEIR; SEIR só no início | Gripe num internato inglês, 1978 | Resíduos padronizados com desvio 0,9. |
| `pk_teofilina` | PK com flip-flop | Teofilina oral, sujeito 1 | Os dois modos (troca de k_a e k_e) ficam dentro do prior. |
| `pk_indometacina` | PK de dois compartimentos | Indometacina intravenosa, sujeito 1 | Ajuste bom. |
| `fisher_kpp_jin`, `fisher_kpp_jin_instante` | Fisher-KPP; Fisher-KPP num instante | Ensaio de arranhão de Jin et al. (2016) | No de um instante, correlação de −0,98 entre proliferação e capacidade. |
| `didinium_veilleux` | Rosenzweig-MacArthur | Paramecium e Didinium, 71 instantes em 35 dias (Veilleux, 1976) | **Fraco**: R² de 0,32 e 0,45. O modelo acerta o período, não a amplitude. Rótulos trocados no pacote TSA, corrigidos no CSV. |
| `didinium_gause` | (predador-presa adicional) | Paramecium e Didinium com imigração, 17 dias (Gause, 1934) | R² de 0,74 e 0,89. A imigração é um pulso suave, com quantidade livre. |
| `alfa_pineno` | Cadeia de decaimento | Isomerização do α-pineno, 5 espécies em 8 instantes | Reproduz as constantes de Box et al. (1973): 5,9; 3,0; 2,0; 27,4 e 4,0 × 10⁻⁵ por min. |
| `hes1` | Goodwin (duas versões) | mRNA de Hes1, 8 medidas (Silk et al., 2011) | R² de 0,74. Valores transcritos de fonte secundária, sem conferência na primária. |
| `pendulo_mocap` | Pêndulo amortecido | Ângulo por captura de movimento, 130 instantes (Giessler, 2025) | R² de 1,00; desvio de 0,005 rad. |
| `fhn_rato` | FitzHugh-Nagumo | Média de 64 disparos de um registro de rato (Metcalfe, 2020) | R² de 0,93. O parâmetro b fica na borda do prior. O dado é uma média de disparos, não o registro bruto. |
| `tracador_esgoto` | Advecção-difusão | Traçador NaCl em 469 m de esgoto (Rieckermann et al., 2005) | Reproduz o artigo: v = 0,38 m/s, D = 0,069 m²/s. R² de 0,99. |

O repressilador sintético já estava coberto pelo problema PEtab de Elowitz e Leibler (2000).

## Encontrados, mas não obtidos

| Modelo sintético | Candidato | Motivo |
|---|---|---|
| Van der Pol | Circuito oscilador forçado, [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.z34tmpg80) (CSV de 32 MB, licença CC0) | O download automático devolveu 401 e 403 (exige autenticação). Baixando o arquivo pelo navegador, dá para montar o modelo. |
| Calor com fonte; calor inverso | Barra aquecida em túnel de vento, [Tomanek e Stutts (2022)](https://core.ac.uk/download/524635783.pdf) | O acesso ao artigo foi recusado (403) e o repositório dos dados não foi localizado. |
| Brusselator | Experimentos de Belousov-Zhabotinsky ([arXiv 2304.04818](https://arxiv.org/pdf/2304.04818)) | Sem série temporal pública pronta. |

## Sem dado real

| Modelo sintético | Motivo |
|---|---|
| Oscilador forçado | O conjunto do pêndulo é de oscilação livre; não se achou medida com forçamento conhecido. |
| Lorenz-63 (duas versões) | Só conjuntos simulados. A [roda d'água de Malkus](https://arxiv.org/pdf/1202.5508) tem experimentos em artigos, sem dados abertos. |
| Burgers, Allen-Cahn (duas versões), onda, FitzHugh-Nagumo com difusão | A busca só achou conjuntos gerados por simulação ([exemplo](https://service.tib.eu/ldmservice/dataset/burgers-equation)). |
| Darcy (duas versões) | Os próprios benchmarks da área são sintéticos ([exemplo](https://researchportal.bath.ac.uk/en/publications/towards-a-community-wide-effort-for-benchmarking-in-subsurface-hy/)). |
| Lorenz-96, Robertson, HIRES, decaimento com escala | Sistemas conceituais ou de teste numérico, sem experimento de referência. |

## Descartados depois de conferidos

- **Registro bruto do rato como série temporal.** São de 110 a 170 disparos irregulares por canal; uma EDO determinística com ruído de medida não os descreve. O modelo `fhn_rato` usa a média dos disparos alinhados.
- **Outros conjuntos do Zenodo de traçador.** Há 56 experimentos; usa-se só o experimento 2 (curva a montante completa e a de jusante com passo de 15 s).

## Endereços dos dados baixados

[Stan (lebre e lince)](https://raw.githubusercontent.com/stan-dev/example-models/master/knitr/lotka-volterra/hudson-bay-lynx-hare.csv), [bsflu](https://kingaa.github.io/sbied/stochsim/bsflu_data.txt), [Theoph](https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Theoph.csv), [Indometh](https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Indometh.csv), [Jin et al. (BINNs)](https://raw.githubusercontent.com/jlager/BINNs/master/Data/cell_density_profiles.npy), [TSA veilleux](https://raw.githubusercontent.com/cran/TSA/master/data/veilleux.rda), [gauseR](https://raw.githubusercontent.com/cran/gauseR/master/data/gause_1934_book_f32.rda), [COPS pinene](https://raw.githubusercontent.com/MadNLP/COPSBenchmark.jl/main/ext/COPSBenchmarkJuMP/pinene.jl), [Hes1 (Schmitt et al.)](https://arxiv.org/abs/2310.04395), [pêndulo (Figshare)](https://figshare.com/articles/dataset/Kinematic_data_of_a_pendulum_serving_as_an_angular_acceleration_reference/28271405), [rato (Mendeley)](https://data.mendeley.com/datasets/ybhwtngzmm/1), [traçador (Zenodo)](https://zenodo.org/records/20831873).
