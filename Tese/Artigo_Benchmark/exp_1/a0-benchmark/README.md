# a0bench: projeto completo do benchmark A0

Benchmark de inferência bayesiana de parâmetros em EDOs e EDPs. Roda 63 modelos (30 com dados reais, 33 sintéticos) com sete métodos de inferência, compara cada um a uma posterior de referência, analisa a identificabilidade por quatro métodos e desenha tudo a partir dos arquivos gravados. A simulação salva o estado a cada n passos, retoma de onde parou e, ao fim, deixa resultados e histórico num formato que pode ser reaberto depois.

## Estrutura do projeto

```
a0-benchmark/
├── README.md              este documento
├── requirements.txt       dependências diretas com versão fixa
├── requirements-lock.txt  o ambiente inteiro do teste (125 pacotes; Linux x86_64, Python 3.12)
├── pyproject.toml         permite `pip install -e .` e o comando `a0bench`
├── executar_reais.bat     lança a trilha de dados reais no WSL (dois cliques); acompanhar_reais.bat mostra o andamento
├── execucao/              lançador da trilha de dados reais (lancar_reais.ps1, rodar_reais.sh) e histórico
├── analises/              scripts das análises e do PDF do relatório, preparação dos dados reais, estimativa de tempo
├── dados/                 OS DADOS REAIS: uma pasta por modelo (CSV ou arquivos PEtab), fontes.json, _brutos/
├── docs/                  resultados preliminares, estimativa de tempo, ajustes dos modelos reais
├── entrega_orientador/    relatório técnico em PDF, dados discretizados e figuras enviados ao orientador
└── a0bench/
    ├── modelos.py             OS MODELOS: 33 sintéticos e 14 com dados reais (equações, prior, dados, simuladores)
    ├── dados_reais/           preparar.py: regenera dados/ a partir de dados/_brutos e calibra o desvio
    ├── cadeias.py             AS CADEIAS: amostradores, gravação a cada n passos, retomada, diagnósticos
    ├── graficos.py            OS GRÁFICOS: todas as figuras, feitas só com os arquivos gravados
    ├── problemas.py           posterior, gradiente e soluções (modelos sintéticos e problemas PEtab)
    ├── inferencia.py          MAP, referência, soluções e superfícies, NPE, NPE-PFN, métricas
    ├── identificabilidade.py  perfil de verossimilhança, estrutura local, subespaço informado, predição
    ├── executor.py            processos em paralelo, ordem das etapas, retomada, arquivo do processo
    ├── monitor.py             painel de terminal
    ├── resultados.py          tabelas, leitura dos resultados (`carregar`), `exportar` e `conferir`
    ├── config.py              presets e todos os parâmetros
    ├── util.py                escrita atômica, registro de eventos, trava, descrição do ambiente
    ├── verificar.py           confere a instalação na máquina em que vai rodar
    └── __main__.py            linha de comando
```

Os três arquivos pedidos como módulos independentes:

| Arquivo | Conteúdo | Para mexer |
|---|---|---|
| `modelos.py` | Lista `MODELOS`. Cada modelo é um dicionário com nome, tipo (EDO ou EDP), grupo, nomes dos parâmetros, limites do prior, valor verdadeiro e três simuladores: `sim` (saídas observadas), `pred` (predição fora da janela) e `traj` (todos os estados numa malha fina). | Escreva `f(y, th, dy)` e a condição inicial com `@njit` e registre com `_m(...)`. |
| `cadeias.py` | `amostrar_pt` (Metropolis adaptativo e temperação paralela), `amostrar_mala`, `amostrar_hmc`, `amostrar_ensemble`, `amostrar_laplace`; a classe `Gravador` (cadeia em disco e estado salvo a cada n passos); `ess`, `rhat`, `autocorrelacao`. | Um amostrador novo cria o estado em `criar()`, chama `g.registrar(x, logp)` a cada passo e `g.talvez_salvar(n)`. |
| `graficos.py` | Uma função por figura, mais `relatorio_geral` e `relatorio_modelo`. Nenhuma roda simulação. | Uma figura nova lê os arquivos com `_npz`, `_json`, `_ref`, `_cadeia` e entra na lista `POR_MODELO`. |

## Instalação

Testado no Ubuntu 24.04 (x86_64) com Python 3.12. O que a máquina precisa ter:

| Item | Para quê | No Ubuntu |
|---|---|---|
| Python 3.12 com `venv` e cabeçalhos | rodar o pacote; o AMICI compila os modelos PEtab | `sudo apt install python3.12 python3.12-venv python3.12-dev` |
| Compilador C++ (`g++`) | compilar os modelos PEtab | `sudo apt install build-essential` |
| `git` | baixar a coleção de modelos PEtab na instalação | `sudo apt install git` |
| LaTeX com `cm-super` e `dvipng` (opcional) | fonte Computer Modern nas figuras; sem ele saem em STIX | `sudo apt install texlive-latex-extra texlive-fonts-recommended cm-super dvipng` |

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # cerca de 6,5 GB (o torch traz as bibliotecas de GPU)
pip install -e .                    # opcional: cria o comando `a0bench`
python -m a0bench verificar         # confere a instalação nesta máquina (1 a 3 minutos)
```

- `cmake`, `ninja` e `swig` vêm pelo pip, dentro do venv. Por isso o venv precisa estar ativado sempre que o pacote rodar.
- `requirements.txt` fixa as 14 dependências diretas. `requirements-lock.txt` fixa os 125 pacotes do ambiente em que o pacote foi testado; use-o no lugar do primeiro para repetir exatamente esse ambiente (só Linux x86_64 com Python 3.12).
- As dependências também resolvem em Python 3.13 e 3.14, mas o pacote só foi executado no 3.12.
- macOS não foi testado. No Windows, use o WSL2.

### Conferir a instalação

`python -m a0bench verificar` roda um pouco de cada parte e mostra uma tabela com ✔, ⚠ ou ✖ por etapa:

| Etapa | O que faz |
|---|---|
| Python e sistema | Versão, núcleos e memória; avisa se a memória não dá para um processo por núcleo. |
| Pacotes | Compara as versões instaladas com as do `requirements.txt`. |
| Simuladores | Compila e roda três modelos (não rígido, rígido e EDP). |
| Pausa e retomada | Roda cada amostrador de uma vez e com uma pausa forçada; as cadeias têm de sair idênticas. |
| Ponta a ponta | Dois modelos com todas as etapas e seis métodos, com orçamento mínimo. |
| Figuras e tabelas | Desenha todas as figuras dessa execução, com LaTeX se houver. |
| Exportar, conferir e reabrir | Gera o `.zip`, confere os sha256 e reabre com `carregar`. |
| PEtab e AMICI | Compila o Boehm e compara −log L no ponto nominal com 138,2220. `--sem-petab` pula. |
| NPE-PFN | Só com `--pfn`: baixa os pesos do TabPFN (precisa de internet), amostra uma posterior no tamanho usado no preset completo e mostra quanto tempo levou. |

Rode `verificar --pfn` uma vez antes da execução completa: o NPE-PFN é a única parte que não pôde ser executada no teste do pacote (ver Limitações).

## Uso

Rode na pasta que contém `a0bench/`, com o venv ativado.

```bash
python -m a0bench verificar                                # confere a instalação (ver acima)
python -m a0bench listar                                   # os 63 modelos, seus grupos e a origem dos dados
python -m a0bench estimar --processos 8                    # tempo previsto nesta máquina
python -m a0bench rodar --preset rapido --sem-pfn --saida teste   # os 37 modelos menores com orçamento pequeno (40 min em 2 núcleos)
python -m a0bench rodar --saida resultados_a0              # execução completa, com painel ao vivo
python -m a0bench identificar --saida resultados_a0        # só MAP, referência, identificabilidade e soluções
python -m a0bench monitor --saida resultados_a0            # acompanha de outro terminal
python -m a0bench relatorio --saida resultados_a0 --processos 8   # tabelas e figuras
python -m a0bench exportar --saida resultados_a0           # tudo num .zip com manifesto
python -m a0bench conferir --saida resultados_a0           # confere os arquivos com o manifesto
```

A trilha de dados reais (23 modelos) tem lançador próprio, com cenários e tempos: veja `execucao/LEIA-ME.md`.

### Roteiro para a primeira execução

1. `python -m a0bench verificar`, até sair "Instalação pronta para rodar o benchmark".
2. `python -m a0bench estimar --sem-pfn --processos N`, para saber o tempo nesta máquina.
3. A execução completa, sem o NPE-PFN, de um jeito que sobreviva ao fechamento do terminal. Comece pelos 37 modelos menores; os 4 PEtab maiores levam mais que todos os outros juntos e entram depois, na mesma pasta:

   ```bash
   nohup python -m a0bench rodar --saida resultados_a0 --sem-pfn --sem-monitor --grupos facil,dificil,petab > a0.log 2>&1 &
   python -m a0bench monitor --saida resultados_a0
   # depois de terminar:
   nohup python -m a0bench rodar --saida resultados_a0 --sem-pfn --sem-monitor --grupos petab_grande > a0_grande.log 2>&1 &
   ```

4. O NPE-PFN depois, na mesma pasta: `python -m a0bench verificar --pfn` mostra se os pesos do TabPFN baixam e quanto tempo leva uma semente; `python -m a0bench rodar --saida resultados_a0 --metodos npe_pfn` calcula só esse método, com o banco de simulações que já está gravado.
5. `python -m a0bench relatorio --saida resultados_a0 --processos N` e `python -m a0bench exportar --saida resultados_a0`.

O NPE-PFN fica para depois porque é a única parte sem teste de execução e sem tempo medido (ver Limitações).

Opções de `rodar`, `identificar` e `estimar`:

| Opção | Efeito |
|---|---|
| `--preset completo\|reais\|padrao\|rapido` | Orçamentos (tabela abaixo). `reais` é o `completo` só nos modelos com dados reais. |
| `--dados reais\|sinteticos\|todos` | Só os modelos com dados reais (PEtab e os de `reais()` em `modelos.py`), só os sintéticos, ou todos. Vale quando `--modelos` não é dado. |
| `--processos N` | Modelos em paralelo, um núcleo cada. Padrão: todos os núcleos. Reserve 1,5 GB de memória por processo (no preset rápido cada um chegou a 1,0 GB). |
| `--pasta-modelos PASTA` | Onde ficam os modelos PEtab compilados. Padrão: `<saida>/_amici_models`. Aponte várias execuções para a mesma pasta para compilar uma vez só. |
| `--salvar-a-cada N` | Grava a cadeia e o estado do amostrador a cada N passos. Padrão: 1000. |
| `--desbaste K` | Guarda 1 a cada K passos das cadeias completas. Padrão: 1 (tudo). |
| `--nivel 0.95\|1sigma\|2sigma` | Nível dos intervalos de confiança da identificabilidade. Padrão: 0,95. `1sigma` (0,683) reproduz os limiares da figura 3 de Raue et al. (2009). |
| `--grupos facil,dificil,petab,petab_grande` | Restringe aos grupos. |
| `--modelos robertson,darcy16` | Restringe a modelos específicos. |
| `--metodos laplace,metropolis,...` | Restringe os métodos. |
| `--sementes 0,1,2` | Uma cadeia por semente. |
| `--sem-pfn` | Não roda o NPE-PFN (precisa baixar os pesos do TabPFN). |
| `--sem-identificabilidade` | Só em `rodar`: pula a identificabilidade, que pode ser feita depois com `identificar` na mesma pasta. Útil nos modelos em que ela é cara (Fujita). |
| `--sem-monitor` | Mensagens de log em vez do painel (útil com `nohup` ou em cluster). |

| Preset | Cadeias | Referência (passos) | Orçamento por cadeia | Simulações do NPE |
|---|---|---|---|---|
| `completo` | 3 | 50 mil, dobrando até 400 mil | 50 mil ESD | 10 mil |
| `reais` | 3 | 50 mil, dobrando até 400 mil | 50 mil ESD | 10 mil (só modelos com dados reais) |
| `padrao` | 2 | 50 mil, dobrando até 200 mil | 50 mil ESD | 5 mil |
| `rapido` | 2 | 4 mil | 6 mil ESD | 1,5 mil |

`estimar` mede a velocidade dos simuladores sintéticos nesta máquina; os tempos dos problemas PEtab vêm de uma tabela medida numa máquina de referência e são os menos precisos. No teste do preset rápido, o tempo medido por modelo ficou entre 0,3 e 3 vezes o previsto (mediana 0,9); os PEtab e o Darcy com 16 coeficientes levaram de 2 a 3 vezes o previsto. Trate a estimativa como ordem de grandeza.

## Pausa e retomada

Para pausar, use Ctrl+C ou encerre o processo. Para continuar, rode o mesmo comando: as etapas concluídas são puladas e a cadeia que estava no meio continua do último estado salvo.

- A cada `--salvar-a-cada N` passos, as linhas novas da cadeia vão para um arquivo `.npy` mapeado em memória e o estado do amostrador (posições, adaptação e gerador aleatório) é trocado de forma atômica. Perde-se no máximo N passos.
- A cadeia retomada dá exatamente os mesmos números que daria sem a pausa (conferido bit a bit nos quatro amostradores e na referência).
- Na identificabilidade, cada perfil pronto vai para o disco; depois de uma pausa a etapa continua do perfil seguinte, com o mesmo resultado. As outras etapas (MAP, soluções, treino do NPE) recomeçam do início da etapa; são curtas.
- Um estado salvo só é reaproveitado se o modelo, os dados e os ajustes do amostrador forem os mesmos; senão a cadeia recomeça.
- Cada retomada fica anotada em `<modelo>/registro.jsonl` e no painel.
- Se o processo principal morrer (`kill` ou `kill -9`), os processos de trabalho saem sozinhos em poucos segundos.
- Ao levar a pasta para outra máquina, as etapas concluídas continuam valendo, mas uma cadeia que estava no meio pode recomeçar: o estado salvo só vale para os mesmos dados sintéticos, que podem diferir no último bit entre processadores.

## O que é calculado

Para cada modelo, nesta ordem:

1. **MAP** por otimização com vários inícios, e a informação de Fisher. Nos PEtab, o ponto nominal da coleção (o ajuste publicado, centro da caixa do prior) também concorre.
2. **Referência** por temperação paralela (6 temperaturas, 2 execuções). O comprimento dobra até R-hat < 1,05 e ESS ≥ 400, ou até `n_ref_max`.
3. **Identificabilidade** por quatro métodos (seção abaixo).
4. **Soluções e superfícies** para os gráficos: saídas no formato dos dados, todos os estados numa malha fina e fatias do log da posterior.
5. **Laplace**.
6. **Quatro amostradores com o mesmo orçamento de ESD:** Metropolis adaptativo, ensemble invariante afim, MALA e HMC.
7. **NPE** e **NPE-PFN**, sobre o mesmo banco de simulações.

Métricas: C2ST contra a referência (0,5 é o ideal), erro da média, razão de desvios-padrão, R-hat entre cadeias e ESS mínimo por mil ESD. Um ESD é o custo de uma solução do modelo; um gradiente custa c_S ESD nos PEtab e d + 1 nos sintéticos. c_S é a mediana do tempo de um gradiente sobre a mediana do tempo de uma solução, medidas no ponto nominal e em 8 pontos sorteados do prior.

Nos PEtab, quando a solução existe mas as sensibilidades falham com a tolerância padrão do AMICI (absoluta 1e-16), o gradiente é tentado de novo com tolerância absoluta 1e-12 e depois 1e-10. Nos modelos em que a tolerância padrão funciona, nada muda. `problema.json` guarda em `grad_ok` a fração dos pontos de teste em que o gradiente saiu, e o painel avisa quando ela fica abaixo de 80%.

## Gráficos

`relatorio` grava as figuras gerais em `relatorio/` (PDF e PNG) e as de cada modelo em `relatorio/modelos/<modelo>/` (PNG; `--pdf` grava também em PDF). `--sem-latex` é bem mais rápido; `--so geral` ou `--so modelos` e `--modelos a,b` restringem.

| Tema | Figura | O que mostra |
|---|---|---|
| Cadeias de Markov | `cadeias_<método>` | Traço de cada parâmetro (uma cor por cadeia, aquecimento sombreado), densidade de cada cadeia contra a referência, traço do log da posterior, R-hat, ESS e aceitação. |
| | `autocorrelacao` | Autocorrelação por parâmetro, um painel por amostrador. |
| | `convergencia` | Erro da média acumulada contra o custo em ESD, uma linha por cadeia. |
| | `diagnosticos_cadeias` (geral) | Mapas de calor de R-hat, ESS por mil ESD e aceitação, por modelo e amostrador. |
| Espaço de solução | `solucao` | Dados com barra de erro, solução no ponto central e faixa de 95% a posteriori. |
| | `estados` | EDOs: todos os estados no tempo e o espaço de fase, com trajetórias da posterior. |
| | `campo` | EDPs: mapa de calor da solução no plano x–t, largura da faixa de 95% e pontos de medida. |
| Posteriores | `marginais` | Marginal de cada parâmetro: referência e cada método. |
| | `pares` | Conjuntas dois a dois: regiões de 50% e 90% da referência e o contorno de 90% de cada método. |
| | `correlacao` | Mapa de calor da correlação a posteriori. |
| Contornos | `contornos` | Regiões de 50, 90 e 99% da posterior com os primeiros passos de uma cadeia de cada amostrador. |
| | `superficie` | Fatia exata do log da posterior em dois parâmetros, contra a região de 90% da conjunta. |
| Identificabilidade | `identificabilidade` (por modelo) | Painel com quatro leituras: intervalos do perfil contra os da posterior, contração, espectros, composição das direções. |
| | `perfis` | Perfis no formato da figura 3 de Raue et al. (2009): χ²_PL contra log10 do parâmetro, aproximação quadrática no ótimo, valor calibrado e limiares pontual e simultâneo. |
| | `caminhos` | Trajetórias dos demais parâmetros ao longo do perfil de cada parâmetro não identificável. |
| | `identificabilidade`, `dimensoes`, `espectros`, `predicao` (gerais) | Classe de cada parâmetro, dimensões informadas por critério, espectros e contribuição para a predição. |
| Desempenho | `c2st_mapa`, `c2st_classe`, `c2st_custo`, `eficiencia` (gerais) | C2ST por modelo e método, por classe de modelo, contra o custo; ESS por mil ESD. |

Nos problemas PEtab não há `estados` nem `campo` (só os observáveis em `solucao`).

## Identificabilidade

### Os quatro métodos

| Método | O que responde | Referência |
|---|---|---|
| Perfil de verossimilhança | Cada parâmetro é limitado pelos dados? De um lado, dos dois, de nenhum? Há mais de um modo? Quem compensa quem? | Raue et al. (2009) |
| Estrutura local | Quantas direções as saídas distinguem (posto do jacobiano), e quais combinações de parâmetros são identificáveis | Stigter e Molenaar (2015); Maclaren et al. (2025) |
| Subespaço informado | Quantas direções a posterior precisa para ficar a menos de ε nats da posterior completa | Zahm et al. (2022); Cui, Tong e Zahm (2022); Cui e Zahm (2021) |
| Predição perfil a perfil | A incerteza dos parâmetros chega às predições? Por qual parâmetro? | Simpson e Maclaren (2023) |

Além desses, saem a contração da posterior e o espectro de Fisher no ótimo.

### Perfil de verossimilhança e intervalos de confiança (Raue et al., 2009)

O perfil do parâmetro θ_i é χ²_PL(θ_i) = min sobre os demais parâmetros de χ²(θ), com χ² = −2 log L. Nos modelos sintéticos (ruído gaussiano de desvio conhecido) isso é a soma dos resíduos ponderados ao quadrado; nos PEtab inclui os termos dos desvios estimados. A varredura segue o artigo: passo adaptativo a partir do ótimo, com reotimização dos demais parâmetros a cada passo. Somam-se uma varredura global de dentro para fora, iniciada também nas amostras da referência, para achar outros modos, e um polimento: cada ponto abaixo de 1,2 vez o limiar simultâneo é tentado de novo a partir do ótimo e dos dois vizinhos (`perfil_polimento` em `config.py`; desligado no preset rápido). O método supõe que os perfis partem do ótimo global: se um perfil acha um ponto melhor que o ótimo por mais de 0,05 em −log L, todos são refeitos a partir dele, até duas vezes, e o painel avisa.

Saem os quatro intervalos da seção 2.2 do artigo, no nível `--nivel`:

| Intervalo | Definição | Limiar |
|---|---|---|
| Assintótico pontual | θ̂_i ± √(Δ_α · C_ii), C = 2 H⁻¹, H = Hessiana de χ² no ótimo (eq. 7) | Δ_α = χ²(α, 1) |
| Assintótico simultâneo | idem | Δ_α = χ²(α, d) |
| Pela verossimilhança, pontual | {θ_i : χ²_PL(θ_i) − χ²(θ̂) < Δ_α} (eq. 8) | Δ_α = χ²(α, 1) |
| Pela verossimilhança, simultâneo | idem | Δ_α = χ²(α, d) |

- **Coordenada.** Parâmetros positivos são tratados em log10, como no artigo. O intervalo do perfil não depende da coordenada; o assintótico depende.
- **Hessiana.** H é a Hessiana exata, por diferenças centrais do gradiente (2d gradientes). Com ela, a parábola da figura é tangente ao perfil no ótimo. A aproximação de Gauss–Newton H ≈ 2JᵀJ (informação de Fisher) sai junto, para comparação (`dp_gauss_newton`): ela despreza o termo dos resíduos vezes a curvatura do modelo e pode errar muito em direções frouxas.
- **Intervalo infinito.** Se a informação de Fisher é singular numa direção que envolve o parâmetro, o intervalo assintótico é infinito (`observacao` = "Fisher singular").
- **Recuo para Gauss–Newton.** Se a Hessiana não tem curvatura positiva para o parâmetro (ótimo no limite do prior, verossimilhança plana) ou não pôde ser calculada, o intervalo sai de Gauss–Newton e a coluna `observacao` diz por quê.
- **Domínio.** O perfil é calculado dentro da caixa do prior do benchmark. Um intervalo que chega ao limite da caixa faz o papel do intervalo infinito do artigo e fica marcado: a coluna `perfil_pontual_extremos` traz o rótulo de cada extremo (tabela abaixo) e `perfil_simultaneo_chega_ao_limite` diz se o simultâneo chega ao limite. O intervalo assintótico ignora a caixa.

`relatorio` grava tudo em `relatorio/intervalos_de_confianca.csv`, uma linha por parâmetro. Em `identificabilidade.json`, cada parâmetro tem `u` e `chi2` (o perfil), `u_otimo`, `assintotico` e `ic_perfil`; os limiares estão em `delta_chi2_pontual` e `delta_chi2_simultaneo`, e os espectros da Hessiana e de Gauss–Newton em `curvatura`.

### Classificação

O conjunto de confiança pontual é a união dos trechos com χ²_PL − χ²(θ̂) abaixo do limiar pontual (3,84 a 95%). Cada extremo recebe um rótulo:

| Rótulo | Significado |
|---|---|
| `dados` | O perfil cruza o limiar. |
| `aberto` | O conjunto chega ao limite do prior. |
| `prior` | O perfil só sobe porque outro parâmetro ficou preso no limite do prior dele (coluna `limitado_por`). |
| `falha` | O perfil acaba antes do limite do prior, porque a simulação deixa de existir. |

Um parâmetro é **I** (identificável) com os dois extremos em `dados`, **P** (parcial) com um, **N** (não identificável) com nenhum. O expoente **m** marca conjunto desconexo (mais de um modo); **e** marca perda de posto do jacobiano em todos os pontos sorteados do prior.

### Cuidados de leitura

- Acima de 1,2 vez o limiar simultâneo, cada ponto do perfil sai de um ou dois inícios, e o valor é um limite superior do perfil. Isso não muda os intervalos.
- Abaixo disso, uma conferência com 11 inícios por ponto nos 33 modelos sintéticos achou valor mais baixo (queda acima de 0,05 em χ²) em 11 de 2 598 pontos, todos no Goodwin com Hill livre, com queda de até 0,73. `perfil_inicios_extras` em `config.py` soma inícios sorteados do prior a cada ponto; com 8, sobraram 6 desses pontos e os intervalos mudaram no máximo 0,001 em log10, a um custo 5 vezes maior.
- O posto é numérico (tolerância relativa de 1e-6 nos sintéticos e 1e-5 nos PEtab). Provar não identificabilidade estrutural pede ferramenta simbólica.
- No subespaço informado, só a versão com o prior normalizado tem cota coberta pela literatura. A constante da versão na caixa do prior é derivação nossa.
- A banda de predição perfil a perfil pode ficar estreita demais quando os parâmetros interagem.

## Modelos

| Grupo | Quantos | Conteúdo |
|---|---|---|
| `facil` | 18 + 6 | 11 EDOs e 7 EDPs sintéticas com posterior quase gaussiana; e 6 com dados reais (lebre e lince, gripe no internato, PK da indometacina, Fisher–KPP de Jin et al., pêndulo, traçador em esgoto). |
| `dificil` | 15 + 8 | Rígidos (Robertson, HIRES), caóticos (Lorenz-63 em janela longa, Lorenz-96), multimodais (cadeia de decaimento, PK com flip-flop), não identificáveis ou pouco informativos (decaimento com escala, SEIR só no início, Fisher–KPP num instante, Allen–Cahn metaestável) e mal-postos (calor inverso, Darcy com 16 coeficientes), entre outros; e 8 com dados reais (gripe só nos 5 primeiros dias, PK da teofilina com flip-flop, Fisher–KPP num instante, Didinium e Paramecium de Veilleux e de Gause, α-pineno, Hes1, disparo médio de rato). |
| `petab` | 9 | Bertozzi, Boehm, Rahman, Crauste, Perelson, SalazarCavazos, Blasi, Liu, Oliveira (dados reais, 3 a 12 parâmetros). |
| `petab_grande` | 7 | Bruno, Sneyd, Fujita, Elowitz, Laske, Armistead, Okuonghae (13 a 21 parâmetros). |

Os sintéticos são integrados em numba: Runge–Kutta de 4ª ordem nos não rígidos e Rosenbrock de 2ª ordem (ROS2, L-estável) nos rígidos. Os PEtab usam o AMICI (CVODES).

### Dados reais e sintéticos

Cada modelo tem `dados` igual a `real` ou `sintetico` em `problema.json` e em `relatorio/problemas.csv`. Os sintéticos geram o próprio dado (simulação no valor verdadeiro, ruído de semente fixa) e formam a trilha de validação, com verdade conhecida. As análises sobre dados reais usam só os modelos com `dados = real`:

```bash
python -m a0bench rodar --preset reais --saida resultados_reais     # o protocolo completo, só com dados reais
python -m a0bench rodar --dados sinteticos --saida resultados_sint  # só a trilha sintética
```

Os 14 modelos numba com dados reais estão no fim de `modelos.py` (`reais()` e `reais2()`). Os dados ficam em `dados/<modelo>/` na raiz (CSV; nos PEtab, os arquivos da coleção com as medições), com fonte, URL, data e sha256 do arquivo bruto em `fontes.json`:

| Modelo | Dados | Escala observada | d |
|---|---|---|---|
| `lebre_lince` | Peles da Hudson's Bay Company, 1900 a 1920 (Howard, 2009; estudo de caso do Stan) | log das peles | 6 (4 taxas e as populações iniciais) |
| `gripe_internato` | Gripe num internato inglês, 1978: acamados e convalescentes por dia (BMJ, 1978; King et al., pomp) | raiz das contagens | 4 |
| `gripe_internato_inicio` | Os mesmos, só os 5 primeiros dias | raiz das contagens | 4 |
| `pk_teofilina` | Teofilina oral, sujeito 1 (Boeckmann et al., 1994; `datasets::Theoph` do R) | log da concentração | 3 |
| `pk_indometacina` | Indometacina intravenosa, sujeito 1 (Kwan et al., 1976; `datasets::Indometh`) | log da concentração | 4 |
| `fisher_kpp_jin` | Ensaio de arranhão com células PC-3, 10 mil células por poço, média de 3 réplicas (Jin et al., 2016) | densidade (1e-3 células/µm²) | 2 |
| `fisher_kpp_jin_instante` | Os mesmos, só o perfil de 48 h, com a capacidade de suporte livre | densidade | 3 |
| `didinium_veilleux` | Paramecium e Didinium a cada 12 h por 35 dias (Veilleux, 1976; pacote TSA). Rosenzweig–MacArthur; ajuste fraco (R² de 0,32 e 0,45) | log das densidades | 8 |
| `didinium_gause` | Paramecium e Didinium com imigração a cada 3 dias, 17 dias (Gause, 1934; pacote gauseR). Lotka–Volterra com pulsos de imigração | raiz das contagens | 6 |
| `alfa_pineno` | Isomerização do α-pineno, 5 espécies em 8 instantes (Fuguitt e Hawkins, 1947; Box et al., 1973; COPS) | porcentagem em massa | 5 |
| `hes1` | mRNA de Hes1 em 8 instantes (Silk et al., 2011); oscilador de Goodwin com Hill livre | variação relativa | 4 |
| `pendulo_mocap` | Ângulo de um pêndulo por captura óptica, 130 instantes em 3,2 s (Giessler, 2025) | rad | 4 |
| `fhn_rato` | Média de 64 disparos de um registro de radícula dorsal de rato (Metcalfe, 2020). FitzHugh–Nagumo com escala de tempo e amplitude livres | potencial médio | 6 |
| `tracador_esgoto` | Traçador NaCl num trecho de esgoto de 469 m (Rieckermann et al., 2005). Advecção–dispersão por convolução com a curva a montante | condutividade | 4 |

O ruído é gaussiano na escala observada, com desvio fixo por grupo de saídas: o desvio residual do ajuste de máxima verossimilhança, com correção de graus de liberdade (o modelo só com o início da gripe usa o desvio do modelo completo). A posterior de referência e todos os métodos usam o mesmo desvio, então a comparação entre métodos não depende dessa escolha; a posterior é condicional a ele. Para refazer os CSVs e o desvio a partir dos arquivos brutos: `python -m a0bench.dados_reais.preparar <pasta_brutos> --sigma`. Os demais modelos sintéticos não têm conjunto de dados público adequado (sistemas idealizados, como Lorenz, ou EDPs de manual) e ficam só na trilha sintética.

## Saída

```
resultados_a0/
├── config.json              parâmetros usados
├── execucao.json            modelos, etapas e uma entrada por sessão (início, fim, situação, hash do código)
├── ambiente.json            Python, sistema, núcleos, versões dos pacotes, hash de cada arquivo do pacote
├── registro.log             registro geral, com data e hora
├── codigo/                  cópia do pacote usado, do requirements.txt e deste README
├── manifesto.json           criado por `exportar`: tamanho e sha256 de cada arquivo
├── <modelo>/
│   ├── problema.json            nomes dos parâmetros, limites do prior, valor verdadeiro, diagnóstico da referência
│   ├── estado.json              andamento (lido pelo monitor)
│   ├── registro.jsonl           um evento por linha: início e fim de cada etapa, retomadas, avisos, erros
│   ├── map.pkl                  ponto de máximo, Fisher, covariância
│   ├── referencia.npz           amostras da referência e diagnóstico
│   ├── identificabilidade.json  perfis, estrutura, subespaço informado, predição
│   ├── solucao.npz              dados, soluções e faixas a posteriori; estados numa malha fina
│   ├── superficie.npz           fatias do log da posterior em pares de parâmetros
│   ├── cadeias/
│   │   ├── referencia.npz       traço das cadeias frias (no máximo 20 mil pontos por cadeia)
│   │   └── <amostrador>.npz     cadeias completas (ver abaixo)
│   ├── <método>_s<semente>.npy  amostras já sem aquecimento, usadas nas métricas
│   ├── <método>_s<semente>.json métricas
│   └── banco.npz                simulações usadas pelo NPE
└── relatorio/                   CSVs e figuras, criado por `relatorio`
    └── intervalos_de_confianca.csv   intervalos assintóticos e pela verossimilhança, pontuais e simultâneos
```

Conteúdo de `cadeias/<amostrador>.npz`:

| Chave | Forma | Conteúdo |
|---|---|---|
| `cadeia` | (C, n, d); no ensemble (C, n, W, d) | Parâmetros a cada passo, com o aquecimento. C cadeias, W caminhantes. |
| `logp` | (C, n) ou (C, n, W) | Log da posterior a cada passo. |
| `aquecimento` | escalar | Passos iniciais descartados nas métricas. |
| `desbaste`, `passos`, `caminhantes` | escalares | Um a cada quantos passos foi guardado; passos rodados; W. |
| `aceitacao`, `rhat`, `esd`, `sementes`, `nomes` | | Diagnósticos, custo por cadeia e nomes dos parâmetros. |

No preset completo, as cadeias ocupam cerca de 2,4 MB por parâmetro mais uns 4 MB por modelo, o que dá da ordem de 700 MB nos 41 modelos (estimativa pelo tamanho dos vetores, não medida); `--desbaste` reduz.

## Reuso dos resultados

```python
from a0bench import carregar

r = carregar("resultados_a0")              # pasta, ou o .zip criado por `exportar`
r.modelos                                  # nomes dos modelos
r.tabela()                                 # métricas por modelo e método (pandas)
c = r.cadeia("lotka_volterra", "hmc")      # dicionário com cadeia, logp, aquecimento, ...
r.amostras("lotka_volterra", "npe")        # amostras da posterior segundo o método
r.referencia("lotka_volterra")             # amostras da referência
r.identificabilidade("lotka_volterra")     # perfis, estrutura, subespaço informado
r.solucao("calor"); r.superficie("calor")  # soluções e superfícies gravadas
r.registro("lotka_volterra")               # histórico do processo
r.para_arviz("lotka_volterra", "hmc")      # InferenceData, se o arviz estiver instalado
```

- `exportar` junta resultados, cadeias, registros, ambiente e a cópia do código num `.zip` e grava `manifesto.json`. `--sem-cadeias`, `--sem-relatorio` e `--sem-banco` deixam o arquivo menor.
- `conferir` compara cada arquivo com o sha256 do manifesto.
- Para refazer as figuras de uma execução antiga basta `relatorio`; nada é simulado de novo.
- Para acrescentar métodos ou modelos a uma execução, rode `rodar` na mesma pasta: só o que falta é calculado. Para refazer uma etapa de um modelo, apague o arquivo correspondente.

## Se algo der errado

| Sintoma | O que fazer |
|---|---|
| Um modelo termina com ✖ | A mensagem está no resumo final, em `<modelo>/estado.json` e em `<modelo>/registro.jsonl`; o rastro completo fica em `<modelo>/erro.txt`. Os outros modelos seguem. Rodar o mesmo comando refaz só o que falhou. |
| "não rodou, porque outro processo está trabalhando nesta pasta" | Há outra execução na mesma pasta de saída. Se não houver, apague `<modelo>/em_execucao.lock`. |
| NPE-PFN falha em todos os modelos | Os pesos do TabPFN são baixados na primeira vez. A mensagem completa, com o endereço e a pasta de destino para baixar à mão, fica em `<saida>/_tabpfn_indisponivel.txt`. Para rodar sem ele: `--sem-pfn`. |
| O AMICI não compila um modelo PEtab | Confira se o venv está ativado (é nele que estão `cmake` e `swig`) e se há `g++` e os cabeçalhos do Python. `verificar` testa isso. |
| Erro de LaTeX nas figuras | `relatorio --sem-latex` usa a fonte STIX. |
| Falta memória | Diminua `--processos`. |

## O que foi testado na versão 3.2

Tudo num ambiente novo, criado só com o `requirements.txt`, no Ubuntu 24.04 com Python 3.12.3, 2 núcleos e 8 GB.

| Teste | Resultado |
|---|---|
| Instalação a partir do `requirements.txt` e do `requirements-lock.txt` | 125 pacotes nos dois casos, as 14 versões diretas conferem, `pip check` sem conflitos. Feita com o instalador `uv`; a resolução pelo `pip` puro não terminou em 40 min no ambiente de teste, por causa da velocidade de download. |
| `verificar` a partir do zip, numa pasta limpa | Todas as etapas passam, em 1 min 30 s, com o AMICI compilando o Boehm do zero. |
| Os 37 modelos menores, preset rápido, Laplace, quatro amostradores e NPE | 444 tarefas (modelo × método × semente), nenhuma falha; 40 min 45 s. No relatório, 611 figuras com LaTeX e uma falha (o painel de identificabilidade do Rahman, por um símbolo que o LaTeX não aceitava), corrigida em seguida. |
| Os 4 PEtab maiores, preset rápido, mesmos métodos | Rodaram de ponta a ponta em 68 min. Foi onde apareceram os defeitos do Fujita (ver abaixo), corrigidos em seguida. |
| Pausa e retomada das cadeias | Idêntica bit a bit nos quatro amostradores e na referência, inclusive quando a referência dobra de comprimento. |
| Pausa e retomada pela linha de comando | Ctrl+C, `kill` e `kill -9` no meio de uma execução de 5 modelos: nenhum processo sobra, nenhum arquivo temporário sobra, e o mesmo comando termina a execução sem erro. |
| Pausa e retomada da identificabilidade | Resultado igual ao da execução direta em 2 modelos. |
| Identificabilidade no preset completo | 37 modelos, 13 min 22 s (rodada da versão 3.1; depois das mudanças da 3.2, os resultados de 2 modelos conferidos são os mesmos). |
| Outros | 4 processos em 2 núcleos; pasta com espaços e acentos no caminho; `relatorio` com LaTeX, com `--pdf`, com `--sem-latex`, durante uma execução e com processos criados por `spawn`; `exportar`, `conferir` (detecta arquivo alterado) e `carregar`; `para_arviz` com o ArviZ 1.3; método acrescentado depois na mesma pasta; `pip install -e .`. |
| Memória | Pico de 1,0 GB por processo no preset rápido. |

## Limitações

- **NPE-PFN sem teste de execução.** O ambiente de teste não tem acesso aos pesos do TabPFN. Foi testado até o ponto do download e o caso de falha (o método é marcado como falho, com aviso, e o resto segue). O uso da GPU (`dispositivo_pfn`) também não foi testado; se a GPU falhar, o código volta para a CPU. O tempo por semente não é conhecido.
- **O preset completo nunca rodou inteiro de uma vez.** As partes foram testadas em separado: referências completas e identificabilidade em 37 modelos, amostradores no preset rápido nos 41.
- **Referências dos 4 PEtab maiores no preset completo não foram rodadas.** No preset rápido (4 mil passos) elas não convergem, como esperado; não se sabe se convergem dentro dos 400 mil passos do preset completo. Se não convergirem, o modelo fica marcado e segue.
- **Fujita.** As sensibilidades do AMICI falham com a tolerância padrão em boa parte dos pontos, inclusive no ponto nominal. Com o recuo de tolerância o gradiente sai, mas custa cerca de 50 a 80 soluções, e a identificabilidade fica cara: no preset rápido, o teste fez 2 de 19 perfis em 10 minutos e foi interrompido, e no completo ela deve levar horas. Use `--sem-identificabilidade` para deixá-la para depois. MALA e HMC ficam com poucos passos nesse modelo (pela conta do orçamento, cerca de 700 e 200 no preset completo), porque o orçamento é contado em soluções.
- **Só Linux foi testado**, com Python 3.12.

- O HMC sorteia de 1 a 8 passos por iteração; não é o NUTS.
- Não há SBC nem aceitação atrasada com substitutos.
- As cadeias de um amostrador rodam em lote com um único gerador aleatório; mudar a lista de sementes muda o lote e recomeça as cadeias.
- Nos modelos caóticos, o MAP por otimização com vários inícios pode parar num ótimo local. A identificabilidade detecta e avisa, mas os amostradores ainda partem do MAP original.
- O R-hat do ensemble trata cada conjunto de caminhantes como uma cadeia.
- A superfície de `superficie` é uma fatia (os demais parâmetros fixos); não é o perfil em duas dimensões.
- O perfil fica dentro da caixa do prior. Não há a comparação com dados adicionais (planejamento de experimentos) da figura 3 de Raue et al. (2009).
