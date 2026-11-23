"""Monta o relatorio tecnico em PDF a partir de <entrega>/dados, <entrega>/figuras e <entrega>/numeros.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "pdf_libs"))
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,  # noqa: E402
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

E = Path(sys.argv[1])
D, FG, FX = E / "dados", E / "figuras", E / "figuras"
N = json.load(open(E / "numeros.json", encoding="utf-8"))
OUTPDF = E / "Relatorio_Tecnico_A0.pdf"

# ------------------------------------------------------------------ fontes e estilos
fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for nome, arq in [("Serif", "DejaVuSerif.ttf"), ("Serif-B", "DejaVuSerif-Bold.ttf"), ("Serif-I", "DejaVuSerif-Italic.ttf"),
                  ("Serif-BI", "DejaVuSerif-BoldItalic.ttf"), ("Sans", "DejaVuSans.ttf"), ("Sans-B", "DejaVuSans-Bold.ttf"),
                  ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(nome, str(fd / arq)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans", boldItalic="Sans-B")
INK, MUTED, ACC = colors.HexColor("#1b1b1b"), colors.HexColor("#555555"), colors.HexColor("#0b4f8a")
BODY = ParagraphStyle("body", fontName="Serif", fontSize=9.4, leading=13.4, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5.5)
H1 = ParagraphStyle("h1", fontName="Sans-B", fontSize=15, leading=19, textColor=ACC, spaceBefore=14, spaceAfter=7, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="Sans-B", fontSize=11.2, leading=14.5, textColor=INK, spaceBefore=10, spaceAfter=4, keepWithNext=1)
H3 = ParagraphStyle("h3", fontName="Sans-B", fontSize=9.6, leading=12.5, textColor=MUTED, spaceBefore=6, spaceAfter=2, keepWithNext=1)
CAP = ParagraphStyle("cap", fontName="Serif-I", fontSize=8.2, leading=11, textColor=MUTED, alignment=TA_JUSTIFY, spaceAfter=8)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=13, bulletIndent=2, spaceAfter=2.5)
CELL = ParagraphStyle("cell", fontName="Serif", fontSize=7.4, leading=9.2, textColor=INK)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Serif-B")
CELLC = ParagraphStyle("cellc", parent=CELL, alignment=TA_CENTER)
MONO = ParagraphStyle("mono", fontName="Mono", fontSize=7.4, leading=10, textColor=INK, backColor=colors.HexColor("#f3f3f3"),
                      borderPadding=4, spaceAfter=8, spaceBefore=2)
TITLE = ParagraphStyle("title", fontName="Sans-B", fontSize=24, leading=30, textColor=ACC, spaceAfter=8)
SUB = ParagraphStyle("sub", fontName="Sans", fontSize=12.5, leading=17, textColor=INK, spaceAfter=6)
META = ParagraphStyle("meta", fontName="Sans", fontSize=9.5, leading=14, textColor=MUTED)


def f2(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(itens):
    return [Paragraph(t, BUL, bulletText="•") for t in itens]


def tab(rows, colw, fs=None, header=1, align_num=True, zebra=True, extra=None, pad=2.2):
    cel = [[Paragraph(str(c), CELLB if i < header else (CELLC if (j > 0 and align_num) else CELL)) for j, c in enumerate(r)]
           for i, r in enumerate(rows)]
    t = Table(cel, colWidths=[w * cm for w in colw], repeatRows=header)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK), ("LINEBELOW", (0, header - 1), (-1, header - 1), 0.5, INK),
          ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK), ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f4f6f8")) for i in range(header + 1, len(rows), 2)]
    t.setStyle(TableStyle(st + (extra or [])))
    return t


def heat(v):
    """cor de fundo para C2ST: 0,5 branco -> 1,0 vermelho-escuro."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return colors.white
    x = min(max((v - 0.5) / 0.5, 0), 1)
    a, b = np.array([255, 255, 255]), np.array([200, 70, 50])
    c = (a + (b - a) * x).astype(int)
    return colors.Color(c[0] / 255, c[1] / 255, c[2] / 255)


def fig(arq, w, cap):
    im = PILImage.open(arq)
    h = w * im.size[1] / im.size[0]
    return KeepTogether([Image(str(arq), width=w * cm, height=h * cm), Spacer(1, 3), Paragraph(cap, CAP)])


class Doc(BaseDocTemplate):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        fr = Frame(2.2 * cm, 2.2 * cm, A4[0] - 4.4 * cm, A4[1] - 4.4 * cm, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[fr], onPage=self.rodape)])

    def rodape(self, c, d):
        c.saveState()
        c.setFont("Sans", 7.5)
        c.setFillColor(MUTED)
        c.drawString(2.2 * cm, 1.2 * cm, "A0 benchmark — relatório técnico dos resultados preliminares (16/11/2026)")
        c.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{d.page}")
        c.restoreState()

    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and fl.style.name in ("h1", "h2"):
            nivel = 0 if fl.style.name == "h1" else 1
            self.notify("TOCEntry", (nivel, fl.getPlainText(), self.page))


# ------------------------------------------------------------------ dados
pr = pd.read_csv(D / "01_modelos_e_referencias.csv")
c2 = pd.read_csv(D / "04_c2st_modelo_x_metodo.csv")
cm_ = pd.read_csv(D / "05_resumo_classe_metodo.csv")
gl = pd.read_csv(D / "06_resumo_global_metodo.csv").set_index("metodo")
tp = pd.read_csv(D / "07_testes_pareados_metodos.csv")
ip = pd.read_csv(D / "12_identificabilidade_por_modelo.csv")
cv = pd.read_csv(D / "14_npe_curva_orcamento.csv")
tm = pd.read_csv(D / "10_tempo_mediano_etapa_classe.csv").set_index("classe")
nu = pd.read_csv(D / "15_piso_de_ruido_c2st.csv")
s0 = pd.read_csv(D / "08_npe_pfn_piloto_semente0.csv")
CLASSES = ["EDO", "EDO difícil", "EDP", "EDP difícil", "PEtab", "PEtab grande"]
MET = ["laplace", "metropolis", "ensemble", "mala", "hmc", "npe"]
NOME = dict(laplace="Laplace", metropolis="Metropolis", ensemble="Ensemble afim", mala="MALA", hmc="HMC", npe="NPE", npe_pfn="NPE-PFN")
NOMEC = dict(laplace="Laplace", metropolis="Metropolis", ensemble="Ensemble", mala="MALA", hmc="HMC", npe="NPE", npe_pfn="NPE-PFN")
g = N["global"]
rk = N["rank_medio"]
vit = N["vitorias"]
pf = N["pfn"]
pz = N["piso"]
refp = pr.ref_passos.value_counts().sort_index()
S = []  # historia (flowables)

# ------------------------------------------------------------------ capa
S += [Spacer(1, 3.2 * cm), P("Inferência bayesiana de parâmetros em EDOs e EDPs: resultados preliminares do benchmark A0", TITLE),
      P("Relatório técnico da execução completa (preset <i>completo</i>, versão 3.2 do pacote)", SUB), Spacer(1, 0.6 * cm),
      P("41 problemas inversos (33 sintéticos e 8 com dados reais no formato PEtab) · 7 métodos de inferência · referência por temperação paralela · "
        "C2ST como métrica de distância à posterior · identificabilidade por perfil de verossimilhança", META), Spacer(1, 1.0 * cm),
      P("Versão 1.0 · 16 de novembro de 2026<br/>Documento preparado para revisão pela orientação. Resultados preliminares: o NPE-PFN foi executado "
        "apenas como piloto (uma semente, 18 modelos) e a identificabilidade do modelo Fujita não foi concluída (ver seções 3.6 e 7).", META),
      Spacer(1, 1.2 * cm),
      P("<b>Como ler este documento.</b> A seção 1 resume tudo em uma página. As seções 2 e 3 descrevem o desenho e o protocolo para que o experimento possa ser "
        "auditado. A seção 4 traz os resultados, a 5 a discussão, e a 6 as ameaças à validade, que merecem leitura atenta antes de qualquer conclusão "
        "ser citada. O apêndice D lista as pastas e os arquivos que acompanham este PDF.", META), PageBreak()]
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("t0", fontName="Sans-B", fontSize=9.5, leading=15, leftIndent=0, spaceBefore=3),
                   ParagraphStyle("t1", fontName="Sans", fontSize=8.8, leading=12.5, leftIndent=14)]
H1X = ParagraphStyle("h1x", parent=H1)
S += [P("Sumário", H1X), toc, PageBreak()]

# ------------------------------------------------------------------ 1 resumo
S += [P("1. Resumo executivo", H1)]
S += [P("O benchmark A0 compara métodos de inferência bayesiana para problemas inversos de equações diferenciais, medindo quão perto cada método chega de uma "
        "posterior de referência com o mesmo orçamento computacional, contado em <i>equivalentes de solução direta</i> (ESD). Esta execução cobriu "
        "41 modelos (11 EDOs e 7 EDPs sintéticas de comportamento regular, 15 sintéticas difíceis, 4 PEtab de porte médio e 4 PEtab grandes) com 3 sementes por método. "
        "Os principais resultados são os seguintes."),
      *bullets([
          f"<b>Amostradores clássicos dominam com orçamento igual.</b> O Metropolis adaptativo sem temperação teve mediana de C2ST de {f2(g['metropolis']['c2st_mediana'], 3)} "
          f"(piso de ruído {f2(pz['mediana'], 3)}), C2ST &lt; 0,6 em {g['metropolis']['frac_c2st_menor_0_6'] * 100:.0f}% dos modelos e posto médio {f2(rk['metropolis'])} entre seis métodos. "
          f"Segue-se o ensemble afim (mediana {f2(g['ensemble']['c2st_mediana'], 3)}). Teste de Friedman: χ² = {N['friedman']['chi2']:.1f}, p &lt; 10⁻¹⁵ (41 modelos).",
          f"<b>O NPE com 10 mil simulações é o pior método em média</b> (mediana {f2(g['npe']['c2st_mediana'], 3)}; C2ST &lt; 0,6 em {g['npe']['frac_c2st_menor_0_6'] * 100:.0f}% dos modelos). "
          "A curva de orçamento mostra saturação nas EDPs a partir de 5 mil simulações e estagnação nos PEtab desde 1 mil; só nas EDOs a curva ainda desce em 10 mil.",
          f"<b>NPE-PFN (piloto).</b> Em 18 modelos de baixa dimensão e uma semente, a mediana foi {f2(pf['mediana_pfn'], 3)} contra {f2(pf['mediana_npe'], 3)} do NPE "
          f"(melhor em {pf['pfn_melhor']}, pior em {pf['npe_melhor']}, empate em {pf['empate']}; Wilcoxon p = {f2(pf['p_wilcoxon'])}). Não há evidência de diferença, e o método não alcança os amostradores "
          f"(Metropolis: {f2(pf['mediana_metropolis'], 3)} nos mesmos modelos). Rodou só em CPU, com contexto de 1000 simulações e mediana de {pf['segundos_mediano'] / 3600:.1f} h por semente.",
          "<b>Sete modelos ficam acima de C2ST 0,65 para todos os métodos.</b> Dois são falhas de inicialização em modelos caóticos (Lorenz-63 em janela longa e Lorenz-96), "
          "dois são multimodais (cadeia de decaimento e PK com flip-flop), e três são PEtab de dimensão alta (Rahman, Sneyd, Elowitz). Nos PEtab grandes, exceto no Bruno, só o Metropolis fica abaixo de 0,7 (Fujita 0,59; Sneyd 0,69), e no Elowitz nenhum método chega a esse nível.",
          "<b>Identificabilidade.</b> Em 40 modelos (207 parâmetros; Fujita ficou de fora), 103 parâmetros são identificáveis, 30 parciais e 74 não identificáveis; 50 parâmetros, de qualquer classe, têm o rótulo de perda estrutural de posto. Não identificabilidade, por si só, não prejudicou o Metropolis: os quatro modelos desenhados como não identificáveis têm C2ST de 0,50 a 0,59.",
          "<b>Ressalva principal.</b> A referência é gerada pelo mesmo núcleo Metropolis adaptativo avaliado, apenas com 6 temperaturas e cadeias muito mais longas. "
          "A vantagem do Metropolis pode estar inflada por erros compartilhados. Esta é a primeira ameaça à validade (seção 6) e a primeira recomendação (seção 7)."]),
      P("<b>O que não está neste relatório.</b> NPE-PFN fora dos 18 modelos do piloto e das sementes 1 e 2; identificabilidade do Fujita; avaliação de calibração (SBC) e de cobertura; "
        "comparação em tempo de parede com controle de carga da máquina.")]

# ------------------------------------------------------------------ 2 intro
S += [P("2. Motivação e objetivos", H1),
      P("Em problemas inversos governados por EDOs e EDPs, a posterior dos parâmetros é cara de explorar porque cada avaliação da verossimilhança exige resolver o modelo direto. "
        "Duas famílias competem: (i) amostradores baseados em cadeias de Markov, que pagam várias soluções por amostra efetiva mas oferecem garantias assintóticas, e "
        "(ii) estimadores neurais que amortizam o custo, treinando uma rede com simulações do prior e produzindo a posterior de forma direta para qualquer observação. "
        "O A0 foi desenhado como um banco de provas para essa comparação, preparando o terreno para avaliar simuladores acelerados por redes neurais. As perguntas desta execução foram:"),
      *bullets(["<b>Q1.</b> Com o mesmo orçamento em ESD, qual método chega mais perto da posterior de referência, e como isso depende da classe de problema?",
                "<b>Q2.</b> Quanto custa e quanto rende a inferência neural (NPE e NPE-PFN) frente aos amostradores, e o orçamento de simulação é o fator limitante?",
                "<b>Q3.</b> Onde todos os métodos falham, e a causa é de amostragem, de inicialização ou do problema?",
                "<b>Q4.</b> Quais parâmetros os dados determinam (identificabilidade), e isso explica o desempenho dos métodos?"])]

# ------------------------------------------------------------------ 3 desenho
S += [P("3. Desenho experimental", H1), P("3.1 Problemas", H2),
      P("Os 41 modelos estão no apêndice A. Os 33 sintéticos têm ruído gaussiano de desvio relativo conhecido, prior uniforme em caixa e valor verdadeiro conhecido, e são integrados em "
        "numba (Runge–Kutta de quarta ordem nos não rígidos, Rosenbrock L-estável de segunda ordem nos rígidos). Os 8 problemas PEtab (Bertozzi, Boehm, Rahman, Crauste, Bruno, Sneyd, Fujita, Elowitz) "
        "vêm da coleção Benchmark-Models-PEtab, com dados reais, e usam o AMICI (CVODES) com sensibilidades diretas; o prior é uniforme em log₁₀ com meia-largura 1 em torno do ponto nominal. "
        "Os dados dos 33 modelos sintéticos são simulados; as fontes dos dados reais, a justificativa dos dados simulados e a situação de cada modelo estão no apêndice F. "
        "Cada problema tem <b>um único conjunto de dados</b>: os resultados descrevem a posterior dessa observação, não uma média sobre conjuntos de dados. "
        "As classes usadas nos resultados são: EDO (11), EDO difícil (10), EDP (7), EDP difícil (5), PEtab (4) e PEtab grande (4)."),
      P("3.2 Métodos", H2)]
S += [tab([["Método", "Descrição", "Custo em ESD"],
           ["Laplace", "Gaussiana no MAP com a inversa da Hessiana (informação de Fisher); MAP por otimização com 8 inícios.", "≈ 660 (otimização)"],
           ["Metropolis", "Metropolis adaptativo (escala por taxa-alvo 0,234 e covariância adaptativa), sem temperação (1 temperatura), 3 cadeias, parâmetros no espaço original. O rótulo PT do pacote refere-se ao código compartilhado com a referência.", "1 por passo; 50 mil por cadeia"],
           ["Ensemble afim", "Movimento de esticamento (Goodman–Weare, a = 2), caminhantes no espaço sem restrições (logit sobre a caixa do prior).", "1 por avaliação"],
           ["MALA", "Langevin com correção de Metropolis, pré-condicionado, no espaço sem restrições; passo e covariância adaptados na primeira metade.", "d+1 (sint.) ou c<sub>S</sub> (PEtab) por gradiente"],
           ["HMC", "Hamiltoniano com massa igual à inversa da covariância; número de passos de leapfrog sorteado em 1 a 8 (não é NUTS).", "idem MALA, por passo de leapfrog"],
           ["NPE", "Estimação neural da posterior (pacote sbi, fluxo padrão), treinada com 10 mil simulações do prior e a observação bruta como entrada.", "10 mil simulações (treino não contado)"],
           ["NPE-PFN", "TabPFN como estimador de densidade sem treino, usando simulações como contexto (pacote sbi).", "1000 simulações de contexto (CPU)"]],
          [2.6, 10.4, 3.4], align_num=False),
      Spacer(1, 4),
      P("O custo de um gradiente é c<sub>S</sub> ESD nos PEtab, medido como a razão entre o tempo mediano de um gradiente e o de uma solução (de 2,6 a 80, apêndice A), e d + 1 ESD nos sintéticos "
        "(diferenças finitas). Esse critério penaliza MALA e HMC quando o gradiente é caro e é uma escolha de contabilidade, não uma propriedade dos métodos (seção 6).", BODY),
      P("3.3 Referência e métricas", H2),
      P("A posterior de referência é obtida por Metropolis adaptativo com <b>temperação paralela</b> (6 temperaturas β = 1; 0,5; 0,25; 0,1; 0,03; 0,01), 2 cadeias independentes, partindo de amostras da gaussiana de Laplace. "
        "O comprimento começa em 50 mil passos e dobra até 400 mil, até R-hat &lt; 1,05 e ESS mínimo ≥ 400. "
        f"Passos usados: 50 mil em {int(refp.get(50000, 0))} modelos, 100 mil em {int(refp.get(100000, 0))}, 200 mil em {int(refp.get(200000, 0))} e 400 mil em {int(refp.get(400000, 0))}. "
        f"{int(pr.ref_ok.sum())} das 41 referências convergiram (ESS mediano {round(pr.ref_ess.median())}; R-hat máximo {f2(pr.ref_rhat.max(), 3)}); <b>o Elowitz não convergiu</b> (R-hat {f2(pr[pr.problema.str.startswith('Elowitz')].ref_rhat.iloc[0], 3)}, "
        f"ESS {int(pr[pr.problema.str.startswith('Elowitz')].ref_ess.iloc[0])}) e seus C2ST são contra uma referência incerta."),
      P("<b>C2ST</b> (<i>classifier two-sample test</i>): um classificador (MLP com validação cruzada, implementação do sbi) tenta distinguir 1000 amostras do método de 1000 da referência, em coordenadas "
        "normalizadas pela caixa do prior. Acurácia 0,5 significa indistinguível; 1,0, completamente distinguível. Também se reportam o erro máximo da média em desvios-padrão da referência, a razão geométrica "
        "dos desvios-padrão, o R-hat máximo entre cadeias e o ESS mínimo por mil ESD."),
      P("3.4 Orçamento, sementes e protocolo", H2),
      P("Cada amostrador roda 3 cadeias de 50 mil ESD (sementes 0, 1, 2). O NPE treina com 10 mil simulações por semente e amostra 2000 pontos. O mesmo banco de simulações (gerado com semente fixa) é usado em todas as sementes do NPE, "
        "de modo que a variação entre sementes reflete só a inicialização e a ordem do treino. O NPE-PFN foi executado com a semente 0 em 18 modelos de 2 e 3 parâmetros (EDO, EDP e as duas classes difíceis), com contexto limitado a 1000 simulações "
        "(limite do pacote para CPU). A configuração efetiva está em <font face='Mono'>dados/protocolo_efetivo.json</font>; o arquivo de configuração da pasta de resultados foi sobrescrito pelas sessões finais e não representa o protocolo."),
      P("3.5 Ambiente", H2),
      P("Notebook com Intel Core i7-1355U (10 núcleos físicos, 12 threads lógicos), 15 GB de RAM, Windows 11 com WSL2 (Ubuntu 24.04), Python 3.12.3, numpy 2.5.4, scipy 1.18.1, torch 2.14.1, sbi 0.27.0, tabpfn 9.2.0, amici 1.1.0, "
        "petab 0.9.0, numba 0.68.1. Sem GPU NVIDIA. Até 10 modelos em paralelo, um núcleo por processo. A instalação seguiu o arquivo de versões fixas do pacote (125 pacotes). Detalhes no apêndice C."),
      P("3.6 Identificabilidade", H2),
      P("Quatro métodos complementares: perfil de verossimilhança no nível 0,95 com intervalos pontuais e simultâneos (Raue et al., 2009), estrutura local por posto do jacobiano (Stigter e Molenaar, 2015), "
        "subespaço informado (Zahm et al., 2022) e predição perfil a perfil (Simpson e Maclaren, 2023). Um parâmetro é identificável (I) se o conjunto de confiança é limitado pelos dados nos dois lados, "
        "parcial (P) se em um, não identificável (N) se em nenhum; o rótulo estrutural indica perda de posto em todos os pontos sorteados do prior. O perfil é calculado dentro da caixa do prior. "
        "A identificabilidade do Fujita foi interrompida (1 de 19 perfis após cerca de 25 h) e fica como pendência.")]

# ------------------------------------------------------------------ 4 resultados
S += [P("4. Resultados", H1), P("4.1 Desempenho global com orçamento igual", H2)]
linhas = [["Método", "Q1", "Mediana", "Q3", "C2ST &lt; 0,6", "Posto médio", "Melhor por modelo", "ESS/mil ESD (mediana)", "Cadeias R-hat &gt; 1,05"]]
for m in MET:
    r = gl.loc[m]
    es = "—" if np.isnan(r.ess_por_mil_esd_mediana) else f2(r.ess_por_mil_esd_mediana, 1)
    rh = "—" if np.isnan(r.cadeias_rhat_maior_1_05) else f"{int(r.cadeias_rhat_maior_1_05)}/{int(r.cadeias_total)}"
    linhas.append([NOME[m], f2(r.c2st_q1, 3), f2(r.c2st_mediana, 3), f2(r.c2st_q3, 3), f"{r.frac_c2st_menor_0_6 * 100:.0f}%", f2(rk[m]), str(vit[m]), es, rh])
S += [tab(linhas, [2.7, 1.2, 1.5, 1.2, 1.7, 1.7, 2.1, 2.2, 2.3]),
      P("Tabela 1. C2ST sobre os 41 modelos (cada modelo é a média das 3 sementes). “Melhor por modelo”: número de modelos em que o método está a até 0,01 do menor C2ST. Posto médio: 1 é o melhor. "
        "A contagem de cadeias com R-hat &gt; 1,05 é por semente (123 por método).", CAP)]
S += [P(f"O teste de Friedman rejeita a igualdade dos seis métodos (χ² = {N['friedman']['chi2']:.1f}, gl = 5, p = {N['friedman']['p']:.1e}; sem o Elowitz, χ² = {N['friedman_sem_elowitz']['chi2']:.1f}). "
        "Nas comparações pareadas (Wilcoxon, 41 modelos, correção de Holm para 15 pares), o Metropolis é melhor que todos os demais (p<sub>Holm</sub> &lt; 10⁻³), o ensemble é melhor que Laplace (p<sub>Holm</sub> = 0,02), MALA, HMC e NPE, "
        "e o NPE é pior que todos. Três pares não se distinguem: Laplace–HMC, Laplace–MALA e MALA–HMC. A tabela completa está em <font face='Mono'>dados/07_testes_pareados_metodos.csv</font>."),
      P(f"A magnitude das diferenças deve ser lida contra o ruído: o C2ST entre duas metades da própria referência fica em {f2(pz['mediana'], 3)} (máximo {f2(pz['maximo'], 3)}, no {pz['maximo_modelo']}), "
        f"e o desvio-padrão do C2ST entre sementes tem mediana de {f2(N['dp_semente_mediano']['metropolis'], 3)} no Metropolis e {f2(N['dp_semente_mediano']['npe'], 3)} no NPE (figura 2). "
        "Diferenças abaixo de cerca de 0,03 não são interpretáveis.")]
S += [fig(FG / "c2st_mapa.png", 11.2, "Figura 1. C2ST contra a referência, por modelo e método (média das 3 sementes; NPE-PFN com 1 semente nos 18 modelos do piloto). O asterisco marca o Elowitz, cuja referência não convergiu."),
      fig(FX / "fig_robustez.png", 15.8, "Figura 2. (a) Piso de ruído do C2ST: a referência comparada a ela mesma, 4 pares disjuntos de 1000 amostras por modelo. (b) Desvio-padrão do C2ST entre as 3 sementes, um valor por modelo.")]

# 4.2 por classe
S += [P("4.2 Por classe de problema", H2)]
lin = [["Classe (n)"] + [NOMEC[m] for m in MET]]
ext = []
for i, c in enumerate(CLASSES, 1):
    r = cm_[cm_.classe == c].set_index("metodo")
    n = int(r.n_modelos.iloc[0])
    lin.append([f"{c} ({n})"] + [f2(r.loc[m, "c2st_mediana"], 2) for m in MET])
    for j, m in enumerate(MET, 1):
        ext.append(("BACKGROUND", (j, i), (j, i), heat(r.loc[m, "c2st_mediana"])))
S += [tab(lin, [3.2, 2.1, 2.1, 2.1, 2.0, 2.0, 2.0], zebra=False, extra=ext),
      P("Tabela 2. Mediana do C2ST (sobre modelos) por classe e método; sombreado proporcional ao C2ST.", CAP),
      P("Nas EDOs e EDPs de comportamento regular, todos os métodos exceto o NPE têm mediana de até 0,55, e a escolha do amostrador pesa pouco. Nas EDOs difíceis, o Metropolis (0,51) separa-se do ensemble (0,58) e dos métodos "
        "que dependem do MAP e de gradientes (Laplace 0,70; MALA 0,73; HMC 0,69). Nos PEtab, o Metropolis (0,56) distancia-se do ensemble (0,65) e dos demais (0,88 a 0,94); nos PEtab grandes, "
        "só o Metropolis (0,64) permanece abaixo de 0,9. A figura 3 mostra a dispersão por modelo; a figura 4, o C2ST contra o custo em ESD.")]
S += [fig(FX / "fig_c2st_pontos_classe.png", 16.4, "Figura 3. C2ST por modelo (pontos) e mediana por classe (traço), para cada método. Lap = Laplace, MH = Metropolis, Ens = ensemble."),
      fig(FG / "c2st_custo.png", 13.2, "Figura 4. C2ST contra o custo em ESD (escala logarítmica). Forma = classe, cor = método. O NPE-PFN aparece em 1000 (simulações de contexto).")]

# 4.3 PEtab
S += [P("4.3 Problemas PEtab", H2)]
pt = c2[c2["classe"].isin(["PEtab", "PEtab grande"])]
lin = [["Modelo", "d"] + [NOMEC[m] for m in MET]]
ext = []
for i, (_, r) in enumerate(pt.iterrows(), 1):
    lin.append([r.rotulo, int(r.d)] + [f2(r[m], 2) for m in MET])
    for j, m in enumerate(MET, 2):
        ext.append(("BACKGROUND", (j, i), (j, i), heat(r[m])))
S += [tab(lin, [3.6, 0.9, 1.9, 2.0, 1.9, 1.7, 1.7, 1.7], zebra=False, extra=ext),
      P("Tabela 3. C2ST nos 8 problemas PEtab (média de 3 sementes). A referência do Elowitz não convergiu.", CAP),
      P("Com orçamento em ESD, o gradiente custa de 2,6 (Bertozzi) a cerca de 80 soluções (Fujita), e MALA e HMC dispõem de poucas centenas de passos nos modelos maiores. O ESS mínimo por mil ESD do Metropolis nos PEtab "
        f"(mediana {f2(cm_[(cm_.classe == 'PEtab') & (cm_.metodo == 'metropolis')].ess_por_mil_esd_mediana.iloc[0], 1)}) é 20 a 33 vezes o de MALA e HMC (figura 5). "
        "No Fujita, o MAP parte do ponto nominal da coleção porque a otimização com vários inícios não o alcançou; no Crauste, existe região inviável colada ao ótimo (20 de 48 perturbações de 0,2% falham). "
        "No Rahman, 8 dos 9 parâmetros não são identificáveis e a referência exigiu 400 mil passos.")]
S += [fig(FG / "eficiencia.png", 14.5, "Figura 5. ESS mínimo por mil ESD, por amostrador e classe (escala logarítmica).")]

# 4.4 falhas
S += [P("4.4 Modelos em que todos os métodos falham", H2)]
fal = c2.copy()
fal["min"] = fal[MET].min(axis=1)
fal = fal[fal["min"] > 0.65].sort_values("min", ascending=False)
causa = {"Lorenz-96": "MAP em ótimo local: a referência achou −log L menor em 1.937. Os amostradores partem do MAP.",
         "Lorenz-63 (janela longa)": "Idem, −log L menor em 1.095 na referência.",
         "Cadeia de decaimento": "Seis modos por simetria (três observados nas cadeias); cada cadeia permanece em um modo.",
         "Elowitz (PEtab)": "Referência não convergida; 21 parâmetros.",
         "PK com flip-flop": "Dois modos (ambiguidade flip-flop); sem temperação, cada cadeia fica em um.",
         "Rahman (PEtab)": "8 de 9 parâmetros não identificáveis; referência de 400 mil passos.",
         "Sneyd (PEtab)": "15 parâmetros; referência de 400 mil passos."}
lin = [["Modelo", "Classe", "d", "Menor C2ST", "Melhor método", "Causa provável"]]
for _, r in fal.iterrows():
    mt = r[MET].astype(float).idxmin()
    lin.append([r.rotulo, r.classe, int(r.d), f2(r["min"], 2), NOME[mt], causa.get(r.rotulo, "")])
S += [tab(lin, [2.9, 1.9, 0.7, 1.5, 2.1, 7.2], align_num=False),
      P("Tabela 4. Modelos com menor C2ST acima de 0,65 entre os seis métodos.", CAP),
      P("Os dois Lorenz devem ser descritos como falha de inicialização (limitação conhecida do pacote para modelos caóticos), não de amostragem: a referência, com temperação e partida dispersa, encontra modos melhores que o MAP. "
        "A recomendação é reiniciar os amostradores a partir de amostras da referência ou refazer o MAP com os inícios da temperação. "
        "Nos dois modelos multimodais, a mesma cadeia fica em um modo, o que o R-hat entre cadeias pode ou não revelar. Esses casos mostram que a vantagem do Metropolis sobre o resto não é "
        "universal: sem temperação ele também falha.")]

# 4.5 NPE
S += [P("4.5 NPE: orçamento de simulação", H2),
      P("Para separar limite do estimador de limite do orçamento, o NPE foi retreinado com os 1, 2 e 5 mil primeiros pontos do banco de cada modelo (semente 0), e o ponto de 10 mil vem do benchmark (média de 3 sementes). "
        "Mediana de C2ST por classe:")]
cc = cv.groupby(["classe", "n_sim"]).c2st.median().unstack()
lin = [["Classe", "1 mil", "2 mil", "5 mil", "10 mil"]] + [[c] + [f2(cc.loc[c, k], 2) for k in cc.columns] for c in CLASSES]
S += [tab(lin, [3.6, 2, 2, 2, 2]), P("Tabela 5. C2ST mediano do NPE por número de simulações de treino.", CAP),
      P("Nas EDPs o NPE satura em 5 mil simulações (0,60 a 0,66): o limite é do estimador, não do orçamento. Nas EDOs a curva ainda cai em 10 mil (0,83 para 0,75; 0,93 para 0,88 nas difíceis), de modo que "
        "pontos em 20 e 50 mil seriam informativos e baratos. Nos PEtab não há melhora (0,97 para 0,94, e 0,99 nos grandes): são dimensões de 8 a 21 com observações altamente estruturadas, "
        "onde um estimador sem estatísticas sumárias nem rede de embedding não se aproxima da posterior. A limitação: uma única semente nos pontos de 1, 2 e 5 mil.")]
S += [fig(FG / "npe_curva.png", 13.5, "Figura 6. Curva de orçamento do NPE: C2ST contra o número de simulações de treino, por classe (semente 0 nos pontos de 1, 2 e 5 mil; média de 3 sementes em 10 mil).")]

# 4.6 PFN
S += [P("4.6 NPE-PFN (piloto)", H2),
      P(f"O TabPFN foi usado como estimador de densidade sem treino em {pf['n']} modelos de 2 e 3 parâmetros, com a semente 0. Como não há GPU, o contexto foi limitado a 1000 simulações, metade do previsto no preset. "
        f"A mediana de C2ST foi {f2(pf['mediana_pfn'], 3)} (NPE: {f2(pf['mediana_npe'], 3)}; Metropolis: {f2(pf['mediana_metropolis'], 3)}). O NPE-PFN foi melhor que o NPE em {pf['pfn_melhor']} modelos e pior em {pf['npe_melhor']} "
        f"(diferença mediana {f2(pf['mediana_dif'], 3)}; Wilcoxon p = {f2(pf['p_wilcoxon'])}), com ganhos concentrados nas EDPs (melhor em 6 dos 8, pior no Allen–Cahn) e perdas, na semente 0, no FitzHugh–Nagumo (0,91 contra 0,84) e no Pêndulo (0,76 contra 0,60). "
        f"O tempo de parede por semente foi de {pf['segundos_mediano'] / 3600:.1f} h em mediana ({pf['segundos_max'] / 3600:.1f} h no máximo) em um núcleo, contra cerca de 1 minuto de um amostrador. "
        "Com uma semente e 18 modelos, o resultado é indicativo; não permite afirmar vantagem nem desvantagem do método. Em Lorenz-96 todos os métodos, inclusive o NPE-PFN, falham.")]
S += [fig(FX / "fig_pfn_vs_npe.png", 8.6, "Figura 7. C2ST do NPE-PFN contra o do NPE, semente 0, nos 18 modelos do piloto. Pontos abaixo da diagonal: NPE-PFN melhor.")]

# 4.7 ident
S += [P("4.7 Identificabilidade", H2)]
ig = ip.groupby("classe")[["identificaveis", "parciais", "nao_identificaveis", "estruturais", "multimodais", "d"]].sum().reindex(CLASSES)
lin = [["Classe", "Modelos", "Parâmetros", "Identificáveis", "Parciais", "Não identif.", "dos quais estruturais", "Com mais de um modo"]]
for c in CLASSES:
    r = ig.loc[c]
    lin.append([c, int((ip.classe == c).sum()), int(r.d), int(r.identificaveis), int(r.parciais), int(r.nao_identificaveis), int(r.estruturais), int(r.multimodais)])
tot = ig.sum()
lin.append(["Total", len(ip), int(tot.d), int(tot.identificaveis), int(tot.parciais), int(tot.nao_identificaveis), int(tot.estruturais), int(tot.multimodais)])
S += [tab(lin, [2.6, 1.4, 1.7, 2.4, 1.5, 1.8, 2.4, 2.4], extra=[("FONTNAME", (0, -1), (-1, -1), "Serif-B")]),
      P("Tabela 6. Classificação dos parâmetros pelo perfil de verossimilhança (nível 0,95). O Fujita não consta. Os rótulos estrutural e multimodal podem recair sobre parâmetros de qualquer classe, por isso não somam com as colunas anteriores.", CAP),
      P("As EDOs de comportamento regular têm todos os parâmetros identificáveis. Nas EDPs difíceis, 26 dos 34 parâmetros são não identificáveis (16 com perda estrutural de posto), por desenho: dados esparsos, modos altos apagados no calor inverso, "
        "coeficientes de Darcy em número maior que as saídas (posto 7 com 7 saídas, 15 de 16 coeficientes não identificáveis). No Boehm, 7 de 9 parâmetros são identificáveis e 1 é estrutural, coerente com a literatura. "
        "Nos PEtab, 21 de 38 parâmetros são não identificáveis; no Bruno, todos os 13 são identificáveis. Nos três PEtab grandes concluídos (Bruno, Sneyd, Elowitz; 49 parâmetros), 22 são não identificáveis e 23 têm perda estrutural de posto, em sua maioria no Elowitz, cuja referência não convergiu."),
      P("<b>Relação com o desempenho dos métodos.</b> A não identificabilidade, isolada, não degrada o Metropolis: Decaimento com escala, Fisher–KPP em um instante e Calor inverso têm C2ST de 0,50 a 0,51 no Metropolis, e o Darcy com 16 coeficientes 0,59. "
        "A dificuldade vem de multimodalidade, caos e dimensão alta com informação fraca. Já MALA, HMC e Laplace sofrem nos modelos com crista ou perda de posto, como o SEIR só com o início (MALA 0,96; HMC 0,93; Metropolis 0,51) e o Darcy com 16 coeficientes (Laplace 0,99).")]
S += [fig(FG / "identificabilidade.png", 12.8, "Figura 8. Classe de cada parâmetro (I identificável, P parcial, N não identificável; e = perda estrutural de posto, m = multimodal) e contração da posterior (cor).")]

# 4.8 custo
S += [P("4.8 Custo computacional", H2),
      P("Tempos de parede medianos por etapa e classe (segundos), em um núcleo, com até 10 processos simultâneos no mesmo notebook (a carga altera os tempos; ver seção 6).")]
cols = [("referencia", "Referência"), ("metropolis", "Metropolis"), ("ensemble", "Ensemble"), ("mala", "MALA"), ("hmc", "HMC"), ("npe", "NPE"), ("npe_pfn", "NPE-PFN"), ("identificabilidade", "Identificab.")]
lin = [["Classe"] + [c[1] for c in cols]]
for c in CLASSES:
    lin.append([c] + [("—" if pd.isna(tm.loc[c, k]) else f"{int(tm.loc[c, k]):,}".replace(",", ".")) for k, _ in cols])
S += [tab(lin, [2.6, 1.9, 1.9, 1.8, 1.5, 1.5, 1.5, 1.9, 2.0]), P("Tabela 7. Tempo de parede mediano por etapa (s). A referência dos PEtab grandes (até 400 mil passos com 6 temperaturas) levou horas e não consta na mediana.", CAP),
      P("Nos sintéticos, o NPE é a etapa mais cara em tempo de parede (cerca de 500 s) apesar de usar 5 vezes menos simulações que uma cadeia de amostrador, porque o treino da rede domina: o custo em ESD não captura isso. "
        "Nos PEtab grandes o NPE é mais barato em tempo de parede (cerca de 200 s contra 640 a 820 s dos amostradores) e muito pior em precisão. O NPE-PFN é duas ordens de grandeza mais caro que um amostrador em CPU.")]

# ------------------------------------------------------------------ 5 discussao
S += [P("5. Discussão", H1),
      P("<b>Amostradores contra inferência neural.</b> Com 50 mil ESD por cadeia, um Metropolis adaptativo bem configurado resolve a maioria dos problemas deste conjunto, de 2 a 21 parâmetros, e o ensemble afim vem logo atrás. "
        "Os estimadores neurais aqui avaliados, no regime de uma única observação, não competem em precisão. Isso é esperado: a inferência amortizada só se justifica quando a mesma rede atende muitas observações. "
        "Com os tempos medianos medidos nas EDOs (NPE 601 s; Metropolis 51 s), o ponto de equilíbrio seria de cerca de 12 observações <i>se</i> o NPE atingisse a mesma precisão, o que não ocorre (0,75 contra 0,51). "
        "A conta é uma ordem de grandeza, não um resultado, e depende de a observação ser nova e do mesmo modelo."),
      P("<b>Por que Metropolis e não gradientes.</b> O resultado é uma consequência da contabilidade em ESD com gradiente a custo d + 1 ou c<sub>S</sub>. Em problemas de dimensão baixa a moderada, o gradiente por diferenças finitas "
        "custa de 3 a 6 soluções (sintéticos) e MALA e HMC ficam com de poucos milhares a 17 mil gradientes, o que ainda os coloca perto do Metropolis nas classes fáceis. Nos PEtab, c<sub>S</sub> chega a dezenas, e o resultado inverte contra os gradientes. "
        "Se o gradiente viesse de diferenciação automática ou de um método adjunto, com custo de poucas soluções, o resultado para MALA e HMC provavelmente melhoraria; a comparação deve ser repetida com custo de gradiente de 1 a 3 ESD "
        "antes de generalizar (seção 7)."),
      P("<b>Implicações para a avaliação de simuladores acelerados por redes neurais (proposta).</b> O eixo de custo em ESD é justo para solvers e injusto para substitutos, cujo custo por avaliação é muito menor mas que introduzem um erro de modelagem. "
        "Uma avaliação consistente deveria (i) manter a referência gerada com o simulador original, para que o erro do substituto apareça na métrica; (ii) relatar o custo em ESD e em tempo de parede, incluindo o treino do substituto e a geração das simulações; "
        "(iii) tratar o custo de treino como amortizável por número de observações, e relatar o ponto de equilíbrio; e (iv) separar o erro de amostragem do erro do substituto, comparando o mesmo amostrador com o simulador original e com o substituto. "
        "Isto é uma proposta de desenho, não um resultado desta execução."),
      P("<b>Falhas de amostragem e de inicialização.</b> Das sete falhas totais, duas são de inicialização em modelos caóticos, duas são multimodalidade (que só a referência, com temperação, resolve) e três são PEtab de dimensão alta com referência cara. "
        "Para o benchmark servir como régua, convém classificá-las separadamente e não somá-las ao desempenho médio dos métodos.")]

# ------------------------------------------------------------------ 6 ameacas
S += [P("6. Ameaças à validade e limitações", H1),
      *bullets([
          "<b>Referência e Metropolis compartilham o núcleo.</b> A referência é o Metropolis adaptativo com 6 temperaturas e 2 cadeias de pelo menos 50 mil passos (custo de 600 mil ESD ou mais por modelo), contra 3 cadeias de 50 mil ESD do método avaliado. "
          "A temperação resolve parte do problema de multimodalidade, mas erros que dependem do núcleo (adaptação de covariância, parametrização no espaço original, partida a partir da gaussiana de Laplace) tendem a ser comuns, o que pode "
          "favorecer o Metropolis. Os demais amostradores usam outro espaço (logit) e outras dinâmicas, e são avaliados contra uma referência de outra família. Mitigação recomendada: referência independente (SMC, nested sampling ou HMC/NUTS longo) em um subconjunto, e modelos com posterior analítica.",
          "<b>A referência não é verdade absoluta.</b> É validada por R-hat e ESS, não por solução exata. O Elowitz não convergiu em 400 mil passos; as demais convergiram segundo esses critérios.",
          f"<b>Limites do C2ST.</b> Com 1000 amostras por classe o piso de ruído é cerca de {f2(pz['mediana'], 3)} (máximo {f2(pz['maximo'], 3)}); o C2ST satura em 1,0 e não ordena métodos ruins entre si (muitos casos de MALA, HMC, NPE em PEtab); "
          "não mede calibração nem cobertura; e é calculado em coordenadas normalizadas pela caixa do prior, o que depende da largura da caixa.",
          "<b>Uma observação por problema.</b> Os resultados descrevem uma realização de dados por modelo. Não há SBC nem estimativa de cobertura sobre muitas realizações. Para o NPE em particular, isso dispensa justamente a propriedade pela qual ele é desenhado.",
          "<b>Sementes.</b> As 3 sementes variam as cadeias, mas o banco de simulações do NPE é o mesmo. O desvio entre sementes subestima a variância total do NPE.",
          "<b>NPE sem ajuste.</b> Fluxo padrão do sbi, sem estatísticas sumárias, sem rede de embedding, sem busca de hiperparâmetros, sem versão sequencial. O resultado é do NPE padrão, não do melhor NPE possível.",
          "<b>NPE-PFN limitado.</b> CPU, contexto de 1000 simulações, uma semente, 18 de 41 modelos (nenhum PEtab nem modelo de dimensão maior que 3). Não se generaliza.",
          "<b>Contabilidade em ESD.</b> O gradiente custa d + 1 ou c<sub>S</sub> ESD, e o treino do NPE não entra. O HMC usa de 1 a 8 passos de leapfrog sorteados (não é NUTS) e a adaptação é mínima. Os resultados de MALA e HMC são condicionais a essas escolhas.",
          "<b>Inicialização.</b> MALA, HMC, ensemble e Laplace partem do MAP por otimização, que falha nos modelos caóticos; esses dois casos medem a inicialização, não o amostrador.",
          "<b>Tempo de parede.</b> Os tempos vêm de um notebook com 10 a 11 processos simultâneos e 12 threads lógicos (2 núcleos de desempenho e 8 de eficiência), incluindo contenção e três suspensões do Windows que congelaram o WSL (cerca de 12 horas no total, sem corrupção de resultados). São indicativos.",
          "<b>Escopo.</b> Um único computador, uma única observação por modelo, só Linux/WSL, e os resultados dos problemas PEtab dependem da largura da caixa do prior (1 em log₁₀)."]),
      ]

# ------------------------------------------------------------------ 7 conclusoes
S += [P("7. Conclusões e próximos passos", H1),
      P("Com o desenho atual, o Metropolis adaptativo é o melhor método em 34 dos 41 modelos, o ensemble afim é o segundo, MALA, HMC e Laplace formam um grupo intermediário sem diferença entre si, e o NPE com 10 mil simulações é o pior. "
        "A inferência neural não é competitiva no regime de uma observação, e o NPE-PFN em CPU não altera esse quadro nos 18 modelos testados. Os resultados são preliminares até que a ameaça de circularidade da referência seja descartada. "
        "Por ordem de prioridade e de custo-benefício:"),
      *bullets([
          "<b>1. Referência independente.</b> Gerar para um subconjunto de 8 a 10 modelos (incluindo PEtab e multimodais) uma referência por outro algoritmo e recalcular o C2ST do Metropolis e do ensemble; incluir problemas com posterior analítica.",
          "<b>2. Sensibilidade ao custo do gradiente.</b> Recontabilizar MALA e HMC com custo de gradiente de 1 a 3 ESD (ou rodar com gradiente adjunto) e reavaliar o posto. Parte pode ser feita sobre as cadeias já gravadas, reescalando o custo.",
          "<b>3. NPE.</b> Ponto de 20 e 50 mil simulações nas EDOs, 3 sementes na curva, estatísticas sumárias ou rede de embedding nos PEtab, variante sequencial, e varredura do ponto de equilíbrio por número de observações.",
          "<b>4. NPE-PFN completo.</b> Três sementes nos 41 modelos exigem GPU (estimativa de 350 horas de núcleo em CPU); alternativamente, as sementes 1 e 2 dos 18 modelos de baixa dimensão cabem em cerca de 4 horas de máquina.",
          "<b>5. Inicialização dos modelos caóticos.</b> Reiniciar Lorenz-63 longo e Lorenz-96 a partir de amostras da referência, e reportá-los separadamente.",
          "<b>6. Pendências menores.</b> Identificabilidade do Fujita (horas a dias), referência do Elowitz além de 400 mil passos, e SBC e cobertura em um subconjunto de modelos de baixa dimensão."])]

# ------------------------------------------------------------------ apendices
S += [P("Apêndice A. Os 41 modelos e as referências", H1)]
lin = [["Modelo", "Classe", "d", "n<sub>obs</sub>", "Característica", "c<sub>S</sub>", "Passos ref.", "ESS ref.", "R-hat ref."]]
for _, r in pr.sort_values(["classe", "problema"]).iterrows():
    cs = "" if pd.isna(r.custo_grad) else f2(r.custo_grad, 1)
    des = "" if pd.isna(r.desafio) else str(r.desafio)
    lin.append([r.rotulo, r.classe, int(r.d), int(r.n_obs), des, cs, f"{int(r.ref_passos) // 1000} mil", int(r.ref_ess) if not pd.isna(r.ref_ess) else "", f2(r.ref_rhat, 3)])
S += [tab(lin, [3.4, 1.8, 0.6, 0.9, 4.1, 1.0, 1.4, 1.2, 1.3], align_num=False, pad=1.3),
      P("Tabela A1. c<sub>S</sub>: custo de um gradiente em soluções (medido); nos sintéticos é d + 1 por construção.", CAP)]

S += [PageBreak(), P("Apêndice B. C2ST por modelo e método", H1)]
lin = [["Modelo", "Classe", "d"] + [NOMEC[m] for m in MET] + ["NPE-PFN"]]
ext = []
for i, (_, r) in enumerate(c2.iterrows(), 1):
    vals = [r[m] for m in MET] + [r.get("npe_pfn", np.nan)]
    lin.append([r.rotulo, r.classe, int(r.d)] + [("" if pd.isna(v) else f2(v, 2)) for v in vals])
    for j, v in enumerate(vals, 3):
        ext.append(("BACKGROUND", (j, i), (j, i), heat(v)))
S += [tab(lin, [3.4, 1.9, 0.6, 1.3, 1.5, 1.5, 1.2, 1.2, 1.2, 1.7], zebra=False, extra=ext, pad=1.5),
      P("Tabela B1. C2ST médio das 3 sementes (NPE-PFN: semente 0). Valores individuais por semente, ESS, R-hat, aceitação e tempos estão em dados/02_resultados_por_semente.csv.", CAP)]

S += [PageBreak(), P("Apêndice C. Reprodutibilidade", H1),
      P("<b>Ambiente.</b> Windows 11 Pro com WSL2 (Ubuntu 24.04, kernel 6.18), Python 3.12.3 em ambiente virtual no sistema de arquivos ext4, instalado a partir de requirements-lock.txt (125 pacotes). "
        "O computador está atrás de um proxy que intercepta TLS; foi necessário um pacote de certificados restrito ao ambiente virtual para o pip clonar a coleção PEtab. O ambiente completo está em ambiente.json na pasta de resultados."),
      P("<b>Comandos principais.</b>"),
      P("python -m a0bench verificar --pfn<br/>python -m a0bench rodar --preset completo --saida resultados_a0 --sem-pfn --sem-monitor --grupos facil,dificil,petab<br/>"
        "python -m a0bench rodar --preset completo --saida resultados_a0 --sem-pfn --sem-monitor --grupos petab_grande --sem-identificabilidade<br/>"
        "python -m a0bench identificar --saida resultados_a0 --modelos Elowitz_Nature2000,Sneyd_PNAS2002<br/>"
        "python -m a0bench rodar --saida resultados_a0 --metodos npe_pfn --sementes 0 --modelos &lt;18 modelos&gt; --sem-identificabilidade<br/>"
        "python -m a0bench relatorio --saida resultados_a0 --sem-latex<br/>python -m a0bench exportar --saida resultados_a0 --destino resultados_a0_final.zip", MONO),
      P("<b>Cronologia e intercorrências.</b> A execução completa começou em 13/11/2026. Foram necessárias duas fases (37 modelos menores, 4 h 37 min; depois os 4 PEtab grandes, que levam mais que todo o resto). "
        "O Windows suspendeu o computador três vezes e congelou o WSL (23:41 a 02:15 entre 13 e 14/11, e 04:12 a 14:11 em 15/11); as execuções retomam de onde pararam, sem corrupção, mas o tempo de parede inclui essas pausas. "
        "Cerca de 30 h de máquina acordada foram gastas no total. A identificabilidade do Fujita foi interrompida após cerca de 25 h (1 de 19 perfis). O NPE-PFN completo (3 sementes, 41 modelos) foi abandonado por inviabilidade em CPU "
        "(nenhuma semente concluída após 2 h 45 min na primeira tentativa) e substituído pelo piloto descrito na seção 3.4, concluído em 16/11 às 02:26 (3 h 20 min). "
        "O arquivo config.json da pasta de resultados reflete a última sessão e não o protocolo; o protocolo efetivo está em dados/protocolo_efetivo.json."),
      P("<b>Pacote completo.</b> O arquivo resultados_a0_final.zip (789 MB, manifesto sha256) contém cadeias, amostras da referência, bancos de simulação, registros, o código usado e as 657 figuras por modelo; "
        "<font face='Mono'>python -m a0bench conferir</font> verifica a integridade e <font face='Mono'>a0bench.carregar</font> o reabre sem reexecutar.")]

S += [P("<b>Citações.</b> Raue et al. (2009), <i>Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood</i>, Bioinformatics 25(15). "
        "Stigter e Molenaar (2015), <i>A fast algorithm to assess local structural identifiability</i>, Automatica 58. Zahm et al. (2022), <i>Certified dimension reduction in nonlinear Bayesian inverse problems</i>, Math. Comp. 91. "
        "Simpson e Maclaren (2023), <i>Profile-wise analysis</i>, PLOS Comput. Biol. 19. Lueckmann et al. (2021), <i>Benchmarking simulation-based inference</i>, AISTATS (origem do C2ST como métrica). "
        "Goodman e Weare (2010), <i>Ensemble samplers with affine invariance</i>, CAMCoS 5. Hass et al. (2019), <i>Benchmark problems for dynamic modeling of intracellular processes</i>, Bioinformatics 35 (coleção de problemas PEtab). As referências bibliográficas devem ser conferidas antes de serem citadas em texto final.", CAP)]
S += [P("Apêndice D. Conteúdo da pasta que acompanha este relatório", H1),
      tab([["Caminho", "Conteúdo"],
           ["Relatorio_Tecnico_A0.pdf", "Este documento."],
           ["LEIA-ME.md", "Guia da pasta e dicionário de colunas."],
           ["dados/01 a 15 (CSV)", "Modelos e referências; resultados por semente; resumos por modelo, classe e método; testes estatísticos; piloto do NPE-PFN; tempos; identificabilidade; intervalos de confiança; curva do NPE; piso de ruído do C2ST."],
           ["dados/protocolo_efetivo.json, ambiente.json", "Configuração efetivamente usada e descrição do ambiente (versões, máquina)."],
           ["figuras/", "Figuras gerais do benchmark (PNG e PDF) e figuras novas deste relatório."],
           ["figuras/modelos_selecionados/", "Cadeias, marginais, pares, solução e identificabilidade dos nove modelos selecionados (apêndice E)."],
           ["codigo/", "Scripts das análises adicionais, das figuras novas e da geração do PDF."]],
          [5.3, 11.3], align_num=False)]


# ------------------------------------------------------------------ apendice E: posteriores e cadeias por modelo
def figh(arq, maxw, maxh, cap):
    im = PILImage.open(arq)
    w = min(maxw, maxh * im.size[0] / im.size[1])
    h = w * im.size[1] / im.size[0]
    return KeepTogether([Image(str(arq), width=w * cm, height=h * cm), Spacer(1, 3), Paragraph(cap, CAP)])

MS = FG / "modelos_selecionados"
NOMES = [("lorenz96", "Lorenz-96"), ("lorenz63_longo", "Lorenz-63 (janela longa)"), ("cadeia_simetrica", "Cadeia simétrica"),
         ("pk_flipflop", "Farmacocinética flip-flop"), ("Rahman_MBS2016", "Rahman (PEtab)"), ("Sneyd_PNAS2002", "Sneyd (PEtab)"),
         ("Elowitz_Nature2000", "Elowitz (PEtab)"), ("Boehm_JProteomeRes2014", "Boehm (PEtab)"), ("lotka_volterra", "Lotka-Volterra")]
S += [PageBreak(), P("Apêndice E. Posteriores e cadeias por modelo", H1),
      P("Para cada um dos nove modelos selecionados (os sete de falha total e dois casos bem-sucedidos), mostram-se as marginais a posteriori (referência contra os métodos), "
        "os gráficos de pares, as cadeias do Metropolis avaliado, as cadeias da referência, a solução ajustada e os perfis de verossimilhança. "
        "Os mesmos arquivos em PNG estão em figuras/modelos_selecionados/.")]
for k, (pasta, nome) in enumerate(NOMES, 1):
    d = MS / pasta
    S += [PageBreak(), P(f"E.{k} {nome}", H2),
          figh(d / "marginais.png", 16.4, 10.5, f"Figura E.{k}a. Posteriores marginais: referência e métodos."),
          figh(d / "pares.png", 16.4, 12.5, f"Figura E.{k}b. Gráficos de pares da posterior.")]
    S += [PageBreak(), figh(d / "cadeias_metropolis.png", 16.4, 11.5, f"Figura E.{k}c. Cadeias do Metropolis avaliado."),
          figh(d / "cadeias_referencia.png", 16.4, 11.5, f"Figura E.{k}d. Cadeias da referência (adaptativo com tempering).")]
    S += [PageBreak(), figh(d / "solucao.png", 16.4, 8.5, f"Figura E.{k}e. Solução do modelo sob amostras da posterior."),
          figh(d / "perfis.png", 16.4, 8.0, f"Figura E.{k}f. Perfis de verossimilhança."),
          figh(d / "identificabilidade.png", 16.4, 7.0, f"Figura E.{k}g. Identificabilidade dos parâmetros.")]

# ------------------------------------------------------------------ apendice F: origem dos dados
S += [PageBreak(), P("Apêndice F. Origem dos dados: fontes dos dados reais e justificativa dos dados simulados", H1),
      P("<b>O que os resultados deste relatório usam.</b> Dos 41 modelos das seções 4 e 5, os 8 problemas PEtab usam medições reais publicadas e os 33 modelos "
        "escritos em numba usam dados simulados: a solução do modelo num valor conhecido dos parâmetros, somada a ruído gaussiano de semente fixa. "
        "Depois dessa execução, o benchmark passou a ter 30 modelos com dados reais (Tabela F1), guardados em dados/&lt;modelo&gt;/ no repositório, e o padrão do código "
        "passou a ser rodar só esses. <b>A execução nos 22 modelos reais novos ainda não foi feita</b>; nenhum número deste relatório vem deles. "
        "Os números entre colchetes remetem à lista de referências da seção F.5."),
      P("F.1 Fontes dos dados reais", H2)]
_real = [["Modelo", "Dados", "Fonte"],
         ["Lebre e lince (Lotka–Volterra)", "Peles compradas pela Hudson's Bay Company, 1900 a 1920; 21 anos, duas espécies", "Howard (2009) [1]; arquivo do estudo de caso do Stan [2]"],
         ["Gripe no internato (duas versões)", "Acamados e convalescentes por dia, 14 dias, 763 meninos em risco; a segunda versão usa só os 5 primeiros dias", "BMJ (1978) [3]; valores lidos do gráfico, conjunto bsflu do pacote pomp [4]"],
         ["PK da teofilina", "Concentração sérica após dose oral, sujeito 1, 10 instantes", "Boeckmann, Sheiner e Beal (1994) [5]; conjunto Theoph do R [6]"],
         ["PK da indometacina", "Concentração plasmática após bolus intravenoso, sujeito 1, 11 instantes", "Kwan et al. (1976) [7]; conjunto Indometh do R [6]"],
         ["Fisher–KPP (Jin et al.; duas versões)", "Densidade de células PC-3 em ensaio de arranhão, 37 colunas, média de 3 réplicas; 4 instantes, ou só o de 48 h", "Jin et al. (2016) [8]; arquivo de Lagergren et al. (2020) [9]"],
         ["Didinium e Paramecium (Veilleux)", "Densidades de presa e predador a cada 12 h por 35 dias, 71 instantes", "Veilleux (1976) [10]; Jost e Ellner (2000) [11]; pacote TSA do R [12]"],
         ["Didinium e Paramecium com imigração (Gause)", "Contagens diárias de presa e predador por 17 dias, com imigração a cada 3 dias", "Gause (1934) [13]; digitalização do pacote gauseR [14]"],
         ["Isomerização do α-pineno", "Porcentagem em massa de 5 espécies em 8 instantes", "Fuguitt e Hawkins (1947) [15]; Box et al. (1973) [16]; valores da coleção COPS [17]"],
         ["Hes1 (oscilador de Goodwin)", "mRNA de Hes1 por PCR quantitativo, 8 instantes de 30 a 240 min", "Silk et al. (2011) [18]; modelo de Filippi et al. (2013) [19]; valores transcritos de Schmitt et al. (2024) [20]"],
         ["Pêndulo (captura de movimento)", "Ângulo por captura óptica a 200 Hz por 3,2 s; 130 instantes", "Giessler (2025) [21]"],
         ["FitzHugh–Nagumo (disparo médio, rato)", "Média de 64 disparos alinhados de um registro extracelular de radícula dorsal; 50 instantes", "Metcalfe (2020) [22]; dados usados por Samson, Tamborrino e Tubikanec (2025) [23]"],
         ["Advecção–dispersão (traçador em esgoto)", "Condutividade a montante e a jusante de um trecho de 469 m, traçador NaCl; 56 medidas", "Rieckermann et al. (2005) [24]; arquivo no Zenodo [25]"],
         ["16 problemas PEtab", "Medições, observáveis e condições experimentais de cada estudo (arquivo measurements.tsv)", "Coleção Benchmark-Models-PEtab [26, 27]; estudos de origem [28]. Os 8 primeiros de [28] são os deste relatório."]]
S += [tab(_real, [4.0, 6.6, 6.0], align_num=False, pad=1.7),
      P("Tabela F1. Os 30 modelos com dados reais e suas fontes. O arquivo dados/fontes.json do repositório traz, para cada conjunto, a referência, a URL, a data de obtenção e o sha256 do arquivo bruto.", CAP),
      P("Nos modelos numba com dados reais, o ruído é gaussiano na escala observada (logaritmo, raiz ou linear), com desvio fixo por grupo de saídas, igual ao desvio residual do ajuste de máxima "
        "verossimilhança corrigido pelos graus de liberdade. A posterior de referência e todos os métodos usam o mesmo desvio; a posterior é, portanto, condicional a ele, "
        "o que é uma limitação a declarar. Escolhas fixadas antes de rodar: sujeito 1 nos dois conjuntos farmacocinéticos; densidade inicial de 10 mil células por poço no Fisher–KPP; "
        "um infectado no dia 0 na gripe; experimento 2 no traçador; primeiro registro em repouso e canal 1 no rato."),
      P("<b>Conferências.</b> Três ajustes reproduzem a literatura: as taxas do Lotka–Volterra ficam nos valores do estudo de caso do Stan [2]; as cinco constantes do α-pineno "
        "(5,9; 3,0; 2,0; 27,4 e 4,0 em 10⁻⁵ por min) são as de Box et al. [16]; e a velocidade e a dispersão do traçador (0,38 m/s e 0,069 m²/s) são as de Rieckermann et al. [24]."),
      P("<b>Ressalvas dos modelos novos.</b> Entraram também modelos em que o ajuste é fraco ou o dado é processado, para ampliar o conjunto com dados reais; a ressalva de cada um acompanha o modelo no código:")]
S += bullets([
    "<b>Veilleux.</b> O modelo de Rosenzweig–MacArthur acerta o período dos ciclos, mas explica só 32% e 45% da variância das duas séries: é um modelo mal especificado. "
    "No pacote TSA os rótulos das colunas estão trocados (a série rotulada como Didinium antecede a outra em cerca de um dia e é, portanto, a presa); o dado curado usa os nomes corrigidos.",
    "<b>Gause.</b> O experimento tem imigração a cada três dias. Ela entra como um pulso suave logo após a contagem, com a mesma quantidade para as duas espécies, e essa quantidade é um parâmetro. "
    "Os valores foram digitalizados de uma figura [14]. O ajuste explica 74% e 89% da variância.",
    "<b>Rato.</b> O registro bruto é um trem de disparos irregulares, que uma equação determinística não descreve; Samson et al. [23] usam a versão estocástica do modelo. "
    "Aqui se ajusta a forma média do disparo, com escala de tempo e amplitude livres. O ajuste explica 93% da variância, um dos parâmetros fica na borda do prior, e os resíduos de uma média são autocorrelacionados.",
    "<b>Hes1.</b> São 8 medidas para 4 parâmetros. Os valores foram transcritos de [20], que reproduz os dados de [18]; a transcrição não foi conferida contra a fonte primária.",
    "<b>α-pineno.</b> Box et al. [16] mostram dependências lineares entre as respostas: parte dos valores publicados foi calculada, não medida."])
S += [P("F.2 Por que os demais modelos usaram dados simulados", H2),
      P("Dado simulado só se justifica quando cumpre uma função que o dado real não cumpre. Nesta execução, as razões foram quatro:")]
S += bullets([
    "<b>Verdade conhecida.</b> Com o valor verdadeiro dos parâmetros conhecido, é possível checar a posterior de referência e, mais adiante, medir cobertura (SBC). Com dados reais não há verdade.",
    "<b>Modelo bem especificado.</b> O dado é gerado pelo próprio modelo, com ruído do tipo suposto na verossimilhança. Assim, a dificuldade medida é a do método de inferência, "
    "não a de um modelo que não descreve o experimento.",
    "<b>Dificuldade controlada.</b> Os modelos difíceis foram desenhados para isolar uma patologia por vez: rigidez (Robertson, HIRES), caos (Lorenz), multimodalidade por simetria "
    "(cadeia de decaimento), não identificabilidade (decaimento com escala, Fisher–KPP num instante), problema mal-posto (calor inverso, Darcy). Dados reais misturam essas patologias.",
    "<b>Falta de dados públicos adequados.</b> Para vários sistemas não há experimento correspondente (Lorenz-96, Robertson e HIRES são sistemas conceituais ou de teste numérico) "
    "ou a busca só encontrou conjuntos simulados (Burgers, Allen–Cahn, onda, Darcy, Lorenz-63)."])
S += [P("Essas razões têm um limite. A comparação entre métodos é feita contra a posterior de referência do mesmo dado, de modo que a origem do dado não favorece nenhum método. "
        "Mas o dado simulado não tem má especificação do modelo, nem ruído não gaussiano, nem efeitos de medição, e por isso os resultados dos 33 modelos sintéticos "
        "<b>não devem ser lidos como desempenho em dados reais</b>. Daqui em diante, eles formam uma trilha de validação separada, fora das estatísticas principais."),
      P("F.3 Situação de cada modelo sintético", H2)]
_sint = [["Modelo sintético", "Situação", "Dados reais: substituto, ou resultado da busca"],
         ["Lotka–Volterra", "Substituído", "Lebre e lince [1, 2]."],
         ["SEIR; SEIR só no início", "Substituídos", "Gripe no internato [3, 4], completa e só os 5 primeiros dias."],
         ["PK de dois compartimentos; PK com flip-flop", "Substituídos", "Indometacina intravenosa [7]; teofilina oral [5]."],
         ["Fisher–KPP; Fisher–KPP num instante", "Substituídos", "Ensaio de arranhão [8, 9]: os 4 instantes, ou só o perfil de 48 h com a capacidade de suporte livre."],
         ["Repressilador", "Coberto", "O problema PEtab de Elowitz e Leibler (2000) [28] usa os dados reais do repressilador."],
         ["Rosenzweig–MacArthur", "Substituído, com ressalva", "Série de Veilleux [10–12] (ajuste fraco) e microcosmo de Gause com imigração [13, 14]."],
         ["FitzHugh–Nagumo", "Substituído, com ressalva", "Disparo médio de radícula dorsal de rato [22, 23]."],
         ["Goodwin (duas versões)", "Substituídos", "Hes1 [18–20]: o mesmo laço de retroalimentação negativa com coeficiente de Hill livre."],
         ["Pêndulo amortecido", "Substituído", "Pêndulo medido por captura de movimento [21]."],
         ["Advecção–difusão", "Substituído", "Traçador em trecho de esgoto [24, 25]; a curva a montante é a condição de entrada."],
         ["Cadeia de decaimento", "Substituído", "Isomerização do α-pineno [15–17], uma cadeia de reações de primeira ordem."],
         ["Oscilador forçado", "Sem dado obtido", "O conjunto do pêndulo [21] é de oscilação livre; não se achou medida com forçamento conhecido."],
         ["Van der Pol", "Não obtido", "Circuito oscilador forçado no repositório Dryad (doi:10.5061/dryad.z34tmpg80): o download automático foi recusado (exige autenticação)."],
         ["Calor com fonte; calor inverso", "Não obtido", "Barra aquecida em túnel de vento (Tomanek e Stutts, 2022, Data in Brief): o acesso ao artigo foi recusado e o repositório não foi localizado."],
         ["Brusselator", "Sem dado público", "Há experimentos de Belousov–Zhabotinsky em artigos, sem série temporal pública pronta."],
         ["Lorenz-63 (duas versões)", "Sem dado público", "Só conjuntos simulados; a roda d'água de Malkus tem experimentos publicados, sem dados abertos."],
         ["Burgers; Allen–Cahn (duas versões); onda; Darcy (duas versões); FitzHugh–Nagumo com difusão", "Sem dado público",
          "A busca só encontrou conjuntos gerados por simulação. No Darcy, os próprios benchmarks da área são sintéticos."],
         ["Lorenz-96; Robertson; HIRES; decaimento com escala", "Sem experimento", "Sistemas conceituais ou de teste numérico, sem experimento de referência."]]
S += [tab(_sint, [4.6, 2.6, 9.4], align_num=False, pad=1.7),
      P("Tabela F2. Os 33 modelos sintéticos desta execução: 15 têm agora um modelo equivalente com dados reais no repositório e 18 não têm. "
        "Não obtido: a busca encontrou um conjunto de dados reais, mas ele não pôde ser baixado.", CAP),
      P("F.4 Como os dados simulados foram gerados", H2),
      P("Em cada modelo sintético, o dado é y = f(θ*) + σ ε, em que f é a solução do modelo nas saídas e nos instantes observados, θ* é o valor verdadeiro (apêndice A), "
        "ε é ruído gaussiano padrão com semente fixa derivada do nome do modelo, e σ é uma fração do desvio-padrão das saídas (0,10 por padrão; 0,03, 0,05 ou 0,20 em alguns modelos difíceis). "
        "Há uma única realização por modelo, a mesma para todos os métodos e sementes."),
      P("F.5 Referências dos dados", H2)]
_refs = [
    "Howard, P. (2009). <i>Modeling Basics</i>. Notas de aula da disciplina M442, Texas A&amp;M University.",
    "Carpenter, B. (2018). <i>Predator-prey population dynamics: the Lotka-Volterra model in Stan</i>. Estudo de caso do Stan, mc-stan.org. Arquivo hudson-bay-lynx-hare.csv do repositório stan-dev/example-models.",
    "Anonymous (1978). Influenza in a boarding school. <i>British Medical Journal</i> 1(6112).",
    "King, A. A., Nguyen, D. e Ionides, E. L. (2016). Statistical inference for partially observed Markov processes via the R package pomp. <i>Journal of Statistical Software</i> 69(12). Conjunto bsflu.",
    "Boeckmann, A. J., Sheiner, L. B. e Beal, S. L. (1994). <i>NONMEM Users Guide: Part V</i>. NONMEM Project Group, University of California, San Francisco. Estudo de R. Upton.",
    "R Core Team. <i>R: A Language and Environment for Statistical Computing</i>. Pacote datasets, conjuntos Theoph e Indometh; cópias em CSV do projeto Rdatasets.",
    "Kwan, K. C., Breault, G. O., Umbenhauer, E. R., McMahon, F. G. e Duggan, D. E. (1976). Kinetics of indomethacin absorption, elimination, and enterohepatic circulation in man. <i>Journal of Pharmacokinetics and Biopharmaceutics</i> 4, 255–280.",
    "Jin, W., Shah, E. T., Penington, C. J., McCue, S. W., Chopin, L. K. e Simpson, M. J. (2016). Reproducibility of scratch assays is affected by the initial degree of confluence: experiments, modelling and model selection. <i>Journal of Theoretical Biology</i> 390, 136–145.",
    "Lagergren, J. H., Nardini, J. T., Baker, R. E., Simpson, M. J. e Flores, K. B. (2020). Biologically-informed neural networks guide mechanistic modeling from sparse experimental data. <i>PLOS Computational Biology</i> 16, e1008462. Arquivo cell_density_profiles.npy do repositório jlager/BINNs.",
    "Veilleux, B. G. (1976). <i>The analysis of a predatory interaction between Didinium and Paramecium</i>. Dissertação de mestrado, University of Alberta.",
    "Jost, C. e Ellner, S. P. (2000). Testing for predator dependence in predator-prey dynamics: a non-parametric approach. <i>Proceedings of the Royal Society B</i> 267, 1611–1620.",
    "Chan, K.-S. e Ripley, B. <i>TSA: Time Series Analysis</i>. Pacote R (CRAN), conjunto veilleux.",
    "Gause, G. F. (1934). <i>The Struggle for Existence</i>. Williams and Wilkins, Baltimore. Figura 32.",
    "Mühlbauer, L. K., Schulze, M., Harpole, W. S. e Clark, A. T. (2020). gauseR: simple methods for fitting Lotka-Volterra models describing Gause's Struggle for Existence. <i>Ecology and Evolution</i> 10, 13275–13283. Conjunto gause_1934_book_f32.",
    "Fuguitt, R. E. e Hawkins, J. E. (1947). Rate of the thermal isomerization of α-pinene in the liquid phase. <i>Journal of the American Chemical Society</i> 69, 319–322.",
    "Box, G. E. P., Hunter, W. G., MacGregor, J. F. e Erjavec, J. (1973). Some problems associated with the analysis of multiresponse data. <i>Technometrics</i> 15, 33–51.",
    "Dolan, E. D., Moré, J. J. e Munson, T. S. (2004). <i>Benchmarking optimization software with COPS 3.0</i>. Relatório ANL/MCS-TM-273, Argonne National Laboratory. Problema pinene; arquivo do repositório MadNLP/COPSBenchmark.jl.",
    "Silk, D., Kirk, P. D. W., Barnes, C. P., Toni, T., Rose, A., Moon, S., Dallman, M. J. e Stumpf, M. P. H. (2011). Designing attractive models via automated identification of chaotic and oscillatory dynamical regimes. <i>Nature Communications</i> 2, doi:10.1038/ncomms1496.",
    "Filippi, S., Barnes, C. P., Cornebise, J. e Stumpf, M. P. H. (2013). On optimality of kernels for approximate Bayesian computation using sequential Monte Carlo. <i>Statistical Applications in Genetics and Molecular Biology</i> 12, 87–107.",
    "Schmitt, M. et al. (2024). Leveraging self-consistency for data-efficient amortized Bayesian inference. arXiv:2310.04395, apêndice E.",
    "Giessler, M. (2025). <i>Kinematic data of a pendulum serving as an angular acceleration reference</i>. figshare, conjunto de dados, doi:10.6084/m9.figshare.28271405.v1. Licença CC BY 4.0.",
    "Metcalfe, B. (2020). <i>Action potentials recorded from the L5 dorsal rootlet of rat using a multiple electrode array</i>. Mendeley Data, v1, doi:10.17632/ybhwtngzmm.1. Licença CC BY 4.0.",
    "Samson, A., Tamborrino, M. e Tubikanec, I. (2025). Inference for the stochastic FitzHugh-Nagumo model from real action potential data via approximate Bayesian computation. <i>Computational Statistics and Data Analysis</i> 204, 108095.",
    "Rieckermann, J., Neumann, M., Ort, C., Huisman, J. L. e Gujer, W. (2005). Dispersion coefficients of sewers from tracer experiments. <i>Water Science and Technology</i> 52(5), 123–133, doi:10.2166/wst.2005.0124.",
    "Rieckermann, J. e Chavarría Vargas, A. (2026). <i>Dispersion coefficients of sewers from tracer experiments: dataset</i>. Zenodo, doi:10.5281/zenodo.20831873. Licença CC0.",
    "Hass, H., Loos, C., Raimúndez-Álvarez, E., Timmer, J., Hasenauer, J. e Kreutz, C. (2019). Benchmark problems for dynamic modeling of intracellular processes. <i>Bioinformatics</i> 35, 3073–3082.",
    "Schmiester, L. et al. (2021). PEtab: interoperable specification of parameter estimation problems in systems biology. <i>PLOS Computational Biology</i> 17, e1008646. Coleção Benchmark-Models-PEtab, github.com/Benchmarking-Initiative/Benchmark-Models-PEtab.",
    "Estudos de origem dos 16 problemas PEtab, identificados na coleção [27] pelo primeiro autor, periódico e ano: Bertozzi et al. (PNAS, 2020); Boehm et al. (J. Proteome Res., 2014); "
    "Rahman et al. (Math. Biosci., 2016); Crauste et al. (Cell Systems, 2017); Bruno et al. (J. Exp. Bot., 2016); Sneyd e Dufour (PNAS, 2002); Fujita et al. (Sci. Signal., 2010); "
    "Elowitz e Leibler (Nature, 2000); Perelson et al. (Science, 1996); Salazar-Cavazos et al. (Mol. Biol. Cell, 2020); Blasi et al. (Cell Systems, 2016); Liu et al. (IFAC-PapersOnLine, 2025); "
    "Oliveira et al. (Nat. Commun., 2021); Laske et al. (PLOS Comput. Biol., 2019); Armistead et al. (Cell Death Dis., 2024); Okuonghae e Omame (Chaos Solitons Fractals, 2020).",
]
REF = ParagraphStyle("ref", parent=BODY, fontSize=8.1, leading=10.5, leftIndent=16, firstLineIndent=-16, spaceAfter=1.6, alignment=0)
S += [Paragraph(f"[{i}] {r}", REF) for i, r in enumerate(_refs, 1)]
S += [P("As referências foram montadas a partir dos metadados dos repositórios e dos pacotes de onde os dados foram baixados; devem ser conferidas nas fontes originais antes de citadas em texto final. "
        "Os títulos completos dos estudos de [28] estão na tabela da coleção [27].", CAP)]

doc = Doc(str(OUTPDF), pagesize=A4, title="A0 benchmark: relatório técnico dos resultados preliminares", author="", subject="Inferência bayesiana em EDOs e EDPs")
doc.multiBuild(S)
print("pdf ok", OUTPDF)
