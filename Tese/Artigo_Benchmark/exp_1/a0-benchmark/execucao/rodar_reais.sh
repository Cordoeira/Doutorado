#!/usr/bin/env bash
# Trilha de dados reais do A0 (preset "reais", sem o NPE-PFN), pronta para rodar no WSL.
#
#   bash execucao/rodar_reais.sh [A|B|C|D] [--plano]
#
#   D  completo (padrao): os 30 modelos com dados reais ............... 1 a 2,4 dias (ate ~6 dias)
#      = 14 numba + 6 PEtab de custo baixo + Laske + SalazarCavazos, e os 8 PEtab antigos reaproveitados
#   C  D sem o SalazarCavazos ......................................... 12 a 22 h (ate ~2,7 dias)
#   B  C sem o Laske .................................................. 2 a 7 h
#   A  so os 14 modelos numba ......................................... 1 a 2h30
#   (tempo de parede com 10 processos no laptop da execucao anterior; ver docs/estimativa_tempo_reais.md)
#
#   --plano  so confere o ambiente e mostra o que seria rodado; nao roda nada.
#
# Os 8 PEtab ja rodados na execucao completa de 13 a 16/11 (resultados_a0) sao copiados para a pasta de saida e
# reaproveitados. O Fujita nao entra na lista de `rodar`: a identificabilidade dele recomecaria, e ela levou mais
# de 25 h sem terminar. Retomada: rode o mesmo comando de novo; o que ja terminou e pulado.
#
# Variaveis (todas opcionais):
#   A0_VENV               venv do pacote (padrao ~/.venvs/a0bench)
#   A0_RUNS               pasta das execucoes (padrao ~/a0bench_runs)
#   A0_SAIDA              saida desta execucao (padrao $A0_RUNS/resultados_reais)
#   A0_ANTIGOS            execucao anterior com os 8 PEtab (padrao $A0_RUNS/resultados_a0)
#   A0_MODELOS_AMICI      cache dos modelos AMICI compilados (padrao $A0_ANTIGOS/_amici_models)
#   A0_PROCESSOS          processos em paralelo (padrao 10)
#   A0_SEM_IDENT_GRANDES  1 = Laske e SalazarCavazos sem identificabilidade (corta cerca de 1/3 do tempo deles)
#   A0_REFAZER_ANTIGOS    1 = roda tambem os PEtab antigos do zero, se A0_ANTIGOS nao existir (mais 1 a 2 dias)
#   A0_TESTE              1 = ensaio do lancador: orcamento pequeno, 3 modelos baratos e 1 PEtab antigo copiado,
#                         em $A0_RUNS/teste_lancador (uns 5 minutos; nao toca na saida real)
set -u
CEN=D
PLANO=0
for a in "$@"; do
  case "$a" in
    A|B|C|D) CEN=$a ;;
    --plano) PLANO=1 ;;
    *) echo "uso: bash execucao/rodar_reais.sh [A|B|C|D] [--plano]"; exit 2 ;;
  esac
done

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${A0_VENV:-$HOME/.venvs/a0bench}
RUNS=${A0_RUNS:-$HOME/a0bench_runs}
ANTIGOS=${A0_ANTIGOS:-$RUNS/resultados_a0}
AMICI=${A0_MODELOS_AMICI:-$ANTIGOS/_amici_models}
PROC=${A0_PROCESSOS:-10}

NUMBA=lebre_lince,gripe_internato,gripe_internato_inicio,pk_teofilina,pk_indometacina,fisher_kpp_jin,fisher_kpp_jin_instante,didinium_veilleux,didinium_gause,alfa_pineno,hes1,pendulo_mocap,fhn_rato,tracador_esgoto
PETAB_NOVOS=Perelson_Science1996,Blasi_CellSystems2016,Liu_IFACPapersOnLine2025,Oliveira_NatCommun2021,Armistead_CellDeathDis2024,Okuonghae_ChaosSolitonsFractals2020
ANTIGOS_PETAB="Bertozzi_PNAS2020 Boehm_JProteomeRes2014 Rahman_MBS2016 Crauste_CellSystems2017 Bruno_JExpBot2016 Sneyd_PNAS2002 Elowitz_Nature2000 Fujita_SciSignal2010"
case $CEN in
  A) PEQ=$NUMBA; GRANDES="" ;;
  B) PEQ=$NUMBA,$PETAB_NOVOS; GRANDES="" ;;
  C) PEQ=$NUMBA,$PETAB_NOVOS; GRANDES=Laske_PLOSComputBiol2019 ;;
  D) PEQ=$NUMBA,$PETAB_NOVOS; GRANDES=Laske_PLOSComputBiol2019,SalazarCavazos_MBoC2020 ;;
esac

TESTE=${A0_TESTE:-0}
if [ "$TESTE" = 1 ]; then
  PRESET=rapido
  PEQ=pk_indometacina,pendulo_mocap
  GRANDES=hes1                       # faz o papel dos modelos grandes, para exercitar o ramo em paralelo
  ANTIGOS_PETAB="Bertozzi_PNAS2020"
  PREF=teste_lancador
  SAIDA=${A0_SAIDA:-$RUNS/teste_lancador}
  MARCA=$RUNS/TESTE_LANCADOR.done
else
  PRESET=reais
  PREF=reais
  SAIDA=${A0_SAIDA:-$RUNS/resultados_reais}
  MARCA=$RUNS/REAIS.done
fi
LOG=$RUNS/${PREF}_lancador.log
ZIP=$RUNS/$(basename "$SAIDA").zip

falha() { echo "ERRO: $*"; exit 1; }
echo "== A0, trilha de dados reais: cenario $CEN"
[ "$TESTE" = 1 ] && echo "   (ENSAIO do lancador: orcamento pequeno, poucos modelos; a saida real nao e tocada)"
[ "$PLANO" = 1 ] && echo "   (plano: so confere e mostra; nada sera rodado)"
echo "   repositorio  $REPO"
echo "   venv         $VENV"
echo "   saida        $SAIDA"
echo "   antigos      $ANTIGOS"
echo "   cache AMICI  $AMICI"
echo "   processos    $PROC"
[ -f "$VENV/bin/activate" ] || falha "venv nao encontrado em $VENV (veja a secao Instalacao do README)"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO" || falha "nao consegui entrar em $REPO"
python -c "import os, a0bench; print('   pacote       ' + os.path.dirname(a0bench.__file__))"
A0_PEQ="$PEQ" A0_GRANDES="$GRANDES" python "$REPO/execucao/conferir_selecao.py" || falha "selecao de modelos invalida"

# PEtab antigos: reaproveitados da execucao anterior
ANT_OK=1
for m in $ANTIGOS_PETAB; do
  [ -f "$ANTIGOS/$m/referencia.npz" ] || ANT_OK=0
done
if [ "$ANT_OK" = 1 ]; then
  echo "   PEtab antigos: $(echo $ANTIGOS_PETAB | wc -w) em $ANTIGOS, serao reaproveitados"
elif [ "${A0_REFAZER_ANTIGOS:-0}" = 1 ]; then
  echo "   PEtab antigos: ausentes; com A0_REFAZER_ANTIGOS=1 os 7 menores entram na lista (o Fujita fica de fora)"
  PEQ=$PEQ,Bertozzi_PNAS2020,Boehm_JProteomeRes2014,Rahman_MBS2016,Crauste_CellSystems2017,Bruno_JExpBot2016,Sneyd_PNAS2002,Elowitz_Nature2000
else
  falha "os PEtab antigos nao estao em $ANTIGOS. Aponte A0_ANTIGOS para eles ou use A0_REFAZER_ANTIGOS=1"
fi
if [ "$TESTE" != 1 ]; then
  FALTA_AMICI=""
  for m in $(echo "$PEQ,$GRANDES" | tr ',' ' '); do
    case " $(echo "$PETAB_NOVOS,Laske_PLOSComputBiol2019,SalazarCavazos_MBoC2020" | tr ',' ' ') " in
      *" $m "*) ls -d "$AMICI"/*/"$m" >/dev/null 2>&1 || FALTA_AMICI="$FALTA_AMICI $m" ;;
    esac
  done
  if [ -z "$FALTA_AMICI" ]; then
    echo "   cache AMICI: os PEtab novos ja estao compilados"
  else
    echo "   cache AMICI: serao compilados na hora (20 s a 1,5 min cada):$FALTA_AMICI"
  fi
fi

COMUM="--preset $PRESET --sem-pfn --sem-monitor --saida $SAIDA --pasta-modelos $AMICI"
EXTRA=""
[ "${A0_SEM_IDENT_GRANDES:-0}" = 1 ] && EXTRA=--sem-identificabilidade
NG=0
[ -n "$GRANDES" ] && NG=2
NP=$(( PROC > NG + 1 ? PROC - NG : 1 ))
echo
echo "Comandos:"
[ -n "$GRANDES" ] && echo "  python -m a0bench rodar $COMUM --processos 2 --modelos $GRANDES $EXTRA   (em paralelo com o seguinte)"
echo "  python -m a0bench rodar $COMUM --processos $NP --modelos $PEQ"
echo "  python -m a0bench relatorio --saida $SAIDA --sem-latex --processos 8"
echo "  python -m a0bench exportar --saida $SAIDA --destino $ZIP"
echo "Registros: $LOG, $RUNS/$PREF.log; marcador de fim: $MARCA"
if [ "$PLANO" = 1 ]; then
  echo
  echo "Plano conferido. Para rodar: tire o --plano, ou use executar_reais.bat no Windows."
  exit 0
fi

mkdir -p "$SAIDA"
exec >> "$LOG" 2>&1
export TERM=dumb COLUMNS=160
echo "=== inicio $(date) cenario $CEN"
rm -f "$MARCA"
if [ "$ANT_OK" = 1 ]; then
  for m in $ANTIGOS_PETAB; do
    if [ ! -d "$SAIDA/$m" ]; then
      cp -a "$ANTIGOS/$m" "$SAIDA/$m" && rm -f "$SAIDA/$m"/*.lock && echo "copiado da execucao anterior: $m"
    fi
  done
fi
PG=""
if [ -n "$GRANDES" ]; then
  # shellcheck disable=SC2086
  python -m a0bench rodar $COMUM --processos 2 --modelos "$GRANDES" $EXTRA > "$RUNS/${PREF}_grandes.log" 2>&1 &
  PG=$!
  echo "grandes em paralelo: pid $PG ($GRANDES)"
  sleep 20
fi
# shellcheck disable=SC2086
python -m a0bench rodar $COMUM --processos "$NP" --modelos "$PEQ" > "$RUNS/$PREF.log" 2>&1
echo "RODAR_EXIT=$? $(date)"
if [ -n "$PG" ]; then
  wait "$PG"
  echo "RODAR_GRANDES_EXIT=$? $(date)"
fi
echo "=== relatorio $(date)"
python -m a0bench relatorio --saida "$SAIDA" --sem-latex --processos 8 > "$RUNS/${PREF}_relatorio.log" 2>&1
echo "RELATORIO_EXIT=$? $(date)"
echo "=== exportar $(date)"
python -m a0bench exportar --saida "$SAIDA" --destino "$ZIP" > "$RUNS/${PREF}_exportar.log" 2>&1
echo "EXPORTAR_EXIT=$? $(date)"
echo "=== fim $(date)"
touch "$MARCA"
