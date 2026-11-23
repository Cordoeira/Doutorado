#!/bin/bash
# Fase 2b (PEtab grandes sem identificabilidade) -> relatorio -> exportar -> RESULTADOS.done -> NPE-PFN -> relatorio -> COMPLETO.done
R=~/a0bench_runs
exec >> "$R/lancador.log" 2>&1
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # raiz do repositorio
source ~/.venvs/a0bench/bin/activate
export TERM=dumb COLUMNS=160
SAIDA="$R/resultados_a0"
COMUM="--preset completo --saida $SAIDA --sem-monitor --processos 10"

echo "=== FASE 2b (petab_grande, sem identificabilidade) inicio $(date)"
python -m a0bench rodar $COMUM --sem-pfn --grupos petab_grande --sem-identificabilidade > "$R/a0_grande2.log" 2>&1
echo "FASE2B_EXIT=$? $(date)"

echo "=== CURVA NPE (modelos novos) $(date)"
python "$SAIDA/relatorio/npe_curva.py" "$SAIDA" --processos 4 > "$R/npe_curva2.log" 2>&1
echo "NPE_CURVA2_EXIT=$? $(date)"

echo "=== RELATORIO (41 modelos) $(date)"
python -m a0bench relatorio --saida "$SAIDA" --sem-latex --processos 8 > "$R/relatorio_final.log" 2>&1
echo "RELATORIO_EXIT=$? $(date)"

echo "=== EXPORTAR $(date)"
python -m a0bench exportar --saida "$SAIDA" > "$R/exportar.log" 2>&1
echo "EXPORTAR_EXIT=$? $(date)"
touch "$R/RESULTADOS.done"

echo "=== VERIFICAR --pfn $(date)"
python -m a0bench verificar --pfn --sem-petab > "$R/verificar_pfn.log" 2>&1
echo "VERIFICAR_PFN_EXIT=$? $(date)"
if grep -q "Instalação pronta" "$R/verificar_pfn.log"; then
  echo "=== NPE-PFN nos 41 modelos $(date)"
  python -m a0bench rodar $COMUM --metodos npe_pfn > "$R/a0_pfn.log" 2>&1
  echo "PFN_EXIT=$? $(date)"
  echo "=== RELATORIO final com PFN $(date)"
  python -m a0bench relatorio --saida "$SAIDA" --sem-latex --processos 8 > "$R/relatorio_pfn.log" 2>&1
  echo "RELATORIO_PFN_EXIT=$? $(date)"
  python -m a0bench exportar --saida "$SAIDA" > "$R/exportar2.log" 2>&1
  echo "EXPORTAR2_EXIT=$? $(date)"
else
  echo "NPE-PFN nao verificado; pulado"
fi
echo "=== FIM $(date)"
touch "$R/COMPLETO.done"
