#!/bin/bash
# Execução completa do A0 (preset completo), fases 3a e 3b do roteiro do README, sem NPE-PFN.
# Saída no ext4 do WSL (rápido); acessível do Windows em \wsl.localhost\Ubuntu-24.04\home\pedrocordoeira\a0bench_runs
R=~/a0bench_runs; mkdir -p "$R"
exec > >(tee -a "$R/lancador.log") 2>&1
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # raiz do repositorio
source ~/.venvs/a0bench/bin/activate
export TERM=dumb COLUMNS=160
SAIDA="$R/resultados_a0"
COMUM="--preset completo --saida $SAIDA --sem-pfn --sem-monitor --processos 10"
echo "=== FASE 1 (facil,dificil,petab) inicio $(date)"
python -m a0bench rodar $COMUM --grupos facil,dificil,petab > "$R/a0.log" 2>&1
echo "FASE1_EXIT=$? $(date)"
echo "=== FASE 2 (petab_grande) inicio $(date)"
python -m a0bench rodar $COMUM --grupos petab_grande > "$R/a0_grande.log" 2>&1
echo "FASE2_EXIT=$? $(date)"
echo "=== FIM $(date)"
touch "$R/COMPLETO.done"
