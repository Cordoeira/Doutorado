#!/bin/bash
# Plano com prazo: NPE-PFN (1 semente, 18 modelos de baixa dimensao, 9 processos) + identificabilidade de
# Elowitz e Sneyd (2 processos) em paralelo; as 05:30 encerra o que faltar e fecha relatorio, estatisticas e exportacao.
R=~/a0bench_runs
SAIDA=$R/resultados_a0
SCR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../analises" && pwd)
exec >> "$R/prazo.log" 2>&1
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # raiz do repositorio
source ~/.venvs/a0bench/bin/activate
export TERM=dumb COLUMNS=160
PRAZO=$(date -d "2026-11-16 05:30" +%s)
MODELOS=adveccao_difusao,allen_cahn,brusselator,burgers,calor,fisher_kpp,lorenz96,onda,pendulo,van_der_pol,robertson,lorenz63,oscilador,fitzhugh_nagumo,seir,allen_cahn_meta,fisher_frente,produto
echo "=== inicio $(date); prazo $(date -d @$PRAZO)"

python -m a0bench rodar --preset completo --saida "$SAIDA" --sem-monitor --processos 9 --modelos $MODELOS \
    --metodos npe_pfn --sementes 0 --sem-identificabilidade > "$R/a0_pfn_prazo.log" 2>&1 &
P1=$!
sleep 30
python -m a0bench identificar --preset completo --saida "$SAIDA" --processos 2 \
    --modelos Elowitz_Nature2000,Sneyd_PNAS2002 > "$R/ident_prazo.log" 2>&1 &
P2=$!
echo "pids: pfn=$P1 ident=$P2"

while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do
  [ "$(date +%s)" -ge "$PRAZO" ] && break
  sleep 30
done
for P in $P1 $P2; do
  if kill -0 $P 2>/dev/null; then echo "prazo atingido: encerrando pid $P ($(date))"; kill -TERM $P; fi
done
for i in $(seq 1 60); do { kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; } || break; sleep 2; done
pkill -9 -f spawn_main 2>/dev/null; sleep 3
rm -f "$SAIDA"/*/em_execucao.lock
echo "=== processamento encerrado $(date)"

echo "=== relatorio $(date)"
python -m a0bench relatorio --saida "$SAIDA" --sem-latex --processos 10 > "$R/relatorio_prazo.log" 2>&1
echo "RELATORIO_EXIT=$? $(date)"
echo "=== resumos $(date)"
python "$SCR/pfn_resumo.py" "$SAIDA" > "$SAIDA/relatorio/npe_pfn_resumo.txt" 2>&1
python "$SCR/stats_artigo.py" "$SAIDA" > "$SAIDA/relatorio/estatisticas_finais.txt" 2>&1
echo "=== exportar $(date)"
python -m a0bench exportar --saida "$SAIDA" --destino "$R/resultados_a0_final.zip" > "$R/exportar_prazo.log" 2>&1
echo "EXPORTAR_EXIT=$? $(date)"
echo "=== FIM $(date)"
touch "$R/PRAZO.done" "$R/COMPLETO.done"
