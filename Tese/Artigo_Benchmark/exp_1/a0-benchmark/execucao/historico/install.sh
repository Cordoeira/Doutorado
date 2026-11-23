#!/bin/bash
# roda desacoplado: nohup dentro do WSL
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # raiz do repositorio
source ~/.venvs/a0bench/bin/activate
rm -f ~/a0bench_install.done
nohup bash -c '
  echo "=== pip install lock $(date)"
  pip install -r requirements-lock.txt; echo "LOCK_EXIT=$?"
  echo "=== pip install -e . $(date)"
  pip install -e .; echo "EDIT_EXIT=$?"
  echo "=== DONE $(date)"
  touch ~/a0bench_install.done
' > ~/a0bench_install.log 2>&1 &
echo "launched pid $!"
