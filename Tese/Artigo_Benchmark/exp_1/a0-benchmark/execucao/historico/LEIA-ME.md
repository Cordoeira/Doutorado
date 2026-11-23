# Scripts da execução completa de 13 a 16/11/2026

Guardados como registro do que foi rodado. Os caminhos do repositório viraram relativos; os da pasta de execução (`~/a0bench_runs`) continuam os da máquina original.

| Script | O que fez |
|---|---|
| `install.sh` | Instalação do venv no WSL (`requirements-lock.txt` e `pip install -e .`), desacoplada do terminal. |
| `ca_setup.sh` | Pacote de CAs do venv com a raiz do proxy Zscaler (`bash ca_setup.sh <certificado.crt>`), para pip, git, curl e o download dos pesos do TabPFN. |
| `rodar_completo.sh` | Fases 1 (facil, dificil, petab) e 2 (petab_grande) do preset completo, sem NPE-PFN, 10 processos. |
| `rodar_final.sh` | PEtab grandes sem identificabilidade, curva do NPE, relatório, exportação e tentativa do NPE-PFN completo. |
| `rodar_prazo.sh` | Plano para o prazo das 7 h de 16/11: NPE-PFN com 1 semente em 18 modelos e identificabilidade de Elowitz e Sneyd em paralelo, corte às 5h30, relatório e exportação final. |
