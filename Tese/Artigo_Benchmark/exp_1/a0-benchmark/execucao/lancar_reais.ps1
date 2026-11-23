# Lanca a trilha de dados reais do A0 no WSL, desacoplada deste terminal (sobrevive ao fechamento da janela),
# e mantem o Windows acordado ate o fim. Rode no PowerShell, na pasta do repositorio:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File execucao\lancar_reais.ps1 -Plano        # so confere
#   powershell -NoProfile -ExecutionPolicy Bypass -File execucao\lancar_reais.ps1               # roda o completo (D)
#   powershell -NoProfile -ExecutionPolicy Bypass -File execucao\lancar_reais.ps1 -Cenario B    # outro cenario
#   powershell -NoProfile -ExecutionPolicy Bypass -File execucao\lancar_reais.ps1 -Teste        # ensaio de 5 minutos
#
# Cenarios: D (completo, padrao: os 30 modelos com dados reais), C (sem o SalazarCavazos), B (sem o Laske tambem),
# A (so os 14 modelos numba). Detalhes, tempos e variaveis em execucao/LEIA-ME.md.
# -Teste: ensaia este mesmo caminho com orcamento pequeno e poucos modelos, em ~/a0bench_runs/teste_lancador.
param(
    [ValidateSet("A", "B", "C", "D")] [string]$Cenario = "D",
    [string]$Distro = "Ubuntu-24.04",
    [switch]$Plano,
    [switch]$Teste,
    [switch]$SemManterAcordado
)
$ErrorActionPreference = "Stop"
$repoWin = Split-Path -Parent $PSScriptRoot
$repoWsl = (& wsl.exe -d $Distro -- wslpath -a ($repoWin -replace '\\', '/')).Trim()
$script = "$repoWsl/execucao/rodar_reais.sh"
Write-Host "Repositorio no WSL: $repoWsl"
$marca = "REAIS.done"
if ($Teste) {
    # a variavel A0_TESTE passa para o WSL (e para os processos lancados daqui) pela lista WSLENV
    $env:A0_TESTE = "1"
    $env:WSLENV = (@($env:WSLENV, "A0_TESTE") | Where-Object { $_ }) -join ":"
    $marca = "TESTE_LANCADOR.done"
}

if ($Plano) {
    & wsl.exe -d $Distro -- bash $script $Cenario --plano
    exit $LASTEXITCODE
}

# confere antes de lancar (o mesmo modo plano); se falhar, nada e lancado
& wsl.exe -d $Distro -- bash $script $Cenario --plano
if ($LASTEXITCODE -ne 0) { Write-Host "O plano falhou; nada foi lancado."; exit $LASTEXITCODE }

$homeWsl = (& wsl.exe -d $Distro -- bash -c 'echo $HOME').Trim()
$runsWsl = "$homeWsl/a0bench_runs"
& wsl.exe -d $Distro -- rm -f "$runsWsl/$marca"
$marcador = "\\wsl.localhost\$Distro" + ($runsWsl -replace '/', '\') + "\$marca"

if (-not $SemManterAcordado) {
    $acordado = Join-Path $PSScriptRoot "manter_acordado.ps1"
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$acordado`"", "-Marcador", "`"$marcador`"")
    Write-Host "Windows mantido acordado ate aparecer $marcador"
}
Start-Process wsl.exe -WindowStyle Hidden -ArgumentList @("-d", $Distro, "--", "bash", "`"$script`"", $Cenario)
Write-Host ""
if ($Teste) {
    Write-Host "Ensaio lancado. Fim: $runsWsl/$marca ; registro: $runsWsl/teste_lancador_lancador.log"
} else {
    Write-Host "Execucao lancada (cenario $Cenario). Pode fechar este terminal."
    Write-Host "Acompanhar: acompanhar_reais.bat (na raiz do repositorio)"
    Write-Host "  registro: $runsWsl/reais_lancador.log  (fim: $runsWsl/REAIS.done)"
    Write-Host "Interromper: crie $env:TEMP\a0bench_parar_acordado e, no WSL, pkill -f 'a0bench rodar'"
}
