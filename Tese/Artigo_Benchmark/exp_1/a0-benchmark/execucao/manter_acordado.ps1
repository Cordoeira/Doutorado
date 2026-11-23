# Mantem o Windows acordado (sem suspensao automatica) enquanto o benchmark A0 roda no WSL.
# Nao altera nenhuma configuracao de energia: usa SetThreadExecutionState, que so vale enquanto este processo existir.
# Termina sozinho quando o arquivo -Marcador aparecer ou quando o arquivo -Parar for criado.
#   powershell -NoProfile -ExecutionPolicy Bypass -File manter_acordado.ps1 -Marcador "\\wsl.localhost\Ubuntu-24.04\home\USUARIO\a0bench_runs\REAIS.done"
param(
    [Parameter(Mandatory = $true)] [string]$Marcador,
    [string]$Parar = "$env:TEMP\a0bench_parar_acordado"
)
Add-Type -Name Pwr -Namespace A0 -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]0x80000000
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
Remove-Item -Force -ErrorAction SilentlyContinue $Parar
while (-not (Test-Path $Marcador) -and -not (Test-Path $Parar)) {
    [A0.Pwr]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
    Start-Sleep -Seconds 60
}
[A0.Pwr]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
