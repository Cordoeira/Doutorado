@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem Executa a trilha de dados reais do benchmark A0 no WSL. Detalhes em execucao\LEIA-ME.md.
rem
rem   executar_reais.bat            execucao COMPLETA (cenario D): confere, pergunta e lanca
rem   executar_reais.bat B          escolhe outro cenario
rem   executar_reais.bat /plano     so confere o ambiente e mostra o que seria rodado
rem
rem   D = completo: os 30 modelos com dados reais ..... 1 a 2,4 dias (ate uns 6 dias)
rem       14 numba + 6 PEtab de custo baixo + Laske + SalazarCavazos; os 8 PEtab antigos sao reaproveitados
rem   C = D sem o SalazarCavazos ...................... 12 a 22 h (ate uns 2,7 dias)
rem   B = C sem o Laske ............................... 2 a 7 h
rem   A = so os 14 modelos numba ...................... 1 a 2h30
rem
rem A execucao roda desacoplada desta janela (pode fechar) e mantem o Windows acordado ate o fim.
rem Rodar de novo o mesmo comando retoma de onde parou. Deixe o computador na tomada.
rem ---------------------------------------------------------------------------------------------
set "CEN=%~1"
set "SOPLANO="
if "%CEN%"=="" set "CEN=D"
if /i "%CEN%"=="/plano" (
    set "CEN=D"
    set "SOPLANO=1"
)
if /i "%~2"=="/plano" set "SOPLANO=1"

set "OK="
for %%c in (A B C D) do if /i "%CEN%"=="%%c" set "OK=1"
if not defined OK (
    echo Cenario invalido: %CEN%
    echo Uso: executar_reais.bat [A^|B^|C^|D] [/plano]
    goto fim
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0execucao\lancar_reais.ps1" -Cenario %CEN% -Plano
if errorlevel 1 (
    echo.
    echo O plano falhou. Nada foi lancado.
    goto fim
)
if defined SOPLANO goto fim_sem_pausa

echo.
choice /C SN /M "Lancar agora a execucao do cenario %CEN%"
if errorlevel 2 (
    echo Nada foi lancado.
    goto fim
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0execucao\lancar_reais.ps1" -Cenario %CEN%
echo.
echo Para acompanhar: acompanhar_reais.bat

:fim
echo.
pause
:fim_sem_pausa
endlocal
