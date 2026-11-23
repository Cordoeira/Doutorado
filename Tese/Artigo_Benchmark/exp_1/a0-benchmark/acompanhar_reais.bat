@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem Acompanha a execucao da trilha de dados reais (lancada por executar_reais.bat):
rem mostra o fim do registro e abre o painel do a0bench. Ctrl+C fecha o painel; a execucao continua.
rem ---------------------------------------------------------------------------------------------
set "DISTRO=Ubuntu-24.04"
set "REPO=%~dp0"
set "REPO=%REPO:~0,-1%"
for /f "usebackq delims=" %%i in (`wsl.exe -d %DISTRO% -- wslpath -a "%REPO:\=/%"`) do set "REPOWSL=%%i"

wsl.exe -d %DISTRO% -- bash -c "test -d ~/a0bench_runs/resultados_reais"
if errorlevel 1 (
    echo A execucao ainda nao foi lancada. Use executar_reais.bat.
    goto fim
)

echo Fim do registro (~/a0bench_runs/reais_lancador.log no WSL):
wsl.exe -d %DISTRO% -- bash -c "tail -n 12 ~/a0bench_runs/reais_lancador.log 2>/dev/null"
echo.

wsl.exe -d %DISTRO% -- bash -c "test -f ~/a0bench_runs/REAIS.done"
if not errorlevel 1 (
    echo A execucao terminou.
    echo Resultados: \\wsl.localhost\%DISTRO%\home\...\a0bench_runs\resultados_reais
    echo Pacote:     resultados_reais.zip na mesma pasta a0bench_runs
    goto fim
)

echo Abrindo o painel. Ctrl+C fecha o painel; a execucao continua.
wsl.exe -d %DISTRO% -- bash -lc "source ~/.venvs/a0bench/bin/activate && cd '%REPOWSL%' && python -m a0bench monitor --saida ~/a0bench_runs/resultados_reais"

:fim
echo.
pause
endlocal
