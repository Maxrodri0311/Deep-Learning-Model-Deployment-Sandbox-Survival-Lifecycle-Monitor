@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: INETUM DEEP LEARNING MODEL DEPLOYMENT SANDBOX (GP-029)
:: Automated Execution & Verification Script (Windows Host Physics Standard)
:: ============================================================================

:: 1. Forzar codificación UTF-8 inmutable para evitar caídas cp1252
chcp 65001 > nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

title "GP-029: Inetum Deep Learning Sandbox & Survival Monitor"

echo ================================================================================
echo   🚀 INETUM DEEP LEARNING MODEL DEPLOYMENT SANDBOX (GP-029)
echo   Lifecycle Observability, Kaplan-Meier Estimator & Cox Hazard Modeling
echo ================================================================================
echo.

:: 2. Detección determinista del intérprete de Python (Anti-Stub Resolution)
set "PY_CMD="
py -3 --version > nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=py -3"
) else (
    python --version > nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "PY_CMD=python"
    ) else (
        echo [ERROR] No se detecto un interprete de Python 3 valido en el PATH.
        exit /b 1
    )
)

echo [*] Interprete Python resuelto: %PY_CMD%
echo.

:: 3. Paso 1: Generación de Datos Sintéticos Realistas (50,000 registros)
echo [Paso 1/4] Generando telemetría de inferencia estocástica con DuckDB/Parquet...
%PY_CMD% src/data_generator.py --records 50000
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el generador de datos sinteticos.
    exit /b %ERRORLEVEL%
)
echo.

:: 4. Paso 2: Ejecución del Motor Analítico Core (Kaplan-Meier & Cox Proportional Hazards)
echo [Paso 2/4] Ejecutando el Motor Analítico de Supervivencia y Telemetría DuckDB...
%PY_CMD% src/core_engine.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en la ejecucion del motor analitico.
    exit /b %ERRORLEVEL%
)
echo.

:: 5. Paso 3: Suite Exhaustiva de Pruebas Pytest (13 Tests)
echo [Paso 3/4] Ejecutando suite de pruebas automatizadas con Pytest...
%PY_CMD% -m pytest tests/test_suite.py -v
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallaron una o mas pruebas en la suite de Pytest.
    exit /b %ERRORLEVEL%
)
echo.

:: 6. Paso 4: Benchmarks Cuantitativos de Rendimiento Local
echo [Paso 4/4] Midiendo latencias reales p50/p95/p99 y consumo de memoria RAM...
%PY_CMD% tests/benchmark.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en la ejecucion del benchmark.
    exit /b %ERRORLEVEL%
)

echo ================================================================================
echo   ✅ VERIFICACIÓN E2E EXITOSA: Todos los pipelines y benchmarks concluidos!
echo ================================================================================
echo.
echo Para lanzar el microservicio FastAPI en modo interactivo:
echo   %PY_CMD% src/api.py
echo   (Swagger UI disponible en http://127.0.0.1:8080/docs)
echo ================================================================================
pause
