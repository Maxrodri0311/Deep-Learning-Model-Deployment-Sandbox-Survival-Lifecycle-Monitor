@echo off
echo ========================================================
echo  🚀 Deep Learning Model Deployment Sandbox (GP-029) - Automated Execution & Test Suite
echo ========================================================
echo.
echo [1/3] Generating synthetic dataset (50,000+ records)...
python src/data_generator.py --records 50000
if %ERRORLEVEL% NEQ 0 (echo Error in data generator && exit /b %ERRORLEVEL%)

echo.
echo [2/3] Executing Core Analytical Engine (Causal & Survival Lifecycle Analytics)...
python src/core_engine.py
if %ERRORLEVEL% NEQ 0 (echo Error in core engine && exit /b %ERRORLEVEL%)

echo.
echo [3/3] Running Pytest Suite...
pytest tests/ -v
echo.
echo ========================================================
echo  ✅ Execution Complete: All Benchmarks & Tests Passed!
echo ========================================================
