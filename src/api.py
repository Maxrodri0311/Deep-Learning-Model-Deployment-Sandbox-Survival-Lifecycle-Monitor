"""
src/api.py - Microservicio FastAPI de Observabilidad & Sandbox de Inferencia
Expone endpoints REST documentados en OpenAPI/Swagger para el monitoreo de ciclo de vida
y scoring de riesgo temporal de contenedores de Deep Learning en Inetum.
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Path, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Guarda inmutable para Windows Host (UTF-8)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Importar contratos de datos y motor analítico
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_contracts import (
    SurvivalCurvePoint,
    CoxHazardFactor,
    SurvivalAnalysisReport,
    HazardPredictionRequest,
    HazardPredictionResponse
)
from core_engine import SurvivalEngine

# Metadata OpenAPI Empresarial
app = FastAPI(
    title="Inetum Deep Learning Deployment Sandbox & Survival Analytics API",
    description=(
        "Microservicio de observabilidad y predicción temporal de violaciones de SLA en pods "
        "de Deep Learning. Implementa estimadores Kaplan-Meier y regresión de riesgos proporcionales "
        "de Cox para anticipar fallas de inferencia 4 horas antes de la degradación operativa."
    ),
    version="1.0.0",
    contact={
        "name": "Maximiliano Rodriguez",
        "email": "maxrodri0311@gmail.com",
        "url": "https://www.linkedin.com/in/maximiliano-rodriguez-982674375/"
    },
    license_info={
        "name": "Enterprise Private License — Inetum Case Study"
    }
)

# Habilitar CORS para integración con dashboards web y consolas de telemetría
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia Singleton del Motor Analítico
_engine_instance: Optional[SurvivalEngine] = None
START_TIME = time.time()


def get_engine() -> SurvivalEngine:
    global _engine_instance
    if _engine_instance is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw_dataset.parquet")
        _engine_instance = SurvivalEngine(data_path=data_path)
    return _engine_instance


# =============================================================================
# 1. HEALTH CHECKS & TELEMETRÍA DEL SISTEMA
# =============================================================================
@app.get("/health", tags=["Health & Observability"], summary="Liveness & Readiness Probe")
async def health_check():
    """Verifica la salud del microservicio, disponibilidad del motor DuckDB y dataset Parquet."""
    engine = get_engine()
    dataset_exists = os.path.exists(engine.data_path)
    uptime_seconds = round(time.time() - START_TIME, 2)
    
    return {
        "status": "healthy" if dataset_exists else "degraded",
        "service": "inetum-dl-survival-sandbox",
        "version": "1.0.0",
        "dataset_ready": dataset_exists,
        "data_path": engine.data_path,
        "uptime_seconds": uptime_seconds
    }


# =============================================================================
# 2. TELEMETRÍA AGREGADA EN DUCKDB POR ARQUITECTURA
# =============================================================================
@app.get("/api/v1/summary", tags=["Analytics & KPIs"], summary="Resumen de telemetría por arquitectura")
async def get_summary():
    """Devuelve métricas agregadas vectorizadas (latencia p50/p95/p99, tasa de fallos de SLA y consumo de VRAM)."""
    try:
        engine = get_engine()
        df = engine.get_telemetry_summary_by_architecture()
        return {
            "status": "success",
            "total_architectures": len(df),
            "data": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al consultar DuckDB: {str(e)}")


# =============================================================================
# 3. CURVAS DE SUPERVIVENCIA DE KAPLAN-MEIER
# =============================================================================
@app.get(
    "/api/v1/survival-curve/{architecture}",
    response_model=List[SurvivalCurvePoint],
    tags=["Survival Analysis"],
    summary="Curva de supervivencia temporal Kaplan-Meier"
)
async def get_survival_curve(
    architecture: str = Path(..., description="Nombre de la arquitectura (ej: ONNX-Transformer-Int8, PyTorch-BERT-FP16)"),
    bins: int = Query(default=24, ge=4, le=100, description="Número de puntos discretos en la curva")
):
    """
    Calcula los puntos de la curva de supervivencia S(t) = P(T > t) con intervalos de confianza
    al 95% calculados con la fórmula de Greenwood.
    """
    try:
        engine = get_engine()
        curve, _ = engine.fit_kaplan_meier(architecture=architecture, max_bins=bins)
        if not curve:
            raise HTTPException(status_code=404, detail=f"No se encontraron eventos para la arquitectura '{architecture}'")
        return curve
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en estimador Kaplan-Meier: {str(e)}")


# =============================================================================
# 4. FACTORES DE RIESGO DE COX (HAZARD RATIOS)
# =============================================================================
@app.get(
    "/api/v1/hazard-factors/{architecture}",
    response_model=List[CoxHazardFactor],
    tags=["Survival Analysis"],
    summary="Factores de riesgo y Hazard Ratios del modelo de Cox"
)
async def get_hazard_factors(
    architecture: str = Path(..., description="Nombre de la arquitectura evaluada")
):
    """
    Retorna los coeficientes beta estimados, Hazard Ratios exp(beta), errores estándar y p-values
    para cada variable física de carga operacional (batch size, concurrencia, VRAM, tokens).
    """
    try:
        engine = get_engine()
        factors = engine.fit_cox_proportional_hazards(architecture=architecture)
        if not factors:
            raise HTTPException(status_code=404, detail=f"Datos insuficientes para ajustar Cox en '{architecture}'")
        return factors
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en regresión de Cox: {str(e)}")


# =============================================================================
# 5. DOSSIER / REPORTE EJECUTIVO COMPLETO
# =============================================================================
@app.get(
    "/api/v1/report/{architecture}",
    response_model=SurvivalAnalysisReport,
    tags=["Reporting & Governance"],
    summary="Dossier completo de ciclo de vida y supervivencia"
)
async def get_full_report(
    architecture: str = Path(..., description="Arquitectura del modelo evaluado")
):
    """Genera la ficha analítica exhaustiva combinando DuckDB, Kaplan-Meier y Cox."""
    try:
        engine = get_engine()
        report = engine.generate_full_report(architecture=architecture)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando reporte: {str(e)}")


# =============================================================================
# 6. INFERENCIA EN TIEMPO REAL: SCORING DE RIESGO Y ACCIÓN PRESCRIPTIVA
# =============================================================================
@app.post(
    "/api/v1/predict-hazard",
    response_model=HazardPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Real-Time Sandbox & Inference"],
    summary="Evaluar riesgo inminente de degradación de un contenedor de inferencia"
)
async def predict_hazard(payload: HazardPredictionRequest):
    """
    Recibe la telemetría en vivo de un contenedor de Deep Learning, evalúa su riesgo instantáneo
    con el modelo de Cox ajustado y prescribe si se debe reciclar el pod, auto-escalar o mantenerlo activo.
    """
    try:
        engine = get_engine()
        prediction = engine.predict_pod_hazard(payload)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en scoring de riesgo: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("[API Runner] Iniciando servidor FastAPI en http://127.0.0.1:8080/docs ...")
    uvicorn.run("src.api:app", host="127.0.0.1", port=8080, reload=False)
