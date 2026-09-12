"""
src/data_contracts.py - Contratos Declarativos y Esquemas de Telemetría (Pydantic v2)
Gobernanza estricta para el sandbox de despliegue de modelos de Deep Learning en Inetum.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone


class InferenceTelemetryEvent(BaseModel):
    """
    Contrato estricto para eventos de telemetría de inferencia en tiempo real.
    Captura la física del contenedor, variables de carga y estado de SLA.
    """
    request_id: str = Field(..., description="Identificador único del request UUID")
    pod_id: str = Field(..., description="ID del worker/pod de inferencia en Kubernetes")
    model_architecture: str = Field(..., description="Arquitectura del modelo (ej: ONNX-Transformer-Int8, PyTorch-BERT-FP16)")
    batch_size: int = Field(..., ge=1, le=128, description="Tamaño del batch procesado en el request")
    payload_token_count: int = Field(..., ge=1, le=8192, description="Número de tokens o longitud del payload de entrada")
    concurrency_level: int = Field(..., ge=1, le=500, description="Nivel de concurrencia concurrente en el pod")
    gpu_memory_used_mb: float = Field(..., ge=0.0, description="Uso de memoria VRAM de la GPU en MB")
    cpu_utilization_pct: float = Field(..., ge=0.0, le=100.0, description="Porcentaje de utilización de CPU")
    accumulated_runtime_hours: float = Field(..., ge=0.0, description="Horas continuas de ejecución del pod en producción")
    latency_ms: float = Field(..., ge=0.0, description="Latencia end-to-end de inferencia medida en milisegundos")
    sla_threshold_ms: float = Field(default=200.0, description="Umbral de SLA acordado con Inetum (200ms)")
    is_sla_breached: int = Field(..., ge=0, le=1, description="1 si latency_ms > sla_threshold_ms, 0 en caso contrario")
    event_observed: int = Field(..., ge=0, le=1, description="1 si ocurrió degradación crítica/falla de SLA, 0 si la observación fue censurada (pod saludable)")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Timestamp ISO-8601 del evento")

    @field_validator("model_architecture")
    @classmethod
    def validate_architecture(cls, v: str) -> str:
        valid_architectures = {
            "ONNX-Transformer-Int8",
            "PyTorch-BERT-FP16",
            "TensorRT-LLM-Q4",
            "TorchScript-ResNet50"
        }
        if v not in valid_architectures:
            # Permitir arquitecturas personalizadas siempre que no sean vacías
            if not v or len(v.strip()) == 0:
                raise ValueError("model_architecture cannot be empty")
        return v


class SurvivalCurvePoint(BaseModel):
    """Punto discreto en la curva de supervivencia temporal S(t)."""
    timeline_hours: float = Field(..., description="Tiempo acumulado t en horas")
    survival_probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de supervivencia S(t) = P(T > t)")
    confidence_interval_lower: float = Field(..., ge=0.0, le=1.0, description="Límite inferior del intervalo de confianza (95%)")
    confidence_interval_upper: float = Field(..., ge=0.0, le=1.0, description="Límite superior del intervalo de confianza (95%)")
    at_risk_count: int = Field(..., ge=0, description="Número de contenedores en riesgo en el tiempo t")


class CoxHazardFactor(BaseModel):
    """Factor de riesgo proporcional derivado del modelo de Cox."""
    covariate: str = Field(..., description="Nombre de la covariable analizada")
    coefficient: float = Field(..., description="Coeficiente beta estimado")
    hazard_ratio: float = Field(..., description="Hazard Ratio exp(beta)")
    p_value: float = Field(..., ge=0.0, le=1.0, description="Significancia estadística del factor")
    confidence_interval: List[float] = Field(..., description="Intervalo de confianza 95% para el Hazard Ratio")


class SurvivalAnalysisReport(BaseModel):
    """Reporte formal de análisis de supervivencia para Inetum."""
    model_architecture: str
    total_pods_analyzed: int
    failures_observed: int
    censored_events: int
    median_survival_time_hours: Optional[float]
    survival_curve: List[SurvivalCurvePoint]
    hazard_factors: List[CoxHazardFactor]
    p95_latency_ms: float
    sla_breach_rate_pct: float


class HazardPredictionRequest(BaseModel):
    """Contrato para predecir el riesgo inminente de degradación de un contenedor."""
    model_architecture: str = Field(default="ONNX-Transformer-Int8")
    batch_size: int = Field(default=16, ge=1, le=128)
    concurrency_level: int = Field(default=25, ge=1, le=500)
    gpu_memory_used_mb: float = Field(default=4500.0, ge=0.0)
    payload_token_count: int = Field(default=512, ge=1, le=8192)
    accumulated_runtime_hours: float = Field(default=8.5, ge=0.0)


class HazardPredictionResponse(BaseModel):
    """Respuesta con el scoring de riesgo temporal del contenedor."""
    pod_status: str = Field(..., description="ESTABLE, EN_RIESGO o CRÍTICO")
    immediate_hazard_multiplier: float = Field(..., description="Multiplicador de riesgo relativo exp(beta*X)")
    predicted_time_to_failure_hours: Optional[float] = Field(None, description="Horas estimadas hasta la violación del SLA")
    recommended_action: str = Field(..., description="Acción de orquestación recomendada (ej: NO_ACTION, RECYCLE_POD, AUTO_SCALE)")
