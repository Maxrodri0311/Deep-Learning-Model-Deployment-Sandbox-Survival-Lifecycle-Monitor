"""
tests/test_suite.py - Suite Exhaustiva de Pruebas Automatizadas (Pytest)
Verifica integridad de datos sintéticos, invariantes matemáticos de supervivencia,
contratos declarativos Pydantic v2 y endpoints REST del microservicio FastAPI.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Guarda inmutable para Windows Host
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Agregar raíz al sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_contracts import (
    InferenceTelemetryEvent,
    HazardPredictionRequest,
    HazardPredictionResponse,
    SurvivalCurvePoint,
    CoxHazardFactor,
    SurvivalAnalysisReport
)
from src.data_generator import generate_synthetic_dataset
from src.core_engine import SurvivalEngine
from src.api import app


# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture(scope="session")
def session_dataset(tmp_path_factory):
    """Genera un dataset sintético aislado de prueba con 5,000 registros."""
    fn = tmp_path_factory.mktemp("test_data") / "test_telemetry.parquet"
    generate_synthetic_dataset(num_records=5000, output_path=str(fn), seed=123)
    return str(fn)


@pytest.fixture(scope="session")
def survival_engine(session_dataset):
    """Instancia del motor analítico vinculada al dataset de prueba."""
    return SurvivalEngine(data_path=session_dataset)


@pytest.fixture(scope="session")
def api_client():
    """Cliente de pruebas HTTP para FastAPI."""
    return TestClient(app)


# =============================================================================
# 1. PRUEBAS DE GENERACIÓN Y CONTRATOS DE DATOS (PYDANTIC V2)
# =============================================================================
class TestDataIngestionAndContracts:
    """Valida la integridad de la telemetría y los esquemas Pydantic."""

    def test_parquet_schema_and_columns(self, session_dataset):
        df = pd.read_parquet(session_dataset)
        assert len(df) == 5000
        expected_columns = [
            "request_id", "pod_id", "model_architecture", "batch_size",
            "payload_token_count", "concurrency_level", "gpu_memory_used_mb",
            "cpu_utilization_pct", "accumulated_runtime_hours", "latency_ms",
            "sla_threshold_ms", "is_sla_breached", "event_observed", "timestamp"
        ]
        for col in expected_columns:
            assert col in df.columns, f"Columna requerida faltante: {col}"

    def test_pydantic_contract_enforcement(self, session_dataset):
        df = pd.read_parquet(session_dataset)
        sample = df.head(50).to_dict(orient="records")
        for record in sample:
            event = InferenceTelemetryEvent(**record)
            assert event.latency_ms > 0.0
            assert event.batch_size in [1, 2, 4, 8, 16, 32, 64]
            assert event.event_observed in (0, 1)

    def test_rejection_of_invalid_schema(self):
        with pytest.raises(Exception):
            # Batch size negativo debe ser rechazado por Pydantic
            InferenceTelemetryEvent(
                request_id="req-invalid",
                pod_id="pod-test",
                model_architecture="ONNX-Transformer-Int8",
                batch_size=-5,  # Invalido (ge=1)
                payload_token_count=128,
                concurrency_level=10,
                gpu_memory_used_mb=1500.0,
                cpu_utilization_pct=45.0,
                accumulated_runtime_hours=2.0,
                latency_ms=45.0,
                sla_threshold_ms=200.0,
                is_sla_breached=0,
                event_observed=0
            )


# =============================================================================
# 2. PRUEBAS DEL MOTOR ANALÍTICO (DUCKDB, KAPLAN-MEIER & COX)
# =============================================================================
class TestSurvivalEngineMathematics:
    """Valida los invariantes matemáticos de los modelos de supervivencia."""

    def test_duckdb_summary_kpi_integrity(self, survival_engine):
        summary = survival_engine.get_telemetry_summary_by_architecture()
        assert len(summary) >= 2
        assert "p50_latency_ms" in summary.columns
        assert "p95_latency_ms" in summary.columns
        assert "p99_latency_ms" in summary.columns

        # Invariante de percentiles: p50 <= p95 <= p99
        for _, row in summary.iterrows():
            assert row["p50_latency_ms"] <= row["p95_latency_ms"] <= row["p99_latency_ms"]

    def test_kaplan_meier_monotonicity_and_bounds(self, survival_engine):
        curve, _ = survival_engine.fit_kaplan_meier("ONNX-Transformer-Int8", max_bins=12)
        assert len(curve) > 0

        # Punto inicial S(0) = 1.0
        assert curve[0].survival_probability == 1.0

        prev_prob = 1.0
        for p in curve:
            # Invariante 1: Probabilidad acotada [0, 1]
            assert 0.0 <= p.survival_probability <= 1.0
            # Invariante 2: Intervalos de Greenwood acotados
            assert 0.0 <= p.confidence_interval_lower <= p.survival_probability <= p.confidence_interval_upper <= 1.0
            # Invariante 3: Monotonicidad no creciente S(t1) >= S(t2) para t1 <= t2
            assert p.survival_probability <= prev_prob + 1e-6
            prev_prob = p.survival_probability

    def test_cox_proportional_hazards_convergence(self, survival_engine):
        hazards = survival_engine.fit_cox_proportional_hazards("PyTorch-BERT-FP16")
        assert len(hazards) == 4
        for h in hazards:
            assert h.hazard_ratio > 0.0, "El Hazard Ratio debe ser estrictamente positivo"
            assert 0.0 <= h.p_value <= 1.0, "El p-value debe ser una probabilidad válida"
            assert h.confidence_interval[0] <= h.confidence_interval[1]

    def test_real_time_hazard_prediction(self, survival_engine):
        # Caso Estable
        stable_req = HazardPredictionRequest(
            model_architecture="ONNX-Transformer-Int8",
            batch_size=2,
            concurrency_level=5,
            gpu_memory_used_mb=1800.0,
            payload_token_count=128,
            accumulated_runtime_hours=2.0
        )
        res_stable = survival_engine.predict_pod_hazard(stable_req)
        assert res_stable.pod_status == "ESTABLE"
        assert res_stable.recommended_action == "NO_ACTION"
        assert res_stable.predicted_time_to_failure_hours >= 12.0

        # Caso Crítico
        stress_req = HazardPredictionRequest(
            model_architecture="PyTorch-BERT-FP16",
            batch_size=64,
            concurrency_level=120,
            gpu_memory_used_mb=8500.0,
            payload_token_count=4000,
            accumulated_runtime_hours=36.0
        )
        res_stress = survival_engine.predict_pod_hazard(stress_req)
        assert res_stress.pod_status in ("CRÍTICO", "EN_RIESGO")
        assert res_stress.recommended_action in ("RECYCLE_POD", "AUTO_SCALE")


# =============================================================================
# 3. PRUEBAS DEL MICROSERVICIO FASTAPI (ENDPOINTS HTTP)
# =============================================================================
class TestFastAPIMicroservice:
    """Valida los contratos de entrada y salida HTTP mediante TestClient."""

    def test_health_endpoint(self, api_client):
        res = api_client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["service"] == "inetum-dl-survival-sandbox"
        assert data["status"] in ("healthy", "degraded")

    def test_summary_endpoint(self, api_client):
        res = api_client.get("/api/v1/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert len(data["data"]) > 0

    def test_survival_curve_endpoint(self, api_client):
        res = api_client.get("/api/v1/survival-curve/ONNX-Transformer-Int8?bins=10")
        assert res.status_code == 200
        points = res.json()
        assert len(points) > 0
        assert "survival_probability" in points[0]

    def test_hazard_factors_endpoint(self, api_client):
        res = api_client.get("/api/v1/hazard-factors/PyTorch-BERT-FP16")
        assert res.status_code == 200
        factors = res.json()
        assert len(factors) == 4
        assert "hazard_ratio" in factors[0]

    def test_full_report_endpoint(self, api_client):
        res = api_client.get("/api/v1/report/ONNX-Transformer-Int8")
        assert res.status_code == 200
        report = res.json()
        assert report["model_architecture"] == "ONNX-Transformer-Int8"
        assert "survival_curve" in report
        assert "hazard_factors" in report

    def test_predict_hazard_endpoint(self, api_client):
        payload = {
            "model_architecture": "ONNX-Transformer-Int8",
            "batch_size": 16,
            "concurrency_level": 25,
            "gpu_memory_used_mb": 2500.0,
            "payload_token_count": 512,
            "accumulated_runtime_hours": 10.0
        }
        res = api_client.post("/api/v1/predict-hazard", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert "pod_status" in body
        assert "recommended_action" in body
        assert body["recommended_action"] in ("NO_ACTION", "AUTO_SCALE", "RECYCLE_POD")
