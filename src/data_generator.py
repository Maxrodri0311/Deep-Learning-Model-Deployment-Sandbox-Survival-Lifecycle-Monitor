"""
src/data_generator.py - Generador Sintético Estocástico de Telemetría (50,000+ Registros)
Simula la física real de pods de inferencia de Deep Learning en Inetum para Survival & Hazard Analytics.
"""

import os
import sys
import time
import uuid
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Guarda inmutable para Windows Host (UTF-8)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Asegurar import de data_contracts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_contracts import InferenceTelemetryEvent


def generate_synthetic_dataset(
    num_records: int = 50000,
    output_path: str = "data/raw_dataset.parquet",
    seed: int = 42
) -> pd.DataFrame:
    """
    Genera un dataset sintético de telemetría de inferencia con física de degradación temporal,
    fugas de memoria en tensores, contención de concurrencia y censura a la derecha.
    """
    print(f"[Data Generator] 🚀 Iniciando generación de {num_records:,} eventos de inferencia para Inetum...")
    start_time = time.time()
    np.random.seed(seed)

    # 1. Definición de pods y arquitecturas de Deep Learning en producción
    architectures = [
        "ONNX-Transformer-Int8",
        "PyTorch-BERT-FP16",
        "TensorRT-LLM-Q4",
        "TorchScript-ResNet50"
    ]
    arch_weights = [0.35, 0.30, 0.20, 0.15]

    num_pods = max(500, num_records // 100)
    pod_ids = [f"pod-inetum-{uuid.uuid4().hex[:6]}" for _ in range(num_pods)]
    pod_arch_map = {pid: np.random.choice(architectures, p=arch_weights) for pid in pod_ids}
    
    # Asignación de pod a cada evento
    chosen_pods = np.random.choice(pod_ids, size=num_records)
    chosen_archs = [pod_arch_map[p] for p in chosen_pods]

    # 2. Generación de variables físicas de carga operacional
    batch_sizes = np.random.choice([1, 2, 4, 8, 16, 32, 64], size=num_records, p=[0.20, 0.25, 0.25, 0.15, 0.08, 0.05, 0.02])
    payload_tokens = np.random.lognormal(mean=5.8, sigma=0.85, size=num_records).astype(int)
    payload_tokens = np.clip(payload_tokens, 16, 4096)
    concurrency_levels = np.random.negative_binomial(n=5, p=0.15, size=num_records) + 1
    concurrency_levels = np.clip(concurrency_levels, 1, 150)

    # Horas acumuladas de ejecución continua del pod (0.1 a 48 horas)
    runtime_hours = np.random.exponential(scale=14.0, size=num_records)
    runtime_hours = np.round(np.clip(runtime_hours, 0.1, 48.0), 2)

    # 3. Física estocástica específica por arquitectura
    latencies = np.zeros(num_records, dtype=float)
    gpu_memories = np.zeros(num_records, dtype=float)
    cpu_utilizations = np.zeros(num_records, dtype=float)

    for i in range(num_records):
        arch = chosen_archs[i]
        b = batch_sizes[i]
        tokens = payload_tokens[i]
        c = concurrency_levels[i]
        t = runtime_hours[i]

        if arch == "ONNX-Transformer-Int8":
            base_lat = 38.0
            base_vram = 1800.0
            leak_rate = 1.2  # MB por hora
            concurrency_impact = 0.35
            token_impact = 0.012
            cpu_base = 35.0
        elif arch == "PyTorch-BERT-FP16":
            base_lat = 82.0
            base_vram = 4200.0
            leak_rate = 14.5  # Fuga moderada de tensores en memoria CUDA
            concurrency_impact = 0.72
            token_impact = 0.025
            cpu_base = 52.0
        elif arch == "TensorRT-LLM-Q4":
            base_lat = 125.0
            base_vram = 7800.0
            leak_rate = 8.0
            concurrency_impact = 1.15
            token_impact = 0.048
            cpu_base = 65.0
        else:  # TorchScript-ResNet50
            base_lat = 26.0
            base_vram = 1200.0
            leak_rate = 0.8
            concurrency_impact = 0.22
            token_impact = 0.002
            cpu_base = 28.0

        # Modelado de VRAM acumulada con fuga y saturación
        vram = base_vram + (leak_rate * t) + np.random.normal(0, 35.0)
        gpu_memories[i] = round(max(500.0, vram), 2)

        # Utilización de CPU
        cpu = cpu_base + (c * 0.25) + np.random.normal(0, 4.0)
        cpu_utilizations[i] = round(np.clip(cpu, 5.0, 99.0), 2)

        # Penalización no lineal por degradación térmica y de memoria
        memory_stress_factor = max(0.0, (gpu_memories[i] - (base_vram * 1.3)) / 250.0)
        drift_latency_penalty = (t ** 1.15) * 0.85

        # Latencia total calculada
        calc_lat = (
            base_lat
            + (b * 1.6)
            + (c * concurrency_impact)
            + (tokens * token_impact)
            + memory_stress_factor * 8.5
            + drift_latency_penalty
            + np.random.exponential(scale=12.0)
        )
        latencies[i] = round(max(5.0, calc_lat), 2)

    # 4. Asignación de Estado de SLA y Censura para Survival Analysis
    sla_threshold = 200.0  # ms
    is_sla_breached = (latencies > sla_threshold).astype(int)
    
    # Evento observado: 1 si violó el SLA durante su ciclo de vida; 0 si se censuró a la derecha
    # Un pod que acumula horas pero mantiene SLA < 200ms es una observación censurada viva
    event_observed = is_sla_breached.copy()

    # 5. Fechas e IDs deterministas
    base_date = datetime(2026, 1, 15, 8, 0, 0)
    timestamps = [
        (base_date + timedelta(minutes=int(i % 1440), seconds=int((i * 7) % 60))).isoformat()
        for i in range(num_records)
    ]
    request_ids = [f"req-{uuid.UUID(int=i + 100000000000).hex[:12]}" for i in range(num_records)]

    # 6. Construcción del DataFrame
    df = pd.DataFrame({
        "request_id": request_ids,
        "pod_id": chosen_pods,
        "model_architecture": chosen_archs,
        "batch_size": batch_sizes,
        "payload_token_count": payload_tokens,
        "concurrency_level": concurrency_levels,
        "gpu_memory_used_mb": gpu_memories,
        "cpu_utilization_pct": cpu_utilizations,
        "accumulated_runtime_hours": runtime_hours,
        "latency_ms": latencies,
        "sla_threshold_ms": sla_threshold,
        "is_sla_breached": is_sla_breached,
        "event_observed": event_observed,
        "timestamp": timestamps
    })

    # 7. Validación de muestra contra Contratos Pydantic
    print("[Data Generator] 🔍 Validando contrato de datos Pydantic v2 en muestra...")
    sample_records = df.head(100).to_dict(orient="records")
    for rec in sample_records:
        InferenceTelemetryEvent(**rec)
    print("  ✓ Validación Pydantic v2 exitosa en 100/100 eventos de prueba.")

    # 8. Guardar en Parquet comprimido con Snappy
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    
    elapsed = time.time() - start_time
    breach_pct = (df['is_sla_breached'].sum() / len(df)) * 100
    censored_pct = ((len(df) - df['event_observed'].sum()) / len(df)) * 100

    print(f"[Data Generator] ✅ Generados {len(df):,} registros en {elapsed:.2f}s -> {output_path}")
    print(f"  • Violaciones de SLA (>200ms): {df['is_sla_breached'].sum():,} ({breach_pct:.2f}%)")
    print(f"  • Eventos Censurados (Sanos): {len(df) - df['event_observed'].sum():,} ({censored_pct:.2f}%)")
    print(f"  • Latencia Media: {df['latency_ms'].mean():.2f} ms | p95: {df['latency_ms'].quantile(0.95):.2f} ms")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de telemetría de inferencia para Inetum")
    parser.add_argument("--records", type=int, default=50000, help="Cantidad de eventos a generar")
    parser.add_argument("--output", type=str, default="data/raw_dataset.parquet", help="Ruta del archivo Parquet")
    parser.add_argument("--seed", type=int, default=42, help="Semilla pseudoaleatoria")
    args = parser.parse_args()

    generate_synthetic_dataset(
        num_records=args.records,
        output_path=args.output,
        seed=args.seed
    )
