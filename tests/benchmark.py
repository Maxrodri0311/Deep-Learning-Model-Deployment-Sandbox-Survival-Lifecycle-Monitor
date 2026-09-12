"""
tests/benchmark.py - Medición Real de Latencia p50, p95 y p99
Sin placeholders: ejecuta 30 iteraciones reales con time.perf_counter().
"""

import time
import numpy as np
from src.core_engine import AnalyticsEngine

def run_benchmarks(iterations=30):
    print(f"[Benchmark] Executing {iterations} iterations of Core Analytics Engine...")
    engine = AnalyticsEngine()
    
    # Warmup
    engine.load_and_transform()
    
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.load_and_transform()
        latencies.append((time.perf_counter() - t0) * 1000)
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print("\n" + "="*50)
    print("  QUANTITATIVE LATENCY BENCHMARK (Real Environment)")
    print("="*50)
    print(f"  p50 Latency: {p50:.2f} ms")
    print(f"  p95 Latency: {p95:.2f} ms")
    print(f"  p99 Latency: {p99:.2f} ms")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_benchmarks()
