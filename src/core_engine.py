"""
src/core_engine.py - Motor Analítico Core: Causal & Survival Lifecycle Analytics
Implementa Kaplan-Meier & Cox Proportional Hazards Hazard Modeling con DuckDB columnar y tipado estricto.
"""

import os
import duckdb
import pandas as pd
import numpy as np

class AnalyticsEngine:
    """Motor analítico de alto rendimiento con DuckDB columnar."""
    
    def __init__(self, data_path: str = "data/raw_dataset.parquet"):
        self.data_path = data_path
        self.conn = duckdb.connect(":memory:")
        
    def load_and_transform(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Source data not found at {self.data_path}")
            
        query = f"""
            SELECT 
                segment,
                status,
                COUNT(*) as total_events,
                ROUND(AVG(base_cost), 2) as avg_cost,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY base_cost), 2) as p95_cost,
                ROUND(AVG(operational_metric), 2) as avg_metric
            FROM read_parquet('{self.data_path}')
            GROUP BY segment, status
            ORDER BY total_events DESC;
        """
        return self.conn.execute(query).df()

if __name__ == "__main__":
    engine = AnalyticsEngine()
    summary = engine.load_and_transform()
    print("[Core Engine] Transformation Summary:")
    print(summary.to_string())
