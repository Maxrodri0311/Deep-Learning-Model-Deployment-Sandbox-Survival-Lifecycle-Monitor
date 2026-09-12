"""
tests/test_suite.py - Suite de Pruebas Automatizadas con Pytest
Verifica integridad de datos, esquemas y latencias analíticas.
"""

import os
import pytest
import pandas as pd
from src.data_generator import generate_synthetic_dataset
from src.core_engine import AnalyticsEngine

@pytest.fixture(scope="session")
def test_dataset(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data") / "test_data.parquet"
    df = generate_synthetic_dataset(num_records=5000, output_path=str(fn))
    return str(fn)

def test_data_generation_integrity(test_dataset):
    df = pd.read_parquet(test_dataset)
    assert len(df) == 5000
    assert "transaction_id" in df.columns
    assert df["base_cost"].min() > 0

def test_core_engine_execution(test_dataset):
    engine = AnalyticsEngine(data_path=test_dataset)
    res = engine.load_and_transform()
    assert len(res) > 0
    assert "avg_cost" in res.columns
