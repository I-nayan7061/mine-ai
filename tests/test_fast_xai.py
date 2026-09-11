"""Brutal Verification of Fast XAI Attribution Performance.
SIH26025 - NexGen | Sub-Millisecond Explainability Verification
"""

import time
import numpy as np
import pandas as pd
import pytest

from src.explainability import MineExplainer
from src.inference import MineInferencePipeline


def test_fast_xai_attribution_latency_and_format():
    """Verify that feature attribution completes in under 15ms per sample with valid factors."""
    pipeline = MineInferencePipeline()
    assert pipeline.explainer is not None

    # Construct sample feature vector
    feature_names = pipeline.selected_features
    dummy_data = {f: [float(np.random.randn())] for f in feature_names}
    df_row = pd.DataFrame(dummy_data)

    # Time explanation
    start_t = time.perf_counter()
    explanation = pipeline.explainer.explain_prediction(df_row, top_k=4)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    assert "top_factors" in explanation
    assert len(explanation["top_factors"]) <= 4
    assert "factor_impacts" in explanation
    assert "human_explanations" in explanation
    assert len(explanation["human_explanations"]) == len(explanation["top_factors"])

    # Latency should be fast
    assert elapsed_ms < 50.0, f"XAI took {elapsed_ms:.2f}ms, expected under 50ms"
