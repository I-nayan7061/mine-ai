import numpy as np
from datetime import datetime, timedelta
from src.inference import MineInferencePipeline

pipeline = MineInferencePipeline()
pipeline.buffer_manager.clear()
now = datetime(2026, 9, 10, 12, 0, 0)
rng = np.random.default_rng(123)

for i in range(25):
    res = pipeline.process_single_reading({
        "node_id": "N03",
        "timestamp": (now + timedelta(seconds=i)).isoformat(),
        "tilt_x": 0.10 + rng.normal(0, 0.05),
        "tilt_y": 0.08 + rng.normal(0, 0.05),
        "vibration": abs(0.04 + rng.normal(0, 0.03)),
        "displacement_mm": 1.00 + rng.normal(0, 0.03)
    })
    if i >= 10:
        print(f"i={i}: score={res['risk_score']}, level={res['risk_level']}, probs={res.get('class_probabilities')}, sub={res.get('subscores')}")
