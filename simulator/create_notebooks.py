"""Generates valid Jupyter Notebooks (.ipynb) for SIH 2026 NexGen Mine AI.
Creates:
01_data_exploration.ipynb
02_signal_processing.ipynb
03_feature_engineering.ipynb
04_anomaly_detection.ipynb
05_model_training.ipynb
06_model_comparison.ipynb
07_model_evaluation.ipynb
08_live_inference_test.ipynb
"""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.14.6"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


def generate_all_notebooks(target_dir="mine-ai/notebooks"):
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # 01_data_exploration.ipynb
    nb1 = make_notebook([
        md_cell("# 01 - Data Exploration & Geotechnical Telemetry EDA\n\n**SIH 2026 | Problem ID: SIH26025 | Team: NexGen**\n\nExplores multi-node vibration, tilt, and displacement sensor streams."),
        code_cell("import pandas as pd\nimport numpy as np\nfrom pathlib import Path\n\ndf = pd.read_csv('../data/synthetic/synthetic_mine_subsidence_dataset.csv')\nprint(f'Total records: {len(df)}')\nprint(df.head())\nprint(df['risk_label'].value_counts())")
    ])
    with open(dest / "01_data_exploration.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)

    # 02_signal_processing.ipynb
    nb2 = make_notebook([
        md_cell("# 02 - Signal Processing: Filtering & FFT Spectral Analysis\n\nButterworth filtering, median filtering, and Fast Fourier Transform (FFT) analysis."),
        code_cell("import numpy as np\nfrom src.signal_processing import compute_fft_spectrum, compute_spectral_features, apply_rolling_median_filter\n\nsig = np.random.normal(0.05, 0.02, 100)\nfreqs, amps = compute_fft_spectrum(sig, sampling_rate_hz=1.0)\nfeats = compute_spectral_features(sig, sampling_rate_hz=1.0)\nprint('Spectral Features:', feats)")
    ])
    with open(dest / "02_signal_processing.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)

    # 03_feature_engineering.ipynb
    nb3 = make_notebook([
        md_cell("# 03 - Feature Engineering: Vibration, Tilt, Displacement & Spatial Gradients\n\nExtracts geomechanical time-series indicators across rolling windows."),
        code_cell("import numpy as np\nfrom src.vibration_features import extract_vibration_features\nfrom src.tilt_features import extract_tilt_features\nfrom src.displacement_features import extract_displacement_features\n\nvib_f = extract_vibration_features(np.array([0.03, 0.05, 0.04, 0.08]))\ntilt_f = extract_tilt_features(np.array([0.1, 0.2]), np.array([0.1, 0.15]))\ndisp_f = extract_displacement_features(np.array([1.0, 1.2, 1.5]))\nprint('Vibration Features count:', len(vib_f))\nprint('Tilt Features count:', len(tilt_f))\nprint('Displacement Features count:', len(disp_f))")
    ])
    with open(dest / "03_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)

    # 04_anomaly_detection.ipynb
    nb4 = make_notebook([
        md_cell("# 04 - Unsupervised Anomaly Detection with Isolation Forest\n\nDetects anomalous strata disturbances without requiring labelled disaster history."),
        code_cell("from src.anomaly_model import MineAnomalyDetector\nfrom src.utils import load_json\nimport pandas as pd\n\nanom_detector = MineAnomalyDetector.load('../models/isolation_forest.pkl')\nfeats = load_json('../models/feature_names.json')['selected_features']\n\n# Sample inference\ntest_row = pd.DataFrame([{f: 0.05 for f in feats}])\nis_anom, score = anom_detector.predict(test_row)\nprint('Is Anomaly:', is_anom[0], 'Anomaly Score:', round(score[0], 4))")
    ])
    with open(dest / "04_anomaly_detection.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb4, f, indent=2)

    # 05_model_training.ipynb
    nb5 = make_notebook([
        md_cell("# 05 - Supervised Strata Risk Model Training\n\nChronological time-series splitting and training Random Forest & XGBoost classifiers."),
        code_cell("from src.classifier import MineRiskClassifier\nfrom src.utils import load_json\nimport pandas as pd\n\nrf = MineRiskClassifier.load('../models/random_forest.pkl')\nxgb = MineRiskClassifier.load('../models/xgboost_model.pkl')\nprint('Loaded RF and XGBoost models successfully.')")
    ])
    with open(dest / "05_model_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb5, f, indent=2)

    # 06_model_comparison.ipynb
    nb6 = make_notebook([
        md_cell("# 06 - Model Comparison & Benchmark Analysis\n\nCompares Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM."),
        code_cell("from src.utils import load_json\nimport pandas as pd\n\nmeta = load_json('../models/model_metadata.json')\ncomp_df = pd.DataFrame(meta['benchmark_comparison'])\nprint(comp_df[['model_type', 'accuracy', 'weighted_f1', 'high_recall', 'critical_recall', 'inference_time_ms_per_sample']])")
    ])
    with open(dest / "06_model_comparison.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb6, f, indent=2)

    # 07_model_evaluation.ipynb
    nb7 = make_notebook([
        md_cell("# 07 - Model Evaluation & Safety Recall Diagnostics\n\nEvaluates Critical & High risk recall to prevent hazardous false negatives in mine safety."),
        code_cell("from src.utils import load_json\nimport pandas as pd\n\nmeta = load_json('../models/model_metadata.json')\nxgb_eval = next(m for m in meta['benchmark_comparison'] if m['model_type'] == 'xgboost')\nprint('XGBoost Evaluation:')\nprint('Accuracy:', xgb_eval['accuracy'])\nprint('Critical Recall:', xgb_eval['critical_recall'])\nprint('Confusion Matrix:\n', pd.DataFrame(xgb_eval['confusion_matrix'], index=xgb_eval['class_names'], columns=xgb_eval['class_names']))")
    ])
    with open(dest / "07_model_evaluation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb7, f, indent=2)

    # 08_live_inference_test.ipynb
    nb8 = make_notebook([
        md_cell("# 08 - Live Telemetry Inference & SHAP Explainability\n\nSimulates live streaming readings and computes real-time SHAP factor attributions."),
        code_cell("from src.inference import MineInferencePipeline\n\npipe = MineInferencePipeline(models_dir='../models')\nres = pipe.process_single_reading({\n    'node_id': 'N03',\n    'timestamp': '2026-09-10T14:00:00',\n    'tilt_x': 1.8,\n    'tilt_y': 1.2,\n    'vibration': 0.45,\n    'displacement_mm': 6.5\n})\nprint('Inference Response:')\nprint('Risk Level:', res['risk_level'])\nprint('Risk Score:', res['risk_score'])\nprint('Top Contributing Factors:', res['top_contributing_features'])\nprint('Human Explanations:', res['human_explanations'])")
    ])
    with open(dest / "08_live_inference_test.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb8, f, indent=2)

    print(f"Generated 8 notebooks in {target_dir}")


if __name__ == "__main__":
    generate_all_notebooks()
