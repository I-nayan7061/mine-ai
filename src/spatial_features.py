"""Multi-Node Spatial Intelligence and Graph Correlation Module (Leakage-Free).
SIH26025 - NexGen | Underground Mine Subsidence Network Analysis

Models physical spatial relationships between sensor nodes across galleries, crosscuts, and faces:
- Calculates physical spatial deformation gradients (delta_disp / distance, delta_tilt / distance)
- Aggregates neighboring node displacement, tilt magnitude, and vibration RMS
- Computes distance-weighted physical neighbor anomaly indicators based purely on physical sensor thresholds
- NEVER uses ground truth risk labels, target classes, or future predictions.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .config import app_config
from .utils import get_logger, safe_divide

logger = get_logger("mine_ai.spatial")


class MineSpatialGraph:
    """Manages spatial gallery topology and computes leakage-free multi-node neighborhood features."""

    def __init__(self, config=None):
        self.config = config or app_config
        graph_cfg = self.config.get("spatial_graph", {})
        self.nodes = graph_cfg.get("nodes", {
            "N01": {"x": 0.0, "y": 0.0, "gallery": "Gallery-North-1"},
            "N02": {"x": 25.0, "y": 0.0, "gallery": "Gallery-North-1"},
            "N03": {"x": 50.0, "y": 0.0, "gallery": "Gallery-North-1"},
            "N04": {"x": 50.0, "y": 30.0, "gallery": "Gallery-Crosscut-2"},
            "N05": {"x": 25.0, "y": 30.0, "gallery": "Gallery-Crosscut-2"}
        })
        self.connections = graph_cfg.get("connections", [
            ["N01", "N02"], ["N02", "N03"], ["N03", "N04"], ["N04", "N05"], ["N02", "N05"]
        ])
        self.epsilon = self.config.get("features.spatial.spatial_epsilon", 0.1)

        # Purely physical threshold constants (independent of any ML target labels)
        self.physical_disp_thresh = 1.50   # mm
        self.physical_tilt_thresh = 0.80   # deg
        self.physical_vib_thresh = 0.20    # g

    def get_neighbors(self, node_id: str) -> List[str]:
        """Return list of adjacent node IDs connected in gallery graph."""
        neighbors = []
        for u, v in self.connections:
            if u == node_id:
                neighbors.append(v)
            elif v == node_id:
                neighbors.append(u)
        return list(set(neighbors))

    def get_distance(self, node_a: str, node_b: str) -> float:
        """Euclidean distance in meters between two nodes."""
        pos_a = self.nodes.get(node_a, {"x": 0.0, "y": 0.0})
        pos_b = self.nodes.get(node_b, {"x": 0.0, "y": 0.0})
        dx = pos_a["x"] - pos_b["x"]
        dy = pos_a["y"] - pos_b["y"]
        return float(math.hypot(dx, dy))

    def compute_spatial_features(
        self,
        target_node_id: str,
        current_node_states: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute spatial features for target_node given the latest physical state of all network nodes.
        
        Strict Leakage-Free Guarantee:
        Accepts ONLY physical parameters ('displacement_mm', 'tilt_magnitude', 'vibration_rms').
        No target labels, no risk scores, and no ground truth information is ever accessed.
        """
        neighbors = self.get_neighbors(target_node_id)
        target_state = current_node_states.get(target_node_id, {})
        target_disp = float(target_state.get("displacement_mm", 0.0))
        target_tilt = float(target_state.get("tilt_magnitude", 0.0))
        target_vib = float(target_state.get("vibration_rms", 0.0))

        if not neighbors:
            return {
                "spatial_num_abnormal_neighbors": 0.0,
                "spatial_pct_abnormal_neighbors": 0.0,
                "spatial_avg_neighbor_disp": 0.0,
                "spatial_max_neighbor_disp": 0.0,
                "spatial_avg_neighbor_tilt": 0.0,
                "spatial_max_neighbor_tilt": 0.0,
                "spatial_avg_neighbor_vib": 0.0,
                "spatial_disp_gradient_max": 0.0,
                "spatial_tilt_gradient_max": 0.0,
                "spatial_dist_weighted_anomaly": 0.0
            }

        abnormal_count = 0
        neighbor_disps = []
        neighbor_tilts = []
        neighbor_vibs = []
        disp_gradients = []
        tilt_gradients = []
        weighted_anomaly_sum = 0.0
        weight_sum = 0.0

        for n_id in neighbors:
            dist = max(1.0, self.get_distance(target_node_id, n_id))
            weight = 1.0 / (dist + self.epsilon)
            weight_sum += weight

            n_state = current_node_states.get(n_id, {})
            n_disp = float(n_state.get("displacement_mm", 0.0))
            n_tilt = float(n_state.get("tilt_magnitude", 0.0))
            n_vib = float(n_state.get("vibration_rms", 0.0))

            # Physical anomaly determination strictly based on physical sensor thresholds:
            # Genuine strata anomaly requires deformation (displacement or tilt) OR severe vibration with movement
            has_deformation = (n_disp >= self.physical_disp_thresh) or (n_tilt >= self.physical_tilt_thresh)
            is_physically_anomalous = has_deformation or (n_vib >= 1.5 and n_disp >= 1.0)

            if is_physically_anomalous:
                abnormal_count += 1
                weighted_anomaly_sum += weight * 1.0

            neighbor_disps.append(n_disp)
            neighbor_tilts.append(n_tilt)
            neighbor_vibs.append(n_vib)

            # Spatial gradients (physical rate of change across tunnel span in mm/m and deg/m)
            disp_gradients.append(abs(n_disp - target_disp) / dist)
            tilt_gradients.append(abs(n_tilt - target_tilt) / dist)

        pct_abnormal = float(abnormal_count / len(neighbors))
        avg_disp = float(np.mean(neighbor_disps)) if neighbor_disps else 0.0
        max_disp = float(np.max(neighbor_disps)) if neighbor_disps else 0.0
        avg_tilt = float(np.mean(neighbor_tilts)) if neighbor_tilts else 0.0
        max_tilt = float(np.max(neighbor_tilts)) if neighbor_tilts else 0.0
        avg_vib = float(np.mean(neighbor_vibs)) if neighbor_vibs else 0.0
        max_disp_grad = float(np.max(disp_gradients)) if disp_gradients else 0.0
        max_tilt_grad = float(np.max(tilt_gradients)) if tilt_gradients else 0.0
        dist_weighted_anomaly = float(safe_divide(weighted_anomaly_sum, weight_sum, default=0.0))

        return {
            "spatial_num_abnormal_neighbors": float(abnormal_count),
            "spatial_pct_abnormal_neighbors": round(pct_abnormal, 4),
            "spatial_avg_neighbor_disp": round(avg_disp, 4),
            "spatial_max_neighbor_disp": round(max_disp, 4),
            "spatial_avg_neighbor_tilt": round(avg_tilt, 4),
            "spatial_max_neighbor_tilt": round(max_tilt, 4),
            "spatial_avg_neighbor_vib": round(avg_vib, 4),
            "spatial_disp_gradient_max": round(max_disp_grad, 5),
            "spatial_tilt_gradient_max": round(max_tilt_grad, 5),
            "spatial_dist_weighted_anomaly": round(dist_weighted_anomaly, 4)
        }
