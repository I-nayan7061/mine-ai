"""Asynchronous SQLite Persistence Layer for Mine Subsidence AI.
SIH26025 - NexGen | Persistent Telemetry, Alert Log & Fleet Registry

Features:
- Thread-safe, non-blocking asynchronous queries via aiosqlite
- Write-Ahead Logging (WAL) mode enabled for high concurrency
- Automatic table schema creation and index optimization
- Zero external service dependencies (embedded, edge-friendly)
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiosqlite

from src.utils import get_logger

logger = get_logger("mine_ai.db")

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "mine_ai.db"


class DatabaseManager:
    """Manages persistent SQLite operations for telemetry, alerts, and nodes."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._lock = asyncio.Lock()
        self._initialized = False

    async def ensure_initialized(self) -> None:
        """Ensure schema is initialized once before queries."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    await self.init_db()

    async def init_db(self) -> None:
        """Initialize database schema, WAL mode, and necessary indexes."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode and synchronous normal for robust performance
            await db.execute("PRAGMA journal_mode = WAL;")
            await db.execute("PRAGMA synchronous = NORMAL;")
            await db.execute("PRAGMA foreign_keys = ON;")

            # 1. Telemetry records table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    vibration REAL NOT NULL,
                    tilt_x REAL NOT NULL,
                    tilt_y REAL NOT NULL,
                    displacement_mm REAL NOT NULL,
                    anomaly INTEGER NOT NULL,
                    anomaly_score REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    class_probabilities TEXT NOT NULL,
                    subscores TEXT,
                    top_factors TEXT,
                    human_explanations TEXT,
                    neighbour_anomalies INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );
            """)

            # 2. Active and historical alerts table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    reason TEXT NOT NULL,
                    top_factors TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

            # 3. Fleet node registry table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS fleet_nodes (
                    node_id TEXT PRIMARY KEY,
                    gallery TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    latest_risk_score REAL NOT NULL,
                    latest_risk_level TEXT NOT NULL,
                    last_updated TEXT,
                    online INTEGER DEFAULT 1
                );
            """)

            # Performance Indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_node ON telemetry_records(node_id);")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry_records(timestamp);")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(timestamp);")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);")

            await db.commit()
            self._initialized = True
            logger.info("Persistent database initialized at %s with WAL mode enabled.", self.db_path)

    async def save_reading(self, result: Dict[str, Any]) -> int:
        """Persist a single inference diagnosis record."""
        await self.ensure_initialized()
        now_utc = datetime.now(timezone.utc).isoformat()
        sv = result.get("sensor_values", {})
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO telemetry_records (
                    node_id, timestamp, vibration, tilt_x, tilt_y, displacement_mm,
                    anomaly, anomaly_score, risk_score, risk_level, confidence,
                    class_probabilities, subscores, top_factors, human_explanations,
                    neighbour_anomalies, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["node_id"],
                    str(result["timestamp"]),
                    float(sv.get("vibration", 0.0)),
                    float(sv.get("tilt_x", 0.0)),
                    float(sv.get("tilt_y", 0.0)),
                    float(sv.get("displacement_mm", 0.0)),
                    1 if result.get("anomaly", False) else 0,
                    float(result.get("anomaly_score", 0.0)),
                    float(result.get("risk_score", 0.0)),
                    str(result.get("risk_level", "NORMAL")),
                    float(result.get("confidence", 1.0)),
                    json.dumps(result.get("class_probabilities", {})),
                    json.dumps(result.get("subscores", {})),
                    json.dumps(result.get("top_contributing_features", [])),
                    json.dumps(result.get("human_explanations", [])),
                    int(result.get("neighbour_anomalies", 0)),
                    now_utc
                )
            )
            await db.commit()
            return cursor.lastrowid

    async def save_alert(self, alert: Dict[str, Any]) -> None:
        """Persist an early warning alert item."""
        await self.ensure_initialized()
        now_utc = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO alerts (
                    id, node_id, timestamp, severity, risk_score, reason, top_factors, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(alert["id"]),
                    str(alert["node_id"]),
                    str(alert["timestamp"]),
                    str(alert["severity"]),
                    float(alert["risk_score"]),
                    str(alert["reason"]),
                    json.dumps(alert.get("top_factors", [])),
                    now_utc
                )
            )
            await db.commit()

    async def upsert_node(
        self,
        node_id: str,
        gallery: str,
        x: float,
        y: float,
        latest_risk_score: float,
        latest_risk_level: str,
        last_updated: Optional[str] = None,
        online: bool = True
    ) -> None:
        """Update or insert a fleet node state."""
        await self.ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO fleet_nodes (
                    node_id, gallery, x, y, latest_risk_score, latest_risk_level, last_updated, online
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    gallery = excluded.gallery,
                    x = excluded.x,
                    y = excluded.y,
                    latest_risk_score = excluded.latest_risk_score,
                    latest_risk_level = excluded.latest_risk_level,
                    last_updated = excluded.last_updated,
                    online = excluded.online;
                """,
                (
                    node_id, gallery, x, y, latest_risk_score, latest_risk_level,
                    last_updated, 1 if online else 0
                )
            )
            await db.commit()

    async def get_recent_history(self, node_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent chronological telemetry records."""
        await self.ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if node_id:
                query = """
                    SELECT * FROM (
                        SELECT * FROM telemetry_records WHERE node_id = ? ORDER BY id DESC LIMIT ?
                    ) ORDER BY id ASC;
                """
                cursor = await db.execute(query, (node_id, limit))
            else:
                query = """
                    SELECT * FROM (
                        SELECT * FROM telemetry_records ORDER BY id DESC LIMIT ?
                    ) ORDER BY id ASC;
                """
                cursor = await db.execute(query, (limit,))

            rows = await cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "success": True,
                    "timestamp": r["timestamp"],
                    "node_id": r["node_id"],
                    "sensor_values": {
                        "vibration": r["vibration"],
                        "tilt_x": r["tilt_x"],
                        "tilt_y": r["tilt_y"],
                        "displacement_mm": r["displacement_mm"]
                    },
                    "anomaly": bool(r["anomaly"]),
                    "anomaly_score": r["anomaly_score"],
                    "risk_score": r["risk_score"],
                    "risk_level": r["risk_level"],
                    "confidence": r["confidence"],
                    "class_probabilities": json.loads(r["class_probabilities"] or "{}"),
                    "subscores": json.loads(r["subscores"] or "{}"),
                    "top_contributing_features": json.loads(r["top_factors"] or "[]"),
                    "human_explanations": json.loads(r["human_explanations"] or "[]"),
                    "neighbour_anomalies": r["neighbour_anomalies"]
                })
            return results

    async def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent active and historical alerts."""
        await self.ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM alerts ORDER BY rowid DESC LIMIT ?", (limit,))
            rows = await cursor.fetchall()
            alerts = []
            for r in rows:
                alerts.append({
                    "id": r["id"],
                    "node_id": r["node_id"],
                    "timestamp": r["timestamp"],
                    "severity": r["severity"],
                    "risk_score": r["risk_score"],
                    "reason": r["reason"],
                    "top_factors": json.loads(r["top_factors"] or "[]")
                })
            return alerts

    async def get_fleet_nodes(self) -> List[Dict[str, Any]]:
        """Retrieve current registry status of all nodes."""
        await self.ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM fleet_nodes ORDER BY node_id ASC;")
            rows = await cursor.fetchall()
            nodes = []
            for r in rows:
                nodes.append({
                    "node_id": r["node_id"],
                    "gallery": r["gallery"],
                    "x": r["x"],
                    "y": r["y"],
                    "latest_risk_score": r["latest_risk_score"],
                    "latest_risk_level": r["latest_risk_level"],
                    "last_updated": r["last_updated"],
                    "online": bool(r["online"])
                })
            return nodes


# Global singleton instance
db_manager = DatabaseManager()
