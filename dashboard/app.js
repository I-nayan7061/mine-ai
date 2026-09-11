/**
 * Mine Subsidence AI Dashboard Client Logic
 * SIH26025 - NexGen | Real-Time Telemetry & Scenario Demonstrator
 */

let selectedNodeId = "N03";
let simulationInterval = null;
let simStep = 0;
let activeScenario = null;

// Chart references
let chartDisp = null;
let chartTilt = null;
let chartVib = null;
let chartRisk = null;

// Color maps
const RISK_COLORS = {
  NORMAL: "#10b981",
  WARNING: "#f59e0b",
  HIGH: "#f97316",
  CRITICAL: "#ef4444"
};

// Initialize Dashboard on Page Load
document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initCharts();
  refreshFleetData();
  // Poll fleet telemetry every 1.5 seconds
  setInterval(refreshFleetData, 1500);
});

function initClock() {
  setInterval(() => {
    const now = new Date();
    document.getElementById("live-clock").textContent = now.toLocaleTimeString();
  }, 1000);
}

function initCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    scales: {
      x: {
        display: false,
        grid: { color: "#1e293b" }
      },
      y: {
        grid: { color: "#1e293b" },
        ticks: { color: "#94a3b8", font: { size: 10 } }
      }
    },
    plugins: {
      legend: {
        labels: { color: "#cbd5e1", font: { size: 11, weight: "bold" }, boxWidth: 12 }
      },
      tooltip: { mode: "index", intersect: false }
    }
  };

  // 1. Displacement Chart
  const ctxDisp = document.getElementById("chart-displacement").getContext("2d");
  chartDisp = new Chart(ctxDisp, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Displacement / Convergence (mm)",
        data: [],
        borderColor: "#06b6d4",
        backgroundColor: "rgba(6, 182, 212, 0.1)",
        borderWidth: 2,
        tension: 0.2,
        fill: true
      }]
    },
    options: commonOptions
  });

  // 2. Tilt Chart
  const ctxTilt = document.getElementById("chart-tilt").getContext("2d");
  chartTilt = new Chart(ctxTilt, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Tilt X (°)",
          data: [],
          borderColor: "#3b82f6",
          borderWidth: 1.5,
          tension: 0.2
        },
        {
          label: "Tilt Y (°)",
          data: [],
          borderColor: "#a855f7",
          borderWidth: 1.5,
          tension: 0.2
        }
      ]
    },
    options: commonOptions
  });

  // 3. Vibration Chart
  const ctxVib = document.getElementById("chart-vibration").getContext("2d");
  chartVib = new Chart(ctxVib, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Vibration Magnitude (g)",
        data: [],
        borderColor: "#f59e0b",
        backgroundColor: "rgba(245, 158, 11, 0.1)",
        borderWidth: 1.5,
        tension: 0.1,
        fill: true
      }]
    },
    options: commonOptions
  });

  // 4. Continuous Risk Score Chart
  const ctxRisk = document.getElementById("chart-risk").getContext("2d");
  chartRisk = new Chart(ctxRisk, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Strata Risk Score (0-100)",
        data: [],
        borderColor: "#ef4444",
        backgroundColor: "rgba(239, 68, 68, 0.15)",
        borderWidth: 2.5,
        tension: 0.3,
        fill: true
      }]
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: {
          min: 0,
          max: 100,
          grid: { color: "#1e293b" },
          ticks: { color: "#94a3b8", stepSize: 25 }
        }
      }
    }
  });
}

function selectNode(nodeId) {
  selectedNodeId = nodeId;
  document.getElementById("selected-node-title").textContent = `Node ${nodeId}`;
  refreshFleetData();
}

async function refreshFleetData() {
  try {
    // 1. Fetch Fleet Nodes
    const resNodes = await fetch("/api/nodes");
    if (resNodes.ok) {
      const nodes = await resNodes.json();
      updateMapNodes(nodes);
    }

    // 2. Fetch Fleet Risk Summary
    const resRisk = await fetch("/api/risk");
    if (resRisk.ok) {
      const riskSummary = await resRisk.json();
      updateRiskKPIs(riskSummary);
    }

    // 3. Fetch History for Selected Node
    const resHist = await fetch(`/api/history?node_id=${selectedNodeId}&limit=30`);
    if (resHist.ok) {
      const histData = await resHist.json();
      updateChartsAndMetrics(histData);
    }

    // 4. Fetch Active Alerts
    const resAlerts = await fetch("/api/alerts?limit=10");
    if (resAlerts.ok) {
      const alerts = await resAlerts.json();
      updateAlertsTable(alerts);
    }
  } catch (err) {
    console.error("Failed to refresh telemetry:", err);
  }
}

function updateMapNodes(nodes) {
  nodes.forEach(n => {
    const marker = document.querySelector(`#node-marker-${n.node_id} circle`);
    if (marker) {
      marker.className.baseVal = `node-circle ${n.latest_risk_level.toLowerCase()} ${n.node_id === "N03" ? "pulse" : ""}`;
    }
  });
}

function updateRiskKPIs(summary) {
  document.getElementById("kpi-max-risk").textContent = summary.max_risk_score.toFixed(1);
  document.getElementById("kpi-active-nodes").textContent = summary.total_nodes;

  const pill = document.getElementById("fleet-status-pill");
  const pillText = document.getElementById("fleet-status-text");
  const dot = pill.querySelector(".status-dot");

  let statusClass = "green";
  let statusLabel = "SYSTEM NORMAL";

  if (summary.critical_nodes_count > 0) {
    statusClass = "red";
    statusLabel = `CRITICAL ALERT (${summary.critical_nodes_count} Nodes)`;
  } else if (summary.high_nodes_count > 0) {
    statusClass = "orange";
    statusLabel = `HIGH RISK DETECTED (${summary.high_nodes_count} Nodes)`;
  } else if (summary.warning_nodes_count > 0) {
    statusClass = "yellow";
    statusLabel = `WARNING LEVEL (${summary.warning_nodes_count} Nodes)`;
  }

  dot.className = `status-dot ${statusClass}`;
  pillText.textContent = statusLabel;
}

function updateChartsAndMetrics(history) {
  if (!history || history.length === 0) return;

  const latest = history[history.length - 1];

  // Update Mini Metrics Ribbon
  document.getElementById("val-vib").textContent = `${latest.sensor_values.vibration.toFixed(3)} g`;
  document.getElementById("val-tilt").textContent = `${latest.sensor_values.tilt_x.toFixed(2)}° / ${latest.sensor_values.tilt_y.toFixed(2)}°`;
  document.getElementById("val-disp").textContent = `${latest.sensor_values.displacement_mm.toFixed(2)} mm`;
  document.getElementById("val-neighbors").textContent = `${latest.neighbour_anomalies} Abnormal`;

  const nodeBadge = document.getElementById("selected-node-badge");
  nodeBadge.textContent = `STATUS: ${latest.risk_level} (Score: ${latest.risk_score.toFixed(1)})`;
  nodeBadge.style.background = `rgba(${hexToRgb(RISK_COLORS[latest.risk_level])}, 0.2)`;
  nodeBadge.style.borderColor = RISK_COLORS[latest.risk_level];
  nodeBadge.style.color = RISK_COLORS[latest.risk_level];

  document.getElementById("kpi-anomaly-status").textContent = latest.anomaly ? "ANOMALY DETECTED" : "NORMAL";
  document.getElementById("val-confidence").textContent = `${(latest.confidence * 100).toFixed(1)}%`;

  // Update Charts Data
  const labels = history.map(h => h.timestamp.split("T")[1].slice(0, 8));
  const dispVals = history.map(h => h.sensor_values.displacement_mm);
  const tiltXVals = history.map(h => h.sensor_values.tilt_x);
  const tiltYVals = history.map(h => h.sensor_values.tilt_y);
  const vibVals = history.map(h => h.sensor_values.vibration);
  const riskVals = history.map(h => h.risk_score);

  // Update Displacement
  chartDisp.data.labels = labels;
  chartDisp.data.datasets[0].data = dispVals;
  chartDisp.update("none");

  // Update Tilt
  chartTilt.data.labels = labels;
  chartTilt.data.datasets[0].data = tiltXVals;
  chartTilt.data.datasets[1].data = tiltYVals;
  chartTilt.update("none");

  // Update Vibration
  chartVib.data.labels = labels;
  chartVib.data.datasets[0].data = vibVals;
  chartVib.update("none");

  // Update Risk
  chartRisk.data.labels = labels;
  chartRisk.data.datasets[0].data = riskVals;
  chartRisk.data.datasets[0].borderColor = RISK_COLORS[latest.risk_level];
  chartRisk.update("none");

  // Update Explainable AI (SHAP breakdown)
  updateXAI(latest);
}

function updateXAI(latest) {
  const container = document.getElementById("xai-factors-container");
  if (!latest.human_explanations || latest.human_explanations.length === 0 || latest.risk_level === "NORMAL") {
    container.innerHTML = `<div class="xai-empty">Strata conditions are currently stable within baseline geotechnical tolerances.</div>`;
    return;
  }

  let html = "";
  latest.human_explanations.forEach(exp => {
    html += `
      <div class="xai-factor-bar">
        <div class="xai-factor-header">
          <span>${exp}</span>
        </div>
        <div class="xai-bar-track">
          <div class="xai-bar-fill" style="width: 85%;"></div>
        </div>
      </div>
    `;
  });
  container.innerHTML = html;
}

function updateAlertsTable(alerts) {
  const tbody = document.getElementById("alert-tbody");
  const countBadge = document.getElementById("alert-count-badge");
  countBadge.textContent = `${alerts.length} Active Alerts`;

  if (alerts.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-table">No critical strata anomalies recorded. All galleries stable.</td></tr>`;
    return;
  }

  let html = "";
  alerts.forEach(a => {
    const timeStr = a.timestamp.includes("T") ? a.timestamp.split("T")[1].slice(0, 8) : a.timestamp;
    html += `
      <tr>
        <td>${timeStr}</td>
        <td><strong>${a.node_id}</strong></td>
        <td><span class="badge-alert ${a.severity}">${a.severity}</span></td>
        <td><strong>${a.risk_score.toFixed(1)}</strong></td>
        <td>${a.reason}</td>
      </tr>
    `;
  });
  tbody.innerHTML = html;
}

function hexToRgb(hex) {
  const bigint = parseInt(hex.replace("#", ""), 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `${r}, ${g}, ${b}`;
}

// ==========================================
// CLIENT-SIDE LIVE SCENARIO SIMULATOR
// ==========================================
function triggerScenario(scenarioType) {
  stopSimulation();
  activeScenario = scenarioType;
  simStep = 0;
  document.getElementById("sim-state").textContent = `STREAMING: ${scenarioType.toUpperCase()}`;

  // Stream a new simulated reading every 1.0 second across nodes
  simulationInterval = setInterval(async () => {
    simStep++;
    const now = new Date();
    const ts = now.toISOString();

    const nodes = ["N01", "N02", "N03", "N04", "N05"];

    for (const node of nodes) {
      let disp = 1.0 + Math.random() * 0.02;
      let tilt_x = 0.10 + Math.random() * 0.02;
      let tilt_y = 0.08 + Math.random() * 0.02;
      let vib = 0.03 + Math.random() * 0.01;

      const isEpicenter = node === "N03";
      const isNeighbor = node === "N02" || node === "N04";

      if (scenarioType === "warning") {
        const creep = (simStep * 0.03) * (isEpicenter ? 1.0 : (isNeighbor ? 0.6 : 0.2));
        disp += creep;
        tilt_x += creep * 0.8;
        if (Math.random() > 0.8) vib += 0.15;
      } else if (scenarioType === "high_risk") {
        const prog = Math.min(simStep * 0.12, 5.0);
        disp += prog * (isEpicenter ? 1.0 : (isNeighbor ? 0.7 : 0.3));
        tilt_x += (prog * 0.5) * (isEpicenter ? 1.0 : 0.5);
        tilt_y += (prog * 0.3) * (isEpicenter ? 1.0 : 0.5);
        vib += 0.2 + Math.random() * 0.3;
      } else if (scenarioType === "critical") {
        const jump = Math.min(simStep * 0.4, 15.0);
        disp += jump * (isEpicenter ? 1.0 : (isNeighbor ? 0.8 : 0.4));
        tilt_x += (jump * 0.4) * (isEpicenter ? 1.0 : 0.7);
        tilt_y += (jump * 0.25) * (isEpicenter ? 1.0 : 0.6);
        vib += 0.8 + Math.random() * 1.5;
      } else if (scenarioType === "transient_noise") {
        // High vibration shock for 10 steps, zero displacement change
        if (simStep >= 5 && simStep <= 15) {
          vib = 1.8 + Math.random() * 0.8;
        }
      }

      const payload = {
        node_id: node,
        timestamp: ts,
        tilt_x: parseFloat(tilt_x.toFixed(4)),
        tilt_y: parseFloat(tilt_y.toFixed(4)),
        vibration: parseFloat(vib.toFixed(4)),
        displacement_mm: parseFloat(disp.toFixed(4))
      };

      try {
        await fetch("/api/sensor-data", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
      } catch (e) {
        console.error("Simulation ingestion error:", e);
      }
    }

    refreshFleetData();

    if (simStep > 60) {
      stopSimulation();
    }
  }, 1000);
}

function stopSimulation() {
  if (simulationInterval) {
    clearInterval(simulationInterval);
    simulationInterval = null;
  }
  document.getElementById("sim-state").textContent = "IDLE";
}
