/**
 * Mine Subsidence AI Dashboard Client Logic
 * SIH26025 - NexGen | Real-Time WebSockets Telemetry & Scenario Demonstrator
 */

let selectedNodeId = "N03";
let simulationInterval = null;
let simStep = 0;
let activeScenario = null;
let ws = null;
let wsReconnectTimer = null;
let fallbackPollInterval = null;
let sirenEnabled = true;
let audioCtx = null;
let lastSirenTime = 0;

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
  initWebSocket();
  // Initial REST fetch to populate before first WebSocket broadcast
  refreshFleetData();
  refreshObservatoryData();
});

function initClock() {
  setInterval(() => {
    const now = new Date();
    document.getElementById("live-clock").textContent = now.toLocaleTimeString();
  }, 1000);
}

// ==========================================
// WEBSOCKET REAL-TIME CONNECTION
// ==========================================
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  const wsUrl = `${protocol}//${host}/api/ws/telemetry`;

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("WebSocket connected to:", wsUrl);
      if (fallbackPollInterval) {
        clearInterval(fallbackPollInterval);
        fallbackPollInterval = null;
      }
      // Send keep-alive ping every 25s
      setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send("ping");
        }
      }, 25000);
    };

    ws.onmessage = (event) => {
      if (event.data === "pong") return;
      try {
        const msg = JSON.parse(event.data);
        handleWebSocketMessage(msg);
      } catch (err) {
        console.error("Error parsing WS message:", err);
      }
    };

    ws.onclose = () => {
      console.warn("WebSocket closed. Attempting reconnect in 3s...");
      scheduleReconnect();
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      ws.close();
    };
  } catch (e) {
    console.error("Failed to initialize WebSocket:", e);
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (wsReconnectTimer) return;
  // Activate gentle fallback polling while disconnected
  if (!fallbackPollInterval) {
    fallbackPollInterval = setInterval(refreshFleetData, 5000);
  }
  wsReconnectTimer = setTimeout(() => {
    wsReconnectTimer = null;
    initWebSocket();
  }, 3000);
}

function handleWebSocketMessage(msg) {
  if (msg.type === "INITIAL_SNAPSHOT") {
    if (msg.nodes) updateMapNodes(msg.nodes);
    if (msg.recent_alerts) updateAlertsTable(msg.recent_alerts);
  } else if (msg.type === "TELEMETRY_UPDATE") {
    const reading = msg.reading;
    const node = msg.node;
    const alert = msg.alert;

    // Refresh nodes and risk KPIs
    refreshRiskKPIsOnly();

    // If reading is for currently selected node, append to charts
    if (reading && reading.node_id === selectedNodeId) {
      appendReadingToCharts(reading);
    }

    // Dynamically update Jharia physical sensor card
    if (reading && reading.node_id) {
      const lower = reading.node_id.toLowerCase();
      const elDisp = document.getElementById(`val-${lower}-disp`);
      if (elDisp && reading.displacement_mm !== undefined) elDisp.textContent = `${Number(reading.displacement_mm).toFixed(2)} mm`;
      const elTilt = document.getElementById(`val-${lower}-tilt`);
      if (elTilt && reading.tilt_vector_norm !== undefined) elTilt.textContent = `${Number(reading.tilt_vector_norm).toFixed(2)}°`;
      const elVib = document.getElementById(`val-${lower}-vib`);
      if (elVib && reading.vibration !== undefined) elVib.textContent = `${Number(reading.vibration).toFixed(3)} g`;
      const elRate = document.getElementById(`val-${lower}-rate`);
      if (elRate && reading.disp_rate !== undefined) elRate.textContent = `${Number(reading.disp_rate * 3600).toFixed(2)} mm/h`;
      if (node && node.latest_risk_level) {
        const badge = document.getElementById(`badge-node-${lower}`);
        if (badge) {
          badge.textContent = node.latest_risk_level;
          badge.className = `badge-sm ${(node.latest_risk_level === "CRITICAL" ? "red" : (node.latest_risk_level === "HIGH" ? "orange" : (node.latest_risk_level === "WARNING" || node.latest_risk_level === "MONITORING" ? "yellow" : "green")))}`;
        }
      }
    }

    // If new alert, prepend to alert table
    if (alert) {
      prependAlert(alert);
      if (alert.severity === "CRITICAL" || alert.severity === "HIGH") {
        playEmergencySiren();
      }
    }
  }
}

// ==========================================
// WEB AUDIO API EMERGENCY SIREN
// ==========================================
function toggleSiren() {
  sirenEnabled = !sirenEnabled;
  const btn = document.getElementById("btn-siren-toggle");
  const icon = document.getElementById("siren-icon");
  const status = document.getElementById("siren-status");

  if (sirenEnabled) {
    btn.className = "btn-siren active";
    icon.textContent = "🔔";
    status.textContent = "ENABLED";
  } else {
    btn.className = "btn-siren muted";
    icon.textContent = "🔕";
    status.textContent = "MUTED";
  }
}

function playEmergencySiren() {
  if (!sirenEnabled) return;
  const now = Date.now();
  if (now - lastSirenTime < 4000) return; // Debounce alarm sound
  lastSirenTime = now;

  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(880, audioCtx.currentTime); // High pitch A5
    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.4); // Down to A4
    osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.8);
    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 1.2);

    gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 1.3);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + 1.3);
  } catch (err) {
    console.debug("Audio play error:", err);
  }
}

// ==========================================
// CHART INITIALIZATION
// ==========================================
function initCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 250 },
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
    const resNodes = await fetch("/api/nodes");
    if (resNodes.ok) {
      const nodes = await resNodes.json();
      updateMapNodes(nodes);
    }

    await refreshRiskKPIsOnly();

    const resHist = await fetch(`/api/history?node_id=${selectedNodeId}&limit=30`);
    if (resHist.ok) {
      const histData = await resHist.json();
      updateChartsAndMetrics(histData);
    }

    const resAlerts = await fetch("/api/alerts?limit=10");
    if (resAlerts.ok) {
      const alerts = await resAlerts.json();
      updateAlertsTable(alerts);
    }
  } catch (err) {
    console.debug("Background fleet refresh notice:", err);
  }
}

async function refreshRiskKPIsOnly() {
  try {
    const resRisk = await fetch("/api/risk");
    if (resRisk.ok) {
      const riskSummary = await resRisk.json();
      updateRiskKPIs(riskSummary);
    }
  } catch (e) {
    // Non-blocking
  }
}

function updateMapNodes(nodes) {
  let hasCritical = false;
  nodes.forEach(n => {
    const marker = document.querySelector(`#node-marker-${n.node_id} circle`);
    if (marker) {
      const isSelected = n.node_id === selectedNodeId;
      const isCriticalOrHigh = n.latest_risk_level === "CRITICAL" || n.latest_risk_level === "HIGH";
      if (n.latest_risk_level === "CRITICAL") hasCritical = true;

      // Pulse if selected OR if currently in critical/high warning
      const shouldPulse = isSelected || isCriticalOrHigh;
      marker.className.baseVal = `node-circle ${n.latest_risk_level.toLowerCase()} ${shouldPulse ? "pulse" : ""}`;
    }
  });

  // Toggle Emergency Banner
  const banner = document.getElementById("emergency-banner");
  if (banner) {
    banner.style.display = hasCritical ? "flex" : "none";
  }
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

function appendReadingToCharts(reading) {
  const timeLabel = reading.timestamp.includes("T") ? reading.timestamp.split("T")[1].slice(0, 8) : reading.timestamp;
  
  // Keep max 30 points
  if (chartDisp.data.labels.length >= 30) {
    chartDisp.data.labels.shift();
    chartDisp.data.datasets[0].data.shift();
    chartTilt.data.labels.shift();
    chartTilt.data.datasets[0].data.shift();
    chartTilt.data.datasets[1].data.shift();
    chartVib.data.labels.shift();
    chartVib.data.datasets[0].data.shift();
    chartRisk.data.labels.shift();
    chartRisk.data.datasets[0].data.shift();
  }

  chartDisp.data.labels.push(timeLabel);
  chartDisp.data.datasets[0].data.push(reading.sensor_values.displacement_mm);
  chartDisp.update("none");

  chartTilt.data.labels.push(timeLabel);
  chartTilt.data.datasets[0].data.push(reading.sensor_values.tilt_x);
  chartTilt.data.datasets[1].data.push(reading.sensor_values.tilt_y);
  chartTilt.update("none");

  chartVib.data.labels.push(timeLabel);
  chartVib.data.datasets[0].data.push(reading.sensor_values.vibration);
  chartVib.update("none");

  chartRisk.data.labels.push(timeLabel);
  chartRisk.data.datasets[0].data.push(reading.risk_score);
  chartRisk.data.datasets[0].borderColor = RISK_COLORS[reading.risk_level] || "#ef4444";
  chartRisk.update("none");

  // Update Mini Metrics Ribbon
  document.getElementById("val-vib").textContent = `${reading.sensor_values.vibration.toFixed(3)} g`;
  document.getElementById("val-tilt").textContent = `${reading.sensor_values.tilt_x.toFixed(2)}° / ${reading.sensor_values.tilt_y.toFixed(2)}°`;
  document.getElementById("val-disp").textContent = `${reading.sensor_values.displacement_mm.toFixed(2)} mm`;
  document.getElementById("val-neighbors").textContent = `${reading.neighbour_anomalies} Abnormal`;

  const nodeBadge = document.getElementById("selected-node-badge");
  nodeBadge.textContent = `STATUS: ${reading.risk_level} (Score: ${reading.risk_score.toFixed(1)})`;
  nodeBadge.style.background = `rgba(${hexToRgb(RISK_COLORS[reading.risk_level])}, 0.2)`;
  nodeBadge.style.borderColor = RISK_COLORS[reading.risk_level];
  nodeBadge.style.color = RISK_COLORS[reading.risk_level];

  document.getElementById("kpi-anomaly-status").textContent = reading.anomaly ? "ANOMALY DETECTED" : "NORMAL";
  document.getElementById("val-confidence").textContent = `${(reading.confidence * 100).toFixed(1)}%`;

  updateXAI(reading);
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
  const labels = history.map(h => h.timestamp.includes("T") ? h.timestamp.split("T")[1].slice(0, 8) : h.timestamp);
  const dispVals = history.map(h => h.sensor_values.displacement_mm);
  const tiltXVals = history.map(h => h.sensor_values.tilt_x);
  const tiltYVals = history.map(h => h.sensor_values.tilt_y);
  const vibVals = history.map(h => h.sensor_values.vibration);
  const riskVals = history.map(h => h.risk_score);

  chartDisp.data.labels = labels;
  chartDisp.data.datasets[0].data = dispVals;
  chartDisp.update("none");

  chartTilt.data.labels = labels;
  chartTilt.data.datasets[0].data = tiltXVals;
  chartTilt.data.datasets[1].data = tiltYVals;
  chartTilt.update("none");

  chartVib.data.labels = labels;
  chartVib.data.datasets[0].data = vibVals;
  chartVib.update("none");

  chartRisk.data.labels = labels;
  chartRisk.data.datasets[0].data = riskVals;
  chartRisk.data.datasets[0].borderColor = RISK_COLORS[latest.risk_level] || "#ef4444";
  chartRisk.update("none");

  updateXAI(latest);
}

function updateXAI(latest) {
  const container = document.getElementById("xai-factors-container");
  if (!latest.top_contributing_features || latest.top_contributing_features.length === 0 || latest.risk_level === "NORMAL") {
    container.innerHTML = `<div class="xai-empty">✅ Strata equilibrium stable. All physical sensors, weather percolation, and satellite creep are within baseline limits.</div>`;
    return;
  }

  let html = "";
  const topFactors = latest.top_contributing_features || [];
  const explanations = latest.human_explanations || [];

  topFactors.forEach((factorKey, idx) => {
    const meta = FEATURE_HUMAN_DICTIONARY[factorKey] || {
      name: factorKey.replace(/_/g, " ").toUpperCase(),
      icon: "⚠️",
      simple_desc: "Sensor or geotechnical indicator showing elevated activity.",
      safe_limit: "Normal",
      danger_trigger: "High deformation"
    };
    const humanExp = explanations[idx] || meta.simple_desc;

    html += `
      <div class="xai-factor-bar" onclick="openFeatureGuideModal('${factorKey}')" style="cursor: pointer; margin-bottom: 0.8rem;" title="Click to open Feature Decoder for ${meta.name}">
        <div class="xai-factor-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.2rem;">
          <span style="font-weight: 700; color: #fff; font-size: 0.88rem;">${meta.icon} ${meta.name}</span>
          <span style="font-size: 0.72rem; color: #93c5fd; background: rgba(59, 130, 246, 0.15); padding: 0.1rem 0.4rem; border-radius: 4px;">${meta.safe_limit} Safe Limit</span>
        </div>
        <p style="font-size: 0.8rem; color: #fde68a; margin: 0.2rem 0 0.4rem 0;">${humanExp}</p>
        <div class="xai-bar-track" style="height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden;">
          <div class="xai-bar-fill" style="width: ${Math.max(40, 95 - idx * 15)}%; height: 100%; background: ${RISK_COLORS[latest.risk_level] || '#ef4444'}; border-radius: 3px;"></div>
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
        <td><strong>${Number(a.risk_score).toFixed(1)}</strong></td>
        <td>${a.reason}</td>
      </tr>
    `;
  });
  tbody.innerHTML = html;
}

function prependAlert(alert) {
  const tbody = document.getElementById("alert-tbody");
  const timeStr = alert.timestamp.includes("T") ? alert.timestamp.split("T")[1].slice(0, 8) : alert.timestamp;
  const newRow = `
    <tr>
      <td>${timeStr}</td>
      <td><strong>${alert.node_id}</strong></td>
      <td><span class="badge-alert ${alert.severity}">${alert.severity}</span></td>
      <td><strong>${Number(alert.risk_score).toFixed(1)}</strong></td>
      <td>${alert.reason}</td>
    </tr>
  `;

  if (tbody.querySelector(".empty-table")) {
    tbody.innerHTML = newRow;
  } else {
    tbody.insertAdjacentHTML("afterbegin", newRow);
  }
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
        console.debug("Simulation ingestion error:", e);
      }
    }

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

// ==========================================
// TAB SWITCHING
// ==========================================
function switchTab(tabId) {
  const tabs = ["telemetry", "observatory", "benchmark"];
  tabs.forEach(t => {
    const btn = document.getElementById(`tab-btn-${t}`);
    const pane = document.getElementById(`tab-pane-${t}`);
    if (btn) btn.classList.remove("active");
    if (pane) pane.style.display = "none";
  });

  const activeBtn = document.getElementById(`tab-btn-${tabId}`);
  const activePane = document.getElementById(`tab-pane-${tabId}`);
  if (activeBtn) activeBtn.classList.add("active");
  if (activePane) activePane.style.display = "flex";

  if (tabId === "observatory") {
    if (!satelliteMap) {
      initSatelliteMap();
    }
    setTimeout(() => {
      if (satelliteMap) satelliteMap.invalidateSize();
    }, 200);
    refreshObservatoryData();
  }
}

// ==========================================
// LEAFLET SATELLITE GIS MAP (JHARIA MOONIDIH)
// ==========================================
let satelliteMap = null;
let gisLayers = {
  insar: null,
  ndvi: null,
  thermal: null,
  nodes: null
};

function initSatelliteMap() {
  const mapElement = document.getElementById("satellite-map");
  if (!mapElement || satelliteMap) return;

  // Moonidih Colliery Coordinates: 23.7438° N, 86.4172° E
  const moonidihLat = 23.7438;
  const moonidihLon = 86.4172;

  try {
    satelliteMap = L.map("satellite-map", {
      center: [moonidihLat, moonidihLon],
      zoom: 15,
      zoomControl: true
    });

    // High-Resolution Esri World Imagery (True Color Optical Satellite)
    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Tiles &copy; Esri, Maxar, Earthstar Geographics",
        maxZoom: 18
      }
    ).addTo(satelliteMap);

    // CartoDB Dark Labels Overlay
    L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png",
      {
        attribution: "&copy; CartoDB",
        maxZoom: 18
      }
    ).addTo(satelliteMap);

    // 1. InSAR Subsidence Heatmap Layer (Centred around Moonidih longwall caving trough)
    const insarCircles = [
      L.circle([moonidihLat, moonidihLon], {
        color: "#ef4444",
        fillColor: "#ef4444",
        fillOpacity: 0.45,
        radius: 260
      }).bindPopup("<b>InSAR Primary Subsidence Bowl</b><br>LOS Velocity: <b>-38.5 to -44.2 mm/yr</b><br>Max Cumulative Sag: -182 mm"),
      L.circle([moonidihLat + 0.0015, moonidihLon + 0.001], {
        color: "#f97316",
        fillColor: "#f97316",
        fillOpacity: 0.35,
        radius: 420
      }).bindPopup("<b>InSAR Secondary Creep Zone</b><br>LOS Velocity: <b>-22.0 to -35.0 mm/yr</b>"),
      L.circle([moonidihLat, moonidihLon], {
        color: "#f59e0b",
        fillColor: "#f59e0b",
        fillOpacity: 0.20,
        radius: 650
      }).bindPopup("<b>InSAR Influence Outer Basin</b><br>LOS Velocity: <b>-10.0 to -20.0 mm/yr</b>")
    ];
    gisLayers.insar = L.layerGroup(insarCircles).addTo(satelliteMap);

    // 2. Sentinel-2 NDVI Tension Crack Anomaly Layer
    const ndviPolygon = L.polygon([
      [moonidihLat + 0.002, moonidihLon - 0.0025],
      [moonidihLat + 0.0025, moonidihLon + 0.003],
      [moonidihLat - 0.0015, moonidihLon + 0.0028],
      [moonidihLat - 0.002, moonidihLon - 0.002]
    ], {
      color: "#f59e0b",
      dashArray: "6, 6",
      fillColor: "#eab308",
      fillOpacity: 0.30
    }).bindPopup("<b>Sentinel-2 Tension Fissure Zone</b><br>NDVI Anomaly: <b>-0.24</b> (Severe root rupture)<br>Surface Cracking Detected");
    gisLayers.ndvi = L.layerGroup([ndviPolygon]).addTo(satelliteMap);

    // 3. Landsat-9 Thermal IR Anomaly Layer
    const thermalCircle = L.circle([moonidihLat - 0.001, moonidihLon + 0.0015], {
      color: "#a855f7",
      fillColor: "#c084fc",
      fillOpacity: 0.40,
      radius: 200
    }).bindPopup("<b>Landsat-9 Thermal IR Anomaly</b><br>Surface Temp: <b>38.6°C</b> (&Delta;T = +4.2K)<br>Subsurface Coal Seam Heating");
    gisLayers.thermal = L.layerGroup([thermalCircle]).addTo(satelliteMap);

    // 4. Projected Underground IoT Nodes
    const nodeMarkers = [
      L.circleMarker([moonidihLat - 0.0012, moonidihLon - 0.0015], {
        radius: 8, color: "#38bdf8", fillColor: "#0284c7", fillOpacity: 0.9
      }).bindPopup("<b>Node N01</b><br>Gallery North-1 Intake<br>Depth: 320m<br>Status: NORMAL"),
      L.circleMarker([moonidihLat - 0.0006, moonidihLon - 0.0005], {
        radius: 8, color: "#38bdf8", fillColor: "#0284c7", fillOpacity: 0.9
      }).bindPopup("<b>Node N02</b><br>Pillar Intersection A<br>Depth: 320m<br>Status: NORMAL"),
      L.circleMarker([moonidihLat, moonidihLon], {
        radius: 11, color: "#ef4444", fillColor: "#dc2626", fillOpacity: 0.95
      }).bindPopup("<b>Node N03* (Epicenter)</b><br>Active Longwall Face<br>Depth: 320m<br>Status: MONITORING"),
      L.circleMarker([moonidihLat + 0.0008, moonidihLon + 0.001], {
        radius: 8, color: "#38bdf8", fillColor: "#0284c7", fillOpacity: 0.9
      }).bindPopup("<b>Node N04</b><br>Return Crosscut 2<br>Depth: 320m<br>Status: NORMAL"),
      L.circleMarker([moonidihLat + 0.0004, moonidihLon + 0.002], {
        radius: 8, color: "#38bdf8", fillColor: "#0284c7", fillOpacity: 0.9
      }).bindPopup("<b>Node N05</b><br>Goaf Boundary Panel<br>Depth: 320m<br>Status: NORMAL")
    ];
    gisLayers.nodes = L.layerGroup(nodeMarkers).addTo(satelliteMap);
  } catch (e) {
    console.error("Failed to initialize Leaflet satellite map:", e);
  }
}

function toggleGisLayer(layerType) {
  const btn = document.getElementById(`btn-toggle-${layerType}`);
  const layer = gisLayers[layerType];
  if (!layer || !satelliteMap) return;

  if (satelliteMap.hasLayer(layer)) {
    satelliteMap.removeLayer(layer);
    if (btn) btn.classList.remove("active");
  } else {
    satelliteMap.addLayer(layer);
    if (btn) btn.classList.add("active");
  }
}

// ==========================================
// OBSERVATORY DATA FETCH & PARAMETER EXPLORER
// ==========================================
let allObservatoryRecords = [];

async function refreshObservatoryData() {
  try {
    const res = await fetch("/api/mine/jharia-moonidih");
    if (!res.ok) throw new Error("Failed to fetch mine observatory");
    const data = await res.json();

    // Update Weather Cards
    const w = data.weather || {};
    const rain24 = parseFloat(w.rain_cum_24h_mm || 0);
    const rain72 = parseFloat(w.rain_cum_72h_mm || 0);
    const soil = parseFloat(w.soil_moisture_deep || 0.35);
    const porePressure = (data.geotech_indices && data.geotech_indices.hydraulic_head_pressure_kpa) || (90 + rain72 * 1.8);

    const elRain24 = document.getElementById("obs-rain-24h");
    if (elRain24) elRain24.textContent = `${rain24.toFixed(1)} mm`;
    const elRain72 = document.getElementById("obs-rain-72h");
    if (elRain72) elRain72.textContent = `${rain72.toFixed(1)} mm`;
    const elSoil = document.getElementById("obs-soil-moisture");
    if (elSoil) elSoil.textContent = `${(soil * 100).toFixed(1)} %`;
    const elPore = document.getElementById("obs-pore-pressure");
    if (elPore) elPore.textContent = `${porePressure.toFixed(1)} kPa`;

    const elBar24 = document.getElementById("bar-rain-24h");
    if (elBar24) elBar24.style.width = `${Math.min(100, (rain24 / 50.0) * 100)}%`;
    const elBar72 = document.getElementById("bar-rain-72h");
    if (elBar72) elBar72.style.width = `${Math.min(100, (rain72 / 120.0) * 100)}%`;
    const elBarSoil = document.getElementById("bar-soil");
    if (elBarSoil) elBarSoil.style.width = `${Math.min(100, (soil / 0.6) * 100)}%`;

    // Extended Weather Telemetry
    const elRainRate = document.getElementById("obs-rain-rate");
    if (elRainRate) elRainRate.textContent = Number(w.rain_rate_mm_h || 0.0).toFixed(1);
    const elRain7d = document.getElementById("obs-rain-7d");
    if (elRain7d) elRain7d.textContent = Number(w.rain_cum_7d_mm || (rain72 * 1.7)).toFixed(1);
    const elSoilShallow = document.getElementById("obs-soil-shallow");
    if (elSoilShallow) elSoilShallow.textContent = `${((w.soil_moisture_shallow || 0.28) * 100).toFixed(1)}%`;
    const elSeepage = document.getElementById("obs-seepage-idx");
    if (elSeepage) elSeepage.textContent = Number(data.geotech_indices?.seepage_infiltration_index || 0.44).toFixed(2);

    // Update Satellite Cards
    const s = data.satellite || {};
    const elInsar = document.getElementById("obs-insar-vel");
    if (elInsar) elInsar.textContent = `${parseFloat(s.sentinel1_insar_velocity_mm_yr || -18.4).toFixed(1)} mm/yr`;
    const elCum = document.getElementById("obs-cum-los");
    if (elCum) elCum.textContent = `${parseFloat(s.sentinel1_cum_los_displacement_mm || -32.6).toFixed(1)} mm`;
    const elNdvi = document.getElementById("obs-ndvi-anom");
    if (elNdvi) elNdvi.textContent = `${parseFloat(s.sentinel2_ndvi_anomaly || -0.14).toFixed(2)}`;
    const elTherm = document.getElementById("obs-thermal-anom");
    if (elTherm) elTherm.textContent = `+${parseFloat(s.landsat_thermal_anomaly_k || 3.4).toFixed(1)} K`;

    const elCoherence = document.getElementById("obs-insar-coherence");
    if (elCoherence) elCoherence.innerHTML = `&gamma; = ${s.sentinel1_coherence ? Number(s.sentinel1_coherence).toFixed(2) : "0.76"}`;
    const elSurfaceTemp = document.getElementById("obs-surface-temp");
    if (elSurfaceTemp) elSurfaceTemp.textContent = `${s.landsat_lst_surface_temp_c ? Number(s.landsat_lst_surface_temp_c).toFixed(1) : "36.8"}°C`;

    // Update Subterranean Physical IoT Node Cards (N01 - N05)
    const ps = data.physical_sensors || {};
    const nodes = ps.nodes || {};
    ["N01", "N02", "N03", "N04", "N05"].forEach(nid => {
      const nd = nodes[nid];
      if (!nd) return;
      const lower = nid.toLowerCase();
      const elDisp = document.getElementById(`val-${lower}-disp`);
      if (elDisp) elDisp.textContent = `${Number(nd.displacement_mm).toFixed(2)} mm`;
      const elTilt = document.getElementById(`val-${lower}-tilt`);
      if (elTilt) elTilt.textContent = `${Number(nd.tilt_vector_norm_deg).toFixed(2)}°`;
      const elVib = document.getElementById(`val-${lower}-vib`);
      if (elVib) elVib.textContent = `${Number(nd.vibration_rms_g).toFixed(3)} g`;
      const elRate = document.getElementById(`val-${lower}-rate`);
      if (elRate) elRate.textContent = `${Number(nd.displacement_rate_mm_hr).toFixed(2)} mm/h`;
      const badge = document.getElementById(`badge-node-${lower}`);
      if (badge) {
        badge.textContent = nd.risk_level || "NORMAL";
        badge.className = `badge-sm ${(nd.risk_level === "CRITICAL" ? "red" : (nd.risk_level === "HIGH" ? "orange" : (nd.risk_level === "WARNING" || nd.risk_level === "MONITORING" ? "yellow" : "green")))}`;
      }
    });

    const elGrad = document.getElementById("val-spatial-grad");
    if (elGrad && ps.spatial_disp_gradient_max_mm_m) {
      elGrad.textContent = `${Number(ps.spatial_disp_gradient_max_mm_m).toFixed(2)} mm/m`;
    }

    // Populate Data Table
    allObservatoryRecords = data.sample_records || [];
    renderParameterTable(allObservatoryRecords);
  } catch (err) {
    console.error("Error refreshing observatory:", err);
  }
}

function renderParameterTable(records) {
  const tbody = document.getElementById("multimodal-tbody");
  if (!tbody) return;
  if (!records || records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="13" class="empty-table">No records available.</td></tr>`;
    return;
  }

  let html = "";
  records.slice(0, 100).forEach(r => {
    const timeStr = r.timestamp ? (r.timestamp.includes("T") ? r.timestamp.split("T")[1].slice(0, 8) : r.timestamp) : "--";
    const riskBadge = `<span class="badge-alert ${r.risk_label || 'NORMAL'}">${r.risk_label || 'NORMAL'}</span>`;
    
    // Status pills for physical parameters
    const dispVal = Number(r.displacement_mm || 0);
    let dispPill = dispVal < 2.0 ? `<span class="status-pill-mini normal">SAFE</span>` : (dispVal < 5.0 ? `<span class="status-pill-mini warning">WARN</span>` : `<span class="status-pill-mini danger">HIGH</span>`);
    
    const rain24Val = r.weather_rain_cum_24h_mm !== undefined ? Number(r.weather_rain_cum_24h_mm) : 14.4;
    let rain24Pill = rain24Val < 25 ? `<span class="status-pill-mini normal">LOW</span>` : (rain24Val < 60 ? `<span class="status-pill-mini warning">MOD</span>` : `<span class="status-pill-mini danger">HEAVY</span>`);

    const rain72Val = r.weather_rain_cum_72h_mm !== undefined ? Number(r.weather_rain_cum_72h_mm) : 48.5;
    let rain72Pill = rain72Val < 50 ? `<span class="status-pill-mini normal">DRY</span>` : (rain72Val < 120 ? `<span class="status-pill-mini warning">WET</span>` : `<span class="status-pill-mini danger">SAT</span>`);

    const soilVal = r.weather_soil_moisture_deep !== undefined ? Number(r.weather_soil_moisture_deep) : 0.36;
    let soilPill = soilVal < 0.35 ? `<span class="status-pill-mini normal">OK</span>` : (soilVal < 0.48 ? `<span class="status-pill-mini warning">DAMP</span>` : `<span class="status-pill-mini danger">WATER</span>`);

    const insarVal = r.satellite_insar_velocity_mm_yr !== undefined ? Number(r.satellite_insar_velocity_mm_yr) : -18.4;
    let insarPill = insarVal > -20 ? `<span class="status-pill-mini normal">STABLE</span>` : (insarVal > -35 ? `<span class="status-pill-mini warning">CREEP</span>` : `<span class="status-pill-mini danger">SINK</span>`);

    const ndvi = r.satellite_ndvi_anomaly !== undefined ? Number(r.satellite_ndvi_anomaly).toFixed(2) : "-0.14";
    const lst = r.satellite_thermal_anomaly_k !== undefined ? `+${Number(r.satellite_thermal_anomaly_k).toFixed(1)}K` : "+3.4K";
    const riskScore = r.risk_label === "CRITICAL" ? "88.5" : (r.risk_label === "HIGH" ? "64.2" : (r.risk_label === "WARNING" ? "38.0" : "12.4"));

    html += `
      <tr>
        <td>${timeStr}</td>
        <td><strong>${r.node_id}</strong></td>
        <td>${dispVal.toFixed(2)} mm ${dispPill}</td>
        <td>${Number(r.tilt_x).toFixed(2)}° / ${Number(r.tilt_y).toFixed(2)}°</td>
        <td>${Number(r.vibration).toFixed(3)} g</td>
        <td>${rain24Val.toFixed(1)} mm ${rain24Pill}</td>
        <td>${rain72Val.toFixed(1)} mm ${rain72Pill}</td>
        <td>${(soilVal * 100).toFixed(0)}% ${soilPill}</td>
        <td>${insarVal.toFixed(1)} mm/y ${insarPill}</td>
        <td>${ndvi}</td>
        <td>${lst}</td>
        <td><strong>${riskScore}</strong></td>
        <td>${riskBadge}</td>
      </tr>
    `;
  });
  tbody.innerHTML = html;
}

function filterParameterTable() {
  const query = (document.getElementById("table-search-input").value || "").toUpperCase();
  if (!query) {
    renderParameterTable(allObservatoryRecords);
    return;
  }
  const filtered = allObservatoryRecords.filter(r => {
    return (r.node_id && r.node_id.toUpperCase().includes(query)) ||
           (r.risk_label && r.risk_label.toUpperCase().includes(query)) ||
           (r.timestamp && r.timestamp.toUpperCase().includes(query));
  });
  renderParameterTable(filtered);
}

// ==========================================
// ENVIRONMENTAL MONSOON & CREEP STRESS SIMULATOR
// ==========================================
async function simulateEnvironmentalScenario(type) {
  const envPresets = {
    dry: {
      rain24: 2.0, rain72: 5.0, rain7d: 12.0, soil: 0.22, shallow: 0.18, insar: -12.0, ndvi: -0.02, thermal: 1.2,
      nodes: {
        N01: { disp: 0.85, tilt: 0.14, vib: 0.082, rate: 0.04, risk: "NORMAL" },
        N02: { disp: 1.25, tilt: 0.28, vib: 0.115, rate: 0.08, risk: "NORMAL" },
        N03: { disp: 1.80, tilt: 0.35, vib: 0.120, rate: 0.09, risk: "NORMAL" },
        N04: { disp: 1.10, tilt: 0.20, vib: 0.095, rate: 0.06, risk: "NORMAL" },
        N05: { disp: 1.65, tilt: 0.41, vib: 0.145, rate: 0.11, risk: "NORMAL" }
      },
      gradient: 0.22, name: "Dry Baseline"
    },
    moderate_rain: {
      rain24: 28.0, rain72: 65.0, rain7d: 95.0, soil: 0.38, shallow: 0.32, insar: -21.0, ndvi: -0.12, thermal: 2.8,
      nodes: {
        N01: { disp: 1.10, tilt: 0.22, vib: 0.110, rate: 0.08, risk: "NORMAL" },
        N02: { disp: 1.65, tilt: 0.38, vib: 0.140, rate: 0.12, risk: "NORMAL" },
        N03: { disp: 2.85, tilt: 0.65, vib: 0.225, rate: 0.22, risk: "MONITORING" },
        N04: { disp: 1.35, tilt: 0.25, vib: 0.105, rate: 0.08, risk: "NORMAL" },
        N05: { disp: 2.10, tilt: 0.48, vib: 0.165, rate: 0.15, risk: "NORMAL" }
      },
      gradient: 0.38, name: "Moderate Shower"
    },
    monsoon_surge: {
      rain24: 82.0, rain72: 175.0, rain7d: 220.0, soil: 0.54, shallow: 0.48, insar: -38.5, ndvi: -0.28, thermal: 4.2,
      nodes: {
        N01: { disp: 1.95, tilt: 0.45, vib: 0.210, rate: 0.25, risk: "MONITORING" },
        N02: { disp: 2.70, tilt: 0.72, vib: 0.265, rate: 0.38, risk: "MONITORING" },
        N03: { disp: 5.40, tilt: 1.65, vib: 0.420, rate: 0.75, risk: "WARNING" },
        N04: { disp: 2.10, tilt: 0.52, vib: 0.190, rate: 0.22, risk: "MONITORING" },
        N05: { disp: 3.85, tilt: 1.15, vib: 0.340, rate: 0.52, risk: "WARNING" }
      },
      gradient: 0.74, name: "Monsoon Surge"
    },
    washout: {
      rain24: 125.0, rain72: 240.0, rain7d: 310.0, soil: 0.58, shallow: 0.54, insar: -45.0, ndvi: -0.35, thermal: 5.5,
      nodes: {
        N01: { disp: 3.40, tilt: 0.95, vib: 0.380, rate: 0.65, risk: "WARNING" },
        N02: { disp: 4.80, tilt: 1.45, vib: 0.450, rate: 0.92, risk: "WARNING" },
        N03: { disp: 14.80, tilt: 3.65, vib: 0.780, rate: 3.20, risk: "CRITICAL" },
        N04: { disp: 3.90, tilt: 1.10, vib: 0.360, rate: 0.72, risk: "WARNING" },
        N05: { disp: 8.50, tilt: 2.25, vib: 0.590, rate: 1.85, risk: "CRITICAL" }
      },
      gradient: 1.85, name: "Overburden Washout"
    }
  };

  const p = envPresets[type] || envPresets.dry;

  // Update UI cards
  const elRain24 = document.getElementById("obs-rain-24h");
  if (elRain24) elRain24.textContent = `${p.rain24.toFixed(1)} mm`;
  const elRain72 = document.getElementById("obs-rain-72h");
  if (elRain72) elRain72.textContent = `${p.rain72.toFixed(1)} mm`;
  const elRain7d = document.getElementById("obs-rain-7d");
  if (elRain7d) elRain7d.textContent = `${p.rain7d.toFixed(1)} mm`;
  const elSoil = document.getElementById("obs-soil-moisture");
  if (elSoil) elSoil.textContent = `${(p.soil * 100).toFixed(1)} %`;
  const elSoilShallow = document.getElementById("obs-soil-shallow");
  if (elSoilShallow) elSoilShallow.textContent = `${(p.shallow * 100).toFixed(1)}%`;
  const elPore = document.getElementById("obs-pore-pressure");
  if (elPore) elPore.textContent = `${(90 + p.rain72 * 1.8).toFixed(1)} kPa`;
  const elInsar = document.getElementById("obs-insar-vel");
  if (elInsar) elInsar.textContent = `${p.insar.toFixed(1)} mm/yr`;
  const elNdvi = document.getElementById("obs-ndvi-anom");
  if (elNdvi) elNdvi.textContent = `${p.ndvi.toFixed(2)}`;
  const elTherm = document.getElementById("obs-thermal-anom");
  if (elTherm) elTherm.textContent = `+${p.thermal.toFixed(1)} K`;

  const elBar24 = document.getElementById("bar-rain-24h");
  if (elBar24) elBar24.style.width = `${Math.min(100, (p.rain24 / 50.0) * 100)}%`;
  const elBar72 = document.getElementById("bar-rain-72h");
  if (elBar72) elBar72.style.width = `${Math.min(100, (p.rain72 / 120.0) * 100)}%`;
  const elBarSoil = document.getElementById("bar-soil");
  if (elBarSoil) elBarSoil.style.width = `${Math.min(100, (p.soil / 0.6) * 100)}%`;

  const elKpiWeather = document.getElementById("kpi-weather-status");
  if (elKpiWeather) elKpiWeather.textContent = `${p.rain24.toFixed(1)} mm`;
  const elKpiSoil = document.getElementById("kpi-soil-status");
  if (elKpiSoil) elKpiSoil.textContent = `Deep Saturation: ${(p.soil * 100).toFixed(1)}%`;
  const elKpiInsar = document.getElementById("kpi-insar-val");
  if (elKpiInsar) elKpiInsar.textContent = `${p.insar.toFixed(1)}`;

  // Update physical nodes deck
  if (p.nodes) {
    Object.keys(p.nodes).forEach(nid => {
      const nd = p.nodes[nid];
      const lower = nid.toLowerCase();
      const elDisp = document.getElementById(`val-${lower}-disp`);
      if (elDisp) elDisp.textContent = `${nd.disp.toFixed(2)} mm`;
      const elTilt = document.getElementById(`val-${lower}-tilt`);
      if (elTilt) elTilt.textContent = `${nd.tilt.toFixed(2)}°`;
      const elVib = document.getElementById(`val-${lower}-vib`);
      if (elVib) elVib.textContent = `${nd.vib.toFixed(3)} g`;
      const elRate = document.getElementById(`val-${lower}-rate`);
      if (elRate) elRate.textContent = `${nd.rate.toFixed(2)} mm/h`;
      const badge = document.getElementById(`badge-node-${lower}`);
      if (badge) {
        badge.textContent = nd.risk;
        badge.className = `badge-sm ${(nd.risk === "CRITICAL" ? "red" : (nd.risk === "HIGH" ? "orange" : (nd.risk === "WARNING" || nd.risk === "MONITORING" ? "yellow" : "green")))}`;
      }
    });
  }
  const elGrad = document.getElementById("val-spatial-grad");
  if (elGrad) elGrad.textContent = `${p.gradient.toFixed(2)} mm/m`;

  // Send reading with this weather & satellite override to Node N03
  const dispBase = (type === "washout") ? 14.8 : ((type === "monsoon_surge") ? 5.4 : 1.8);
  const tiltBase = (type === "washout") ? 3.65 : ((type === "monsoon_surge") ? 1.65 : 0.35);

  const payload = {
    node_id: "N03",
    timestamp: new Date().toISOString(),
    tilt_x: tiltBase,
    tilt_y: tiltBase * 0.7,
    vibration: (type === "washout") ? 0.78 : (type === "monsoon_surge" ? 0.42 : 0.12),
    displacement_mm: dispBase,
    weather: {
      rain_rate_mm_h: p.rain24 > 50 ? 15.0 : (p.rain24 > 20 ? 4.0 : 0.0),
      rain_cum_24h_mm: p.rain24,
      rain_cum_72h_mm: p.rain72,
      soil_moisture_deep: p.soil
    },
    satellite: {
      sentinel1_insar_velocity_mm_yr: p.insar,
      sentinel2_ndvi_anomaly: p.ndvi,
      landsat_thermal_anomaly_k: p.thermal
    }
  };

  try {
    const res = await fetch("/api/sensor-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (result.success) {
      const elMaxRisk = document.getElementById("kpi-max-risk");
      if (elMaxRisk) elMaxRisk.textContent = Number(result.risk_score).toFixed(1);
    }
  } catch (e) {
    console.debug("Error simulating environmental payload:", e);
  }
}

// ==============================================================
// PLAIN-ENGLISH FEATURE DICTIONARY & INTERACTIVE GUIDE MODAL
// ==============================================================
const FEATURE_HUMAN_DICTIONARY = {
  // --- 1. SUBTERRANEAN HARDWARE SENSORS (ROOF, PILLAR, SEISMIC) ---
  disp_rate: {
    category: "hardware",
    category_name: "⚡ Subterranean Hardware",
    name: "Roof Sagging Speed",
    icon: "📏",
    simple_desc: "Measures how many millimetres the underground tunnel roof is sinking downward every minute via digital extensometers.",
    safe_limit: "< 0.5 mm/min",
    alarm_threshold: "> 1.5 mm/min",
    danger_trigger: "Rapid bed separation of immediate shale roof from competent sandstone overburden, indicating imminent fall.",
    dgms_clause: "DGMS Tech Circular No. 6 (SMP Strata Control)"
  },
  displacement_mm: {
    category: "hardware",
    category_name: "⚡ Subterranean Hardware",
    name: "Cumulative Roof Sag",
    icon: "📐",
    simple_desc: "Total downward physical drop of the immediate mine roof strata from its original baseline height.",
    safe_limit: "< 3.0 mm",
    alarm_threshold: "> 8.0 mm",
    danger_trigger: "Exceeds safe yield capacity of steel roof bolts and hydraulic props, leading to roadway collapse.",
    dgms_clause: "Coal Mines Regulations (CMR) 1957 / 2017 Reg. 111"
  },
  disp_rate_max: {
    category: "hardware",
    category_name: "⚡ Subterranean Hardware",
    name: "Peak Roof Drop Velocity",
    icon: "⚡",
    simple_desc: "Fastest downward convergence acceleration spike recorded within the rolling 60-second observation window.",
    safe_limit: "< 0.8 mm/s",
    alarm_threshold: "> 2.0 mm/s",
    danger_trigger: "Tertiary creep inflection point indicating dynamic tensile snapping of roof anchoring layers.",
    dgms_clause: "DGMS Circular No. 2 (Mandatory Convergence Evacuation)"
  },
  tilt_vector_norm: {
    category: "hardware",
    category_name: "⚡ Subterranean Hardware",
    name: "Pillar Lean / Tilt Angle",
    icon: "📐",
    simple_desc: "Resultant angular inclination from true vertical of structural coal pillars or steel standing arches.",
    safe_limit: "< 0.8°",
    alarm_threshold: "> 2.5°",
    danger_trigger: "Eccentric bending and spalling of coal rib pillars under severe abutment pressure transfer.",
    dgms_clause: "DGMS (Tech) S&T Guidelines on Pillar Stability"
  },
  tilt_x_current: {
    category: "hardware",
    category_name: "⚡ Subterranean Hardware",
    name: "Lateral Pillar Tilt (E-W)",
    icon: "↔️",
    simple_desc: "Transverse axis rotation angle across the roadway width, tracking asymmetric side-wall thrust.",
    safe_limit: "± 0.5°",
    alarm_threshold: "> 2.0°",
    danger_trigger: "Transverse structural buckling of standing steel supports and rib-side crushing.",
    dgms_clause: "DGMS Guidelines for Support in Roadways"
  },
  tilt_rate_max: {
    category: "hardware",
    category_name: "⚡ Subterranean Hardware",
    name: "Pillar Tilting Speed",
    icon: "🔄",
    simple_desc: "Angular velocity measuring how rapidly support arches or pillars are twisting out of vertical alignment.",
    safe_limit: "< 0.1°/min",
    alarm_threshold: "> 0.5°/min",
    danger_trigger: "Active unseating and dislodgement of support props under dynamic weighting.",
    dgms_clause: "DGMS Strata Control Manual"
  },
  vibration: {
    category: "hardware",
    category_name: "⚡ Subterranean Hardware",
    name: "Micro-Seismic Tremor Energy",
    icon: "〰️",
    simple_desc: "RMS seismic vibration amplitude caused when brittle rock strata fractures and breaks under overburden stress.",
    safe_limit: "< 0.08 g",
    alarm_threshold: "> 0.40 g",
    danger_trigger: "Brittle rockburst or violent seismic fracture of main sandstone roof beam.",
    dgms_clause: "DGMS Guidelines on Rockburst & Bump Monitoring"
  },
  vib_kurtosis: {
    category: "hardware",
    category_name: "⚡ Subterranean Hardware",
    name: "Vibration Burst Spikiness",
    icon: "⚡",
    simple_desc: "Statistical peak sharpness indicator that distinguishes impulsive rock cracking from continuous mining machine hum.",
    safe_limit: "< 4.0",
    alarm_threshold: "> 9.0",
    danger_trigger: "Impulsive snapping of thick sandstone beds prior to large-scale roof collapse.",
    dgms_clause: "National Institute of Rock Mechanics (NIRM) Standards"
  },
  vib_spectral_centroid: {
    category: "hardware",
    category_name: "⚡ Subterranean Hardware",
    name: "Seismic Acoustic Pitch",
    icon: "🔊",
    simple_desc: "Spectral center of gravity of rock acoustic emissions. Shifts from low rumble to high pitch as micro-fissures propagate.",
    safe_limit: "< 18 Hz",
    alarm_threshold: "> 35 Hz",
    danger_trigger: "High-frequency micro-cracking propagation throughout the immediate beam foundation.",
    dgms_clause: "NIRM Acoustic Emission Monitoring Benchmark"
  },

  // --- 2. MONSOON & METEOROLOGY ---
  rain_cum_24h_mm: {
    category: "weather",
    category_name: "🌧️ Surface Meteorology",
    name: "24-Hour Rain Infiltration",
    icon: "🌧️",
    simple_desc: "Total rain that soaked into topsoil directly over the Jharia coalfield seam within the past 24 hours.",
    safe_limit: "< 25.0 mm",
    alarm_threshold: "> 75.0 mm",
    danger_trigger: "Intense monsoon downpours saturating fractured overburden and weakening fault barriers.",
    dgms_clause: "IMD Monsoon Disaster Protocol / DGMS Flooding Warning"
  },
  rain_cum_72h_mm: {
    category: "weather",
    category_name: "🌧️ Surface Meteorology",
    name: "72-Hour Monsoon Accumulation",
    icon: "🌊",
    simple_desc: "3-day cumulative precipitation depth building massive hydrostatic groundwater pressure on the mine ceiling.",
    safe_limit: "< 50.0 mm",
    alarm_threshold: "> 140.0 mm",
    danger_trigger: "Hydrostatic water weight saturating porous sandstone and triggering sudden inundation / roof fall.",
    dgms_clause: "DGMS Standing Order on Surface Inundation & Goaf Water"
  },
  soil_moisture_deep: {
    category: "weather",
    category_name: "🌧️ Surface Meteorology",
    name: "Deep Overburden Water Saturation",
    icon: "💧",
    simple_desc: "Volumetric moisture fraction in deep subsoil (28-100cm depth) percolating downward through rock joints.",
    safe_limit: "< 35%",
    alarm_threshold: "> 52%",
    danger_trigger: "Loss of overburden shear strength and clay liquefaction above active longwall panels.",
    dgms_clause: "Central Mine Planning & Design Institute (CMPDI) Geotech"
  },
  hydro_mechanical_coupling_risk: {
    category: "weather",
    category_name: "🌧️ Surface Meteorology",
    name: "Water-Weighted Sag Velocity",
    icon: "⚠️",
    simple_desc: "NexGen AI index multiplying groundwater saturation weight by underground roof convergence speed.",
    safe_limit: "< 0.50",
    alarm_threshold: "> 2.20",
    danger_trigger: "Synergistic catastrophe: high groundwater thrust accelerating structural roof collapse.",
    dgms_clause: "SIH26025 Multi-Modal Geotechnical Standard"
  },
  env_pore_pressure_kpa: {
    category: "weather",
    category_name: "🌧️ Surface Meteorology",
    name: "Pore Water Pressure",
    icon: "🚰",
    simple_desc: "Fluid pressure inside rock fissures that reduces Terzaghi effective rock clamping friction.",
    safe_limit: "< 120 kPa",
    alarm_threshold: "> 240 kPa",
    danger_trigger: "Hydrostatic thrust opening joints and reducing normal clamping force to zero (hydro-fracturing).",
    dgms_clause: "Terzaghi Effective Stress Geotechnical Code"
  },

  // --- 3. SATELLITE EARTH OBSERVATION ---
  sentinel1_insar_velocity_mm_yr: {
    category: "satellite",
    category_name: "🛰️ Satellite Remote Sensing",
    name: "Satellite Ground Sinking Velocity",
    icon: "🛰️",
    simple_desc: "European Space Agency Sentinel-1 C-Band radar interferometry tracking surface subsidence rates over Jharia.",
    safe_limit: "> -15 mm/yr",
    alarm_threshold: "< -35 mm/yr",
    danger_trigger: "Rapidly deepening surface subsidence bowl above extraction panel threatening surface infrastructure.",
    dgms_clause: "Copernicus Sentinel-1 / NRSC InSAR Mining Guidelines"
  },
  sentinel1_cum_los_displacement_mm: {
    category: "satellite",
    category_name: "🛰️ Satellite Remote Sensing",
    name: "Total Surface Depression",
    icon: "🌐",
    simple_desc: "Total cumulative vertical ground depression bowl tracked by satellite line-of-sight radar.",
    safe_limit: "> -20 mm",
    alarm_threshold: "< -50 mm",
    danger_trigger: "Regional subsidence trough causing tension cracks, road fracture, and surface building damage.",
    dgms_clause: "National Remote Sensing Centre (NRSC) Guidelines"
  },
  sentinel2_ndvi_anomaly: {
    category: "satellite",
    category_name: "🛰️ Satellite Remote Sensing",
    name: "Vegetation Strain & Tension Fissure",
    icon: "🌱",
    simple_desc: "Sentinel-2 multispectral NDVI drop occurring when ground tension cracks sever tree and shrub root systems.",
    safe_limit: "> -0.05",
    alarm_threshold: "< -0.22",
    danger_trigger: "Surface tension fissures actively ripping open across agricultural and residential buffer zones.",
    dgms_clause: "Sentinel-2 Multi-Spectral Surface Crack Detection"
  },
  landsat_thermal_anomaly_k: {
    category: "satellite",
    category_name: "🛰️ Satellite Remote Sensing",
    name: "Coal Fire Thermal Anomaly",
    icon: "🔥",
    simple_desc: "Landsat-9 Thermal Infrared surface heat spikes caused by spontaneous combustion in Jharia coal seams.",
    safe_limit: "< +2.0 K",
    alarm_threshold: "> +4.5 K",
    danger_trigger: "Active subsurface coal fires pyrolyzing and degrading rock tensile strength above mining galleries.",
    dgms_clause: "BCCL / DGMS Coal Fire Safety & Hazard Mapping"
  },

  // --- 4. SPATIAL MULTI-NODE MESH ---
  spatial_disp_gradient_max: {
    category: "spatial",
    category_name: "🕸️ Spatial Mesh Network",
    name: "Differential Sag (Pillar A vs B)",
    icon: "🕸️",
    simple_desc: "Maximum difference in roof drop between two adjacent sensor nodes in the gallery grid.",
    safe_limit: "< 0.8 mm/m",
    alarm_threshold: "> 2.5 mm/m",
    danger_trigger: "Severe shear bending strain tearing the roof beam between adjacent support arches.",
    dgms_clause: "Spatial Differential Subsidence Safety Code"
  },
  spatial_tilt_divergence: {
    category: "spatial",
    category_name: "🕸️ Spatial Mesh Network",
    name: "Support Arch Twisting / Divergence",
    icon: "🔄",
    simple_desc: "Adjacent sensor nodes tilting in opposite directions, revealing torsional strain along the goaf edge.",
    safe_limit: "< 0.5°/m",
    alarm_threshold: "> 1.8°/m",
    danger_trigger: "Torsional arch deformation and catastrophic buckling at the collapse boundary.",
    dgms_clause: "DGMS Powered Support Arch Convergence Standards"
  },
  spatial_fleet_cluster_sag_score: {
    category: "spatial",
    category_name: "🕸️ Spatial Mesh Network",
    name: "Fleet-Wide Collapse Consensus",
    icon: "📍",
    simple_desc: "Consensus metric confirming whether multiple independent nodes agree that the whole strata block is settling.",
    safe_limit: "< 25.0",
    alarm_threshold: "> 70.0",
    danger_trigger: "Systemic multi-pillar failure across the entire mining district (not a single sensor glitch).",
    dgms_clause: "DGMS Real-time Fleet Monitoring Compliance"
  }
};

let currentGuideCategory = "all";
let currentSearchQuery = "";

function onFeatureSearchInput(val) {
  currentSearchQuery = (val || "").trim().toLowerCase();
  const clearBtn = document.getElementById("modal-search-clear");
  if (clearBtn) {
    clearBtn.style.display = currentSearchQuery ? "inline-flex" : "none";
  }
  renderFeatureGuideGrid(currentGuideCategory, null, currentSearchQuery);
}

function clearFeatureSearch() {
  const input = document.getElementById("modal-search-input");
  if (input) input.value = "";
  currentSearchQuery = "";
  const clearBtn = document.getElementById("modal-search-clear");
  if (clearBtn) clearBtn.style.display = "none";
  renderFeatureGuideGrid(currentGuideCategory, null, "");
}

function openFeatureGuideModal(highlightKey = null) {
  const modal = document.getElementById("feature-guide-modal");
  if (!modal) return;
  modal.style.display = "flex";
  document.body.classList.add("modal-open");

  // If a specific key is passed, switch to its category or stay on 'all'
  if (highlightKey && FEATURE_HUMAN_DICTIONARY[highlightKey]) {
    currentGuideCategory = FEATURE_HUMAN_DICTIONARY[highlightKey].category;
  } else {
    currentGuideCategory = "all";
  }

  // Clear search on new open
  const searchInput = document.getElementById("modal-search-input");
  if (searchInput) searchInput.value = "";
  currentSearchQuery = "";
  const clearBtn = document.getElementById("modal-search-clear");
  if (clearBtn) clearBtn.style.display = "none";

  updateModalTabButtons();
  renderFeatureGuideGrid(currentGuideCategory, highlightKey, "");

  // Reset scroll to top
  const grid = document.getElementById("feature-guide-grid");
  if (grid) {
    grid.scrollTop = 0;
  }

  if (highlightKey) {
    setTimeout(() => {
      const card = document.getElementById(`guide-card-${highlightKey}`);
      if (card) {
        card.scrollIntoView({ behavior: "smooth", block: "center" });
        card.classList.add("highlighted-card");
        setTimeout(() => card.classList.remove("highlighted-card"), 3000);
      }
    }, 180);
  }
}

function closeFeatureGuideModal() {
  const modal = document.getElementById("feature-guide-modal");
  if (modal) modal.style.display = "none";
  document.body.classList.remove("modal-open");
}

function filterGuideCategory(cat) {
  currentGuideCategory = cat;
  updateModalTabButtons();
  renderFeatureGuideGrid(cat, null, currentSearchQuery);
  const grid = document.getElementById("feature-guide-grid");
  if (grid) grid.scrollTop = 0;
}

function updateModalTabButtons() {
  const categories = ["all", "hardware", "weather", "satellite", "spatial"];
  categories.forEach(cat => {
    const btn = document.getElementById(`modal-tab-${cat}`);
    if (btn) {
      if (cat === currentGuideCategory) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    }
  });
}

function renderFeatureGuideGrid(category = "all", highlightKey = null, searchQuery = "") {
  const grid = document.getElementById("feature-guide-grid");
  if (!grid) return;

  const entries = Object.entries(FEATURE_HUMAN_DICTIONARY);
  let filtered = (category === "all") 
    ? entries 
    : entries.filter(([_, item]) => item.category === category);

  if (searchQuery) {
    filtered = filtered.filter(([key, item]) => {
      return key.toLowerCase().includes(searchQuery) ||
             item.name.toLowerCase().includes(searchQuery) ||
             item.simple_desc.toLowerCase().includes(searchQuery) ||
             item.danger_trigger.toLowerCase().includes(searchQuery) ||
             item.category_name.toLowerCase().includes(searchQuery);
    });
  }

  // Update count badge
  const countBadge = document.getElementById("modal-features-count");
  if (countBadge) {
    countBadge.textContent = `${filtered.length} Parameter${filtered.length === 1 ? '' : 's'} Shown`;
  }

  if (filtered.length === 0) {
    grid.innerHTML = `
      <div class="guide-empty-state">
        <div class="empty-icon">🔍</div>
        <h3>No parameters match "${searchQuery}"</h3>
        <p>Try searching for words like 'rain', 'tilt', 'vibration', 'insar', or switch categories.</p>
        <button class="btn-guide-reset" onclick="clearFeatureSearch()">Reset Search</button>
      </div>
    `;
    return;
  }

  let html = "";
  filtered.forEach(([key, item]) => {
    const isHighlight = (key === highlightKey);
    const highlightClass = isHighlight ? "highlighted-card" : "";

    html += `
      <div class="guide-card ${item.category} ${highlightClass}" id="guide-card-${key}">
        <div class="guide-card-topbar">
          <span class="guide-cat-badge ${item.category}">${item.category_name}</span>
          <span class="guide-card-code">${key}</span>
        </div>
        
        <div class="guide-card-title-row">
          <span class="guide-icon">${item.icon}</span>
          <h3 class="guide-card-name">${item.name}</h3>
        </div>

        <p class="guide-card-desc">${item.simple_desc}</p>
        
        <div class="guide-card-metrics">
          <div class="guide-metric-col safe">
            <span class="metric-heading">NORMAL / SAFE</span>
            <div class="metric-val-box safe">
              <span class="metric-dot green"></span>
              <span class="metric-value">${item.safe_limit}</span>
            </div>
          </div>
          <div class="guide-metric-col danger">
            <span class="metric-heading">ALARM THRESHOLD</span>
            <div class="metric-val-box danger">
              <span class="metric-dot red"></span>
              <span class="metric-value">${item.alarm_threshold}</span>
            </div>
          </div>
        </div>

        <div class="guide-card-danger">
          <div class="danger-header">
            <span class="danger-icon">⚠️</span>
            <strong>FAILURE MECHANISM & IMPACT</strong>
          </div>
          <p class="danger-text">${item.danger_trigger}</p>
          <div class="dgms-tag">
            <span>📋</span> ${item.dgms_clause}
          </div>
        </div>
      </div>
    `;
  });

  grid.innerHTML = html;
}

// Close modal when user clicks outside the modal dialog or presses ESC
window.addEventListener("click", (e) => {
  const modal = document.getElementById("feature-guide-modal");
  if (e.target === modal) {
    closeFeatureGuideModal();
  }
});

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeFeatureGuideModal();
  }
});
