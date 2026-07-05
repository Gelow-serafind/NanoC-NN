const state = {
  model: null,
  job: null,
};

const el = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", () => {
  el("fileInput").addEventListener("change", onFileSelected);
  el("convertBtn").addEventListener("click", runConvert);
  el("regressionBtn").addEventListener("click", runRegression);
  document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => useSample(button.dataset.sample));
  });
});

async function onFileSelected(event) {
  const file = event.target.files[0];
  if (!file) return;
  setLog(`Uploading ${file.name} ...`);
  const response = await fetch("/api/models/upload", {
    method: "POST",
    headers: { "X-Filename": encodeURIComponent(file.name) },
    body: file,
  });
  const data = await response.json();
  if (!response.ok || data.error) {
    setLog(data.error || "Upload failed");
    return;
  }
  setModel(data.model);
}

async function useSample(name) {
  setLog(`Loading sample model: ${name}`);
  const response = await fetch(`/api/models/sample?name=${encodeURIComponent(name)}`, {
    method: "POST",
  });
  const data = await response.json();
  if (!response.ok || data.error) {
    setLog(data.error || "Failed to load sample");
    return;
  }
  setModel(data.model);
}

function setModel(model) {
  state.model = model;
  el("modelName").textContent = model.name;
  el("modelMeta").textContent = `${formatBytes(model.sizeBytes)} · opset ${formatOpsets(model.opsets)} · ${model.quantizedHint ? "量化" : "未识别量化"}`;
  el("inputSummary").textContent = summarizeTensor(model.inputs[0]);
  el("outputSummary").textContent = summarizeTensor(model.outputs[0]);
  el("nodeCount").textContent = String(model.nodeCount);
  el("quantHint").textContent = model.quantizedHint ? "Q/DQ 或 QLinear" : "未知";
  renderGraph(model.nodes);
  renderOps(model.opCounts);
  setLog(`Loaded ${model.name}\nNodes: ${model.nodeCount}\nOps: ${model.opCounts.map((x) => `${x.op} x${x.count}`).join(", ")}`);
}

async function runConvert() {
  if (!state.model) {
    setLog("请先导入 ONNX 模型。");
    return;
  }
  setStatus("running", "转换中", "正在执行 converter + codegen");
  setLog("Running conversion ...");
  const payload = {
    modelId: state.model.id,
    outRoot: el("outRoot").value,
    prefix: el("prefix").value,
    target: el("target").value,
    backend: el("backend").value,
    sramBudget: el("sramBudget").value,
    flashBudget: el("flashBudget").value,
  };
  const response = await fetch("/api/convert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || data.error) {
    setStatus("unsupported", "error", data.error || "转换失败");
    setLog(data.error || "转换失败");
    return;
  }
  state.job = data.job;
  const job = data.job;
  setStatus(job.status, job.status, statusText(job.status));
  setLog([
    `$ ${job.command.join(" ")}`,
    "",
    `exit=${job.exitCode} duration=${job.durationSec}s status=${job.status}`,
    "",
    job.stdout || "",
    job.stderr ? `\nSTDERR:\n${job.stderr}` : "",
  ].join("\n"));
  renderLinks("reportsList", job.reports);
  renderLinks("filesList", job.files);
  el("jobInfo").textContent = job.outRoot;
}

async function runRegression() {
  setLog("Running TDD stable regression ...");
  const response = await fetch("/api/tdd/regression", { method: "POST" });
  const data = await response.json();
  setLog([
    `$ ${data.command.join(" ")}`,
    "",
    `exit=${data.exitCode} duration=${data.durationSec}s`,
    "",
    data.stdout || "",
    data.stderr ? `\nSTDERR:\n${data.stderr}` : "",
  ].join("\n"));
}

function renderGraph(nodes) {
  const canvas = el("graphCanvas");
  canvas.innerHTML = "";
  const visible = nodes.slice(0, 12);
  visible.forEach((node, index) => {
    if (index > 0) {
      const edge = document.createElement("div");
      edge.className = "edge";
      canvas.appendChild(edge);
    }
    const item = document.createElement("div");
    item.className = "node";
    item.textContent = node.op;
    item.title = node.name;
    canvas.appendChild(item);
  });
  if (nodes.length > visible.length) {
    const edge = document.createElement("div");
    edge.className = "edge";
    canvas.appendChild(edge);
    const more = document.createElement("div");
    more.className = "node";
    more.textContent = `+${nodes.length - visible.length}`;
    canvas.appendChild(more);
  }
}

function renderOps(ops) {
  const tbody = el("opTable");
  tbody.innerHTML = "";
  ops.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${escapeHtml(item.op)}</td><td>${item.count}</td>`;
    tbody.appendChild(row);
  });
}

function renderLinks(targetId, items) {
  const box = el(targetId);
  box.innerHTML = "";
  if (!items.length) {
    box.innerHTML = '<span class="muted">暂无</span>';
    return;
  }
  items.forEach((item) => {
    const button = document.createElement("button");
    button.textContent = item.path || item.name;
    button.addEventListener("click", () => openFile(item.path));
    box.appendChild(button);
  });
}

async function openFile(path) {
  if (!state.job) return;
  const response = await fetch(`/api/file?jobId=${encodeURIComponent(state.job.id)}&path=${encodeURIComponent(path)}`);
  const data = await response.json();
  el("fileTitle").textContent = data.path || path;
  el("fileContent").textContent = data.content || data.error || "";
}

function setStatus(kind, status, text) {
  const box = el("statusBox");
  box.className = `status-box ${kind}`;
  el("codegenStatus").textContent = status;
  el("statusText").textContent = text;
}

function setLog(text) {
  el("logBox").textContent = text;
}

function statusText(status) {
  return {
    ok: "生成语义通过，可导出 C 工程",
    blocked: "存在算子、量化或 renderer 缺口",
    unsupported: "当前明确不支持该模型形态",
    oversize: "生成语义通过，但目标平台预算不足",
  }[status] || "查看日志和报告";
}

function summarizeTensor(tensor) {
  if (!tensor) return "-";
  return `${tensor.name} [${tensor.shape.join(", ")}]`;
}

function formatOpsets(opsets) {
  if (!opsets || !opsets.length) return "-";
  return opsets.map((item) => item.version).join("/");
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
