const $ = (id) => document.getElementById(id);

const fileInput = $("file-input");
const fileLabel = $("file-label");
const dropzone = $("dropzone");
const submitBtn = $("submit-btn");
const sampleBtn = $("sample-btn");
const downloadBtn = $("download-btn");
const malOnly = $("mal-only");
const themeBtn = $("theme-toggle");
const statusText = $("status-text");
const spinner = $("spinner");
const results = $("results");
const metricsCard = $("metrics-card");
const rowsCap = $("rows-cap");

let pickedFile = null;
let lastRows = [];
let sort = null;
let page = 1;
let pageSize = 100;

const storedTheme = localStorage.getItem("theme");
if (storedTheme) document.documentElement.dataset.theme = storedTheme;
themeBtn.addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme;
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("theme", next);
});

fetch("/api/health")
  .then((r) => r.json())
  .then((j) => {
    const p = j.predictor;
    const date = p.trained_at ? ` · ${p.trained_at}` : "";
    $("predictor-tag").textContent = `${p.name} v${p.version}${date}`;
  })
  .catch(() => ($("predictor-tag").textContent = "unreachable"));

fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
});

function setFile(f) {
  pickedFile = f || null;
  if (!pickedFile) {
    fileLabel.textContent = "Choose a .csv file or drop it here";
    submitBtn.disabled = true;
    return;
  }
  if (!pickedFile.name.toLowerCase().endsWith(".csv")) {
    setStatus(`"${pickedFile.name}" is not a .csv file.`, "error");
    pickedFile = null;
    submitBtn.disabled = true;
    fileLabel.textContent = "Choose a .csv file or drop it here";
    fileInput.value = "";
    return;
  }
  fileLabel.textContent = `${pickedFile.name} (${(pickedFile.size / 1024 / 1024).toFixed(2)} MB)`;
  submitBtn.disabled = false;
  setStatus("", "");
}

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!pickedFile) return;
  await runPrediction(() => {
    const fd = new FormData();
    fd.append("file", pickedFile);
    return fetch("/api/predict", { method: "POST", body: fd });
  });
});

sampleBtn.addEventListener("click", async () => {
  pickedFile = null;
  fileInput.value = "";
  fileLabel.textContent = "Choose a .csv file or drop it here";
  submitBtn.disabled = true;
  await runPrediction(() => fetch("/api/sample"));
});

downloadBtn.addEventListener("click", async () => {
  if (!pickedFile && !downloadBtn.dataset.source) return;
  downloadBtn.disabled = true;
  setStatus("Preparing download…", "info", true);
  try {
    let res;
    if (pickedFile) {
      const fd = new FormData();
      fd.append("file", pickedFile);
      res = await fetch("/api/predict.csv", { method: "POST", body: fd });
    } else {
      const sample = await fetch("/api/sample").then((r) => r.json());
      const blob = buildCsvFromPreview(sample);
      triggerDownload(blob, "predictions_sample.csv");
      setStatus("Sample predictions downloaded (preview rows only).", "info");
      downloadBtn.disabled = false;
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    triggerDownload(blob, "predictions.csv");
    setStatus("Predictions downloaded.", "info");
  } catch (err) {
    setStatus(`Download failed: ${err.message}`, "error");
  } finally {
    downloadBtn.disabled = false;
  }
});

malOnly.addEventListener("change", () => { page = 1; renderRows(); });

$("page-first").addEventListener("click", () => { page = 1; renderRows(); });
$("page-prev").addEventListener("click", () => { page = Math.max(1, page - 1); renderRows(); });
$("page-next").addEventListener("click", () => { page = page + 1; renderRows(); });
$("page-last").addEventListener("click", () => { page = Number.MAX_SAFE_INTEGER; renderRows(); });
$("page-size").addEventListener("change", (e) => {
  pageSize = parseInt(e.target.value, 10) || 100;
  page = 1;
  renderRows();
});

document.querySelectorAll("#rows-table thead th.sortable").forEach((th) => {
  th.addEventListener("click", () => {
    const key = th.dataset.sortKey;
    const type = th.dataset.sortType || "str";
    if (sort?.key === key) {
      sort.dir = sort.dir === "asc" ? "desc" : "asc";
    } else {
      sort = { key, dir: type === "num" ? "desc" : "asc", type };
    }
    page = 1;
    updateSortIndicators();
    renderRows();
  });
});

function updateSortIndicators() {
  document.querySelectorAll("#rows-table thead th.sortable").forEach((th) => {
    th.classList.remove("sort-asc", "sort-desc");
    if (sort && th.dataset.sortKey === sort.key) {
      th.classList.add(sort.dir === "asc" ? "sort-asc" : "sort-desc");
    }
  });
}

async function runPrediction(fetcher) {
  submitBtn.disabled = true;
  sampleBtn.disabled = true;
  downloadBtn.disabled = true;
  setStatus("Scoring…", "info", true);
  try {
    const res = await fetcher();
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail));
    }
    const data = await res.json();
    render(data);
    setStatus(`${data.n_rows.toLocaleString()} rows scored.`, "info");
    downloadBtn.disabled = false;
    downloadBtn.dataset.source = data.source || "";
  } catch (err) {
    setStatus(err.message, "error");
    results.classList.add("hidden");
  } finally {
    submitBtn.disabled = !pickedFile;
    sampleBtn.disabled = false;
  }
}

function setStatus(msg, kind, busy = false) {
  statusText.textContent = msg;
  statusText.className = kind || "";
  spinner.classList.toggle("hidden", !busy);
}

function render(d) {
  results.classList.remove("hidden");

  $("s-rows").textContent = d.n_rows.toLocaleString();
  $("s-benign").textContent = d.summary.benign.toLocaleString();
  $("s-mal").textContent = d.summary.malicious.toLocaleString();
  $("s-pct").textContent = d.summary.malicious_pct + "%";
  $("s-baseline").textContent = `dataset baseline ~${d.summary.baseline_malicious_pct}%`;

  if (d.metrics) {
    metricsCard.classList.remove("hidden");
    renderMetrics(d.metrics);
  } else {
    metricsCard.classList.add("hidden");
  }

  lastRows = d.rows;
  page = 1;
  rowsCap.textContent = d.rows_truncated
    ? `(first ${d.rows.length.toLocaleString()} of ${d.n_rows.toLocaleString()})`
    : `(${d.rows.length.toLocaleString()} rows)`;
  renderRows();
}

function renderMetrics(m) {
  const pct = (x) => (x * 100).toFixed(2) + "%";
  $("m-acc").textContent = pct(m.accuracy);
  $("m-f1").textContent = pct(m.f1);
  $("m-auc").textContent = m.auc == null ? "n/a" : formatAuc(m.auc);
  $("m-prec").textContent = pct(m.precision);
  $("m-rec").textContent = pct(m.recall);

  const cm = m.confusion_matrix;
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const rowBen = cm.tn + cm.fp;
  const rowMal = cm.fn + cm.tp;
  const colBen = cm.tn + cm.fn;
  const colMal = cm.fp + cm.tp;
  const setCell = (n, v, denom) => {
    $(n).textContent = v.toLocaleString();
    const el = $(n + "-pct");
    if (el) el.textContent = denom ? `${((v / denom) * 100).toFixed(1)}%` : "";
  };
  setCell("cm-tn", cm.tn, rowBen);
  setCell("cm-fp", cm.fp, rowBen);
  setCell("cm-fn", cm.fn, rowMal);
  setCell("cm-tp", cm.tp, rowMal);
  $("cm-row-ben").textContent = rowBen.toLocaleString();
  $("cm-row-mal").textContent = rowMal.toLocaleString();
  $("cm-col-ben").textContent = colBen.toLocaleString();
  $("cm-col-mal").textContent = colMal.toLocaleString();
  $("cm-col-all").textContent = total.toLocaleString();

  const curve = $("roc-curve");
  if (m.roc && m.roc.fpr.length) {
    const pts = m.roc.fpr.map((f, i) => `${(f * 100).toFixed(2)},${((1 - m.roc.tpr[i]) * 100).toFixed(2)}`).join(" ");
    curve.setAttribute("points", pts);
    curve.style.display = "";
  } else {
    curve.setAttribute("points", "");
    curve.style.display = "none";
  }
}

function renderRows() {
  let rows = malOnly.checked ? lastRows.filter((r) => r.predicted === "Malicious") : lastRows.slice();
  if (sort) {
    const mul = sort.dir === "asc" ? 1 : -1;
    const isNum = sort.type === "num";
    rows.sort((a, b) => {
      const av = a[sort.key];
      const bv = b[sort.key];
      const aNull = av == null;
      const bNull = bv == null;
      if (aNull && bNull) return 0;
      if (aNull) return 1;
      if (bNull) return -1;
      if (isNum) return (av - bv) * mul;
      return String(av).localeCompare(String(bv)) * mul;
    });
  }

  const total = rows.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  if (page > totalPages) page = totalPages;
  if (page < 1) page = 1;
  const start = (page - 1) * pageSize;
  const pageRows = rows.slice(start, start + pageSize);

  const info = $("page-info");
  if (info) {
    info.textContent = total === 0
      ? "No rows"
      : `Page ${page} of ${totalPages} · rows ${start + 1}–${Math.min(start + pageSize, total)} of ${total.toLocaleString()}`;
  }
  $("page-first").disabled = page <= 1;
  $("page-prev").disabled = page <= 1;
  $("page-next").disabled = page >= totalPages;
  $("page-last").disabled = page >= totalPages;

  const tbody = document.querySelector("#rows-table tbody");
  tbody.innerHTML = "";
  for (const r of pageRows) {
    const p = (r.proba_malicious * 100).toFixed(1);
    const cls = r.predicted === "Malicious" ? "malicious" : "benign";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.row}</td>
      <td><span class="badge ${cls}">${r.predicted}</span></td>
      <td><span class="bar ${cls}"><span style="width:${p}%"></span></span>${p}%</td>
      <td>${fmtTrue(r.true)}</td>
      <td>${numOrDash(r["Destination Port"])}</td>
      <td>${numOrDash(r["Flow Duration"])}</td>
      <td>${numOrDash(r["Total Fwd Packets"])}</td>
      <td>${numOrDash(r["SYN Flag Count"])}</td>
      <td>${numOrDash(r["FIN Flag Count"])}</td>
      <td>${numOrDash(r["RST Flag Count"])}</td>
    `;
    tbody.appendChild(tr);
  }
  if (!pageRows.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="muted small" style="text-align:center;padding:20px;">No rows.</td></tr>';
  }
}

function formatAuc(x) {
  if (x >= 0.9995) return x.toFixed(6);
  return x.toFixed(3);
}

function fmtTrue(v) {
  if (v == null) return '<span class="muted">—</span>';
  return String(v).toUpperCase() === "BENIGN" ? v : `<span class="true-attack">${v}</span>`;
}

function numOrDash(v) {
  if (v == null || Number.isNaN(v)) return '<span class="muted">—</span>';
  return typeof v === "number" ? v.toLocaleString() : v;
}

function triggerDownload(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function buildCsvFromPreview(d) {
  const header = ["row", "predicted", "proba_malicious"];
  if (d.rows[0]?.true != null) header.push("true");
  const lines = [header.join(",")];
  for (const r of d.rows) {
    const row = [r.row, r.predicted, r.proba_malicious];
    if (d.rows[0]?.true != null) row.push(r.true ?? "");
    lines.push(row.join(","));
  }
  return new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
}
