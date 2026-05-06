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
const modelSelect = $("model-select");
const sampleSelect = $("sample-select");
const compareBtn = $("compare-btn");
const compareCard = $("compare-results");
const compareSource = $("compare-source");

const DEFAULT_THRESHOLD = 0.5;

let pickedFile = null;
let lastRows = [];
let lastMetrics = null;
let lastTotals = null; // {nBenign, nMal, total} cached from server-side CM
let lastBaseline = null;
let lastExtraCols = [];
let lastSource = null;
let openRow = null; // currently-expanded row number, or null
let threshold = DEFAULT_THRESHOLD;
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
    if (j.status === "no_models") {
      $("predictor-tag").textContent = "no models loaded";
      return;
    }
    const p = j.predictor;
    const date = p.trained_at ? ` · ${p.trained_at}` : "";
    $("predictor-tag").textContent = `${p.name} v${p.version}${date}`;
  })
  .catch(() => ($("predictor-tag").textContent = "unreachable"));

fetch("/api/models")
  .then((r) => r.json())
  .then((j) => {
    modelSelect.innerHTML = "";
    if (!j.models || j.models.length === 0) {
      modelSelect.innerHTML = '<option value="">(no models found)</option>';
      modelSelect.disabled = true;
      return;
    }
    for (const m of j.models) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      if (m.id === j.default_id) opt.selected = true;
      modelSelect.appendChild(opt);
    }
    refreshCompareEnabled();
  })
  .catch(() => {
    modelSelect.innerHTML = '<option value="">(model list unavailable)</option>';
  });

fetch("/api/samples")
  .then((r) => r.json())
  .then((j) => {
    sampleSelect.innerHTML = "";
    if (!j.samples || j.samples.length === 0) {
      sampleSelect.innerHTML = '<option value="">(none bundled)</option>';
      sampleSelect.disabled = true;
      sampleBtn.disabled = true;
      return;
    }
    for (const s of j.samples) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = s.label;
      if (s.id === j.default_id) opt.selected = true;
      sampleSelect.appendChild(opt);
    }
    refreshCompareEnabled();
  })
  .catch(() => {
    sampleSelect.innerHTML = '<option value="">(unavailable)</option>';
    sampleSelect.disabled = true;
  });

fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));
sampleSelect.addEventListener("change", refreshCompareEnabled);

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
    refreshCompareEnabled();
    return;
  }
  if (!pickedFile.name.toLowerCase().endsWith(".csv")) {
    setStatus(`"${pickedFile.name}" is not a .csv file.`, "error");
    pickedFile = null;
    submitBtn.disabled = true;
    fileLabel.textContent = "Choose a .csv file or drop it here";
    fileInput.value = "";
    refreshCompareEnabled();
    return;
  }
  fileLabel.textContent = `${pickedFile.name} (${(pickedFile.size / 1024 / 1024).toFixed(2)} MB)`;
  submitBtn.disabled = false;
  refreshCompareEnabled();
  setStatus("", "");
}

function refreshCompareEnabled() {
  // Compare needs either an upload or at least one bundled sample to score.
  const haveSample = sampleSelect && sampleSelect.value && !sampleSelect.disabled;
  const haveModels = modelSelect && modelSelect.options.length > 0 && !modelSelect.disabled;
  compareBtn.disabled = !haveModels || (!pickedFile && !haveSample);
}

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!pickedFile) return;
  await runPrediction(() => {
    const fd = new FormData();
    fd.append("file", pickedFile);
    if (modelSelect.value) fd.append("model_id", modelSelect.value);
    return fetch("/api/predict", { method: "POST", body: fd });
  });
});

sampleBtn.addEventListener("click", async () => {
  pickedFile = null;
  fileInput.value = "";
  fileLabel.textContent = "Choose a .csv file or drop it here";
  submitBtn.disabled = true;
  const params = new URLSearchParams();
  if (modelSelect.value) params.set("model_id", modelSelect.value);
  if (sampleSelect.value) params.set("sample_id", sampleSelect.value);
  const qs = params.toString();
  await runPrediction(() => fetch("/api/sample" + (qs ? `?${qs}` : "")));
});

compareBtn.addEventListener("click", async () => {
  const ids = [...modelSelect.options].map((o) => o.value).filter(Boolean);
  if (!ids.length) return;
  compareBtn.disabled = true;
  submitBtn.disabled = true;
  sampleBtn.disabled = true;
  setStatus("Scoring all models…", "info", true);
  try {
    const fd = new FormData();
    fd.append("model_ids", ids.join(","));
    if (pickedFile) {
      fd.append("file", pickedFile);
    } else if (sampleSelect.value) {
      fd.append("sample_id", sampleSelect.value);
    } else {
      throw new Error("pick a file or a bundled sample first");
    }
    const res = await fetch("/api/predict_compare", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail));
    }
    const data = await res.json();
    renderCompare(data);
    setStatus(`${data.results.length} models scored on ${data.source}.`, "info");
  } catch (err) {
    setStatus(`Compare failed: ${err.message}`, "error");
    compareCard.classList.add("hidden");
  } finally {
    refreshCompareEnabled();
    submitBtn.disabled = !pickedFile;
    sampleBtn.disabled = false;
  }
});

function renderCompare(data) {
  compareCard.classList.remove("hidden");
  compareSource.textContent = data.source ? `input: ${data.source}` : "";
  const tbody = compareCard.querySelector("tbody");
  tbody.innerHTML = "";

  const haveTruth = data.results.some((r) => r.metrics);
  $("compare-no-truth-note").style.display = haveTruth ? "none" : "";

  // Pull the comparable numbers per row first so we can compute the
  // best-of-column up front. FPR is "lower is better"; everything else is
  // "higher is better"; pred_mal is informational, not a quality metric,
  // so we don't crown a winner there.
  const stats = data.results.map((r) => {
    if (r.error || !r.metrics) return null;
    const cm = r.metrics.confusion_matrix;
    return {
      accuracy: r.metrics.accuracy,
      precision: r.metrics.precision,
      recall: r.metrics.recall,
      f1: r.metrics.f1,
      fpr: cm.fp / Math.max(1, cm.fp + cm.tn),
      auc: r.metrics.auc,
    };
  });

  const bestPerCol = {};
  if (haveTruth) {
    const cols = [
      ["accuracy", "max"], ["precision", "max"], ["recall", "max"],
      ["f1", "max"], ["fpr", "min"], ["auc", "max"],
    ];
    // Float compare with a small epsilon so tied winners both get the highlight
    // (e.g. two models that round to the same percentage shouldn't fight over it).
    const EPS = 1e-9;
    for (const [col, dir] of cols) {
      const vals = stats.map((s) => (s ? s[col] : null)).filter((v) => v != null);
      if (!vals.length) continue;
      bestPerCol[col] = dir === "max" ? Math.max(...vals) : Math.min(...vals);
      bestPerCol[col + "_eps"] = EPS;
    }
  }

  const isBest = (col, val) => {
    if (val == null || bestPerCol[col] == null) return false;
    return Math.abs(val - bestPerCol[col]) <= bestPerCol[col + "_eps"];
  };

  data.results.forEach((r, i) => {
    const tr = document.createElement("tr");
    if (r.error) {
      tr.classList.add("errored");
      tr.innerHTML = `<td>${escapeHtml(r.model_id)}</td><td colspan="7">error: ${escapeHtml(r.error)}</td>`;
      tbody.appendChild(tr);
      return;
    }
    const s = stats[i];
    const cell = (col, text) => `<td class="num${isBest(col, s?.[col]) ? " best" : ""}">${text}</td>`;
    const m = r.metrics;
    tr.innerHTML = `
      <td>${escapeHtml(r.model_id)}</td>
      ${cell("accuracy",  m ? (m.accuracy  * 100).toFixed(2) + "%" : "—")}
      ${cell("precision", m ? (m.precision * 100).toFixed(2) + "%" : "—")}
      ${cell("recall",    m ? (m.recall    * 100).toFixed(2) + "%" : "—")}
      ${cell("f1",        m ? (m.f1        * 100).toFixed(2) + "%" : "—")}
      ${cell("fpr",       s ? (s.fpr       * 100).toFixed(2) + "%" : "—")}
      ${cell("auc",       m && m.auc != null ? formatAuc(m.auc) : "—")}
      <td class="num">${r.n_malicious.toLocaleString()} / ${r.n_rows.toLocaleString()}</td>
    `;
    tbody.appendChild(tr);
  });
}

downloadBtn.addEventListener("click", async () => {
  if (!pickedFile && !downloadBtn.dataset.source) return;
  downloadBtn.disabled = true;
  setStatus("Preparing download…", "info", true);
  try {
    let res;
    if (pickedFile) {
      const fd = new FormData();
      fd.append("file", pickedFile);
      if (modelSelect.value) fd.append("model_id", modelSelect.value);
      res = await fetch("/api/predict.csv", { method: "POST", body: fd });
    } else {
      const params = new URLSearchParams();
      if (modelSelect.value) params.set("model_id", modelSelect.value);
      if (sampleSelect.value) params.set("sample_id", sampleSelect.value);
      const qs = params.toString();
      const sample = await fetch("/api/sample" + (qs ? `?${qs}` : "")).then((r) => r.json());
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

const thrSlider = $("thr-slider");
const thrValue = $("thr-value");
const thrDefault = $("thr-default");
const thrReset = $("thr-reset");
thrSlider.addEventListener("input", (e) => {
  threshold = parseFloat(e.target.value);
  applyThreshold();
});
thrReset.addEventListener("click", () => {
  threshold = DEFAULT_THRESHOLD;
  thrSlider.value = String(DEFAULT_THRESHOLD);
  applyThreshold();
});

$("metrics-csv-btn").addEventListener("click", () => {
  if (!lastMetrics) return;
  const blob = new Blob([buildMetricsCsv()], { type: "text/csv" });
  triggerDownload(blob, "metrics.csv");
});

function buildMetricsCsv() {
  const m = lastMetrics;
  const lines = ["section,key,value"];
  const head = $("predictor-tag").textContent || "";
  lines.push(`run,predictor,${csvCell(head)}`);
  if (lastSource) lines.push(`run,source,${csvCell(lastSource)}`);
  lines.push(`run,threshold,${threshold.toFixed(4)}`);

  // Headline metrics reflect the live threshold (slider) when truth is present.
  const cell = lastTotals ? rocCellAt(m, threshold) : null;
  let acc, prec, rec, f1, tp, fp, fn, tn;
  if (cell && lastTotals) {
    tp = Math.round(cell.tpr * lastTotals.nMal);
    fn = lastTotals.nMal - tp;
    fp = Math.round(cell.fpr * lastTotals.nBenign);
    tn = lastTotals.nBenign - fp;
    acc = lastTotals.total ? (tp + tn) / lastTotals.total : 0;
    prec = (tp + fp) ? tp / (tp + fp) : 0;
    rec = lastTotals.nMal ? tp / lastTotals.nMal : 0;
    f1 = (prec + rec) ? (2 * prec * rec) / (prec + rec) : 0;
  } else {
    acc = m.accuracy; prec = m.precision; rec = m.recall; f1 = m.f1;
    const cm = m.confusion_matrix;
    tn = cm.tn; fp = cm.fp; fn = cm.fn; tp = cm.tp;
  }
  lines.push(`metrics,accuracy,${acc.toFixed(6)}`);
  lines.push(`metrics,precision,${prec.toFixed(6)}`);
  lines.push(`metrics,recall,${rec.toFixed(6)}`);
  lines.push(`metrics,f1,${f1.toFixed(6)}`);
  if (m.auc != null) lines.push(`metrics,auc,${m.auc.toFixed(6)}`);
  lines.push(`confusion_matrix,tn,${tn}`);
  lines.push(`confusion_matrix,fp,${fp}`);
  lines.push(`confusion_matrix,fn,${fn}`);
  lines.push(`confusion_matrix,tp,${tp}`);

  if (m.per_attack && m.per_attack.length) {
    lines.push("");
    lines.push("per_attack,attack,support,tp,fn,recall");
    for (const a of m.per_attack) {
      lines.push(`per_attack,${csvCell(a.label)},${a.support},${a.tp},${a.fn},${a.recall.toFixed(6)}`);
    }
  }
  return lines.join("\n") + "\n";
}

function csvCell(s) {
  const v = String(s ?? "");
  return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
}

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
    const n = Number(data.n_rows) || 0;
    setStatus(`${n.toLocaleString()} rows scored.`, "info");
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

  lastRows = d.rows;
  lastBaseline = d.feature_baseline || {};
  lastExtraCols = d.row_extra_features || [];
  lastSource = d.source || (pickedFile ? pickedFile.name : null);
  openRow = null;
  page = 1;
  rowsCap.textContent = d.rows_truncated
    ? `(first ${d.rows.length.toLocaleString()} of ${d.n_rows.toLocaleString()})`
    : `(${d.rows.length.toLocaleString()} rows)`;

  lastMetrics = d.metrics || null;
  lastTotals = null;
  if (lastMetrics) {
    const cm = lastMetrics.confusion_matrix;
    lastTotals = {
      nBenign: cm.tn + cm.fp,
      nMal: cm.fn + cm.tp,
      total: cm.tn + cm.fp + cm.fn + cm.tp,
    };
    metricsCard.classList.remove("hidden");
    threshold = DEFAULT_THRESHOLD;
    thrSlider.value = String(DEFAULT_THRESHOLD);
    renderRocStatic(lastMetrics);
    renderPerAttack(lastMetrics.per_attack);
    applyThreshold(); // also renders rows
  } else {
    metricsCard.classList.add("hidden");
    renderRows();
  }
}

function renderRocStatic(m) {
  const curve = $("roc-curve");
  const aucText = $("roc-auc-text");
  if (m.roc && m.roc.fpr.length) {
    const pts = m.roc.fpr.map((f, i) => `${(f * 100).toFixed(2)},${((1 - m.roc.tpr[i]) * 100).toFixed(2)}`).join(" ");
    curve.setAttribute("points", pts);
    curve.style.display = "";
  } else {
    curve.setAttribute("points", "");
    curve.style.display = "none";
  }
  aucText.textContent = (m.auc == null || Number.isNaN(m.auc)) ? "" : `AUC = ${formatAuc(m.auc)}`;
}

// Find the ROC sample whose threshold is closest to t, fall back to nearest
// (fpr, tpr) bracket. roc_curve in sklearn yields thresholds in DESCENDING
// order, so a binary scan would be overkill -- linear is fine for ~120 pts.
function rocCellAt(m, t) {
  if (!m.roc || !m.roc.fpr.length) return null;
  const { fpr, tpr, thr } = m.roc;
  let best = 0;
  let bestDelta = Infinity;
  for (let i = 0; i < thr.length; i++) {
    const d = Math.abs(thr[i] - t);
    if (d < bestDelta) { bestDelta = d; best = i; }
  }
  return { fpr: fpr[best], tpr: tpr[best], thr: thr[best] };
}

function applyThreshold() {
  thrValue.textContent = threshold.toFixed(2);
  thrDefault.classList.toggle("hidden", Math.abs(threshold - DEFAULT_THRESHOLD) > 1e-9);

  if (lastMetrics && lastTotals) {
    const cell = rocCellAt(lastMetrics, threshold) || { fpr: 0, tpr: 0 };
    // tpr/fpr are rates over the *full* dataset; convert back to counts.
    const tp = Math.round(cell.tpr * lastTotals.nMal);
    const fn = lastTotals.nMal - tp;
    const fp = Math.round(cell.fpr * lastTotals.nBenign);
    const tn = lastTotals.nBenign - fp;
    const acc = lastTotals.total ? (tp + tn) / lastTotals.total : 0;
    const prec = (tp + fp) ? tp / (tp + fp) : 0;
    const rec = lastTotals.nMal ? tp / lastTotals.nMal : 0;
    const f1 = (prec + rec) ? (2 * prec * rec) / (prec + rec) : 0;

    const pct = (x) => (x * 100).toFixed(2) + "%";
    $("m-acc").textContent = pct(acc);
    $("m-f1").textContent = pct(f1);
    $("m-auc").textContent = (lastMetrics.auc == null || Number.isNaN(lastMetrics.auc))
      ? "n/a" : formatAuc(lastMetrics.auc);
    $("m-prec").textContent = pct(prec);
    $("m-rec").textContent = pct(rec);

    const rowBen = tn + fp;
    const rowMal = fn + tp;
    const colBen = tn + fn;
    const colMal = fp + tp;
    const setCell = (n, v, denom) => {
      $(n).textContent = v.toLocaleString();
      const el = $(n + "-pct");
      if (el) el.textContent = denom ? `${((v / denom) * 100).toFixed(1)}%` : "";
    };
    setCell("cm-tn", tn, rowBen);
    setCell("cm-fp", fp, rowBen);
    setCell("cm-fn", fn, rowMal);
    setCell("cm-tp", tp, rowMal);
    $("cm-row-ben").textContent = rowBen.toLocaleString();
    $("cm-row-mal").textContent = rowMal.toLocaleString();
    $("cm-col-ben").textContent = colBen.toLocaleString();
    $("cm-col-mal").textContent = colMal.toLocaleString();
    $("cm-col-all").textContent = lastTotals.total.toLocaleString();

    // Move the dot on the ROC. Coords match the polyline's coordinate space.
    const dot = $("roc-dot");
    dot.setAttribute("cx", String(cell.fpr * 100));
    dot.setAttribute("cy", String((1 - cell.tpr) * 100));
  }

  renderRows();
}

function renderPerAttack(rows) {
  const wrap = $("per-attack-wrap");
  const body = $("per-attack-body");
  if (!rows || !rows.length) {
    wrap.classList.add("hidden");
    body.innerHTML = "";
    return;
  }
  wrap.classList.remove("hidden");
  body.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    if (r.recall < 0.8) tr.classList.add("weak");
    tr.innerHTML = `
      <td>${escapeHtml(r.label)}</td>
      <td class="num">${r.support.toLocaleString()}</td>
      <td class="num">${r.tp.toLocaleString()}</td>
      <td class="num">${r.fn.toLocaleString()}</td>
      <td class="num">${(r.recall * 100).toFixed(1)}%</td>
    `;
    body.appendChild(tr);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function predLabel(r) {
  // Use the live threshold so the rows table tracks the slider too. The
  // server-side `predicted` field is left intact -- it's the 0.5 baseline.
  return r.proba_malicious >= threshold ? "Malicious" : "Benign";
}

function renderRows() {
  let rows = malOnly.checked
    ? lastRows.filter((r) => predLabel(r) === "Malicious")
    : lastRows.slice();
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
    const label = predLabel(r);
    const cls = label === "Malicious" ? "malicious" : "benign";
    const isOpen = openRow === r.row;
    const tr = document.createElement("tr");
    tr.className = "data-row" + (isOpen ? " open" : "");
    tr.dataset.row = String(r.row);
    tr.innerHTML = `
      <td><span class="row-toggle">${isOpen ? "▾" : "▸"}</span>${r.row}</td>
      <td><span class="badge ${cls}">${label}</span></td>
      <td><span class="bar ${cls}"><span style="width:${p}%"></span></span>${p}%</td>
      <td>${fmtTrue(r.true)}</td>
      <td>${numOrDash(r["Destination Port"])}</td>
      <td>${numOrDash(r["Flow Duration"])}</td>
      <td>${numOrDash(r["Total Fwd Packets"])}</td>
      <td>${numOrDash(r["SYN Flag Count"])}</td>
      <td>${numOrDash(r["FIN Flag Count"])}</td>
      <td>${numOrDash(r["RST Flag Count"])}</td>
    `;
    tr.addEventListener("click", () => toggleRow(r.row));
    tbody.appendChild(tr);
    if (isOpen) tbody.appendChild(buildDetailRow(r));
  }
  if (!pageRows.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="muted small" style="text-align:center;padding:20px;">No rows.</td></tr>';
  }
}

function toggleRow(rowId) {
  openRow = openRow === rowId ? null : rowId;
  renderRows();
}

function buildDetailRow(r) {
  const tr = document.createElement("tr");
  tr.className = "detail-row";
  const td = document.createElement("td");
  td.colSpan = 10;
  td.innerHTML = renderDetail(r);
  tr.appendChild(td);
  return tr;
}

// Render the click-expand panel: row's value vs the upload's median per
// feature. We flag any value that falls outside [p25, p75] so the viewer's
// eye lands on what's atypical relative to *this file*. No global stats,
// no ML attribution -- just an honest "here's what stands out."
function renderDetail(r) {
  const cols = lastExtraCols.length ? lastExtraCols : Object.keys(r).filter((k) => !["row", "predicted", "proba_malicious", "true"].includes(k));
  const cells = cols.map((c) => {
    const v = r[c];
    const base = lastBaseline?.[c];
    if (v == null) return "";
    const num = Number(v);
    let badge = "";
    if (base && Number.isFinite(num)) {
      if (num > base.p75) badge = '<span class="dev-badge hi">↑</span>';
      else if (num < base.p25) badge = '<span class="dev-badge lo">↓</span>';
    }
    const median = base ? `<span class="muted small">med ${fmtNum(base.median)}</span>` : "";
    return `
      <div class="detail-cell">
        <div class="detail-k">${escapeHtml(c)}</div>
        <div class="detail-v">${badge}${fmtNum(num)}</div>
        ${median}
      </div>
    `;
  }).join("");

  const truth = r.true == null
    ? ""
    : `<div class="detail-meta">True label: <strong>${escapeHtml(String(r.true))}</strong></div>`;

  return `
    <div class="detail-panel">
      <div class="detail-meta">
        Row ${r.row} · server-side prediction at threshold 0.5: <strong>${escapeHtml(r.predicted)}</strong>
        · P(Malicious) = ${(r.proba_malicious * 100).toFixed(2)}%
      </div>
      ${truth}
      <div class="detail-grid">${cells}</div>
      <p class="muted small">↑ above 75th pct of this file · ↓ below 25th pct.</p>
    </div>
  `;
}

function fmtNum(v) {
  if (v == null || Number.isNaN(v)) return "—";
  if (typeof v !== "number") return String(v);
  if (!Number.isFinite(v)) return v > 0 ? "∞" : "-∞";
  if (Math.abs(v) >= 1e6 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(2);
  return v.toLocaleString(undefined, { maximumFractionDigits: 3 });
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
