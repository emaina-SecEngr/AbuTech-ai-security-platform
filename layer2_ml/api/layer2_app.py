"""
Layer 2 — ML Detection
FastAPI Dashboard App

Serves a live, self-explaining dashboard for the
detection layer — one card per model explaining its
algorithm, tech stack, learning type, what it detects,
how it scores, and what it catches that rules miss.
Built as a selling point: the page explains the
platform's detection sophistication so it can be
walked through, not recited from memory.

ENDPOINTS:
    GET  /                  Live Layer 2 dashboard (HTML)
    GET  /health           Health check
    GET  /api/layer-info   What Layer 2 does + value
    GET  /api/models       The model catalog
    GET  /api/models/{id}  One model's detail

RUN IT:
    uvicorn layer2_ml.api.layer2_app:app --reload --port 8002
    Open http://localhost:8002
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from layer2_ml.api.model_catalog import (
    LAYER_INFO,
    STATUS_META,
    LEARNING_META,
    get_models,
    get_model,
    get_categories,
    catalog_stats,
)

app = FastAPI(
    title="AbuTech Layer 2 — ML Detection",
    description=(
        "Specialized ML models score every event for "
        "threat, combined by an ensemble."
    ),
    version="1.0.0",
)


@app.get("/health")
def health():
    """Liveness check."""
    stats = catalog_stats()
    return {
        "status": "healthy",
        "service": "layer2_ml",
        "detection_models": stats["total_models"],
    }


@app.get("/api/layer-info")
def layer_info():
    """What Layer 2 does and the value it adds."""
    return LAYER_INFO


@app.get("/api/models")
def list_models():
    """Return the full model catalog grouped by category."""
    return {
        "catalog_stats": catalog_stats(),
        "status_meta": STATUS_META,
        "learning_meta": LEARNING_META,
        "categories": get_categories(),
    }


@app.get("/api/models/{model_id}")
def model_detail(model_id: str):
    """Return one model's detail."""
    model = get_model(model_id)
    if not model:
        return JSONResponse(
            status_code=404,
            content={"error": "model not found"},
        )
    return model


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the live Layer 2 dashboard."""
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AbuTech — Layer 2 ML Detection</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.1.0/dist/tabler-icons.min.css">
<style>
  :root {
    --bg:#0a0e14; --panel:#111720; --panel2:#18202c;
    --border:#1f2b3a; --text:#e6edf3; --dim:#7d8da3;
    --accent:#00e5a0; --purple:#7C5CDB;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background:var(--bg); color:var(--text);
    font-family:'Segoe UI',system-ui,sans-serif;
    padding:28px 20px; line-height:1.55;
  }
  .wrap { max-width:1150px; margin:0 auto; }
  .head { text-align:center; margin-bottom:24px; }
  .logo {
    font-family:'Courier New',monospace; font-size:12px;
    letter-spacing:4px; color:var(--accent);
    text-transform:uppercase;
  }
  .head h1 { font-size:30px; font-weight:800; margin-top:8px; }
  .head .tag {
    color:var(--dim); margin-top:8px; font-size:15px;
    max-width:720px; margin-left:auto; margin-right:auto;
  }
  .info {
    display:grid; grid-template-columns:repeat(3,1fr);
    gap:14px; margin:24px 0;
  }
  .info-card {
    background:var(--panel); border:1px solid var(--border);
    border-radius:12px; padding:16px;
  }
  .info-card h3 {
    font-size:12px; color:var(--accent); letter-spacing:1px;
    text-transform:uppercase; margin-bottom:8px;
  }
  .info-card p { font-size:13px; color:var(--dim); }
  .stats {
    display:flex; gap:12px; justify-content:center;
    margin:20px 0; flex-wrap:wrap;
  }
  .stat {
    background:var(--panel); border:1px solid var(--border);
    border-radius:10px; padding:10px 18px; text-align:center;
  }
  .stat .num { font-size:22px; font-weight:800; color:var(--accent); }
  .stat .lbl {
    font-size:10px; color:var(--dim); letter-spacing:1px;
    text-transform:uppercase;
  }
  .cat-title {
    font-size:13px; font-weight:600; color:var(--dim);
    margin:22px 0 12px; padding-bottom:6px;
    border-bottom:1px solid var(--border);
  }
  .grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
    gap:14px;
  }
  .card {
    background:var(--panel); border:1px solid var(--border);
    border-radius:14px; padding:18px; transition:border-color .15s;
  }
  .card:hover { border-color:var(--accent); }
  .card-head {
    display:flex; align-items:center; gap:11px; margin-bottom:6px;
  }
  .card-head i { font-size:24px; color:var(--accent); }
  .card-head .name { font-size:16px; font-weight:700; flex:1; }
  .learn {
    font-size:9px; font-weight:700; padding:2px 7px;
    border-radius:6px; color:#0a0e14; white-space:nowrap;
  }
  .algo {
    font-family:'Courier New',monospace; font-size:11px;
    color:var(--purple); margin-bottom:9px;
  }
  .stack {
    display:flex; flex-wrap:wrap; gap:5px; margin-bottom:12px;
  }
  .pill {
    font-size:10px; color:var(--dim);
    background:var(--panel2); border:1px solid var(--border);
    border-radius:6px; padding:2px 8px;
    font-family:'Courier New',monospace;
  }
  .row { margin-bottom:10px; }
  .row .k {
    font-size:10px; color:var(--dim); letter-spacing:1px;
    text-transform:uppercase; margin-bottom:2px;
  }
  .row .v { font-size:13px; color:var(--text); }
  .row .v.miss { color:var(--accent); }
  .card-foot {
    display:flex; align-items:center; justify-content:space-between;
    margin-top:12px; padding-top:12px;
    border-top:1px solid var(--border);
  }
  .mitre {
    font-family:'Courier New',monospace; font-size:11px;
    color:var(--dim);
  }
  .badge {
    font-size:10px; font-weight:700; padding:3px 9px;
    border-radius:7px; color:#06241a; white-space:nowrap;
  }
  .foot {
    text-align:center; margin-top:30px; font-size:12px;
    color:var(--dim); font-family:'Courier New',monospace;
  }
  @media(max-width:760px){ .info{grid-template-columns:1fr;} }
</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <div class="logo">AbuTech AI Security Platform</div>
    <h1>Layer 2 — ML Detection</h1>
    <div class="tag" id="tagline"></div>
  </div>

  <div class="info" id="info"></div>
  <div class="stats" id="stats"></div>
  <div id="catalog"></div>

  <div class="foot">Every score is explainable and maps to MITRE ATT&CK · combined by the ensemble into one risk score.</div>
</div>

<script>
async function load() {
  const [infoRes, modelsRes] = await Promise.all([
    fetch('/api/layer-info'),
    fetch('/api/models')
  ]);
  const info = await infoRes.json();
  const data = await modelsRes.json();

  document.getElementById('tagline').textContent = info.tagline;

  document.getElementById('info').innerHTML = `
    <div class="info-card"><h3>What it does</h3><p>${info.what_it_does}</p></div>
    <div class="info-card"><h3>Why it matters</h3><p>${info.why_it_matters}</p></div>
    <div class="info-card"><h3>How scores combine</h3><p>${info.how_scores_combine}</p></div>
  `;

  const cs = data.catalog_stats;
  let statHtml = `<div class="stat"><div class="num">${cs.total_models}</div><div class="lbl">Models</div></div>`;
  for (const [lt, count] of Object.entries(cs.by_learning || {})) {
    const meta = data.learning_meta[lt];
    if (!meta) continue;
    statHtml += `<div class="stat"><div class="num">${count}</div><div class="lbl">${meta.label}</div></div>`;
  }
  document.getElementById('stats').innerHTML = statHtml;

  const cat = document.getElementById('catalog');
  cat.innerHTML = data.categories.map(g => `
    <div class="cat-title">${g.category} · ${g.models.length}</div>
    <div class="grid">
      ${g.models.map(m => {
        const sm = data.status_meta[m.status] || {label:m.status,color:'#888'};
        const lm = data.learning_meta[m.learning_type] || null;
        const learnBadge = lm ? `<span class="learn" style="background:${lm.color}">${lm.label}</span>` : '';
        return `
        <div class="card">
          <div class="card-head">
            <i class="ti ${m.icon}"></i>
            <span class="name">${m.name}</span>
            ${learnBadge}
          </div>
          <div class="algo">${m.algorithm}</div>
          <div class="stack">${(m.tech_stack||[]).map(t => `<span class="pill">${t}</span>`).join('')}</div>
          <div class="row"><div class="k">Detects</div><div class="v">${m.detects}</div></div>
          <div class="row"><div class="k">How it scores</div><div class="v">${m.how_it_scores}</div></div>
          <div class="row"><div class="k">What rules miss</div><div class="v miss">${m.rules_miss}</div></div>
          <div class="card-foot">
            <span class="mitre">${m.mitre}</span>
            <span class="badge" style="background:${sm.color}">${sm.label}</span>
          </div>
        </div>`;
      }).join('')}
    </div>
  `).join('');
}
load();
</script>
</body>
</html>"""