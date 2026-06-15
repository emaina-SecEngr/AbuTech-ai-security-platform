"""
Layer 3 — Knowledge & Enrichment
FastAPI Dashboard App

Serves a live, self-explaining dashboard for the
enrichment layer — one card per enricher explaining
what it adds, the source it draws on, how it enriches
an event, and the value it provides. Built as a
selling point, matching the Layer 1 and Layer 2 style.

ENDPOINTS:
    GET  /                     Live Layer 3 dashboard
    GET  /health              Health check
    GET  /api/layer-info      What Layer 3 does + value
    GET  /api/enrichers       The enrichment catalog
    GET  /api/enrichers/{id}  One enricher's detail

RUN IT:
    uvicorn layer3_knowledge.api.layer3_app:app --reload --port 8003
    Open http://localhost:8003
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from layer3_knowledge.api.enrichment_catalog import (
    LAYER_INFO,
    STATUS_META,
    get_enrichers,
    get_enricher,
    get_categories,
    catalog_stats,
)

app = FastAPI(
    title="AbuTech Layer 3 — Knowledge & Enrichment",
    description=(
        "Maps events to MITRE ATT&CK, enriches with "
        "threat intelligence, and correlates in a "
        "knowledge graph."
    ),
    version="1.0.0",
)


@app.get("/health")
def health():
    """Liveness check."""
    stats = catalog_stats()
    return {
        "status": "healthy",
        "service": "layer3_knowledge",
        "enrichers": stats["total_enrichers"],
        "intel_sources": stats["intel_sources"],
    }


@app.get("/api/layer-info")
def layer_info():
    """What Layer 3 does and the value it adds."""
    return LAYER_INFO


@app.get("/api/enrichers")
def list_enrichers():
    """Return the enrichment catalog grouped by category."""
    return {
        "catalog_stats": catalog_stats(),
        "status_meta": STATUS_META,
        "categories": get_categories(),
    }


@app.get("/api/enrichers/{enricher_id}")
def enricher_detail(enricher_id: str):
    """Return one enricher's detail."""
    enricher = get_enricher(enricher_id)
    if not enricher:
        return JSONResponse(
            status_code=404,
            content={"error": "enricher not found"},
        )
    return enricher


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the live Layer 3 dashboard."""
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AbuTech — Layer 3 Knowledge & Enrichment</title>
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
    max-width:740px; margin-left:auto; margin-right:auto;
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
  .card-head .name { font-size:16px; font-weight:700; }
  .src {
    font-family:'Courier New',monospace; font-size:11px;
    color:var(--purple); margin-bottom:12px;
  }
  .row { margin-bottom:10px; }
  .row .k {
    font-size:10px; color:var(--dim); letter-spacing:1px;
    text-transform:uppercase; margin-bottom:2px;
  }
  .row .v { font-size:13px; color:var(--text); }
  .row .v.val { color:var(--accent); }
  .card-foot {
    display:flex; align-items:center; justify-content:flex-end;
    margin-top:12px; padding-top:12px;
    border-top:1px solid var(--border);
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
    <h1>Layer 3 — Knowledge &amp; Enrichment</h1>
    <div class="tag" id="tagline"></div>
  </div>

  <div class="info" id="info"></div>
  <div class="stats" id="stats"></div>
  <div id="catalog"></div>

  <div class="foot">A score tells you something is risky · enrichment tells you why, who, and what to do.</div>
</div>

<script>
async function load() {
  const [infoRes, enrRes] = await Promise.all([
    fetch('/api/layer-info'),
    fetch('/api/enrichers')
  ]);
  const info = await infoRes.json();
  const data = await enrRes.json();

  document.getElementById('tagline').textContent = info.tagline;

  document.getElementById('info').innerHTML = `
    <div class="info-card"><h3>What it does</h3><p>${info.what_it_does}</p></div>
    <div class="info-card"><h3>Why it matters</h3><p>${info.why_it_matters}</p></div>
    <div class="info-card"><h3>How it works</h3><p>${info.how_it_works}</p></div>
  `;

  const cs = data.catalog_stats;
  document.getElementById('stats').innerHTML = `
    <div class="stat"><div class="num">${cs.total_enrichers}</div><div class="lbl">Enrichers</div></div>
    <div class="stat"><div class="num">${cs.intel_sources}</div><div class="lbl">Intel sources</div></div>
  `;

  const cat = document.getElementById('catalog');
  cat.innerHTML = data.categories.map(g => `
    <div class="cat-title">${g.category} · ${g.enrichers.length}</div>
    <div class="grid">
      ${g.enrichers.map(e => {
        const sm = data.status_meta[e.status] || {label:e.status,color:'#888'};
        return `
        <div class="card">
          <div class="card-head">
            <i class="ti ${e.icon}"></i>
            <span class="name">${e.name}</span>
          </div>
          <div class="src">${e.source}</div>
          <div class="row"><div class="k">Adds</div><div class="v">${e.adds}</div></div>
          <div class="row"><div class="k">How</div><div class="v">${e.how}</div></div>
          <div class="row"><div class="k">Value</div><div class="v val">${e.value}</div></div>
          <div class="card-foot">
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