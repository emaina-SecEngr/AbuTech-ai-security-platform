"""
Layer 1 — Data Ingestion
FastAPI Ingestion App

The network-facing front end for Layer 1. This is the
"how do you actually ingest events" answer: real HTTP
endpoints that receive events and drive them through
the ingestion pipeline you built.

ENDPOINTS:
    GET  /                     Live dashboard (HTML)
    GET  /health               Health check
    GET  /api/sources          The 28-source catalog
    GET  /api/sources/{name}   One source's detail
    POST /api/ingest/{source}  Ingest one event (PUSH)
                               source is known from path
    POST /api/ingest           Ingest, source inferred
    POST /api/ingest/batch     Ingest a list of events
    GET  /api/stats            Live pipeline statistics

THE PUSH PATTERN (real production shape):
    CrowdStrike's webhook is configured to POST to
        /api/ingest/crowdstrike
    The path segment is the source hint. The handler
    passes the body straight into the pipeline.

RUN IT:
    uvicorn layer1_ingestion.api.ingestion_app:app --reload
    Open http://localhost:8000
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from layer1_ingestion.pipeline.ingestion_pipeline\
    import IngestionPipeline
from layer1_ingestion.api.source_catalog import (
    get_catalog,
    get_categories,
    get_source,
    catalog_stats,
    INGESTION_METHODS,
)

app = FastAPI(
    title="AbuTech Layer 1 — Ingestion",
    description=(
        "Receives security events from any source and "
        "routes them through the ingestion pipeline."
    ),
    version="1.0.0",
)

# One shared pipeline instance for the app lifetime.
# It accumulates live statistics across requests.
pipeline = IngestionPipeline()


# ============================================================
# HEALTH + CATALOG
# ============================================================

@app.get("/health")
def health():
    """Liveness check."""
    return {
        "status": "healthy",
        "service": "layer1_ingestion",
        "supported_sources": len(
            pipeline.supported_sources()
        ),
    }


@app.get("/api/sources")
def list_sources():
    """Return the full source catalog grouped by category."""
    return {
        "catalog_stats": catalog_stats(),
        "methods": INGESTION_METHODS,
        "categories": get_categories(),
    }


@app.get("/api/sources/{source_name}")
def source_detail(source_name: str):
    """Return one source's catalog entry."""
    entry = get_source(source_name)
    if not entry:
        return JSONResponse(
            status_code=404,
            content={"error": "source not found"},
        )
    routable = (
        source_name in pipeline.supported_sources()
    )
    return {**entry, "routable_today": routable}


# ============================================================
# INGESTION
# ============================================================

# ============================================================
# INGESTION
# ============================================================
@app.post("/api/ingest/batch")
async def ingest_batch(request: Request):
    """
    Ingest a LIST of events (PULL path).

    Defined before /api/ingest/{source} so the literal
    'batch' path wins over the {source} wildcard.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid JSON body"},
        )

    if not isinstance(body, list):
        return JSONResponse(
            status_code=400,
            content={
                "error": "body must be a JSON array"
            },
        )

    results = pipeline.ingest_batch(body)
    return {
        "accepted": len(results),
        "submitted": len(body),
        "dropped": len(body) - len(results),
    }


@app.post("/api/ingest/{source}")
async def ingest_with_source(
    source: str, request: Request
):
    """
    Ingest one event for a KNOWN source (PUSH path).
    The {source} path segment is the source hint -
    this is how a webhook from CrowdStrike or Okta
    delivers events to us.
    """
    # Reject sources the pipeline cannot route
    if source not in pipeline.supported_sources():
        return JSONResponse(
            status_code=422,
            content={
                "accepted": False,
                "source": source,
                "reason": "unsupported source",
            },
        )

    try:
        raw_event = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid JSON body"},
        )
    normalized = pipeline.ingest(
        raw_event, source=source
    )

    if normalized is None:
        return JSONResponse(
            status_code=422,
            content={
                "accepted": False,
                "source": source,
                "reason": (
                    "event could not be normalized "
                    "for this source"
                ),
            },
        )

    return {
        "accepted": True,
        "source": normalized.get(
            "ingestion_source", source
        ),
        "risk_score": normalized.get("risk_score"),
        "mitre_technique": normalized.get(
            "mitre_technique", ""
        ),
        "normalized_event": normalized,
    }


@app.post("/api/ingest")
async def ingest_inferred(request: Request):
    """
    Ingest one event with NO source hint (STREAM path).

    The pipeline's detector inspects the event and
    infers the source. This is how mixed syslog or
    Kafka streams are handled.
    """
    try:
        raw_event = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid JSON body"},
        )

    normalized = pipeline.ingest(raw_event)

    if normalized is None:
        return JSONResponse(
            status_code=422,
            content={
                "accepted": False,
                "reason": (
                    "source could not be detected or "
                    "event could not be normalized"
                ),
            },
        )

    return {
        "accepted": True,
        "source": normalized.get(
            "ingestion_source", "unknown"
        ),
        "risk_score": normalized.get("risk_score"),
        "mitre_technique": normalized.get(
            "mitre_technique", ""
        ),
        "normalized_event": normalized,
    }


@app.get("/api/stats")
def stats():
    """Return live pipeline statistics."""
    return pipeline.get_statistics()


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the live ingestion dashboard."""
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AbuTech — Layer 1 Ingestion</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.1.0/dist/tabler-icons.min.css">
<style>
  :root {
    --bg:#0a0e14; --panel:#111720; --panel2:#18202c;
    --border:#1f2b3a; --text:#e6edf3; --dim:#7d8da3;
    --push:#378ADD; --pull:#1D9E75; --stream:#D85A30;
    --accent:#00e5a0;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background:var(--bg); color:var(--text);
    font-family:'Segoe UI',system-ui,sans-serif;
    padding:28px 20px; line-height:1.5;
  }
  .wrap { max-width:1100px; margin:0 auto; }
  .head { text-align:center; margin-bottom:28px; }
  .logo {
    font-family:'Courier New',monospace; font-size:12px;
    letter-spacing:4px; color:var(--accent);
    text-transform:uppercase;
  }
  .head h1 { font-size:30px; font-weight:800; margin-top:8px; }
  .head p { color:var(--dim); margin-top:6px; font-size:14px; }
  .stats {
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:12px; margin:24px 0;
  }
  .stat {
    background:var(--panel); border:1px solid var(--border);
    border-radius:12px; padding:16px; text-align:center;
  }
  .stat .num {
    font-size:26px; font-weight:800; color:var(--accent);
  }
  .stat .lbl {
    font-size:11px; color:var(--dim); letter-spacing:1px;
    text-transform:uppercase; margin-top:4px;
  }
  .legend {
    display:flex; gap:16px; flex-wrap:wrap;
    margin-bottom:18px; font-size:13px; color:var(--dim);
  }
  .legend span { display:flex; align-items:center; gap:6px; }
  .dot { width:10px; height:10px; border-radius:3px; }
  .cat-title {
    font-size:13px; font-weight:600; color:var(--dim);
    margin:20px 0 10px; padding-bottom:6px;
    border-bottom:1px solid var(--border);
  }
  .grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
    gap:12px;
  }
  .card {
    background:var(--panel); border:1px solid var(--border);
    border-radius:12px; padding:14px; cursor:pointer;
    transition:border-color .15s;
  }
  .card:hover { border-color:var(--accent); }
  .card-top {
    display:flex; align-items:center; justify-content:space-between;
    margin-bottom:8px;
  }
  .card-name {
    display:flex; align-items:center; gap:8px;
    font-size:14px; font-weight:600;
  }
  .card-name i { font-size:20px; color:var(--dim); }
  .badge {
    font-size:11px; font-weight:600; padding:2px 8px;
    border-radius:8px; color:#06241a;
  }
  .badge.push { background:var(--push); }
  .badge.pull { background:var(--pull); }
  .badge.stream { background:var(--stream); }
  .card-desc { font-size:12px; color:var(--dim); }
  .foot {
    text-align:center; margin-top:28px; font-size:12px;
    color:var(--dim); font-family:'Courier New',monospace;
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <div class="logo">AbuTech AI Security Platform</div>
    <h1>Layer 1 — Data Ingestion</h1>
    <p>Receives security events from every source and routes them to the right normalizer.</p>
  </div>

  <div class="stats" id="stats"></div>

  <div class="legend">
    <span><span class="dot" style="background:var(--push)"></span>Push — source sends to us</span>
    <span><span class="dot" style="background:var(--pull)"></span>Pull — we fetch on schedule</span>
    <span><span class="dot" style="background:var(--stream)"></span>Stream — continuous flow</span>
  </div>

  <div id="catalog"></div>

  <div class="foot">All sources normalize to one standard event before ML scoring.</div>
</div>

<script>
async function load() {
  const res = await fetch('/api/sources');
  const data = await res.json();
  const cs = data.catalog_stats;

  document.getElementById('stats').innerHTML = `
    <div class="stat"><div class="num">${cs.total_sources}</div><div class="lbl">Sources</div></div>
    <div class="stat"><div class="num">${cs.by_method.push||0}</div><div class="lbl">Push</div></div>
    <div class="stat"><div class="num">${cs.by_method.pull||0}</div><div class="lbl">Pull</div></div>
    <div class="stat"><div class="num">${cs.by_method.stream||0}</div><div class="lbl">Stream</div></div>
  `;

  const cat = document.getElementById('catalog');
  cat.innerHTML = data.categories.map(g => `
    <div class="cat-title">${g.category} · ${g.sources.length}</div>
    <div class="grid">
      ${g.sources.map(s => `
        <div class="card" onclick="window.location='/api/sources/'+'${s.source}'">
          <div class="card-top">
            <div class="card-name"><i class="ti ${s.icon}"></i>${s.name}</div>
            <span class="badge ${s.method}">${s.method}</span>
          </div>
          <div class="card-desc">${s.description}</div>
        </div>
      `).join('')}
    </div>
  `).join('');
}
load();
</script>
</body>
</html>"""