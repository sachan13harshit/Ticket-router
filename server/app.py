"""
FastAPI application for the Customer Support Ticket Router.

Standard OpenEnv endpoints:
  POST /reset   — Load a ticket (preset or dynamic free-form)
  POST /step    — Submit routing decision
  GET  /state   — Internal state
  GET  /schema  — Action/observation schemas
  GET  /health  — Health check
  WS   /ws      — WebSocket for persistent sessions

Demo UI:
  GET  /demo    — Interactive web interface (type any ticket, get scored)
"""

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core is required. pip install openenv-core") from e

try:
    from ..models import TicketRouterAction, TicketRouterObservation
    from .ticket_router_environment import TicketRouterEnvironment
except (ImportError, ModuleNotFoundError):
    from models import TicketRouterAction, TicketRouterObservation
    from server.ticket_router_environment import TicketRouterEnvironment


app = create_app(
    TicketRouterEnvironment,
    TicketRouterAction,
    TicketRouterObservation,
    env_name="ticket_router",
    max_concurrent_envs=1,
)

# ── Shared demo environment (stateful, for /demo UI only) ────────────────────
_demo_env = TicketRouterEnvironment()


class DemoResetRequest(BaseModel):
    ticket_body:    Optional[str] = None
    ticket_subject: Optional[str] = None
    customer_tier:  str = "standard"
    task_type:      str = "easy"
    seed:           int = 0


class DemoStepRequest(BaseModel):
    primary_team: str
    priority:     str
    urgency:      str


@app.post("/demo/reset")
async def demo_reset(req: DemoResetRequest):
    obs = _demo_env.reset(
        ticket_body=req.ticket_body or None,
        ticket_subject=req.ticket_subject or None,
        customer_tier=req.customer_tier,
        task_type=req.task_type,
        seed=req.seed,
    )
    return JSONResponse(obs.model_dump())


@app.post("/demo/step")
async def demo_step(req: DemoStepRequest):
    try:
        action = TicketRouterAction(
            primary_team=req.primary_team,
            priority=req.priority,
            urgency=req.urgency,
        )
        obs = _demo_env.step(action)
        return JSONResponse({"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done})
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# ── Demo UI ──────────────────────────────────────────────────────────────────

_DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ticket Router — Demo</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f7fa;color:#1a1a2e;min-height:100vh}
  .header{background:#1a1a2e;color:#fff;padding:18px 32px;display:flex;align-items:center;gap:12px}
  .header h1{font-size:1.3rem;font-weight:600}
  .badge{background:#4caf50;color:#fff;font-size:.7rem;padding:2px 8px;border-radius:12px;font-weight:600}
  .container{max-width:900px;margin:32px auto;padding:0 16px;display:grid;gap:20px}
  .card{background:#fff;border-radius:12px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
  .card h2{font-size:1rem;font-weight:600;margin-bottom:16px;color:#1a1a2e}
  label{display:block;font-size:.82rem;font-weight:500;color:#555;margin-bottom:6px}
  textarea{width:100%;border:1.5px solid #ddd;border-radius:8px;padding:12px;font-size:.9rem;resize:vertical;min-height:120px;font-family:inherit;transition:border .2s}
  textarea:focus{outline:none;border-color:#4caf50}
  input[type=text],select{width:100%;border:1.5px solid #ddd;border-radius:8px;padding:10px 12px;font-size:.9rem;font-family:inherit;background:#fff;transition:border .2s}
  input[type=text]:focus,select:focus{outline:none;border-color:#4caf50}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
  .btn{padding:11px 28px;border:none;border-radius:8px;font-size:.9rem;font-weight:600;cursor:pointer;transition:all .2s}
  .btn-primary{background:#4caf50;color:#fff}
  .btn-primary:hover{background:#43a047}
  .btn-secondary{background:#1a1a2e;color:#fff}
  .btn-secondary:hover{background:#2d2d44}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .actions{display:flex;gap:12px;margin-top:8px;align-items:center}
  .score-box{border-radius:10px;padding:20px;text-align:center;margin-bottom:12px}
  .score-box .score-val{font-size:2.8rem;font-weight:700;line-height:1}
  .score-box .score-label{font-size:.85rem;margin-top:4px;opacity:.8}
  .score-perfect{background:#e8f5e9;color:#2e7d32}
  .score-good{background:#fff8e1;color:#f57f17}
  .score-poor{background:#fce4ec;color:#c62828}
  .result-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px}
  .result-item{background:#f9f9f9;border-radius:8px;padding:12px;text-align:center}
  .result-item .label{font-size:.75rem;color:#777;margin-bottom:4px}
  .result-item .val{font-size:.95rem;font-weight:600}
  .correct{color:#2e7d32} .wrong{color:#c62828}
  .tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.78rem;font-weight:600}
  .tag-green{background:#e8f5e9;color:#2e7d32}
  .tag-red{background:#fce4ec;color:#c62828}
  .tag-blue{background:#e3f2fd;color:#1565c0}
  .hidden{display:none}
  .spinner{display:inline-block;width:16px;height:16px;border:2px solid #fff;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .info-bar{background:#e3f2fd;border-left:4px solid #1565c0;padding:10px 14px;border-radius:0 8px 8px 0;font-size:.85rem;color:#1565c0;margin-bottom:16px}
  .mode-tabs{display:flex;gap:8px;margin-bottom:20px}
  .tab{padding:8px 18px;border-radius:8px;font-size:.85rem;font-weight:600;cursor:pointer;border:2px solid #ddd;background:#fff;color:#777;transition:all .2s}
  .tab.active{border-color:#4caf50;color:#4caf50;background:#f1f8e9}
  .breakdown{margin-top:12px;display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
  .breakdown-item{background:#f5f5f5;border-radius:8px;padding:10px 12px;font-size:.82rem}
  .breakdown-item .bk-label{color:#777;font-size:.75rem}
  .breakdown-item .bk-val{font-weight:600;margin-top:2px}
</style>
</head>
<body>

<div class="header">
  <span>🎫</span>
  <h1>Customer Support Ticket Router</h1>
  <span class="badge">OpenEnv Demo</span>
</div>

<div class="container">

  <!-- Mode tabs -->
  <div class="mode-tabs">
    <div class="tab active" id="tab-free" onclick="switchTab('free')">✏️ Free-form Ticket</div>
    <div class="tab" id="tab-preset" onclick="switchTab('preset')">📋 Preset Scenarios</div>
  </div>

  <!-- Free-form input card -->
  <div class="card" id="panel-free">
    <h2>📝 Describe the Issue</h2>
    <div class="info-bar">
      Type any support ticket. The system will auto-infer the correct routing and score your decision.
    </div>
    <div style="margin-bottom:14px">
      <label>Ticket Body *</label>
      <textarea id="ticket-body" placeholder="Describe the issue, request, or complaint...&#10;&#10;Example: I have faced an issue, I am not able to see the login button to a specific portal, kindly give permission to do that"></textarea>
    </div>
    <div class="row" style="margin-bottom:14px">
      <div>
        <label>Subject (optional)</label>
        <input type="text" id="ticket-subject" placeholder="One-line summary (auto-generated if blank)">
      </div>
      <div>
        <label>Customer Tier</label>
        <select id="customer-tier">
          <option value="standard">Standard</option>
          <option value="premium">Premium</option>
          <option value="enterprise">Enterprise</option>
        </select>
      </div>
    </div>
    <button class="btn btn-primary" onclick="loadTicket()">Load Ticket →</button>
  </div>

  <!-- Preset input card -->
  <div class="card hidden" id="panel-preset">
    <h2>📋 Preset Scenario</h2>
    <div class="info-bar">Select a benchmark scenario with a deterministic expected answer.</div>
    <div class="row">
      <div>
        <label>Difficulty</label>
        <select id="task-type">
          <option value="easy">Easy — Clear signals, balanced teams</option>
          <option value="medium">Medium — Multi-intent, moderate load</option>
          <option value="hard">Hard — Ambiguous + overloaded teams</option>
        </select>
      </div>
      <div>
        <label>Scenario Seed (0–4)</label>
        <select id="seed">
          <option value="0">Seed 0</option>
          <option value="1">Seed 1</option>
          <option value="2">Seed 2</option>
          <option value="3">Seed 3</option>
          <option value="4">Seed 4</option>
        </select>
      </div>
    </div>
    <button class="btn btn-secondary" style="margin-top:16px" onclick="loadPreset()">Load Scenario →</button>
  </div>

  <!-- Ticket display (shown after load) -->
  <div class="card hidden" id="ticket-card">
    <h2>🎫 Ticket <span id="scenario-id" style="font-size:.8rem;font-weight:400;color:#777"></span></h2>
    <div style="margin-bottom:10px">
      <div style="font-weight:600;margin-bottom:4px" id="display-subject"></div>
      <div style="font-size:.9rem;color:#444;line-height:1.6" id="display-body"></div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px">
      <span class="tag tag-blue" id="display-tier"></span>
      <span class="tag" style="background:#f3e5f5;color:#6a1b9a" id="display-mode"></span>
    </div>

    <!-- Team loads -->
    <div style="margin-top:16px">
      <div style="font-size:.82rem;font-weight:600;color:#777;margin-bottom:8px">TEAM STATUS</div>
      <div id="team-table" style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px"></div>
    </div>
  </div>

  <!-- Routing decision -->
  <div class="card hidden" id="routing-card">
    <h2>🔀 Your Routing Decision</h2>
    <div class="row3">
      <div>
        <label>Primary Team</label>
        <select id="sel-team">
          <option>Billing</option>
          <option>Tech Support</option>
          <option>Account</option>
          <option>Product</option>
          <option>Escalations</option>
        </select>
      </div>
      <div>
        <label>Priority</label>
        <select id="sel-priority">
          <option value="low">Low</option>
          <option value="medium" selected>Medium</option>
          <option value="high">High</option>
        </select>
      </div>
      <div>
        <label>Urgency</label>
        <select id="sel-urgency">
          <option value="low">Low</option>
          <option value="medium" selected>Medium</option>
          <option value="high">High</option>
        </select>
      </div>
    </div>
    <div class="actions" style="margin-top:16px">
      <button class="btn btn-primary" id="score-btn" onclick="scoreRouting()">
        Score Routing
      </button>
      <button class="btn" style="background:#eee;color:#333" onclick="resetAll()">
        ↩ New Ticket
      </button>
    </div>
  </div>

  <!-- Result -->
  <div class="card hidden" id="result-card">
    <h2>📊 Result</h2>
    <div class="score-box" id="score-box">
      <div class="score-val" id="score-val">—</div>
      <div class="score-label" id="score-label">Score</div>
    </div>

    <div class="breakdown">
      <div class="breakdown-item">
        <div class="bk-label">Team</div>
        <div class="bk-val" id="bk-team"></div>
      </div>
      <div class="breakdown-item">
        <div class="bk-label">Priority</div>
        <div class="bk-val" id="bk-priority"></div>
      </div>
      <div class="breakdown-item">
        <div class="bk-label">Urgency</div>
        <div class="bk-val" id="bk-urgency"></div>
      </div>
    </div>

    <div class="result-grid" style="margin-top:12px">
      <div class="result-item">
        <div class="label">Expected Team</div>
        <div class="val" id="res-exp-team">—</div>
      </div>
      <div class="result-item">
        <div class="label">Expected Priority</div>
        <div class="val" id="res-exp-priority">—</div>
      </div>
      <div class="result-item">
        <div class="label">Expected Urgency</div>
        <div class="val" id="res-exp-urgency">—</div>
      </div>
    </div>

    <div id="overload-warning" class="hidden" style="margin-top:12px;background:#fff3cd;border-left:4px solid #ff9800;padding:10px 14px;border-radius:0 8px 8px 0;font-size:.85rem;color:#e65100">
      ⚠️ Overload penalty applied: chosen team has queue &gt; 10 and a better alternative exists (−0.2)
    </div>

    <div style="margin-top:16px">
      <button class="btn btn-secondary" onclick="resetAll()">↩ Try Another Ticket</button>
    </div>
  </div>

</div>

<script>
let currentMode = 'free';
let ticketLoaded = false;

function switchTab(mode) {
  currentMode = mode;
  document.getElementById('tab-free').classList.toggle('active', mode==='free');
  document.getElementById('tab-preset').classList.toggle('active', mode==='preset');
  document.getElementById('panel-free').classList.toggle('hidden', mode!=='free');
  document.getElementById('panel-preset').classList.toggle('hidden', mode!=='preset');
}

function show(id){ document.getElementById(id).classList.remove('hidden'); }
function hide(id){ document.getElementById(id).classList.add('hidden'); }

async function loadTicket() {
  const body = document.getElementById('ticket-body').value.trim();
  if(!body){ alert('Please enter a ticket description.'); return; }
  const subject = document.getElementById('ticket-subject').value.trim();
  const tier = document.getElementById('customer-tier').value;

  const payload = { ticket_body: body, customer_tier: tier };
  if(subject) payload.ticket_subject = subject;

  await doReset(payload, 'dynamic');
}

async function loadPreset() {
  const task_type = document.getElementById('task-type').value;
  const seed = parseInt(document.getElementById('seed').value);
  await doReset({ task_type, seed }, 'preset');
}

async function doReset(payload, mode) {
  try {
    const r = await fetch('/demo/reset', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await r.json();
    const obs = data;

    document.getElementById('display-subject').textContent = obs.ticket_subject || '';
    document.getElementById('display-body').textContent    = obs.ticket_body    || '';
    document.getElementById('display-tier').textContent    = '👤 ' + (obs.customer_tier || 'standard');
    document.getElementById('display-mode').textContent    = mode==='dynamic' ? '✏️ Free-form' : '📋 Preset';
    document.getElementById('scenario-id').textContent     = obs.scenario_id ? `(${obs.scenario_id})` : '';

    // Team status table
    const tbl = document.getElementById('team-table');
    tbl.innerHTML = '';
    (obs.team_status || []).forEach(t => {
      const overloaded = t.queue_length > 10;
      tbl.innerHTML += `<div style="background:${overloaded?'#fce4ec':'#f9f9f9'};border-radius:8px;padding:8px;text-align:center;font-size:.78rem">
        <div style="font-weight:600;color:${overloaded?'#c62828':'#333'}">${t.name}</div>
        <div style="color:#777;margin-top:2px">Queue: <b>${t.queue_length}</b>${overloaded?' 🔴':''}</div>
        <div style="color:#999;font-size:.72rem">${t.avg_resolution_time_min}min avg</div>
      </div>`;
    });

    show('ticket-card');
    show('routing-card');
    hide('result-card');
    ticketLoaded = true;
  } catch(e) {
    alert('Error loading ticket: ' + e.message);
  }
}

async function scoreRouting() {
  const btn = document.getElementById('score-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Scoring...';

  const team     = document.getElementById('sel-team').value;
  const priority = document.getElementById('sel-priority').value;
  const urgency  = document.getElementById('sel-urgency').value;

  try {
    const r = await fetch('/demo/step', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ primary_team: team, priority, urgency })
    });
    const data = await r.json();
    if(data.error){ alert('Error: ' + data.error); btn.disabled=false; btn.innerHTML='Score Routing'; return; }
    const meta = (data.observation || {}).metadata || {};
    const score = meta.score ?? 0;
    const reward = data.reward ?? 0;

    // Score display
    const box = document.getElementById('score-box');
    const val = document.getElementById('score-val');
    const lbl = document.getElementById('score-label');
    val.textContent = (score * 100).toFixed(0) + '%';
    lbl.textContent = `Reward: ${reward >= 0 ? '+' : ''}${reward.toFixed(2)}`;
    box.className = 'score-box ' + (score>=0.8?'score-perfect':score>=0.4?'score-good':'score-poor');

    // Breakdown
    document.getElementById('bk-team').innerHTML = `${team} <span class="tag ${meta.team_correct?'tag-green':'tag-red'}">${meta.team_correct?'✓ Correct':'✗ Wrong'}</span>`;
    document.getElementById('bk-priority').innerHTML = `${priority} <span class="tag ${meta.priority_correct?'tag-green':'tag-red'}">${meta.priority_correct?'✓':'✗'}</span>`;
    document.getElementById('bk-urgency').innerHTML = `${urgency} <span class="tag ${meta.urgency_correct?'tag-green':'tag-red'}">${meta.urgency_correct?'✓':'✗'}</span>`;

    document.getElementById('res-exp-team').textContent     = meta.expected_team     || '—';
    document.getElementById('res-exp-priority').textContent = meta.expected_priority || '—';
    document.getElementById('res-exp-urgency').textContent  = meta.expected_urgency  || '—';

    if(meta.overload_penalty) show('overload-warning'); else hide('overload-warning');

    show('result-card');
    hide('routing-card');
  } catch(e) {
    alert('Error scoring: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Score Routing';
  }
}

function resetAll() {
  hide('ticket-card'); hide('routing-card'); hide('result-card');
  document.getElementById('ticket-body').value = '';
  document.getElementById('ticket-subject').value = '';
  ticketLoaded = false;
}
</script>
</body>
</html>
"""


@app.get("/demo", response_class=HTMLResponse)
async def demo_ui(request: Request):
    return HTMLResponse(content=_DEMO_HTML)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
