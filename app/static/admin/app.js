'use strict';

// agent_server admin UI - talks to the admin API (see app/admin_api.py).
// Relative base ('api', resolved against the page URL) so the UI works both
// natively at /admin/ and behind the reverse proxy at /llm/admin/.
const API = 'api';
let editing = null;        // name of the agent in the form, or null when creating
let testMessages = [];     // the in-UI tester conversation

// --------------------------------------------------------------------------
// HTTP helper
// --------------------------------------------------------------------------
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(API + path, opts);
  const text = await res.text();
  let data = null;
  if (text) { try { data = JSON.parse(text); } catch (e) { data = text; } }
  if (!res.ok) throw new Error(errText(data, res.status));
  return data;
}

// Pulls a message out of FastAPI ({detail:"..."} / {detail:{message,errors}})
// and OpenAI ({detail:{error:{message}}}) error shapes.
function errText(data, status) {
  if (data && typeof data === 'object') {
    const d = data.detail !== undefined ? data.detail : data;
    if (typeof d === 'string') return d;
    if (d && typeof d === 'object') {
      if (d.error && d.error.message) return d.error.message;
      if (d.message) {
        const errs = Array.isArray(d.errors) ? ': ' + d.errors.join('; ') : '';
        return d.message + errs;
      }
    }
  }
  return 'HTTP ' + status;
}

let toastTimer = null;
function toast(msg, kind) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast ' + (kind || 'ok');
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4500);
}

// --------------------------------------------------------------------------
// Tabs
// --------------------------------------------------------------------------
function showTab(which) {
  document.getElementById('view-dashboard').hidden = which !== 'dashboard';
  document.getElementById('view-agents').hidden = which !== 'agents';
  document.getElementById('view-clients').hidden = which !== 'clients';
  document.getElementById('view-config').hidden = which !== 'config';
  document.getElementById('tab-dashboard').classList.toggle('active', which === 'dashboard');
  document.getElementById('tab-agents').classList.toggle('active', which === 'agents');
  document.getElementById('tab-clients').classList.toggle('active', which === 'clients');
  document.getElementById('tab-config').classList.toggle('active', which === 'config');
  if (which === 'dashboard') { loadDashboard(true); startDashTimer(); }
  else { stopDashTimer(); }
  if (which === 'agents') loadAgents();
  if (which === 'clients') loadClients();
  if (which === 'config') loadConfig();
}

// --------------------------------------------------------------------------
// Dashboard
// --------------------------------------------------------------------------
let dashTimer = null;
let selectedModel = null;
let switching = false;
let modelMeta = {};        // model_id -> {vision, reasoning, context, family, file}
let activeContext = null;  // active model's current ctx-size (tokens)
let activeModelId = null;

// Human-readable GGUF size, e.g. 5966095584 -> "5.6 GB".
function fmtSize(bytes) {
  if (!bytes || bytes < 0) return '';
  const gb = bytes / 1073741824;
  if (gb >= 1) return (gb >= 10 ? Math.round(gb) : gb.toFixed(1)) + ' GB';
  return Math.round(bytes / 1048576) + ' MB';
}

// Context-size choices for the dashboard selector. Always includes the
// model's current value so it shows even if it isn't a round number.
function ctxOptions(current) {
  const set = new Set([4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072]);
  if (current) set.add(current);
  return Array.from(set).sort((a, b) => a - b);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// GET an endpoint at the server root (not under /admin/api), e.g. v1/models.
// The admin UI lives at /admin/, so '../' resolves to the server root.
async function rootGet(path) {
  const res = await fetch('../' + path);
  const text = await res.text();
  let data = null;
  if (text) { try { data = JSON.parse(text); } catch (e) { data = text; } }
  if (!res.ok) throw new Error(errText(data, res.status));
  return data;
}

function badge(text, kind) {
  const s = document.createElement('span');
  s.className = 'badge' + (kind ? ' badge-' + kind : '');
  s.textContent = text;
  return s;
}

function startDashTimer() {
  stopDashTimer();
  // Light periodic refresh; skipped while a switch is in progress.
  dashTimer = setInterval(() => { if (!switching) loadDashboard(false); }, 6000);
}
function stopDashTimer() { if (dashTimer) { clearInterval(dashTimer); dashTimer = null; } }

async function loadDashboard(first) {
  if (first) await loadConfigMeta();   // per-model metadata, fetched once
  await Promise.all([loadModels(), loadStatus(), loadCalls(), loadMemory()]);
}

// One-time: pull per-model metadata (vision/reasoning/context/file) from the
// live config so the model cards can show badges /v1/models doesn't carry.
async function loadConfigMeta() {
  try {
    const res = await api('GET', '/config');
    const chat = (((res.live || {}).models || {}).chat) || [];
    modelMeta = {};
    for (const m of chat) {
      const bk = (m.backends || {})[m.active_backend] || {};
      modelMeta[m.model_id] = {
        vision: !!m.vision, reasoning: !!m.reasoning, context: m.context,
        family: m.family, file: (bk.model_file || '').split('/').pop(),
      };
    }
  } catch (e) { /* non-fatal: cards just show less */ }
}

async function loadModels() {
  let data;
  try { data = await rootGet('v1/models'); } catch (e) { return; }
  const chat = (data.data || []).filter((m) => m.kind === 'chat');
  const active = chat.find((m) => m.active) || null;
  renderActiveCard(active);
  renderModelList(chat);
}

function renderActiveCard(active) {
  const el = document.getElementById('dash-active');
  if (!active) { el.innerHTML = '<span class="muted">unknown</span>'; return; }
  const meta = modelMeta[active.id] || {};
  const ctx = active.context != null ? active.context : meta.context;
  activeModelId = active.id;
  activeContext = ctx || null;
  el.innerHTML =
      '<div class="active-name"><span class="dot on"></span><b></b>'
    + '<small class="active-size muted"></small></div>'
    + '<div class="active-badges"></div>'
    + '<div class="active-file muted"></div>'
    + '<div class="ctx-control"><label>Context</label>'
    + '<select id="ctx-select"></select>'
    + '<button id="btn-ctx-apply" class="ghost">Apply</button>'
    + '<span class="ctx-hint muted">restarts ~40s · VRAM grows with context</span></div>';
  el.querySelector('b').textContent = active.display_name || active.id;
  if (active.size_bytes)
    el.querySelector('.active-size').textContent = ' (' + fmtSize(active.size_bytes) + ')';
  const b = el.querySelector('.active-badges');
  b.append(badge(meta.family || active.family || '?', 'fam'));
  if (meta.reasoning) b.append(badge('reasoning'));
  if (meta.vision) b.append(badge('vision', 'vision'));
  if (ctx) b.append(badge((ctx / 1024) + 'K ctx'));
  el.querySelector('.active-file').textContent = meta.file || active.id;

  const sel = el.querySelector('#ctx-select');
  for (const v of ctxOptions(ctx)) {
    const o = document.createElement('option');
    o.value = String(v);
    o.textContent = (v / 1024) + 'K';
    if (v === ctx) o.selected = true;
    sel.appendChild(o);
  }
  el.querySelector('#btn-ctx-apply').onclick = applyContext;
}

function renderModelList(chat) {
  const box = document.getElementById('model-list');
  box.innerHTML = '';
  for (const m of chat) {
    const meta = modelMeta[m.id] || {};
    const row = document.createElement('label');
    row.className = 'model-row' + (m.active ? ' is-active' : '');
    row.innerHTML = '<input type="radio" name="dash-model">'
      + '<span class="m-name"></span><span class="m-badges"></span>';
    const nameEl = row.querySelector('.m-name');
    nameEl.textContent = m.display_name || m.id;
    if (m.size_bytes) {
      const sz = document.createElement('small');
      sz.className = 'muted m-size';
      sz.textContent = ' (' + fmtSize(m.size_bytes) + ')';
      nameEl.appendChild(sz);
    }
    const bb = row.querySelector('.m-badges');
    bb.append(badge(meta.family || m.family || '?', 'fam'));
    if (meta.reasoning) bb.append(badge('reasoning'));
    if (meta.vision) bb.append(badge('vision', 'vision'));
    if (meta.context) bb.append(badge((meta.context / 1024) + 'K'));
    const radio = row.querySelector('input');
    if (m.active) { radio.checked = true; bb.append(badge('active', 'on')); }
    radio.onchange = () => {
      selectedModel = m.id;
      document.getElementById('btn-activate').disabled = false;
    };
    box.appendChild(row);
  }
}

async function loadStatus() {
  const el = document.getElementById('dash-status');
  let s;
  try { s = await api('GET', '/status'); }
  catch (e) { el.innerHTML = '<span class="err-text">status unavailable</span>'; return; }
  let html = '';
  if (s.gpu) {
    const g = s.gpu, pct = g.total_mb ? Math.round(100 * g.used_mb / g.total_mb) : 0;
    html += '<div class="vram-row"><span>VRAM</span><span class="muted">'
      + (g.used_mb / 1024).toFixed(1) + ' / ' + (g.total_mb / 1024).toFixed(1)
      + ' GB · ' + g.util_pct + '% util</span></div>'
      + '<div class="progress sm"><div class="progress-bar" style="width:' + pct + '%"></div></div>';
  } else {
    html += '<div class="muted">GPU stats unavailable</div>';
  }
  const dot = s.router_reachable ? '<span class="dot on"></span>healthy'
    : '<span class="dot off"></span>router unreachable';
  html += '<div class="status-line">llama-vision: ' + dot + '</div>';
  if (s.resident && s.resident.length) {
    // Model ids are safe (kebab-case from config); inline small tags for size/state.
    html += '<div class="status-line muted">resident: '
      + s.resident.map((r) => r.id
          + (r.size ? ' <small>(' + fmtSize(r.size) + ')</small>' : '')
          + (r.state && r.state !== 'loaded' ? ' <small>· ' + r.state + '</small>' : '')
        ).join(', ')
      + '</div>';
  }
  el.innerHTML = html;
}

async function loadCalls() {
  let res;
  try { res = await api('GET', '/calls?limit=40'); } catch (e) { return; }
  const body = document.getElementById('calls-body');
  const calls = res.calls || [];
  document.getElementById('calls-empty').hidden = calls.length > 0;
  body.innerHTML = '';
  for (const c of calls) {
    const tr = document.createElement('tr');
    const tin = c.prompt_tokens != null ? c.prompt_tokens : '·';
    const tout = c.completion_tokens != null ? c.completion_tokens : (c.chunks != null ? c.chunks + 'c' : '·');
    tr.innerHTML = '<td class="muted"></td><td></td><td class="muted"></td>'
      + '<td class="num"></td><td class="num"></td><td></td>';
    const td = tr.children;
    td[0].textContent = fmtTime(c.ts);
    td[1].textContent = c.model || '?';
    td[2].textContent = c.agent || (c.stream ? 'stream' : '');
    td[3].textContent = tin + '/' + tout;
    td[4].textContent = c.latency_ms != null ? c.latency_ms : '';
    td[5].innerHTML = '';
    td[5].append(badge(c.status || '?', c.status === 'ok' ? 'on'
      : (c.status === 'error' ? 'err' : 'warn')));
    body.appendChild(tr);
  }
}

async function loadMemory() {
  let res;
  try { res = await api('GET', '/memory'); } catch (e) { return; }
  document.getElementById('mem-note').textContent = res.note || '';
  const body = document.getElementById('mem-body');
  const threads = res.threads || [];
  document.getElementById('mem-empty').hidden = threads.length > 0;
  body.innerHTML = '';
  for (const t of threads) {
    const tr = document.createElement('tr');
    tr.className = 'clickable';
    tr.innerHTML = '<td></td><td class="num"></td><td class="muted"></td>';
    tr.children[0].textContent = t.thread_id;
    tr.children[1].textContent = t.messages;
    tr.children[2].textContent = (t.last_role ? t.last_role + ': ' : '') + (t.last_preview || '');
    tr.onclick = () => openThread(t.thread_id);
    body.appendChild(tr);
  }
}

async function openThread(id) {
  let res;
  try { res = await api('GET', '/memory/' + encodeURIComponent(id)); }
  catch (e) { toast('Load failed: ' + e.message, 'err'); return; }
  document.getElementById('mem-modal-title').textContent = id + ' (' + res.count + ' msgs)';
  const box = document.getElementById('mem-modal-body');
  box.innerHTML = '';
  for (const m of res.messages) {
    const d = document.createElement('div');
    d.className = 'msg msg-' + (m.role || 'user');
    d.innerHTML = '<span class="msg-role"></span><span class="msg-text"></span>';
    d.querySelector('.msg-role').textContent = m.role;
    d.querySelector('.msg-text').textContent = m.content;
    box.appendChild(d);
  }
  document.getElementById('mem-modal').hidden = false;
}

async function toggleLogs() {
  const box = document.getElementById('logs-box');
  if (!box.hidden) { box.hidden = true; return; }
  box.textContent = 'loading…';
  box.hidden = false;
  try {
    const res = await api('GET', '/logs?tail=200&container=agent_server');
    box.textContent = res.logs || '(empty)';
    box.scrollTop = box.scrollHeight;
  } catch (e) { box.textContent = 'logs unavailable: ' + e.message; }
}

// ---- Switch flow with milestone progress -------------------------------
function showSwitchOverlay(target) {
  document.getElementById('switch-title').textContent = 'Switching to ' + target + '…';
  document.getElementById('switch-steps').innerHTML = '';
  setBar(0);
  document.getElementById('switch-overlay').hidden = false;
}
function hideSwitchOverlay() { document.getElementById('switch-overlay').hidden = true; }
function setBar(pct) { document.getElementById('switch-bar').style.width = pct + '%'; }
function addStep(text, kind) {
  const li = document.createElement('li');
  li.className = kind ? 'step-' + kind : '';
  li.textContent = text;
  document.getElementById('switch-steps').appendChild(li);
}

async function activateSelected() {
  if (!selectedModel || switching) return;
  if (!confirm('Switch the active model to "' + selectedModel + '"?\n\n'
    + 'This restarts llama-vision + agent_server (~40s) and drops in-flight chats.')) return;

  switching = true;
  stopDashTimer();
  document.getElementById('btn-activate').disabled = true;
  const target = selectedModel;
  showSwitchOverlay(target);

  // elapsed-time ticker
  const t0 = Date.now();
  const ticker = setInterval(() => {
    document.getElementById('switch-elapsed').textContent =
      Math.round((Date.now() - t0) / 1000) + 's';
  }, 500);
  const done = () => { clearInterval(ticker); switching = false; };

  addStep('Requesting switch…');
  let resp;
  try { resp = await api('POST', '/active-model', { model_id: target }); }
  catch (e) { addStep('✗ ' + e.message, 'err'); setBar(0); done(); setTimeout(hideSwitchOverlay, 4000); return; }

  if (resp.noop) { addStep('Already active.', 'ok'); setBar(100); finishSwitch(done); return; }
  if (resp.status !== 'switching') {
    addStep(resp.note || 'Config written; auto-restart unavailable — restart manually.', 'warn');
    setBar(50); done(); return;
  }
  addStep('✓ config flipped', 'ok'); setBar(25);
  addStep('Restarting llama-vision + agent_server…');

  // Poll /v1/models until agent_server is back AND the target is active.
  const deadline = Date.now() + 150000;
  let sawDown = false, sawBack = false;
  while (Date.now() < deadline) {
    await sleep(2500);
    let data = null;
    try { data = await rootGet('v1/models'); }
    catch (e) {
      if (!sawDown) { addStep('agent_server is down (restarting)…'); setBar(50); sawDown = true; }
      continue;
    }
    if (!sawBack) { addStep('✓ agent_server back', 'ok'); setBar(75); sawBack = true; }
    const chat = (data.data || []).filter((m) => m.kind === 'chat');
    const active = chat.find((m) => m.active);
    if (active && active.id === target) {
      addStep('✓ ' + (active.display_name || target) + ' is active & serving', 'ok');
      setBar(100); finishSwitch(done); return;
    }
    addStep('waiting for ' + target + ' to become active…');
  }
  addStep('✗ timed out waiting for the switch', 'err');
  done();
}

function finishSwitch(done) {
  if (done) done();
  selectedModel = null;
  toast('Active model switched', 'ok');
  setTimeout(() => { hideSwitchOverlay(); loadDashboard(false); startDashTimer(); }, 1400);
}

// Change the active model's context window. Same restart+poll machinery as
// activateSelected, but verifies the active model's `context` instead of id.
async function applyContext() {
  if (switching) return;
  const sel = document.getElementById('ctx-select');
  if (!sel) return;
  const ctx = parseInt(sel.value, 10);
  if (!ctx) return;
  if (ctx === activeContext) { toast('Context unchanged', 'warn'); return; }
  if (!confirm('Set context for "' + (activeModelId || 'active model') + '" to '
    + (ctx / 1024) + 'K tokens?\n\nThis restarts llama-vision + agent_server (~40s) '
    + 'and drops in-flight chats. Larger contexts use more VRAM and may fail to load.')) return;

  switching = true;
  stopDashTimer();
  showSwitchOverlay((ctx / 1024) + 'K context');
  const t0 = Date.now();
  const ticker = setInterval(() => {
    document.getElementById('switch-elapsed').textContent =
      Math.round((Date.now() - t0) / 1000) + 's';
  }, 500);
  const done = () => { clearInterval(ticker); switching = false; };

  addStep('Requesting context change…');
  let resp;
  try { resp = await api('POST', '/active-context', { context: ctx }); }
  catch (e) { addStep('✗ ' + e.message, 'err'); setBar(0); done(); setTimeout(hideSwitchOverlay, 4000); return; }

  if (resp.noop) { addStep('Already at that context.', 'ok'); setBar(100); finishCtx(done); return; }
  if (resp.status !== 'switching') {
    addStep(resp.note || 'Config written; auto-restart unavailable — restart manually.', 'warn');
    setBar(50); done(); return;
  }
  addStep('✓ config updated (ctx=' + ctx + ')', 'ok'); setBar(25);
  addStep('Restarting llama-vision + agent_server…');

  const deadline = Date.now() + 150000;
  let sawDown = false, sawBack = false;
  while (Date.now() < deadline) {
    await sleep(2500);
    let data = null;
    try { data = await rootGet('v1/models'); }
    catch (e) {
      if (!sawDown) { addStep('agent_server is down (restarting)…'); setBar(50); sawDown = true; }
      continue;
    }
    if (!sawBack) { addStep('✓ agent_server back', 'ok'); setBar(75); sawBack = true; }
    const chat = (data.data || []).filter((m) => m.kind === 'chat');
    const active = chat.find((m) => m.active);
    if (active && active.context === ctx) {
      addStep('✓ context now ' + (ctx / 1024) + 'K & serving', 'ok');
      setBar(100); finishCtx(done); return;
    }
    addStep('waiting for ctx=' + (ctx / 1024) + 'K to apply…');
  }
  addStep('✗ timed out waiting for the context change', 'err');
  done();
}

function finishCtx(done) {
  if (done) done();
  toast('Context updated', 'ok');
  setTimeout(() => { hideSwitchOverlay(); loadDashboard(false); startDashTimer(); }, 1400);
}

function fmtTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// --------------------------------------------------------------------------
// Agents
// --------------------------------------------------------------------------
async function loadAgents() {
  const tbody = document.querySelector('#agent-table tbody');
  tbody.innerHTML = '';
  let data;
  try { data = await api('GET', '/agents'); }
  catch (e) { toast('Load failed: ' + e.message, 'err'); return; }
  for (const a of data.agents) {
    const tr = document.createElement('tr');
    tr.dataset.name = a.name;
    tr.innerHTML = '<td class="a-name"></td><td class="a-mem"></td>';
    tr.querySelector('.a-name').textContent = a.name + (a.protected ? '  🔒' : '');
    tr.querySelector('.a-mem').textContent = a.memory_policy;
    tr.onclick = () => editAgent(a.name);
    tbody.appendChild(tr);
  }
  // Auto-select the first agent so its details are shown immediately, unless
  // the user is mid-edit (e.g. creating a new agent).
  if (data.agents.length && !editing) editAgent(data.agents[0].name);
}

function _highlightAgentRow(name) {
  document.querySelectorAll('#agent-table tbody tr').forEach((tr) =>
    tr.classList.toggle('selected', tr.dataset.name === name));
}

async function editAgent(name) {
  let a;
  try { a = await api('GET', '/agents/' + encodeURIComponent(name)); }
  catch (e) { toast('Load failed: ' + e.message, 'err'); return; }
  editing = a.name;
  _highlightAgentRow(a.name);
  document.getElementById('form-title').textContent = 'Edit: ' + a.name;
  document.getElementById('agent-form').hidden = false;
  const fn = document.getElementById('f-name');
  fn.value = a.name;
  fn.disabled = true;
  document.getElementById('f-prompt').value = a.system_prompt || '';
  document.getElementById('f-memory').value = a.memory_policy || 'none';
  document.getElementById('f-tts').value = a.tts_field || '';
  document.getElementById('f-params').value =
    JSON.stringify(a.params_override || {}, null, 2);
  document.getElementById('btn-delete').hidden = !!a.protected;
  showTestPanel(a.name);
}

function newAgent() {
  editing = null;
  document.getElementById('form-title').textContent = 'New agent';
  document.getElementById('agent-form').hidden = false;
  const fn = document.getElementById('f-name');
  fn.value = '';
  fn.disabled = false;
  document.getElementById('f-prompt').value = '';
  document.getElementById('f-memory').value = 'none';
  document.getElementById('f-tts').value = '';
  document.getElementById('f-params').value = '{}';
  document.getElementById('btn-delete').hidden = true;
  hideTestPanel();
  fn.focus();
}

function cancelForm() {
  document.getElementById('agent-form').hidden = true;
  document.getElementById('form-title').textContent = 'Select an agent, or create one';
  editing = null;
  hideTestPanel();
}

async function saveAgent(ev) {
  ev.preventDefault();
  const name = document.getElementById('f-name').value.trim().toLowerCase();
  let params;
  try {
    params = JSON.parse(document.getElementById('f-params').value || '{}');
  } catch (e) {
    toast('params_override is not valid JSON: ' + e.message, 'err');
    return;
  }
  if (params === null || typeof params !== 'object' || Array.isArray(params)) {
    toast('params_override must be a JSON object', 'err');
    return;
  }
  const tts = document.getElementById('f-tts').value.trim();
  const body = {
    name: name,
    system_prompt: document.getElementById('f-prompt').value,
    params_override: params,
    memory_policy: document.getElementById('f-memory').value,
    tts_field: tts || null,
  };
  try {
    if (editing) {
      await api('PUT', '/agents/' + encodeURIComponent(editing), body);
    } else {
      await api('POST', '/agents', body);
    }
  } catch (e) {
    toast('Save failed: ' + e.message, 'err');
    return;
  }
  toast('Saved "' + name + '" - live now (hot-reloaded). Test it below.', 'ok');
  await loadAgents();
  // Re-open on the saved agent so the tester is right there.
  await editAgent(name);
}

async function deleteAgent() {
  if (!editing) return;
  if (!confirm('Delete agent "' + editing + '"? This cannot be undone.')) return;
  try {
    await api('DELETE', '/agents/' + encodeURIComponent(editing));
  } catch (e) {
    toast('Delete failed: ' + e.message, 'err');
    return;
  }
  toast('Deleted "' + editing + '"', 'ok');
  cancelForm();
  loadAgents();
}

// --------------------------------------------------------------------------
// In-UI agent tester  ->  POST /v1/chat/completions  (model = agent name)
// --------------------------------------------------------------------------
function showTestPanel(name) {
  document.getElementById('test-title').textContent = 'Test: ' + name;
  document.getElementById('test-panel').hidden = false;
  resetTest();
}

function hideTestPanel() {
  document.getElementById('test-panel').hidden = true;
  resetTest();
}

function resetTest() {
  testMessages = [];
  document.getElementById('test-log').innerHTML = '';
  const m = document.getElementById('test-msg');
  if (m) m.value = '';
}

function appendBubble(role, text) {
  const log = document.getElementById('test-log');
  const div = document.createElement('div');
  div.className = 'bubble ' + role;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

async function sendTest() {
  if (!editing) { toast('Save the agent first, then test it', 'err'); return; }
  const inp = document.getElementById('test-msg');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  testMessages.push({ role: 'user', content: text });
  appendBubble('user', text);
  const pending = appendBubble('assistant', 'thinking...');
  const btn = document.getElementById('btn-test-send');
  btn.disabled = true;
  try {
    const res = await fetch('../v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: editing, messages: testMessages, stream: false }),
    });
    const raw = await res.text();
    let data = null;
    if (raw) { try { data = JSON.parse(raw); } catch (e) {} }
    if (!res.ok) {
      pending.textContent = 'Error: ' + errText(data, res.status);
      pending.classList.add('err');
      testMessages.pop();             // drop the dangling user turn
      return;
    }
    const msg = (data && data.choices && data.choices[0] && data.choices[0].message) || {};
    const answer = msg.content || '';
    pending.textContent = answer ||
      (msg.reasoning_content
        ? '(model produced only reasoning, no answer text - try a larger max_tokens)'
        : '(empty response)');
    testMessages.push({ role: 'assistant', content: answer });
  } catch (e) {
    pending.textContent = 'Request failed: ' + e.message;
    pending.classList.add('err');
    testMessages.pop();
  } finally {
    btn.disabled = false;
  }
}

// --------------------------------------------------------------------------
// Service config
// --------------------------------------------------------------------------
async function loadConfig() {
  let data;
  try { data = await api('GET', '/config'); }
  catch (e) { toast('Config load failed: ' + e.message, 'err'); return; }
  const ed = document.getElementById('config-editor');
  if (data.on_disk !== null && data.on_disk !== undefined) {
    ed.value = JSON.stringify(data.on_disk, null, 2);
  } else {
    ed.value = '';
    toast('Could not read config: ' + (data.read_error || 'unknown'), 'err');
  }
  document.getElementById('config-path').textContent =
    'File: ' + (data.config_path || '');
  setBanner(data.restart_pending);
}

function setBanner(pending) {
  const b = document.getElementById('restart-banner');
  if (pending) {
    b.textContent = 'Restart pending - the on-disk config differs from the ' +
      'running config. Run "docker restart agent_server" to apply.';
    b.hidden = false;
  } else {
    b.hidden = true;
  }
}

async function saveConfig() {
  let cfg;
  try {
    cfg = JSON.parse(document.getElementById('config-editor').value);
  } catch (e) {
    toast('Config is not valid JSON: ' + e.message, 'err');
    return;
  }
  let res;
  try {
    res = await api('PUT', '/config', cfg);
  } catch (e) {
    toast('Config save failed: ' + e.message, 'err');
    return;
  }
  toast('Config written - restart agent_server to apply', 'ok');
  setBanner(res.restart_pending);
}

// --------------------------------------------------------------------------
// Clients
// --------------------------------------------------------------------------
function _fmtDuration(s) {
  if (s == null) return '—';
  if (s < 60) return Math.round(s) + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm ' + Math.round(s % 60) + 's';
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
}

// ISO 3166-1 alpha-2 -> full country name in the browser locale ("PT" ->
// "Portugal"). Intl.DisplayNames is in all modern browsers since 2021; falls
// back to the raw code if unavailable or unknown.
const _countryNamer = (typeof Intl !== 'undefined'
  && typeof Intl.DisplayNames === 'function')
  ? new Intl.DisplayNames(['en'], { type: 'region' })
  : null;
function _countryName(cc) {
  if (!cc) return null;
  if (!_countryNamer) return cc;
  try { return _countryNamer.of(cc.toUpperCase()) || cc; }
  catch (_) { return cc; }
}
function _fmtLocation(geo) {
  if (!geo || !geo.country_code) return '—';
  const parts = [];
  if (geo.city) parts.push(geo.city);
  parts.push(_countryName(geo.country_code));
  return parts.join(', ');
}

async function loadClients() {
  const tbody = document.getElementById('clients-body');
  const hint = document.getElementById('clients-hint');
  const empty = document.getElementById('clients-empty');
  tbody.innerHTML = '';
  let data;
  try { data = await api('GET', '/clients'); }
  catch (e) { toast('Load failed: ' + e.message, 'err'); return; }

  if (data.geoip_db_present) {
    const which = [];
    if (data.geoip_ipv4_present) which.push('IPv4');
    if (data.geoip_ipv6_present) which.push('IPv6');
    hint.textContent = data.count + ' client(s) — geo enrichment active ('
      + which.join(' + ') + '). Private/Docker IPs do not geolocate.';
  } else {
    hint.textContent = data.count + ' client(s) — geo DB not loaded (drop '
      + 'geolite2-city-ipv4.mmdb / -ipv6.mmdb into agent_server/data/geoip/).';
  }
  empty.hidden = (data.count > 0);

  for (const c of data.clients) {
    const tr = document.createElement('tr');
    const add = (txt) => {
      const td = document.createElement('td');
      td.textContent = txt;
      tr.appendChild(td);
    };
    const kindTd = document.createElement('td');
    kindTd.appendChild(badge(c.kind, c.kind === 'socket' ? 'ok' : 'muted'));
    tr.appendChild(kindTd);
    add(c.client_id || c.id || '—');
    add(c.ip || '—');
    // Location: flag SVG (best-effort; falls back to text if no asset) + text.
    const locTd = document.createElement('td');
    const cc = c.geo && c.geo.country_code;
    if (cc) {
      const img = document.createElement('img');
      img.src = 'flags/' + cc.toLowerCase() + '.svg';
      img.alt = cc;
      img.className = 'flag';
      img.onerror = () => img.remove();
      locTd.appendChild(img);
      locTd.appendChild(document.createTextNode(' '));
    }
    locTd.appendChild(document.createTextNode(_fmtLocation(c.geo)));
    tr.appendChild(locTd);
    add(c.calls == null ? '—' : String(c.calls));
    add(_fmtDuration(c.connected_for_s));
    add(_fmtDuration(c.idle_for_s));
    tbody.appendChild(tr);
  }
}

// --------------------------------------------------------------------------
// Wire up
// --------------------------------------------------------------------------
document.getElementById('tab-dashboard').onclick = () => showTab('dashboard');
document.getElementById('tab-agents').onclick = () => showTab('agents');
document.getElementById('tab-clients').onclick = () => showTab('clients');
document.getElementById('btn-clients-refresh').onclick = loadClients;
document.getElementById('tab-config').onclick = () => showTab('config');
document.getElementById('btn-activate').onclick = activateSelected;
document.getElementById('btn-logs').onclick = toggleLogs;
document.getElementById('mem-modal-close').onclick =
  () => { document.getElementById('mem-modal').hidden = true; };
document.getElementById('btn-new').onclick = newAgent;
document.getElementById('btn-cancel').onclick = cancelForm;
document.getElementById('btn-delete').onclick = deleteAgent;
document.getElementById('agent-form').onsubmit = saveAgent;
document.getElementById('btn-test-send').onclick = sendTest;
document.getElementById('btn-test-reset').onclick = resetTest;
document.getElementById('btn-config-save').onclick = saveConfig;
document.getElementById('btn-config-reload').onclick = loadConfig;

// Enter sends in the tester; Shift+Enter inserts a newline.
document.getElementById('test-msg').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendTest();
  }
});

showTab('dashboard');
