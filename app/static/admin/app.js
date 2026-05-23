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
  document.getElementById('view-agents').hidden = which !== 'agents';
  document.getElementById('view-config').hidden = which !== 'config';
  document.getElementById('tab-agents').classList.toggle('active', which === 'agents');
  document.getElementById('tab-config').classList.toggle('active', which === 'config');
  if (which === 'config') loadConfig();
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
    tr.innerHTML = '<td class="a-name"></td><td class="a-mem"></td>';
    tr.querySelector('.a-name').textContent = a.name + (a.protected ? '  🔒' : '');
    tr.querySelector('.a-mem').textContent = a.memory_policy;
    tr.onclick = () => editAgent(a.name);
    tbody.appendChild(tr);
  }
}

async function editAgent(name) {
  let a;
  try { a = await api('GET', '/agents/' + encodeURIComponent(name)); }
  catch (e) { toast('Load failed: ' + e.message, 'err'); return; }
  editing = a.name;
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
// Wire up
// --------------------------------------------------------------------------
document.getElementById('tab-agents').onclick = () => showTab('agents');
document.getElementById('tab-config').onclick = () => showTab('config');
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

loadAgents();
