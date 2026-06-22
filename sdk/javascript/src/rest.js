// REST client for agent_server's OpenAI-compatible API (browser fetch).
// Streaming SSE + non-streaming, thinking toggle, discovery, active-model.

import { StreamParser, splitResponse } from './parser.js';

// Families whose chat template uses a non-default kwarg to toggle reasoning.
const THINKING_KWARG = { granite: 'thinking' };
const THINKING_UNSUPPORTED = new Set(['ministral']);

/** Build the chat_template_kwargs that toggle reasoning for a model family. */
export function thinkingKwargs(value, family = '') {
  if (value === null || value === undefined) return {};
  const fam = String(family).toLowerCase();
  if (THINKING_UNSUPPORTED.has(fam)) return {};
  const key = THINKING_KWARG[fam] || 'enable_thinking';
  return { [key]: !!value };
}

export class AgentRestClient {
  /**
   * @param {object} opts
   * @param {string} [opts.baseUrl] agent_server origin (default same-origin '').
   * @param {string} [opts.apiKey] Bearer token (only if the server sets one).
   */
  constructor({ baseUrl = '', apiKey = null } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
  }

  _headers() {
    const h = { 'Content-Type': 'application/json' };
    if (this.apiKey) h.Authorization = `Bearer ${this.apiKey}`;
    return h;
  }

  _payload(model, messages, { stream, thinking, thinkingFamily, sampling, extraBody }) {
    const msgs = typeof messages === 'string' ? [{ role: 'user', content: messages }] : messages;
    const p = { model, messages: msgs, stream: !!stream, ...(sampling || {}) };
    const ctk = { ...(p.chat_template_kwargs || {}), ...thinkingKwargs(thinking, thinkingFamily) };
    if (Object.keys(ctk).length) p.chat_template_kwargs = ctk;
    return { ...p, ...(extraBody || {}) };
  }

  /** Non-streaming completion. Resolves to { answer, thinking, voice, raw, usage }. */
  async chat(model, messages, opts = {}) {
    const body = this._payload(model, messages, { stream: false, ...opts });
    const resp = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: 'POST', headers: this._headers(), body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`agent_server ${resp.status}: ${await resp.text()}`);
    const data = await resp.json();
    const msg = (data.choices && data.choices[0] && data.choices[0].message) || {};
    let content = msg.content || '';
    if (msg.reasoning_content && content.indexOf('<think>') < 0) {
      content = `<think>${msg.reasoning_content}</think>${content}`;
    }
    return { ...splitResponse(content), raw: content, model: data.model || model, usage: data.usage || {} };
  }

  /**
   * Streaming completion. Calls handlers as channels arrive.
   * @param {object} handlers { onAnswer(text), onThinking(text), onVoice(text), onDone(), onError(err) }
   * @returns {{ abort: () => void }} controller to cancel the stream.
   */
  chatStream(model, messages, opts = {}, handlers = {}) {
    const ctrl = new AbortController();
    const body = this._payload(model, messages, { stream: true, ...opts });
    (async () => {
      const parser = new StreamParser();
      const dispatch = (ev) => {
        if (ev.kind === 'answer' && handlers.onAnswer) handlers.onAnswer(ev.text);
        else if (ev.kind === 'thinking' && handlers.onThinking) handlers.onThinking(ev.text, ev.final);
        else if (ev.kind === 'voice' && handlers.onVoice && ev.final) handlers.onVoice(ev.text);
      };
      try {
        const resp = await fetch(`${this.baseUrl}/v1/chat/completions`, {
          method: 'POST', headers: this._headers(), body: JSON.stringify(body), signal: ctrl.signal,
        });
        if (!resp.ok || !resp.body) throw new Error(`agent_server ${resp.status}`);
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop();
          for (const line of lines) {
            const t = line.trim();
            if (!t.startsWith('data:')) continue;
            const d = t.slice(5).trim();
            if (d === '[DONE]') { buf = ''; break; }
            let obj; try { obj = JSON.parse(d); } catch { continue; }
            const delta = (obj.choices && obj.choices[0] && obj.choices[0].delta) || {};
            let piece = delta.content;
            if (delta.reasoning_content && !piece) piece = `<think>${delta.reasoning_content}</think>`;
            if (!piece) continue;
            for (const ev of parser.feed(piece)) dispatch(ev);
          }
        }
        for (const ev of parser.flush()) dispatch(ev);
        if (handlers.onDone) handlers.onDone();
      } catch (err) {
        if (err.name === 'AbortError') { if (handlers.onDone) handlers.onDone(); return; }
        if (handlers.onError) handlers.onError(err); else throw err;
      }
    })();
    return { abort: () => ctrl.abort() };
  }

  /** All chat models (+active/family) and agent names. */
  async listModels() {
    const resp = await fetch(`${this.baseUrl}/v1/models`, { headers: this._headers() });
    if (!resp.ok) throw new Error(`agent_server ${resp.status}`);
    const data = await resp.json();
    return (data.data || []).map((m) => ({
      id: m.id, active: !!m.active, kind: m.kind || 'model', family: m.family || '', raw: m,
    }));
  }

  async activeModel() {
    return (await this.listModels()).find((m) => m.active) || null;
  }

  async getAgent(name) {
    const resp = await fetch(`${this.baseUrl}/v1/agents/${encodeURIComponent(name)}`, { headers: this._headers() });
    if (!resp.ok) throw new Error(`agent_server ${resp.status}`);
    return resp.json();
  }

  /** Switch the resident chat model. Optionally wait until back up (~30-45s). */
  async setActiveModel(modelId, { wait = true, timeoutMs = 90000 } = {}) {
    const resp = await fetch(`${this.baseUrl}/admin/api/active-model`, {
      method: 'POST', headers: this._headers(), body: JSON.stringify({ model_id: modelId }),
    });
    if (!resp.ok) throw new Error(`switch failed ${resp.status}: ${await resp.text()}`);
    const result = await resp.json();
    if (wait) await this.waitUntilReady({ expectActive: modelId, timeoutMs });
    return result;
  }

  async waitUntilReady({ expectActive = null, timeoutMs = 90000, pollMs = 2000 } = {}) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      try {
        const models = await this.listModels();
        if (!expectActive) return true;
        if (models.some((m) => m.active && m.id === expectActive)) return true;
      } catch { /* server restarting */ }
      await new Promise((r) => setTimeout(r, pollMs));
    }
    throw new Error(`agent_server not ready within ${timeoutMs}ms`);
  }
}
