// Interactive client over agent_server's Socket.IO interface.
// Chat -> RunStarted / ChatChunk / ChatDone / Error / Interrupted, with each
// ChatChunk piped through the channel parser so callers get answer/thinking/
// voice events instead of raw tagged text.
//
// The socket.io client (`io`) is injected so this SDK bundles no transport —
// pass the vendored/imported `io` from your page.

import { StreamParser } from './parser.js';

export class AgentSocketClient {
  /**
   * @param {object} opts
   * @param {function} opts.io  socket.io-client factory (required).
   * @param {string} [opts.url] agent_server origin (default same-origin '').
   * @param {string} [opts.path] socket.io path (default '/socket.io').
   */
  constructor({ io, url = '', path = '/socket.io' } = {}) {
    if (typeof io !== 'function') throw new Error('AgentSocketClient needs an `io` factory');
    this._io = io;
    this.url = url;
    this.path = path;
    this.socket = null;
    this._parser = null;
    this._handlers = {};
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.socket = this._io(this.url, { path: this.path, transports: ['websocket', 'polling'], forceNew: true });
      this.socket.once('connect', () => resolve(true));
      this.socket.once('connect_error', (e) => reject(e));
      this.socket.on('RunStarted', (d) => this._emit('runStarted', d));
      this.socket.on('ChatChunk', (d) => {
        if (!this._parser) return;
        for (const ev of this._parser.feed((d && d.chunk) || '')) this._dispatch(ev);
      });
      this.socket.on('ChatDone', (d) => {
        if (this._parser) for (const ev of this._parser.flush()) this._dispatch(ev);
        this._emit('done', d);
      });
      this.socket.on('Interrupted', (d) => {
        if (this._parser) for (const ev of this._parser.flush()) this._dispatch(ev);
        this._emit('interrupted', d);
      });
      this.socket.on('Error', (d) => this._emit('error', d));
      this.socket.on('UserTranscript', (d) => this._emit('userTranscript', d));
    });
  }

  get connected() { return !!(this.socket && this.socket.connected); }

  disconnect() { if (this.socket) try { this.socket.disconnect(); } catch { /* ignore */ } }

  /**
   * Send a turn. `handlers`: { onAnswer, onThinking, onVoice, onDone, onError }.
   * Thinking on/off over Socket.IO is governed by the agent PRESET; use the
   * REST client's `thinking=` for per-request control.
   */
  chat(agent, text, { threadId, memory } = {}, handlers = {}) {
    this._handlers = handlers;
    this._parser = new StreamParser();
    const payload = { agent, text };
    if (threadId) payload.thread_id = threadId;
    if (memory !== undefined) payload.memory = memory;
    this.socket.emit('Chat', payload);
  }

  interrupt() { if (this.socket) this.socket.emit('Interrupt'); }

  on(event, fn) { (this._listeners ||= {})[event] = fn; return this; }

  _dispatch(ev) {
    const h = this._handlers;
    if (ev.kind === 'answer' && h.onAnswer) h.onAnswer(ev.text);
    else if (ev.kind === 'thinking' && h.onThinking) h.onThinking(ev.text, ev.final);
    else if (ev.kind === 'voice' && h.onVoice && ev.final) h.onVoice(ev.text);
  }

  _emit(name, data) {
    if (name === 'done' && this._handlers.onDone) this._handlers.onDone(data);
    if (name === 'error' && this._handlers.onError) this._handlers.onError(data);
    if (this._listeners && this._listeners[name]) this._listeners[name](data);
  }
}
