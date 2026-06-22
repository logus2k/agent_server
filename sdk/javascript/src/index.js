// agent-server-client — client-side (browser) SDK for agent_server.
//
// Exposes the tested channel parser + the REST and Socket.IO chat clients, plus
// OPTIONAL, fail-soft TTS / STT / avatar integrations. Each optional service is
// independent: if it's down, that one feature degrades and everything else
// keeps working.

export { StreamParser, splitResponse, sanitizeForTTS } from './parser.js';
export { AgentRestClient, thinkingKwargs } from './rest.js';
export { AgentSocketClient } from './socket.js';
export { TtsClient } from './tts.js';
export { SttClient } from './stt.js';
export { AvatarClient } from './avatar.js';

import { AgentRestClient } from './rest.js';
import { TtsClient } from './tts.js';
import { SttClient } from './stt.js';
import { AvatarClient } from './avatar.js';

/** Random stable client id linking TTS/STT/avatar for one browser session. */
export function newClientId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return 'cvc-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

/**
 * Convenience facade that wires chat + the optional voice/avatar services and
 * reports which are actually available. Build only what you pass `io` for.
 *
 *   const c = new AgentClient({ io, baseUrl: '', clientId: newClientId() });
 *   await c.enableVoice();           // tries TTS; false if unavailable
 *   await c.enableMic(t => ...);     // tries STT; false if unavailable
 *   await c.enableAvatar(videoEl);   // tries avatar; false if unavailable
 *   c.ask('cv_assistant_e2b', 'Hi', { thinking:true }, { onAnswer, onVoice });
 */
export class AgentClient {
  constructor({ io = null, baseUrl = '', apiKey = null, clientId = null,
                ttsPath, sttPath, avatarPath, voice, speed } = {}) {
    this.io = io;
    this.clientId = clientId || newClientId();
    this.rest = new AgentRestClient({ baseUrl, apiKey });
    this._ttsOpts = { io, clientId: this.clientId, url: baseUrl, path: ttsPath, voice, speed };
    this._sttOpts = { io, clientId: this.clientId, url: baseUrl, path: sttPath };
    this._avatarOpts = { io, clientId: this.clientId, url: baseUrl, path: avatarPath };
    this.tts = null; this.stt = null; this.avatar = null;
    this.capabilities = { chat: true, tts: false, stt: false, avatar: false };
  }

  /** Try to enable spoken replies. Returns true if TTS is available. */
  async enableVoice() {
    if (!this.io) return false;
    this.tts = new TtsClient(this._cleanOpts(this._ttsOpts));
    const ok = await this.tts.connect();
    this.capabilities.tts = ok;
    if (!ok) this.tts = null;
    return ok;
  }

  /** Try to enable mic input. `onTranscript(text,{final})`. Returns availability. */
  async enableMic(onTranscript) {
    if (!this.io) return false;
    this.stt = new SttClient(this._cleanOpts(this._sttOpts));
    this.stt.onTranscript = onTranscript;
    const ok = await this.stt.start();
    this.capabilities.stt = ok;
    if (!ok) this.stt = null;
    return ok;
  }

  /** Try to enable the avatar bound to a <video>. Returns availability. */
  async enableAvatar(videoEl) {
    if (!this.io) return false;
    this.avatar = new AvatarClient(this._cleanOpts(this._avatarOpts));
    const ok = await this.avatar.attach(videoEl);
    this.capabilities.avatar = ok;
    if (ok && this.tts) this.tts.setMode('avatar_only');   // route audio to avatar
    if (!ok) this.avatar = null;
    return ok;
  }

  /** Ask via REST streaming. Voice is auto-spoken if TTS is enabled. */
  ask(model, messages, opts = {}, handlers = {}) {
    const wrapped = { ...handlers };
    if (this.tts) {
      const userVoice = handlers.onVoice;
      wrapped.onVoice = (t) => { this.tts.speak(t); if (userVoice) userVoice(t); };
    }
    return this.rest.chatStream(model, messages, opts, wrapped);
  }

  bargeIn() { if (this.tts) this.tts.stop(); }

  _cleanOpts(o) {
    const out = {};
    for (const k of Object.keys(o)) if (o[k] !== undefined) out[k] = o[k];
    return out;
  }
}
