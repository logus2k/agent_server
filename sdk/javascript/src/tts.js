// Optional TTS integration (tts_server, socket.io path /tts/socket.io).
//
// FAIL SOFT: if the TTS service is absent, connect() resolves false and every
// method becomes a no-op — chat and everything else keep working. The text you
// pass to speak() is ALWAYS run through sanitizeForTTS so a stray <voice> tag
// can never be spoken.

import { sanitizeForTTS } from './parser.js';

export class TtsClient {
  /**
   * @param {object} opts
   * @param {function} opts.io socket.io-client factory (required).
   * @param {string} opts.clientId stable id linking TTS/STT/avatar.
   * @param {string} [opts.url] origin (default same-origin '').
   * @param {string} [opts.path] default '/tts/socket.io'.
   * @param {string} [opts.voice] default voice profile.
   * @param {number} [opts.speed] default 1.0.
   */
  constructor({ io, clientId, url = '', path = '/tts/socket.io', voice = 'af_heart', speed = 1.0 } = {}) {
    if (typeof io !== 'function') throw new Error('TtsClient needs an `io` factory');
    this._io = io;
    this.clientId = clientId;
    this.url = url; this.path = path;
    this.voice = voice; this.speed = speed;
    this.available = false;
    this.socket = null;
    this._ctx = null;
    this._queue = Promise.resolve();
    this._barged = false;
    this._mode = 'browser';        // 'browser' (play here) or 'avatar_only'
    this.onSpeakingChange = null;   // (bool) => void
  }

  /** Connect + register. Resolves true if voice is available, false otherwise. */
  async connect() {
    try {
      const ok = await new Promise((resolve) => {
        this.socket = this._io(this.url, {
          path: this.path, transports: ['websocket', 'polling'], forceNew: true,
          query: { client_id: this.clientId, format: 'binary' },
        });
        this.socket.once('connect', () => resolve(true));
        this.socket.once('connect_error', () => resolve(false));
      });
      if (!ok) { this._cleanup(); return false; }
      this.socket.emit('register_audio_client', {
        main_client_id: this.clientId, connection_type: 'browser',
        mode: this._mode, format: 'binary', voice: this.voice, speed: this.speed,
      });
      this.socket.emit('tts_configure_client', { client_id: this.clientId, voice: this.voice, speed: this.speed });
      this.socket.emit('set_client_mode', { mode: this._mode, client_id: this.clientId });
      this.socket.on('tts_audio_chunk', (e) => this._onChunk(e));
      this.socket.on('tts_stop_immediate', () => this._stopLocal());
      this.available = true;
      return true;
    } catch { this._cleanup(); return false; }
  }

  /** Speak text (sanitised). No-op if TTS unavailable. */
  speak(text) {
    if (!this.available || !this.socket) return;
    const clean = sanitizeForTTS(text);
    if (!clean) return;
    this._barged = false;
    this.socket.emit('tts_text_chunk', { client_id: this.clientId, text: clean });
  }

  /** Route audio to the avatar ('avatar_only') or back to the browser ('browser'). */
  setMode(mode) {
    this._mode = mode;
    if (this.socket) this.socket.emit('set_client_mode', { mode, client_id: this.clientId });
  }

  configure({ voice, speed } = {}) {
    if (voice !== undefined) this.voice = voice;
    if (speed !== undefined) this.speed = speed;
    if (this.socket) this.socket.emit('tts_configure_client', { client_id: this.clientId, voice: this.voice, speed: this.speed });
  }

  /** Barge-in: stop current + future audio for this turn. */
  stop() {
    this._barged = true;
    if (this.socket) this.socket.emit('stop_generation', { client_id: this.clientId });
    this._stopLocal();
  }

  disconnect() { this._cleanup(); }

  // -- internals --
  _onChunk(evt) {
    if (this._mode === 'avatar_only') return;       // audio goes to the avatar
    const buf = evt && evt.audio_buffer;
    if (!buf || this._barged) return;
    if (!this._ctx) this._ctx = new (window.AudioContext || window.webkitAudioContext)();
    let ab;
    if (buf instanceof ArrayBuffer) ab = buf.slice(0);
    else if (buf && buf.buffer) ab = buf.buffer.slice(0);
    else return;
    this._ctx.decodeAudioData(ab).then((audioBuf) => {
      if (this._barged || !this._ctx) return;
      if (this.onSpeakingChange) this.onSpeakingChange(true);
      this._queue = this._queue.then(() => new Promise((res) => {
        if (this._barged || !this._ctx) return res();
        const src = this._ctx.createBufferSource();
        src.buffer = audioBuf;
        src.connect(this._ctx.destination);
        this._cur = src;
        src.onended = () => { if (this.onSpeakingChange) this.onSpeakingChange(false); res(); };
        src.start();
      }));
    }).catch(() => {});
  }

  _stopLocal() {
    this._barged = true;
    try { if (this._cur) this._cur.stop(); } catch { /* ignore */ }
    this._queue = Promise.resolve();
    if (this.onSpeakingChange) this.onSpeakingChange(false);
  }

  _cleanup() {
    this.available = false;
    try { if (this.socket) this.socket.disconnect(); } catch { /* ignore */ }
    this.socket = null;
  }
}
