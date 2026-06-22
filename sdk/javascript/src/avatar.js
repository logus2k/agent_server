// Optional avatar integration (avatar_server, socket.io path /avatar/socket.io).
//
// FAIL SOFT: if the avatar service is absent, attach() resolves false and audio
// still plays through the browser (TTS in 'browser' mode). The avatar consumes
// the TTS audio server-side and streams muxed video chunks here, appended into
// a <video> via MediaSource.
//
// When the avatar is active, set the TtsClient to mode 'avatar_only' so the
// server routes audio to the avatar instead of the browser.

const DEFAULT_CODECS = [
  'video/mp4; codecs="avc1.42E01E, mp4a.40.2"',
  'video/webm; codecs="vp8, opus"',
];

export class AvatarClient {
  /**
   * @param {object} opts
   * @param {function} opts.io socket.io-client factory (required).
   * @param {string} opts.clientId stable id linking TTS/STT/avatar.
   * @param {string} [opts.url] origin (default '').
   * @param {string} [opts.path] default '/avatar/socket.io'.
   */
  constructor({ io, clientId, url = '', path = '/avatar/socket.io' } = {}) {
    if (typeof io !== 'function') throw new Error('AvatarClient needs an `io` factory');
    this._io = io;
    this.clientId = clientId;
    this.url = url; this.path = path;
    this.available = false;
    this.socket = null;
    this._video = null; this._ms = null; this._sb = null; this._buf = [];
  }

  /** Connect + bind to a <video> element. Resolves true if the avatar is up. */
  async attach(videoEl) {
    if (!('MediaSource' in window)) return false;
    this._video = videoEl;
    try {
      const ok = await new Promise((resolve) => {
        this.socket = this._io(this.url, {
          path: this.path, transports: ['websocket', 'polling'], forceNew: true,
          query: { client_id: this.clientId },
        });
        this.socket.once('connect', () => resolve(true));
        this.socket.once('connect_error', () => resolve(false));
      });
      if (!ok) { this._cleanup(); return false; }
      this.socket.on('avatar_init', (d) => this._init(d));
      this.socket.on('avatar_video_chunk', (d) => this._chunk(d));
      this.socket.on('avatar_reset', () => this._reset());
      this.available = true;
      return true;
    } catch { this._cleanup(); return false; }
  }

  disconnect() { this._cleanup(); }

  // -- internals --
  _pickCodec(hint) {
    const list = hint ? [hint, ...DEFAULT_CODECS] : DEFAULT_CODECS;
    return list.find((c) => window.MediaSource && MediaSource.isTypeSupported(c)) || null;
  }

  _init(d) {
    const codec = this._pickCodec(d && d.mime);
    if (!codec) return;
    this._ms = new MediaSource();
    this._video.src = URL.createObjectURL(this._ms);
    this._ms.addEventListener('sourceopen', () => {
      try {
        this._sb = this._ms.addSourceBuffer(codec);
        this._sb.addEventListener('updateend', () => this._pump());
        this._pump();
      } catch { /* unsupported */ }
    }, { once: true });
  }

  _chunk(d) {
    const buf = d && (d.video_buffer || d.chunk || d);
    if (buf) { this._buf.push(buf instanceof ArrayBuffer ? buf : (buf.buffer || buf)); this._pump(); }
  }

  _pump() {
    if (!this._sb || this._sb.updating || !this._buf.length) return;
    try { this._sb.appendBuffer(this._buf.shift()); } catch { /* buffer full / closed */ }
  }

  _reset() {
    this._buf = [];
    try { if (this._sb && this._ms && this._ms.readyState === 'open') this._sb.abort(); } catch { /* ignore */ }
  }

  _cleanup() {
    this.available = false;
    try { if (this.socket) this.socket.disconnect(); } catch { /* ignore */ }
    this.socket = null; this._ms = null; this._sb = null; this._buf = [];
  }
}
