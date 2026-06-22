// Optional STT integration (stt_server, socket.io path /stt/socket.io).
//
// FAIL SOFT: if the STT service or the microphone is unavailable, start()
// resolves false and nothing else breaks — the user can still type.
//
// Captures mic audio, downsamples to 16 kHz PCM16, and emits `audio_data`;
// receives `transcription` (final) and `transcription_partial`. A simple
// energy gate avoids streaming silence (skip with { gate:false }).

export class SttClient {
  /**
   * @param {object} opts
   * @param {function} opts.io socket.io-client factory (required).
   * @param {string} opts.clientId stable id linking TTS/STT/avatar.
   * @param {string} [opts.url] origin (default '').
   * @param {string} [opts.path] default '/stt/socket.io'.
   * @param {boolean} [opts.gate] energy-gate sends (default true).
   */
  constructor({ io, clientId, url = '', path = '/stt/socket.io', gate = true } = {}) {
    if (typeof io !== 'function') throw new Error('SttClient needs an `io` factory');
    this._io = io;
    this.clientId = clientId;
    this.url = url; this.path = path; this.gate = gate;
    this.available = false;
    this.listening = false;
    this.socket = null;
    this.onTranscript = null;        // (text, { final }) => void
    this._media = null; this._ctx = null; this._node = null; this._src = null;
  }

  /** Connect + open mic. Resolves true if listening, false if unavailable. */
  async start() {
    try {
      const ok = await new Promise((resolve) => {
        this.socket = this._io(this.url, { path: this.path, transports: ['websocket', 'polling'], forceNew: true });
        this.socket.once('connect', () => resolve(true));
        this.socket.once('connect_error', () => resolve(false));
      });
      if (!ok) { this._cleanup(); return false; }
      this.socket.on('transcription', (d) => this._onText(d, true));
      this.socket.on('transcription_partial', (d) => this._onText(d, false));
      this.socket.emit('register_audio_client', { main_client_id: this.clientId, connection_type: 'browser' });

      this._media = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      this._ctx = new (window.AudioContext || window.webkitAudioContext)();
      this._src = this._ctx.createMediaStreamSource(this._media);
      // ScriptProcessor is deprecated but needs no worklet file — fine for a
      // reference SDK. Swap for an AudioWorklet in production if you prefer.
      this._node = this._ctx.createScriptProcessor(4096, 1, 1);
      const inRate = this._ctx.sampleRate;
      this._node.onaudioprocess = (e) => this._onAudio(e.inputBuffer.getChannelData(0), inRate);
      this._src.connect(this._node);
      this._node.connect(this._ctx.destination);
      this.available = true; this.listening = true;
      return true;
    } catch { this._cleanup(); return false; }
  }

  /** Pause/resume sending without tearing down the socket. */
  setListening(on) { this.listening = !!on; }

  stop() { this._cleanup(); }

  // -- internals --
  _onText(d, final) {
    const text = (d && (d.text || d.transcript)) || (typeof d === 'string' ? d : '');
    if (text && this.onTranscript) this.onTranscript(text, { final });
  }

  _onAudio(float32, inRate) {
    if (!this.listening || !this.socket || !this.socket.connected) return;
    // Simple energy gate.
    if (this.gate) {
      let sum = 0;
      for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
      if (Math.sqrt(sum / float32.length) < 0.008) return;
    }
    const pcm = this._downsampleTo16kPCM16(float32, inRate);
    if (pcm) this.socket.emit('audio_data', { client_id: this.clientId, audio: pcm });
  }

  _downsampleTo16kPCM16(input, inRate) {
    const outRate = 16000;
    const ratio = inRate / outRate;
    const outLen = Math.floor(input.length / ratio);
    const out = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const s = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out.buffer;
  }

  _cleanup() {
    this.available = false; this.listening = false;
    try { if (this._node) this._node.disconnect(); } catch { /* ignore */ }
    try { if (this._src) this._src.disconnect(); } catch { /* ignore */ }
    try { if (this._media) this._media.getTracks().forEach((t) => t.stop()); } catch { /* ignore */ }
    try { if (this.socket) this.socket.disconnect(); } catch { /* ignore */ }
    this.socket = null; this._node = null; this._src = null; this._media = null;
  }
}
