// Streaming + whole-response parser for agent_server output.
//
// agent_server folds reasoning and the spoken-summary into the content stream
// as literal tags:
//
//   <think>...reasoning...</think><voice>...spoken...</voice>answer body
//
// This is the one tested parser so no client re-implements it (a hand-rolled
// version once read a whole answer aloud + spoke a literal "</voice>" tag).
//
// Semantics, identical to the Python SDK:
//   * thinking streams incrementally (clients show it live);
//   * voice is BUFFERED until </voice> and emitted once (final) — a partial or
//     forgotten voice block is therefore NEVER handed to TTS;
//   * answer streams incrementally, holding back a partial tag at a boundary.
//
// sanitizeForTTS() is whitespace/format tolerant: it strips even malformed or
// truncated tags so a stray tag can never be spoken.

const TAGS = ['<think>', '</think>', '<voice>', '</voice>'];
const MAX_TAG = '</voice >'.length;

const RE = {
  openThink: /<\s*think\s*>/i,
  openVoice: /<\s*voice\s*>/i,
  closeThink: /<\s*\/\s*think\s*>/i,
  closeVoice: /<\s*\/\s*voice\s*>/i,
  thinkBlock: /<\s*think\s*>[\s\S]*?<\s*\/\s*think\s*>/i,
  voiceBlock: /<\s*voice\s*>([\s\S]*?)<\s*\/\s*voice\s*>/i,
};

/** Strip text that must never reach TTS: malformed/truncated think|voice tags
 *  plus markdown/citation noise. */
export function sanitizeForTTS(text) {
  return String(text)
    .replace(/<\s*think\s*>[\s\S]*?<\s*\/\s*think\s*>/gi, '')
    .replace(/<\s*think\s*>[\s\S]*$/i, '')
    .replace(/<\s*voice\s*>[\s\S]*?<\s*\/\s*voice\s*>/gi, '')
    .replace(/<\s*voice\s*>[\s\S]*$/i, '')
    .replace(/<\s*\/?\s*(?:think|voice)\s*>/gi, '')        // bare/spaced tags
    .replace(/<\s*\/?\s*(?:think|voice)\b[^>]*$/i, '')     // truncated at end
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\[(?:markdown_chunk|E|R):[^\]]+\]/g, '')
    .replace(/\[C\d+\]/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/^\s*#{1,6}\s+/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\n{2,}/g, '. ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Split a COMPLETE content string into { thinking, voice, answer }. */
export function splitResponse(content) {
  content = content || '';
  let thinking = '';
  const mt = content.match(RE.thinkBlock);
  if (mt) thinking = mt[0].replace(/^<\s*think\s*>/i, '').replace(/<\s*\/\s*think\s*>$/i, '').trim();
  let voice = '';
  const mv = content.match(RE.voiceBlock);
  if (mv) voice = mv[1].trim();
  let answer = content
    .replace(/<\s*think\s*>[\s\S]*?<\s*\/\s*think\s*>/gi, '')
    .replace(/<\s*voice\s*>[\s\S]*?<\s*\/\s*voice\s*>/gi, '')
    .replace(/<\s*think\s*>[\s\S]*$/i, '')
    .replace(/<\s*voice\s*>[\s\S]*$/i, '')
    .trim();
  return { thinking, voice, answer };
}

function isPartialTag(frag) {
  const f = frag.replace(/\s/g, '').toLowerCase();
  return TAGS.some((t) => t.startsWith(f) && f !== t);
}

/**
 * Incremental parser. `feed(delta)` returns an array of events; each event is
 * `{ kind: 'thinking'|'voice'|'answer', text, final }`. Call `flush()` when the
 * stream ends. A `voice` event with `final:true` is the complete spoken summary
 * (safe to TTS via sanitizeForTTS).
 */
export class StreamParser {
  constructor() {
    this._buf = '';
    this._mode = 'answer';
    this._voicePending = '';
    this.thinkingText = '';
    this.voiceText = '';
    this.answerText = '';
  }

  feed(delta) {
    this._buf += delta || '';
    const events = [];
    while (true) {
      let progressed = false;
      if (this._mode === 'answer') progressed = this._scanAnswer(events);
      else if (this._mode === 'thinking') progressed = this._scanThinking(events);
      else if (this._mode === 'voice') progressed = this._scanVoice(events);
      if (!progressed) break;
    }
    return events;
  }

  flush() {
    const events = [];
    if (this._mode === 'thinking') {
      if (this._buf) { this.thinkingText += this._buf; events.push({ kind: 'thinking', text: this._buf, final: false }); }
    } else if (this._mode === 'voice') {
      // Forgotten closer: surface held-back voice as ANSWER, never spoken.
      const leftover = this._voicePending + this._buf;
      this._voicePending = '';
      if (leftover) { this.answerText += leftover; events.push({ kind: 'answer', text: leftover, final: false }); }
    } else if (this._buf) {
      this.answerText += this._buf; events.push({ kind: 'answer', text: this._buf, final: false });
    }
    this._buf = '';
    this._mode = 'answer';
    return events;
  }

  _scanAnswer(events) {
    const ot = this._buf.match(RE.openThink);
    const ov = this._buf.match(RE.openVoice);
    let next = null;
    if (ot && ov) next = ot.index <= ov.index ? ot : ov;
    else next = ot || ov;
    if (!next) return this._emitSafePrefix(events, 'answer');
    const before = this._buf.slice(0, next.index);
    if (before) { this.answerText += before; events.push({ kind: 'answer', text: before, final: false }); }
    this._mode = next === ot ? 'thinking' : 'voice';
    this._buf = this._buf.slice(next.index + next[0].length);
    return true;
  }

  _scanThinking(events) {
    const m = this._buf.match(RE.closeThink);
    if (!m) return this._emitSafePrefix(events, 'thinking');
    const inner = this._buf.slice(0, m.index);
    if (inner) { this.thinkingText += inner; events.push({ kind: 'thinking', text: inner, final: false }); }
    events.push({ kind: 'thinking', text: '', final: true });
    this._buf = this._buf.slice(m.index + m[0].length);
    this._mode = 'answer';
    return true;
  }

  _scanVoice(events) {
    const m = this._buf.match(RE.closeVoice);
    if (!m) {
      // Buffer everything except a possible partial closer at the tail.
      let keep = 0;
      const from = this._buf.length > MAX_TAG ? this._buf.slice(this._buf.length - MAX_TAG) : this._buf;
      const lt = from.lastIndexOf('<');
      if (lt !== -1 && isPartialTag(from.slice(lt))) keep = from.length - lt;
      const move = keep === 0 ? this._buf : this._buf.slice(0, this._buf.length - keep);
      if (move) { this._voicePending += move; this._buf = this._buf.slice(move.length); }
      return false;
    }
    const full = this._voicePending + this._buf.slice(0, m.index);
    this._voicePending = '';
    this.voiceText += full;
    events.push({ kind: 'voice', text: full, final: true });
    this._buf = this._buf.slice(m.index + m[0].length);
    this._mode = 'answer';
    return true;
  }

  _emitSafePrefix(events, kind) {
    if (!this._buf) return false;
    let keep = 0;
    const from = this._buf.length > MAX_TAG ? this._buf.slice(this._buf.length - MAX_TAG) : this._buf;
    const lt = from.lastIndexOf('<');
    if (lt !== -1 && isPartialTag(from.slice(lt))) keep = from.length - lt;
    const emit = keep === 0 ? this._buf : this._buf.slice(0, this._buf.length - keep);
    if (emit) {
      if (kind === 'thinking') this.thinkingText += emit;
      else if (kind === 'voice') this.voiceText += emit;
      else this.answerText += emit;
      events.push({ kind, text: emit, final: false });
      this._buf = this._buf.slice(emit.length);
    }
    return false;
  }
}
