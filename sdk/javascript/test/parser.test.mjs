// Node test for the JS parser — mirrors the Python suite so both SDKs agree.
// Run: node test/parser.test.mjs

import { StreamParser, splitResponse, sanitizeForTTS } from '../src/parser.js';

let pass = 0, fail = 0;
function ok(cond, name) { if (cond) { pass++; } else { fail++; console.log('FAIL', name); } }
function eq(a, b, name) { ok(JSON.stringify(a) === JSON.stringify(b), `${name}  (${JSON.stringify(a)} !== ${JSON.stringify(b)})`); }

function channels(text, chunk) {
  const p = new StreamParser();
  const out = { thinking: '', voice: '', answer: '' };
  const push = (evs) => evs.forEach((e) => { out[e.kind] += e.text; });
  for (let i = 0; i < text.length; i += chunk) push(p.feed(text.slice(i, i + chunk)));
  push(p.flush());
  return out;
}

// splitResponse
eq(splitResponse('<think>r</think><voice>v</voice>body.'), { thinking: 'r', voice: 'v', answer: 'body.' }, 'split complete');
eq(splitResponse('plain answer'), { thinking: '', voice: '', answer: 'plain answer' }, 'split none');

// streaming at many chunk sizes
const text = '<think>thinking</think><voice>say this</voice>Visible answer with detail.';
for (const c of [1, 2, 3, 7, 1000]) {
  eq(channels(text, c), { thinking: 'thinking', voice: 'say this', answer: 'Visible answer with detail.' }, `stream chunk=${c}`);
}

// voice final marker
{
  const p = new StreamParser();
  let finals = 0;
  const all = [...p.feed('<voice>hi</voice>body'), ...p.flush()];
  all.forEach((e) => { if (e.kind === 'voice' && e.final) finals++; });
  eq(finals, 1, 'one final voice event');
}

// unclosed voice -> answer, never spoken
{
  const ch = channels('<voice>this never closes and runs into the body', 1);
  eq(ch.voice, '', 'unclosed voice committed nothing to voice channel');
  ok(ch.answer.includes('runs into the body'), 'unclosed voice surfaced as answer');
}

// partial tag across boundary
{
  const p = new StreamParser();
  const evs = [...p.feed('Answer <voi'), ...p.feed('ce>spoken</voice>more'), ...p.flush()];
  const out = { thinking: '', voice: '', answer: '' };
  evs.forEach((e) => { out[e.kind] += e.text; });
  ok(!out.answer.includes('<voi'), 'no partial tag leaked into answer');
  eq(out.voice, 'spoken', 'voice extracted across boundary');
  eq(out.answer, 'Answer more', 'answer reassembled across boundary');
}

// sanitiser: malformed closers (the "slash voice" class of bug)
eq(sanitizeForTTS('António was CTO.</voice>'), 'António was CTO.', 'sanitize exact closer');
eq(sanitizeForTTS('António was CTO.</ voice>'), 'António was CTO.', 'sanitize spaced closer');
eq(sanitizeForTTS('António was CTO.</voice >'), 'António was CTO.', 'sanitize trailing-space closer');
eq(sanitizeForTTS('António was CTO.</voice'), 'António was CTO.', 'sanitize truncated closer');
eq(sanitizeForTTS('He filed every invoice on time.'), 'He filed every invoice on time.', 'sanitize keeps "invoice"');
ok(!sanitizeForTTS('<think>secret</think>Answer [markdown_chunk:ab12] here.').includes('secret'), 'sanitize drops think + citation');

// fuzz random chunking
{
  const t = '<think>r</think><voice>v</voice>answer-text';
  let seed = 7; const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let n = 0; n < 50; n++) {
    const p = new StreamParser();
    const out = { thinking: '', voice: '', answer: '' };
    let i = 0;
    while (i < t.length) { const k = 1 + Math.floor(rnd() * 5); p.feed(t.slice(i, i + k)).forEach((e) => { out[e.kind] += e.text; }); i += k; }
    p.flush().forEach((e) => { out[e.kind] += e.text; });
    eq(out, { thinking: 'r', voice: 'v', answer: 'answer-text' }, `fuzz ${n}`);
  }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
