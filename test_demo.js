// Headless functional test for the streaming demo.
// Stubs just enough DOM + Audio to drive the card state machine and assert that
// model outputs appear exactly when the audio reaches each chunk's end.
const fs = require('fs');
const vm = require('vm');
const path = require('path');

// The build is published as index.html but kept as streaming_160ms_demo.html in
// the development tree; accept either so this test runs in both.
const PAGE = [process.env.DEMO_HTML, 'streaming_160ms_demo.html', 'index.html']
  .filter(Boolean)
  .map(f => path.join(__dirname, f))
  .find(f => fs.existsSync(f));
if (!PAGE) throw new Error('no demo html found next to test_demo.js');
const html = fs.readFileSync(PAGE, 'utf8');
console.log(`testing ${path.basename(PAGE)}`);
let js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
js += "\nglobalThis.__T = {CASES, buildCard, render, tasks};\n";

// ---------- DOM stub ----------
function mkEl(tag) {
  const e = {
    tag, children: [], _cls: new Set(),
    style: { setProperty(k, v) { this[k] = v; } },
    dataset: {},
    title: '', textContent: '', scrollTop: 0, scrollHeight: 0,
    onclick: null, _parent: null, _html: '',
    appendChild(c) { c._parent = this; this.children.push(c); return c; },
    insertBefore(c, ref) {
      c._parent = this;
      const i = ref ? this.children.indexOf(ref) : -1;
      if (i < 0) this.children.push(c); else this.children.splice(i, 0, c);
      return c;
    },
    append(...cs) { cs.forEach(c => this.appendChild(c)); },
    remove() {
      if (this._parent) {
        const i = this._parent.children.indexOf(this);
        if (i >= 0) this._parent.children.splice(i, 1);
      }
    },
    querySelector() { return mkEl('div'); },
  };
  Object.defineProperty(e, 'className', {
    get() { return [...e._cls].join(' '); },
    set(v) { e._cls = new Set(String(v || '').split(/\s+/).filter(Boolean)); },
  });
  Object.defineProperty(e, 'innerHTML', {
    get() { return e._html; },
    set(v) { e._html = v; if (v === '') e.children = []; },
  });
  e.classList = {
    add: (...c) => c.forEach(x => e._cls.add(x)),
    remove: (...c) => c.forEach(x => e._cls.delete(x)),
    toggle: (c, f) => { const on = f === undefined ? !e._cls.has(c) : !!f; on ? e._cls.add(c) : e._cls.delete(c); return on; },
    contains: c => e._cls.has(c),
  };
  return e;
}
function walk(e, out = []) { out.push(e); e.children.forEach(c => walk(c, out)); return out; }
const has = (e, c) => e._cls.has(c);

const registry = {};
const doc = {
  createElement: mkEl,
  getElementById: id => (registry[id] = registry[id] || mkEl('div')),
};
const AUDIOS = [];
let pending = null;
const sandbox = {
  document: doc,
  Audio: function (src) {
    const a = {
      src, paused: true, ended: false, currentTime: 0, duration: 0, preload: '',
      onpause: null, onended: null,
      play() { a.paused = false; a.ended = false; return { then() {} }; },
      pause() { a.paused = true; if (a.onpause) a.onpause(); },
    };
    AUDIOS.push(a); return a;
  },
  requestAnimationFrame: cb => { pending = cb; return 1; },
  console, Math, Number, Set, Array, Object, JSON, String, Boolean, isNaN,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(js, sandbox);

const T = sandbox.__T;
let fails = 0;
const ok = (cond, msg) => { if (!cond) { fails++; console.log('  FAIL  ' + msg); } };

// ---------- drive each card ----------
for (const task of T.tasks) {
  const audioBase = AUDIOS.length;
  T.render(task);
  const cards = registry.grid.children;
  const want = T.CASES.filter(c => c.task === task);
  ok(cards.length === want.length, `${task}: card count ${cards.length} != ${want.length}`);

  cards.forEach((card, ci) => {
    const c = want[ci];
    pending = null;
    const all = walk(card);
    const play = all.find(e => has(e, 'play'));
    const logEl = all.find(e => e.tag === 'div' && has(e, 'log'));
    const cells = all.filter(e => e.tag === 'i');
    const gtEls = all.filter(e => has(e, 'gtpos'));
    const audio = AUDIOS[audioBase + ci];
    audio.duration = c.duration;

    ok(cells.length === c.nChunks, `${c.id}: cells ${cells.length} != nChunks ${c.nChunks}`);
    ok(gtEls.length === c.gt.length, `${c.id}: gtEls ${gtEls.length} != gt ${c.gt.length}`);
    ok(gtEls.every((g, i) => g._html.includes(c.gt[i].word)), `${c.id}: gt label text`);

    const rows = () => logEl.children.filter(e => has(e, 'row'));
    const spkCells = () => cells.filter(e => has(e, 'spk')).length;
    const hits = () => gtEls.filter(e => has(e, 'hit')).length;

    // ---- stream 1 (audio / ASR-like) and stream 2 (model output) ----
    const txA = all.find(e => has(e, 'tx-audio'));
    const txB = all.find(e => has(e, 'tx-out'));
    ok(!!txA && !!txB, `${c.id}: the two streams were not both created`);
    const wordEls = txA.children;
    const outEls = txB.children;
    const chips = () => outEls.filter(e => has(e, 'em'));
    const sils = () => outEls.filter(e => has(e, 'sil'));

    // stream 1 voices the whole sentence; only the queried targets are marked
    ok(wordEls.length === c.words.length,
      `${c.id}: words ${wordEls.length} != ${c.words.length}`);
    ok(wordEls.every((e, i) => e._html === c.words[i].w),
      `${c.id}: word text mismatch`);
    ok(wordEls.filter(e => has(e, 'tgt')).length === c.gt.length,
      `${c.id}: target words != gt count`);
    ok(c.words.filter(w => w.ev >= 0).length === c.gt.length,
      `${c.id}: events not all linked to a heard word`);
    ok(c.gt.every(ev => ev.spoken), `${c.id}: event missing spoken word`);

    // stream 2 is the real output: exactly one decision token per chunk
    ok(outEls.length === c.outTokens.length,
      `${c.id}: out tokens ${outEls.length} != ${c.outTokens.length}`);
    ok(outEls.length === c.nChunks,
      `${c.id}: out tokens ${outEls.length} != chunks ${c.nChunks}`);
    ok(sils().length === c.nChunks - c.speaks.length,
      `${c.id}: silence tokens ${sils().length} != chunks-emissions ${c.nChunks - c.speaks.length}`);
    ok(chips().length === c.speaks.length,
      `${c.id}: emission tokens ${chips().length} != speaks ${c.speaks.length}`);
    // tokens must appear in chunk order, one per chunk, and the model is silent
    // for the large majority of them
    ok(c.outTokens.every((t, i) => t.chunk === i),
      `${c.id}: out tokens are not in chunk order`);
    ok(4 * chips().length <= outEls.length,
      `${c.id}: output is not dominated by silence (${chips().length}/${outEls.length})`);
    // the output stream must never reveal the words the model heard
    ok(c.outTokens.every(t => t.kind !== 'em' || t.text.length > 0),
      `${c.id}: empty emission token`);
    // reveal times must be chronological in both streams
    {
      const rw = c.words.map(w => w.start);
      ok(rw.every((v, i) => i === 0 || v >= rw[i - 1]), `${c.id}: words not in time order`);
      const ro = c.outTokens.map(t => t.reveal);
      ok(ro.every((v, i) => i === 0 || v >= ro[i - 1]), `${c.id}: out tokens not in time order`);
    }

    // initial
    ok(rows().length === 0, `${c.id}: rows at start`);
    ok(spkCells() === 0, `${c.id}: spk cells at start`);
    ok(wordEls.every(e => !has(e, 'on')), `${c.id}: words visible at start`);
    ok(outEls.every(e => !has(e, 'on')), `${c.id}: out tokens visible at start`);
    ok(chips().filter(e => has(e, 'on')).length === 0, `${c.id}: an emission is visible at start`);

    play.onclick();                       // start playback -> registers tick
    const step = t => { audio.currentTime = t; const cb = pending; pending = null; if (cb) cb(); };

    // just before the first reveal nothing should be shown
    const firstRev = c.speaks[0].reveal;
    step(Math.max(0, firstRev - 0.001));
    ok(rows().length === 0, `${c.id}: row shown before reveal (t=${(firstRev - 0.001).toFixed(3)})`);
    ok(!has(cells[c.speaks[0].chunk], 'spk'), `${c.id}: spk cell lit before reveal`);
    ok(chips().filter(e => has(e, 'on')).length === 0, `${c.id}: chip shown before reveal`);

    // walk to each reveal and assert it appears exactly then
    for (let i = 0; i < c.speaks.length; i++) {
      const sp = c.speaks[i];
      step(sp.reveal);
      ok(rows().length === i + 1, `${c.id}: rows at reveal #${i} (got ${rows().length})`);
      ok(has(cells[sp.chunk], 'spk'), `${c.id}: chunk ${sp.chunk} not marked spk at reveal`);
      ok(!has(cells[sp.chunk], 'bad'), `${c.id}: chunk ${sp.chunk} wrongly flagged bad (gt=${sp.gt_idx})`);
      if (sp.gt_idx >= 0) ok(has(gtEls[sp.gt_idx], 'hit'), `${c.id}: gt ${sp.gt_idx} not hit`);
      ok(chips().filter(e => has(e, 'on')).length === i + 1,
        `${c.id}: chips shown at reveal #${i} (got ${chips().filter(e => has(e, 'on')).length})`);
      ok(chips()[i]._html === sp.text, `${c.id}: chip text at #${i}`);
    }

    // both streams must advance in step with the audio, and only when due
    {
      const lastRev = c.speaks[c.speaks.length - 1].reveal;
      const dueW = c.words.filter(w => w.start <= lastRev).length;
      const onW = wordEls.filter(e => has(e, 'on')).length;
      ok(onW === dueW, `${c.id}: words revealed ${onW} != due ${dueW}`);
      const dueO = c.outTokens.filter(t => t.reveal <= lastRev).length;
      const onO = outEls.filter(e => has(e, 'on')).length;
      ok(onO === dueO, `${c.id}: out tokens revealed ${onO} != due ${dueO}`);
      // every target word whose event fired must be marked hit
      c.speaks.forEach(sp => {
        if (sp.gt_idx >= 0) {
          const w = wordEls.find((e, j) => c.words[j].ev === sp.gt_idx);
          ok(w && has(w, 'hit'), `${c.id}: target word for event ${sp.gt_idx} not hit`);
        }
      });
    }

    // a matched word must show a positive delay
    for (const sp of c.speaks) {
      if (sp.gt_idx >= 0) {
        ok(sp.delay >= 0, `${c.id}: negative delay ${sp.delay}`);
        ok(sp.onset >= c.gt[sp.gt_idx].start - 1e-9, `${c.id}: onset before t_start`);
      }
    }

    // run to the end
    for (let t = 0; t <= c.duration + 0.3; t += 0.05) step(Math.min(t, c.duration));
    if (audio.onended) { audio.ended = true; audio.onended(); }
    ok(rows().length === c.speaks.length, `${c.id}: final rows ${rows().length} != speaks ${c.speaks.length}`);
    ok(spkCells() === c.speaks.length, `${c.id}: final spk cells ${spkCells()} != ${c.speaks.length}`);
    ok(hits() === c.gt.length, `${c.id}: gt hits ${hits()} != ${c.gt.length}`);
    ok(wordEls.every(e => has(e, 'on')), `${c.id}: not all words revealed at end`);
    ok(outEls.every(e => has(e, 'on')), `${c.id}: not all out tokens revealed at end`);
    ok(chips().length === c.speaks.length, `${c.id}: final chips ${chips().length} != speaks`);
    ok(chips().every(e => has(e, 'on')), `${c.id}: chip never faded in`);
    // stream-2 DOM order must be the chunk order, with an emission exactly at
    // the chunks where the model spoke
    {
      const emIdx = outEls.map((e, i) => has(e, 'em') ? i : -1).filter(i => i >= 0);
      ok(emIdx.length === c.speaks.length,
        `${c.id}: emission slots ${emIdx.length} != speaks`);
      ok(emIdx.join(',') === c.speaks.map(s => s.chunk).join(','),
        `${c.id}: emissions at chunks ${emIdx.join(',')} != speaks ${c.speaks.map(s => s.chunk).join(',')}`);
      ok(outEls.every(e => has(e, 'em') || has(e, 'sil')),
        `${c.id}: every output token must be an emission or a silence`);
    }

    // reset must wipe everything
    all.find(e => has(e, 'reset')).onclick();
    ok(rows().length === 0, `${c.id}: rows after reset`);
    ok(spkCells() === 0, `${c.id}: spk cells after reset`);
    ok(hits() === 0, `${c.id}: hits after reset`);
    ok(wordEls.every(e => !has(e, 'on')), `${c.id}: words after reset`);
    ok(outEls.every(e => !has(e, 'on')), `${c.id}: out tokens after reset`);
    ok(chips().filter(e => has(e, 'on')).length === 0, `${c.id}: chips after reset`);

    console.log(`  ok  ${c.id.padEnd(18)} ${c.nChunks} chunks  ${c.speaks.length} speaks  ` +
      `${wordEls.length} words  ${outEls.length} out tokens  ` +
      `delay=${c.speaks.map(s => s.delay).join(',')}`);
  });
}
// ---------- tab switching ----------
{
  const tabs = registry.tabs.children;
  ok(tabs.length === T.tasks.length, `tabs count ${tabs.length} != ${T.tasks.length}`);
  T.tasks.forEach((task, i) => {
    ok(typeof tabs[i].onclick === 'function', `tab ${i} has no handler`);
    tabs[i].onclick();
    const shown = registry.grid.children.length;
    const want = T.CASES.filter(c => c.task === task).length;
    ok(shown === want, `tab ${task}: grid shows ${shown} != ${want}`);
    ok(tabs[i]._cls.has('on'), `tab ${task} not marked active`);
    const others = tabs.filter((_, j) => j !== i);
    ok(others.every(t => !t._cls.has('on')), `tab ${task}: stale active tab`);
  });
  console.log(`  ok  tab switching (${tabs.length} tabs)`);
}

console.log(fails ? `\n${fails} FAILURE(S)` : '\nALL CHECKS PASSED');
process.exit(fails ? 1 : 0);
