// Render the demo in the DOM stub and print both streams as text, so the
// interleaving of heard words and model outputs can be eyeballed directly.
const fs = require('fs');
const vm = require('vm');
const path = require('path');

// The project page lives at the repository root; the player is under demo/.
const PAGE = [process.env.DEMO_HTML, 'demo/index.html', 'streaming_160ms_demo.html']
  .filter(Boolean)
  .map(f => path.join(__dirname, f))
  .find(f => fs.existsSync(f));
if (!PAGE) throw new Error('no demo html found next to render_transcript.js');
const html = fs.readFileSync(PAGE, 'utf8');
let js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
js += "\nglobalThis.__T = {CASES, buildCard, render, tasks};\n";

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
const AUDIOS = [];
let pending = null;
const sandbox = {
  document: { createElement: mkEl, getElementById: id => (registry[id] = registry[id] || mkEl('div')) },
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

for (const task of T.tasks) {
  const base = AUDIOS.length;
  T.render(task);
  const cards = registry.grid.children;
  const want = T.CASES.filter(c => c.task === task);

  cards.forEach((card, ci) => {
    const c = want[ci];
    pending = null;
    const all = walk(card);
    const audio = AUDIOS[base + ci];
    audio.duration = c.duration;
    all.find(e => has(e, 'play')).onclick();
    const step = t => { audio.currentTime = t; const cb = pending; pending = null; if (cb) cb(); };
    for (let t = 0; t <= c.duration + 0.3; t += 0.02) step(Math.min(t, c.duration));
    audio.ended = true;
    audio.onended();

    const txA = all.find(e => has(e, 'tx-audio'));
    const txB = all.find(e => has(e, 'tx-out'));
    console.log(`\n=== ${c.id}  [${c.taskLabel}]  query="${c.query}"  ` +
      `delay=${c.stat.delayMean.toFixed(3)}s ===`);
    console.log(`prompt: ${c.prompt}`);

    const wrap = (parts, width = 76) => {
      let line = '';
      for (const p of parts) {
        if ((line + p).length > width) { console.log('  ' + line.trim()); line = ''; }
        line += p + ' ';
      }
      if (line.trim()) console.log('  ' + line.trim());
    };

    console.log(`--- stream 1: audio (${txA.children.length} words) ---`);
    // word index -> the reveal time it faded in at, so the transcript can be
    // printed with timestamps and the emission point is visible inline
    const revealAt = new Map();
    c.words.forEach((w, i) => revealAt.set(txA.children[i], w.start));
    wrap(txA.children.map(n =>
      `${revealAt.get(n).toFixed(2)}:${n._html}${has(n, 'hit') ? '*' : ''}`));
    console.log('  (* = target word the model decided on)');

    console.log(`--- stream 2: model output, one token per chunk ` +
      `(${txB.children.length} tokens = ${c.nChunks} chunks) ---`);
    // compact mark sequence: one glyph per chunk, so the per-chunk decision
    // cadence and the sparsity of the emissions are both directly visible
    const marks = txB.children.map(n => has(n, 'em') ? `[${n._html}]` : '\u00b7');
    for (let i = 0; i < marks.length; i += 38) {
      const lo = i, hi = Math.min(i + 38, marks.length);
      console.log(`  c${String(lo).padStart(3, '0')}-${String(hi - 1).padStart(3, '0')}  ` +
        marks.slice(lo, hi).join(' '));
    }
    console.log(`  (\u00b7 = <silence>, [x] = emitted target; ` +
      `${c.nChunks - c.speaks.length}/${c.nChunks} chunks are silent)`);
    c.speaks.forEach(sp => {
      console.log(`  emission @ chunk ${sp.chunk} (reveal ${sp.reveal.toFixed(2)}s) ` +
        `-> "${sp.text}"  delay=${sp.delay.toFixed(3)}s`);
    });
  });
}
