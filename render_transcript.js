// Render the demo in the DOM stub and print the transcript as text, so the
// interleaving of spoken words and model outputs can be eyeballed directly.
const fs = require('fs');
const vm = require('vm');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
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

    const tx = all.find(e => has(e, 'tx'));
    console.log(`\n=== ${c.id}  [${c.taskLabel}]  query="${c.query}"  ` +
      `delay=${c.stat.delayMean.toFixed(3)}s ===`);
    console.log(`prompt: ${c.prompt}`);
    console.log('--- transcript after playback ---');
    const parts = [];
    tx.children.forEach(n => {
      if (has(n, 'em')) parts.push(`  ⟹${n._html.replace(/&#8680;/g, '')}  `);
      else parts.push(n._html + (has(n, 'hit') ? '*' : ''));
    });
    // wrap at ~78 chars for readability
    let line = '';
    for (const p of parts) {
      if ((line + p).length > 78) { console.log('  ' + line.trim()); line = ''; }
      line += p + ' ';
    }
    if (line.trim()) console.log('  ' + line.trim());
    console.log('  (* = target word the model responded to; ⟹ = model output at commit time)');
  });
}
