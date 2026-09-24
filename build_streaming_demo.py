#!/usr/bin/env python3
"""Build a self-contained HTML demo for the 160 ms streaming QA model.

Picks 2 clean cases per task from the 160 ms chunk model's test-clean run,
converts the LibriSpeech audio to mp3 (base64-embedded so the HTML is a single
portable file), and emits a page that plays the audio and replays the model's
per-chunk outputs in real time.

Timing convention (matches the evaluation package):
  * chunk k covers audio [k*CHUNK_SEC, (k+1)*CHUNK_SEC)
  * the token emitted for chunk k becomes *available* at (k+1)*CHUNK_SEC,
    because the chunk must be fully received before inference runs
  * the metric's onset for chunk k is k*CHUNK_SEC
  * delay = onset - t_start of the matched ground-truth event

Usage:
    python build_streaming_demo.py
"""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path

OUT_ROOT = Path(
    "/mnt/afs_1/zhanzewen/Qwen2.5-Omni/finetune_realtime/outputs"
)
RUN = "phase5_joint_posneg080_chunk016_v2"
EVAL = OUT_ROOT / RUN / "eval_test_clean"
METRICS = EVAL / "stream_s2t_metrics_ms"
DEST = Path(__file__).resolve().parent / "streaming_160ms_demo.html"

CHUNK_SEC = 0.16
BITRATE = "48k"

# Word-level forced alignment of the LibriSpeech transcript (see align_words.py),
# used to replay the words of the whole sentence in the order they are heard.
WORD_ALIGN = Path(__file__).resolve().parent / "word_align.json"

# 2 curated clean cases per task (all: clean==1.0, 0 false alarms, short audio)
CASES: list[tuple[str, str]] = [
    ("lexical_detection", "2830-3980-0032"),
    ("lexical_detection", "8555-284447-0023"),
    ("semantic_detection", "3575-170457-0051"),
    ("semantic_detection", "7176-88083-0002"),
    ("lexical_counting", "4970-29095-0005"),
    ("lexical_counting", "2094-142345-0039"),
    ("semantic_counting", "260-123286-0011"),
    ("semantic_counting", "4446-2275-0043"),
]

TASK_LABEL = {
    "lexical_detection": "Lexical detection",
    "semantic_detection": "Semantic detection",
    "lexical_counting": "Lexical counting",
    "semantic_counting": "Semantic counting",
}

TASK_INTRO = {
    "lexical_detection": "Say the target <em>word</em> the moment it is heard.",
    "semantic_detection": "Say any word from the target <em>category</em> the moment it is heard.",
    "lexical_counting": "Keep a running tally of the target <em>word</em>; say the updated count.",
    "semantic_counting": "Keep a running tally of the target <em>category</em>; say the updated count.",
}


def mp3_data_uri(flac: Path) -> str:
    """Transcode flac -> mono 16 kHz mp3 and return a data URI."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clip.mp3"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(flac),
                "-ac", "1", "-ar", "16000", "-b:a", BITRATE,
                str(out),
            ],
            check=True,
        )
        b64 = base64.b64encode(out.read_bytes()).decode("ascii")
    return f"data:audio/mpeg;base64,{b64}"


def match_speaks_to_gt(speaks: list[dict], gt: list[dict]) -> None:
    """Ordered-exclusive match (same rule as the eval package).

    A prediction at onset ``o`` matches GT event with window
    ``[t_start, t_end]`` -- left boundary is strict (no early tolerance).
    """
    used: set[int] = set()
    for sp in speaks:
        o = sp["onset"]
        for i, ev in enumerate(gt):
            if i in used:
                continue
            if ev["start"] <= o <= ev["end"]:
                used.add(i)
                sp["gt_idx"] = i
                sp["delay"] = round(o - ev["start"], 3)
                break
        else:
            sp["gt_idx"] = -1
            sp["delay"] = None


def overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def attach_words(sid: str, gt: list[dict], align: dict[str, list[dict]]) -> list[dict]:
    """Replay the sentence word by word, and link each event to the word heard.

    For detection tasks the annotated label already is the spoken word, but for
    counting tasks the label is the emitted count (``one``/``two``), so the word
    that actually triggered it has to be recovered by temporal overlap.
    """
    words = [
        {"w": a["w"], "start": a["start"], "end": a["end"], "ev": -1}
        for a in align.get(sid, [])
    ]
    for i, ev in enumerate(gt):
        best_j, best_ov = -1, 0.0
        for j, w in enumerate(words):
            ov = overlap(ev["start"], ev["end"], w["start"], w["end"])
            if ov > best_ov:
                best_ov, best_j = ov, j
        if best_j >= 0:
            words[best_j]["ev"] = i
            ev["spoken"] = words[best_j]["w"]
    return words


def build_output_stream(
    speaks: list[dict], n_chunks: int, duration: float, chunk_sec: float,
) -> list[dict]:
    """The model's own output as a wait-token timeline: exactly one token per chunk.

    The model decides once at every chunk boundary, so the output stream has the
    same length as the chunk sequence: it emits the queried target at the chunks
    where it chooses to speak, and ``<silence>`` everywhere else. A decision for
    chunk ``k`` becomes available at the chunk's right boundary, which for the
    final (possibly partial) chunk is the end of the audio. This is the model's
    real output, not a transcription: it says nothing about the words it heard,
    only whether it chose to speak.
    """
    by_chunk = {sp["chunk"]: i for i, sp in enumerate(speaks)}
    stream: list[dict] = []
    for k in range(n_chunks):
        reveal = round(min((k + 1) * chunk_sec, duration), 3)
        i = by_chunk.get(k)
        if i is None:
            stream.append({
                "kind": "sil", "chunk": k, "reveal": reveal,
                "start": round(k * chunk_sec, 3),
            })
        else:
            sp = speaks[i]
            stream.append({
                "kind": "em", "chunk": k, "reveal": reveal, "idx": i,
                "text": sp["text"], "matched": sp["gt_idx"] >= 0,
            })
    return stream


def build_case(task: str, sid: str, align: dict[str, list[dict]]) -> dict:
    log = json.loads((EVAL / f"inference_log_{sid}_{task}.json").read_text())
    per = json.loads(
        (METRICS / task / "metrics" / f"{sid}_stream_s2t.json").read_text()
    )

    dur = float(log["audio_duration_sec"])
    n_chunks = max(int(o["chunk_idx"]) for o in log["model_outputs"]) + 1

    # one entry per chunk: text if SPEAK, else None (= keep silence)
    chunks: list[str | None] = [None] * n_chunks
    for o in log["model_outputs"]:
        k = int(o["chunk_idx"])
        if o.get("action") == "SPEAK" and str(o.get("text", "")).strip():
            chunks[k] = str(o["text"]).strip()

    gt = [
        {"start": round(float(a), 3), "end": round(float(b), 3), "word": str(w)}
        for a, b, w in log["ground_truth"]
    ]

    speaks = [
        {
            "chunk": k,
            "onset": round(k * CHUNK_SEC, 3),
            "reveal": round((k + 1) * CHUNK_SEC, 3),
            "text": t,
        }
        for k, t in enumerate(chunks)
        if t
    ]
    match_speaks_to_gt(speaks, gt)
    words = attach_words(sid, gt, align)
    out_tokens = build_output_stream(speaks, n_chunks, dur, CHUNK_SEC)

    return {
        "id": sid,
        "task": task,
        "taskLabel": TASK_LABEL[task],
        "taskIntro": TASK_INTRO[task],
        "query": log["query"],
        "prompt": log["prompt"],
        "chunkSec": CHUNK_SEC,
        "duration": dur,
        "nChunks": n_chunks,
        "gt": gt,
        "words": words,
        "outTokens": out_tokens,
        "chunks": chunks,
        "speaks": speaks,
        "audio": mp3_data_uri(Path(log["audio_path"])),
        "stat": {
            "f1": per["emission_f1_fuzzy_during_word"],
            "recall": per["emission_recall_fuzzy_during_word"],
            "precision": per["emission_precision_fuzzy_during_word"],
            "clean": per["sample_clean_fuzzy_during_word"],
            "fa": int(per["n_false_alarm"]),
            "delayMean": per["delay_mean_fuzzy_during_word"],
            "nRef": int(per["n_ref_emissions"]),
        },
    }


def main() -> None:
    align = json.loads(WORD_ALIGN.read_text())
    cases = [build_case(t, s, align) for t, s in CASES]
    for c in cases:
        assert c["stat"]["clean"] == 1.0 and c["stat"]["fa"] == 0, c["id"]
        nEm = sum(1 for t in c["outTokens"] if t["kind"] == "em")
        nSil = sum(1 for t in c["outTokens"] if t["kind"] == "sil")
        print(
            f"  {c['task']:18s} {c['id']:18s} {c['duration']:5.2f}s "
            f"{c['nChunks']:4d} chunks  {len(c['gt'])} events  "
            f"words={len(c['words']):3d}  out={len(c['outTokens'])} ({nEm} em/{nSil} sil)  "
            f"delay={c['stat']['delayMean']:.3f}"
        )
        assert len(c["speaks"]) == len(c["gt"]), (c["id"], len(c["speaks"]), len(c["gt"]))
        assert len(c["words"]) > 0, c["id"]
        assert all(ev.get("spoken") for ev in c["gt"]), c["id"]
        # stream 1: the audio stream must voice every target exactly once
        assert sum(1 for w in c["words"] if w["ev"] >= 0) == len(c["gt"]), c["id"]
        # stream 2: exactly one decision per chunk, emissions only where it spoke
        assert len(c["outTokens"]) == c["nChunks"], (c["id"], len(c["outTokens"]))
        assert nEm == len(c["speaks"]) == len(c["gt"]), (c["id"], nEm)
        assert nSil == c["nChunks"] - nEm, (c["id"], nSil, c["nChunks"], nEm)
        assert [t["chunk"] for t in c["outTokens"]] == list(range(c["nChunks"])), c["id"]
        # the output is dominated by silence: most chunks produce nothing
        assert 4 * nEm <= c["nChunks"], (c["id"], nEm, c["nChunks"])
        # reveal times must be non-decreasing and fit inside the utterance
        rev = [t["reveal"] for t in c["outTokens"]]
        assert rev == sorted(rev), (c["id"], rev)
        assert rev[-1] <= c["duration"] + 1e-9, (c["id"], rev[-1], c["duration"])

    html = TEMPLATE.replace("__DATA__", json.dumps(cases, ensure_ascii=False))
    DEST.write_text(html, encoding="utf-8")
    print(f"\nWrote {DEST}  ({DEST.stat().st_size/1e6:.2f} MB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LibriSpeech-Real-QA &middot; streaming demo</title>
<style>
  :root{
    --bg:#0a0d13; --panel:#141a24; --panel2:#1a2130; --line:#252e3d;
    --txt:#e8eef6; --dim:#8a97a8; --dim2:#5a6678;
    --accent:#4c9aff; --accSoft:rgba(76,154,255,.15); --accLine:rgba(76,154,255,.45);
    --ok:#3fb950; --bad:#f85149; --gt:#a78bfa;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -14px rgba(0,0,0,.85);
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--bg); color:var(--txt);
    font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif;
    -webkit-font-smoothing:antialiased;
    background-image:
      radial-gradient(900px 400px at 10% -10%, rgba(76,154,255,.11), transparent 62%),
      radial-gradient(800px 400px at 92% 0%, rgba(167,139,250,.09), transparent 62%);
    background-attachment:fixed;
  }
  .wrap{max-width:1200px;margin:0 auto;padding:34px 22px 90px}

  /* ---------------- header ---------------- */
  header h1{
    font-size:22px;margin:0 0 13px;font-weight:680;letter-spacing:.1px;
    background:linear-gradient(92deg,#eaf2ff 0%,#9dc4ff 56%,#b79cff 100%);
    -webkit-background-clip:text;background-clip:text;color:transparent;
  }
  header h1 .sep{-webkit-text-fill-color:var(--dim2);color:var(--dim2);margin-right:2px;font-weight:400}
  header .model{color:var(--dim);font-size:13px;margin:0 0 13px;font-weight:500}
  header .model b{color:#9dc4ff;font-weight:650}
  header .sub{display:flex;gap:7px;flex-wrap:wrap}
  .pill{
    font-size:11.5px;padding:3.5px 11px;border-radius:999px;
    border:1px solid var(--line);background:rgba(255,255,255,.025);
    color:var(--dim);white-space:nowrap;
  }
  .pill.ok{color:#7ee787;border-color:rgba(63,185,80,.32);background:rgba(63,185,80,.09)}
  .pill i{font-style:normal;font-family:ui-monospace,monospace;color:#c9d1d9}

  /* ---------------- tabs ---------------- */
  .tabs{display:flex;gap:8px;flex-wrap:wrap;margin:26px 0 4px}
  .tab{
    padding:8px 17px;border-radius:999px;border:1px solid var(--line);
    background:rgba(255,255,255,.02);color:var(--dim);cursor:pointer;
    font-size:13px;font-weight:550;
    transition:color .18s,border-color .18s,background .18s,transform .18s,box-shadow .18s;
  }
  .tab:hover{color:var(--txt);border-color:#3a4658;transform:translateY(-1px)}
  .tab.on{
    color:#fff;border-color:var(--accLine);background:var(--accSoft);
    box-shadow:0 6px 20px -10px var(--accent),inset 0 1px 0 rgba(255,255,255,.06);
  }
  .intro{color:var(--dim);font-size:13px;margin:14px 0 20px;min-height:20px}
  .intro em{color:#9dc4ff;font-style:normal;font-weight:600}
  .intro .dim{color:var(--dim2)}

  /* ---------------- cards ---------------- */
  .grid{display:grid;grid-template-columns:1fr;gap:20px;max-width:1000px}
  .card{
    position:relative;border:1px solid var(--line);border-radius:15px;overflow:hidden;
    background:linear-gradient(180deg,rgba(255,255,255,.030),rgba(255,255,255,.007));
    box-shadow:var(--shadow);
    transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;
    animation:cardIn .38s cubic-bezier(.2,.7,.3,1) both;
  }
  @keyframes cardIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
  .card::before{
    content:"";position:absolute;inset:0 0 auto 0;height:2px;
    background:linear-gradient(90deg,var(--accent),transparent 76%);
    opacity:.8;
  }
  .card:hover{
    transform:translateY(-2px);border-color:var(--accLine);
    box-shadow:var(--shadow),0 0 0 1px var(--accSoft);
  }
  .chead{padding:15px 17px 13px;border-bottom:1px solid var(--line)}
  .crow{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
  .cid{font:600 14px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;color:#dbe5f0}
  .chip{
    font-size:11px;padding:2.5px 9px;border-radius:999px;white-space:nowrap;
    border:1px solid var(--line);color:var(--dim);background:rgba(255,255,255,.02);
  }
  .query{
    margin-top:10px;font-size:13px;color:var(--dim);
    display:flex;align-items:center;gap:7px;flex-wrap:wrap;
  }
  .tlabel{
    font-size:11px;font-weight:650;padding:2.5px 9px;border-radius:999px;
    color:var(--accent);background:var(--accSoft);border:1px solid var(--accLine);
    letter-spacing:.2px;
  }
  .qbox{
    padding:2.5px 10px;border-radius:6px;
    background:var(--accSoft);color:var(--accent);font-weight:650;
    border:1px solid var(--accLine);
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
  }
  .prompt{
    margin-top:9px;padding:6px 10px;border-radius:7px;
    background:rgba(255,255,255,.022);border-left:2px solid var(--line);
    color:var(--dim2);font-size:12px;font-style:italic;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  }
  .body{padding:15px 17px 17px}

  /* ---------------- transport ---------------- */
  .ctl{display:flex;align-items:center;gap:13px}
  button.play{
    width:40px;height:40px;flex:none;border-radius:50%;border:none;cursor:pointer;
    background:var(--accent);color:#08111d;font-size:14px;display:grid;place-items:center;
    box-shadow:0 5px 16px -6px var(--accent),inset 0 -1px 0 rgba(0,0,0,.22);
    transition:transform .15s,filter .15s,box-shadow .15s;
  }
  button.play:hover{filter:brightness(1.1);transform:scale(1.05);box-shadow:0 7px 22px -6px var(--accent)}
  button.play:disabled{opacity:.4;cursor:default}
  .bar{flex:1;height:6px;border-radius:99px;background:#1b2231;position:relative;cursor:pointer;
       overflow:hidden;transition:height .15s}
  .bar:hover{height:9px}
  .bar .fill{
    position:absolute;inset:0 auto 0 0;width:0;border-radius:99px;
    background:linear-gradient(90deg,var(--accLine),var(--accent));
    box-shadow:0 0 12px -2px var(--accent);
  }
  .time{font:500 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim);min-width:78px;text-align:right}
  .reset{
    border:1px solid var(--line);background:rgba(255,255,255,.02);color:var(--dim);
    border-radius:8px;padding:7px 12px;font-size:12px;cursor:pointer;transition:.15s;
  }
  .reset:hover{color:var(--txt);border-color:#3d4757;background:rgba(255,255,255,.05)}

  /* ---------------- sections ---------------- */
  .sec{margin-top:17px}
  .lbl{
    font-size:10.5px;letter-spacing:1.1px;text-transform:uppercase;
    color:var(--dim2);margin-bottom:8px;font-weight:650;
  }

  /* ---------------- timeline ---------------- */
  .tl{position:relative;height:34px;border-radius:9px;background:#0c111a;
      border:1px solid var(--line);overflow:hidden}
  .tl .gtpos{
    position:absolute;top:6px;bottom:6px;border-radius:5px;
    background:linear-gradient(180deg,rgba(167,139,250,.55),rgba(167,139,250,.32));
    border:1px solid rgba(167,139,250,.85);
    transition:background .3s,border-color .3s,box-shadow .3s;
  }
  .tl .gtpos.hit{
    background:linear-gradient(180deg,rgba(63,185,80,.60),rgba(63,185,80,.36));
    border-color:rgba(63,185,80,.95);
    box-shadow:0 0 15px -2px rgba(63,185,80,.65);
  }
  .tl .gtpos span{
    position:absolute;inset:0;display:grid;place-items:center;overflow:hidden;
    font:650 10.5px/1 ui-monospace,monospace;color:#fff;white-space:nowrap;
    text-shadow:0 1px 2px rgba(0,0,0,.55);
  }
  .tl .head{
    position:absolute;top:0;bottom:0;width:2px;background:#fff;opacity:.9;
    pointer-events:none;box-shadow:0 0 8px rgba(255,255,255,.55);
  }
  .chunks{display:flex;gap:2px;height:18px;margin-top:7px}
  .chunks i{flex:1;background:#1c2431;border-radius:2px;
            transition:background .12s,transform .12s,box-shadow .12s}
  .chunks i.live{background:#4a5878;transform:scaleY(1.2)}
  .chunks i.done{background:#2f3a4b}
  .chunks i.spk{background:var(--ok);box-shadow:0 0 10px -1px rgba(63,185,80,.8)}
  .chunks i.spk.bad{background:var(--bad);box-shadow:0 0 10px -1px rgba(248,81,73,.8)}

  /* ---------------- live transcript ---------------- */
  .tx{
    padding:13px 15px;border-radius:10px;min-height:82px;
    background:#0a0e15;border:1px solid var(--line);
    font:14.5px/2.1 ui-monospace,SFMono-Regular,Menlo,monospace;color:#c9d1d9;
  }
  .tx .w{
    opacity:0;transform:translateY(3px);display:inline-block;margin-right:7px;
    transition:opacity .3s ease,transform .3s ease,color .3s ease,text-shadow .3s ease;
  }
  .tx .w.on{opacity:1;transform:none}
  .tx .w.tgt{color:#b79cff;font-weight:700}
  .tx .w.tgt.on{text-shadow:0 0 16px rgba(167,139,250,.55)}
  .tx .w.tgt.hit{color:#7ee787;text-shadow:0 0 16px rgba(63,185,80,.55)}
  .tx .w.tgt.miss{color:#ff9b95;text-decoration:line-through;text-shadow:none}
  .tx .w.sil{
    color:#3f4a59;font-weight:400;font-size:12px;letter-spacing:.2px;
    padding:1px 7px;border-radius:5px;margin-right:6px;
    background:rgba(255,255,255,.028);border:1px solid rgba(255,255,255,.055);
  }
  .tx .em{
    display:inline-flex;align-items:center;margin:0 8px;padding:2.5px 10px;
    border-radius:999px;font-size:11.5px;font-weight:700;vertical-align:1px;
    background:rgba(63,185,80,.16);color:#7ee787;border:1px solid rgba(63,185,80,.45);
    opacity:0;transform:scale(.85);
    transition:opacity .26s ease,transform .26s cubic-bezier(.2,1.6,.4,1);
  }
  .tx .em.on{opacity:1;transform:none;animation:pulse 1.1s ease-out 1}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(63,185,80,.5)}100%{box-shadow:0 0 0 11px rgba(63,185,80,0)}}
  .tx .em.bad{background:rgba(248,81,73,.15);color:#ff9b95;border-color:rgba(248,81,73,.45)}
  .tx .cnt{color:#5a6678;font-size:11.5px;font-weight:600}
  .tx .ph{color:#3d4757}
  .tx .nch{color:#5a6678;font-size:11.5px;font-weight:600}
  .tx-audio{border-left:2px solid rgba(167,139,250,.35)}
  .tx-out{border-left:2px solid rgba(63,185,80,.35);line-height:1.95}
  /* one <silence> token per chunk: keep each one narrow so the per-chunk
     cadence stays readable even when the utterance spans dozens of chunks */
  .tx-out .w.sil{
    font-size:10px;padding:0 5px;margin-right:4px;letter-spacing:0;
  }
  .tx-out .em{margin:0 4px}
  .hint{
    margin-top:7px;color:var(--dim2);font-size:11.5px;line-height:1.6;
  }
  .hint b{color:#9aa7b8;font-weight:600}

  /* ---------------- output log ---------------- */
  .log{
    height:156px;overflow:auto;border-radius:10px;
    background:#080c12;border:1px solid var(--line);padding:9px 12px;
  }
  .log::-webkit-scrollbar{width:8px}
  .log::-webkit-scrollbar-thumb{background:#2b3546;border-radius:4px}
  .log::-webkit-scrollbar-thumb:hover{background:#3a4658}
  .log::-webkit-scrollbar-track{background:transparent}
  .log .row{
    display:flex;gap:10px;align-items:baseline;padding:3.5px 0;
    font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
    opacity:0;animation:in .24s ease forwards;
  }
  @keyframes in{from{opacity:0;transform:translateY(-3px)}to{opacity:1;transform:none}}
  .log .t{color:#55627a;min-width:58px}
  .log .k{color:#55627a;min-width:66px}
  .log .v{font-weight:600}
  .log .v.spk{color:var(--ok)}
  .log .v.spk.bad{color:var(--bad)}
  .log .v.wait{color:#4a5563;font-weight:400}
  .log .d{color:var(--dim);font-size:11.5px}
  .log .d.good{color:#7ee787}
  .log .d.bad{color:#ff9b95}
  .log .empty{color:#44505f;font:12.5px/1.5 ui-monospace,monospace}

  /* ---------------- stats ---------------- */
  .stats{
    display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
    margin-top:15px;padding-top:15px;border-top:1px solid var(--line);
  }
  .st{display:flex;flex-direction:column;gap:3px;padding-left:14px;
      border-left:1px solid var(--line)}
  .st:first-child{border-left:none;padding-left:0}
  .st b{font:650 16px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--txt)}
  .st b.g{color:#7ee787}
  .st span{font-size:10px;letter-spacing:.7px;text-transform:uppercase;
           color:var(--dim2);font-weight:650}

  /* ---------------- footer ---------------- */
  footer{margin-top:42px;color:var(--dim);font-size:12.5px;
         border-top:1px solid var(--line);padding-top:18px}
  footer code{background:var(--panel2);padding:1.5px 6px;border-radius:5px;
              color:#c9d1d9;font-size:11.5px}
  .fnote{color:var(--dim)}
  details{margin-top:12px}
  summary{
    cursor:pointer;color:#9dc4ff;font-size:12.5px;font-weight:550;
    list-style:none;display:inline-flex;align-items:center;gap:6px;
    user-select:none;
  }
  summary::-webkit-details-marker{display:none}
  summary::before{content:"\203A";display:inline-block;font-size:15px;line-height:1;
                  transition:transform .2s}
  details[open] summary::before{transform:rotate(90deg)}
  details ul{margin:11px 0 0;padding-left:20px;line-height:1.9;color:var(--dim)}
  details li{margin-bottom:6px}
  details .tw{color:#b79cff;font-weight:700}
  details .hw{color:#7ee787;font-weight:700}
  details .ew{color:#7ee787;font-weight:700}

  @media (max-width:560px){
    .wrap{padding:24px 14px 70px}
    .grid{grid-template-columns:1fr}
    header h1{font-size:18px}
    .stats{grid-template-columns:repeat(2,1fr);gap:12px 1px}
    .st:nth-child(3){border-left:none;padding-left:0}
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>LibriSpeech-Real-QA<span class="sep">:</span> A Benchmark for Streaming Question Answering</h1>
    <div class="model"><b>Qwen2.5-Omni&nbsp;+&nbsp;SFT</b> &middot; 160&nbsp;ms chunk streaming demo</div>
    <div class="sub">
      <span class="pill"><i>test-clean</i></span>
      <span class="pill">chunk <i>0.16&nbsp;s</i></span>
      <span class="pill">two streams: audio vs.&nbsp;model output</span>
      <span class="pill ok">real model output</span>
    </div>
  </header>

  <div class="tabs" id="tabs"></div>
  <div class="intro" id="intro"></div>
  <div class="grid" id="grid"></div>

  <footer>
    <div class="fnote">
      Every clip and every output is the model's real <code>test-clean</code> prediction
      &mdash; nothing is scripted or simulated.
    </div>
    <details>
      <summary>How to read this display</summary>
      <ul>
        <li><b>Stream 1 &mdash; audio.</b> Every word of the sentence, faded in at its
            spoken onset (wav2vec2 forced alignment). This is the input side; it is
            <em>not</em> the model's output.
            <span class="tw">Purple</span> marks the queried target, turning
            <span class="hw">green</span> the moment the model responds.</li>
        <li><b>Stream 2 &mdash; model output.</b> One token per chunk, because the model
            decides exactly once at every chunk boundary: a
            <code>&lt;silence&gt;</code> token whenever it emits nothing, and a
            <span class="ew">green chip</span> at the chunks where it commits the
            queried target. The output is dominated by silence &mdash; the model is not
            transcribing.</li>
        <li><b>Timing.</b> Chunk <i>k</i> covers <code>[k&middot;0.16,(k+1)&middot;0.16)</code>&nbsp;s and is fully
            consumed at <code>(k+1)&middot;0.16</code>&nbsp;s &mdash; the earliest instant the model can answer.
            An emission therefore appears at that instant, and the metric scores its
            <i>onset</i> <code>k&middot;0.16</code>&nbsp;s against <code>[t_start, t_end]</code>.</li>
        <li><b>Scoring.</b> The metric matches the chunk's <i>onset</i>
            <code>k&middot;0.16</code>&nbsp;s against <code>[t_start, t_end]</code>, so
            <code>delay</code> is measured from the word's onset.</li>
      </ul>
    </details>
  </footer>
</div>

<script>
const CASES = __DATA__;

const SIL_TOKEN = "&lt;silence&gt;";
const fmt = (x, n = 2) => Number(x).toFixed(n);

// One accent per task, so each card reads as a distinct "channel".
const ACCENT = {
  lexical_detection:  "#4c9aff",
  semantic_detection: "#a78bfa",
  lexical_counting:   "#f0b429",
  semantic_counting:  "#2dd4bf",
};

function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function applyAccent(node, task) {
  const hex = ACCENT[task] || "#4c9aff";
  node.style.setProperty("--accent", hex);
  node.style.setProperty("--accSoft", rgba(hex, 0.15));
  node.style.setProperty("--accLine", rgba(hex, 0.45));
}

function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
}

function buildCard(c) {
  const card = el("div", "card");
  applyAccent(card, c.task);

  const head = el("div", "chead");
  head.appendChild(el("div", "crow",
    `<div class="cid">${c.id}</div>
     <div class="chip">${c.nChunks} chunks &middot; ${fmt(c.duration)} s</div>`));
  head.appendChild(el("div", "query",
    `<span class="tlabel">${c.taskLabel}</span>
     target ${c.task.includes("semantic") ? "category" : "word"}
     <span class="qbox">${c.query}</span>`));
  const pr = el("div", "prompt", c.prompt);
  pr.title = c.prompt;
  head.appendChild(pr);
  card.appendChild(head);

  const body = el("div", "body");

  // ---- transport ----
  const ctl = el("div", "ctl");
  const play = el("button", "play", "&#9658;");
  play.title = "play / pause";
  const bar = el("div", "bar", '<div class="fill"></div>');
  const fill = bar.querySelector(".fill");
  const time = el("div", "time", `0.00 / ${fmt(c.duration)}`);
  const reset = el("button", "reset", "reset");
  ctl.append(play, bar, time, reset);
  body.appendChild(ctl);

  // ---- stream 1: what is heard, word by word (for reference) ----
  const secA = el("div", "sec");
  secA.appendChild(el("div", "lbl", "Stream 1 &middot; audio"));
  const txA = el("div", "tx tx-audio");
  const wordEls = c.words.map(w => {
    const isTgt = w.ev >= 0;
    const s = el("span", "w" + (isTgt ? " tgt" : ""), w.w);
    s.title = `${w.w}  [${fmt(w.start)}, ${fmt(w.end)}] s` +
      (isTgt ? `\ntarget event ${w.ev + 1} of ${c.gt.length}` : "");
    txA.appendChild(s);
    return s;
  });
  const eventWord = {};                       // gt index -> its word span
  c.words.forEach((w, i) => { if (w.ev >= 0) eventWord[w.ev] = wordEls[i]; });
  secA.appendChild(txA);
  const hintA = el("div", "hint",
    "The input side &mdash; it is <b>not</b> the model's output.");
  secA.appendChild(hintA);
  body.appendChild(secA);

  // ---- stream 2: the model's real output = one wait token per chunk ----
  const secB = el("div", "sec");
  secB.appendChild(el("div", "lbl",
    `Stream 2 &middot; model output &nbsp;<span class="nch">1 token per chunk ` +
    `&middot; ${c.nChunks} chunks</span>`));
  const txB = el("div", "tx tx-out");
  const outEls = [];
  c.outTokens.forEach(t => {
    let n;
    if (t.kind === "sil") {
      n = el("span", "w sil", SIL_TOKEN);
      n.title = `chunk ${t.chunk}  [${fmt(t.start)}, ${fmt(t.reveal)}] s\n` +
        `<silence>  -  the model emits nothing at this chunk boundary`;
    } else {
      const sp = c.speaks[t.idx];
      n = el("span", "em" + (t.matched ? "" : " bad"), t.text);
      n.title = `chunk ${sp.chunk} &middot; decided at ${fmt(sp.reveal)} s` +
        (t.matched ? ` &middot; delay +${fmt(sp.delay)} s` : " &middot; false alarm");
    }
    txB.appendChild(n);
    outEls.push(n);
  });
  secB.appendChild(txB);
  const hintB = el("div", "hint",
    "The model's actual output: it decides once at every chunk boundary and " +
    "transcribes nothing. It stays silent at " + (c.nChunks - c.gt.length) +
    " of the " + c.nChunks + " chunks and emits the queried target at the " +
    c.gt.length + " chunk" + (c.gt.length > 1 ? "s" : "") + " where it fires.");
  secB.appendChild(hintB);
  body.appendChild(secB);

  // ---- ground-truth timeline ----
  const secTl = el("div", "sec");
  secTl.appendChild(el("div", "lbl", "Timeline &nbsp;&middot;&nbsp; ground truth vs. emission"));
  const tl = el("div", "tl");
  const head2 = el("div", "head");
  tl.appendChild(head2);
  const gtEls = c.gt.map(g => {
    const d = el("div", "gtpos",
      `<span>${g.word}</span>`);
    d.style.left = (100 * g.start / c.duration) + "%";
    d.style.width = Math.max(1.2, 100 * (g.end - g.start) / c.duration) + "%";
    d.title = `${g.word}  [${fmt(g.start)}, ${fmt(g.end)}] s`;
    tl.appendChild(d);
    return d;
  });
  secTl.appendChild(tl);

  // ---- chunk strip ----
  const strip = el("div", "chunks");
  const cells = [];
  for (let i = 0; i < c.nChunks; i++) {
    const cell = el("i");
    cell.title = `chunk ${i}  [${fmt(i * c.chunkSec)}, ${fmt((i + 1) * c.chunkSec)}) s`;
    strip.appendChild(cell);
    cells.push(cell);
  }
  secTl.appendChild(strip);
  body.appendChild(secTl);

  // ---- live output log ----
  const secLog = el("div", "sec");
  secLog.appendChild(el("div", "lbl", "Output"));
  const log = el("div", "log");
  const empty = el("div", "empty", "&#9658;&nbsp; press play");
  log.appendChild(empty);
  secLog.appendChild(log);
  body.appendChild(secLog);

  // ---- stats ----
  const s = c.stat;
  const stats = el("div", "stats");
  stats.innerHTML = `
    <div class="st"><b class="g">${fmt(s.f1, 3)}</b><span>F1</span></div>
    <div class="st"><b>${fmt(s.recall, 3)}</b><span>Recall</span></div>
    <div class="st"><b>${fmt(s.precision, 3)}</b><span>Precision</span></div>
    <div class="st"><b>${fmt(s.delayMean, 3)}s</b><span>Mean delay</span></div>`;
  body.appendChild(stats);

  card.appendChild(body);

  // ---------------- runtime ----------------
  const audio = new Audio(c.audio);
  audio.preload = "auto";

  let wShown = 0;             // how many stream-1 words already revealed
  let oShown = 0;             // how many stream-2 tokens already revealed
  let raf = null;
  let rows = [];

  // A chunk is fully received at its END: chunk k completes at (k+1)*chunkSec,
  // which is exactly when its token becomes available.
  function paintChunks(received) {
    for (let i = 0; i < c.nChunks; i++) {
      const done = i < received;
      cells[i].classList.toggle("done", done);
      const isSpk = done && c.chunks[i] != null;
      cells[i].classList.toggle("spk", isSpk);
      if (isSpk) {
        const sp = c.speaks.find(x => x.chunk === i);
        cells[i].classList.toggle("bad", !!(sp && sp.gt_idx < 0));
      } else {
        cells[i].classList.remove("bad");
      }
      cells[i].classList.toggle("live", i === received);
    }
  }

  function resetState() {
    wShown = 0;
    oShown = 0;
    wordEls.forEach(x => x.classList.remove("on", "hit", "miss"));
    outEls.forEach(x => x.classList.remove("on", "hit", "miss"));
    rows.forEach(r => r.remove());
    rows = [];
    log.innerHTML = "";
    log.appendChild(empty);
    empty.style.display = "";
    paintChunks(0);
    gtEls.forEach(x => x.classList.remove("hit"));
    fill.style.width = "0%";
    head2.style.left = "0%";
    time.textContent = `0.00 / ${fmt(c.duration)}`;
  }

  // Reveal both streams strictly in time order: stream-1 words at their spoken
  // onset, stream-2 tokens (silence or emission) at the instant they appear.
  function advance(t) {
    for (;;) {
      const hasW = wShown < c.words.length;
      const hasO = oShown < c.outTokens.length;
      if (!hasW && !hasO) return;
      const tw = hasW ? c.words[wShown].start : Infinity;
      const to = hasO ? c.outTokens[oShown].reveal : Infinity;
      if (Math.min(tw, to) > t + 1e-6) return;
      if (hasW && tw <= to) wordEls[wShown++].classList.add("on");
      else revealOut();
    }
  }

  // Reveal one stream-2 token. An emission is the model committing a decision,
  // so it also lights the ground-truth cell and is mirrored in the log.
  function revealOut() {
    const tk = c.outTokens[oShown];
    const node = outEls[oShown];
    oShown++;
    void node.offsetWidth;          // flush styles so the fade-in actually runs
    node.classList.add("on");
    if (tk.kind !== "em") return;
    fire(tk);
  }

  // The model commits a decision: light the word it responded to and log it.
  function fire(tk) {
    const sp = c.speaks[tk.idx];
    if (sp.gt_idx >= 0 && eventWord[sp.gt_idx]) {
      eventWord[sp.gt_idx].classList.add("hit");
    }
    appendRow(sp);
  }

  function appendRow(speak) {
    empty.style.display = "none";
    const matched = speak.gt_idx >= 0;
    const row = el("div", "row");

    const t = fmt(speak.reveal) + "s";
    const k = "chunk " + speak.chunk;
    row.appendChild(el("div", "t", t));
    row.appendChild(el("div", "k", k));
    row.appendChild(el("div", "v spk" + (matched ? "" : " bad"), `&ldquo;${speak.text}&rdquo;`));

    let d;
    if (matched) {
      d = `&#10003; onset ${fmt(speak.onset)}s &nbsp; delay +${fmt(speak.delay)}s`;
      row.appendChild(el("div", "d good", d));
      gtEls[speak.gt_idx].classList.add("hit");
    } else {
      row.appendChild(el("div", "d bad", `&#10007; onset ${fmt(speak.onset)}s &nbsp; false alarm`));
    }
    log.appendChild(row);
    rows.push(row);
    log.scrollTop = log.scrollHeight;
  }

  function tick() {
    const t = audio.currentTime;
    const d = audio.duration || c.duration;

    fill.style.width = Math.min(100, 100 * t / d) + "%";
    head2.style.left = Math.min(100, 100 * t / d) + "%";
    time.textContent = `${fmt(t)} / ${fmt(d)}`;

    // chunk strip: chunk k is complete once t >= (k+1)*chunkSec
    paintChunks(Math.min(c.nChunks, Math.floor(t / c.chunkSec + 1e-9)));

    // reveal words heard so far, and any model output already committed
    advance(t);

    if (!audio.paused && !audio.ended) raf = requestAnimationFrame(tick);
  }

  play.onclick = () => {
    if (audio.paused || audio.ended) {
      if (audio.ended) { resetState(); audio.currentTime = 0; }
      audio.play();
      play.innerHTML = "&#10073;&#10073;";
      raf = requestAnimationFrame(tick);
    } else {
      audio.pause();
      play.innerHTML = "&#9658;";
    }
  };
  audio.onpause = () => { play.innerHTML = "&#9658;"; };
  audio.onended = () => {
    play.innerHTML = "&#9658;";
    paintChunks(c.nChunks);
    advance(Infinity);
    c.gt.forEach((ev, i) => {
      if (eventWord[i] && !eventWord[i].classList.contains("hit")) {
        eventWord[i].classList.add("miss");
      }
    });
    fill.style.width = "100%";
    head2.style.left = "100%";
  };

  reset.onclick = () => {
    audio.pause();
    audio.currentTime = 0;
    resetState();
    play.innerHTML = "&#9658;";
  };
  bar.onclick = (e) => {
    const r = bar.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    const wasPlaying = !audio.paused;
    audio.pause();
    audio.currentTime = frac * (audio.duration || c.duration);
    resetState();
    // fast-forward the visual state without animation
    const t = audio.currentTime;
    paintChunks(Math.min(c.nChunks, Math.floor(t / c.chunkSec + 1e-9)));
    advance(t);
    fill.style.width = (100 * t / (audio.duration || c.duration)) + "%";
    head2.style.left = (100 * t / (audio.duration || c.duration)) + "%";
    if (wasPlaying) { audio.play(); play.innerHTML = "&#10073;&#10073;"; raf = requestAnimationFrame(tick); }
  };

  return card;
}

// ---- tabs ----
const tasks = [...new Set(CASES.map(c => c.task))];
const tabs = document.getElementById("tabs");
const grid = document.getElementById("grid");
const intro = document.getElementById("intro");

function render(task) {
  [...tabs.children].forEach(t => t.classList.toggle("on", t.dataset.task === task));
  grid.innerHTML = "";
  intro.innerHTML = c_intro(task);
  CASES.filter(c => c.task === task).forEach(c => grid.appendChild(buildCard(c)));
}
function c_intro(task) {
  const c = CASES.find(x => x.task === task);
  return `${c.taskIntro} <span class="dim">&middot; ${CASES.filter(x => x.task === task).length} real cases</span>`;
}

tasks.forEach((task, i) => {
  const t = el("div", "tab", CASES.find(c => c.task === task).taskLabel);
  t.dataset.task = task;
  applyAccent(t, task);
  t.onclick = () => render(task);
  tabs.appendChild(t);
  if (i === 0) t.classList.add("on");
});
render(tasks[0]);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
