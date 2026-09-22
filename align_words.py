#!/usr/bin/env python3
"""Forced-align LibriSpeech transcripts with wav2vec2 to get word-level timing.

LibriSpeech ships the exact transcript for every utterance, so we do forced
alignment (known text) rather than ASR: we push the utterance through
wav2vec2-base-960h, then use torchaudio.functional.forced_align to find the
frame span of every known token, and merge tokens back into words.

Output: JSON {utterance_id: [{"w": word, "start": s, "end": e}, ...]}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F

CORPUS = Path("/mnt/afs_1/zhanzewen/luhui/corpus/librispeech_orig/LibriSpeech")
CASES = [
    ("test-clean", "2830-3980-0032"),
    ("test-clean", "8555-284447-0023"),
    ("test-clean", "3575-170457-0051"),
    ("test-clean", "7176-88083-0002"),
    ("test-clean", "4970-29095-0005"),
    ("test-clean", "2094-142345-0039"),
    ("test-clean", "260-123286-0011"),
    ("test-clean", "4446-2275-0043"),
]
OUT = Path(__file__).resolve().parent / "word_align.json"


def load_transcript(split: str, uid: str) -> str:
    sp, ch, _ = uid.split("-")
    txt = CORPUS / split / sp / ch / f"{sp}-{ch}.trans.txt"
    for line in txt.read_text().splitlines():
        if line.startswith(uid + " "):
            return line[len(uid) + 1:].strip()
    raise KeyError(uid)


def tokenize(text: str, dictionary: dict[str, int]) -> tuple[list[int], list[int]]:
    """Return (token ids, word id per token). `|` is the word-separator label."""
    ids: list[int] = []
    wid: list[int] = []
    w = 0
    for word in text.upper().split():
        for ch in word:
            if ch not in dictionary:
                continue
            ids.append(dictionary[ch])
            wid.append(w)
        ids.append(dictionary["|"])   # word boundary
        wid.append(w)
        w += 1
    return ids, wid


def main() -> None:
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    print("loading wav2vec2 ...", flush=True)
    model = bundle.get_model()
    labels = bundle.get_labels()
    dictionary = {c: i for i, c in enumerate(labels)}
    print("labels:", labels, flush=True)

    out: dict[str, list[dict]] = {}
    for split, uid in CASES:
        sp, ch, _ = uid.split("-")
        wav_path = CORPUS / split / sp / ch / f"{uid}.flac"
        wav, sr = torchaudio.load(str(wav_path))
        wav = wav.mean(0, keepdim=True)
        if sr != bundle.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)

        text = load_transcript(split, uid)
        targets, word_ids = tokenize(text, dictionary)

        with torch.inference_mode():
            emission, _ = model(wav)

        ratio = wav.shape[-1] / emission.shape[1] / bundle.sample_rate  # sec per frame
        aligned, scores = F.forced_align(emission, torch.tensor([targets]), blank=0)
        spans = F.merge_tokens(aligned[0], scores[0])

        # accumulate token spans into word spans (skip the '|' separator, id=1)
        words: dict[int, dict] = {}
        for sp_, wid_ in zip(spans, word_ids):
            if labels[sp_.token] == "|":
                continue
            d = words.setdefault(wid_, {"start": sp_.start * ratio, "end": sp_.end * ratio})
            d["start"] = min(d["start"], sp_.start * ratio)
            d["end"] = max(d["end"], sp_.end * ratio)

        toks = text.upper().split()
        items = [
            {"w": toks[i], "start": round(words[i]["start"], 3), "end": round(words[i]["end"], 3)}
            for i in sorted(words)
        ]
        out[uid] = items
        print(f"  {uid}: {len(items)} words  last_end={items[-1]['end']:.2f}s", flush=True)

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
