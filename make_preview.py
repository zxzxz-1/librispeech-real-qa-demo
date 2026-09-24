#!/usr/bin/env python3
"""Screenshot the demo for the README preview.

The page drives its animation from ``requestAnimationFrame`` reading
``audio.currentTime``, and the audio element is created with ``new Audio()`` (so it
is not in the DOM). To get a deterministic frame we install a stub ``Audio`` before
the page script runs, hand it a clock of our choosing, and let the real render loop
paint from it. That exercises the same code path the browser does, just without
waiting 6 seconds of wall clock.
"""
import pathlib
import sys

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent
PAGE = ROOT / "index.html"
OUT = ROOT / "preview.png"

# Stop the clock just past the last emission of the first case, so the card shows
# both streams fully populated, plus the timeline and output log.
STOP_AT = 6.0

# A minimal Audio stand-in: enough for the player, with the clock under our control.
AUDIO_STUB = """
(() => {
  const real = window.Audio;
  window.__audios = [];
  window.Audio = function (src) {
    const a = {
      src, paused: true, ended: false, currentTime: 0, duration: 0, preload: '',
      onpause: null, onended: null,
      play() { a.paused = false; a.ended = false; return Promise.resolve(); },
      pause() { a.paused = true; if (a.onpause) a.onpause(); },
      load() {},
      addEventListener() {}, removeEventListener() {},
    };
    window.__audios.push(a);
    return a;
  };
  window.Audio.prototype = real ? real.prototype : {};
})();
"""


def main() -> int:
    if not PAGE.exists():
        print(f"missing {PAGE}", file=sys.stderr)
        return 1

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1120, "height": 1150},
                                device_scale_factor=2)
        page.add_init_script(AUDIO_STUB)
        page.goto(PAGE.as_uri())
        page.wait_for_timeout(500)

        # start the first card so the render loop is running
        page.eval_on_selector(".card button.play", "b => b.click()")
        page.wait_for_timeout(150)

        # move the stub clock and let a few frames paint from it
        page.evaluate(
            """(t) => {
                 const a = window.__audios[0];
                 a.duration = t + 1.5;
                 a.currentTime = t;
               }""",
            STOP_AT,
        )
        page.wait_for_timeout(400)

        page.screenshot(path=str(OUT))
        browser.close()

    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
