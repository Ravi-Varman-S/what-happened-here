r"""
What Happened Here? — recorder
==============================

Records 2–3 minutes of an everyday setting from a microphone, then hands the
file to the analysis tool.

    python record.py                          # 180 s, default mic, then analyse
    python record.py --seconds 150            # 2 min 30 s
    python record.py --list-devices           # show microphones
    python record.py --device 2 --lead-in 0   # pick a mic, no countdown
    python record.py --no-analyze             # just record
    python record.py --no-live-labels         # meter only, no YAMNet watching

The level meter runs while recording so you can tell the microphone is live,
and so does a **live label**: as you record, the meter shows what YAMNet hears
right now — ``LIVE: Speaking 92%``, ``LIVE: Dog barking 71%``, or ``LIVE:
Silent`` when the level falls back under the ambient floor +6 dB (the same
rule the detector uses, so the live readout and the final report agree).
Press Ctrl+C to stop early; whatever was captured is still saved, and a short
tally of what was heard is printed at the end.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
import queue
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

import what_happened_here as eng

PREFERRED_RATES = (48_000, 44_100, 32_000, 16_000)
MIN_SECONDS = 1
MAX_SECONDS = 600
TASK_MIN, TASK_MAX = 120, 180          # "2-3 minutes" from the brief


# --------------------------------------------------------------------------- #
# Device helpers
# --------------------------------------------------------------------------- #

def list_devices() -> int:
    """Print every recording-capable device. Returns the default input index."""
    default_in = sd.default.device[0]
    print("Input (recording) devices:\n")
    found = False
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            found = True
            mark = "*" if i == default_in else " "
            print(f"  [{i:2d}] {mark} {d['name'][:56]:56s} "
                  f"channels={d['max_input_channels']} "
                  f"sr={d['default_samplerate']:.0f}")
    if not found:
        print("  (none found)")
    else:
        print(f"\n  * = default input device ({default_in})")
    return default_in


def pick_device(spec: str | None) -> int:
    """Turn a --device argument (index or name fragment) into a device index.

    Falls back to the system default input device when nothing is specified.
    """
    if spec is None:
        return int(sd.default.device[0])
    if spec.lstrip("-").isdigit():
        return int(spec)
    matches = [i for i, d in enumerate(sd.query_devices())
               if d["max_input_channels"] > 0 and spec.lower() in d["name"].lower()]
    if not matches:
        names = [d["name"] for i, d in enumerate(sd.query_devices())
                 if d["max_input_channels"] > 0]
        print(f"error: no input device matching {spec!r}", file=sys.stderr)
        print("  available: " + "; ".join(names[:12]), file=sys.stderr)
        raise SystemExit(1)
    return matches[0]


def choose_samplerate(device: int, requested: float | None) -> float:
    """Use the requested rate, else the first preferred rate the mic accepts."""
    dev = sd.query_devices(device)
    default_rate = float(dev["default_samplerate"])
    if requested:
        return float(requested)
    for rate in PREFERRED_RATES:
        try:
            sd.check_input_settings(device=device, channels=1, samplerate=rate)
            return float(rate)
        except Exception:
            continue
    return default_rate


# --------------------------------------------------------------------------- #
# Live labelling — "what is the mic hearing *right now*?"
# --------------------------------------------------------------------------- #

LIVE_WINDOW = 0.96        # seconds of audio each live score looks at
LIVE_HOP = 0.48           # how often the label refreshes (YAMNet's own hop)
LIVE_FLOOR_SEC = 5.0      # same rolling-floor length the detector uses


class LiveLabeller:
    """Keeps a current "what's that?" label while the recording is running.

    The audio callback must never block, so it only drops copied chunks into a
    queue.  A worker thread loads YAMNet once (a few seconds — it starts while
    the lead-in countdown is still ticking) and then every ``LIVE_HOP``
    seconds scores the last ``LIVE_WINDOW`` seconds of audio, exactly like the
    final analysis will.

    Silence is decided the same way Step 2 decides it: the level has to fall
    below the rolling 25th-percentile floor + ``ON_MARGIN_DB``.  Labels get the
    same treatment as ``label_events()`` — Silence/Noise/Static are blocked
    before ranking, and low-confidence or substituted answers carry a `` (?)``.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self.state: dict = {
            "ready": False, "label": "loading YAMNet…", "conf": 0.0,
            "flag": False, "silent": False, "level_db": -90.0,
            "tally": {}, "error": None,
        }
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="live-labeller")
        if enabled:
            self._thread.start()

    # ---- called from the audio callback: copy, don't block ---------------- #
    def feed(self, chunk: np.ndarray, samplerate: int) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait((chunk.copy(), int(samplerate)))
        except queue.Full:
            pass                      # drop a chunk rather than stall the mic

    def stop(self) -> None:
        self._stop.set()
        if self.enabled:
            self._thread.join(timeout=2.0)

    def display(self) -> str:
        """What the meter line should show right now."""
        st = self.state
        if not self.enabled:
            return "live labels off"
        if st["error"]:
            return f"labels unavailable"
        if not st["ready"]:
            return st["label"]                      # "loading YAMNet…"
        if st["label"] == "listening…":
            return st["label"]                      # nothing scored yet
        if st["silent"]:
            return "Silent"
        return f"{st['label']}{' (?)' if st['flag'] else ''} {st['conf']:.0%}"

    def tally_text(self) -> str:
        """A short "what I heard" line for after the recording."""
        st = self.state
        if not self.enabled or st["error"] or not st["tally"]:
            return ""
        heard = sorted(st["tally"].items(), key=lambda kv: -kv[1])[:8]
        return ", ".join(f"{name} {secs:.0f}s" for name, secs in heard)

    # ---- the worker -------------------------------------------------------- #
    def _run(self) -> None:
        try:
            model = eng.YamnetLabeller()            # a few seconds, once
        except Exception as exc:
            self.state = {**self.state, "error": str(exc),
                          "label": "model failed to load"}
            return

        labels = model.labels
        blocked = [i for i, n in enumerate(labels) if n in eng.NON_EVENT_LABELS]

        buf = np.zeros(0, dtype=np.float32)         # device-rate rolling audio
        dev_sr = eng.TARGET_SR
        levels: list[float] = []                    # recent level readings (dB)
        tally: dict[str, float] = {}                # label -> seconds heard
        recent: deque[str] = deque(maxlen=3)        # stability filter
        last_score = time.monotonic()
        self.state = {**self.state, "ready": True, "label": "listening…"}

        while not self._stop.is_set():
            try:
                chunk, dev_sr = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            buf = np.concatenate([buf, chunk]) if buf.size else chunk
            keep = int(LIVE_WINDOW * dev_sr) * 2
            if buf.size > keep:
                buf = buf[-keep:]

            now = time.monotonic()
            if now - last_score < LIVE_HOP:
                continue
            last_score = now
            if buf.size < int(0.3 * dev_sr):
                continue                              # too little audio yet

            # level of the freshest quarter-second
            tail = buf[-int(0.25 * dev_sr):]
            level_db = float(20 * np.log10(
                np.sqrt(np.mean(tail.astype(np.float64) ** 2)) + 1e-9))
            levels.append(level_db)
            if len(levels) >= 3:
                floor = float(np.percentile(levels[-int(LIVE_FLOOR_SEC / LIVE_HOP):],
                                            25))
                silent = level_db < floor + eng.ON_MARGIN_DB
            else:
                silent = level_db < -45.0             # not enough history yet

            # score the last LIVE_WINDOW seconds, exactly as Step 3 will
            window = buf[-int(LIVE_WINDOW * dev_sr):]
            x16 = (eng.resample(window, dev_sr, eng.TARGET_SR)
                   if dev_sr != eng.TARGET_SR else window)
            need = int(eng.MIN_SEGMENT_SEC * eng.TARGET_SR)
            if x16.size < need:              # YAMNet needs a whole 0.96 s
                x16 = np.pad(x16, (0, need - x16.size))

            try:
                scores, _ = model.scores(x16.astype(np.float32), eng.TARGET_SR)
                if len(scores) == 0:
                    continue
                row = scores.mean(axis=0)             # average the chunk's windows
            except Exception as exc:                  # never kill the recording
                self.state = {**self.state, "error": str(exc)}
                continue

            raw = int(np.argmax(row))
            ranked = row.copy()
            ranked[blocked] = -1.0                    # silence can never win
            best = int(np.argmax(ranked))
            conf = float(row[best])
            flag = conf < eng.MIN_CONFIDENCE or raw in blocked

            name = "Silent" if silent else labels[best]
            tally[name] = tally.get(name, 0.0) + LIVE_HOP

            # show a label only once it repeats, so it doesn't strobe
            recent.append(name)
            counts = Counter(recent)
            top, n = counts.most_common(1)[0]
            shown = top if n >= 2 else recent[-1]

            self.state = {
                "ready": True, "label": shown, "conf": conf, "flag": flag,
                "silent": silent, "level_db": level_db,
                "tally": dict(tally), "error": None,
            }


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #

def record(seconds: float, device: int | None, samplerate: float | None,
           labeller: LiveLabeller | None = None) -> tuple[np.ndarray, float]:
    """Capture mono audio. Returns ``(samples, samplerate)``."""
    rate = int(round(choose_samplerate(device, samplerate)))
    channels = 1
    total = int(round(seconds * rate))
    buffer = np.zeros(total, dtype=np.float32)
    written = {"n": 0, "peak": 0.0, "clipped": False}
    stop_now = {"flag": False}

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n  ! {status}", file=sys.stderr)
        n = min(frames, total - written["n"])
        if n <= 0:
            stop_now["flag"] = True
            return
        chunk = indata[:n, 0]
        buffer[written["n"]: written["n"] + n] = chunk
        written["n"] += n
        written["peak"] = max(written["peak"], float(np.max(np.abs(chunk))))
        if written["peak"] >= 0.999:
            written["clipped"] = True
        if labeller is not None:
            labeller.feed(chunk, rate)          # watch it live (never blocks)
        if written["n"] >= total:
            stop_now["flag"] = True

    bar_len = 30
    with sd.InputStream(device=device, samplerate=rate, channels=channels,
                        dtype="float32", blocksize=0, callback=callback):
        start = time.monotonic()
        try:
            while not stop_now["flag"]:
                time.sleep(0.2)
                done = written["n"] / rate
                level = written["peak"]
                filled = int(round(bar_len * min(1.0, done / seconds)))
                db = 20 * np.log10(level + 1e-9)
                meter = "#" * filled + "-" * (bar_len - filled)
                clip = "  CLIPPING - move away from the mic!" if written["clipped"] else ""
                live = (f"   LIVE: {labeller.display()}"
                        if labeller is not None else "")
                print(f"\r  [{meter}] {done:6.1f}/{seconds:.0f}s   "
                      f"peak {db:6.1f} dB{live}{clip}   (Ctrl+C to stop early)",
                      end="", flush=True)
                if done >= seconds:
                    break
        except KeyboardInterrupt:
            stop_now["flag"] = True
            print("\n  stopped early", end="")
        finally:
            elapsed = time.monotonic() - start
            if labeller is not None:
                labeller.stop()              # let the last score finish
        print()

    n = written["n"]
    if n == 0:
        print("error: no audio captured", file=sys.stderr)
        raise SystemExit(1)
    return buffer[:n].copy(), rate


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Record 2-3 minutes of an everyday setting, "
                    "then analyse what happened.")
    p.add_argument("--seconds", type=float, default=180.0,
                   help="Recording length in seconds (default 180 = 3 min)")
    p.add_argument("--device", default=None,
                   help="Input device index or name fragment (see --list-devices)")
    p.add_argument("--samplerate", type=float, default=None,
                   help="Force a sample rate (default: best the mic supports)")
    p.add_argument("--lead-in", type=float, default=3.0,
                   help="Countdown before recording starts (default 3 s)")
    p.add_argument("--out", default=None, help="Output .wav path")
    p.add_argument("--outdir", default="output_recording",
                   help="Where the analysis writes its results")
    p.add_argument("--list-devices", action="store_true",
                   help="List microphones and exit")
    p.add_argument("--no-analyze", action="store_true",
                   help="Record only, skip the analysis")
    p.add_argument("--no-live-labels", action="store_true",
                   help="Show only the level meter while recording "
                        "(no YAMNet watching the mic)")
    args = p.parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    if not (MIN_SECONDS <= args.seconds <= MAX_SECONDS):
        print(f"error: --seconds must be between {MIN_SECONDS} and {MAX_SECONDS}",
              file=sys.stderr)
        return 1

    if not (TASK_MIN <= args.seconds <= TASK_MAX):
        print(f"  note: the brief asks for 2-3 minutes; you asked for "
              f"{args.seconds:.0f} s\n")

    device = pick_device(args.device)
    dev_name = sd.query_devices(device)["name"]
    rate = int(round(choose_samplerate(device, args.samplerate)))

    print("=" * 66)
    print("RECORDING")
    print("=" * 66)
    print(f"  device    : {dev_name[:60]}")
    print(f"  length    : {args.seconds:.0f} s at {rate:.0f} Hz, mono")

    # Start the live labeller now and wait for YAMNet: loading it mid-recording
    # hogs the CPU (PortAudio overflows) and wastes the first seconds of audio.
    labeller = LiveLabeller(enabled=not args.no_live_labels)
    if labeller.enabled:
        print("  live label: loading YAMNet…", end="", flush=True)
        t0 = time.monotonic()
        while (not labeller.state["ready"] and not labeller.state["error"]
               and time.monotonic() - t0 < 30.0):
            time.sleep(0.2)
        if labeller.state["error"]:
            print(f" failed ({labeller.state['error'][:40]}) — recording anyway")
        else:
            print(f" ready in {time.monotonic() - t0:.1f}s — "
                  f"the meter will name what it hears")
    else:
        print("  live label: off (--no-live-labels)")

    if args.lead_in > 0:
        print("  starting in:", end=" ", flush=True)
        try:
            for n in range(int(args.lead_in), 0, -1):
                print(f"{n}...", end=" ", flush=True)
                time.sleep(1)
            print("go")
        except KeyboardInterrupt:
            print("\n  cancelled")
            labeller.stop()
            return 130

    audio, rate = record(args.seconds, device, rate, labeller)

    heard = labeller.tally_text()
    if heard:
        print(f"  live heard : {heard}")

    peak = float(np.max(np.abs(audio)))
    rms_db = 20 * np.log10(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)
    if peak < 0.005:
        print("  warning: the recording is almost silent - check the microphone "
              "or use --device to pick another one.", file=sys.stderr)

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("audio") / f"recording_{stamp}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.clip(audio, -1.0, 1.0)
    sf.write(str(out_path), (data * 32767).astype(np.int16), int(round(rate)),
             subtype="PCM_16")

    print(f"\n  saved     : {out_path}  ({len(audio)/rate:.1f} s, "
          f"peak {20*np.log10(peak+1e-9):.1f} dB, rms {rms_db:.1f} dB)")

    if args.no_analyze:
        print(f"\nNow analyse it with:\n"
              f"  python what_happened_here.py \"{out_path}\" --outdir {args.outdir}")
        return 0

    print("\n" + "=" * 66)
    print("ANALYSING")
    print("=" * 66)
    import what_happened_here as analysis
    return analysis.main([str(out_path), "--outdir", args.outdir])


if __name__ == "__main__":
    raise SystemExit(main())
