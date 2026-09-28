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

The level meter runs while recording so you can tell the microphone is live.
Press Ctrl+C to stop early; whatever was captured is still saved.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

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
# Recording
# --------------------------------------------------------------------------- #

def record(seconds: float, device: int | None, samplerate: float | None
           ) -> tuple[np.ndarray, float]:
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
                print(f"\r  [{meter}] {done:6.1f}/{seconds:.0f}s   "
                      f"peak {db:6.1f} dB{clip}   (Ctrl+C to stop early)",
                      end="", flush=True)
                if done >= seconds:
                    break
        except KeyboardInterrupt:
            stop_now["flag"] = True
            print("\n  stopped early", end="")
        finally:
            elapsed = time.monotonic() - start
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

    if args.lead_in > 0:
        print("  starting in:", end=" ", flush=True)
        try:
            for n in range(int(args.lead_in), 0, -1):
                print(f"{n}...", end=" ", flush=True)
                time.sleep(1)
            print("go")
        except KeyboardInterrupt:
            print("\n  cancelled")
            return 130

    audio, rate = record(args.seconds, device, rate)

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
