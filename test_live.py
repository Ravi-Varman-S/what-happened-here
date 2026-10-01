"""Exercise LiveLabeller without a microphone: feed the street clip chunk by
chunk at real-time speed and check the label stream behaves."""
import time

import numpy as np
import soundfile as sf

import record

audio, sr = sf.read("audio/street_3min.wav", dtype="float32", always_2d=True)
audio = audio.mean(axis=1)
start = int(60 * sr)                                # a busier stretch
audio = audio[start: start + int(20 * sr)]          # seconds 60-80

lab = record.LiveLabeller(enabled=True)
print("initial display :", lab.display())

step = int(0.05 * sr)
t0 = time.monotonic()
seen = []
scores = 0
for i in range(0, len(audio), step):
    lab.feed(audio[i:i + step], sr)
    d = lab.display()
    if not seen or seen[-1] != d:
        seen.append(d)
        scores += 1
    target = (i + step) / sr                         # pace like a live mic
    delay = target - (time.monotonic() - t0)
    if delay > 0:
        time.sleep(delay)
lab.stop()

st = lab.state
print("ready           :", st["ready"])
print("error           :", st["error"])
print("final display   :", lab.display())
print("label changes   :", len(seen))
print("history         :", seen)
print("tally           :", lab.tally_text())
print("label, conf, flg:", st["label"], round(st["conf"], 3), st["flag"])

ok = (st["error"] is None and st["ready"] and bool(st["tally"])
      and sum(st["tally"].values()) > 8
      and any(l not in ("Silent",) and "(?)" not in l for l in seen))
print("\n" + ("LIVE LABELS OK" if ok else "LIVE LABELS FAILED"))
raise SystemExit(0 if ok else 1)
