# What Happened Here?

Give it a 2–3 minute recording of an everyday setting and it tells you **what
happened and when** — as a labelled spectrogram and as a timing table.

## Run it — one command

```
run.bat            (Windows)      ./run.sh            (macOS / Linux)
```

That single command creates a virtual environment on first run, installs
`requirements.txt`, analyses the bundled demo clip and prints the event table —
afterwards it just runs. Other things you can pass to the same command:

```
run.bat my.wav                      analyse your own file
run.bat my.wav my_out               ... into a different output folder
run.bat record                      record 3 min from the mic, then analyse
run.bat record --seconds 150 --device 1
```

Prefer to do it by hand? `pip install -r requirements.txt`, then
`python what_happened_here.py audio\street_3min.wav --outdir output_street`.

See **[APPROACH.md](APPROACH.md)** for the written explanation: how it works,
where it fails, and what I would improve.

## Record your own

```
python record.py                 # 180 s countdown -> records -> analyses
python record.py --seconds 150   # 2 min 30 s
python record.py --list-devices  # show microphones
python record.py --device 5 --lead-in 0 --no-analyze
```

`record.py` shows a live level meter while it captures (with a CLIPPING
warning), writes a 16-bit WAV into `audio\`, then hands the file straight to
the analyser. **Ctrl+C stops early and still saves.** Everything is 2–3
minutes by default, matching the brief.

## Analyse an existing file

```
python what_happened_here.py audio\street_3min.wav --outdir output_street
```

## What it does

| # | Step | How |
|---|------|-----|
| 1 | **Load & clean** | decode (soundfile, ffmpeg fallback) → mono → resample to 16 kHz → RMS-normalise to −26 dB with a peak ceiling |
| 2 | **Detect start/stop** | *hand-written*: 25 ms frames / 10 ms hop → RMS → dB → rolling 25th-percentile ambient floor → **hysteresis** (start at +6 dB, stop at +4 dB) → merge gaps < 0.40 s → drop blips < 0.10 s |
| 3 | **Label** | YAMNet (pretrained, 521 AudioSet classes) run once over the file; each event is labelled by the mean score of the analysis windows that fall inside it |
| 4 | **Report** | two-panel figure (spectrogram with shaded, numbered events + the loudness curve that produced them), a fixed-width table, `events.csv`, `summary.txt` |
| + | **Bonus stats** | counts per label, speech coverage %, total loud vs quiet time |

Step 2 is deliberately **not** a library call — the detector is implemented in
this file from energy over time, as the task requires.

## Files

```
run.bat / run.sh               one-command setup + run
what_happened_here.py          the tool (single file, CLI)
record.py                      recorder: 2-3 min from the mic, live level meter
APPROACH.md                    write-up: approach, failures, improvements
README.md, requirements.txt
audio\street_3min.wav          sample: street outside a window, 3:00, public domain
audio\kitchen_3min.wav         sample: kitchen ambience, 3:00, public domain
output_street\                 spectrogram.png, events.csv, events.txt, summary.txt
output_kitchen\                same, for the kitchen clip
```

## CLI

```
python what_happened_here.py AUDIO [--outdir DIR] [--duration N]
                             [--min-duration S] [--merge-gap S]
                             [--on-margin DB] [--off-margin DB]
                             [--attack S] [--release S] [--skip-model]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--duration` | 0 (whole file) | analyse only the first N seconds |
| `--min-duration` | 0.10 | ignore events shorter than this |
| `--merge-gap` | 0.40 | glue events separated by less than this |
| `--on-margin` | 6.0 | dB above the ambient floor that **starts** an event |
| `--off-margin` | 4.0 | dB above the ambient floor that **ends** it |
| `--attack` | 0.10 | seconds the level must stay high before an event opens |
| `--release` | 0.10 | seconds the level must stay low before it closes |
| `--skip-model` | off | detection only — no TensorFlow, no network |

Labels ending in **`(?)`** are the model's best guess when it was unsure, or
when its own favourite answer was *Silence/Noise/Static* — classes the detector
has already ruled out.

## The detector in one paragraph

The signal is cut into 25 ms frames every 10 ms and each frame's RMS is turned
into decibels, giving a loudness curve. A rolling low-percentile over 5 s
estimates the *ambient* floor of each neighbourhood of the recording (a low
percentile, rather than the median, deliberately ignores the loud parts so that
a long event is still measured against background). Loudness crossing **6 dB
above** that floor opens an event; dropping back **under 4 dB** closes it. The
two separate levels are hysteresis — without them a noisy event would flicker
into dozens of micro-events. Segments separated by less than 0.40 s are merged
(a clinking cutlery burst becomes one event rather than twenty), anything
shorter than 0.10 s is discarded.

## Sample audio credits

Both samples come from Wikimedia Commons and are public domain (PDSounds
transfer):

* *Sunday in the city street noise1* by *cori* — cars, seagulls, wood pigeons,
  planes over a city-by-the-sea street.
* *Ambient Kitchen Sounds* — kitchen ambience.

## Requirements

```
pip install -r requirements.txt
```

TensorFlow downloads YAMNet from TensorFlow Hub on first run (cached afterwards
in `%TEMP%\tfhub_modules`). The 521 class names ship inside the model as an
asset, so labelling needs no extra network calls.
