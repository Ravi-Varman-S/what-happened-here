# What Happened Here? — approach, failures, improvements

*A short write-up of how the tool works, where it breaks, and what I would do
next.*

---

## Approach

The pipeline is four steps in one file (`what_happened_here.py`), each one
feeding the next:

**1. Load & clean.** Decode with `soundfile`, falling back to a bundled ffmpeg
binary for containers it can't read (Speex/Opus OGG, MP3) → average channels to
mono → resample to 16 kHz (YAMNet's native rate) → RMS-normalise to −26 dB with
a 0.99 peak ceiling. Normalising every input to the same loudness is what lets
one fixed dB threshold work across a quiet bedroom and a busy street.

**2. Detect start/stop — written by hand.** The signal is cut into 25 ms frames
every 10 ms and each frame's RMS is converted to decibels, giving a loudness
curve. A rolling **25th percentile over 5 s** estimates the *ambient* floor of
each neighbourhood of the recording — a low percentile rather than a median
because a sound that stays loud for several seconds must still be measured
against background, not against itself. An event **opens** 6 dB above that floor
and **closes** 4 dB below it: two levels (hysteresis) so a wobbling sound cannot
flicker. On top of that, the level must hold for 0.10 s before an event opens
and 0.10 s before it closes (attack/release debounce). Finally gaps < 0.40 s are
merged and blips < 0.10 s dropped. No library event detector is used anywhere —
this is the requirement, and it is ~100 lines of NumPy.

**3. Label.** YAMNet (pretrained, 521 AudioSet classes) runs once over the whole
recording and returns 521 probabilities every 0.48 s. Each detected event is
labelled by the **mean score of the windows whose centre falls inside it**, so
the answer describes the event rather than one arbitrary frame. Because step 2
only fires where sound is demonstrably present, the classes `Silence`, `Noise`
and `Static` are ruled out before ranking — but if YAMNet's own favourite *was*
one of them, the substituted label is flagged `(?)` instead of being presented
as a confident answer. The same flag marks anything below 0.20 confidence.

**4. Report.** A two-panel figure: mel spectrogram with shaded, numbered event
bands and packed text labels on top; underneath, the loudness curve with the
ambient floor and both thresholds, so every boundary in the figure can be
audited against the numbers that produced it. Plus a fixed-width table,
`events.csv` (with runner-up labels), and `summary.txt` with counts, speech
coverage and total loud/quiet time.

**Bonus statistic:** speech coverage is computed frame-by-frame (any window with
`Speech > 0.30`, × 0.48 s), independent of the event list, so speech the
detector failed to segment still counts.

**One-command run:** `run.bat` (Windows) / `run.sh` (macOS/Linux) creates a
virtual environment if needed, installs `requirements.txt`, then analyses the
bundled demo clip — `run.bat record` records from the microphone first.

---

## Where it fails

- **Flat, low-frequency rumble.** The detector is energy-only, so a fan, an air
  conditioner, mic handling noise or wind — anything with a strong 20–200 Hz
  component — modulates the energy curve and opens events that contain nothing
  recognisable. In a quiet room this is the dominant error: YAMNet answers
  *Silence* for those windows, which is why the `(?)` flag exists at all. A
  spectral-flux or band-limited detector would ignore them.
- **Thresholds are tuned, not learned.** The defaults (6/4 dB, 0.10 s
  attack/release, 25th percentile over 5 s) were picked by sweeping roughly 20
  combinations on three clips — two Wikimedia samples and one microphone
  recording. That is a very small, non-representative sample; they are a
  reasonable middle ground, not an optimum, and they will mis-tune on music,
  rain, or a recording with a strongly changing background.
- **Labelling granularity.** YAMNet's 0.48 s hop and 0.96 s window mean events
  shorter than ~0.5 s are labelled from a single window that is mostly *not* the
  event; a 0.13 s clink gets whatever else is in that window. Its 521 classes
  also have no notion of *what kind* of event the boundaries describe — a door
  slam and a book drop are simply *Impact*.
- **No ground truth.** There is no labelled test set, so "33 events" cannot be
  scored as precision/recall — only eyeballed against the spectrogram. I cannot
  currently say the detector is *x%* accurate, only that it produces plausible,
  auditable boundaries on the recordings I have.
- **Assumptions that don't always hold.** Mono/stationary microphone; a single
  dominant sound at a time (two overlapping sounds merge into one event);
  the whole file is one acoustic scene, so a recording with a loud section
  followed by a quiet one is judged against a baseline that spans both.
- **Cosmetic:** spectrogram label widths are estimated from character count
  (0.58 × font size per glyph), so a few labels can overlap at extreme zoom;
  very dense passages fall back to the table.

---

## What I would improve

1. **Detect in the frequency domain.** Spectral flux / onset detection (or a
   band-weighted loudness curve that ignores <200 Hz) would remove the
   rumble-triggered false events — the single biggest win on quiet recordings.
2. **Evaluate properly.** Hand-label 10 minutes of recordings with true
   event boundaries and classes, then report boundary error in ms, precision /
   recall at a ±0.2 s tolerance, and label accuracy. That turns parameter
   choices from judgement into measurement, and would catch overfitting.
3. **Learn the parameters.** Grid-search the margins/debounce against that
   ground truth, or make the floor adaptive (longer window, per-band
   percentiles) so fewer knobs exist at all.
4. **Better labelling.** For short events, feed YAMNet only the event segment
   (zero-padded and scored) instead of a window shared with its surroundings;
   or move to a stronger model (PANNs, AST, CLAP) with the same wrapper. Add
   per-event onset/offset classes so the label describes the *transition*.
5. **Split events by content, not just energy.** Re-segment each event at
   internal spectral changes (e.g. novelty matrix peaks) so one long burst can
   contain several distinct sounds.
6. **Engineering:** a `--json` output, a `pytest` suite with fixture
   waveforms for the detector (impulses, gaps, ramps), streaming analysis for
   files longer than 3 minutes, and packaging as a real `pip`-installable
   console script.

---

## Honest assessment

The energy detector does what the brief asks — transparent, hand-written,
auditable boundaries — and the second plot panel is the proof: you can see
exactly why each event starts and stops. Labelling is reliable on clear,
isolated sounds at ≥0.5 s (street vehicles, coughs, whistles all come back at
0.7–0.99 confidence) and unreliable on short or buried ones, which the tool
admits with `(?)` rather than guessing confidently. The weakest link is the
energy-only detection in the presence of steady low-frequency noise.
