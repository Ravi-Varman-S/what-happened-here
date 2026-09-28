# What Happened Here? — how it works, what it gets wrong, what I'd fix

## The idea

The brief was simple enough to sound easy: record two or three minutes of
something ordinary — a kitchen, a street, an office — and tell me what happened
and when. I ended up with a single Python file that plays the recording,
chops it into sound events, names each one, and hands you a picture and a table
to check its work against.

The one rule I took seriously from the start was that finding the *boundaries*
had to be my own code. So there's no library hiding in step two. The detector
is about a hundred lines of NumPy, and you can read every line of it.

## How it actually works

Everything lives in `what_happened_here.py`, and it runs in four steps.

First, the audio gets tidied up. Decode it (with a bundled ffmpeg sitting
behind `soundfile` for the odd formats that refuse to open), average the
channels down to mono, resample to 16 kHz, then normalise the whole thing to
−26 dB RMS with a peak ceiling so nothing ever clips. That last part matters
more than it sounds: once every recording is at the same loudness, one fixed
dB threshold can work on a quiet bedroom and a busy street alike.

Then comes the detector, which is the part I'd point you at first. I cut the
signal into 25 ms frames, stepping 10 ms at a time, and turn each frame's RMS
into decibels. That gives a loudness curve — basically a line that says how
loud the room is, a hundred times a second. From there I need an answer to one
question: louder than *what*?

My answer is a rolling 25th percentile over five seconds. Why 25 and not the
median? Because a lorry can rumble past for eight seconds, and if the
"background" estimate includes it, the threshold drifts up to meet the sound
and the event never registers. A low percentile deliberately ignores the loud
stuff and stays anchored to the quiet parts of each neighbourhood of the
recording.

An event opens once the loudness climbs 6 dB above that floor, and closes once
it falls back under 4 dB. Two levels rather than one — hysteresis, if you want
the technical word — because with a single threshold a noisy event flickers in
and out and turns one door slam into forty micro-events. On top of that I added
a debounce: the level has to hold for 0.10 s before an event opens, and 0.10 s
of quiet before it closes. The timestamp still goes at the *first* qualifying
frame, so the debounce delays the decision but never the reported time.
Finally I glue together anything separated by less than 0.40 s and throw away
blips shorter than 0.10 s.

Naming the events is YAMNet's job. It's a pretrained model with 521 sound
classes, and I run it once over the whole file rather than once per event. It
hands back 521 probabilities every 0.48 s; each of my events is labelled by the
average of the windows whose centre falls inside it, so the answer describes
the event and not some arbitrary frame.

There's a wrinkle worth mentioning. On quiet recordings the detector
occasionally opens an event that contains nothing recognisable — a fan
changing speed, someone shifting in a chair — and YAMNet, being honest, answers
"Silence". That's a contradiction: I already proved with the loudness curve
that sound is present. So Silence, Noise and Static are ruled out before
ranking. But I don't want to quietly swap in a second-best answer and pretend
it was confident, so if the model's own favourite was one of those three, the
label gets a `(?)` appended. Anything under 0.20 confidence gets the same mark.
You can see at a glance which labels to trust.

The last step is showing the work. There's a two-panel figure: the top is a
mel spectrogram with every event shaded and numbered, and underneath is the
loudness curve alongside the ambient floor and both thresholds. That second
panel is my favourite part of the project, because you can literally watch each
event start where the grey line crosses the green dashed one and end where it
drops below the red dotted one. Nothing is hidden. Next to the figure you get
a plain table, a CSV with the runner-up labels, and a summary with counts and
speech coverage.

On the speech number specifically: I count it frame by frame rather than
through the events, so speech that the detector failed to segment still shows
up in the total.

## Where it falls over

The honest version, in rough order of how much it bugs me:

**Low-frequency rumble is the big one.** The detector only looks at energy, so
a fan, an air conditioner, mic handling noise or wind will happily open an
event. In a quiet room this is the dominant failure — the loudness curve wobbles,
an event opens, and YAMNet says "Silence" because there's genuinely nothing
there to name. That's exactly where the `(?)` flag earns its keep, but it's
still a false positive, and it's the first thing I'd fix.

**The thresholds are tuned, not learned.** I picked 6/4 dB, 0.10 s
attack/release, the 25th percentile and the 5-second window by sweeping around
twenty combinations across three recordings — two public-domain samples and one
microphone capture of my own room. Three clips is not much of a sample set.
These defaults are a reasonable middle ground; I wouldn't claim they're an
optimum, and they'd probably misbehave on music, rain, or a room where the
background changes halfway through.

**I can't quote you an accuracy number.** There's no ground truth. I can show
you that the boundaries are plausible and auditable against the spectrogram,
and I did check every event on the demo clips, but I can't tell you the
detector is 92% accurate because I never labelled a test set to measure it
against. That's the biggest gap between this and something I'd call finished.

**Short events get labelled badly.** YAMNet's window is 0.96 s long and it
steps 0.48 s at a time. A 0.13 s clink therefore gets one window that is mostly
*not* the clink, so it inherits whatever else is happening in that second. The
521 classes also don't distinguish a door slam from a book dropping — both come
back as something like "Impact".

**Assumptions that don't always hold.** One microphone, roughly stationary.
One dominant sound at a time, so two overlapping sounds merge into one event.
And one acoustic scene per file, which means a recording that starts loud and
ends quiet is judged against a baseline that spans both halves.

On a cosmetic note, the labels on the spectrogram are placed by estimating
their width from the character count. It works well at normal density, but in
the busiest passages they can overlap — the table always has the full list.

## What I'd do next

If I had another week, the order would be:

Detect in the frequency domain. Spectral flux, or simply a loudness curve that
weights everything below 200 Hz lightly, would kill the rumble problem in one
move and it's the single biggest win available.

Then build some ground truth. Hand-label ten minutes of recordings with real
boundaries and classes, and report boundary error in milliseconds plus
precision and recall at a ±0.2 s tolerance. That turns every parameter choice
from a judgement call into something I can measure — and it would tell me
quickly whether I've overfitted to three clips.

With measurement in place, let the parameters follow from it: grid-search the
margins and debounce against that set, or make the floor adaptive enough that
there are fewer knobs to turn at all.

For the labelling, feed YAMNet only the event itself (padded, scored, done)
instead of a shared window, or swap in something newer like PANNs or an audio
transformer behind the same small wrapper. It'd also be worth labelling the
*transition* — onset versus offset — rather than just the stretch of sound.

On the engineering side: a `--json` output so other tools can consume this, a
pytest suite with generated fixture waveforms (impulses, gaps, ramps) to pin
the detector's behaviour down, streaming analysis for files longer than three
minutes, and proper packaging so it installs as a console command.

## What I'd say about it today

The detector does the job it was asked to do, and the second plot panel is the
proof — the reasoning is visible rather than asserted. Labelling is reliable
when the sound is clear and lasts more than half a second; the street clip's
vehicles and the whistles and coughs in my own recording all come back between
0.7 and 0.99 confidence. It's unreliable on short or buried sounds, and in
those cases it says so instead of bluffing. The weak link is energy-only
detection around steady low-frequency noise, and I'd start there.

To check it end to end, `run.bat` builds a clean virtual environment, installs
`requirements.txt` and runs the demo — I tested that path from scratch and got
the same 33 events as the local run, so the one-command claim isn't just hope.
