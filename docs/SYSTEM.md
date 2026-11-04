# How the system works

This page explains the whole chain, from a trajectory file to what you hear. For every option see
[REPLAY.md](REPLAY.md), for every OSC address [OSC.md](OSC.md), for the code structure
[ARCHITECTURE.md](ARCHITECTURE.md), and for the Max patches [MAX_SETUP.md](MAX_SETUP.md) and
[MAX_PATCH_CHANGES.md](MAX_PATCH_CHANGES.md).

## Contents

- [The idea](#the-idea)
- [Pipeline](#pipeline)
- [Sources and views](#sources-and-views)
- [Descriptors](#descriptors)
- [Scene quantities](#scene-quantities)
- [Sound profile synthfish](#sound-profile-synthfish)
- [Sound profile scene](#sound-profile-scene)
- [The feature-target territory and the RafaelV2 terrain](#the-feature-target-territory-and-the-rafaelv2-terrain)
- [Events and flashes](#events-and-flashes)
- [The Max side](#the-max-side)
- [Limits](#limits)

## The idea

Recorded crowds play music. A trajectory file (who was where, frame by frame) is replayed in real
time, and what the crowd does (how many people, how fast and how orderly they walk, how close they
are, who arrives, who starts walking together) drives **PLAUD**, a neural DDSP synthesis model, and
two small Max patches: a per-person granulator and a kick. Nobody plays notes. The picture shows
the same people, on a map or over the original video, so you can point at what you hear.

The project grew out of **SynthFish**, where a simulated fish school played PLAUD. The simulation
is still there and still sounds exactly as before.

## Pipeline

```
 trajectory file (ETH/UCY .txt, or CSV: id, frame or t, x, y)            simulation (boids)
   |  loader: frame step, units, gaps -> pieces                                |
   |  interpolation in data time (cubic Hermite / Catmull-Rom), 60 fps         |
   |  calibration: the file's p95 speed and acceleration                       |
   v                                                                           v
 agents on screen (up to 100, stable slots) ----------------------------- agents (the fish)
   |
   |-> descriptors (descriptors.py, 0..1, smoothed): count, mean_speed, energy, centroid, spread, ...
   |-> scene quantities (replay/mapping.py, profile scene): count, speed, disorder, closeness,
   |     pairs, arrivals, agitation, step rate
   v
 sound profile: synthfish (the simulation's mapping) or scene (the crowd's state)
   v
 OSC over UDP -> 127.0.0.1:7400, ~15 Hz (+ events when they happen)
   v
 Max/MSP
   |- PLAUD device: prior (territory) -> latents (+ dials, looper, posterior) -> nn~ decoder -> bending
   |- shoal_gran: one grain voice per person, panned; slices on arrivals
   '- shoal_fx: kick (+ PLAUD shake, reverb)
```

The window draws the same agents. Everything runs in **data time**: `SPACE` pauses the data (and
the sound state), `R` restarts the file, `--replay-speed` scales it.

## Sources and views

**Simulation source** (no `--replay`): the SynthFish boids, played with the mouse, the keyboard and
MIDI ([CONTROLS.md](CONTROLS.md)). Its output is fixed: every change in this project is checked
against hashes of the original OSC stream and pixels.

**Replay source** (`--replay FILE`):
- **Loader.** ETH/UCY `frame id x y` files (metres, 2.5 rows per second, frame step detected), or
  any CSV with `id`, `x`, `y` and `frame` or `t`. A track with a gap longer than one missing row is
  split into pieces.
- **Interpolation.** Each piece is a cubic Hermite (Catmull-Rom) curve in data time, sampled at 60
  fps. Velocity and acceleration come from the same curve.
- **Calibration.** The p95 of speed and acceleration over the whole file becomes the agents'
  `max_speed` and `max_force`, so `mean_speed` and agitation mean the same 0–1 in every scene.
- **Slots.** At most 100 agents (`--replay-max-agents`). An agent keeps its slot while its piece
  lasts, so grain voice i stays the same person.
- **No forces.** The mouse rings are drawn but move nobody: the data already contains the people's
  reactions.

**Views** (data look, `--replay-view`), display only except for the pan and events of profile scene:

| View | Background | Scenes |
|---|---|---|
| `map` | metric grid (1 m / 5 m) and scale bar | all |
| `camera` | OpenTraj reference frame, people projected through its homography | zara1, hotel |
| `video` | the original video at frame round(25 × data time) (needs OpenCV) | zara1, hotel |

Layers on any view: `N` social graph (lines between people closer than 2 m, thick when they walk
together), `L` long exposure (walked routes as light), `V` video look (normal / dark / off). People
are dots coloured by speed with a heading tick and the last 10 s of path; `I` labels, `H` HUD.

## Descriptors

`descriptors.py` (unchanged from SynthFish) measures the agents every frame; a one-pole smoother
(α 0.16 per frame, about 0.1 s) follows. In replay mode the reference speed and force are the
file's p95, so the values are comparable between scenes.

| Descriptor | Computation |
|---|---|
| `fish_count` | agents / 100 |
| `mean_speed` | mean \|v\| / max_speed (the p95 in replay) |
| `energy` (agitation) | mean \|a\| / (6 × max_force) |
| `center_x_raw`, `center_y_raw` | centroid / window size |
| `center_x`, `center_y` | the centroid expanded × 2.2 around 0.5 |
| `spread` | mean distance to the centroid / (0.28 × window diagonal) |
| `nearest_distance` | mean nearest-neighbour distance / 120 px; `density` = 1 − that |
| `direction_coherence` | length of the mean unit heading |
| `cluster_count` | occupied 180 px cells / 12 |

With `--replay-normalise` every descriptor is mapped to the file's p5–p95 and `fish_count` to
people / most at once.

## Scene quantities

Profile scene (`--replay-profile scene`) measures the crowd directly, in metres and m/s, in
`replay/mapping.py`. Positions and velocities come from the interpolated agents. Disorder,
closeness, the pair check and the territory lookup run at 10 Hz; the rest every frame.

| Quantity | How it is computed | Range and glide |
|---|---|---|
| **count** | people on screen, relative to the file: a log curve from the file's p5 (at least one person) to its p95 (at least two more), sampled every 0.1 s over the whole file at load | 0–1, instant (the sounds that use it glide). zara1 2–10 people, hotel 1–11, univ 27–65 |
| **speed** | mean \|v\| of everybody on screen, 0.4 → 1.5 m/s mapped to 0 → 1 | 0–1, glide τ 2 s |
| **disorder** | axial: 1 − the length of the mean of the walkers' doubled heading angles (walkers: faster than 0.25 m/s). People walking both ways along one line give 0 (a two-way street is ordered); headings everywhere give ~1. Fewer than two walkers: 0 | 0–1, glide τ 2 s |
| **closeness** | the mean distance from each person to their nearest neighbour, 2.5 m → 0.7 m mapped to 0 → 1. One person: 0 | 0–1, glide τ 2 s |
| **agitation** | mean \|a\| / the file's p95 acceleration | glide τ 4 s |
| **step rate** | mean speed of the walkers / 0.7 m (step length) | Hz; 0 when everybody stands |
| **visible** | a person is clearly in the picture: at least 40 px inside the frame (camera and video views) or 1.5 m inside the data's bounding box (map view) | yes / no |
| **arrival** | a person not seen before becomes visible (people present at the start are not arrivals) | event |
| **pair** | two visible people walk together for 2 s: closer than 2 m, both faster than 0.4 m/s, headings within 30°, speeds within 0.3 m/s (the same rule as the social graph) | event |
| **pan** | each person's column in the picture shown: the video column (through the camera homography) in the camera and video views, the map x otherwise | 0–1 per person |

When the scene is empty the glided values hold (loudness falls to 0 and presence to silence).
A pause holds everything; `R` and a loop start again from the scene as it is.

## Sound profile synthfish

The default (`--replay-profile synthfish`), and the only mapping of the simulation. The crowd is
treated like the fish school.

| Quantity | Formula | OSC | Max object / parameter | What you hear |
|---|---|---|---|---|
| centroid x | `center_x_raw` | `/plaud/x` | device → `s plaudX` → prior~ territory_x | PLAUD's style moves with where the crowd stands, left / right |
| centroid y | `center_y_raw` (0 = top) | `/plaud/y` | `s plaudY` → `!- 1.` → territory_y | the same, up / down |
| count | `0.6 + 0.8 × fish_count` | `/plaud/loudness` | Loudness dial (clipped to 1) and, through `scale 0.7 1.4 1. 0.`, the Limit dial | louder and fuller with more fish. In replays fish_count is ~0.1, so Limit sits at 1: a single partial |
| agitation | `clip(0.2 + 1.3 × energy)` | `/plaud/temp` | L2 dial (4th latent offset) | timbre moves with agitation |
| per-person x | x / window width, first 32 | `/gran/fish_x` | shoal_gran, one voice each (16 used) | grains panned where each person is |
| count | `100 × fish_count` | `/gran/density` | shoal_gran onset rate `0.5 + 0.075 × density` grains/s per voice | denser grains with more people |
| agitation, disorder | `clip(1.2 × energy + 0.7 × predator / 3 + 0.2 × (1 − direction_coherence))` | `/gran/noise` | shoal_gran onset jitter and region spread | rougher grains |
| acceleration event | onset 1, release 0; strength 0.3–1.0 | `/fx/predator`, `/fx/predator_strength` | shoal_fx | a kick at once and every 800 ms while it lasts, with a spectral shake of PLAUD |

## Sound profile scene

`--replay-profile scene`, optionally with `--replay-terrain` (the PLAUD model's terrain map, e.g.
`RafaelV2.json`, which comes with the model and is not in this repository). Same addresses where
possible, plus three that need one route in the PLAUD device ([MAX_PATCH_CHANGES.md](MAX_PATCH_CHANGES.md)).

| Quantity | Formula / curve / glide | OSC | Max object / parameter | What you hear |
|---|---|---|---|---|
| count, speed, disorder | feature targets → best terrain cell near the current one, re-picked at 10 Hz, glide τ 4 s ([below](#the-feature-target-territory-and-the-rafaelv2-terrain)) | `/plaud/x`, `/plaud/y` | prior~ territory_x / territory_y | busier and faster: denser, more rhythmic styles; disordered crossing: percussive, bright, noisy styles; ordered flow: fuller, louder ones |
| count | `0.15 + 0.85 × count`; 0 when empty; glide τ 1.5 s | `/plaud/loudness` | Loudness dial → 1st latent offset | PLAUD's loudness latent rises with the crowd |
| count, closeness | `clip(0.8 × (1 − count) + 0.3 × closeness)`; glide τ 1.5 s; held when empty | `/plaud/limit` | Limit dial "lim" → decoder `limit_components` (0 = all partials, 1 = one) | more people, more partials; people pressed together thin them again |
| closeness | `0.5 × closeness` | `/plaud/stretch` | SpectralStretch dial "str" → decoder `spectral_stretch` (+ = toward highs) | a crowded scene stretches the spectrum upward (tension) |
| speed | speed (0.4–1.5 m/s → 0–1) | `/plaud/centroid` | Centroid dial → 2nd latent offset | brighter when people hurry |
| agitation | `clip(0.2 + 1.3 × agitation)` | `/plaud/temp` | L2 dial → 4th latent offset | timbre moves with agitation, over seconds |
| absolute number of people | 0 when empty; otherwise a log curve up to −6 dB, 6 dB wide (one person −10.6 dB in a file with at most 20 at once); fade-in 1.5 s, fade-out 3 s | `/mix/master` | Dry `live.gain~` (−70..0 dB) | silence when nobody is there |
| pan | per person, from the picture shown | `/gran/fish_x` | shoal_gran voice pan | each grain voice sits where its person is in the picture |
| step rate | density so that `0.5 + 0.075 × density` = the step rate | `/gran/density` | shoal_gran onset rate | grains pulse at walking cadence (~1.9 Hz at 1.3 m/s) |
| agitation, disorder | as in synthfish | `/gran/noise` | shoal_gran | rougher grains |
| arrival | slice 1–4 by the newcomer's pan (left → right), at most one per 15 s | `/pad/sample` | shoal_gran slice player (1.5 s at 10 / 30 / 55 / 80 % of the source) | a breath-like slice when someone enters the picture |
| pair | at most one per 15 s; strength 0 first so there is no shake | `/fx/predator_strength 0`, `/pad/kick` | shoal_fx one kick | one kick when two people start walking together |

`/plaud/loudness` also moves the Limit dial inside the device (the SynthFish `scale`); the scene's
`/plaud/limit` is sent right after it, so its value is the one that stays. The acceleration kick
is off in this profile. The HUD (`H`) shows three lines: the scene quantities, the PLAUD values and
the events.

## The feature-target territory and the RafaelV2 terrain

PLAUD's prior is steered by a 2D style point, the **territory**. The model comes with a terrain
map (`RafaelV2.json`): six audio features measured on 8 s of generated audio at a 24×24 grid of
territory points (cfg 1.5, temperature 0.6, one seed) and interpolated to 128×128, each 0–1:

| Feature | p5 / median / p95 | Layout |
|---|---|---|
| rhythmicity | 0.19 / 0.41 / 0.66 | small patches, slightly higher on the left |
| onset_density | 0.09 / 0.40 / 0.74 | patchy, higher on the left |
| percussiveness | 0.10 / 0.28 / 0.69 | smooth, high at the top |
| loudness | 0.13 / 0.50 / 0.78 | smooth, high at the bottom |
| brightness | 0.13 / 0.31 / 0.59 | smooth, high at the top |
| noisiness | 0.10 / 0.19 / 0.59 | smooth, high at the top |

They are strongly correlated: brightness and noisiness r = +0.91, percussiveness with both
+0.7, loudness against all three −0.7 to −0.9; rhythmicity and onset_density +0.43, and mostly
independent of the rest. In practice the map has about two and a half directions (59 % of the
variance on the percussive / bright / noisy ↔ loud axis, 21 % on rhythm and onsets).

So profile scene uses three targets, each scaled to the map's own p5–p95:
- **onset_density ← count**;
- **rhythmicity ← speed**;
- **texture ← disorder** (disorder / 0.6, clipped). Texture is one value per cell for the main
  direction: the mean z-score of percussiveness, brightness and noisiness, minus that of loudness.

Walking together is deliberately not on the map (it is planned for the prior's cfg).

**Lookup.** The map is averaged to 32×32 cells. Ten times a second the cell with the lowest
`Σ (feature − target)² + 0.3 × (distance from the current territory)²` is chosen: the penalty keeps
the point from jumping between far patches that match equally well. The territory glides to it
(τ 4 s). Without a terrain map: x = 1 − (count + speed) / 2, y = disorder.

Measured on whole files, the median gap between the wanted and the reached feature is 0.1–0.23; big
target changes (more than 0.3 of the map) happen 1–3 times a minute, each glided over seconds.

## Events and flashes

| Event | Profile | Rule | Sound |
|---|---|---|---|
| acceleration | synthfish | z-score of agitation over the last 10 s of data time: onset at z > 3 (rising, after 5 s), ends below 1.5 for 0.3 s, then 5 s without events (`--replay-event-z`, `--replay-event-refractory`) | `/fx/predator` 1 / 0: a kick at once and every 800 ms, strength 0.3–1.0, with the PLAUD shake; the predator rings are drawn on the person who accelerates most |
| arrival | scene | a new person becomes visible; at most one per 15 s | `/pad/sample 1–4` |
| pair | scene | two visible people walk together for 2 s; at most one per 15 s | one kick, no shake |

In profile scene every sent event **flashes** on its people at the same moment, in every view: a
ring that grows and fades over 1.2 s, labelled "arrival" or "walking together"; a pair is joined by
a line. Someone who only walks along the edge of the picture never makes a sound. On zara1 every
sent event was at least 40 px inside the frame, with a moving person under the flash in 32 of 37.

## The Max side

Three patches run in one Max session: the **PLAUD device** (not in this repository), and
`max/shoal_gran.maxpat` and `max/shoal_fx.maxpat` (in this repository). How the device receives
OSC is in [MAX_SETUP.md](MAX_SETUP.md); the routes this project added are in
[MAX_PATCH_CHANGES.md](MAX_PATCH_CHANGES.md).

### The PLAUD device

What happens inside the device, as read from its patch and from the model's TorchScript code:

```
 prior (nn~ ... prior): territory x/y, tmp, cfg, smh -> 4 latents
   + Loudness, Centroid, L1, L2 dials (offsets on latents 1-4)
   -> looper (passes the latents through, or plays a recorded latent path)
   -> posterior: x ch1-4, then + ch1-4
   -> decoder (nn~ ... decode): DDSP synthesis, with the bending controls
   -> Dry gain -> output;  Wet gain -> vst~ (Portal) -> output
```

| Part | Control | Range | What it does | OSC today |
|---|---|---|---|---|
| prior | territory X / Y | 0–1 | the style point (the terrain map's coordinates) | `/plaud/x`, `/plaud/y` |
| prior | cfg | 0–20 | guidance strength toward the territory: `uncond + cfg × (cond − uncond)` on the token logits; also the terrain display's contrast | no (planned) |
| prior | tmp | 0–5 | sampling temperature of the prior's tokens | no (planned; `/plaud/temp` is L2, not this) |
| prior | smh | 0–1 | smoothing of the prior's output latents (exponential, capped at 0.999) | no (planned) |
| prior | on / off | | mutes the prior | no |
| prior | conditioning inputs 0–3 | | the final latents fed back, delayed; in this patch they pass through a `*~` with no argument, so they are 0 | — |
| prior | reseed (input 9) | | a rising edge restarts the prior's sequence; not connected in the patch | — |
| dials | Loudness, Centroid | 0–1 | offsets on latents 1 and 2, scaled by LdMin/LdMax, CdMin/CdMax (0–1) | `/plaud/loudness`; `/plaud/centroid` (added) |
| dials | L1, L2 | −1–1 | offsets on latents 3 and 4. L1 is driven by an internal LFO (`cycle~ 0.054`, ~18.5 s period) | `/plaud/temp` → L2; L1 no |
| dials | bypass | on / off | freezes the four dials: neither the UI nor OSC reaches the latents while it is on (off at load) | no |
| looper | on | on / off | plays the recorded latent path instead of the live latents | no (planned) |
| looper | rec | on / off | records the latents into the buffer | no (planned) |
| looper | sync, quantize + grid | on / off, 1/16 … 1 bar | keeps the four playheads together; snaps lengths to the transport grid | no |
| looper | buffer + set, start, length, speed, direction, channel, preserve | 1–60 s, 0–100 %, 0.125×–32×, → / ←, all / ch1–4 | the loop window and playback; preserve writes the buffer to a file | no |
| posterior | × ch1–4 | −5–5 (1) | multiplies each final latent | no |
| posterior | + ch1–4 | −1–1 (0) | adds to each final latent | no |
| bending | lim (Limit) | 0–1 | how many partials survive: 0 = all, 1 = one | through `/plaud/loudness`; `/plaud/limit` (added) |
| bending | typ (PartialsMode) | loudest, density, lower, higher, peaks, stochastic | which partials survive the limit | no |
| bending | shp (Shape) | 0–1 | waveshape: 0 noise, 0.5 sine, 1 square | the shoal_fx shake only |
| bending | rol (SpectralRoll) | 0–1 | shifts the spectrum up, wrapping | no |
| bending | war (SpectralWarp) | −1–1 | skews the spectrum (+ = toward lows) | the shoal_fx shake only |
| bending | str (SpectralStretch) | −1–1 | stretches the spectrum (+ = toward highs) | `/plaud/stretch` (added) |
| toggles | rythme | on / off (off) | every 1 s: plays the device's kick sample and ramps Shape to 0.5 over 0.8 s, then back to 0 in 0.1 s | no |
| toggles | kick | on / off (off) | mutes / unmutes that kick sample | no |
| toggles | latent | on / off | display only: the latent meters | — |
| toggles | prior glitch | on / off, 108 ms | retriggers a sample into a meter only: no sound, no effect on the prior | — |
| output | Dry, Wet | −70–6 dB (−6) | PLAUD's direct level; the level into the Portal effect (`vst~`) | `/mix/master`, `/mix/wet` |

### shoal_gran.maxpat

One grain voice per person (16 MC voices; voices above the number of people fade out in ~50 ms).
Each voice has its own onset clock at `(0.5 + 0.075 × density)` grains/s with a slow random drift
(depth grows with `noise`), reads a Hann-windowed grain from a random position in a region of the
source, and is panned by its person's `/gran/fish_x` (equal power). `/pad/sample 1–4` plays a 1.5 s
stereo slice from 10 / 30 / 55 / 80 % of the source. Details and defaults: [MAX_SETUP.md](MAX_SETUP.md#shoal_granmaxpat-one-grain-voice-per-fish).

### shoal_fx.maxpat

The kick: while `/fx/predator` is 1 it fires at once and every 800 ms; `/pad/kick` fires one. Each
kick shakes PLAUD (waveshaping and spectral warp jump to base + shake × the last strength, then
glide back in 400 ms) and feeds a long reverb. Details: [MAX_SETUP.md](MAX_SETUP.md#shoal_fxmaxpat-predator-kick-spectral-shake-and-reverb).

## Limits

- **Data.**
  - `eth` moves about twice too fast and is skipped.
  - Camera and video views exist only for zara1 and hotel; zara1 needs an image correction, and
    its data points sit about 26 px from the people in the video.
  - hotel's annotations leave out people at the frame edges and at the tram stop: about 28 % of
    the visible person-sized motion has a data point, against 77 % in zara1.
- **Sound mapping.**
  - Absolute scales for disorder and closeness: univ is disordered and packed all the time, so
    its texture target and stretch stay near the top.
  - Arrivals fire when the newcomer crosses the 40 px edge line, so nearly all play slice 1
    (from the left) or 4 (from the right).
  - The terrain map was measured at cfg 1.5 and temperature 0.6 with one seed; rhythm and onsets
    are patchy, so the reached features miss the targets by 0.1–0.23 (median).
  - shoal_gran has 16 voices: people beyond the 16th have no grain voice.
  - In profile synthfish, replays keep Limit at 1 (a single partial), as described above.
- **Max.**
  - The prior's cfg, tmp and smh, the posterior, the looper and the L1 LFO have no OSC route yet.
  - The prior's conditioning inputs are multiplied by 0 in the patch.
  - Bypass blocks the dials, OSC included.
- **Platform.** Tested on Windows only. The descriptor rate is ~15 Hz (a 1/20 s gate checked once
  per 60 fps frame).
