# Replay: real trajectories instead of the boids

`--replay FILE` moves the agents along recorded trajectories (pedestrians from the ETH/UCY
datasets, or any CSV of positions) instead of the boid rules. Everything downstream is unchanged:
the same descriptors, the same OSC to Max, the same PLAUD, `shoal_fx` and `shoal_gran` patches. So
the music now follows real crowds.

Without `--replay` nothing changes: the simulation, its look and its OSC output are identical to
before the replay work (checked with OSC and pixel hashes after every commit).

```
python src/main.py --replay <DATA>/eth_ucy/zara1/crowds_zara01.txt --replay-flip-y
```

## Contents

- [What happens in replay mode](#what-happens-in-replay-mode)
- [Data sources and citations](#data-sources-and-citations)
- [Getting the files](#getting-the-files)
- [Python environments](#python-environments)
- [Command line](#command-line)
- [Looks and views](#looks-and-views)
- [Calibration and normalisation](#calibration-and-normalisation)
- [Acceleration events](#acceleration-events)
- [Sound profiles](#sound-profiles)
- [Known limits](#known-limits)

## What happens in replay mode

**Motion.**
- Positions come from the file, scaled uniformly into the 1200x750 world (60 px margin, aspect kept).
- Between keyframes (2.5 per second in ETH/UCY) a cubic Hermite spline gives smooth position,
  velocity and acceleration at 60 fps.
- One missing row is bridged; a longer gap ends the track, and the person comes back as a new agent.

**Agents.**
- Each person is an ordinary `Boid` object that is moved, never flocked. Size and colour come from
  the track id, never from `random`.
- A person keeps the same list index while present, so `/gran/fish_x` voice *i* stays the same person.
  When someone leaves, the last agent takes the freed index; newcomers are appended, up to the cap.

**Inputs.**
- **MIDI is off** in replay mode. The UDP control port 7500 stays open.
- The mouse food and predator show their rings and the predator still sends `/fx/predator`, but they do
  not move anyone: the data already contains the people's reactions.
- The boid weight keys and `-`/`=` have no effect.
- `SPACE` pauses the data, `R` restarts the file, and at the end of the data the last frame holds
  (unless `--replay-loop`).

**Output.** The OSC addresses are the same as in the simulation ([OSC.md](OSC.md)). The values
describe the crowd instead of the flock: see [Calibration and normalisation](#calibration-and-normalisation).

## Data sources and citations

None of the data is in this repository. The files are downloaded once into a folder of your choice
(see [Getting the files](#getting-the-files)).

| What | From | Terms | Cite |
|---|---|---|---|
| ETH: `eth`, `hotel` (BIWI Walking Pedestrians) | Trajectron++ copies (below) | no licence information is distributed with the data | S. Pellegrini, A. Ess, K. Schindler, L. van Gool, *You'll never walk alone: Modeling social behavior for multi-target tracking*, ICCV 2009 |
| UCY: `zara1`, `zara2`, `univ` (Crowds by Example) | Trajectron++ copies (below) | free to use with credit to the original authors | A. Lerner, Y. Chrysanthou, D. Lischinski, *Crowds by Example*, Computer Graphics Forum 26(3), 2007 |
| The trajectory files (Social GAN leave-one-out splits) | [StanfordASL/Trajectron-plus-plus](https://github.com/StanfordASL/Trajectron-plus-plus), `experiments/pedestrians/raw/<scene>/test/`, commit `1031c7bd` | MIT (repository) | when the splits are used as in the benchmark: Social GAN (A. Gupta et al., CVPR 2018) and Trajectron++ (T. Salzmann et al., ECCV 2020) |
| Reference frames, homographies, videos (`zara01`, `seq_hotel`) | [crowdbotp/OpenTraj](https://github.com/crowdbotp/OpenTraj), `datasets/UCY/zara01/`, `datasets/ETH/seq_hotel/`, commit `97ec1d5e` | MIT (repository); the data keeps the terms above | J. Amirian et al., *OpenTraj: Assessing Prediction Complexity in Human Trajectories Datasets*, ACCV 2020 |

The replay credits its source on screen:
- The window title and the caption name the scene and its dataset, e.g. `real trajectories (UCY: Lerner et al. 2007)`.
- The HUD shows the full citation.
- The camera and video views add `+ OpenTraj frame / video (Amirian et al. 2020)`.

## Getting the files

Put the data in any folder outside the repository, called `<DATA>` here, for example
`D:\data`. Keep a short `SOURCE.txt` next to the files: URL, commit, date, checksums, and the
citations above.

**1. Trajectories (ETH/UCY, needed for any replay).** The test split of each scene is the whole
scene. A shallow, sparse clone fetches only those folders:

```
git clone --depth 1 --filter=blob:none --sparse https://github.com/StanfordASL/Trajectron-plus-plus.git tpp
cd tpp
git sparse-checkout set experiments/pedestrians/raw/eth/test experiments/pedestrians/raw/hotel/test experiments/pedestrians/raw/univ/test experiments/pedestrians/raw/zara1/test experiments/pedestrians/raw/zara2/test
```

Copy the `.txt` files into `<DATA>/eth_ucy/<scene>/`:

```
<DATA>/eth_ucy/eth/biwi_eth.txt
<DATA>/eth_ucy/hotel/biwi_hotel.txt
<DATA>/eth_ucy/univ/students001.txt, students003.txt
<DATA>/eth_ucy/zara1/crowds_zara01.txt
<DATA>/eth_ucy/zara2/crowds_zara02.txt
```

**Format** (as found): tab-separated, no header, `frame id x y`, with x and y in metres. The frame
numbers advance by 10 per row, at 2.5 rows per second (0.4 s); the loader detects the step.

**2. OpenTraj frames, homographies and videos (camera and video views, zara1 and hotel only).**
Download these files from
`https://raw.githubusercontent.com/crowdbotp/OpenTraj/97ec1d5e1b579f26febeaa9083705952deac338a/datasets/`:

```
UCY/zara01/reference.png  UCY/zara01/H-cam.txt  UCY/zara01/video.avi   (53 MB)
ETH/seq_hotel/reference.png  ETH/seq_hotel/H.txt  ETH/seq_hotel/video.avi   (50 MB)
```

Lay them out as in OpenTraj:

```
<DATA>/opentraj/zara01/reference.png, H-cam.txt, video.avi
<DATA>/opentraj/seq_hotel/reference.png, H.txt, video.avi
```

Then pass `--replay-camera-dir <DATA>/opentraj`, or set the environment variable
`PLAUD_OPENTRAJ_DIR` once.

**3. Your own data.** Use a CSV with a header row containing `id,frame,x,y` (any order and case;
extra columns are ignored). An optional `t` column in seconds replaces `frame / --replay-fps`.

## Python environments

Two separate environments keep the base install small and untouched:

| Environment | Install | Gives |
|---|---|---|
| base | `pip install -r requirements.txt` (pygame, python-osc) | the simulation and every replay feature except the video view |
| video | `pip install -r requirements-video.txt` (the base set + OpenCV and numpy, pinned) | also `--replay-view video` |

For example, with Python 3.13 on Windows:

```
python -m venv .venv-video
.venv-video\Scripts\activate
pip install -r requirements-video.txt
```

OpenCV is imported only for the video view. Without it, the app prints how to install it and
falls back to the camera view.

## Command line

All the [general options](CONTROLS.md#command-line) still apply (`--classic`, `--mode`,
`--fullscreen`, `--size`).

| Option | Default | Effect |
|---|---|---|
| `--replay FILE` | off | replay this trajectory file instead of the simulation |
| `--replay-format auto\|csv\|eth-ucy` | `auto` | `auto`: `.txt` is ETH/UCY (`frame id x y`, no header), anything else is CSV |
| `--replay-fps F` | 2.5 | data rows per second (when there is no `t` column); the frame step is detected |
| `--replay-speed S` | 1 | playback speed (1 = real time); `mean_speed` and `energy` do not depend on it |
| `--replay-loop` | off | start again at the end of the data |
| `--replay-flip-y` | off | the data's y points up (ETH/UCY: use it for a map-style view, x right, y up) |
| `--replay-max-agents N` | 100 | most agents on screen at once; extra people wait for a free slot |
| `--replay-look data\|thermal` | `data` | the data look (below), or people drawn in the thermal look |
| `--replay-view map\|camera\|video` | `map` | data look only: metric map, OpenTraj reference frame, or the original video |
| `--replay-camera-dir DIR` | `$PLAUD_OPENTRAJ_DIR` | the folder with `zara01/` and `seq_hotel/` from OpenTraj |
| `--replay-normalise` | off | map every descriptor to this file's p5–p95 range |
| `--replay-no-events` | off | no acceleration events |
| `--replay-profile synthfish\|scene` | `synthfish` | sound mapping: the simulation's, or one that follows the scene ([Sound profiles](#sound-profiles)) |
| `--replay-terrain FILE` | `$PLAUD_TERRAIN` | profile `scene`: the PLAUD model's terrain map (e.g. `RafaelV2.json`, not included); without it, plain axes |
| `--replay-event-z Z` | 3 | event onset threshold (z-score) |
| `--replay-event-refractory S` | 5 | seconds of data time without a new event after one ends |

Examples:

```
# a dense scene, calibrated to its own range, looping
python src/main.py --replay <DATA>/eth_ucy/univ/students001.txt --replay-flip-y --replay-normalise --replay-loop

# hotel over its original video, twice as fast (video environment)
python src/main.py --replay <DATA>/eth_ucy/hotel/biwi_hotel.txt --replay-view video --replay-camera-dir <DATA>/opentraj --replay-speed 2

# zara1 on the reference frame, fewer and stronger events
python src/main.py --replay <DATA>/eth_ucy/zara1/crowds_zara01.txt --replay-view camera --replay-event-z 3.5 --replay-event-refractory 8

# your own CSV at 10 rows per second
python src/main.py --replay walk.csv --replay-fps 10
```

Keys in replay mode are listed in [CONTROLS.md](CONTROLS.md#replay-mode).

## Looks and views

**Data look** (default). It is drawn like tracking output, not like the aquarium:
- a neutral dark grey background;
- each person as a dot with a heading tick, coloured by speed (violet = still, yellow = the file's p95, with a legend);
- a thin line through each person's recorded positions of the last 10 s;
- `I` toggles track id labels, `H` toggles the HUD;
- a caption at the bottom-left with the scene, the citation and the data time (`mm:ss / total`).

It has three views:

| View | Background | Notes |
|---|---|---|
| `map` | a metric grid (1 m minor, 5 m major) and a scale bar | every scene |
| `camera` | the scene's OpenTraj reference frame, dimmed | zara1 and hotel. People, paths, rings and labels are projected into the frame through its homography. No grid or scale bar (the perspective changes the scale). The frame is one still photo: the people in it are not the people in the data |
| `video` | the scene's original video, dimmed, at frame round(25 × data time) | zara1 and hotel; needs OpenCV. Pause, `R`, loop and `--replay-speed` follow the data time. Decoding runs in a background thread; jumps (restart, loop) seek, and the previous frame stays up meanwhile |

**Optional layers** (all views; off at start, so the look above is the default):

| Key | Layer | What it shows |
|---|---|---|
| `N` | social graph | a thin line between two people closer than 2 m; when they walk together (heading within 30°, speed within 0.3 m/s, both moving) the line thickens and brightens from 2 s to 6 s of walking together |
| `L` | long exposure | each person lays light along the way they walk (per distance, so standing adds nothing); it fades one grey level every 4 s of data time, so one pass stays visible about 1.5 min and a much-used route about 17 min: the street's habitual routes appear |
| `V` | video look (video view) | normal → dark (grey, darker street; what moves since the previous frame keeps its colour) → off (people and layers on a plain background) |

The layers follow the data time (pause holds them; `R` and a loop restart clear the timers). All
views and layers are display only: a replay sends exactly the same OSC in map, camera and video
view, with or without the layers (except with `--replay-profile scene`, where pan and the events
follow the picture shown: [Sound profiles](#sound-profiles)).
Measured frame rates are about 60 fps in a 1200x750 window and fullscreen, for all three views.

**Thermal look** (`--replay-look thermal`). People as heads with a heading tick in the aquarium's
thermal style, with longer trails. **Classic look** (`--classic`). Dots with a tick on the flat
background. Both show the same caption and replay HUD.

## Calibration and normalisation

The descriptors (`descriptors.py`) are unchanged. They were designed for 100 fish moving up to
4 px per frame, so each file is calibrated when it loads:

**Speed and force (always).**
- Every agent's `max_speed` is the 95th percentile of |v| over the whole interpolated file at the
  chosen speed, and `max_force` is the 95th percentile of |a| divided by 6.
- So `mean_speed` and `energy` mean "fast / agitated for this dataset", on the same 0–1 scale as in
  the aquarium.
- Without this, pedestrians would sit near `energy` 0.01. With it, zara1 spans about 0.07–0.54.

**`--replay-normalise` (optional).**
- A pass over the file collects each raw descriptor (no smoothing) and stores its 5th and 95th
  percentile.
- During the replay each value becomes (v − p5) / (p95 − p5), clipped to 0–1, before the smoother.
- `fish_count` becomes the number of people on screen divided by the most at once in the file,
  instead of divided by 100.
- Use it when a scene's values crowd into a narrow band. The values then mean "relative to this
  file".

## Acceleration events

The predator kick in Max (`/fx/predator`) has a data-driven version: a sudden change in how the
people move.

**Signal.** Agitation, i.e. the mean |a| over the calibrated p95, sampled every frame in data time.

**Detector** (z-score over the last 10 s of data time):
- an onset when z > 3 and rising, after a 5 s warm-up;
- an event lasts at least 0.5 s and ends when z stays below 1.5 for 0.3 s;
- after an event, no new one for 5 s.

**Effect.**
- The repeller is placed on the person who accelerates most. The existing predator path then
  sends `/fx/predator 1`/`0`, and every look draws the predator rings there. The data look also
  writes "acceleration event z …" next to them.
- `predator_strength` follows the event: 3 × (0.3 + 0.7 × clip((peak z − 3) / 2.5)). So
  `/fx/predator_strength` is 0.3–1.0 and every onset is audible. The previous value comes back
  after the event.
- The repeller exerts no force.

**Rates** with the defaults, whole files:
- zara1: 5.2 events per minute;
- hotel: 4.4 per minute;
- univ (students001): 5.1 per minute.

Raise `--replay-event-z` or `--replay-event-refractory` for fewer events. For example, z 4 with 8 s
gives about 1–3.5 per minute.

## Sound profiles

`--replay-profile` chooses how the crowd drives the sound. Both use only addresses the Max patches
already receive; nothing changes on the Max side.

**`synthfish`** (default): the simulation's mapping, unchanged. The PLAUD territory follows the
crowd's centroid in the frame, `/plaud/loudness` follows the count, `/gran/density` the count, and
acceleration events trigger the kick ([above](#acceleration-events)).

**`scene`**: PLAUD is a slow background bed that follows the scene's state; what you hear as an
event is something you can point at in the picture, at the same moment.

| Sound | From | Address |
|---|---|---|
| PLAUD's style territory | **hurry** (mean speed / the file's p95) → horizontal: hurried = left, where the RafaelV2 terrain is more rhythmic with denser onsets; **crowding** (people / most at once in the file) → vertical: crowded = down (fuller, louder), sparse = up (more percussive, brighter). With `--replay-terrain` the point is the terrain cell whose measured features match that state best, kept close to those axes. It glides over ~25 s and stays within the central 60 % of the map | `/plaud/x`, `/plaud/y` |
| presence | the number of people: **silence when the scene is empty** (fade-out ~3 s), a soft fade-in on the first arrival (~1.5 s), and a gentle log curve with the count, 4 dB wide, up to −10 dB (below the device's −6 dB default: a bed) | `/mix/master` (PLAUD dry level) |
| PLAUD L2 offset | agitation (mean acceleration / the file's p95) averaged over ~10 s: `0.2 + 1.3 ×` that | `/plaud/temp` |
| pan of the grain voices | each person's position across the **picture shown**: the video column in the camera and video views, the map's x in the map view | `/gran/fish_x` |
| grain density | the walkers' mean step rate (speed / 0.7 m): about 1.9 grains/s per voice at 1.3 m/s; 0 when everybody stands | `/gran/density` |
| arrivals | a new person is first **clearly in the picture**: a slice of the grain source, slice 1–4 by where they are (left to right) | `/pad/sample` |
| walking together | two people, both clearly in the picture, start walking together (the social-graph rule, held 2 s): one kick, without the PLAUD shake (strength 0, then the kick) | `/fx/predator_strength`, `/pad/kick` |

- Clearly in the picture: at least 40 px inside the video frame (camera and video views), or 1.5 m
  inside the data's bounding box (map view). Someone who only walks along the edge makes no event.
- Each sent event flashes on the people at the same moment (a ring growing and fading over 1.2 s,
  labelled "arrival" or "walking together"; a pair is joined by a line), in every view.
- Each event type is capped at one per 15 s of data time (about 4 per minute).
- The acceleration kick is off in this profile.
- PLAUD's internal L1 LFO is not reachable over OSC; it stays as it is (a Max change, phase B).
- Everything runs in data time: pause holds it, `R` and a loop restart it.
- With the data HUD (`H`) a line shows the territory, hurry, crowding, presence, step rate and
  events sent / detected.

Measured on zara1, whole file, video view (`scene` with the terrain):

| | zara1 |
|---|---|
| territory x / y (p5–p95) | 0.35–0.45 / 0.53–0.70; at most 0.10 of movement in any 10 s |
| presence (not silent, p5–p95) | −13.3 … −10.8 dB |
| L2 offset (p5–p95) | 0.38–0.58 |
| arrivals detected → sent | 140 → 19 in 6 min |
| walking together detected → sent | 120 → 18 in 6 min |
| sent events at least 40 px inside the frame | 37 / 37; a moving person under the flash in the video: 32 / 37 |
| pan vs the column of the motion under each person | r = +0.998, median 8.7 px apart |

Silence takes a few seconds to reach, so empty stretches shorter than that are only quieter. `/mix/master` is the PLAUD **dry** level only: the wet path
(Portal), the granulator and the kick keep their own levels. The granulator's voices follow the number of people, so it should fall silent when no one is
left, but whether an empty `/gran/fish_x` list reaches `shoal_gran` has not been checked in Max yet.

## Known limits

**Data.**
- **ETH `eth` (biwi_eth) moves too fast.** The median walking speed comes out at 2.45 m/s, about twice a
  normal pace (hotel 1.27, zara1 1.15 m/s). The time scale of that file may be off; this has not
  been checked. Use zara1 and hotel as reference scenes.
- **zara1 needs a correction in the camera and video views.** The Trajectron++ zara coordinates are
  not exactly in the world frame of OpenTraj's `H-cam.txt`. After the homography, the projection is
  mirrored left-right, scaled ×1.2 about the frame centre and shifted (+10, +40) px
  (`ZARA01_FIX` in `replay/camera.py`).
  - This was found by syncing the data with the video at the same frame number: the share of dots
    on moving people went from 0.29 to 0.86, against 0.82 for hotel, which needs no correction.
  - Where the difference comes from is not known. The correction is an image-space fit, accurate to
    a few pixels.
- **The y direction is not stored in the files.** `--replay-flip-y` gives a map view (y up); without
  it the scene is mirrored. The camera and video views do not depend on it.

**Signal quality.**
- **Acceleration is approximate.** Between keyframes it is piecewise linear and can jump at a
  keyframe. On a test circle sampled like the data, |a| is within about 6% and its direction within
  about 14°. That adds a small ripple to `energy` at the keyframe rate.
- **hotel's events partly mark people entering or leaving.** Onsets coincide with a change in the
  number of people 69% of the time, against 38% at random moments. zara1 and univ show no such
  effect.

**Scale.**
- **At most 100 agents** (`--replay-max-agents`). The nearest-neighbour descriptor is O(n²). univ
  (up to 75 people) still runs at 60 fps.

**Views.**
- **Camera and video views exist for zara1 and hotel only.** OpenTraj has frames for other scenes,
  but they have not been aligned.

**Sound.**
- **The music layer (`/music/*`) differs from the simulation's** even for identical crowds: the boids
  draw random numbers every frame, the replay does not, so the Markov layer's random sequence
  changes. `UP`/`DOWN` and `1`–`6` still control it.
