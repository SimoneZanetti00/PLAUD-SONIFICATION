# PLAUD-SONIFICATION — Roadmap

Where the project can go next. Ordered from "finish what we have" to "new pieces".
Each item says what it is, why it matters, and roughly what it needs.

Status at the time of writing (November 2026): pedestrian replay (ETH/UCY) works end to end,
with map, camera and video views, visible sound events, and a `scene` sound profile in which
PLAUD follows the state of the street (people count, walking speed, disorder, crowding).
zara1 is the reference scene.

---

## 1. Finish and consolidate (short term)

| # | Item | Why | Needs |
|---|---|---|---|
| 1.1 | **One grain voice per person** (phase B, item a) | The clearest link between image and sound: you follow a person and hear them cross the space. Today each person already has a grain voice (up to 16), panned by position, but all voices share one rate; missing: speed → grain rate/pitch per voice, fade in on arrival and out on leaving, and a voice that stays with its person when someone else leaves. | Python lists per voice + `shoal_gran` change in the Max copy |
| 1.2 | **Darken untracked motion** (item d) | In the video view, only people who are in the data keep their colour, so nobody visible is "silent". | Python only (display) |
| 1.3 | **Prior character: cfg / tmp / smh** (item B) | Order → PLAUD follows the scene closely; disorder → it improvises; calm → smoother sound. Pairs walking together → higher cfg ("focus"). Keep near cfg 1.5 / tmp 0.6, where the terrain map is valid. | 3 OSC routes in the device + receives in `prior~` (Max copy, with backups) |
| 1.4 | **Latent looper on pairs** (item D, optional) | A "duet": while two people walk together, their latent path is recorded and replayed softly. | Max routes to rec/on/length; test without Live transport |
| 1.5 | **L1 LFO under control** | L1 still moves on an internal 18.5 s LFO, unrelated to the scene. | Small Max change |
| 1.6 | **Validation (REPLAY_04 commit 5)** | Proves the replay path feeds the descriptors exactly like the simulation; needed for a paper. | Harness recorder + `--replay-exact` |
| 1.7 | **Blind listening test** | Listeners hear audio only and guess crowded/empty, calm/hurried, arrivals. Shows the mapping is legible, `synthfish` vs `scene`. | 10–12 listeners, thresholds fixed in advance |
| 1.8 | **Demo video** | 1–2 min, zara1 video view + Max audio, for the README and applications. | Screen recording (OBS / Win+G) |

## 2. Make it public

| # | Item | Needs |
|---|---|---|
| 2.1 | OK from Alessandro (the code base comes from SynthFish) and a licence (MIT?) | conversation |
| 2.2 | Błażej's answer on publishing our modified PLAUD device; until then `docs/MAX_PATCH_CHANGES.md` explains how to add our routes by hand | his reply |
| 2.3 | Rafael: whether `RafaelV2.ts` can be shared (e.g. as a GitHub Release) | his reply |
| 2.4 | Switch the repository to public, add `LICENSE`, fill the demo slot | GitHub settings |

## 3. New data: other collective behaviours

| # | Dataset | What is interesting | Notes |
|---|---|---|---|
| 3.1 | **Ants** (*Aphaenogaster senilis*, foraging in a Y-maze, with and without food; CSV, 2 Hz, CC BY 4.0) | Trail following, recruitment waves, food discovery. Sessions with and without food can be compared by ear. | 3 h sessions → `--replay-speed`; possibly > 100 ants → grid speed-up in descriptors |
| 3.2 | **Fish** (zebrafish, idtracker.ai / SLEAP, CC BY 4.0) | Chases, dominance fights, escapes: a natural follow-up to SynthFish's virtual fish. | Large files (10 GB), mostly pairs |
| 3.3 | **Traffic** (drone datasets such as inD; or own recordings) | Cars, bikes and pedestrians sharing a crossing: flows, stops at lights, near misses. | Some datasets require registration and are research-only |
| 3.4 | **Own recordings + automatic tracking** (YOLO + ByteTrack on a video of a street, a square, an aquarium, an ant nest) | Fully original material, and later live input from a camera. | New tool: video → `id,frame,x,y` CSV; then the existing pipeline |

Each new dataset needs: a converter to `id,frame,x,y`, a calibration check, its own event types
(e.g. ants: trail formed, food found; fish: chase, escape), and a listening pass.

## 4. New PLAUD models: the timbre should belong to the place

PLAUD is designed for small personal datasets (~20–30 min of audio). Today every scene sounds
with Rafael's material. Training scene-specific models makes the timbre itself coherent.

| # | Model | Used with |
|---|---|---|
| 4.1 | **Open-air conversations and footsteps** (streets, squares, markets) | pedestrian scenes |
| 4.2 | **Traffic and street noise** (engines, tyres, horns, trams) | traffic scenes, crowded streets |
| 4.3 | **Nature** (wind, leaves, birds, water, insects) | ants, fish, calm moments |
| 4.4 | **Multichannel model** (`audio.n_channels`, e.g. 4) | spatial installations: each zone of the scene on its own speaker |

Needs: recordings with a clear licence (own field recordings preferred; Freesound CC0/CC-BY),
GPU training (DDSP, then the prior), a terrain map for each model so the feature-target
territory works with it unchanged.

## 5. Artistic piece: "Crash" (traffic → nature)

**Idea.** Two PLAUD models, one trained on traffic and street noise, one on nature.
The system follows a busy scene with the traffic model. As traffic, density and disorder grow,
the sound gets denser, dirtier and more bent (bending, limit, noise). Past a threshold the
system **"crashes"**: a short breakdown, then the model switches to nature, and the calm is heard.
Later, when the scene empties, or as a loop, the cycle can start again.

**Why it works.** It turns the mapping into a dramaturgy with a clear arc, it makes a statement
about the city and its noise, and the moment of the switch is an unmistakable event for the
audience.

**How (first sketch).**
- Two decoders loaded in parallel and crossfaded, rather than reloading one `nn~` (a 200 MB
  model takes time to load, which would leave a gap).
- A "saturation" value from the scene (count, disorder, closeness, events per minute), integrated
  over time; the crash fires when it stays above a threshold, with hysteresis.
- The crash itself: a few seconds of extreme bending / glitch (prior reseed: the model's prior has
  a reseed input, input 9, but the device does not connect it yet), then the crossfade to the
  nature model with a slow fade.
- Visual counterpart: the video saturates or freezes, then fades to the dark/clean look.
- Scenes: a traffic dataset (3.3) or a dense pedestrian scene (univ), or own recordings.

## 6. Visual and immersive

| # | Item | Notes |
|---|---|---|
| 6.1 | **Sound → image**: Max sends PLAUD loudness/brightness back to Python (UDP 7500, already open), and light and colour breathe with the sound | closes the loop: one organism |
| 6.2 | **TouchDesigner** as the visual engine, receiving the same OSC | richer visuals; Python stays the data engine |
| 6.3 | **Installation**: large projection + multichannel audio (the granulator is already MC); top-down map projected on the floor | needs a venue |
| 6.4 | **Long-exposure score**: export the accumulated paths as an image — a visual "score" of the piece | small Python addition |

## 7. Suggested order for the next weeks

1. 1.1 one voice per person, 1.2 darken untracked → the pedestrian piece is complete.
2. 1.8 demo video, 2.1–2.4 permissions and public release.
3. 3.1 ants (first non-human dataset).
4. 4.1 / 4.3 first scene-specific models (own recordings).
5. 5 "Crash" as the first artistic piece built on two models.
6. 1.6 / 1.7 validation and listening test, if a paper or thesis is the goal.
