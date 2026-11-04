# Ants: replaying ant colonies

Experimental (branch `ants`). Two ant tracking datasets can be replayed with the same pipeline as
the pedestrians: a converter turns them into the replay CSV, and a **scene config** tells the
replay what the data is (units, body length, nest, food, maze, image and video).

Status: step 1, the base. Data, converters, scene config and views are in place. The sound is
still the pedestrian one (`synthfish` or `scene` profile, unchanged): the ant events and mapping
(food discovery, recruitment wave, activity, distance from the nest, encounter density, arrivals
at the nest) are the next step. Without `--replay-config` nothing changes for the pedestrian
scenes.

## Contents

- [The datasets](#the-datasets)
- [Getting the files](#getting-the-files)
- [Converting](#converting)
- [The scene config](#the-scene-config)
- [Views](#views)
- [Known limits](#known-limits)

## The datasets

| | ANTS | Foraging Ants as Liquid Brains |
|---|---|---|
| What | ant colonies filmed from above: 5 indoor sequences (10 ants in a dish) and 5 outdoor ones (11–61 ants around a nest entrance) | *Aphaenogaster senilis* foraging in a Y-maze, from the nest at the bottom, with food (10 trials) or without (12 trials) |
| Data | `gt.txt` per sequence: `frame,id,left,top,w,h,1` in image pixels, fixed-size boxes; jpg frames | one CSV per trial: id, frame (2 Hz), x, y in mm, contact flag, maze node; maze nodes; food items with their discovery time |
| Length | 14–40 s per sequence, 25 fps (indoor) or 30 fps (outdoor) | about 3 hours per trial |
| Images | yes (jpg frames; the converter makes an mp4) | none: map view only |
| Licence | CC0 1.0 | CC BY 4.0 (cite it) |
| Cite | M. Wu, X. Cao, M. Yang, X. Cao, S. Guo, *A dataset of ant colonies' motion trajectories in indoor and outdoor scenes to study clustering behavior*, GigaScience 11, giac096, 2022. Data: [doi:10.5524/102254](https://doi.org/10.5524/102254), [Mendeley doi:10.17632/9ws98g4npw.4](https://doi.org/10.17632/9ws98g4npw.4) | P. Fernández-López, D. Oro, R. Lloret-Cabot, M. Genovart, J. Garriga, F. Bartumeus, *Foraging ants as liquid brains: movement heterogeneity shapes collective efficiency*, PNAS, 2025, [doi:10.1073/pnas.2506930122](https://doi.org/10.1073/pnas.2506930122). Data: [Mendeley doi:10.17632/4xvxkd6jxh.1](https://doi.org/10.17632/4xvxkd6jxh.1); code: [github.com/Polfe94/liquid_brains](https://github.com/Polfe94/liquid_brains) |

The data is **not** in this repository. Keep it in a folder of your own (called `<DATA>` below),
for example `<DATA>/ants/`, with a note of where each file came from.

## Getting the files

Both datasets are on Mendeley Data and download without an account.

- **ANTS**: `Ant_dataset.zip` (750 MB) has, per sequence, `seqinfo.ini`, `gt/gt.txt` and `img/`.
  Unzip it under `<DATA>/ants/ANTS/`.
- **Liquid Brains**: from `raw_data/Food/` and `raw_data/NoFood/` the trial CSVs you want (e.g.
  `Food_Exp_1_data.csv`, `NoFood_Exp_1_data.csv`), and from `processed_data/` the maze nodes
  `hex.csv` and `det.RData` (the food items of the Food trials). Put them under
  `<DATA>/ants/liquid_brains/`, keeping the two folder names.
- The food items are in R format. Turn them into a CSV once with R (base R, no packages); the
  command is in the docstring of [tools/convert_ants.py](../tools/convert_ants.py). It writes
  `food_items.csv` with `exp, patch, item, x, y, t` (t = the discovery frame).

## Converting

[tools/convert_ants.py](../tools/convert_ants.py) writes the replay CSV (`id,frame,t,x,y`) next to
the data, a places file (Liquid Brains) and a starter scene config `<name>.scene.json`. A config
that already exists is kept, so you can edit it.

```
# ANTS: one or more sequence folders; --video also builds <name>.mp4 from img/ (OpenCV)
python tools/convert_ants.py ants <DATA>/ants/ANTS/Ant_dataset/OutdoorDataset/Seq0009Object50Image64 --out <DATA>/ants/converted/ants --video

# Liquid Brains: a Food trial with its food items, and a No-Food trial
python tools/convert_ants.py liquid <DATA>/ants/liquid_brains/raw_data/Food/Food_Exp_1_data.csv --hex <DATA>/ants/liquid_brains/processed_data/hex.csv --food <DATA>/ants/liquid_brains/food_items.csv --exp 1 --out <DATA>/ants/converted/liquid_brains
python tools/convert_ants.py liquid <DATA>/ants/liquid_brains/raw_data/NoFood/NoFood_Exp_1_data.csv --hex <DATA>/ants/liquid_brains/processed_data/hex.csv --out <DATA>/ants/converted/liquid_brains
```

What the converters do:

| | ANTS | Liquid Brains |
|---|---|---|
| Position | box centre: `left + w/2`, `top + h/2`, in image pixels, y down (no flip, no homography) | `Xmm`, `Ymm` as they are |
| Frame and time | `frame = gt frame - 1` (frame 0 = `img/000001.jpg` = video frame 0), `t = frame / frameRate` of the sequence's `seqinfo.ini` | `Frame` as it is, `t = Time_sec` (= frame / 2) |
| Other | the box size becomes the body length (an estimate); name from the folder (the outdoor `seqinfo.ini` names say "Object10" for every sequence) | the R row names are dropped (each row has one more field than the header); places file: nest = maze node 634 (as in the authors' figure code), maze edges between arena nodes (y > 1000 mm) closer than 51 mm, food items with `t` in seconds |

## The scene config

`--replay-config FILE.json` (kept with the data). Relative paths in it are relative to the file.
Without it the replay is the pedestrian one, unchanged. The starter config of `Food_Exp_1`, as
written by the converter:

```json
{
  "name": "Food_Exp_1",
  "citation": "Liquid Brains: Fernandez-Lopez et al. 2025",
  "reference": "Fernandez-Lopez, Oro, Lloret-Cabot, Genovart, Garriga & Bartumeus, Foraging ants as liquid brains: movement heterogeneity shapes collective efficiency, PNAS 2025 (data CC BY 4.0)",
  "agents": "ants",
  "units": "mm",
  "body_length": 6.0,
  "body_length_is_estimate": true,
  "comment": "Aphaenogaster senilis in a Y-maze; body length 6 mm is an estimate; y flipped so the nest is at the bottom as in the paper",
  "flip_y": true,
  "grid": 50,
  "thresholds": {"standing_bl_s": 0.3, "speed_bl_s": [0.5, 5.0], "close_bl": [0.5, 3.0]},
  "places_file": "Food_Exp_1.places.json",
  "places": {"nest": {"r": 60}},
  "events": {"arrivals": "nest", "together": false, "food_discovery": true, "recruitment": true, "encounters": true}
}
```

| Key | Meaning | Used now |
|---|---|---|
| `name`, `citation`, `reference` | scene name; short citation (window title, caption); long reference (HUD `source:` line) | yes |
| `agents` | what is on screen, for the HUD ("ants on screen 27") | yes |
| `units` | units of x and y: `mm`, `px`, `m`... (scale bar, speed legend) | yes |
| `body_length`, `body_length_is_estimate` | one body length (BL) in those units; the scale bar also shows BL | yes |
| `flip_y` | the data's y points up (same as `--replay-flip-y`; either one flips) | yes |
| `grid` | map grid step in data units (default 10 BL; without a body length, 1 unit) | yes |
| `thresholds` | in body lengths: `standing_bl_s` (slower = standing), `speed_bl_s` (low, high), `close_bl` (near, far) | read and checked; used by the ant mapping (next step) |
| `places_file` | a places file (below) | yes |
| `places.nest` | `x`, `y`, `r` (data units); its keys win over the places file's (e.g. only `r`); default r = 10 BL | drawn |
| `places.food` | a list of `{x, y, patch, t}`; replaces the places file's food | drawn |
| `events` | which events apply: `arrivals` (`anywhere`, `nest` or `off`), `together`, `food_discovery`, `recruitment`, `encounters` (true / false) | read and checked; used next step |
| `image`, `video` | a frame and a video in the data's pixel coordinates (camera / video views); `--replay-image`, `--replay-video` override them | yes |
| `image_credit`, `video_credit` | credit shown in the caption in those views | yes |
| `comment` | free text, ignored | |

The config is checked when it is loaded. An unknown key, thresholds without a body length,
`arrivals: "nest"` without a nest, or `food_discovery` without any food time stops the app with
the reason.

**Places file** (`<name>.places.json`, written by the converter for Liquid Brains):
`{"units": "mm", "source": ..., "nest": {"x", "y", "node"}, "food": [{"patch", "item", "x", "y",
"t"}], "maze_edges": [[x1, y1, x2, y2], ...]}`. `t` is the discovery time in data seconds, the
same clock as the CSV's `t`; `null` = never found.

For ANTS the converter writes `units: "px"`, the box size as the body length, `image` (the first
jpg) and, with `--video`, `video`. The outdoor sequences have a nest entrance in the picture: its
place is not in the data, so add it by hand (e.g. `"places": {"nest": {"x": 720, "y": 300, "r":
128}}` for Seq0009, placed by eye on frame 1).

## Views

```
# Liquid Brains, 20x: the maze, the nest, the food items (filled once found)
python src/main.py --replay <DATA>/ants/converted/liquid_brains/Food_Exp_1.csv --replay-config <DATA>/ants/converted/liquid_brains/Food_Exp_1.scene.json --replay-speed 20

# ANTS outdoor over its video (video environment), looping
python src/main.py --replay <DATA>/ants/converted/ants/Seq0009Object50Image64.csv --replay-config <DATA>/ants/converted/ants/Seq0009Object50Image64.scene.json --replay-view video --replay-loop

# any data already in image pixels, without a config
python src/main.py --replay <FILE>.csv --replay-view video --replay-image <FRAME>.jpg --replay-video <VIDEO>.mp4
```

- **Map** (both datasets): the grid at the config's step, and the scale bar in data units and
  body lengths. With a config there are also the maze edges, the nest (a ring) and the food items
  (hollow squares, filled from their discovery time on), drawn in every view.
- **Camera** (`--replay-view camera`): the frame from `--replay-image` (or the config's `image`)
  behind the ants, with x, y taken as its pixels (no homography). It is a still frame; the ants
  move over it.
- **Video** (`--replay-view video`): the video from `--replay-video` (or the config's `video`),
  frame = fps × data time, so frame 0 is the first jpg. It needs an image too (the camera view is
  the fallback). The `V` key (normal, dark, off) works as for the pedestrians.
- The sound profiles, events and keys are those of the pedestrian replay ([REPLAY.md](REPLAY.md)).

## Known limits

- The *A. senilis* body length (6 mm) is an estimate. For ANTS the body length is the box size,
  also an estimate. The thresholds in body lengths inherit both.
- The ANTS nest position is placed by eye.
- ANTS sequences are short (14–40 s): loops, or half speed, rather than a piece with an arc.
- Some ants visible in the ANTS outdoor frames have no box, so they are not in the data.
- Liquid Brains ids are fragmented: an ant lost and found again gets a new id (613 ids in
  Food_Exp_1, median track 14 s). Arrivals will count only tracks that start near the nest.
- A 3-hour Liquid Brains trial takes about 10 s to load (speed calibration over the whole file).
- The first ~10 minutes of a Liquid Brains trial are almost empty (the ants have not come out yet).
