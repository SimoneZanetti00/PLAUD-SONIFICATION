# Changes to the PLAUD device

The PLAUD device is not in this repository. This page lists **every object and connection this
project added** to our modified copy of the PLAUD device, so you can make the same changes by hand
in yours.

The SynthFish routes that came before (`[udpreceive 7400]`, the `/plaud/*`, `/fx/*`, `/gran/*` and
`/mix/*` routes, the `prior~` edits) are in [MAX_SETUP.md](MAX_SETUP.md). Add those first.

Only `--replay-profile scene` uses these changes ([SYSTEM.md](SYSTEM.md#sound-profile-scene)). A
device without them works with every other mode: the new addresses are simply ignored.

## Summary

| Added | Where | Connects to | Used by |
|---|---|---|---|
| object `[route /plaud/limit /plaud/centroid /plaud/stretch]` | the device's main patcher, next to the existing `[route /plaud/x /plaud/y /plaud/loudness /plaud/temp]` | inlet ← `[udpreceive 7400]` outlet 0 | profile scene |
| patchline | route outlet 0 (`/plaud/limit`) | **Limit** dial (label "lim"), left inlet | profile scene |
| patchline | route outlet 1 (`/plaud/centroid`) | **Centroid** dial, left inlet | profile scene |
| patchline | route outlet 2 (`/plaud/stretch`) | **SpectralStretch** dial (label "str"), left inlet | profile scene |

The route's last outlet (no match) stays unconnected. Nothing existing was removed or changed.

| Address | Value | Dial | Dial range | Scripting name |
|---|---|---|---|---|
| `/plaud/limit` | 0–1 | Limit ("lim", bending): 0 = all partials, 1 = one | 0–1 | `limit` |
| `/plaud/centroid` | 0–1 | Centroid (latent 2 offset) | 0–1 | `centroid` |
| `/plaud/stretch` | 0–0.5 (profile scene) | SpectralStretch ("str", bending): + = toward highs | −1–1 | `spectral_stretch` |

## How to add it by hand

1. Open the device's main patcher and unlock it (patching mode, not presentation).
2. Find `[udpreceive 7400]` and the existing `[route /plaud/x /plaud/y /plaud/loudness /plaud/temp]`
   below it.
3. Make a new object next to that route (double-click, or `N`) and type:
   `route /plaud/limit /plaud/centroid /plaud/stretch`
   It gets four outlets: one per address, and the last for anything else.
4. Connect the left outlet of `[udpreceive 7400]` to the left inlet of the new route.
   `udpreceive` already feeds the other routes; one more connection from the same outlet is fine.
5. Connect the new route's outlets to the dials' **left inlets**:
   - outlet 0 → the **Limit** dial (in presentation it is labelled "lim", in the bending section,
     bottom right);
   - outlet 1 → the **Centroid** dial (top left, next to Loudness);
   - outlet 2 → the **SpectralStretch** dial (labelled "str", bending section, bottom left).
   To find a dial in patching mode, select it in presentation mode first, then switch view: it
   stays selected. Its Inspector shows the Long Name above (Limit, Centroid, SpectralStretch).
6. Save the device (keep a backup copy of the file first).

A number into a `live.dial`'s left inlet sets it and passes the value on, so each address moves
its dial exactly as if you turned it. Like the other macro dials (Loudness, L1, L2), the Centroid
dial does not reach the sound while **bypass** is on.

## Check it

- Run a replay with the scene profile, for example
  `python src/main.py --replay <DATA>/eth_ucy/zara1/crowds_zara01.txt --replay-profile scene`,
  and watch the three dials move (the HUD, `H`, shows the values sent: lim, centroid, str).
- Or test without Python: a `[udpsend 127.0.0.1 7400]` fed by message boxes
  `/plaud/limit 0.5`, `/plaud/centroid 0.8`, `/plaud/stretch 0.3`.

## Interaction with the SynthFish routes

`/plaud/loudness` also sets the Limit dial through the SynthFish `[scale 0.7 1.4 1. 0.]`. Profile
scene sends `/plaud/limit` right after `/plaud/loudness`, every time, so its value is the one that
stays. Profile synthfish never sends `/plaud/limit`, so there the SynthFish behaviour is unchanged.

## To undo

Delete the new route object; its three patchlines go with it.

## Planned (not added yet)

The prior's **cfg**, **tmp** and **smh** (inside `prior~`) have no OSC route. They will need a
route in the main patcher and a `send` / `receive` pair into `prior~`, the same way `/plaud/x`
reaches territory_x (`s plaudX` → `r plaudX`). They will be listed here when they are added.
