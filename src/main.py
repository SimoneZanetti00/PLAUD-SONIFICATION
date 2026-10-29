from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import pygame
from pygame.math import Vector2

from aquarium_boids.config import (
    BOID_COUNT,
    BPM,
    CONTROL_PORT,
    ENABLE_MIDI_INPUT,
    FPS,
    GRAN_MAX_VOICES,
    HEIGHT,
    JOY_DEAD_ZONE,
    JOY_INVERT_Y,
    JOY_MARGIN,
    JOY_SMOOTHING,
    MIDI_CC_MAPPING,
    MIDI_DEBUG,
    MIDI_INPUT_NAME,
    MIDI_PROFILES,
    OSC_HOST,
    OSC_PORT,
    PAD_PREDATOR_BURST_S,
    SEND_DESCRIPTOR_HZ,
    WIDTH,
)
from aquarium_boids.control_in import ControlReceiver
from aquarium_boids.controls import RuntimeControls, clamp01
from aquarium_boids.descriptors import DESCRIPTOR_ORDER, OnePoleSmoother, compute_descriptors
from aquarium_boids.markov import LAYERS, SECTIONS, SymbolicGenerator
from aquarium_boids.midi_in import MidiControllerInput
from aquarium_boids.osc_io import OscSender
from aquarium_boids.render import ClassicRenderer, Display, ThermalRenderer
from aquarium_boids.render_data import CAMERA_DIM, DataRenderer
from aquarium_boids.replay.camera import ENV_DIR as CAMERA_ENV_DIR, find_camera, find_video
from aquarium_boids.replay.events import EventParams
from aquarium_boids.replay.loader import FORMATS as REPLAY_FORMATS
from aquarium_boids.sources import ReplaySource, SimulationSource


BASE_DIR = Path(__file__).resolve().parent


def write_log_header(path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", *DESCRIPTOR_ORDER, "density_fader", "section"])


def append_log(path: Path, now: float, descriptors: dict[str, float], controls: RuntimeControls) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{now:.3f}",
            *[f"{descriptors.get(key, 0.0):.5f}" for key in DESCRIPTOR_ORDER],
            f"{controls.density_fader:.5f}",
            controls.section_name,
        ])


def handle_key(
    event: pygame.event.Event, controls: RuntimeControls, source: SimulationSource | ReplaySource
) -> None:
    step = 0.05
    if event.key == pygame.K_ESCAPE:
        pygame.quit()
        sys.exit(0)
    if event.key == pygame.K_SPACE:
        controls.paused = not controls.paused
    elif event.key == pygame.K_UP:
        controls.density_fader = clamp01(controls.density_fader + step)
    elif event.key == pygame.K_DOWN:
        controls.density_fader = clamp01(controls.density_fader - step)
    elif event.key == pygame.K_a:
        controls.alignment_weight += 0.1
    elif event.key == pygame.K_z:
        controls.alignment_weight = max(0.0, controls.alignment_weight - 0.1)
    elif event.key == pygame.K_s:
        controls.cohesion_weight += 0.1
    elif event.key == pygame.K_x:
        controls.cohesion_weight = max(0.0, controls.cohesion_weight - 0.1)
    elif event.key == pygame.K_d:
        controls.separation_weight += 0.1
    elif event.key == pygame.K_c:
        controls.separation_weight = max(0.0, controls.separation_weight - 0.1)
    elif event.key == pygame.K_n:
        controls.noise_weight += 0.03
    elif event.key == pygame.K_m:
        controls.noise_weight = max(0.0, controls.noise_weight - 0.03)
    elif event.key == pygame.K_MINUS:
        controls.population = max(1, controls.population - 5)
    elif event.key == pygame.K_EQUALS:
        controls.population = min(BOID_COUNT, controls.population + 5)
    elif event.key == pygame.K_r:
        source.reset(controls.population)
    elif pygame.K_1 <= event.key <= pygame.K_6:
        controls.section_id = event.key - pygame.K_1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SynthFish: a flock of fish -> OSC -> Max")
    parser.add_argument(
        "--classic",
        action="store_true",
        help="original look (flat background, triangle fish, text HUD) instead of the thermal look",
    )
    parser.add_argument(
        "--mode",
        choices=ThermalRenderer.MODES,
        default="art",
        help="thermal look: 'art' = no text (default, for recording), 'data' = with HUD; H toggles",
    )
    parser.add_argument("--no-trails", action="store_true", help="thermal look: start without trails (T toggles)")
    parser.add_argument("--no-glow", action="store_true", help="thermal look: start without glow (G toggles)")
    parser.add_argument(
        "--no-particles", action="store_true", help="thermal look: start without drifting particles (P toggles)"
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        default=(WIDTH, HEIGHT),
        metavar="WxH",
        help=f"window size, e.g. 1920x1080 (default {WIDTH}x{HEIGHT}); the simulation stays {WIDTH}x{HEIGHT}",
    )
    parser.add_argument("--fullscreen", action="store_true", help="start fullscreen (F11 toggles)")
    replay = parser.add_argument_group(
        "replay",
        "move the fish along recorded trajectories (CSV id,frame,x,y[,t] or ETH/UCY .txt); no MIDI in this mode",
    )
    replay.add_argument("--replay", metavar="FILE", help="trajectory file to replay instead of the boid simulation")
    replay.add_argument(
        "--replay-format", choices=REPLAY_FORMATS, default="auto",
        help="file format; auto (default) = eth-ucy for .txt (frame id x y, no header), csv otherwise",
    )
    replay.add_argument(
        "--replay-fps", type=positive_float, default=2.5, metavar="F",
        help="data rows per second when the CSV has no t column (default 2.5)",
    )
    replay.add_argument(
        "--replay-speed", type=positive_float, default=1.0, metavar="S", help="playback speed, 1 = real time (default 1)"
    )
    replay.add_argument("--replay-loop", action="store_true", help="start again at the end of the data")
    replay.add_argument("--replay-flip-y", action="store_true", help="data y points up (pygame's y points down)")
    replay.add_argument(
        "--replay-max-agents", type=positive_int, default=BOID_COUNT, metavar="N",
        help=f"most agents on screen at once (default {BOID_COUNT})",
    )
    replay.add_argument(
        "--replay-look", choices=("data", "thermal"), default="data",
        help="data (default): grid, scale bar, paths, speed colours, I = id labels; thermal: people in the "
        "thermal look; --classic overrides both",
    )
    replay.add_argument(
        "--replay-view", choices=("map", "camera", "video"), default="map",
        help="data look: map (default, metric grid), camera (the scene's OpenTraj reference frame, dimmed, "
        "with the people projected into it) or video (the original video behind them, frame = 25 x data "
        "time; needs OpenCV, see requirements-video.txt); camera and video: zara1 and hotel only",
    )
    replay.add_argument(
        "--replay-camera-dir", metavar="DIR",
        help=f"folder with the OpenTraj frames, homographies and videos (zara01/, seq_hotel/); "
        f"default ${CAMERA_ENV_DIR}",
    )
    replay.add_argument(
        "--replay-no-events", action="store_true",
        help="no acceleration events (by default an agitation spike sets the predator: /fx/predator + rings)",
    )
    replay.add_argument(
        "--replay-event-z", type=positive_float, default=3.0, metavar="Z",
        help="event onset threshold, z-score of the agitation over the last 10 s (default 3)",
    )
    replay.add_argument(
        "--replay-event-refractory", type=positive_float, default=5.0, metavar="S",
        help="seconds of data time without a new event after one ends (default 5)",
    )
    replay.add_argument(
        "--replay-normalise", action="store_true",
        help="map each descriptor to this file's p5..p95 range (fish_count: / most agents on screen)",
    )
    return parser.parse_args(argv)


def positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}")
    if not (value > 0 and value != float("inf")):
        raise argparse.ArgumentTypeError(f"must be a finite number > 0, got {text}")
    return value


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {text}")
    return value


def load_replay(args: argparse.Namespace) -> ReplaySource:
    """Load --replay before any window opens; a bad file exits with its message."""
    try:
        source = ReplaySource(
            args.replay,
            fmt=args.replay_format,
            fps=args.replay_fps,
            speed=args.replay_speed,
            loop=args.replay_loop,
            flip_y=args.replay_flip_y,
            max_agents=args.replay_max_agents,
            normalise=args.replay_normalise,
            events=not args.replay_no_events,
            event_params=EventParams(on=args.replay_event_z, refractory=args.replay_event_refractory),
        )
    except (OSError, ValueError) as exc:
        sys.exit(f"--replay: {exc}")
    print(f"[replay] {source.summary}")
    return source


def replay_camera(args: argparse.Namespace):
    """--replay-view camera / video -> (camera, video); anything missing falls back with the reason:
    video -> camera view -> map view (None, None)."""
    if args.replay_view == "map":
        return None, None
    if args.classic or args.replay_look != "data":
        print(f"[replay] {args.replay_view} view needs the data look; using the map view")
        return None, None
    camera, why = find_camera(args.replay, args.replay_camera_dir)
    if camera is None:
        print(f"[replay] {args.replay_view} view not available: {why}; using the map view")
        return None, None
    if args.replay_view == "video":
        from aquarium_boids.replay.video import VideoFrames  # cv2 is imported only inside

        path, why = find_video(args.replay, args.replay_camera_dir)
        if path is not None:
            try:
                video = VideoFrames(path, dim=CAMERA_DIM)
            except RuntimeError as exc:
                why = str(exc)
            else:
                print(f"[replay] video view: {path} ({video.count} frames at {video.fps:g} fps)")
                return camera, video
        print(f"[replay] video view not available: {why}; using the camera view")
    print(f"[replay] camera view: {camera.image_path}")
    return camera, None


def parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WxH, e.g. 1920x1080, got {text!r}")
    if w < 320 or h < 200:
        raise argparse.ArgumentTypeError("size must be at least 320x200")
    return (w, h)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    replay = load_replay(args) if args.replay else None
    pygame.init()
    title = replay.title if replay is not None else "SynthFish - boids -> OSC -> Max"
    display = Display((WIDTH, HEIGHT), args.size, title, fullscreen=args.fullscreen)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    camera, video = replay_camera(args) if replay is not None else (None, None)
    # Replay only: people instead of fish, a caption with the scene and the data time, replay HUD text.
    replay_look = {"people": True, "caption": replay.caption, "hud": replay.hud_lines} if replay is not None else {}
    renderer = (
        ClassicRenderer(font, **replay_look)
        if args.classic
        else DataRenderer(font, replay, mode=args.mode, camera=camera, video=video)
        if replay is not None and args.replay_look == "data"
        else ThermalRenderer(
            font,
            mode=args.mode,
            trails=not args.no_trails,
            glow=not args.no_glow,
            particles=not args.no_particles,
            **replay_look,
        )
    )

    controls = RuntimeControls()
    source = replay if replay is not None else SimulationSource(BOID_COUNT)
    smoother = OnePoleSmoother(alpha=0.16)
    osc = OscSender(OSC_HOST, OSC_PORT)
    generator = SymbolicGenerator()
    control_receiver = ControlReceiver(port=CONTROL_PORT)
    control_receiver.start()
    print(f"Listening for Max/MIDIMIX controls on UDP {CONTROL_PORT}")

    midi_input = MidiControllerInput(
        device_name=MIDI_INPUT_NAME,
        mapping=MIDI_CC_MAPPING,
        debug=MIDI_DEBUG,
        profiles=MIDI_PROFILES,
        joy_dead_zone=JOY_DEAD_ZONE,
        joy_invert_y=JOY_INVERT_Y,
    )
    if ENABLE_MIDI_INPUT and replay is None:
        midi_input.start()
    if replay is not None:
        print("[midi] off in replay mode")
    else:
        print(f"[midi] active profile: {midi_input.profile_name if ENABLE_MIDI_INPUT else 'none (MIDI disabled in config)'}")

    log_path = BASE_DIR / "logs" / f"descriptors_{int(time.time())}.csv"
    write_log_header(log_path)

    last_descriptor_send = 0.0
    last_log = 0.0
    last_print = 0.0
    next_event_time = 0.0
    descriptors: dict[str, float] = {}
    gran = [0.0, 0.0, 1, 0.0]
    plaud_xy = (0.0, 0.0)
    predator_active = False
    predator_burst_until = 0.0  # pad 2
    food_pad_on = False  # pad 3
    joy_food: Vector2 | None = None  # MPK joystick food point (smoothed), None while the stick rests
    joy_food_since = 0.0
    mouse_food_since = 0.0
    mouse_left_was = False

    while True:
        dt = clock.tick(FPS) / (1000.0 / FPS)
        now = time.time()

        applied_controls = control_receiver.drain(controls)
        for command in applied_controls:
            print(f"[control] {command.name} = {command.value}")

        midi_controls = midi_input.poll(controls)
        for command in midi_controls:
            print(f"[midi-map] cc={command.cc} -> {command.parameter} = {command.scaled_value}")
            if command.parameter.startswith("/"):  # knob mapped to a Max mix level, not to the flock
                osc.send_mix(command.parameter, command.scaled_value)

        for pad in midi_input.take_pads():  # only controllers whose profile defines pads (MPK)
            if pad == 1:
                osc.send_pad_kick()
            elif pad == 2:  # predator at the centre for a moment: same path as the mouse right button
                predator_burst_until = now + PAD_PREDATOR_BURST_S
            elif pad == 3:  # food at the centre on/off
                food_pad_on = not food_pad_on
            elif pad == 4:  # clear food and predator
                food_pad_on = False
                predator_burst_until = 0.0
                controls.food_amount = 0.0
                controls.predator_amount = 0.0
            elif 5 <= pad <= 8:
                osc.send_pad_sample(pad - 4)

        transpose = midi_input.take_transpose()  # MPK keys: latched transpose for the grain source
        if transpose is not None:
            osc.send_transpose(transpose)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                osc.send_fx_predator(False)  # never leave Max in a predator scare
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    osc.send_fx_predator(False)  # handle_key quits on ESC
                if event.key == pygame.K_F11:
                    display.toggle_fullscreen()
                    continue
                if renderer.handle_key(event.key):
                    continue
                handle_key(event, controls, source)

        mouse_buttons = pygame.mouse.get_pressed(num_buttons=3)
        mouse_pos = display.view.to_world(pygame.mouse.get_pos())
        attractor = mouse_pos if mouse_buttons[0] else None
        if mouse_buttons[0] and not mouse_left_was:
            mouse_food_since = now
        mouse_left_was = mouse_buttons[0]
        # MPK joystick: food at centre + deflection (full = near the edge), glides there, gone in the dead zone.
        joy = midi_input.joystick()
        if joy is None:
            joy_food = None
        else:
            target = Vector2(WIDTH * 0.5 + joy[0] * (WIDTH * 0.5 - JOY_MARGIN),
                             HEIGHT * 0.5 - joy[1] * (HEIGHT * 0.5 - JOY_MARGIN))  # joystick up = screen up
            if joy_food is None:
                joy_food = Vector2(WIDTH * 0.5, HEIGHT * 0.5)
                joy_food_since = now
            joy_food += (target - joy_food) * JOY_SMOOTHING
        if joy_food is not None and (attractor is None or joy_food_since > mouse_food_since):
            attractor = Vector2(joy_food)  # same food path as the mouse; the most recent source wins
        repeller = mouse_pos if mouse_buttons[2] else None
        if attractor is None and food_pad_on:
            attractor = Vector2(WIDTH * 0.5, HEIGHT * 0.5)
        if repeller is None and now < predator_burst_until:
            repeller = Vector2(WIDTH * 0.5, HEIGHT * 0.5)
        if attractor is None and controls.food_amount > 0.01:
            attractor = Vector2(WIDTH * 0.5, HEIGHT * 0.5)
        if repeller is None and controls.predator_amount > 0.01:
            repeller = Vector2(WIDTH * 0.5, HEIGHT * 0.5)
        if repeller is None and source.repeller is not None:  # replay: an acceleration event
            repeller = source.repeller

        # Predator scare onset/release for Max: the same state the boids react to (repeller set).
        if (repeller is not None) != predator_active:
            predator_active = repeller is not None
            if predator_active:
                osc.send_fx_predator_strength(controls)  # so Max has the strength before the first kick
            osc.send_fx_predator(predator_active)

        flock = source.step(dt, controls, attractor, repeller)

        descriptors = smoother.process(source.normalise(compute_descriptors(flock, WIDTH, HEIGHT)))

        if now - last_descriptor_send >= 1.0 / SEND_DESCRIPTOR_HZ:
            osc.send_descriptors(descriptors)
            osc.send_controls(controls)
            osc.send_direct_mapping(descriptors, controls)
            gran = osc.send_granular(descriptors, controls)
            plaud_xy = osc.send_plaud(descriptors)
            if predator_active:
                osc.send_fx_predator_strength(controls)
            osc.send_fish_x(flock, WIDTH, GRAN_MAX_VOICES)
            last_descriptor_send = now

        # Clean terminal readout: granular controls + PLAUD latent point (~2x/sec).
        if now - last_print >= 0.5:
            pan, density, nchan, noise = gran
            px, py = plaud_xy
            print(
                f"GRAN fish={controls.population:3d} density={density:5.1f} nchan={nchan:2d} noise={noise:.2f}  "
                f"|  PLAUD x={px:+.2f} y={py:+.2f}  |  coherence={descriptors.get('direction_coherence', 0):.2f} "
                f"spread={descriptors.get('spread', 0):.2f}"
            )
            last_print = now

        if now - last_log >= 0.25:
            append_log(log_path, now, descriptors, controls)
            last_log = now

        # Event rate: collective/slow at low density, granular/fast at high density and high fish speed.
        if now >= next_event_time:
            burst = 1 + int(controls.density_fader * 3.2)
            for _ in range(burst):
                event = generator.generate(descriptors, controls)
                osc.send_music_event(event)
                if event.event_type == "note" and flock:
                    random.choice(flock).flash = 1.0
            events_per_second = 0.45 + controls.density_fader * 5.5 + descriptors.get("mean_speed", 0.0) * 2.0
            next_event_time = now + max(0.08, 1.0 / events_per_second)

        renderer.draw(display.canvas, display.view, flock, attractor, repeller, controls, descriptors)
        pygame.display.flip()


if __name__ == "__main__":
    main()
