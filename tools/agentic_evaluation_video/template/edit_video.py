# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Edit KitchenBench construction, policy selection and verified policy rollouts."""
import argparse
import json
import math
import subprocess
import textwrap
from functools import lru_cache
from pathlib import Path

from narration import export_timing, rollout_length, rollout_rate, timeline, visual_time
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent


def ease(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)


BG, PANEL, INK, MUTED = "#ffffff", "#f7f9f5", "#20251c", "#687160"
NVIDIA_GREEN, ACCENT_TEXT, BORDER = "#76b900", "#466f00", "#b2bea7"
TEAL, BLUE, GOLD = ACCENT_TEXT, ACCENT_TEXT, ACCENT_TEXT
CORAL, VIOLET = ACCENT_TEXT, MUTED


@lru_cache(None)
def font(size, mono=False, bold=None):
    use_bold = (mono or size >= 33) if bold is None else bold
    weight = "Bd" if use_bold else "Rg"
    return ImageFont.truetype(str(ROOT / "assets/fonts" / f"NVIDIASans_{weight}.woff2"), size)


@lru_cache(None)
def terminal_font(size=28):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)


@lru_cache(None)
def symbol_font(size, bold=False):
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf", size)


def draw_text(draw, xy, value, font, fill, anchor=None, spacing=15):
    # NVIDIA's Latin webfonts omit pi and arrows; keep the surrounding type in NVIDIA Sans.
    if not any(c in value for c in "π→✓"):
        draw.text(xy, value, font=font, fill=fill, anchor=anchor, spacing=spacing)
        return
    runs = []
    for c in value:
        chosen = symbol_font(font.size, font.getname()[1] == "Bold") if c in "π→✓" else font
        if runs and runs[-1][1] is chosen:
            runs[-1] = (runs[-1][0] + c, chosen)
        else:
            runs.append((c, chosen))
    width = sum(f.getlength(part) for part, f in runs)
    boxes = [f.getbbox(part, anchor="ls") for part, f in runs]
    top = min(b[1] for b in boxes)
    bottom = max(b[3] for b in boxes)
    anchor = anchor or "la"
    x, y = xy
    if anchor[0] == "m":
        x -= width / 2
    elif anchor[0] == "r":
        x -= width
    if anchor[1] == "m":
        y -= (top + bottom) / 2
    elif anchor[1] == "a":
        y += font.getmetrics()[0]
    elif anchor[1] == "t":
        y -= top
    elif anchor[1] == "b":
        y -= bottom
    for part, f in runs:
        draw.text((x, y), part, font=f, fill=fill, anchor="ls")
        x += f.getlength(part)


def text(draw, xy, value, size=30, color=INK, mono=False):
    draw_text(draw, xy, value, font=font(size, mono), fill=color, spacing=15)


def page_title(draw):
    draw_text(draw, (55, 22), "Isaac Lab - Arena", font=font(52, bold=True), fill="#000000")
    draw_text(draw, (55, 84), "Agentic Evaluation Authoring", font=font(36, bold=False), fill="#737373")


def arrow(draw, start, end, color=ACCENT_TEXT, progress=1):
    import math

    end = tuple(a + (b - a) * progress for a, b in zip(start, end))
    draw.line((start, end), fill=color, width=3)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    points = [end]
    for offset in (-0.48, 0.48):
        points.append((end[0] - 12 * math.cos(angle + offset), end[1] - 12 * math.sin(angle + offset)))
    draw.polygon(points, fill=color)


STEMS = ["01_kitchen", "02_wooden_bowl", "03_robot", "04_fruits_far_bowl"]
STARTS = [2, 9, 18, 27]
LENGTHS = [7, 9, 9, 12]
AUTHORING_SPEED = 1.4
FPS = 30
SAMPLE_COUNT = 5
SAMPLE_INTERVAL = 0.7
CAPTURE_SPEED = 3.5
PROMPT = "Design an evaluation environment in a kitchen where two pieces of fruit must be placed into two bowls."
COMMANDS = [
    "Adding the peninsula kitchen...",
    "Adding the wooden bowl...",
    "Placing the robot at the island...",
    "Adding banana, orange, and red bowl...",
    "Defining two pick & place tasks...",
    "Adding π0.5 as the policy...",
    "Running the evaluation...",
]
SCENE_BOUNDS = (1050, 550, 1950, 1010)


@lru_cache(None)
def isaac_lab_icon():
    return Image.open(ROOT / "assets/isaaclab_icon.png").convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)


@lru_cache(None)
def codex_icon():
    return Image.open(ROOT / "assets/codex_icon.png").convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)


@lru_cache(None)
def physical_intelligence_icon():
    source = Image.open(ROOT / "assets/physical_intelligence_highres.png").convert("RGB")
    alpha = ImageChops.invert(source.convert("L"))
    icon = Image.new("RGBA", source.size, (0, 0, 0, 0))
    icon.putalpha(alpha)
    return icon.crop(icon.getbbox()).resize((64, 62), Image.Resampling.LANCZOS)


@lru_cache(None)
def omniverse_icon():
    return Image.open(ROOT / "assets/omniverse_icon.png").convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)


CAPTIONS = [
    "Anchor the kitchen workspace.",
    "Add the wooden bowl as a spatial reference.",
    "Place and orient the robot through relations.",
    "Add the banana, orange and farther bowl together.",
]


@lru_cache(None)
def capture_directory(stage):
    sources = ROOT / "capture_sources.json"
    if sources.exists():
        return ROOT / json.loads(sources.read_text())[STEMS[stage]]
    return ROOT / "captures" / STEMS[stage]


@lru_cache(150)
def frame(stage, layout, index, size=(840, 630)):
    path = capture_directory(stage) / f"layout_{layout:02d}" / f"frame_{index:03d}.jpg"
    return Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)


@lru_cache(None)
def period(stage):
    return json.loads((capture_directory(stage) / "captures.json").read_text())[0]["trajectory"][1]["time_s"]


def section(d, bounds, title, color, center=False):
    d.rounded_rectangle(bounds, radius=12, fill="#eeeeee", outline=BORDER, width=4)
    x1, y1, x2, y2 = bounds
    draw_text(
        d,
        ((x1 + x2) / 2 if center else x1 + 22, y1 + 25),
        title,
        font=font(25),
        fill=INK,
        anchor="mm" if center else "lm",
    )


# Individually rendered USD assets with depth-derived transparent backgrounds.
@lru_cache(None)
def asset_thumbnail(key):
    source = Image.open(ROOT / "assets/rendered" / f"{key}.png").convert("RGBA")
    source = source.crop(source.getbbox())
    crop = ImageOps.contain(source, (42, 42), Image.Resampling.LANCZOS)
    thumb = Image.new("RGBA", (42, 42), (0, 0, 0, 0))
    thumb.paste(crop, ((42 - crop.width) // 2, (42 - crop.height) // 2))
    return thumb


def dashed_box(d, bounds, color):
    x1, y1, x2, y2 = bounds
    for x in range(int(x1), int(x2), 15):
        d.line((x, y1, min(x + 8, x2), y1), fill=color, width=3)
        d.line((x, y2, min(x + 8, x2), y2), fill=color, width=3)
    for y in range(int(y1), int(y2), 15):
        d.line((x1, y, x1, min(y + 8, y2)), fill=color, width=3)
        d.line((x2, y, x2, min(y + 8, y2)), fill=color, width=3)


def node(canvas, d, xy, key, label, color, reference=False):
    x, y = xy
    left = x - 110
    bounds = (left, y - 26, x + 110, y + 26)
    if reference:
        d.rectangle(bounds, fill="#ffffff")
        dashed_box(d, bounds, color)
    else:
        d.rounded_rectangle(bounds, radius=9, fill="#ffffff", outline=color, width=3)
    canvas.paste(asset_thumbnail(key), (int(left + 6), int(y - 21)), asset_thumbnail(key))
    if label == "WOODEN BOWL":
        d.multiline_text((x + 26, y), "WOODEN\nBOWL", font=font(20), fill=INK, anchor="mm", align="center", spacing=0)
    else:
        draw_text(d, (x + 26, y), label, font=font(22), fill=INK, anchor="mm")


def spatial_graph(canvas, d, stage, compact=False, task_added=False):
    if stage < 0:
        return
    positions = {
        "banana": (1300, 660),
        "orange": (1700, 660),
        "near": (1300, 780),
        "far": (1700, 780),
        "robot": (1210, 890),
        "island": (1770, 890),
        "floor": (1770, 975),
    }
    # Floor and island top are references into the kitchen USD, not separate placed assets.
    d.rounded_rectangle((1440, 845, 1925, 1005), radius=12, fill="#f8f8f8", outline=BORDER, width=3)
    canvas.paste(asset_thumbnail("kitchen"), (1452, 850), asset_thumbnail("kitchen"))
    text(d, (1506, 854), "KITCHEN", 21, INK)
    labels = []

    def edge(a, b, label, label_xy, start_offset=(0, 0), end_offset=(0, 0)):
        start = tuple(x + y for x, y in zip(positions[a], start_offset))
        end = tuple(x + y for x, y in zip(positions[b], end_offset))
        arrow(d, start, end, "#a6b398")
        labels.append((label, label_xy))

    if stage >= 1:
        edge("near", "island", "on", (1505, 831), (65, 30), (-65, -30))
    if stage >= 2:
        edge("robot", "floor", "on", (1430, 947), (0, 30), (-115, 0))
        edge("robot", "island", "align with", (1490, 905), (115, 0), (-115, 0))
    if stage >= 3:
        edge("far", "island", "on", (1735, 835), (0, 30), (0, -30))
        edge("banana", "island", "on", (1490, 753), (65, 30), (-65, -30))
        edge("orange", "island", "on", (1830, 711), (115, 0), (115, 0))
    for label, (x, y) in labels:
        half_width = font(21).getlength(label) / 2 + 7
        fill = "#f8f8f8" if x > 1440 and y > 845 else "#eeeeee"
        d.rounded_rectangle((x - half_width, y - 14, x + half_width, y + 14), radius=4, fill=fill)
        draw_text(d, (x, y), label, font=font(21), fill="#58624f", anchor="mm")
    for key, label, color, added in [
        ("floor", "FLOOR", BLUE, 0),
        ("island", "ISLAND TOP", BLUE, 0),
        ("near", "WOODEN BOWL", BLUE if task_added else TEAL, 1),
        ("robot", "ROBOT", TEAL, 2),
        ("far", "RED BOWL", BLUE if task_added else TEAL, 3),
        ("banana", "BANANA", CORAL if task_added else TEAL, 3),
        ("orange", "ORANGE", GOLD if task_added else TEAL, 3),
    ]:
        if stage >= added:
            node(canvas, d, positions[key], key, label, color, reference=key in ("floor", "island"))


def components(canvas, d, stage, task_mode, task_added, policy_mode, t):
    section(d, (1050, 310, 1950, 525), "TASK", GOLD)
    section(d, SCENE_BOUNDS, "SCENE", BLUE, True)
    color = TEAL if t >= 51.2 else GOLD
    section(d, (1050, 1030, 1950, 1190), "POLICY", color)
    spatial_graph(canvas, d, stage, compact=True, task_added=task_added)
    if task_added:
        d.rounded_rectangle((1100, 366, 1900, 511), radius=18, fill="#f1f6e9", outline=GOLD, width=4)
        draw_text(d, (1500, 389), "COMPOSITE TASK", font=font(23), fill=GOLD, anchor="mm")
        for index, (label, x, color) in enumerate([("BANANA", 1120, CORAL), ("ORANGE", 1520, GOLD)]):
            if t < 41 + index * 0.6:
                continue
            d.rounded_rectangle((x, 412, x + 360, 495), radius=12, fill="#ffffff", outline=color, width=3)
            draw_text(d, (x + 180, 435), "PICK & PLACE", font=font(20), fill=color, anchor="mm")
            destination = "WOODEN BOWL" if index == 0 else "RED BOWL"
            draw_text(d, (x + 180, 470), label + " → " + destination, font=font(22), fill=INK, anchor="mm")
            fruit_x = 1300 if index == 0 else 1700
            arrow(d, (fruit_x, 511), (fruit_x, 628), color)
            # Each task reaches its bowl along the outside of its own column.
            route_x = 1140 if index == 0 else 1860
            bowl_edge_x = 1185 if index == 0 else 1815
            d.line([(route_x, 495), (route_x, 780)], fill=color, width=3)
            arrow(d, (route_x, 780), (bowl_edge_x, 780), color)
    if policy_mode and t >= 48.1:
        color = TEAL if t >= 51.2 else GOLD
        d.rounded_rectangle((1190, 1050, 1630, 1170), radius=16, fill="#ffffff", outline=TEAL, width=4)
        icon = physical_intelligence_icon()
        canvas.paste(icon, (1220, 1080), icon)
        draw_text(d, (1470, 1090), "π0.5", font=font(42), fill=INK, anchor="mm")
        draw_text(d, (1470, 1140), "Closed-loop policy", font=font(21), fill=MUTED, anchor="mm")
        text(d, (1665, 1084), "EVALUATING" if t >= 51.2 else "READY", 25, color)
    elif policy_mode:
        text(d, (1130, 1090), "Adding a policy…", 30, MUTED)


def state(t):
    stage = max((i for i, start in enumerate(STARTS) if t >= start), default=0)
    local = t - STARTS[stage]
    task_mode = t >= 39
    outro = 44 <= t < 47
    graph_stage = stage if local >= 1.8 else stage - 1
    render_stage = stage if local >= 3.5 else stage - 1
    layout, index, flash = 0, 33, 0.0
    if task_mode:
        graph_stage = render_stage = 3
        layout = SAMPLE_COUNT - 1
        if outro:
            interval = 3 / SAMPLE_COUNT
            layout = min(SAMPLE_COUNT - 1, int((t - 44) / interval))
            age = t - (44 + layout * interval)
            index = min(33, int(max(0, age) * CAPTURE_SPEED / period(3)))
            flash = 1 - ease(age / 0.32) if age < 0.32 else 0
    elif render_stage >= 0:
        if render_stage < stage:
            layout = 0 if render_stage == 0 else SAMPLE_COUNT - 1
        else:
            elapsed = max(0, local - 3.5)
            layout = min(SAMPLE_COUNT - 1, int(elapsed / SAMPLE_INTERVAL)) if stage else 0
            age = elapsed - layout * SAMPLE_INTERVAL
            index = min(33, int(age * CAPTURE_SPEED / period(stage))) if stage else 33
            flash = 1 - ease(age / 0.32) if stage and age < 0.32 else 0
    return stage, local, graph_stage, render_stage, layout, index, flash


def flow_arrow(d, center, label, color, t, active):
    """Draw a curved panel connector with a moving pulse while data flows."""
    points = []
    for i in range(81):
        p = i / 80
        points.append((center - 250 + 500 * p, 130 - 70 * 4 * p * (1 - p)))
    d.line(points, fill="#e9eee4", width=4, joint="curve")
    d.line(points, fill=NVIDIA_GREEN if active else "#b9c3ae", width=3, joint="curve")
    arrow(d, points[-4], points[-1], NVIDIA_GREEN if active else "#b9c3ae")
    d.ellipse((points[0][0] - 8, 122, points[0][0] + 8, 138), fill=color)
    half = 160 if len(label) > 10 else 92
    d.rounded_rectangle((center - half, 8, center + half, 49), radius=16, fill=BG, outline=None, width=1)
    draw_text(d, (center, 28), label, font=font(24), fill=color if active else MUTED, anchor="mm")
    if active:
        progress = (t % 1.1) / 1.1
        stop = max(1, min(80, int(progress * 80)))
        d.line(points[max(0, stop - 12) : stop + 1], fill=NVIDIA_GREEN, width=5, joint="curve")
        x, y = points[stop]
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=NVIDIA_GREEN)


def board(t):
    t *= AUTHORING_SPEED
    intro = t < 2
    task_mode = t >= 39
    outro = 44 <= t < 47
    policy_mode = t >= 47
    stage, local, graph_stage, render_stage, layout, index, flash = state(t)
    canvas = Image.new("RGB", (3000, 1320), BG)
    d = ImageDraw.Draw(canvas)
    focus = 0 if local < 1.8 else 1 if local < 3.5 else 2
    for i, (title, label, color) in enumerate(
        zip(
            ["AGENT", "Isaac Lab - Arena", "Isaac Lab"],
            ["Benchmark author", "Evaluation description framework", "GPU-accelerated simulator"],
            [TEAL, BLUE, GOLD],
        )
    ):
        x = 40 + i * 990
        active = not intro and not outro and ((task_mode and i == 1) or (not task_mode and i == focus))
        d.rounded_rectangle(
            (x, 130, x + 940, 1210),
            radius=16,
            fill=PANEL,
            outline=NVIDIA_GREEN if active else BORDER,
            width=5 if active else 4,
        )
        icon = codex_icon() if i == 0 else isaac_lab_icon()
        canvas.paste(icon, (x + 45, 157), icon)
        if i == 2:
            omni_icon = omniverse_icon()
            canvas.paste(omni_icon, (x + 123, 157), omni_icon)
        text(d, (x + (210 if i == 2 else 130), 175), title, 33, INK)
        text(d, (x + 45, 228), label, 27, MUTED)
        d.line((x + 45, 290, x + 895, 290), fill="#b2bea7", width=2)
    authoring = (not intro and not task_mode and local < 3.5) or 39 <= t < 41.5 or 47 <= t < 51.2
    building = (not intro and not task_mode and local >= 3.5) or outro or t >= 51.2
    flow_arrow(d, 1005, "AUTHOR", TEAL, t, authoring)
    flow_arrow(d, 1995, "RUN", GOLD, t, building)
    visible = list(range(stage + 1)) if not intro else []
    if task_mode:
        visible = list(range(5))
    if policy_mode:
        visible = list(range(7 if t >= 49.7 else 6))
    # macOS window chrome surrounds the seven typed commands without scrolling.
    d.rounded_rectangle((78, 326, 948, 1181), radius=18, fill="#e2e4df")
    d.rounded_rectangle((75, 320, 945, 1175), radius=18, fill="#ffffff", outline="#b8b8b8", width=3)
    d.rounded_rectangle((77, 322, 943, 395), radius=16, fill="#e8e8e8")
    d.rectangle((77, 370, 943, 395), fill="#e8e8e8")
    d.line((77, 395, 943, 395), fill="#c8c8c8", width=2)
    for x, fill, outline in ((110, "#ff5f57", "#e0443e"), (148, "#febc2e", "#dea123"), (186, "#28c840", "#1aab29")):
        d.ellipse((x - 11, 347, x + 11, 369), fill=fill, outline=outline, width=1)
    draw_text(d, (510, 358), "Codex", font=font(27), fill="#505050", anchor="mm")
    prompt = PROMPT
    prompt_lines = textwrap.wrap(prompt, width=43)
    prompt_bottom = 438 + 40 * len(prompt_lines)
    d.rectangle((92, 421, 928, prompt_bottom), fill="#eeeeee")
    d.text((105, 433), "›", font=terminal_font(), fill=INK)
    d.text((145, 433), "\n".join(prompt_lines), font=terminal_font(), fill=INK, spacing=12)
    for row, j in enumerate(visible):
        y = prompt_bottom + 37 + j * 52
        if j >= 4:
            start = {4: 39, 5: 47, 6: 49.7}[j]
            fraction = min(1, max(0, (t - start) / 0.8))
        else:
            fraction = min(1, max(0, local / 1.2)) if j == stage and local < 1.8 else 1
        phrase = COMMANDS[j][: int(len(COMMANDS[j]) * fraction)]
        d.text((105, y), "•", font=terminal_font(), fill=INK)
        d.text((145, y), phrase, font=terminal_font(), fill=INK)
    components(canvas, d, graph_stage, task_mode, t >= 40, policy_mode, t)
    if flash:
        region = SCENE_BOUNDS
        crop = canvas.crop(region)
        glow = Image.new("RGB", crop.size, "#d9edb8")
        mask = Image.new("L", crop.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, crop.width - 1, crop.height - 1), radius=20, fill=255)
        canvas.paste(Image.blend(crop, glow, 0.2 * flash), (region[0], region[1]), mask)
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle(region, radius=20, outline="#76b900", width=max(2, round(8 * flash)))
    if render_stage >= 0:
        canvas.paste(frame(render_stage, layout, index), (2070, 325))
        d = ImageDraw.Draw(canvas)
        if render_stage != 0:
            text(d, (2070, 975), f"SAMPLE {layout+1:02d} / {SAMPLE_COUNT:02d}", 30, GOLD, True)
        if render_stage:
            if task_mode and not outro:
                text(d, (2460, 982), "TASK DEFINED", 24, MUTED)
            for k in range(SAMPLE_COUNT):
                x = 2070 + k * 170
                canvas.paste(frame(render_stage, k, 33, (156, 117)), (x, 1035))
                text(d, (x + 60, 1165), f"0{k+1}", 22, GOLD if k == layout else MUTED, True)
                if k == layout:
                    d.rectangle((x - 3, 1032, x + 159, 1155), outline=GOLD, width=4)
    # Keep panel headings and the title stationary throughout authoring.
    cx, cy, cw = 1500, 660, 3040
    ch = cw * 860 / 1920
    left, top, right, bottom = int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2)
    viewport = Image.new("RGB", (right - left, bottom - top), BG)
    viewport.paste(canvas, (-left, -top))
    out = Image.new("RGB", (1920, 1080), BG)
    out.paste(viewport.resize((1920, 860), Image.Resampling.LANCZOS), (0, 145))
    d = ImageDraw.Draw(out)
    page_title(d)
    return out


@lru_cache(90)
def rollout_frame(folder, index):
    return (
        Image.open(ROOT / folder / f"frame_{index:05d}.jpg")
        .convert("RGB")
        .resize((1000, 750), Image.Resampling.LANCZOS)
    )


def rollout_board(local, entry, number, rollouts, rate=1.0):
    clip_time = max(0, local - 0.5) * rate
    index = min(entry["frame_count"] - 1, int(clip_time * FPS))
    finished = clip_time >= entry["frame_count"] / FPS
    out = Image.new("RGB", (1920, 1080), BG)
    d = ImageDraw.Draw(out)
    page_title(d)
    out.paste(rollout_frame(entry["frames"], index), (815, 160))
    d = ImageDraw.Draw(out)
    # The task and policy stay fixed while the three rollout clips change on the right.
    d.rounded_rectangle((45, 260, 710, 515), radius=16, fill="#eeeeee", outline=BORDER, width=4)
    text(d, (80, 285), "COMPOSITE TASK", 25, GOLD)
    text(d, (80, 365), "BANANA → WOODEN BOWL", 29)
    text(d, (80, 435), "ORANGE → RED BOWL", 29)
    d.rounded_rectangle((45, 550, 710, 690), radius=16, fill="#eeeeee", outline=BORDER, width=4)
    text(d, (80, 570), "POLICY", 25, GOLD)
    icon = physical_intelligence_icon().resize((36, 35), Image.Resampling.LANCZOS)
    out.paste(icon, (80, 625), icon)
    text(d, (135, 615), "π0.5", 36)
    width = (1810 - 60) / 3
    for slot, result in enumerate(rollouts, 1):
        complete = slot < number or (slot == number and finished)
        active = slot == number and not finished
        fill = (NVIDIA_GREEN if result["success"] else "#c43d36") if complete else "#eeeeee"
        border = fill if complete else NVIDIA_GREEN if active else BORDER
        label = f"Rollout {slot}"
        if complete:
            label += " · Success" if result["success"] else " · Failed"
        elif active:
            label += " · Running"
        left = 55 + (slot - 1) * (width + 30)
        d.rounded_rectangle(
            (left, 928, left + width, 978), radius=10, fill=fill, outline=border, width=4 if active else 2
        )
        draw_text(
            d,
            (left + width / 2, 953),
            label,
            font=font(25),
            fill="#ffffff" if complete and not result["success"] else INK,
            anchor="mm",
        )
    return out


# Authoring coordinates match the original 3000 x 1320 board.
REFERENCE_BOUNDS = {
    "agent": (40, 130, 980, 1210),
    "arena": (1030, 130, 1970, 1210),
    "simulator": (2020, 130, 2960, 1210),
    "prompt": (92, 421, 928, 438 + 40 * len(textwrap.wrap(PROMPT, width=43))),
    "task": (1050, 310, 1950, 525),
    "scene": SCENE_BOUNDS,
    "policy": (1050, 1030, 1950, 1190),
    "composite": (1100, 366, 1900, 511),
    "pick_banana": (1120, 412, 1480, 495),
    "pick_orange": (1520, 412, 1880, 495),
    "kitchen": (1440, 845, 1925, 1005),
    "near": (1190, 754, 1410, 806),
    "far": (1590, 754, 1810, 806),
    "banana": (1190, 634, 1410, 686),
    "orange": (1590, 634, 1810, 686),
    "robot": (1100, 864, 1320, 916),
}
ROLLOUT_REFERENCE_BOUNDS = {
    "rollout_video": (815, 160, 1815, 910),
    "rollout_task": (45, 260, 710, 515),
    "rollout_policy": (45, 550, 710, 690),
    "rollout_banana": (65, 350, 690, 410),
    "rollout_orange": (65, 420, 690, 480),
    "results": (55, 928, 1866, 978),
}


def reference_bounds(target):
    if target in ROLLOUT_REFERENCE_BOUNDS:
        return ROLLOUT_REFERENCE_BOUNDS[target]
    x1, y1, x2, y2 = REFERENCE_BOUNDS[target]
    return tuple(
        round(v)
        for v in (
            (x1 + 20) * 1920 / 3040,
            145 + (y1 + 20) * 860 / 1360,
            (x2 + 20) * 1920 / 3040,
            145 + (y2 + 20) * 860 / 1360,
        )
    )


def reference_overlay(out, t, data):
    for event in data["references"]:
        if not event["start"] <= t < event["end"]:
            continue
        phase = (t - event["start"]) / (event["end"] - event["start"])
        strength = math.sin(math.pi * phase) ** 0.7
        for target in event["targets"]:
            bounds = reference_bounds(target)
            crop = out.crop(bounds)
            emphasis = crop.copy()
            d = ImageDraw.Draw(emphasis)
            # A soft green wash and a bold edge draw attention without hiding content.
            if target != "results":
                wash = Image.new("RGB", crop.size, "#d9edb8")
                emphasis = Image.blend(emphasis, wash, 0.22)
                d = ImageDraw.Draw(emphasis)
            d.rounded_rectangle((1, 1, crop.width - 2, crop.height - 2), radius=8, outline=NVIDIA_GREEN, width=5)
            out.paste(Image.blend(crop, emphasis, strength), bounds[:2])
    return out


def caption_overlay(out, t, data):
    d = ImageDraw.Draw(out)
    d.line((55, 994, 1865, 994), fill="#dddddd", width=1)
    for cue in data["cues"]:
        if cue["start"] <= t < cue["end"]:
            assert font(32, bold=False).getlength(cue["text"]) < 1810, cue["text"]
            draw_text(d, (960, 1033), cue["text"], font=font(32, bold=False), fill=INK, anchor="mm")
            break
    d.rectangle((0, 1076, int(1920 * t / data["duration"]), 1079), fill=NVIDIA_GREEN)
    return out


def movie_frame(t, data, rollouts):
    if t < data["authoring_duration"]:
        out = board(visual_time(t, data["cues"]) / AUTHORING_SPEED)
    else:
        offset = t - data["authoring_duration"]
        for number, entry in enumerate(rollouts, 1):
            length = rollout_length(entry, data)
            if offset < length or number == len(rollouts):
                out = rollout_board(offset, entry, number, rollouts, rate=rollout_rate(data))
                break
            offset -= length
    return caption_overlay(reference_overlay(out, t, data), t, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stills", action="store_true")
    args = parser.parse_args()
    data = timeline()
    export_timing(data)
    rollouts = json.loads((ROOT / "rollouts.json").read_text())
    assert len(rollouts) == 3 and all(isinstance(e["success"], bool) for e in rollouts)
    if args.stills:
        folder = ROOT / "narration_review"
        folder.mkdir(exist_ok=True)
        for i, cue in enumerate(data["cues"]):
            movie_frame((cue["start"] + cue["end"]) / 2, data, rollouts).save(folder / f"cue_{i+1:02d}.jpg", quality=94)
        movie_frame(data["authoring_duration"] - 0.4, data, rollouts).save(ROOT / "preview_52.jpg", quality=94)
        movie_frame(data["cues"][-1]["start"] + 1.5, data, rollouts).save(
            ROOT / "preview_evaluation_layout.jpg", quality=94
        )
        return
    assert json.loads((ROOT / "evaluation_summary.json").read_text())["finalized"]
    duration = data["duration"]
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1920x1080",
        "-r",
        "30",
        "-i",
        "-",
    ]
    if data["audio_included"]:
        command += [
            "-i",
            str(ROOT / data["audio_file"]),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
    else:
        command += ["-an"]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(ROOT / "kitchenbench_split_bowls_authoring.mp4"),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i in range(round(duration * FPS)):
            t = i / FPS
            out = movie_frame(t, data, rollouts)
            process.stdin.write(out.tobytes())
            if i % (FPS * 5) == 0:
                print(f"Encoded {t:.0f}/{duration:.1f} seconds", flush=True)
    finally:
        process.stdin.close()
    assert process.wait() == 0


if __name__ == "__main__":
    main()
