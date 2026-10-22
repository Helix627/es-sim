"""
Find an object in an image (or live webcam frame), draw a box around it,
estimate its 3D coordinates, and calculate the 5-DOF robotic arm joint angles
and movement sequence.

OWLv2 finds the precise boxes; LLM (or visual attribute ranker when offline)
decides which candidate box matches the user's query.

Usage:
    python owl_locate.py <image_path> "where is the white mixed foam board"
    python owl_locate.py --webcam "Pick up the red bottle and place it in the blue tray"
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Auto-attach parent workspace and .venv site-packages
_repo_dir = Path(__file__).resolve().parent
_root_dir = _repo_dir.parent
_venv_site = _root_dir / ".venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from vision_test import FRIDAY_ENV, load_env
from nlp_parser import CommandParser, _clean_phrase
from calibration import WorkspaceCalibrator
from kinematics import KinematicsSolver
from motion_planner import MotionPlanner

MODEL_ID = "google/owlv2-base-patch16-ensemble"
GENERIC_PHRASES = ["object", "box", "block", "electronic component"]
MAX_BOX_AREA = 0.8
NMS_IOU = 0.5
MIN_SCORE = 0.05
SPECIFIC_SLOTS = 3
TILE_SIZE = 256

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#9a6324", "#808000", "#000075"
]

PHRASE_PROMPT = """Turn the user's question into a short noun phrase naming the
object to find in an image, keeping colour and other visual details.
Reply with the phrase only, lowercase, no punctuation.
Examples:
"where is the white color breadboard" -> white breadboard
"can you find my red multimeter on the desk" -> red multimeter"""

PICK_PROMPT = """You get two images.
The first is the full photo with numbered coloured boxes drawn on it. Each
number is written in a filled square at a corner of its box.
The second is a sheet of enlarged crops, one per box, each labelled with the
same number. Use the crops to see small details like colour and markings,
and the full photo to see where things are and how they relate.

The user asks where an object is. Decide which numbered box contains that
object. Pick the box that fits the object best, not a box that only contains part of
it or a much larger area around it.
If none of the boxes contain the object, answer 0."""

PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "box": {"type": "integer"},
    },
    "required": ["reason", "box"],
    "additionalProperties": False,
}


def llm_client():
    env_candidates = [FRIDAY_ENV, _repo_dir / ".env", _root_dir / ".env"]
    for ep in env_candidates:
        if ep.exists():
            env = load_env(ep)
            if "AZURE_OPENAI_ENDPOINT" in env and "AZURE_OPENAI_API_KEY" in env:
                from openai import OpenAI
                client = OpenAI(
                    base_url=env["AZURE_OPENAI_ENDPOINT"],
                    api_key=env["AZURE_OPENAI_API_KEY"],
                )
                return client, env["AZURE_OPENAI_DEPLOYMENT_MAIN"]
    return None, None


def question_to_phrase(client, model: str, question: str) -> str:
    if client is None:
        return _clean_phrase(question)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": PHRASE_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": question}]},
        ],
        temperature=0,
    )
    return response.output_text.strip().strip('"').lower()


def to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def pick_box(client, model: str, numbered: Image.Image, tiles: Image.Image, question: str, candidates: list[dict], raw_image: Image.Image) -> dict:
    if client is not None:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": PICK_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {"type": "input_image", "image_url": to_data_url(numbered), "detail": "high"},
                        {"type": "input_image", "image_url": to_data_url(tiles), "detail": "high"},
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "pick_box",
                    "schema": PICK_SCHEMA,
                    "strict": True,
                }
            },
            temperature=0,
        )
        return json.loads(response.output_text)

    # Local visual verification over OWLv2 candidates when Azure OpenAI .env is not present
    import cv2
    import numpy as np
    from vision import VisionDetector

    cv_img = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(cv2.GaussianBlur(cv_img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    vd = VisionDetector(use_simulation=True)

    best_idx = 1
    best_combined = -1.0
    best_reason = ""

    for idx, c in enumerate(candidates, start=1):
        x0, y0, x1, y1 = [int(round(v)) for v in c["box"]]
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        prop = {
            "bbox": (x0, y0, w, h),
            "box_xyxy": (x0, y0, x1, y1),
            "center": (x0 + w // 2, y0 + h // 2),
            "area": float(w * h),
            "features": vd._extract_region_features(cv_img, hsv_img, (x0, y0, w, h)),
        }
        vis_score, _, reason = vd._score_proposal_for_query(prop, question, cv_img.shape)
        # Combine OWLv2 confidence + visual feature verification
        owl_bonus = 0.25 if c["phrase"] not in GENERIC_PHRASES else 0.0
        combined = vis_score + c["score"] * 0.5 + owl_bonus
        if combined > best_combined:
            best_combined = combined
            best_idx = idx
            best_reason = f"The { _clean_phrase(question) } is clearly visible in box {idx} ({reason})"

    return {"box": best_idx, "reason": best_reason}


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))


def find_candidates(image: Image.Image, phrases: list[str], limit: int) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()

    inputs = processor(text=[phrases], images=image, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"OWLv2 inference: {(time.perf_counter() - start) * 1000:.0f} ms on {device}")

    width, height = image.size
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=0.0, target_sizes=[(height, width)]
    )[0]

    scores = results["scores"].cpu()
    labels = results["labels"].cpu()
    boxes = results["boxes"].cpu().clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = (areas < MAX_BOX_AREA * width * height) & (scores >= MIN_SCORE)
    scores, labels, boxes = scores[keep], labels[keep], boxes[keep]

    order = scores.argsort(descending=True).tolist()
    specific = [i for i in order if labels[i] == 0][:SPECIFIC_SLOTS]
    rest = [i for i in order if i not in specific]

    candidates = []
    for i in specific + rest:
        if any(iou(boxes[i], c["tensor"]) > NMS_IOU for c in candidates):
            continue
        candidates.append({
            "tensor": boxes[i],
            "box": tuple(boxes[i].tolist()),
            "score": scores[i].item(),
            "phrase": phrases[labels[i].item()],
        })
        if len(candidates) == limit:
            break

    for c in candidates:
        del c["tensor"]
    return candidates


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arialbd.ttf", max(16, size))
    except OSError:
        return ImageFont.load_default()


def draw_label(draw, xy, text, font, fill):
    x, y = xy
    text_box = draw.textbbox((0, 0), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    draw.rectangle([x, y, x + text_w + 10, y + text_h + 10], fill=fill)
    draw.text((x + 5 - text_box[0], y + 5 - text_box[1]), text, fill="white", font=font)


def overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def draw_numbered(image: Image.Image, candidates: list[dict]) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = load_font(min(image.size) // 18)
    line_width = max(2, min(image.size) // 200)

    for n, c in enumerate(candidates, start=1):
        color = COLORS[(n - 1) % len(COLORS)]
        draw.rectangle(c["box"], outline=color, width=line_width)

    placed = []
    for n, c in enumerate(candidates, start=1):
        color = COLORS[(n - 1) % len(COLORS)]
        text_box = draw.textbbox((0, 0), str(n), font=font)
        w = text_box[2] - text_box[0] + 10
        h = text_box[3] - text_box[1] + 10
        x0, y0, x1, y1 = c["box"]
        spots = [
            (x0, y0), (x1 - w, y0), (x0, y1 - h), (x1 - w, y1 - h),
            (x0, y0 - h), (x1 - w, y0 - h), (x0, y1), (x1 - w, y1)
        ]
        spots = [(min(max(0, x), image.width - w), min(max(0, y), image.height - h)) for x, y in spots]
        spot = next(
            (s for s in spots if not any(overlaps((*s, s[0] + w, s[1] + h), p) for p in placed)),
            spots[0],
        )
        placed.append((*spot, spot[0] + w, spot[1] + h))
        draw_label(draw, spot, str(n), font, color)

    return image


def draw_tiles(image: Image.Image, candidates: list[dict]) -> Image.Image:
    columns = min(4, len(candidates))
    rows = (len(candidates) + columns - 1) // columns
    gap = 8
    sheet = Image.new(
        "RGB",
        (columns * (TILE_SIZE + gap) + gap, rows * (TILE_SIZE + gap) + gap),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = load_font(TILE_SIZE // 7)

    for n, c in enumerate(candidates, start=1):
        x0, y0, x1, y1 = c["box"]
        pad_x, pad_y = (x1 - x0) * 0.15, (y1 - y0) * 0.15
        crop = image.crop((
            max(0, x0 - pad_x), max(0, y0 - pad_y),
            min(image.width, x1 + pad_x), min(image.height, y1 + pad_y),
        ))
        scale = TILE_SIZE / max(crop.size)
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))

        row, col = divmod(n - 1, columns)
        tile_x = gap + col * (TILE_SIZE + gap)
        tile_y = gap + row * (TILE_SIZE + gap)
        sheet.paste(crop, (tile_x + (TILE_SIZE - crop.width) // 2, tile_y + (TILE_SIZE - crop.height) // 2))

        color = COLORS[(n - 1) % len(COLORS)]
        draw.rectangle([tile_x, tile_y, tile_x + TILE_SIZE, tile_y + TILE_SIZE], outline=color, width=4)
        draw_label(draw, (tile_x, tile_y), str(n), font, color)

    return sheet


def draw_answer(image: Image.Image, candidate: dict, label: str, coords_3d: tuple[float, float, float]) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = load_font(min(image.size) // 30)
    line_width = max(3, min(image.size) // 150)

    x0, y0, x1, y1 = candidate["box"]
    draw.rectangle([x0, y0, x1, y1], outline="red", width=line_width)

    rx, ry, rz = coords_3d
    full_label = f"{label} | 3D: ({rx:+.1f}, {ry:+.1f}, {rz:+.1f}) cm"
    text_box = draw.textbbox((0, 0), full_label, font=font)
    text_w = text_box[2] - text_box[0] + 10
    text_h = text_box[3] - text_box[1] + 10
    label_x = min(max(0, x0), max(0, image.width - text_w))
    label_y = y0 - text_h if y0 - text_h > 0 else y0
    draw_label(draw, (label_x, label_y), full_label, font, "red")

    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Find an object with OWLv2 + LLM and compute 5-DOF Arm Kinematics.")
    parser.add_argument("image", nargs="?", default="", help="Path to image (or omit with --webcam)")
    parser.add_argument("question", nargs="?", default="", help="Natural language query")
    parser.add_argument("--webcam", action="store_true", help="Capture frame from laptop webcam")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    if args.webcam or not args.image:
        import cv2
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            sys.exit("Could not capture frame from webcam.")
        image_path = _repo_dir / "webcam_capture.jpg"
        cv2.imwrite(str(image_path), frame)
        question = args.image if (args.image and not args.question) else (args.question or "red object")
    else:
        image_path = Path(args.image)
        if not image_path.exists() and (_repo_dir / args.image).exists():
            image_path = _repo_dir / args.image
        question = args.question or input("What should I find? ")

    image = Image.open(image_path).convert("RGB")
    client, model = llm_client()

    cmd_parser = CommandParser()
    task = cmd_parser.parse(question)
    phrase = question_to_phrase(client, model, task.pick_phrase)
    print(f"Searching for: {phrase!r} + {GENERIC_PHRASES}")

    candidates = find_candidates(image, [phrase] + GENERIC_PHRASES, args.candidates)
    if not candidates:
        print("OWLv2 found nothing.")
        return

    for n, c in enumerate(candidates, start=1):
        box = ", ".join(f"{v:.0f}" for v in c["box"])
        print(f"  [{n}] score {c['score']:.2f} ({c['phrase']})  box ({box})")

    numbered = draw_numbered(image, candidates)
    candidates_path = image_path.with_name(f"{image_path.stem}_candidates.jpg")
    numbered.save(candidates_path, quality=95)

    tiles = draw_tiles(image, candidates)
    tiles.save(image_path.with_name(f"{image_path.stem}_tiles.jpg"), quality=95)

    start = time.perf_counter()
    pick = pick_box(client, model, numbered, tiles, task.pick_phrase, candidates, image)
    print(f"LLM pick: {(time.perf_counter() - start) * 1000:.0f} ms")
    print(f"LLM chose box {pick['box']}: {pick['reason']}")

    if not 1 <= pick["box"] <= len(candidates):
        print(f"None of the boxes match. Candidates: {candidates_path}")
        return

    chosen = candidates[pick["box"] - 1]
    x0, y0, x1, y1 = chosen["box"]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    # Compute 3D Coordinates & Robotic Arm Kinematics
    calibrator = WorkspaceCalibrator(frame_w=image.width, frame_h=image.height)
    coords_3d = calibrator.pixel_to_robot(cx, cy, z=0.0)
    solver = KinematicsSolver()
    planner = MotionPlanner(solver=solver)

    grasp_ik = solver.inverse_kinematics(coords_3d[0], coords_3d[1], 2.5, gripper_closed=False)
    print("\n" + "=" * 96)
    print(f" 3D COORDINATES (Robot Base Frame) : X={coords_3d[0]:+.2f} cm, Y={coords_3d[1]:+.2f} cm, Z={coords_3d[2]:+.2f} cm")
    print(f" CALCULATED JOINT & SERVO ANGLES   : {grasp_ik}")
    print("=" * 96)

    place_xyz = (round(min(25.0, max(15.0, coords_3d[0] + 2.0)), 2), round(-coords_3d[1] if abs(coords_3d[1]) > 4.0 else -10.0, 2), 0.0)
    plan = planner.plan_pick_and_place(coords_3d, place_xyz, task_label=question)
    if plan.is_valid:
        plan.print_plan_table()

    out_path = image_path.with_name(f"{image_path.stem}_owl.jpg")
    draw_answer(image, chosen, phrase, coords_3d).save(out_path, quality=95)
    print(f"Saved: {out_path}")
    print(f"Candidates: {candidates_path}")

    if sys.platform == "win32" and not args.no_open:
        os.startfile(out_path)


if __name__ == "__main__":
    main()
