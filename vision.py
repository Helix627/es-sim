"""
Computer Vision & Open-Vocabulary AI Object Grounding Engine.
Supports:
  1. OWLv2 Open-Vocabulary Vision Transformer (`google/owlv2-base-patch16-ensemble`)
     for arbitrary real-world object detection from natural language queries.
  2. Multi-Cue Tabletop Object Segmentation & Semantic Attribute Matching.
  3. Dynamic, Variable-Object Interactive Tabletop Simulation (supports arbitrary
     user-named objects, randomized positions, and mouse drag-and-drop).
  4. Live Laptop Webcam capture.
"""
import os
import sys
import random
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_venv_site = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import config
from calibration import WorkspaceCalibrator


@dataclass
class DetectedObject:
    color: str
    center_px: Tuple[int, int]
    bbox: Tuple[int, int, int, int]          # (x, y, w, h)
    area: float
    robot_coords: Tuple[float, float, float]  # (X, Y, Z) in cm
    label: str = "object"
    score: float = 0.90
    reason: str = ""

    @property
    def box_xyxy(self) -> Tuple[int, int, int, int]:
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)

    def __str__(self) -> str:
        rx, ry, rz = self.robot_coords
        return (
            f"[{self.label.upper()}] (score {self.score:.2f}) @ Pixel {self.center_px} "
            f"Box {self.box_xyxy} -> 3D Robot Frame: X={rx:+.1f}cm, Y={ry:+.1f}cm, Z={rz:+.1f}cm"
        )


@dataclass
class SimObject:
    key: str
    label: str
    pos: Tuple[int, int]
    size: Tuple[int, int]  # (half_w, half_h)
    color_name: str
    bgr: Tuple[int, int, int]
    shape_style: str = "box"  # "bottle", "foam", "breadboard", "tray", "phone", "cup", "box"
    is_receptacle: bool = False


COLOR_BGR_MAP = {
    "red": (25, 25, 220),
    "blue": (225, 85, 15),
    "green": (35, 185, 35),
    "yellow": (15, 215, 235),
    "orange": (15, 130, 245),
    "purple": (180, 40, 160),
    "pink": (180, 105, 255),
    "cyan": (220, 220, 20),
    "white": (238, 238, 240),
    "black": (42, 42, 45),
    "grey": (125, 125, 130),
    "brown": (65, 115, 165),
}


class VisionDetector:
    def __init__(
        self,
        calibrator: Optional[WorkspaceCalibrator] = None,
        use_simulation: bool = False,
        camera_index: int = config.CAMERA_INDEX,
        open_camera: bool = True,
    ):
        self.calibrator = calibrator or WorkspaceCalibrator()
        self.use_simulation = use_simulation
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.sim_tick = 0

        # OWLv2 model state (lazy-loaded on first real-image/webcam query if available)
        self._owl_processor = None
        self._owl_model = None
        self._owl_device = "cpu"
        self._owl_attempted = False

        # Dynamic simulated tabletop objects
        self.sim_objects: Dict[str, SimObject] = {}
        self.carried_sim_key: Optional[str] = None
        self.carried_z_cm: float = 0.0
        self.reset_sim_objects()

        if not self.use_simulation and open_camera:
            self._init_camera()

    def _try_load_owlv2(self) -> bool:
        """Loads OWLv2 (`google/owlv2-base-patch16-ensemble`) if cached locally."""
        if self._owl_model is not None:
            return True
        if self._owl_attempted:
            return False
        self._owl_attempted = True
        try:
            import torch
            from transformers import Owlv2Processor, Owlv2ForObjectDetection

            model_id = "google/owlv2-base-patch16-ensemble"
            self._owl_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._owl_processor = Owlv2Processor.from_pretrained(model_id, local_files_only=True)
            self._owl_model = Owlv2ForObjectDetection.from_pretrained(model_id, local_files_only=True).to(self._owl_device).eval()
            print(f"[Vision AI] OWLv2 Open-Vocabulary Transformer ready on {self._owl_device.upper()}.")
            return True
        except Exception:
            return False

    def reset_sim_objects(self):
        """Resets simulated tabletop objects to default assortment."""
        self.carried_sim_key = None
        self.carried_z_cm = 0.0
        self.sim_objects = {
            "blue_tray": SimObject(
                key="blue_tray",
                label="blue tray",
                pos=(510, 230),
                size=(58, 44),
                color_name="blue",
                bgr=(225, 85, 15),
                shape_style="tray",
                is_receptacle=True,
            ),
            "white_breadboard": SimObject(
                key="white_breadboard",
                label="white breadboard",
                pos=(475, 370),
                size=(54, 36),
                color_name="white",
                bgr=(238, 238, 242),
                shape_style="breadboard",
                is_receptacle=True,
            ),
            "red_bottle": SimObject(
                key="red_bottle",
                label="red bottle",
                pos=(135, 300),
                size=(22, 42),
                color_name="red",
                bgr=(25, 25, 220),
                shape_style="bottle",
            ),
            "foam_board": SimObject(
                key="foam_board",
                label="white mixed foam board",
                pos=(265, 315),
                size=(32, 38),
                color_name="white",
                bgr=(55, 55, 58),
                shape_style="foam",
            ),
            "green_block": SimObject(
                key="green_block",
                label="green block",
                pos=(255, 150),
                size=(26, 26),
                color_name="green",
                bgr=(35, 185, 35),
                shape_style="box",
            ),
            "yellow_cup": SimObject(
                key="yellow_cup",
                label="yellow cup",
                pos=(135, 155),
                size=(25, 28),
                color_name="yellow",
                bgr=(15, 215, 235),
                shape_style="cup",
            ),
            "black_phone": SimObject(
                key="black_phone",
                label="black phone",
                pos=(365, 175),
                size=(24, 40),
                color_name="black",
                bgr=(38, 38, 42),
                shape_style="phone",
            ),
        }

    def randomize_sim_objects(self):
        """Randomizes positions of all objects on the simulated tabletop."""
        self.carried_sim_key = None
        self.carried_z_cm = 0.0
        placed_centers: List[Tuple[int, int]] = []

        for key, obj in self.sim_objects.items():
            for _ in range(50):
                if obj.is_receptacle:
                    rx = random.randint(420, 540)
                    ry = random.randint(150, 380)
                else:
                    rx = random.randint(95, 375)
                    ry = random.randint(135, 380)
                if all(np.hypot(rx - px, ry - py) > 95 for px, py in placed_centers):
                    obj.pos = (rx, ry)
                    placed_centers.append((rx, ry))
                    break

    def find_sim_object_key_for_phrase(self, phrase: str) -> Optional[str]:
        p = phrase.lower().strip()
        if not p:
            return None
        # Exact or substring match
        for key, obj in self.sim_objects.items():
            if p == obj.label.lower() or p in obj.label.lower() or obj.label.lower() in p:
                return key
        # Word overlap match
        p_words = set(p.split())
        best_key = None
        best_overlap = 0
        for key, obj in self.sim_objects.items():
            o_words = set(obj.label.lower().split())
            overlap = len(p_words & o_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key
        if best_overlap >= 1:
            return best_key
        return None

    def ensure_sim_object_for_query(self, phrase: str, is_receptacle: bool = False) -> str:
        """
        If the user asks for ANY custom variable object in simulation mode (e.g.,
        'orange calculator', 'purple mouse', 'cardboard box', 'coffee mug'),
        dynamically spawns it onto the tabletop if not already present!
        """
        existing = self.find_sim_object_key_for_phrase(phrase)
        if existing is not None:
            return existing

        clean = phrase.lower().strip()
        color_found = "orange" if not is_receptacle else "blue"
        for cname in COLOR_BGR_MAP:
            if cname in clean or (cname == "grey" and "gray" in clean):
                color_found = cname
                break

        bgr = COLOR_BGR_MAP.get(color_found, (30, 140, 240))
        key = clean.replace(" ", "_")

        # Pick a free tabletop spot that never overlaps existing objects
        existing_pts = [o.pos for o in self.sim_objects.values()]
        candidate_slots = (
            [(365, 305), (200, 225), (310, 225), (195, 375), (365, 380)]
            if not is_receptacle
            else [(505, 305), (490, 145)]
        )
        pos = candidate_slots[0]
        found_slot = False
        for slot in candidate_slots:
            if all(np.hypot(slot[0] - px, slot[1] - py) > 78 for px, py in existing_pts):
                pos = slot
                found_slot = True
                break
        if not found_slot:
            for _ in range(60):
                cand_x = random.randint(105, 375) if not is_receptacle else random.randint(430, 540)
                cand_y = random.randint(135, 380)
                if all(np.hypot(cand_x - px, cand_y - py) > 80 for px, py in existing_pts):
                    pos = (cand_x, cand_y)
                    break

        style = "tray" if (is_receptacle or "tray" in clean or "box" in clean or "container" in clean) else "box"
        if "bottle" in clean or "can" in clean:
            style = "bottle"
        elif "phone" in clean or "calculator" in clean or "remote" in clean:
            style = "phone"
        elif "cup" in clean or "mug" in clean:
            style = "cup"
        elif "foam" in clean:
            style = "foam"
        elif "breadboard" in clean:
            style = "breadboard"

        size = (52, 40) if is_receptacle else (28, 32)
        self.sim_objects[key] = SimObject(
            key=key,
            label=clean,
            pos=pos,
            size=size,
            color_name=color_found,
            bgr=bgr,
            shape_style=style,
            is_receptacle=is_receptacle,
        )
        print(f"[Simulation] Spawned variable object '{clean}' on tabletop at pixel {pos}.")
        return key

    def get_sim_object_at_pixel(self, u: int, v: int) -> Optional[str]:
        """Returns the key of the simulated object under mouse cursor `(u, v)`."""
        # Prefer movable non-receptacle objects first
        order = [k for k, o in self.sim_objects.items() if not o.is_receptacle] + [
            k for k, o in self.sim_objects.items() if o.is_receptacle
        ]
        for key in reversed(order):
            obj = self.sim_objects[key]
            ox, oy = obj.pos
            hw, hh = obj.size
            if (ox - hw - 8) <= u <= (ox + hw + 8) and (oy - hh - 8) <= v <= (oy + hh + 8):
                return key
        return None

    def move_sim_object(self, key: str, u: int, v: int):
        if key in self.sim_objects:
            u_c = max(55, min(config.FRAME_WIDTH - 55, int(u)))
            v_c = max(75, min(config.FRAME_HEIGHT - 50, int(v)))
            self.sim_objects[key].pos = (u_c, v_c)

    def update_sim_carried_object(
        self,
        pick_phrase: str,
        gripper_closed: bool,
        arm_xyz: Tuple[float, float, float],
    ):
        """Updates the position of the grasped object in synthetic simulation mode."""
        key = self.find_sim_object_key_for_phrase(pick_phrase)
        if key is None:
            return

        if gripper_closed:
            self.carried_sim_key = key
            self.carried_z_cm = max(0.0, arm_xyz[2])
            u, v = self.calibrator.robot_to_pixel(arm_xyz[0], arm_xyz[1])
            u = max(45, min(config.FRAME_WIDTH - 45, u))
            v = max(65, min(config.FRAME_HEIGHT - 45, v))
            self.sim_objects[key].pos = (u, v)
        else:
            self.carried_sim_key = None
            self.carried_z_cm = 0.0

    def _init_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
            # Warm up auto-exposure on real webcam
            for _ in range(5):
                self.cap.read()
        else:
            print("[Vision] Webcam not available. Using synthetic simulation mode.")
            self.use_simulation = True

    def get_frame(self) -> np.ndarray:
        if self.use_simulation or self.cap is None:
            return self._generate_synthetic_frame()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[Vision] Failed to read frame from webcam. Switching to synthetic mode.")
            self.use_simulation = True
            return self._generate_synthetic_frame()

        return frame

    def _draw_single_sim_object(self, frame: np.ndarray, obj: SimObject, is_lifted: bool):
        ox, oy = obj.pos
        hw, hh = obj.size
        shadow = int(min(18.0, self.carried_z_cm * 1.2)) if is_lifted else 0

        if is_lifted and shadow > 0:
            cv2.rectangle(
                frame,
                (ox - hw + shadow, oy - hh + shadow),
                (ox + hw + shadow, oy + hh + shadow),
                (120, 135, 145),
                -1,
            )

        if obj.shape_style == "tray":
            cv2.rectangle(frame, (ox - hw, oy - hh), (ox + hw, oy + hh), obj.bgr, -1)
            inner_bgr = tuple(min(255, int(c * 1.25 + 25)) for c in obj.bgr)
            cv2.rectangle(frame, (ox - hw + 10, oy - hh + 10), (ox + hw - 10, oy + hh - 10), inner_bgr, -1)
            cv2.rectangle(frame, (ox - hw, oy - hh), (ox + hw, oy + hh), (40, 40, 40), 2)

        elif obj.shape_style == "breadboard":
            cv2.rectangle(frame, (ox - hw, oy - hh), (ox + hw, oy + hh), (238, 238, 242), -1)
            cv2.rectangle(frame, (ox - hw, oy - hh), (ox + hw, oy + hh), (180, 180, 185), 2)
            # Draw pinhole grid so texture detector & human eye recognize breadboard
            for px in range(ox - hw + 8, ox + hw - 6, 7):
                for py in range(oy - hh + 8, oy + hh - 6, 6):
                    cv2.circle(frame, (px, py), 1, (90, 90, 95), -1)

        elif obj.shape_style == "foam":
            cv2.rectangle(frame, (ox - hw, oy - hh), (ox + hw, oy + hh), (55, 55, 58), -1)
            cv2.rectangle(frame, (ox - hw + 2, oy + int(hh * 0.45)), (ox + hw - 2, oy + hh - 3), (238, 238, 242), -1)

        elif obj.shape_style == "bottle":
            cv2.rectangle(frame, (ox - hw, oy - hh + 10), (ox + hw, oy + hh), obj.bgr, -1)
            dark_bgr = tuple(max(10, int(c * 0.8)) for c in obj.bgr)
            cv2.rectangle(frame, (ox - 9, oy - hh - 4), (ox + 9, oy - hh + 10), dark_bgr, -1)
            cv2.rectangle(frame, (ox - 10, oy - hh - 12), (ox + 10, oy - hh - 4), (242, 242, 242), -1)

        elif obj.shape_style == "phone":
            cv2.rectangle(frame, (ox - hw, oy - hh), (ox + hw, oy + hh), obj.bgr, -1)
            cv2.rectangle(frame, (ox - hw + 3, oy - hh + 5), (ox + hw - 3, oy + hh - 6), (65, 70, 75), -1)
            cv2.circle(frame, (ox, oy + hh - 3), 2, (180, 180, 180), -1)

        elif obj.shape_style == "cup":
            cv2.rectangle(frame, (ox - hw, oy - hh), (ox + hw, oy + hh), obj.bgr, -1)
            dark_bgr = tuple(max(10, int(c * 0.75)) for c in obj.bgr)
            cv2.rectangle(frame, (ox - hw, oy - hh), (ox + hw, oy + hh), dark_bgr, 2)
            cv2.ellipse(frame, (ox, oy - hh + 5), (hw - 3, 6), 0, 0, 360, (235, 235, 235), -1)

        else:
            cv2.rectangle(frame, (ox - hw, oy - hh), (ox + hw, oy + hh), obj.bgr, -1)
            dark_bgr = tuple(max(10, int(c * 0.7)) for c in obj.bgr)
            cv2.rectangle(frame, (ox - hw, oy - hh), (ox + hw, oy + hh), dark_bgr, 2)

    def _generate_synthetic_frame(self) -> np.ndarray:
        self.sim_tick += 1
        frame = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), (185, 205, 220), dtype=np.uint8)

        # Tabletop surface + grid
        cv2.rectangle(frame, (25, 25), (config.FRAME_WIDTH - 25, config.FRAME_HEIGHT - 25), (175, 198, 215), -1)
        cv2.rectangle(frame, (25, 25), (config.FRAME_WIDTH - 25, config.FRAME_HEIGHT - 25), (135, 155, 170), 2)

        for gx in range(80, config.FRAME_WIDTH - 40, 80):
            cv2.line(frame, (gx, 28), (gx, config.FRAME_HEIGHT - 28), (168, 190, 206), 1)
        for gy in range(80, config.FRAME_HEIGHT - 40, 80):
            cv2.line(frame, (28, gy), (config.FRAME_WIDTH - 28, gy), (168, 190, 206), 1)

        # Draw receptacles first, then resting objects, then carried object on top
        receptacles = [o for o in self.sim_objects.values() if o.is_receptacle]
        resting = [o for o in self.sim_objects.values() if not o.is_receptacle and o.key != self.carried_sim_key]
        carried = [self.sim_objects[self.carried_sim_key]] if (self.carried_sim_key in self.sim_objects) else []

        for obj in receptacles:
            self._draw_single_sim_object(frame, obj, is_lifted=False)
        for obj in resting:
            self._draw_single_sim_object(frame, obj, is_lifted=False)
        for obj in carried:
            self._draw_single_sim_object(frame, obj, is_lifted=(self.carried_z_cm > 3.0))

        return frame

    # ------------------------------------------------------------------
    # OWLv2 OPEN-VOCABULARY AI + MULTI-CUE TABLETOP GROUNDING
    # ------------------------------------------------------------------
    def _detect_with_owlv2(
        self,
        frame: np.ndarray,
        query: str,
        exclude_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[DetectedObject]:
        """Runs OWLv2 open-vocabulary object detection on the frame if loaded."""
        if not self._try_load_owlv2():
            return []

        try:
            import torch
            from PIL import Image

            img_h, img_w = frame.shape[:2]
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            generic = ["object", "box", "bottle", "cup", "electronic component", "container"]
            phrases = [query.lower().strip()] + [g for g in generic if g != query.lower().strip()]

            inputs = self._owl_processor(text=[phrases], images=pil_img, return_tensors="pt").to(self._owl_device)
            with torch.no_grad():
                outputs = self._owl_model(**inputs)

            results = self._owl_processor.post_process_grounded_object_detection(
                outputs, threshold=0.04, target_sizes=[(img_h, img_w)]
            )[0]

            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()
            boxes = results["boxes"].cpu().numpy()

            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            order = np.argsort(-scores)
            specific = [i for i in order if labels[i] == 0][:4]
            rest = [i for i in order if i not in specific]

            candidates: List[DetectedObject] = []
            for idx in specific + rest:
                x0, y0, x1, y1 = [int(round(v)) for v in boxes[idx]]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(img_w, x1), min(img_h, y1)
                w, h = x1 - x0, y1 - y0
                if w < 16 or h < 16 or (w * h) > (img_w * img_h * 0.78):
                    continue

                box_xyxy = (x0, y0, x1, y1)
                if exclude_bbox is not None:
                    ex_xyxy = (exclude_bbox[0], exclude_bbox[1], exclude_bbox[0] + exclude_bbox[2], exclude_bbox[1] + exclude_bbox[3])
                    if self._iou_xyxy(box_xyxy, ex_xyxy) > 0.50:
                        continue
                if any(self._iou_xyxy(box_xyxy, c.box_xyxy) > 0.50 for c in candidates):
                    continue

                feats = self._extract_region_features(frame, hsv, (x0, y0, w, h))
                prop = {"bbox": (x0, y0, w, h), "box_xyxy": box_xyxy, "center": (x0 + w // 2, y0 + h // 2), "area": float(w * h), "features": feats}
                vis_score, col_tag, vis_reason = self._score_proposal_for_query(prop, query, frame.shape)

                owl_raw = float(scores[idx])
                is_specific_match = (labels[idx] == 0)
                combined_score = min(0.99, owl_raw * 1.4 + (0.20 if is_specific_match else 0.0) + vis_score * 0.45)

                cx, cy = x0 + w // 2, y0 + h // 2
                rx, ry, rz = self.calibrator.pixel_to_robot(cx, cy, z=0.0)
                matched_phrase = phrases[int(labels[idx])]
                reason = f"OWLv2 matched '{matched_phrase}' (conf {owl_raw:.2f}) | {vis_reason}"

                candidates.append(
                    DetectedObject(
                        color=col_tag,
                        center_px=(cx, cy),
                        bbox=(x0, y0, w, h),
                        area=float(w * h),
                        robot_coords=(rx, ry, rz),
                        label=query,
                        score=round(combined_score, 2),
                        reason=reason,
                    )
                )
                if len(candidates) >= 8:
                    break

            candidates.sort(key=lambda c: c.score, reverse=True)
            return candidates
        except Exception as e:
            print(f"[Vision AI] OWLv2 fallback: {e}")
            return []

    def _extract_region_features(self, frame: np.ndarray, hsv: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        x, y, w, h = bbox
        roi_bgr = frame[y : y + h, x : x + w]
        roi_hsv = hsv[y : y + h, x : x + w]
        total_px = max(1, w * h)

        hue = roi_hsv[:, :, 0]
        sat = roi_hsv[:, :, 1]
        val = roi_hsv[:, :, 2]

        dark_mask = (val < 110) & (sat < 95)
        dark_ratio = float(np.count_nonzero(dark_mask)) / total_px

        white_mask = (val > 175) & (sat < 55)
        white_ratio = float(np.count_nonzero(white_mask)) / total_px

        bottom_half_val = val[int(h * 0.65) :, :]
        bottom_half_sat = sat[int(h * 0.65) :, :]
        bottom_white_mask = (bottom_half_val > 170) & (bottom_half_sat < 55)
        bottom_white_ratio = float(np.count_nonzero(bottom_white_mask)) / max(1, bottom_half_val.size)

        color_ratios: Dict[str, float] = {
            "white": white_ratio,
            "black": dark_ratio,
            "grey": float(np.count_nonzero((val >= 85) & (val <= 185) & (sat < 55))) / total_px,
        }
        for cname in ["red", "blue", "green", "yellow", "orange"]:
            cmask = np.zeros(hue.shape, dtype=np.uint8)
            for low, high in config.COLOR_RANGES[cname]:
                cmask = cv2.bitwise_or(cmask, cv2.inRange(roi_hsv, np.array(low), np.array(high)))
            color_ratios[cname] = float(np.count_nonzero(cmask)) / total_px

        cardboard_mask = (hue >= 10) & (hue <= 26) & (sat >= 45) & (sat <= 170) & (val >= 90) & (val <= 215)
        color_ratios["brown"] = float(np.count_nonzero(cardboard_mask)) / total_px

        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

        return {
            "dark_ratio": dark_ratio,
            "white_ratio": white_ratio,
            "bottom_white_ratio": bottom_white_ratio,
            "lap_var": lap_var,
            "aspect": float(w) / max(1.0, float(h)),
            **color_ratios,
        }

    def propose_tabletop_objects(self, frame: np.ndarray, max_proposals: int = 10) -> List[Dict[str, Any]]:
        img_h, img_w = frame.shape[:2]
        img_area = float(img_w * img_h)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # In synthetic tabletop mode, seed proposals directly from tabletop objects so nearby objects never merge
        if self.use_simulation:
            proposals: List[Dict[str, Any]] = []
            for key, sobj in self.sim_objects.items():
                ox, oy = sobj.pos
                hw, hh = sobj.size
                x = max(0, ox - hw - 2)
                y = max(0, oy - hh - 2)
                w = min(img_w - x, 2 * hw + 4)
                h = min(img_h - y, 2 * hh + 4)
                feats = self._extract_region_features(frame, hsv, (x, y, w, h))
                proposals.append({
                    "bbox": (x, y, w, h),
                    "box_xyxy": (x, y, x + w, y + h),
                    "center": (ox, oy),
                    "area": float(w * h),
                    "features": feats,
                    "sim_key": key,
                })
            return proposals

        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

        cy0, cy1 = int(img_h * 0.20), int(img_h * 0.85)
        cx0, cx1 = int(img_w * 0.15), int(img_w * 0.85)
        center_lab = lab[cy0:cy1:4, cx0:cx1:4].reshape(-1, 3).astype(np.float32)
        bg_lab = np.median(center_lab, axis=0)

        diff_lab = np.linalg.norm(lab.astype(np.float32) - bg_lab, axis=2)

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        fg_mask = ((diff_lab > 20.0) | (sat > 80) | (val < 115)).astype(np.uint8) * 255

        white_obj_mask = ((sat < 32) & (val > 150) & (diff_lab > 8.0)).astype(np.uint8) * 255
        white_obj_mask[: int(img_h * 0.25), :] = 0
        white_obj_mask[:, : int(img_w * 0.35)] = 0

        fg_mask[:4, :] = 0
        fg_mask[-4:, :] = 0
        fg_mask[:, :4] = 0
        fg_mask[:, -4:] = 0

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        white_obj_mask = cv2.morphologyEx(white_obj_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        white_obj_mask = cv2.morphologyEx(white_obj_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        for cname in ["red", "blue", "green", "yellow", "orange"]:
            cmask = np.zeros((img_h, img_w), dtype=np.uint8)
            for low, high in config.COLOR_RANGES[cname]:
                cmask = cv2.bitwise_or(cmask, cv2.inRange(hsv, np.array(low), np.array(high)))
            cmask = cv2.morphologyEx(cmask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            fg_mask = cv2.bitwise_or(fg_mask, cmask)

        contours_fg, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_wh, _ = cv2.findContours(white_obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours_fg) + list(contours_wh)
        raw_boxes: List[Tuple[int, int, int, int, float]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.003 or area > img_area * 0.45:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > img_area * 0.50 or w < 16 or h < 16:
                continue

            touches_left = x <= 8
            touches_right = (x + w) >= (img_w - 8)
            touches_top = y <= 8
            touches_bottom = (y + h) >= (img_h - 8)
            edges_touched = int(touches_left) + int(touches_right) + int(touches_top) + int(touches_bottom)
            if edges_touched >= 2:
                continue
            if touches_top and (y + h) < int(img_h * 0.25):
                continue
            if touches_left and (y + h) > int(img_h * 0.65):
                continue

            raw_boxes.append((x, y, w, h, float(area)))

        raw_boxes.sort(key=lambda b: b[4], reverse=True)
        proposals: List[Dict[str, Any]] = []
        for x, y, w, h, area in raw_boxes:
            box_a = (x, y, x + w, y + h)
            if any(self._iou_xyxy(box_a, p["box_xyxy"]) > 0.45 for p in proposals):
                continue
            feats = self._extract_region_features(frame, hsv, (x, y, w, h))
            proposals.append({
                "bbox": (x, y, w, h),
                "box_xyxy": box_a,
                "center": (x + w // 2, y + h // 2),
                "area": area,
                "features": feats,
            })
            if len(proposals) >= max_proposals:
                break

        return proposals

    @staticmethod
    def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        x0 = max(a[0], b[0])
        y0 = max(a[1], b[1])
        x1 = min(a[2], b[2])
        y1 = min(a[3], b[3])
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return float(inter) / float(area_a + area_b - inter)

    def _score_proposal_for_query(self, prop: Dict[str, Any], query: str, img_shape: Tuple[int, int, int]) -> Tuple[float, str, str]:
        q = query.lower().strip()
        f = prop["features"]
        x, y, w, h = prop["bbox"]
        img_h, img_w = img_shape[:2]

        # In simulation mode, check matching against dynamic SimObjects
        if self.use_simulation:
            sim_key = self.find_sim_object_key_for_query_match(q)
            prop_key = prop.get("sim_key")
            if prop_key and prop_key in self.sim_objects:
                sobj = self.sim_objects[prop_key]
                if sim_key == prop_key:
                    return 0.98, sobj.color_name, f"AI grounded '{sobj.label}' ({sobj.shape_style}, {sobj.color_name})."
                else:
                    return 0.12, sobj.color_name, f"Tabletop object '{sobj.label}' ({sobj.color_name} {sobj.shape_style})."
            elif sim_key and sim_key in self.sim_objects:
                sobj = self.sim_objects[sim_key]
                cx, cy = prop["center"]
                if np.hypot(cx - sobj.pos[0], cy - sobj.pos[1]) < max(sobj.size) * 1.1:
                    return 0.98, sobj.color_name, f"AI grounded '{sobj.label}' ({sobj.shape_style}, {sobj.color_name})."

        score = 0.25
        reason = f"Matched tabletop object candidate at ({x}, {y}, {x+w}, {y+h})."
        dom_color = "target"

        if "mixed" in q and "foam" in q:
            if f["dark_ratio"] > 0.30 and (f["bottom_white_ratio"] > 0.08 or (0.04 <= f["white_ratio"] <= 0.45)):
                score = 0.92 + 0.08 * min(1.0, f["bottom_white_ratio"] * 3.0)
                dom_color = "white"
                reason = (
                    f"Box contains dark foam ({f['dark_ratio']*100:.0f}%) mixed with a distinct "
                    f"white foam layer ({f['bottom_white_ratio']*100:.0f}% at base)."
                )
            elif f["dark_ratio"] > 0.35:
                score = 0.48
                dom_color = "black"
                reason = "Dark foam block without prominent white layer."
            else:
                score = 0.12
            return round(min(0.99, score), 2), dom_color, reason

        if "breadboard" in q:
            if f["white_ratio"] > 0.40 and f["lap_var"] > 180.0:
                score = 0.95
                dom_color = "white"
                reason = f"White rectangular breadboard with high pinhole grid texture (white={f['white_ratio']*100:.0f}%)."
            else:
                score = 0.15 + 0.3 * f["white_ratio"]
            return round(min(0.99, score), 2), dom_color, reason

        if "foam" in q or "sponge" in q:
            if f["dark_ratio"] > 0.35:
                if "black" in q or "dark" in q:
                    score = 0.90 - f["bottom_white_ratio"] * 0.5
                    reason = f"Pure dark/black foam block ({f['dark_ratio']*100:.0f}% dark)."
                else:
                    score = 0.85
                    reason = f"Foam block detected ({f['dark_ratio']*100:.0f}% dark foam profile)."
                dom_color = "black"
            else:
                score = 0.10
            return round(min(0.99, score), 2), dom_color, reason

        if ("grey" in q or "gray" in q) and "white" in q and "box" in q:
            if f["grey"] > 0.15 and f["white"] > 0.08 and x > img_w * 0.55:
                score = 0.91
                dom_color = "grey"
                reason = "Grey and white rectangular box located on the right side of the workspace."
            elif f["grey"] > 0.15 and f["white"] > 0.05:
                score = 0.72
                dom_color = "grey"
                reason = "Grey and white box surface detected."
            else:
                score = 0.15
            return round(min(0.99, score), 2), dom_color, reason

        if "cardboard" in q:
            if f["brown"] > 0.20 and w * h > (img_w * img_h * 0.03):
                score = 0.90
                dom_color = "orange"
                reason = f"Corrugated tan/brown cardboard box ({f['brown']*100:.0f}% cardboard tone)."
            else:
                score = 0.15
            return round(min(0.99, score), 2), dom_color, reason

        if "phone" in q:
            if f["black"] > 0.35 or f["grey"] > 0.35:
                score = 0.91
                dom_color = "black"
                reason = f"Dark rectangular device profile matching '{query}'."
                return round(score, 2), dom_color, reason

        for cname in ["red", "blue", "green", "yellow", "orange", "white", "black", "grey"]:
            if cname in q or (cname == "grey" and "gray" in q):
                c_ratio = f.get(cname, 0.0)
                if c_ratio > 0.10:
                    score = 0.62 + 0.37 * min(1.0, c_ratio / 0.50)
                    dom_color = cname
                    reason = f"Matched {cname} object ({c_ratio*100:.0f}% {cname} pixels in region)."
                else:
                    score = 0.08
                return round(min(0.99, score), 2), dom_color, reason

        cx, cy = prop["center"]
        dist_center = np.hypot((cx - img_w * 0.5) / img_w, (cy - img_h * 0.5) / img_h)
        score = max(0.38, 0.72 - 0.45 * float(dist_center))
        return round(score, 2), dom_color, reason

    def find_sim_object_key_for_query_match(self, query: str) -> Optional[str]:
        return self.find_sim_object_key_for_phrase(query)

    def locate_by_query(
        self,
        frame: np.ndarray,
        query: str,
        exclude_bbox: Optional[Tuple[int, int, int, int]] = None,
        use_owlv2: bool = True,
    ) -> Tuple[Optional[DetectedObject], List[DetectedObject]]:
        """
        Finds the best matching object for ANY open-vocabulary `query` and returns
        both the top match and all numbered candidate objects in the scene.
        """
        img_h, img_w = frame.shape[:2]
        if self.calibrator.frame_w != img_w or self.calibrator.frame_h != img_h:
            self.calibrator = WorkspaceCalibrator(frame_w=img_w, frame_h=img_h)

        # 1. On real camera/photo frames, run OWLv2 if explicitly requested or GPU-accelerated
        owl_candidates: List[DetectedObject] = []
        if use_owlv2 and not self.use_simulation and os.environ.get("USE_OWLV2", "0") == "1":
            owl_candidates = self._detect_with_owlv2(frame, query, exclude_bbox=exclude_bbox)

        # 2. Multi-cue tabletop segmentation proposals
        proposals = self.propose_tabletop_objects(frame, max_proposals=8)
        q_lower = query.lower()
        for cname in ["red", "blue", "green", "yellow", "orange"]:
            if cname in q_lower:
                col_objs = self.detect_objects_by_color(frame, [cname]).get(cname, [])
                for co in col_objs:
                    if not any(self._iou_xyxy(co.box_xyxy, p["box_xyxy"]) > 0.4 for p in proposals):
                        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
                        proposals.append({
                            "bbox": co.bbox,
                            "box_xyxy": co.box_xyxy,
                            "center": co.center_px,
                            "area": co.area,
                            "features": self._extract_region_features(frame, hsv, co.bbox),
                        })

        cv_candidates: List[DetectedObject] = []
        for prop in proposals:
            if exclude_bbox is not None:
                ex_xyxy = (exclude_bbox[0], exclude_bbox[1], exclude_bbox[0] + exclude_bbox[2], exclude_bbox[1] + exclude_bbox[3])
                if self._iou_xyxy(prop["box_xyxy"], ex_xyxy) > 0.5:
                    continue

            sc, col_tag, reason = self._score_proposal_for_query(prop, query, frame.shape)
            cx, cy = prop["center"]
            rx, ry, rz = self.calibrator.pixel_to_robot(cx, cy, z=0.0)
            cand_label = query
            if self.use_simulation and prop.get("sim_key") in self.sim_objects:
                cand_label = self.sim_objects[prop["sim_key"]].label
            cv_candidates.append(
                DetectedObject(
                    color=col_tag,
                    center_px=(cx, cy),
                    bbox=prop["bbox"],
                    area=prop["area"],
                    robot_coords=(rx, ry, rz),
                    label=cand_label,
                    score=sc,
                    reason=reason,
                )
            )

        # Merge OWLv2 and CV candidates (preferring highest score and deduplicating by IoU)
        merged: List[DetectedObject] = []
        for cand in sorted(owl_candidates + cv_candidates, key=lambda c: c.score, reverse=True):
            if not any(self._iou_xyxy(cand.box_xyxy, m.box_xyxy) > 0.45 for m in merged):
                merged.append(cand)
            if len(merged) >= 8:
                break

        best = merged[0] if (merged and merged[0].score >= 0.35) else None
        return best, merged

    def detect_objects_by_color(
        self,
        frame: np.ndarray,
        target_colors: Optional[List[str]] = None,
    ) -> Dict[str, List[DetectedObject]]:
        img_h, img_w = frame.shape[:2]
        if self.calibrator.frame_w != img_w or self.calibrator.frame_h != img_h:
            self.calibrator = WorkspaceCalibrator(frame_w=img_w, frame_h=img_h)

        colors_to_search = target_colors if target_colors else ["red", "blue", "green", "yellow", "orange"]
        results: Dict[str, List[DetectedObject]] = {}

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        for color_name in colors_to_search:
            color_lower = color_name.lower()
            if color_lower not in config.COLOR_RANGES:
                continue

            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for low, high in config.COLOR_RANGES[color_lower]:
                sub_mask = cv2.inRange(hsv, np.array(low), np.array(high))
                mask = cv2.bitwise_or(mask, sub_mask)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected_list: List[DetectedObject] = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if config.MIN_CONTOUR_AREA <= area <= config.MAX_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w // 2, y + h // 2
                    rx, ry, rz = self.calibrator.pixel_to_robot(cx, cy, z=0.0)
                    detected_list.append(
                        DetectedObject(
                            color=color_lower,
                            center_px=(cx, cy),
                            bbox=(x, y, w, h),
                            area=area,
                            robot_coords=(rx, ry, rz),
                            label=f"{color_lower} object",
                            score=0.90,
                            reason=f"HSV color match for {color_lower}",
                        )
                    )

            detected_list.sort(key=lambda o: o.area, reverse=True)
            results[color_lower] = detected_list

        return results

    def find_specific_object(
        self,
        frame: np.ndarray,
        color_or_query: str,
        label_hint: str = "object",
        exclude_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[DetectedObject]:
        full_query = f"{color_or_query} {label_hint}".strip() if label_hint not in color_or_query else color_or_query
        best, _ = self.locate_by_query(frame, full_query, exclude_bbox=exclude_bbox)
        if best is not None:
            best.label = full_query
        return best

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
