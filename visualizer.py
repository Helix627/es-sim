"""
Visualizer & HUD Overlay Engine for Vision-Guided Robotic Arm.
Renders:
  1. Left Panel: Camera / Tabletop feed with bounding boxes, 3D coordinates,
     planned trajectory vector, AND a live top-down robotic arm overlay reaching
     for and carrying the object.
  2. Right Panel: 2D Sagittal Side Elevation view (showing the arm, target markers,
     and grasped object in the claw), Top-Down Base Yaw radar, and ESP32 servo telemetry.
"""
import cv2
import numpy as np
import math
from typing import List, Optional, Tuple
import config
from calibration import WorkspaceCalibrator
from kinematics import KinematicsSolver, JointState
from vision import DetectedObject


class DashboardVisualizer:
    def __init__(
        self,
        solver: Optional[KinematicsSolver] = None,
        calibrator: Optional[WorkspaceCalibrator] = None,
    ):
        self.solver = solver or KinematicsSolver()
        self.calibrator = calibrator or WorkspaceCalibrator()
        self.telemetry_w = 500
        self.height = config.FRAME_HEIGHT
        self.cam_w = config.FRAME_WIDTH
        self.total_w = self.cam_w + self.telemetry_w

    def draw_camera_hud(
        self,
        frame: np.ndarray,
        detected_objects: List[DetectedObject],
        pick_obj: Optional[DetectedObject] = None,
        place_obj: Optional[DetectedObject] = None,
        status_msg: str = "SYSTEM READY",
        command_text: str = "",
        current_angles: Optional[Tuple[float, float, float, float]] = None,
        gripper_state: str = "OPEN",
        typing_mode: bool = False,
        typing_buffer: str = "",
    ) -> np.ndarray:
        """Overlays bounding boxes, 3D coordinates, planned path, and live top-down arm on camera frame."""
        if frame.shape[1] != self.cam_w or frame.shape[0] != self.height:
            vis = cv2.resize(frame, (self.cam_w, self.height))
            sx = self.cam_w / frame.shape[1]
            sy = self.height / frame.shape[0]
        else:
            vis = frame.copy()
            sx, sy = 1.0, 1.0

        # Top Banner
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (self.cam_w, 60), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.80, vis, 0.20, 0, vis)

        cv2.putText(
            vis,
            "VISION-GUIDED ROBOTIC ARM | AI OBJECT TRACKER",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cmd_display = f"CMD: \"{command_text[:58]}\"" if command_text else "CMD: Waiting for command..."
        cv2.putText(
            vis,
            cmd_display,
            (12, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw transfer arrow from Pick to Place if both are present
        if pick_obj and place_obj:
            p1 = (int(pick_obj.center_px[0] * sx), int(pick_obj.center_px[1] * sy))
            p2 = (int(place_obj.center_px[0] * sx), int(place_obj.center_px[1] * sy))
            cv2.arrowedLine(vis, p1, p2, (0, 255, 255), 2, tipLength=0.06)
            mid = ((p1[0] + p2[0]) // 2 - 45, (p1[1] + p2[1]) // 2 - 10)
            cv2.putText(
                vis,
                "PLANNED PATH",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Pass 1: Draw all bounding box outlines and centroids first
        for idx, obj in enumerate(detected_objects, start=1):
            ox, oy, ow, oh = obj.bbox
            x, y, w, h = int(ox * sx), int(oy * sy), int(ow * sx), int(oh * sy)
            cx, cy = int(obj.center_px[0] * sx), int(obj.center_px[1] * sy)

            is_pick = pick_obj is not None and (obj is pick_obj or obj.bbox == pick_obj.bbox)
            is_place = place_obj is not None and (obj is place_obj or obj.bbox == place_obj.bbox)

            if is_pick:
                bgr, thickness = (0, 0, 255), 3
            elif is_place:
                bgr, thickness = (255, 130, 0), 3
            else:
                bgr, thickness = config.DISPLAY_COLORS.get(obj.color, (180, 180, 60)), 1

            cv2.rectangle(vis, (x, y), (x + w, y + h), bgr, thickness)
            if is_pick or is_place:
                cv2.line(vis, (cx - 10, cy), (cx + 10, cy), (0, 255, 255), 2)
                cv2.line(vis, (cx, cy - 10), (cx, cy + 10), (0, 255, 255), 2)
                cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)

        # Pass 1.5: Draw live Top-Down Robotic Arm projection onto Tabletop View
        if current_angles is not None:
            pts3d = self.solver.get_3d_joint_positions(*current_angles)
            base_px = (self.cam_w // 2, self.height - 12)
            elb_px = self.calibrator.robot_to_pixel(pts3d.elbow[0], pts3d.elbow[1])
            wri_px = self.calibrator.robot_to_pixel(pts3d.wrist[0], pts3d.wrist[1])
            grp_px = self.calibrator.robot_to_pixel(pts3d.gripper[0], pts3d.gripper[1])

            elb_px = (max(10, min(self.cam_w - 10, elb_px[0])), max(62, min(self.height - 10, elb_px[1])))
            wri_px = (max(10, min(self.cam_w - 10, wri_px[0])), max(62, min(self.height - 10, wri_px[1])))
            grp_px = (max(10, min(self.cam_w - 10, grp_px[0])), max(62, min(self.height - 10, grp_px[1])))

            arm_overlay = vis.copy()
            # Arm links
            cv2.line(arm_overlay, base_px, elb_px, (40, 195, 245), 8, cv2.LINE_AA)
            cv2.line(arm_overlay, elb_px, wri_px, (30, 140, 220), 6, cv2.LINE_AA)
            cv2.line(arm_overlay, wri_px, grp_px, (80, 220, 80), 5, cv2.LINE_AA)

            # Base pedestal & joints
            cv2.circle(arm_overlay, base_px, 14, (70, 70, 70), -1, cv2.LINE_AA)
            cv2.circle(arm_overlay, base_px, 8, (220, 220, 220), -1, cv2.LINE_AA)
            cv2.circle(arm_overlay, elb_px, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(arm_overlay, wri_px, 5, (255, 255, 255), -1, cv2.LINE_AA)

            # End-Effector Claw Ring
            grip_open = gripper_state.upper() == "OPEN"
            claw_col = (60, 230, 60) if grip_open else (40, 40, 255)
            claw_radius = 14 if grip_open else 8
            cv2.circle(arm_overlay, grp_px, claw_radius, claw_col, 2, cv2.LINE_AA)
            cv2.circle(arm_overlay, grp_px, 3, (255, 255, 255), -1, cv2.LINE_AA)

            cv2.addWeighted(arm_overlay, 0.82, vis, 0.18, 0, vis)

        # Pass 2: Draw all labels/badges on top so no outline cuts through text
        for idx, obj in enumerate(detected_objects, start=1):
            ox, oy, ow, oh = obj.bbox
            x, y, w, h = int(ox * sx), int(oy * sy), int(ow * sx), int(oh * sy)
            rx, ry, rz = obj.robot_coords

            is_pick = pick_obj is not None and (obj is pick_obj or obj.bbox == pick_obj.bbox)
            is_place = place_obj is not None and (obj is place_obj or obj.bbox == place_obj.bbox)

            if is_pick:
                bgr = (0, 0, 255)
            elif is_place:
                bgr = (255, 130, 0)
            else:
                bgr = config.DISPLAY_COLORS.get(obj.color, (180, 180, 60))

            if is_pick or is_place:
                role_prefix = "[PICK]" if is_pick else "[PLACE]"
                target_label = pick_obj.label if (is_pick and pick_obj) else (place_obj.label if (is_place and place_obj) else obj.label)
                label_str = f"{role_prefix} {target_label.upper()}"
                coord_str = f"3D: X={rx:+.1f} Y={ry:+.1f} Z={rz:+.1f}cm"

                (tw1, th1), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                (tw2, _), _ = cv2.getTextSize(coord_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                box_w = max(tw1, tw2) + 14

                if is_pick:
                    tag_y = y - 6 if y > 96 else y + h + 38
                else:
                    tag_y = y + h + 38 if (y + h + 42 < self.height - 38) else max(96, y - 6)
                tag_x = max(4, min(x, self.cam_w - box_w - 4))

                cv2.rectangle(vis, (tag_x, tag_y - 36), (tag_x + box_w, tag_y), (20, 20, 20), -1)
                cv2.rectangle(vis, (tag_x, tag_y - 36), (tag_x + box_w, tag_y), bgr, 1)
                cv2.putText(
                    vis,
                    label_str,
                    (tag_x + 6, tag_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    bgr if not is_pick else (100, 100, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    coord_str,
                    (tag_x + 6, tag_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.rectangle(vis, (x, y), (x + 26, y + 20), bgr, -1)
                cv2.putText(
                    vis,
                    f"{idx}",
                    (x + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Status badge bottom-left
        status_w = min(self.cam_w - 20, max(430, len(status_msg) * 8 + 90))
        cv2.rectangle(vis, (10, self.height - 34), (10 + status_w, self.height - 8), (18, 18, 18), -1)
        status_col = (0, 230, 255) if "READY TO MOVE" in status_msg else (80, 255, 80)
        cv2.putText(
            vis,
            f"STATUS: {status_msg}",
            (18, self.height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            status_col,
            1,
            cv2.LINE_AA,
        )

        # Interactive On-Screen Typing Bar (when user presses [T] to type a variable object)
        if typing_mode:
            bar_y1, bar_y2 = self.height // 2 - 38, self.height // 2 + 38
            overlay_bar = vis.copy()
            cv2.rectangle(overlay_bar, (30, bar_y1), (self.cam_w - 30, bar_y2), (15, 20, 30), -1)
            cv2.addWeighted(overlay_bar, 0.92, vis, 0.08, 0, vis)
            cv2.rectangle(vis, (30, bar_y1), (self.cam_w - 30, bar_y2), (0, 215, 255), 2)
            cv2.putText(
                vis,
                "TYPE VARIABLE OBJECT OR COMMAND (Press ENTER to Identify, ESC to Cancel):",
                (44, bar_y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (180, 230, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"> {typing_buffer}_",
                (44, bar_y1 + 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )

        return vis

    def draw_telemetry_panel(
        self,
        current_angles: Tuple[float, float, float, float],
        gripper_state: str,
        current_xyz: Tuple[float, float, float],
        target_xyz: Optional[Tuple[float, float, float]] = None,
        phase_label: str = "IDLE",
        step_progress: Tuple[int, int] = (0, 10),
        pick_xyz: Optional[Tuple[float, float, float]] = None,
        place_xyz: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """
        Renders the right telemetry dashboard containing:
        - 2D Sagittal (Side) elevation schematic of the 5-DOF Robotic Arm + grasped object
        - Top-Down Base Yaw radar (B servo / theta_1)
        - Joint angles & ESP32 Servo values (B, S, E, W, C)
        - Cartesian XYZ pose & Gripper status
        """
        panel = np.full((self.height, self.telemetry_w, 3), 28, dtype=np.uint8)

        # Header
        cv2.rectangle(panel, (0, 0), (self.telemetry_w, 42), (42, 42, 42), -1)
        cv2.putText(
            panel,
            "ROBOT ARM KINEMATICS & ESP32 TELEMETRY",
            (14, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        t1, t2, t3, t4 = current_angles
        grip_open = gripper_state.upper() == "OPEN"
        js = JointState(t1, t2, t3, t4, gripper_closed=not grip_open, is_reachable=True)

        # -------------------------------------------------------------
        # 1. Side Elevation View (Left viewport) & Top-Down Radar (Right viewport)
        # -------------------------------------------------------------
        cv2.rectangle(panel, (14, 50), (325, 255), (18, 18, 18), -1)
        cv2.rectangle(panel, (14, 50), (325, 255), (58, 58, 58), 1)
        cv2.putText(
            panel,
            "SIDE VIEW (REACH & HEIGHT)",
            (22, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (150, 150, 150),
            1,
        )

        scale = 4.4
        origin_x = 75
        origin_y = 228

        # Table ground line
        cv2.line(panel, (22, origin_y), (315, origin_y), (80, 80, 80), 2)
        cv2.putText(
            panel,
            "Z=0 cm",
            (265, origin_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (110, 110, 110),
            1,
        )

        # Draw target reach markers on table surface in Side View
        if pick_xyz is not None:
            r_pick = math.hypot(pick_xyz[0], pick_xyz[1])
            px_side = origin_x + int(r_pick * scale)
            if 25 <= px_side <= 310:
                cv2.rectangle(panel, (px_side - 6, origin_y - 10), (px_side + 6, origin_y), (60, 60, 220), 1)
        if place_xyz is not None:
            r_place = math.hypot(place_xyz[0], place_xyz[1])
            lx_side = origin_x + int(r_place * scale)
            if 25 <= lx_side <= 310:
                cv2.rectangle(panel, (lx_side - 10, origin_y - 6), (lx_side + 10, origin_y), (255, 130, 0), 2)

        t2_r = math.radians(t2)
        t3_r = math.radians(t3)
        t4_r = math.radians(t4)

        p0 = (origin_x, origin_y)
        p1 = (origin_x, origin_y - int(self.solver.l1 * scale))

        r_elbow = self.solver.l2 * math.cos(t2_r)
        z_elbow = self.solver.l1 + self.solver.l2 * math.sin(t2_r)
        p2 = (origin_x + int(r_elbow * scale), origin_y - int(z_elbow * scale))

        r_wrist = r_elbow + self.solver.l3 * math.cos(t2_r + t3_r)
        z_wrist = z_elbow + self.solver.l3 * math.sin(t2_r + t3_r)
        p3 = (origin_x + int(r_wrist * scale), origin_y - int(z_wrist * scale))

        phi = t2_r + t3_r + t4_r
        r_grip = r_wrist + self.solver.l4 * math.cos(phi)
        z_grip = z_wrist + self.solver.l4 * math.sin(phi)
        p4 = (origin_x + int(r_grip * scale), origin_y - int(z_grip * scale))

        # Draw Arm Links
        cv2.line(panel, p0, p1, (160, 160, 160), 6, cv2.LINE_AA)
        cv2.line(panel, p1, p2, (40, 195, 245), 5, cv2.LINE_AA)
        cv2.line(panel, p2, p3, (30, 140, 220), 4, cv2.LINE_AA)
        cv2.line(panel, p3, p4, (80, 220, 80), 3, cv2.LINE_AA)

        for pt, rad in [(p0, 6), (p1, 6), (p2, 5), (p3, 5)]:
            cv2.circle(panel, pt, rad, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel, pt, max(2, rad - 3), (30, 30, 30), -1, cv2.LINE_AA)

        # Draw Gripper Claw Fingers & Grasped Object if CLOSED
        finger_spread = 10 if grip_open else 5
        norm_x = -math.sin(phi)
        norm_y = math.cos(phi)
        f1 = (int(p4[0] + norm_x * finger_spread), int(p4[1] + norm_y * finger_spread))
        f2 = (int(p4[0] - norm_x * finger_spread), int(p4[1] - norm_y * finger_spread))
        grip_col = (60, 220, 60) if grip_open else (50, 60, 255)
        cv2.line(panel, p4, f1, grip_col, 2, cv2.LINE_AA)
        cv2.line(panel, p4, f2, grip_col, 2, cv2.LINE_AA)

        if not grip_open:
            # Show carried object held in the claw in side view
            cv2.rectangle(panel, (p4[0] - 5, p4[1] - 4), (p4[0] + 5, p4[1] + 8), (30, 30, 230), -1)
            cv2.rectangle(panel, (p4[0] - 5, p4[1] - 4), (p4[0] + 5, p4[1] + 8), (255, 255, 255), 1)

        # Top-Down Base Yaw Radar (Right Box)
        cv2.rectangle(panel, (335, 50), (self.telemetry_w - 14, 255), (18, 18, 18), -1)
        cv2.rectangle(panel, (335, 50), (self.telemetry_w - 14, 255), (58, 58, 58), 1)
        cv2.putText(
            panel,
            "TOP VIEW (BASE B)",
            (345, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (150, 150, 150),
            1,
        )

        radar_cx, radar_cy = 410, 195
        radar_r = 58
        cv2.ellipse(panel, (radar_cx, radar_cy), (radar_r, radar_r), 0, 180, 360, (65, 65, 65), 1)
        cv2.line(panel, (radar_cx - radar_r, radar_cy), (radar_cx + radar_r, radar_cy), (65, 65, 65), 1)

        yaw_rad = math.radians(t1)
        reach_norm = min(1.0, max(0.2, r_grip / 32.0))
        arm_len_px = int(radar_r * reach_norm)
        tip_x = radar_cx - int(arm_len_px * math.sin(yaw_rad))
        tip_y = radar_cy - int(arm_len_px * math.cos(yaw_rad))
        cv2.line(panel, (radar_cx, radar_cy), (tip_x, tip_y), (0, 215, 255), 3, cv2.LINE_AA)
        cv2.circle(panel, (radar_cx, radar_cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, (tip_x, tip_y), 5, grip_col, -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"B={js.servo_B} deg",
            (375, 238),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 215, 255),
            1,
        )

        # -------------------------------------------------------------
        # 2. Digital Telemetry Readouts
        # -------------------------------------------------------------
        curr_step, max_steps = step_progress
        phase_y = 266
        cv2.rectangle(panel, (14, phase_y), (self.telemetry_w - 14, phase_y + 32), (42, 52, 65), -1)
        cv2.putText(
            panel,
            f"PHASE {curr_step}/{max_steps}: {phase_label}",
            (22, phase_y + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (100, 225, 255),
            1,
            cv2.LINE_AA,
        )

        box_y = 316
        cv2.putText(
            panel,
            "ESP32 SERVO ANGLES (0-180 deg):",
            (16, box_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 200, 200),
            1,
        )

        servo_rows = [
            ("B (Base / Pin 13)",     f"{js.servo_B:>3} deg  (t1:{t1:+5.1f})"),
            ("S (Shoulder / Pin 26)", f"{js.servo_S:>3} deg  (t2:{t2:+5.1f})"),
            ("E (Elbow / Pin 14)",    f"{js.servo_E:>3} deg  (t3:{t3:+5.1f})"),
            ("W (Wrist / Pin 27)",    f"{js.servo_W:>3} deg  (t4:{t4:+5.1f})"),
            ("C (Gripper / Pin 15)",  f"{js.servo_C:>3} deg  ({gripper_state.upper()})"),
        ]

        for i, (name, val) in enumerate(servo_rows):
            row_y = box_y + 20 + i * 19
            cv2.putText(panel, name, (20, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (175, 175, 175), 1)
            cv2.putText(panel, val, (172, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (80, 220, 255), 1)

        # Right column: 3D Cartesian Pose & Serial Command
        col2_x = 335
        cv2.putText(
            panel,
            "3D POSE (cm):",
            (col2_x, box_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 200, 200),
            1,
        )

        cx, cy, cz = current_xyz
        cv2.putText(panel, f"X: {cx:+6.1f} cm", (col2_x + 4, box_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (235, 235, 235), 1)
        cv2.putText(panel, f"Y: {cy:+6.1f} cm", (col2_x + 4, box_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (235, 235, 235), 1)
        cv2.putText(panel, f"Z: {cz:+6.1f} cm", (col2_x + 4, box_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (235, 235, 235), 1)

        # Serial Command Box
        ser_y = 424
        cv2.rectangle(panel, (14, ser_y), (self.telemetry_w - 14, ser_y + 26), (16, 16, 16), -1)
        cv2.putText(
            panel,
            f"ESP32 UART:  {js.serial_command()}",
            (22, ser_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 255, 80),
            1,
            cv2.LINE_AA,
        )

        # Controls Hint Bottom
        cv2.rectangle(panel, (0, self.height - 26), (self.telemetry_w, self.height), (18, 18, 18), -1)
        cv2.putText(
            panel,
            "[T] Type Object [SPACE] Move [N] Randomize [Drag] Move Obj [1-6] Presets [S] Sim/Cam",
            (8, self.height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (175, 175, 175),
            1,
        )

        return panel

    def combine_dashboard(self, camera_hud: np.ndarray, telemetry_panel: np.ndarray) -> np.ndarray:
        """Horizontally concatenates Camera HUD and Telemetry panel."""
        return np.hstack([camera_hud, telemetry_panel])
