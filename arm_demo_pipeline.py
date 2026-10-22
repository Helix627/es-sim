"""
========================================================================================
AI-POWERED VISION-GUIDED ROBOTIC ARM - COMPLETE SOFTWARE PROTOTYPE
========================================================================================
Integrates:
  1. Variable Object & Natural Language Command Understanding (Open-Vocabulary)
  2. Open-Vocabulary Object Grounding (OWLv2 + Multi-Cue Tabletop Vision)
  3. 2D-to-3D Tabletop Coordinate Estimation (Robot Base Frame X, Y, Z in cm)
  4. 5-DOF Analytical Inverse Kinematics (t1..t4 & ESP32 Servos B, S, E, W, C)
  5. Two-Stage Autonomous Workflow:
       - Stage 1 (IDENTIFY & GET READY): AI identifies any variable object, calculates
         3D coordinates & joint angles, and moves the arm to the APPROACH_PICK hover
         pose right above the target object ("READY TO MOVE").
       - Stage 2 (EXECUTE MOVE): Pressing [SPACE] executes the grasp, lift, transit,
         and placement into the destination tray.
  6. Interactive Tabletop: Type ANY object name on-screen ([T]), drag-and-drop objects
     with the mouse to variable (X, Y) coordinates, or press [N] to randomize positions.
========================================================================================
"""
import os
import sys
import time
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_venv_site = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

import cv2
import numpy as np

import config
from nlp_parser import CommandParser
from calibration import WorkspaceCalibrator
from kinematics import KinematicsSolver, JointState
from vision import VisionDetector, DetectedObject
from motion_planner import MotionPlanner, Waypoint
from visualizer import DashboardVisualizer


def run_pipeline(
    query: str = "Pick up the red bottle and place it in the blue tray.",
    image_path_str: str = "",
    webcam_index: int = 0,
    use_simulation: bool = False,
    headless: bool = False,
):
    print("=" * 108)
    print(" AI-POWERED VISION-GUIDED ROBOTIC ARM - VARIABLE OBJECT PROTOTYPE")
    print("=" * 108)
    print(f"[Command] \"{query}\"")

    cmd_parser = CommandParser()
    task = cmd_parser.parse(query)
    print(f"[NLP]     {task}")

    calibrator = WorkspaceCalibrator()
    solver = KinematicsSolver()
    planner = MotionPlanner(solver=solver)
    visualizer = DashboardVisualizer(solver=solver, calibrator=calibrator)

    static_frame = None
    sample_path = None

    if image_path_str:
        sample_path = Path(image_path_str)
        if not sample_path.exists():
            alt_path = Path("5-DOF-Robotic-Arm-Basic") / image_path_str
            if alt_path.exists():
                sample_path = alt_path
            else:
                sys.exit(f"[Error] Image not found: {image_path_str}")
        static_frame = cv2.imread(str(sample_path))
        if static_frame is None:
            sys.exit(f"[Error] Could not decode image: {sample_path}")
        print(f"[Input]   Static Image ({static_frame.shape[1]}x{static_frame.shape[0]}): {sample_path}")
        detector = VisionDetector(calibrator=calibrator, use_simulation=False, open_camera=False)
    else:
        detector = VisionDetector(
            calibrator=calibrator,
            use_simulation=use_simulation,
            camera_index=webcam_index,
        )
        mode_desc = "SYNTHETIC SIMULATION" if detector.use_simulation else f"LIVE WEBCAM (Device {webcam_index})"
        print(f"[Input]   {mode_desc}")

    print("-" * 108)

    def ensure_variable_sim_objects():
        """Ensures any user-requested variable object exists on the simulated tabletop."""
        if detector.use_simulation and static_frame is None:
            detector.ensure_sim_object_for_query(task.pick_phrase, is_receptacle=False)
            if task.place_phrase:
                detector.ensure_sim_object_for_query(task.place_phrase, is_receptacle=True)

    def detect_and_plan(frame: np.ndarray):
        t0 = time.perf_counter()
        pick_obj, candidates = detector.locate_by_query(frame, task.pick_phrase)

        place_obj = None
        if task.place_phrase:
            exclude = pick_obj.bbox if pick_obj else None
            place_obj, _ = detector.locate_by_query(frame, task.place_phrase, exclude_bbox=exclude)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\n[Vision AI] Variable Object Search for '{task.pick_phrase}' completed in {elapsed_ms:.1f} ms ({len(candidates)} objects detected):")
        for idx, cand in enumerate(candidates, start=1):
            bx0, by0, bx1, by1 = cand.box_xyxy
            rx, ry, rz = cand.robot_coords
            print(
                f"  [{idx}] score {cand.score:.2f} | label '{cand.label}' | box ({bx0:>3}, {by0:>3}, {bx1:>3}, {by1:>3}) "
                f"| 3D ({rx:+5.1f}, {ry:+5.1f}, {rz:+5.1f}) cm | {cand.reason}"
            )

        if pick_obj is None:
            print(f"[Vision AI] Could not locate '{task.pick_phrase}' in current view.")
            return None, None, candidates, None, None

        print(f"\n[AI Identified Target]  -> {pick_obj}")
        print(f"  Match Reason: {pick_obj.reason}")

        if place_obj is not None:
            print(f"[AI Identified Place]   -> {place_obj}")
            place_xyz = place_obj.robot_coords
        else:
            px, py, _ = pick_obj.robot_coords
            place_xyz = (round(min(25.0, max(15.0, px + 2.0)), 2), round(-py if abs(py) > 4.0 else -10.0, 2), 0.0)
            print(f"[Default Place Target]  -> Drop Zone @ 3D Robot Frame: X={place_xyz[0]:+.1f}cm, Y={place_xyz[1]:+.1f}cm, Z=+0.0cm")

        px, py, pz = pick_obj.robot_coords
        ready_ik = solver.inverse_kinematics(px, py, config.Z_HOVER, gripper_closed=False)
        grasp_ik = solver.inverse_kinematics(px, py, config.Z_GRIP, gripper_closed=False)
        fk_verify = solver.forward_kinematics(*grasp_ik.as_tuple())

        print("\n" + "=" * 108)
        print(" AI IDENTIFIED OBJECT -> READY TO MOVE TELEMETRY:")
        print("=" * 108)
        print(f"  Identified Variable Object            : '{pick_obj.label}' (Confidence Score: {pick_obj.score:.2f})")
        print(f"  Object 3D Position (Robot Base Frame) : X = {px:+6.2f} cm, Y = {py:+6.2f} cm, Z = {pz:+6.2f} cm")
        print(f"  Ready/Hover Pose (APPROACH_PICK)      : X = {px:+6.2f} cm, Y = {py:+6.2f} cm, Z = {config.Z_HOVER:+6.2f} cm")
        print(f"  Ready Servo Angles (ESP32)            : B={ready_ik.servo_B} deg, S={ready_ik.servo_S} deg, E={ready_ik.servo_E} deg, W={ready_ik.servo_W} deg, C={ready_ik.servo_C} deg")
        print(f"  Grasp Target Pose (DESCEND_PICK)      : X = {px:+6.2f} cm, Y = {py:+6.2f} cm, Z = {config.Z_GRIP:+6.2f} cm")
        print(f"  Grasp Joint Angles (t1..t4)           : t1={grasp_ik.theta_1:+5.1f} deg, t2={grasp_ik.theta_2:+5.1f} deg, t3={grasp_ik.theta_3:+5.1f} deg, t4={grasp_ik.theta_4:+5.1f} deg")
        print(f"  Grasp Servo Angles (0-180 deg)        : B={grasp_ik.servo_B} deg, S={grasp_ik.servo_S} deg, E={grasp_ik.servo_E} deg, W={grasp_ik.servo_W} deg, C={grasp_ik.servo_C} deg")
        print(f"  ESP32 UART Command String             : \"{grasp_ik.serial_command()}\"")
        print(f"  Forward Kinematics Verification       : X = {fk_verify[0]:+6.2f} cm, Y = {fk_verify[1]:+6.2f} cm, Z = {fk_verify[2]:+6.2f} cm")
        print("=" * 108)

        plan = planner.plan_pick_and_place(
            pick_xyz=pick_obj.robot_coords,
            place_xyz=place_xyz,
            task_label=task.raw_command,
        )
        if plan.is_valid:
            plan.print_plan_table()
            print(f"\n>>> [AI READY] Target '{pick_obj.label}' locked! Arm moving to APPROACH_PICK ready pose above X={px:+.1f}, Y={py:+.1f} cm.")
            print(">>> Press [SPACE] in the GUI window to execute the Pick & Place movement!")
        else:
            print(f"[Planner Error] {plan.error_message}")

        return pick_obj, place_obj, candidates, plan, place_xyz

    # Initial Setup & Detection
    detector.reset_sim_objects()
    ensure_variable_sim_objects()
    initial_frame = static_frame.copy() if static_frame is not None else detector.get_frame()
    pick_obj, place_obj, candidates, current_plan, place_xyz = detect_and_plan(initial_frame)

    # Save annotated snapshot showing the arm positioned at the Ready-to-Move (APPROACH_PICK) / Grasp pose
    home_js = solver.home_joint_state()
    display_wp = current_plan.waypoints[1] if (current_plan and current_plan.is_valid) else Waypoint(
        1, "STANDBY_HOME", "Home", solver.forward_kinematics(*home_js.as_tuple()), home_js, "OPEN", 1.0
    )

    cam_hud = visualizer.draw_camera_hud(
        frame=initial_frame,
        detected_objects=candidates,
        pick_obj=pick_obj,
        place_obj=place_obj,
        status_msg=f"READY TO MOVE '{pick_obj.label.upper()}' [PRESS SPACE]" if pick_obj else "SEARCHING",
        command_text=task.raw_command,
        current_angles=display_wp.joint_angles.as_tuple(),
        gripper_state=display_wp.gripper_state,
    )
    tel_panel = visualizer.draw_telemetry_panel(
        current_angles=display_wp.joint_angles.as_tuple(),
        gripper_state=display_wp.gripper_state,
        current_xyz=display_wp.target_xyz,
        phase_label=f"{display_wp.phase_name} (READY)",
        step_progress=(display_wp.step_id, 10),
        pick_xyz=pick_obj.robot_coords if pick_obj else None,
        place_xyz=place_xyz,
    )
    dashboard_img = visualizer.combine_dashboard(cam_hud, tel_panel)
    out_path = Path("demo_output_result.jpg")
    cv2.imwrite(str(out_path), dashboard_img)
    print(f"[Saved] Annotated Dashboard Image -> {out_path.resolve()}")

    if headless:
        detector.release()
        return

    window_name = "Vision-Guided Robotic Arm Prototype - Live Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n[Interactive GUI Controls - Variable Object & Two-Stage Move]")
    print("  [T] or [/]  : Type ANY variable object or command directly on-screen (e.g. 'yellow cup', 'black phone', 'orange mouse')")
    print("  [SPACE]     : Execute Pick & Place Move (once AI is READY above object) or Re-run")
    print("  [Mouse Drag]: Click & drag any object on the tabletop to variable (X, Y) coordinates!")
    print("  [N]         : Randomize tabletop object coordinates & make AI re-identify and get ready")
    print("  [1..6]      : Quick variable objects: [1] Red Bottle [2] Foam Board [3] Green Block [4] Yellow Cup [5] Black Phone [6] Desk Photo")
    print("  [S]         : Toggle Live Webcam / Synthetic Simulation Feed")
    print("  [R]         : Reset Arm & Objects to Home Pose")
    print("  [Q / ESC]   : Quit Demo")
    print("-" * 108)

    # Two-Stage Arm State Machine:
    #   "IDLE"           : Arm at home, no valid target
    #   "GETTING_READY"  : AI identified object; arm is moving to APPROACH_PICK hover pose above object
    #   "READY_TO_MOVE"  : Arm is hovering right above the identified object, waiting for [SPACE] to pick & place
    #   "EXECUTING_MOVE" : Arm is executing DESCEND_PICK -> GRASP -> LIFT -> TRANSIT -> PLACE -> HOME
    #   "COMPLETED"      : Object placed in target tray
    arm_state = "IDLE"
    plan_step_idx = 0
    target_end_idx = 0
    interp_substeps = []
    substep_idx = 0

    current_angles = home_js.as_tuple()
    current_gripper = "OPEN"
    current_xyz = solver.forward_kinematics(*current_angles)
    status_msg = "SEARCHING FOR OBJECT"

    typing_mode = False
    typing_buffer = ""
    dragging_sim_key = None

    def start_get_ready():
        """Stage 1: Move arm from current pose to Waypoint 1 (APPROACH_PICK hover right above identified object)."""
        nonlocal arm_state, plan_step_idx, target_end_idx, interp_substeps, substep_idx, status_msg
        if current_plan and current_plan.is_valid and pick_obj is not None:
            arm_state = "GETTING_READY"
            plan_step_idx = 1  # Waypoint index 1 is APPROACH_PICK (hover above target)
            target_end_idx = 1
            curr_js = JointState(*current_angles, gripper_closed=(current_gripper == "CLOSED"))
            start_wp = Waypoint(0, "CURRENT", "Current", current_xyz, curr_js, current_gripper, 0.5)
            interp_substeps = planner.interpolate_trajectory(start_wp, current_plan.waypoints[1], steps=16)
            substep_idx = 0
            status_msg = f"AI LOCKED '{pick_obj.label.upper()}' -> MOVING TO READY POSE"
        else:
            arm_state = "IDLE"
            status_msg = f"NOT FOUND: '{task.pick_phrase}' (PRESS [T] TO TYPE OR [S] FOR SIM)"

    def start_execute_move():
        """Stage 2: Execute pick-and-place from APPROACH_PICK through RETURN_HOME."""
        nonlocal arm_state, plan_step_idx, target_end_idx, interp_substeps, substep_idx, status_msg
        if current_plan and current_plan.is_valid and pick_obj is not None:
            arm_state = "EXECUTING_MOVE"
            plan_step_idx = 2  # Waypoint index 2 is DESCEND_PICK
            target_end_idx = len(current_plan.waypoints) - 1
            curr_js = JointState(*current_angles, gripper_closed=(current_gripper == "CLOSED"))
            start_wp = Waypoint(1, "READY", "Ready", current_xyz, curr_js, current_gripper, 0.5)
            interp_substeps = planner.interpolate_trajectory(start_wp, current_plan.waypoints[2], steps=14)
            substep_idx = 0
            status_msg = f"MOVING '{pick_obj.label.upper()}' TO TARGET"

    # Mouse callback for dragging objects to variable positions on the tabletop
    def on_mouse(event, x, y, flags, param):
        nonlocal dragging_sim_key, pick_obj, place_obj, candidates, current_plan, place_xyz
        if not detector.use_simulation or static_frame is not None or x >= config.FRAME_WIDTH:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            hit_key = detector.get_sim_object_at_pixel(x, y)
            if hit_key:
                dragging_sim_key = hit_key
        elif event == cv2.EVENT_MOUSEMOVE and dragging_sim_key is not None:
            detector.move_sim_object(dragging_sim_key, x, y)
        elif event == cv2.EVENT_LBUTTONUP and dragging_sim_key is not None:
            detector.move_sim_object(dragging_sim_key, x, y)
            dragging_sim_key = None
            frame_now = detector.get_frame()
            pick_obj, place_obj, candidates, current_plan, place_xyz = detect_and_plan(frame_now)
            start_get_ready()

    cv2.setMouseCallback(window_name, on_mouse)

    # Trigger Stage 1 (Identify & Get Ready) on startup
    start_get_ready()

    live_frame_counter = 0

    try:
        while True:
            live_frame_counter += 1

            phase_label = "STANDBY_HOME"
            step_progress = (1, 10)

            if arm_state in ("GETTING_READY", "EXECUTING_MOVE") and current_plan and current_plan.is_valid:
                step_progress = (plan_step_idx + 1, len(current_plan.waypoints))
                wp = current_plan.waypoints[plan_step_idx]
                phase_label = wp.phase_name

                if substep_idx < len(interp_substeps):
                    t1, t2, t3, t4, grip = interp_substeps[substep_idx]
                    current_angles = (t1, t2, t3, t4)
                    current_gripper = grip
                    current_xyz = solver.forward_kinematics(t1, t2, t3, t4)
                    substep_idx += 1

                    # Move grasped object along with the claw in simulation mode
                    if detector.use_simulation and static_frame is None:
                        detector.update_sim_carried_object(
                            pick_phrase=task.pick_phrase,
                            gripper_closed=(current_gripper == "CLOSED"),
                            arm_xyz=current_xyz,
                        )
                        if pick_obj is not None and current_gripper == "CLOSED":
                            u_c, v_c = calibrator.robot_to_pixel(current_xyz[0], current_xyz[1])
                            _, _, pw, ph = pick_obj.bbox
                            pick_obj.center_px = (u_c, v_c)
                            pick_obj.bbox = (u_c - pw // 2, v_c - ph // 2, pw, ph)
                            pick_obj.robot_coords = current_xyz
                else:
                    if arm_state == "GETTING_READY":
                        # Reached APPROACH_PICK hover position! Pause and wait for user to trigger move
                        arm_state = "READY_TO_MOVE"
                        obj_name = pick_obj.label.upper() if pick_obj else "OBJECT"
                        status_msg = f"READY TO MOVE '{obj_name}' - PRESS [SPACE] TO PICK & PLACE"
                    elif arm_state == "EXECUTING_MOVE":
                        plan_step_idx += 1
                        if plan_step_idx <= target_end_idx:
                            next_wp = current_plan.waypoints[plan_step_idx]
                            interp_substeps = planner.interpolate_trajectory(wp, next_wp, steps=15)
                            substep_idx = 0
                        else:
                            arm_state = "COMPLETED"
                            detector.carried_sim_key = None
                            status_msg = "OBJECT PLACED! PRESS [T] FOR NEW OBJECT OR [N] RANDOMIZE"
                            print("\n[Pipeline] Pick-and-Place complete! Object placed in destination.")
            elif arm_state == "READY_TO_MOVE":
                phase_label = "APPROACH_PICK (READY)"
                step_progress = (2, 10)
            elif arm_state == "COMPLETED":
                phase_label = "COMPLETED (AT HOME)"
                step_progress = (10, 10)

            # Acquire frame AFTER updating simulation object position
            frame = static_frame.copy() if static_frame is not None else detector.get_frame()

            # On live webcam (when not executing a move), continuously track variable object every 20 frames
            if (
                static_frame is None
                and not detector.use_simulation
                and arm_state not in ("EXECUTING_MOVE", "GETTING_READY")
                and (live_frame_counter % 20 == 0)
            ):
                new_pick, new_cands = detector.locate_by_query(frame, task.pick_phrase)
                candidates = new_cands
                if new_pick is not None:
                    # If target moved significantly on live webcam, update plan and ready position
                    moved_dist = (
                        np.hypot(
                            new_pick.robot_coords[0] - pick_obj.robot_coords[0],
                            new_pick.robot_coords[1] - pick_obj.robot_coords[1],
                        )
                        if pick_obj is not None
                        else 99.0
                    )
                    pick_obj = new_pick
                    if task.place_phrase:
                        place_obj, _ = detector.locate_by_query(frame, task.place_phrase, exclude_bbox=pick_obj.bbox)
                    if moved_dist > 1.8 or current_plan is None:
                        pick_obj, place_obj, candidates, current_plan, place_xyz = detect_and_plan(frame)
                        start_get_ready()

            cam_hud = visualizer.draw_camera_hud(
                frame=frame,
                detected_objects=candidates,
                pick_obj=pick_obj,
                place_obj=place_obj,
                status_msg=status_msg,
                command_text=task.raw_command,
                current_angles=current_angles,
                gripper_state=current_gripper,
                typing_mode=typing_mode,
                typing_buffer=typing_buffer,
            )
            tel_panel = visualizer.draw_telemetry_panel(
                current_angles=current_angles,
                gripper_state=current_gripper,
                current_xyz=current_xyz,
                phase_label=phase_label,
                step_progress=step_progress,
                pick_xyz=pick_obj.robot_coords if pick_obj else None,
                place_xyz=place_xyz,
            )
            dashboard = visualizer.combine_dashboard(cam_hud, tel_panel)
            cv2.imshow(window_name, dashboard)

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            # Handle On-Screen Typing Mode for Variable Objects/Commands
            if typing_mode:
                if key == 27:  # ESC cancels typing
                    typing_mode = False
                    typing_buffer = ""
                elif key in (13, 10):  # ENTER submits variable object/command
                    typing_mode = False
                    new_cmd = typing_buffer.strip()
                    typing_buffer = ""
                    if new_cmd:
                        # If user typed just an object name like "yellow cup", default to placing in blue tray
                        if "place" not in new_cmd.lower() and "put" not in new_cmd.lower() and " to " not in new_cmd.lower():
                            full_cmd = f"Pick up the {new_cmd} and place it in the blue tray."
                        else:
                            full_cmd = new_cmd
                        task = cmd_parser.parse(full_cmd)
                        print(f"\n[Variable Query] {task}")
                        ensure_variable_sim_objects()
                        frame = static_frame.copy() if static_frame is not None else detector.get_frame()
                        pick_obj, place_obj, candidates, current_plan, place_xyz = detect_and_plan(frame)
                        start_get_ready()
                elif key in (8, 127):  # Backspace
                    typing_buffer = typing_buffer[:-1]
                elif 32 <= key <= 126 and len(typing_buffer) < 52:
                    typing_buffer += chr(key)
                continue

            # Normal Hotkeys
            if key in (ord("q"), ord("Q"), 27):
                break

            elif key in (ord("t"), ord("T"), ord("/")):
                typing_mode = True
                typing_buffer = ""

            elif key in (ord(" "), ord("m"), ord("M")):
                if arm_state == "READY_TO_MOVE":
                    # Stage 2: Execute the Pick & Place move!
                    start_execute_move()
                else:
                    # If completed or idle, re-identify and get ready (or if already at home, get ready first)
                    if detector.use_simulation and static_frame is None and arm_state == "COMPLETED":
                        detector.reset_sim_objects()
                        ensure_variable_sim_objects()
                    frame = static_frame.copy() if static_frame is not None else detector.get_frame()
                    pick_obj, place_obj, candidates, current_plan, place_xyz = detect_and_plan(frame)
                    start_get_ready()

            elif key in (ord("n"), ord("N")):
                # Randomize tabletop object positions so coordinates are completely variable
                static_frame = None
                detector.use_simulation = True
                detector.randomize_sim_objects()
                ensure_variable_sim_objects()
                print("\n[Randomize] Randomized tabletop object coordinates!")
                frame = detector.get_frame()
                pick_obj, place_obj, candidates, current_plan, place_xyz = detect_and_plan(frame)
                start_get_ready()

            elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                preset_map = {
                    ord("1"): "Pick up the red bottle and place it in the blue tray.",
                    ord("2"): "Pick up the white mixed foam board and place it in the blue tray.",
                    ord("3"): "Pick up the green block and place it in the blue tray.",
                    ord("4"): "Pick up the yellow cup and place it in the blue tray.",
                    ord("5"): "Pick up the black phone and place it in the blue tray.",
                }
                static_frame = None
                detector.use_simulation = True
                ensure_variable_sim_objects()
                task = cmd_parser.parse(preset_map[key])
                print(f"\n[Variable Target] Switched target object: {task}")
                frame = detector.get_frame()
                pick_obj, place_obj, candidates, current_plan, place_xyz = detect_and_plan(frame)
                start_get_ready()

            elif key == ord("6"):
                p_path = Path("5-DOF-Robotic-Arm-Basic") / "foam.png"
                if p_path.exists():
                    static_frame = cv2.imread(str(p_path))
                    detector.use_simulation = False
                    task = cmd_parser.parse("Pick up the white mixed foam board and place it on the white breadboard")
                    print(f"\n[Preset] Loaded Real Desk Photo {p_path}: {task}")
                    pick_obj, place_obj, candidates, current_plan, place_xyz = detect_and_plan(static_frame)
                    start_get_ready()

            elif key in (ord("r"), ord("R")):
                arm_state = "IDLE"
                detector.reset_sim_objects()
                ensure_variable_sim_objects()
                current_angles = home_js.as_tuple()
                current_gripper = "OPEN"
                current_xyz = solver.forward_kinematics(*current_angles)
                frame = static_frame.copy() if static_frame is not None else detector.get_frame()
                pick_obj, place_obj, candidates, current_plan, place_xyz = detect_and_plan(frame)
                start_get_ready()
                print("\n[Control] Reset tabletop objects and moved arm to Ready Pose.")

            elif key in (ord("s"), ord("S")):
                static_frame = None
                detector.use_simulation = not detector.use_simulation
                if detector.use_simulation:
                    detector.reset_sim_objects()
                    ensure_variable_sim_objects()
                elif detector.cap is None:
                    detector._init_camera()
                mode_str = "SYNTHETIC SIMULATION" if detector.use_simulation else "LIVE WEBCAM"
                print(f"\n[Input] Switched to {mode_str}")
                frame = detector.get_frame()
                pick_obj, place_obj, candidates, current_plan, place_xyz = detect_and_plan(frame)
                start_get_ready()

            elif key in (ord("c"), ord("C")):
                print("\n" + "=" * 60)
                print("Enter any variable object or command (e.g., 'orange calculator', 'Pick up the yellow cup'):")
                try:
                    new_cmd = input("Command >> ").strip()
                    if new_cmd:
                        if "place" not in new_cmd.lower() and "put" not in new_cmd.lower() and " to " not in new_cmd.lower():
                            new_cmd = f"Pick up the {new_cmd} and place it in the blue tray."
                        task = cmd_parser.parse(new_cmd)
                        print(f"[NLP] {task}")
                        ensure_variable_sim_objects()
                        frame = static_frame.copy() if static_frame is not None else detector.get_frame()
                        pick_obj, place_obj, candidates, current_plan, place_xyz = detect_and_plan(frame)
                        start_get_ready()
                except Exception as e:
                    print(f"[Input Error] {e}")
    finally:
        detector.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="AI-Powered Vision-Guided Robotic Arm Software Prototype")
    ap.add_argument(
        "--query",
        "--command",
        dest="query",
        type=str,
        default="Pick up the red bottle and place it in the blue tray.",
        help="Any variable natural language command or object query",
    )
    ap.add_argument("--image", type=str, default="", help="Optional static image path (e.g. 5-DOF-Robotic-Arm-Basic/foam.png)")
    ap.add_argument("--webcam", type=int, default=0, help="Webcam device index (default 0)")
    ap.add_argument("--sim", action="store_true", help="Start in synthetic tabletop simulation mode")
    ap.add_argument("--headless", action="store_true", help="Run pipeline and print results without opening GUI window")
    args = ap.parse_args()

    run_pipeline(
        query=args.query,
        image_path_str=args.image,
        webcam_index=args.webcam,
        use_simulation=args.sim,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
