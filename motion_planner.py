"""
Motion Planner & Trajectory Generator for Robotic Pick-and-Place tasks.
Generates waypoints, validates kinematic limits, interpolates joint trajectories,
and formats step-by-step movement plans with ESP32 servo commands.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import config
from kinematics import KinematicsSolver, JointState


@dataclass
class Waypoint:
    step_id: int
    phase_name: str
    description: str
    target_xyz: Tuple[float, float, float]       # (X, Y, Z) in cm
    joint_angles: JointState                    # (theta_1, theta_2, theta_3, theta_4) + servos
    gripper_state: str                          # "OPEN" or "CLOSED"
    duration_s: float                           # estimated phase execution time

    def format_row(self) -> str:
        x, y, z = self.target_xyz
        if self.joint_angles.is_reachable:
            t1, t2, t3, t4 = self.joint_angles.as_rounded(1)
            angles_str = f"[{t1:+5.1f}, {t2:+5.1f}, {t3:+5.1f}, {t4:+5.1f}]"
            serial_str = self.joint_angles.serial_command()
        else:
            angles_str = "[UNREACHABLE]               "
            serial_str = "ERR_UNREACHABLE     "

        return (
            f"| {self.step_id:>2} | {self.phase_name:<14} | "
            f"({x:>5.1f}, {y:>5.1f}, {z:>5.1f}) | "
            f"{angles_str:<28} | {serial_str:<20} | {self.gripper_state:<6} |"
        )


@dataclass
class TrajectoryPlan:
    task_name: str
    waypoints: List[Waypoint]
    is_valid: bool = True
    error_message: Optional[str] = None
    total_duration_s: float = 0.0

    def print_plan_table(self):
        print("\n" + "=" * 108)
        print(f" MOTION PLAN: {self.task_name.upper()}")
        print("=" * 108)
        header = (
            f"| {'#':>2} | {'Phase':<14} | {'Target XYZ (cm)':<19} | "
            f"{'Joint Angles (t1,t2,t3,t4)°':<28} | {'ESP32 Servos (BSEWC)':<20} | {'Grip':<6} |"
        )
        print(header)
        print("-" * 108)
        for wp in self.waypoints:
            print(wp.format_row())
        print("=" * 108)
        print(
            f"Total Trajectory Phases: {len(self.waypoints)} | "
            f"Est. Duration: {self.total_duration_s:.1f}s | "
            f"Status: {'VALID' if self.is_valid else 'FAILED'}\n"
        )


class MotionPlanner:
    def __init__(self, solver: Optional[KinematicsSolver] = None):
        self.solver = solver or KinematicsSolver()

    def plan_pick_and_place(
        self,
        pick_xyz: Tuple[float, float, float],
        place_xyz: Tuple[float, float, float],
        task_label: str = "Pick and Place",
    ) -> TrajectoryPlan:
        """
        Generates full 10-phase motion plan from Pick coordinates to Place coordinates.
        """
        px, py, _ = pick_xyz
        lx, ly, _ = place_xyz

        z_hover = config.Z_HOVER
        z_pick = config.Z_GRIP
        z_place = config.Z_PLACE

        home_joints = self.solver.home_joint_state()
        home_fk = self.solver.forward_kinematics(*home_joints.as_tuple())

        pick_hover_open = self.solver.inverse_kinematics(px, py, z_hover, gripper_closed=False)
        pick_grip_open = self.solver.inverse_kinematics(px, py, z_pick, gripper_closed=False)
        pick_grip_closed = self.solver.inverse_kinematics(px, py, z_pick, gripper_closed=True)
        pick_hover_closed = self.solver.inverse_kinematics(px, py, z_hover, gripper_closed=True)

        place_hover_closed = self.solver.inverse_kinematics(lx, ly, z_hover, gripper_closed=True)
        place_drop_closed = self.solver.inverse_kinematics(lx, ly, z_place, gripper_closed=True)
        place_drop_open = self.solver.inverse_kinematics(lx, ly, z_place, gripper_closed=False)
        place_hover_open = self.solver.inverse_kinematics(lx, ly, z_hover, gripper_closed=False)

        for name, js in [
            ("Pick Hover", pick_hover_open),
            ("Pick Grip", pick_grip_open),
            ("Place Hover", place_hover_closed),
            ("Place Drop", place_drop_closed),
        ]:
            if not js.is_reachable:
                return TrajectoryPlan(
                    task_name=task_label,
                    waypoints=[],
                    is_valid=False,
                    error_message=f"Kinematic constraint violation at '{name}': {js.error_msg}",
                )

        definitions = [
            (1,  "STANDBY_HOME",  "Arm in initial standby home pose",     home_fk,            home_joints,        "OPEN",   1.0),
            (2,  "APPROACH_PICK", "Move to hover height above object",    (px, py, z_hover),  pick_hover_open,    "OPEN",   1.5),
            (3,  "DESCEND_PICK",  "Descend vertically towards object",    (px, py, z_pick),   pick_grip_open,     "OPEN",   1.0),
            (4,  "GRASP",         "Actuate gripper to grasp object",      (px, py, z_pick),   pick_grip_closed,   "CLOSED", 0.8),
            (5,  "LIFT_OBJECT",   "Ascend with object to safe clearance", (px, py, z_hover),  pick_hover_closed,  "CLOSED", 1.0),
            (6,  "TRANSIT",       "Horizontal transfer to destination",   (lx, ly, z_hover),  place_hover_closed, "CLOSED", 2.0),
            (7,  "DESCEND_PLACE", "Lower object into destination tray",   (lx, ly, z_place),  place_drop_closed,  "CLOSED", 1.0),
            (8,  "RELEASE",       "Open gripper to release object",       (lx, ly, z_place),  place_drop_open,    "OPEN",   0.8),
            (9,  "RETRACT",       "Ascend clear of destination tray",     (lx, ly, z_hover),  place_hover_open,   "OPEN",   1.0),
            (10, "RETURN_HOME",   "Return to standby home pose",          home_fk,            home_joints,        "OPEN",   1.5),
        ]

        waypoints = [
            Waypoint(
                step_id=step,
                phase_name=name,
                description=desc,
                target_xyz=xyz,
                joint_angles=js,
                gripper_state=grip,
                duration_s=dur,
            )
            for (step, name, desc, xyz, js, grip, dur) in definitions
        ]

        total_time = sum(wp.duration_s for wp in waypoints)
        return TrajectoryPlan(
            task_name=task_label,
            waypoints=waypoints,
            is_valid=True,
            total_duration_s=total_time,
        )

    def interpolate_trajectory(
        self,
        wp_start: Waypoint,
        wp_end: Waypoint,
        steps: int = 15,
    ) -> List[Tuple[float, float, float, float, str]]:
        """
        Smooth cosine S-curve joint-space interpolation between two waypoints.
        Returns list of (theta_1, theta_2, theta_3, theta_4, gripper_state).
        """
        s_angles = np.array(wp_start.joint_angles.as_tuple())
        e_angles = np.array(wp_end.joint_angles.as_tuple())

        interpolated = []
        for i in range(steps + 1):
            alpha = i / float(steps)
            smooth_alpha = 0.5 * (1.0 - np.cos(alpha * np.pi))
            current_angles = s_angles + smooth_alpha * (e_angles - s_angles)
            grip = wp_end.gripper_state if alpha >= 0.5 else wp_start.gripper_state
            interpolated.append((
                float(current_angles[0]),
                float(current_angles[1]),
                float(current_angles[2]),
                float(current_angles[3]),
                grip,
            ))
        return interpolated
