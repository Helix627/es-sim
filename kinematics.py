"""
Forward and Analytical Inverse Kinematics for the 5-DOF Robotic Arm.
Matches the physical ESP32 servo hardware in `5-DOF-Robotic-Arm-Basic/base_test/base_test.ino`:
  - B (Base, Pin 13):     0..180 deg (90 = straight forward +X)
  - S (Shoulder, Pin 26): 15..165 deg (90 = vertical upright +Z)
  - E (Elbow, Pin 14):    0..180 deg (90 = 90-deg forward bend from upper arm)
  - W (Wrist, Pin 27):    0..180 deg (90 = straight extension of forearm)
  - C (Gripper, Pin 15):  20..160 deg (20 = closed/grasped, 90 = open)
"""
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import config


@dataclass
class JointState:
    theta_1: float  # Base Yaw (deg): 0 = +X forward, + = +Y left, - = -Y right
    theta_2: float  # Shoulder Pitch (deg from +X horizontal): 90 = vertical up
    theta_3: float  # Elbow Pitch (deg relative to upper arm): -90 = 90-deg forward bend
    theta_4: float  # Wrist Pitch (deg relative to forearm): 0 = aligned with forearm
    gripper_closed: bool = False
    is_reachable: bool = True
    error_msg: Optional[str] = None

    @property
    def servo_B(self) -> int:
        return max(0, min(180, int(round(90.0 - self.theta_1))))

    @property
    def servo_S(self) -> int:
        return max(15, min(165, int(round(self.theta_2))))

    @property
    def servo_E(self) -> int:
        return max(0, min(180, int(round(180.0 + self.theta_3))))

    @property
    def servo_W(self) -> int:
        return max(0, min(180, int(round(90.0 + self.theta_4))))

    @property
    def servo_C(self) -> int:
        return 20 if self.gripper_closed else 90

    def servos_dict(self) -> Dict[str, int]:
        return {
            "B": self.servo_B,
            "S": self.servo_S,
            "E": self.servo_E,
            "W": self.servo_W,
            "C": self.servo_C,
        }

    def serial_command(self) -> str:
        return f"B{self.servo_B} S{self.servo_S} E{self.servo_E} W{self.servo_W} C{self.servo_C}"

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.theta_1, self.theta_2, self.theta_3, self.theta_4)

    def as_rounded(self, decimals: int = 1) -> Tuple[float, float, float, float]:
        return (
            round(self.theta_1, decimals),
            round(self.theta_2, decimals),
            round(self.theta_3, decimals),
            round(self.theta_4, decimals),
        )

    def __str__(self) -> str:
        if not self.is_reachable:
            return f"UNREACHABLE: {self.error_msg}"
        t1, t2, t3, t4 = self.as_rounded(1)
        return (
            f"Joints[t1:{t1:+5.1f} deg, t2:{t2:+5.1f} deg, t3:{t3:+5.1f} deg, t4:{t4:+5.1f} deg] "
            f"| Servos[{self.serial_command()}]"
        )


@dataclass
class ArmJointPositions3D:
    base: Tuple[float, float, float]
    shoulder: Tuple[float, float, float]
    elbow: Tuple[float, float, float]
    wrist: Tuple[float, float, float]
    gripper: Tuple[float, float, float]


class KinematicsSolver:
    def __init__(
        self,
        l1: float = config.LINK_1_BASE_HEIGHT,
        l2: float = config.LINK_2_UPPER_ARM,
        l3: float = config.LINK_3_FOREARM,
        l4: float = config.LINK_4_GRIPPER,
    ):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4

    @staticmethod
    def home_joint_state() -> JointState:
        """
        Matches ESP32 ANGLE_HOME = {90, 90, 90, 90, 90}:
          B=90 -> theta_1 = 0 deg
          S=90 -> theta_2 = 90 deg (vertical upper arm)
          E=90 -> theta_3 = -90 deg (horizontal forearm)
          W=90 -> theta_4 = 0 deg (horizontal gripper)
        """
        return JointState(0.0, 90.0, -90.0, 0.0, gripper_closed=False, is_reachable=True)

    def inverse_kinematics(
        self,
        x: float,
        y: float,
        z: float,
        pitch_deg: float = -75.0,
        elbow_up: bool = True,
        gripper_closed: bool = False,
    ) -> JointState:
        """
        Analytical Inverse Kinematics with adaptive wrist pitch fallback.
        Tries preferred grasp pitch first (e.g., -75 deg top-down angled grasp),
        and if out of reach, searches candidate pitches from -90 deg to -15 deg.
        """
        theta_1_rad = math.atan2(y, x)
        theta_1_deg = math.degrees(theta_1_rad)
        r = math.hypot(x, y)

        # Candidate tool pitches to try (preferred pitch first, then adaptive angles)
        candidate_pitches = [pitch_deg] + [p for p in range(-90, 0, 5) if p != pitch_deg]

        max_span = self.l2 + self.l3
        min_span = abs(self.l2 - self.l3)

        for phi_deg in candidate_pitches:
            phi_rad = math.radians(phi_deg)
            r_w = r - self.l4 * math.cos(phi_rad)
            z_w = z - self.l1 - self.l4 * math.sin(phi_rad)

            if r_w < 0:
                continue

            d_sq = r_w * r_w + z_w * z_w
            d = math.sqrt(d_sq)

            if min_span <= d <= max_span:
                cos_t3 = (d_sq - self.l2 * self.l2 - self.l3 * self.l3) / (2.0 * self.l2 * self.l3)
                cos_t3 = max(-1.0, min(1.0, cos_t3))
                t3_mag = math.acos(cos_t3)
                t3_rad = -t3_mag if elbow_up else t3_mag

                gamma = math.atan2(
                    self.l3 * math.sin(t3_rad),
                    self.l2 + self.l3 * math.cos(t3_rad),
                )
                alpha = math.atan2(z_w, r_w)
                t2_rad = alpha - gamma
                t4_rad = phi_rad - (t2_rad + t3_rad)

                t2_deg = math.degrees(t2_rad)
                t3_deg = math.degrees(t3_rad)
                t4_deg = math.degrees(t4_rad)

                # Verify physical servo ranges: S in [15, 165], E in [0, 180], W in [0, 180]
                if 15.0 <= t2_deg <= 165.0 and -180.0 <= t3_deg <= 0.0 and -90.0 <= t4_deg <= 90.0:
                    return JointState(
                        theta_1=theta_1_deg,
                        theta_2=t2_deg,
                        theta_3=t3_deg,
                        theta_4=t4_deg,
                        gripper_closed=gripper_closed,
                        is_reachable=True,
                    )

        # If target is slightly beyond max span, project onto reachable boundary
        total_max = self.l2 + self.l3 + self.l4 * 0.95
        dist_from_shoulder = math.hypot(r, z - self.l1)
        if dist_from_shoulder > total_max and dist_from_shoulder > 1e-6:
            scale = (total_max * 0.95) / dist_from_shoulder
            r_clamped = r * scale
            z_clamped = self.l1 + (z - self.l1) * scale
            x_c = r_clamped * math.cos(theta_1_rad)
            y_c = r_clamped * math.sin(theta_1_rad)
            clamped_state = self.inverse_kinematics(
                x_c, y_c, z_clamped, pitch_deg=-35.0, elbow_up=elbow_up, gripper_closed=gripper_closed
            )
            if clamped_state.is_reachable:
                return clamped_state

        return JointState(
            theta_1_deg,
            90.0,
            -90.0,
            0.0,
            gripper_closed=gripper_closed,
            is_reachable=False,
            error_msg=f"Target ({x:.1f}, {y:.1f}, {z:.1f}) cm exceeds arm workspace ({total_max:.1f} cm)",
        )

    def forward_kinematics(self, t1: float, t2: float, t3: float, t4: float) -> Tuple[float, float, float]:
        """Calculates 3D end-effector coordinate (X, Y, Z) in cm from joint angles in degrees."""
        pts = self.get_3d_joint_positions(t1, t2, t3, t4)
        gx, gy, gz = pts.gripper
        return round(gx, 2), round(gy, 2), round(gz, 2)

    def get_3d_joint_positions(self, t1: float, t2: float, t3: float, t4: float) -> ArmJointPositions3D:
        """Returns 3D (x, y, z) coordinates for Base, Shoulder, Elbow, Wrist, and Gripper tip."""
        t1_r = math.radians(t1)
        t2_r = math.radians(t2)
        t3_r = math.radians(t3)
        t4_r = math.radians(t4)

        p_base = (0.0, 0.0, 0.0)
        p_shoulder = (0.0, 0.0, self.l1)

        r_elbow = self.l2 * math.cos(t2_r)
        z_elbow = self.l1 + self.l2 * math.sin(t2_r)
        p_elbow = (r_elbow * math.cos(t1_r), r_elbow * math.sin(t1_r), z_elbow)

        r_wrist = r_elbow + self.l3 * math.cos(t2_r + t3_r)
        z_wrist = z_elbow + self.l3 * math.sin(t2_r + t3_r)
        p_wrist = (r_wrist * math.cos(t1_r), r_wrist * math.sin(t1_r), z_wrist)

        phi = t2_r + t3_r + t4_r
        r_grip = r_wrist + self.l4 * math.cos(phi)
        z_grip = z_wrist + self.l4 * math.sin(phi)
        p_gripper = (r_grip * math.cos(t1_r), r_grip * math.sin(t1_r), z_grip)

        return ArmJointPositions3D(
            base=p_base,
            shoulder=p_shoulder,
            elbow=p_elbow,
            wrist=p_wrist,
            gripper=p_gripper,
        )


if __name__ == "__main__":
    solver = KinematicsSolver()
    home = solver.home_joint_state()
    print("Home State:", home)
    print("Home FK XYZ:", solver.forward_kinematics(*home.as_tuple()))

    for pt in [(18.0, 5.0, 2.5), (24.0, -8.0, 2.5), (30.0, 10.0, 2.5)]:
        ik = solver.inverse_kinematics(*pt)
        fk = solver.forward_kinematics(*ik.as_tuple())
        print(f"Target {pt} -> {ik} -> FK {fk}")
