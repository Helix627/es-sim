"""
Configuration parameters for Vision-Guided Robotic Arm Prototype.
All dimensions are in centimeters (cm) and angles in degrees unless specified.
"""
from typing import Dict

# ==========================================
# ROBOT ARM PHYSICAL SPECIFICATIONS (5-DOF)
# ==========================================
LINK_1_BASE_HEIGHT = 10.0   # L1: Table to shoulder joint (cm)
LINK_2_UPPER_ARM   = 15.0   # L2: Shoulder to elbow joint (cm)
LINK_3_FOREARM     = 15.0   # L3: Elbow to wrist joint (cm)
LINK_4_GRIPPER     = 8.0    # L4: Wrist to gripper tip (cm)

# Workspace Reach Limits
MIN_REACH = 6.0
MAX_REACH = LINK_2_UPPER_ARM + LINK_3_FOREARM + LINK_4_GRIPPER * 0.85

# Joint Limits in Degrees
JOINT_LIMITS = {
    'theta_1': (-90.0, 90.0),    # Base Yaw rotation
    'theta_2': (15.0, 165.0),    # Shoulder pitch (90 = vertical upright)
    'theta_3': (-180.0, 0.0),    # Elbow pitch (-90 = 90-deg forward bend)
    'theta_4': (-90.0, 90.0),    # Wrist pitch (0 = aligned with forearm)
}

# Default Home / Standby Joint Angles (degrees) -> corresponds to B90 S90 E90 W90 C90
HOME_ANGLES = (0.0, 90.0, -90.0, 0.0)

# Waypoint Heights relative to tabletop (Z = 0)
Z_HOVER = 12.0  # Safe transit height (cm)
Z_GRIP  = 2.5   # Grasping height (cm)
Z_PLACE = 3.5   # Placing height (cm)

# ==========================================
# COMPUTER VISION & CAMERA SPECIFICATIONS
# ==========================================
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Tabletop Workspace in Camera Field of View (Calibration defaults)
# Maps image pixels (u, v) to robot frame (X, Y) in cm
TABLE_X_MIN = 13.0   # Robot X (forward) near edge (cm)
TABLE_X_MAX = 27.0   # Robot X (forward) far edge (cm)
TABLE_Y_MIN = -14.0  # Robot Y (right side) (cm)
TABLE_Y_MAX = 14.0   # Robot Y (left side) (cm)

# HSV Color Ranges for Object Detection
# Hue: [0, 179], Saturation: [0, 255], Value: [0, 255]
COLOR_RANGES: Dict[str, list] = {
    'red': [
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (180, 255, 255))
    ],
    'blue': [
        ((98, 110, 60), (132, 255, 255))
    ],
    'green': [
        ((38, 90, 50), (85, 255, 255))
    ],
    'yellow': [
        ((20, 110, 90), (35, 255, 255))
    ],
    'orange': [
        ((11, 140, 90), (20, 255, 255))
    ],
    'white': [
        ((0, 0, 185), (180, 45, 255))
    ],
    'black': [
        ((0, 0, 10), (180, 90, 75))
    ],
    'grey': [
        ((0, 0, 75), (180, 50, 180))
    ],
}

# Contour Area Thresholds (filter noise)
MIN_CONTOUR_AREA = 500
MAX_CONTOUR_AREA = 120000

# Visualization Display Colors (BGR for OpenCV)
DISPLAY_COLORS = {
    'red': (0, 0, 255),
    'blue': (255, 110, 20),
    'green': (0, 210, 0),
    'yellow': (0, 225, 255),
    'orange': (0, 140, 255),
    'white': (240, 240, 240),
    'black': (60, 60, 60),
    'grey': (150, 150, 150),
    'cyan': (255, 255, 0),
    'gray': (120, 120, 120),
    'dark_gray': (40, 40, 40),
    'green_status': (50, 220, 100),
    'accent': (255, 180, 30),
    'target': (0, 0, 255),
    'dest': (255, 120, 0),
}
