"""
Calibration & Coordinate Transformation Module.
Maps 2D pixel coordinates (u, v) from the webcam to 3D Robot Base Frame (X, Y, Z) in cm.
"""
import numpy as np
import cv2
from typing import Tuple, Optional
import config

class WorkspaceCalibrator:
    def __init__(self, frame_w: int = config.FRAME_WIDTH, frame_h: int = config.FRAME_HEIGHT):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Default Pixel Points (Corners of table workspace in image)
        # Margin of 40px from edges
        margin_x = int(frame_w * 0.1)
        margin_y = int(frame_h * 0.1)

        self.pixel_corners = np.array([
            [margin_x, margin_y],                           # Top-Left (Far-Left in workspace)
            [frame_w - margin_x, margin_y],                 # Top-Right (Far-Right in workspace)
            [frame_w - margin_x, frame_h - margin_y],       # Bottom-Right (Near-Right in workspace)
            [margin_x, frame_h - margin_y]                  # Bottom-Left (Near-Left in workspace)
        ], dtype=np.float32)

        # Corresponding Robot Workspace Coordinates (X_robot forward, Y_robot lateral) in cm
        # When camera is facing the table:
        # Top of image corresponds to farther away (higher X)
        # Bottom of image corresponds to closer to robot (lower X)
        # Left of image corresponds to +Y (Robot's left)
        # Right of image corresponds to -Y (Robot's right)
        self.robot_corners = np.array([
            [config.TABLE_X_MAX, config.TABLE_Y_MAX],  # Top-Left: Far Left
            [config.TABLE_X_MAX, config.TABLE_Y_MIN],  # Top-Right: Far Right
            [config.TABLE_X_MIN, config.TABLE_Y_MIN],  # Bottom-Right: Near Right
            [config.TABLE_X_MIN, config.TABLE_Y_MAX]   # Bottom-Left: Near Left
        ], dtype=np.float32)

        # Compute Homography Matrix: Pixel -> Robot Plane (Z=0)
        self.H_pix2rob, _ = cv2.findHomography(self.pixel_corners, self.robot_corners)
        self.H_rob2pix, _ = cv2.findHomography(self.robot_corners, self.pixel_corners)

    def set_custom_calibration(self, pixel_pts: np.ndarray, robot_pts: np.ndarray):
        """Update calibration using user-supplied 4 corner points."""
        assert len(pixel_pts) == 4 and len(robot_pts) == 4, "Must provide 4 corresponding points."
        self.pixel_corners = np.array(pixel_pts, dtype=np.float32)
        self.robot_corners = np.array(robot_pts, dtype=np.float32)
        self.H_pix2rob, _ = cv2.findHomography(self.pixel_corners, self.robot_corners)
        self.H_rob2pix, _ = cv2.findHomography(self.robot_corners, self.pixel_corners)

    def pixel_to_robot(self, u: float, v: float, z: float = 0.0) -> Tuple[float, float, float]:
        """
        Transforms pixel coordinate (u, v) to Robot Base Frame (X, Y, Z) in cm.
        Robot Frame:
          +X: Forward (towards the workspace)
          +Y: Left
          +Z: Upwards from table surface
        """
        p = np.array([[[u, v]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, self.H_pix2rob)
        x_rob = float(transformed[0, 0, 0])
        y_rob = float(transformed[0, 0, 1])
        return round(x_rob, 2), round(y_rob, 2), round(z, 2)

    def robot_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Maps Robot Frame (X, Y) to camera pixel coordinates (u, v)."""
        p = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, self.H_rob2pix)
        u = int(round(transformed[0, 0, 0]))
        v = int(round(transformed[0, 0, 1]))
        return u, v


if __name__ == "__main__":
    calib = WorkspaceCalibrator()
    print("Testing Calibration Transformation:")
    # Center pixel
    cx, cy = config.FRAME_WIDTH / 2, config.FRAME_HEIGHT / 2
    rx, ry, rz = calib.pixel_to_robot(cx, cy, 0.0)
    print(f"Center pixel ({cx}, {cy}) -> Robot Frame: X={rx} cm, Y={ry} cm, Z={rz} cm")

    # Round trip
    u_back, v_back = calib.robot_to_pixel(rx, ry)
    print(f"Round trip -> Pixel: ({u_back}, {v_back})")
