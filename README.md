# Vision-Guided 5-DOF Robotic Arm — Software Prototype & Demo Guide

## 1. Objective & What This Prototype Demonstrates
This software prototype implements the complete perception-to-control pipeline for your **5-DOF Vision-Guided Robotic Arm** (aligned with [`base_test.ino`](file:///c:/Users/JESTA/OneDrive/Documents/Btech_stuffs/es/5-DOF-Robotic-Arm-Basic/base_test/base_test.ino) and [`arm_demo_pipeline.py`](file:///c:/Users/JESTA/OneDrive/Documents/Btech_stuffs/es/arm_demo_pipeline.py)):

1. **Variable Object & Natural Language Command Parsing ([`nlp_parser.py`](file:///c:/Users/JESTA/OneDrive/Documents/Btech_stuffs/es/nlp_parser.py))**:
   - Understands **any variable object query** or full pick-and-place command (e.g., *"Pick up the red bottle and place it in the blue tray"*, *"white mixed foam board"*, *"yellow cup"*, *"black phone"*, *"orange calculator"*, etc.).
2. **AI Open-Vocabulary Object Grounding ([`vision.py`](file:///c:/Users/JESTA/OneDrive/Documents/Btech_stuffs/es/vision.py))**:
   - Combines **OWLv2 (`google/owlv2-base-patch16-ensemble`)** open-vocabulary vision transformer with multi-cue tabletop segmentation (LAB background distance + HSV color/material + texture profiling) and optional GPT-4o VLM indexing.
   - Dynamically spawns any custom-named variable object onto the simulated tabletop if not already present.
3. **2D-to-3D Coordinate Estimation ([`calibration.py`](file:///c:/Users/JESTA/OneDrive/Documents/Btech_stuffs/es/calibration.py))**:
   - Projects 2D pixel centroids $(u, v)$ onto the physical tabletop plane in the **Robot Base Frame $(X, Y, Z)$ in centimeters**.
4. **5-DOF Analytical Inverse Kinematics & ESP32 Mapping ([`kinematics.py`](file:///c:/Users/JESTA/OneDrive/Documents/Btech_stuffs/es/kinematics.py))**:
   - Solves geometric joint angles $(\theta_1, \theta_2, \theta_3, \theta_4)$ and maps them directly to the 5 ESP32 servos (`B` Pin 13, `S` Pin 26, `E` Pin 14, `W` Pin 27, `C` Pin 15) with Forward Kinematics verification.
5. **Two-Stage Autonomous Workflow ([`arm_demo_pipeline.py`](file:///c:/Users/JESTA/OneDrive/Documents/Btech_stuffs/es/arm_demo_pipeline.py) & [`visualizer.py`](file:///c:/Users/JESTA/OneDrive/Documents/Btech_stuffs/es/visualizer.py))**:
   - **Stage 1 — AI Identifies & Gets Ready to Move**: As soon as a variable object is specified (or moved), the AI locates it, prints its 3D coordinates and servo angles, and **automatically moves the robotic arm from Home to the `APPROACH_PICK` hover position right above the target object (`READY TO MOVE`)** with the gripper open.
   - **Stage 2 — Execute Pick & Place**: Pressing **`[SPACE]`** executes the grasp, lift, transit, placement into the target tray, and return to Home.

---

## 2. How to Run Tomorrow's Demo

Use the local virtual environment (`.venv`) where all dependencies (`opencv-python`, `numpy`, `scipy`, `torch`, `transformers`, `openai`, `pillow`) are installed:

### A. Interactive Tabletop Simulation (Variable Objects + Live Arm Animation)
```powershell
.\.venv\Scripts\python.exe arm_demo_pipeline.py --sim
```
Or pass any variable object directly on the command line:
```powershell
.\.venv\Scripts\python.exe arm_demo_pipeline.py --sim --query "Pick up the yellow cup and place it in the blue tray"
```

### B. Live Laptop Webcam Mode
```powershell
.\.venv\Scripts\python.exe arm_demo_pipeline.py --webcam 0 --query "Pick up the red bottle and place it in the blue tray"
```

### C. Real Cluttered Desk Scene Photos (`foam.png` / `test.png`)
```powershell
.\.venv\Scripts\python.exe arm_demo_pipeline.py --image 5-DOF-Robotic-Arm-Basic/foam.png --query "Pick up the white mixed foam board and place it on the white breadboard"
```

---

## 3. Interactive GUI Controls (Inside the OpenCV Window)

| Key / Action | Description |
| :--- | :--- |
| **`[T]` or `[/]`** | **Type ANY variable object or command directly on-screen** (e.g., `yellow cup`, `black phone`, `orange calculator`, `green block`), then press **`[ENTER]`** — the AI identifies it and moves the arm to the **Ready-to-Move** hover position above it! |
| **`[SPACE]`** | **Execute Pick & Place Move** once the arm is hovering in `READY TO MOVE` state above the identified object |
| **Mouse Drag & Drop** | Click and drag **any object** on the tabletop to a new $(X, Y)$ position — when you release the mouse, the AI re-identifies its new 3D coordinates and moves the arm to the new ready position! |
| **`[N]`** | **Randomize** all object positions on the tabletop and watch the AI re-identify the target and move to the new ready pose |
| **`[1]` – `[5]`** | Quick variable object targets (`[1]` Red Bottle, `[2]` White Mixed Foam Board, `[3]` Green Block, `[4]` Yellow Cup, `[5]` Black Phone) |
| **`[6]`** | Load real cluttered desk photo (`5-DOF-Robotic-Arm-Basic/foam.png`) |
| **`[S]`** | Toggle between **Live Laptop Webcam** and **Interactive Tabletop Simulation** |
| **`[R]`** | Reset arm and tabletop objects to initial state |
| **`[Q]` / `[ESC]`** | Quit the application |
