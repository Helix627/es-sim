"""
Vision-Guided Robotic Arm Prototype - Main Entry Point.
Delegates to the unified pipeline in `arm_demo_pipeline.py`.
"""
import argparse
from arm_demo_pipeline import run_pipeline


def main():
    parser_cli = argparse.ArgumentParser(description="Vision-Guided Robotic Arm Software Prototype")
    parser_cli.add_argument(
        "--command",
        "--query",
        dest="command",
        type=str,
        default="Pick up the red bottle and place it in the blue tray.",
        help="Natural language command or object query to execute",
    )
    parser_cli.add_argument(
        "--image",
        type=str,
        default="",
        help="Optional static image path (e.g. 5-DOF-Robotic-Arm-Basic/foam.png)",
    )
    parser_cli.add_argument(
        "--webcam",
        type=int,
        default=0,
        help="Webcam device index (default 0)",
    )
    parser_cli.add_argument(
        "--sim",
        action="store_true",
        default=False,
        help="Start in synthetic tabletop simulation mode (no physical props required)",
    )
    parser_cli.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless terminal verification mode",
    )
    args = parser_cli.parse_args()

    run_pipeline(
        query=args.command,
        image_path_str=args.image,
        webcam_index=args.webcam,
        use_simulation=args.sim,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
