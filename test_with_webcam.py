"""Local webcam debug entrypoint for the shared VLM driving pipeline."""

import json
from datetime import datetime

import cv2

from vlm_driving_common import (
    CAPTURE_INTERVAL,
    ROS_TOPIC_NAME,
    analyze_frame_with_vlm,
    build_overlay_lines,
    initialize_ros_publisher,
    publish_analysis_to_ros,
    shutdown_ros_publisher,
)


WEBCAM_INDEX = 0  # Usually 0 for the default webcam


def main():
    print("=" * 80)
    print("Real-time Webcam Driving Decision Monitor")
    print("=" * 80)
    print(f"Capture interval: {CAPTURE_INTERVAL} seconds")
    print("Press 'q' in the webcam window or Ctrl+C to stop\n")

    cap = None
    backends_to_try = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Default"),
    ]
    camera_indices = [WEBCAM_INDEX] + [idx for idx in [0, 1, 2] if idx != WEBCAM_INDEX]

    for camera_idx in camera_indices:
        for backend, backend_name in backends_to_try:
            print(f"Trying camera {camera_idx} with {backend_name} backend...")
            cap = cv2.VideoCapture(camera_idx, backend)

            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"Successfully opened camera {camera_idx} with {backend_name}\n")
                    break
                cap.release()
                cap = None
            else:
                if cap:
                    cap.release()
                cap = None

        if cap and cap.isOpened():
            break

    if not cap or not cap.isOpened():
        print("\n" + "=" * 80)
        print("ERROR: Could not open any webcam")
        print("=" * 80)
        print("\nTroubleshooting steps:")
        print("1. Close any other applications using the camera (Camera app, Teams, etc.)")
        print("2. Check Windows Settings > Privacy > Camera")
        print("3. Make sure camera permissions are enabled for Python")
        print("4. Try running as administrator")
        print("\nPress Enter to exit...")
        input()
        return

    print("Webcam initialized successfully\n")
    ros_publisher = initialize_ros_publisher()

    window_name = "Driving Decision Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    last_analysis_time = 0.0
    last_result = None

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break

            display_frame = frame_bgr.copy()
            if last_result:
                y_offset = 30
                overlay_lines = build_overlay_lines(frame_count, last_result)
                for line in overlay_lines:
                    font_scale = 0.65 if line.startswith("Frame:") else 0.55
                    thickness = 2 if line.startswith("Frame:") else 1
                    cv2.putText(
                        display_frame,
                        line,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 0),
                        thickness,
                    )
                    y_offset += 30

            cv2.imshow(window_name, display_frame)

            current_time = datetime.now().timestamp()
            if current_time - last_analysis_time >= CAPTURE_INTERVAL:
                frame_count += 1
                last_analysis_time = current_time
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(f"[Frame {frame_count}] {timestamp}")
                print("-" * 80)

                raw_result = None
                try:
                    raw_result, parsed_result = analyze_frame_with_vlm(frame_bgr)

                    print(f"Response Time: {raw_result['elapsed_time']:.2f}s")
                    print("Structured Output:")
                    print(json.dumps(parsed_result, indent=2))
                    if publish_analysis_to_ros(ros_publisher, parsed_result):
                        print(f"Published to ROS 2 topic: {ROS_TOPIC_NAME}")

                    last_result = parsed_result

                except Exception as error:
                    print(f"Error: {error}")
                    if raw_result and raw_result.get("response"):
                        print("Raw model response:")
                        print(raw_result["response"])

                print("-" * 80 + "\n")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nMonitoring stopped by user")
                break

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        shutdown_ros_publisher(ros_publisher)
        print("Webcam released")


if __name__ == "__main__":
    main()
