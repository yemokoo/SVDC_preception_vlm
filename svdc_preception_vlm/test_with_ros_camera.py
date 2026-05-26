"""ROS 2 camera subscriber entrypoint for the shared VLM driving pipeline."""

import argparse
import json
import threading
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
except ImportError:
    rclpy = None
    Node = object
    Image = None

from .vlm_driving_common import (
    CAPTURE_INTERVAL,
    MAX_IN_FLIGHT_ANALYSES,
    ROS_TOPIC_NAME,
    analyze_frame_with_vlm,
    build_overlay_lines,
    configure_vlm_backend,
    describe_vlm_backend,
    initialize_ros_publisher,
    publish_analysis_to_ros,
    shutdown_ros_publisher,
)


CAMERA_TOPIC_NAME = "/image/image_raw"
CAMERA_QOS_DEPTH = 10
WINDOW_NAME = "ROS Camera Driving Decision Monitor"
NODE_NAME = "vlm_ros_camera_monitor"


def ros_image_to_bgr(message: Image):
    """Convert a sensor_msgs/Image message into an OpenCV BGR frame."""
    encoding = message.encoding.lower()
    channels_by_encoding = {
        "mono8": 1,
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
    }

    if encoding not in channels_by_encoding:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")

    channels = channels_by_encoding[encoding]
    row_width_bytes = message.width * channels
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected_size = message.height * message.step
    if raw.size < expected_size:
        raise ValueError("ROS image buffer is smaller than expected.")

    image = raw[:expected_size].reshape((message.height, message.step))
    image = image[:, :row_width_bytes]

    if channels == 1:
        image = image.reshape((message.height, message.width))
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = image.reshape((message.height, message.width, channels))
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


class RosCameraDrivingMonitor(Node):
    """Subscribe to a ROS image topic and keep the latest frame for analysis."""

    def __init__(self, topic_name: str, qos_depth: int, node_name: str):
        super().__init__(node_name)
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            qos_depth,
        )
        self.topic_name = topic_name
        self.latest_frame_bgr = None
        self.last_result = None
        self.last_result_frame_count = 0
        self.frame_count = 0
        self.received_frame_count = 0
        self.last_processed_frame_count = 0
        self.last_analysis_time = 0.0
        self.last_error_message = None
        self.analysis_lock = threading.Lock()
        self.analysis_in_flight = 0
        self.completed_analyses = {}
        self.next_completed_frame_count = 1

    def start_analysis(self, frame_bgr, frame_count: int, timestamp: str):
        """Start VLM analysis without blocking the camera display loop."""
        with self.analysis_lock:
            if self.analysis_in_flight >= MAX_IN_FLIGHT_ANALYSES:
                return False
            self.analysis_in_flight += 1

        worker = threading.Thread(
            target=self._run_analysis,
            args=(frame_bgr, frame_count, timestamp),
            daemon=True,
        )
        worker.start()
        return True

    def _run_analysis(self, frame_bgr, frame_count: int, timestamp: str):
        raw_result = None
        try:
            raw_result, parsed_result = analyze_frame_with_vlm(frame_bgr)
            analysis = {
                "frame_count": frame_count,
                "timestamp": timestamp,
                "raw_result": raw_result,
                "parsed_result": parsed_result,
                "error": None,
            }
        except Exception as error:
            analysis = {
                "frame_count": frame_count,
                "timestamp": timestamp,
                "raw_result": raw_result,
                "parsed_result": None,
                "error": error,
            }

        with self.analysis_lock:
            self.completed_analyses[frame_count] = analysis
            self.analysis_in_flight -= 1

    def pop_completed_analysis_fifo(self):
        with self.analysis_lock:
            analysis = self.completed_analyses.pop(self.next_completed_frame_count, None)
            if analysis:
                self.next_completed_frame_count += 1
        return analysis

    def can_start_analysis(self):
        with self.analysis_lock:
            return self.analysis_in_flight < MAX_IN_FLIGHT_ANALYSES

    def get_analysis_in_flight(self):
        with self.analysis_lock:
            return self.analysis_in_flight

    def image_callback(self, message: Image):
        """Store the newest frame from the ROS camera stream."""
        try:
            self.latest_frame_bgr = ros_image_to_bgr(message)
            self.received_frame_count += 1
            self.last_error_message = None
        except Exception as error:
            error_message = str(error)
            if error_message != self.last_error_message:
                self.get_logger().warning(error_message)
                self.last_error_message = error_message


def build_waiting_frame(camera_topic_name: str):
    """Create a blank frame while waiting for the first ROS image."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"Waiting for {camera_topic_name}",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return frame


def run_ros_camera_monitor(
    camera_topic_name: str = CAMERA_TOPIC_NAME,
    window_name: str = WINDOW_NAME,
    node_name: str = NODE_NAME,
    rclpy_args=None,
):
    """Run the VLM monitor by subscribing to a ROS image topic."""
    if rclpy is None or Image is None:
        print("ROS 2 camera mode requires rclpy and sensor_msgs to be installed and sourced.")
        return

    print("=" * 80)
    print("ROS 2 Camera Driving Decision Monitor")
    print("=" * 80)
    print(f"Subscribed topic: {camera_topic_name}")
    print(f"Capture interval: {CAPTURE_INTERVAL} seconds")
    print(f"Max in-flight analyses: {MAX_IN_FLIGHT_ANALYSES}")
    print(f"VLM backend: {describe_vlm_backend()}")
    print("Press 'q' in the monitor window or Ctrl+C to stop\n")

    if not rclpy.ok():
        rclpy.init(args=rclpy_args)

    ros_publisher = initialize_ros_publisher()
    subscriber = RosCameraDrivingMonitor(camera_topic_name, CAMERA_QOS_DEPTH, node_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(subscriber, timeout_sec=0.1)

            current_frame = subscriber.latest_frame_bgr
            if current_frame is None:
                display_frame = build_waiting_frame(camera_topic_name)
            else:
                display_frame = current_frame.copy()

            if subscriber.last_result:
                y_offset = 30
                overlay_lines = build_overlay_lines(
                    subscriber.last_result_frame_count,
                    subscriber.last_result,
                )
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

            while True:
                completed_analysis = subscriber.pop_completed_analysis_fifo()
                if not completed_analysis:
                    break
                frame_count = completed_analysis["frame_count"]
                timestamp = completed_analysis["timestamp"]
                raw_result = completed_analysis["raw_result"]
                parsed_result = completed_analysis["parsed_result"]
                error = completed_analysis["error"]

                print(f"[Frame {frame_count}] {timestamp}")
                print("-" * 80)
                if error is None:
                    print(f"Response Time: {raw_result['elapsed_time']:.2f}s")
                    print("Structured Output:")
                    print(json.dumps(parsed_result, indent=2))
                    if publish_analysis_to_ros(ros_publisher, parsed_result):
                        print(f"Published to ROS 2 topic: {ROS_TOPIC_NAME}")

                    subscriber.last_result = parsed_result
                    subscriber.last_result_frame_count = frame_count
                else:
                    print(f"Error: {error}")
                    if raw_result and raw_result.get("response"):
                        print("Raw model response:")
                        print(raw_result["response"])
                print("-" * 80 + "\n")

            has_new_frame = (
                subscriber.received_frame_count > subscriber.last_processed_frame_count
                and subscriber.latest_frame_bgr is not None
            )
            is_analysis_due = time.time() - subscriber.last_analysis_time >= CAPTURE_INTERVAL
            if has_new_frame and is_analysis_due and subscriber.can_start_analysis():
                subscriber.frame_count += 1
                subscriber.last_processed_frame_count = subscriber.received_frame_count
                subscriber.last_analysis_time = time.time()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                frame_for_analysis = subscriber.latest_frame_bgr.copy()

                if subscriber.start_analysis(
                    frame_for_analysis,
                    subscriber.frame_count,
                    timestamp,
                ):
                    in_flight = subscriber.get_analysis_in_flight()
                    print(
                        f"[Frame {subscriber.frame_count}] {timestamp} "
                        f"- VLM analysis started ({in_flight}/{MAX_IN_FLIGHT_ANALYSES} in flight)"
                    )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nMonitoring stopped by user")
                break

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

    finally:
        cv2.destroyAllWindows()
        subscriber.destroy_node()
        shutdown_ros_publisher(ros_publisher)
        if ros_publisher is None and rclpy.ok():
            rclpy.shutdown()
        print("ROS camera subscriber stopped")


def parse_args(argv=None):
    """Parse app-specific arguments while leaving ROS arguments for rclpy."""
    parser = argparse.ArgumentParser(
        description="Run the ROS camera VLM driving monitor.",
    )
    parser.add_argument(
        "--provider",
        "--vlm-provider",
        choices=("vllm", "gemini"),
        default=None,
        help="Vision model backend to use. Defaults to SVDC_VLM_PROVIDER or vllm.",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=None,
        help="OpenAI-compatible vLLM base URL.",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="OpenAI-compatible vLLM model name.",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Gemini API key. Can also be set with SVDC_GEMINI_API_KEY or GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="Gemini model name. Defaults to SVDC_GEMINI_MODEL or gemini-3-flash-preview.",
    )
    parser.add_argument(
        "--camera-topic",
        default=CAMERA_TOPIC_NAME,
        help=f"ROS image topic to subscribe to. Defaults to {CAMERA_TOPIC_NAME}.",
    )
    return parser.parse_known_args(argv)


def main(argv=None):
    """Default ROS 2 camera entrypoint using the ROS raw image topic."""
    args, rclpy_args = parse_args(argv)
    try:
        configure_vlm_backend(
            provider=args.provider,
            vllm_base_url=args.vllm_base_url,
            vllm_model=args.vlm_model,
            gemini_api_key=args.gemini_api_key,
            gemini_model=args.gemini_model,
        )
    except ValueError as error:
        print(f"Configuration error: {error}")
        return

    run_ros_camera_monitor(
        camera_topic_name=args.camera_topic,
        rclpy_args=rclpy_args or None,
    )


if __name__ == "__main__":
    main()
