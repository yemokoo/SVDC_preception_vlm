"""ROS 2 camera subscriber entrypoint for the shared VLM driving pipeline."""

import json
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

from vlm_driving_common import (
    CAPTURE_INTERVAL,
    ROS_TOPIC_NAME,
    analyze_frame_with_vlm,
    build_overlay_lines,
    initialize_ros_publisher,
    publish_analysis_to_ros,
    shutdown_ros_publisher,
)


CAMERA_TOPIC_NAME = "/camera/image_raw"
CAMERA_QOS_DEPTH = 10
WINDOW_NAME = "ROS Camera Driving Decision Monitor"


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
    """Subscribe to /camera/image_raw and keep the latest frame for analysis."""

    def __init__(self, topic_name: str, qos_depth: int):
        super().__init__("vlm_ros_camera_monitor")
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            qos_depth,
        )
        self.topic_name = topic_name
        self.latest_frame_bgr = None
        self.last_result = None
        self.frame_count = 0
        self.received_frame_count = 0
        self.last_processed_frame_count = 0
        self.last_analysis_time = 0.0
        self.last_error_message = None

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


def build_waiting_frame():
    """Create a blank frame while waiting for the first ROS image."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"Waiting for {CAMERA_TOPIC_NAME}",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return frame


def main():
    if rclpy is None or Image is None:
        print("ROS 2 camera mode requires rclpy and sensor_msgs to be installed and sourced.")
        return

    print("=" * 80)
    print("ROS 2 Camera Driving Decision Monitor")
    print("=" * 80)
    print(f"Subscribed topic: {CAMERA_TOPIC_NAME}")
    print(f"Capture interval: {CAPTURE_INTERVAL} seconds")
    print("Press 'q' in the monitor window or Ctrl+C to stop\n")

    if not rclpy.ok():
        rclpy.init(args=None)

    ros_publisher = initialize_ros_publisher()
    subscriber = RosCameraDrivingMonitor(CAMERA_TOPIC_NAME, CAMERA_QOS_DEPTH)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(subscriber, timeout_sec=0.1)

            current_frame = subscriber.latest_frame_bgr
            if current_frame is None:
                display_frame = build_waiting_frame()
            else:
                display_frame = current_frame.copy()

            if subscriber.last_result:
                y_offset = 30
                overlay_lines = build_overlay_lines(subscriber.frame_count, subscriber.last_result)
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

            cv2.imshow(WINDOW_NAME, display_frame)

            has_new_frame = (
                subscriber.received_frame_count > subscriber.last_processed_frame_count
                and subscriber.latest_frame_bgr is not None
            )
            is_analysis_due = time.time() - subscriber.last_analysis_time >= CAPTURE_INTERVAL
            if has_new_frame and is_analysis_due:
                subscriber.frame_count += 1
                subscriber.last_processed_frame_count = subscriber.received_frame_count
                subscriber.last_analysis_time = time.time()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(f"[Frame {subscriber.frame_count}] {timestamp}")
                print("-" * 80)

                raw_result = None
                try:
                    raw_result, parsed_result = analyze_frame_with_vlm(subscriber.latest_frame_bgr)

                    print(f"Response Time: {raw_result['elapsed_time']:.2f}s")
                    print("Structured Output:")
                    print(json.dumps(parsed_result, indent=2))
                    if publish_analysis_to_ros(ros_publisher, parsed_result):
                        print(f"Published to ROS 2 topic: {ROS_TOPIC_NAME}")

                    subscriber.last_result = parsed_result

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
        cv2.destroyAllWindows()
        subscriber.destroy_node()
        shutdown_ros_publisher(ros_publisher)
        if ros_publisher is None and rclpy.ok():
            rclpy.shutdown()
        print("ROS camera subscriber stopped")


if __name__ == "__main__":
    main()
