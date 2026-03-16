"""Debug entrypoint that consumes webcam frames from the usb_cam ROS topic."""

from .test_with_ros_camera import run_ros_camera_monitor


USB_CAM_TOPIC_NAME = "/image/image_raw"
WINDOW_NAME = "USB Cam Driving Decision Monitor"
NODE_NAME = "vlm_usb_cam_monitor"


def main():
    """Debug using usb_cam instead of opening the webcam directly with OpenCV."""
    print("Expecting usb_cam to publish webcam images before this node starts.\n")
    run_ros_camera_monitor(
        camera_topic_name=USB_CAM_TOPIC_NAME,
        window_name=WINDOW_NAME,
        node_name=NODE_NAME,
    )


if __name__ == "__main__":
    main()
