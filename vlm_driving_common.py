"""Shared VLM driving-analysis helpers for local webcam and ROS 2 pipelines."""

import base64
import json
import time
from io import BytesIO

import cv2
import requests
from PIL import Image

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    rclpy = None
    Node = object

try:
    from std_msgs.msg import String
except ImportError:
    String = None


MODEL_NAME = "qwen3vl_8b"
VLLM_BASE_URL = "http://192.168.0.87:8000"
CAPTURE_INTERVAL = 3  # seconds
ROS_NODE_NAME = "vlm_driving_decision_publisher"
ROS_TOPIC_NAME = "/svdc/driving_decision"
ROS_QOS_DEPTH = 10

ROAD_TYPES = {"highway", "city", "unknown"}
ROAD_SURFACES = {"dry", "wet", "unknown"}
HAZARD_TYPES = {"none", "obstacle", "pedestrian_intrusion_risk", "unknown"}

SYSTEM_PROMPT = """You are an autonomous driving scene analyzer.

Analyze the driving scene and return exactly one JSON object.

Output requirements:
- Return JSON only.
- Do not wrap the JSON in markdown fences.
- Do not add explanations before or after the JSON.
- Use the exact keys shown below.
- Use only the allowed values for enum fields.

Required JSON schema:
{
  "road_type": "highway | city | unknown",
  "road_surface": "dry | wet | unknown",
  "hazard_present": true,
  "hazard_type": "none | obstacle | pedestrian_intrusion_risk | unknown",
  "hazard_reason": "short evidence-based explanation",
  "driving_action": "accelerate | maintain_speed | decelerate",
  "decision_reason": "short explanation based on the visible scene"
}

Decision rules:
- road_type: use highway for freeway or expressway scenes, city for urban or local roads, unknown if unclear.
- road_surface: use wet only when the road visibly appears wet, rainy, or puddled; use dry only when it visibly appears dry.
- hazard_present: true only when there is a forward hazard such as an obstacle or a person likely to enter the lane.
- hazard_type: use none when hazard_present is false.
- hazard_reason and decision_reason must be brief and based only on visible evidence.
"""

USER_PROMPT = (
    "Analyze this driving scene and respond with the required JSON object only."
)


class DrivingDecisionPublisher(Node):
    """ROS 2 publisher node for streaming structured driving decisions."""

    def __init__(self, topic_name: str, qos_depth: int):
        super().__init__(ROS_NODE_NAME)
        self.publisher = self.create_publisher(String, topic_name, qos_depth)

    def publish_analysis(self, analysis_result: dict):
        """Publish the normalized JSON result as a ROS 2 String message."""
        message = String()
        message.data = json.dumps(analysis_result, ensure_ascii=False)
        self.publisher.publish(message)


def smart_resize(image: Image.Image, factor: int = 28) -> Image.Image:
    """Resize image to dimensions divisible by factor."""
    width, height = image.size
    target_width = round(width / factor) * factor
    target_height = round(height / factor) * factor
    target_width = max(target_width, factor)
    target_height = max(target_height, factor)

    if target_width != width or target_height != height:
        image = image.resize((target_width, target_height), resample=Image.BICUBIC)

    return image


def encode_image_to_base64(image: Image.Image) -> str:
    """Resize and encode image to base64."""
    processed_image = smart_resize(image)
    buffered = BytesIO()
    processed_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def query_vllm_server(system_prompt: str, user_text: str, image_base64: str) -> dict:
    """Send request to the VLM server and return response with timing."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        },
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    start_time = time.time()
    response = requests.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    elapsed_time = time.time() - start_time

    response.raise_for_status()
    result = response.json()
    model_response = result["choices"][0]["message"]["content"]

    return {
        "elapsed_time": elapsed_time,
        "response": model_response,
    }


def extract_json_object(response_text: str) -> str:
    """Extract a JSON object even if the model adds extra text or code fences."""
    cleaned_response = response_text.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response.replace("```json", "", 1).replace("```", "").strip()
    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response.replace("```", "").strip()

    start_idx = cleaned_response.find("{")
    end_idx = cleaned_response.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError("Model response did not contain a valid JSON object.")

    return cleaned_response[start_idx:end_idx + 1]


def normalize_enum(value, allowed_values, default="unknown") -> str:
    """Normalize string enum values for stable downstream parsing."""
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "_")
        if normalized in allowed_values:
            return normalized
    return default


def normalize_bool(value) -> bool:
    """Normalize common boolean-like values from model output."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def sanitize_reason(value, fallback: str) -> str:
    """Collapse whitespace and provide a predictable fallback reason."""
    if isinstance(value, str):
        cleaned = " ".join(value.strip().split())
        if cleaned:
            return cleaned
    return fallback


def determine_driving_action(
    road_type: str,
    road_surface: str,
    hazard_present: bool,
    hazard_type: str,
):
    """Derive the final vehicle action from the first scene judgments."""
    if hazard_present and hazard_type in {"obstacle", "pedestrian_intrusion_risk", "unknown"}:
        return "decelerate", "Hazard detected ahead, so slowing down is safest."

    if road_surface == "wet":
        return "decelerate", "Road appears wet, so reducing speed is safer."

    if road_type == "highway":
        return "accelerate", "Highway scene appears clear and dry."

    if road_type == "city":
        return "maintain_speed", "City-road scene is clearer but still calls for steady speed."

    return "maintain_speed", "Scene certainty is limited, so maintaining speed is safer."


def parse_analysis_response(response_text: str) -> dict:
    """Parse model JSON and normalize it into a stable key-value structure."""
    parsed = json.loads(extract_json_object(response_text))

    road_type = normalize_enum(parsed.get("road_type"), ROAD_TYPES)
    road_surface = normalize_enum(parsed.get("road_surface"), ROAD_SURFACES)
    hazard_present = normalize_bool(parsed.get("hazard_present"))
    hazard_type = normalize_enum(parsed.get("hazard_type"), HAZARD_TYPES)

    if not hazard_present:
        hazard_type = "none"
        hazard_reason = "No forward hazard detected."
    else:
        if hazard_type == "none":
            hazard_type = "unknown"
        hazard_reason = sanitize_reason(
            parsed.get("hazard_reason"),
            "Potential forward hazard is visible.",
        )

    driving_action, decision_reason = determine_driving_action(
        road_type=road_type,
        road_surface=road_surface,
        hazard_present=hazard_present,
        hazard_type=hazard_type,
    )

    return {
        "road_type": road_type,
        "road_surface": road_surface,
        "hazard_present": hazard_present,
        "hazard_type": hazard_type,
        "hazard_reason": hazard_reason,
        "driving_action": driving_action,
        "decision_reason": decision_reason,
    }


def analyze_frame_with_vlm(frame_bgr) -> tuple[dict, dict]:
    """Run the full VLM pipeline for a BGR frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    image_base64 = encode_image_to_base64(frame_pil)
    raw_result = query_vllm_server(SYSTEM_PROMPT, USER_PROMPT, image_base64)
    parsed_result = parse_analysis_response(raw_result["response"])
    return raw_result, parsed_result


def wrap_text(text: str, max_chars_per_line: int) -> list[str]:
    """Wrap overlay text for compact display in the OpenCV window."""
    words = text.split()
    if not words:
        return []

    lines = []
    current_line = ""
    for word in words:
        projected_length = len(current_line) + len(word) + (1 if current_line else 0)
        if projected_length <= max_chars_per_line:
            current_line = f"{current_line} {word}".strip()
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def build_overlay_lines(frame_count: int, analysis_result: dict) -> list[str]:
    """Build compact overlay lines from the normalized JSON result."""
    hazard_label = analysis_result["hazard_type"] if analysis_result["hazard_present"] else "none"
    lines = [
        f"Frame: {frame_count}",
        f"Road: {analysis_result['road_type']}",
        f"Surface: {analysis_result['road_surface']}",
        f"Hazard: {hazard_label}",
        f"Action: {analysis_result['driving_action']}",
    ]
    lines.extend(wrap_text(f"Why: {analysis_result['decision_reason']}", 55)[:2])
    return lines


def initialize_ros_publisher():
    """Create a ROS 2 publisher if rclpy is available in the runtime."""
    if rclpy is None or String is None:
        print("ROS 2 publisher disabled: rclpy/std_msgs are not available in this environment.\n")
        return None

    if not rclpy.ok():
        rclpy.init(args=None)

    publisher_node = DrivingDecisionPublisher(ROS_TOPIC_NAME, ROS_QOS_DEPTH)
    print(f"ROS 2 publisher ready on topic: {ROS_TOPIC_NAME}\n")
    return publisher_node


def publish_analysis_to_ros(ros_publisher, analysis_result: dict) -> bool:
    """Publish the structured result to ROS 2 when a publisher is available."""
    if ros_publisher is None:
        return False

    ros_publisher.publish_analysis(analysis_result)
    rclpy.spin_once(ros_publisher, timeout_sec=0.0)
    return True


def shutdown_ros_publisher(ros_publisher):
    """Clean up ROS 2 resources on shutdown."""
    if ros_publisher is None or rclpy is None:
        return

    ros_publisher.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
