"""Shared VLM driving-analysis helpers for local webcam and ROS 2 pipelines."""

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

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
    from std_msgs.msg import Bool, String
except ImportError:
    Bool = None
    String = None



def load_local_vlm_config() -> dict:
    """Load optional local, untracked VLM settings for one-command startup."""
    config_paths = []
    configured_path = os.getenv("SVDC_VLM_CONFIG")
    if configured_path:
        config_paths.append(Path(configured_path).expanduser())

    repo_root = Path(__file__).resolve().parent.parent
    config_paths.extend(
        [
            Path.cwd() / "svdc_vlm_config.local.json",
            repo_root / "svdc_vlm_config.local.json",
            Path.home() / ".config" / "svdc_preception_vlm" / "config.json",
        ]
    )

    for config_path in config_paths:
        if not config_path.is_file():
            continue
        with config_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    return {}


LOCAL_VLM_CONFIG = load_local_vlm_config()

MODEL_NAME = os.getenv(
    "SVDC_VLM_MODEL",
    LOCAL_VLM_CONFIG.get("vllm_model", "qwen3-vl:8b-instruct"),
)
VLLM_BASE_URL = os.getenv(
    "SVDC_VLM_BASE_URL",
    LOCAL_VLM_CONFIG.get("vllm_base_url", "http://192.168.64.1:11434"),
)
VLM_PROVIDER = os.getenv(
    "SVDC_VLM_PROVIDER",
    LOCAL_VLM_CONFIG.get("provider", "vllm"),
).strip().lower()
GEMINI_MODEL_NAME = os.getenv(
    "SVDC_GEMINI_MODEL",
    LOCAL_VLM_CONFIG.get("gemini_model", "gemini-3-flash-preview"),
)
GEMINI_API_KEY = (
    os.getenv("SVDC_GEMINI_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or LOCAL_VLM_CONFIG.get("gemini_api_key")
)
GEMINI_BASE_URL = os.getenv(
    "SVDC_GEMINI_BASE_URL",
    LOCAL_VLM_CONFIG.get(
        "gemini_base_url",
        "https://generativelanguage.googleapis.com/v1beta",
    ),
)
CAPTURE_INTERVAL = 3  # seconds
ROS_NODE_NAME = "vlm_drive_context_publisher"
ROS_TOPIC_NAME = "/drive_context"
SCENE_CONTEXT_TOPIC = "/scene_context"
TRAFFIC_CONE_PRESENT_TOPIC = "/traffic_cone/present"
TRAFFIC_SIGNAL_PRESENT_TOPIC = "/traffic_signal/present"
TRAFFIC_SIGNAL_RED_TOPIC = "/traffic_signal/red"
TRAFFIC_SIGNAL_GREEN_TOPIC = "/traffic_signal/green"
ROS_QOS_DEPTH = 10

SCENE_CONTEXTS = {"simple", "complex", "unknown"}
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
  "scene_context": "simple | complex | unknown",
  "traffic_cone_present": true,
  "traffic_cone_reason": "short evidence-based explanation",
  "road_type": "highway | city | unknown",
  "road_surface": "dry | wet | unknown",
  "hazard_present": true,
  "hazard_type": "none | obstacle | pedestrian_intrusion_risk | unknown",
  "hazard_reason": "short evidence-based explanation",
  "traffic_signal_present": true,
  "traffic_signal_red": false,
  "traffic_signal_green": false,
  "driving_action": "accelerate | maintain_speed | decelerate",
  "decision_reason": "short explanation based on the visible scene"
}

Decision rules:
- scene_context: use complex whenever a traffic cone, road cone, construction cone,
  pylon cone, or rubber cone shape is visible anywhere in the driving scene. Use
  simple only when no such cone-like object is visible and the scene is otherwise
  clear enough to judge. Use unknown only when the image is too unclear.
- traffic_cone_present: true when a cone-like road marker is visible. Judge by
  shape, not color: look for a tapered cone/frustum body, wide base, stack/ring
  bands, or pylon-like road marker silhouette. Cones may be orange, yellow, blue,
  green, white, black, or other colors; do not require orange.
- traffic_cone_reason: briefly name the visible shape evidence, not the color.
- road_type: use highway for freeway or expressway scenes, city for urban or local roads, unknown if unclear.
- road_surface: use wet only when the road visibly appears wet, rainy, or puddled; use dry only when it visibly appears dry.
- hazard_present: true only when there is a forward hazard such as an obstacle, traffic cone blocking or narrowing the lane, or a person likely to enter the lane.
- hazard_type: use none when hazard_present is false.
- hazard_type: use obstacle when a traffic cone is blocking, narrowing, or placed in the likely driving path.
- traffic_signal_present: true only when a traffic signal is visibly present.
- traffic_signal_red and traffic_signal_green: true only when that light state is clearly visible.
- If traffic_signal_present is false, set traffic_signal_red and traffic_signal_green to false.
- traffic_cone_reason, hazard_reason, and decision_reason must be brief and based only on visible evidence.
"""

USER_PROMPT = (
    "Analyze this driving scene, pay special attention to cone-shaped road markers, "
    "and respond with the required JSON object only."
)


def configure_vlm_backend(
    provider=None,
    vllm_base_url=None,
    vllm_model=None,
    gemini_api_key=None,
    gemini_model=None,
):
    """Apply runtime model backend settings from CLI arguments."""
    global GEMINI_API_KEY
    global GEMINI_MODEL_NAME
    global MODEL_NAME
    global VLLM_BASE_URL
    global VLM_PROVIDER

    normalized_provider = (provider or VLM_PROVIDER).strip().lower()
    if normalized_provider not in {"vllm", "gemini"}:
        raise ValueError("VLM provider must be either 'vllm' or 'gemini'.")
    VLM_PROVIDER = normalized_provider

    if vllm_base_url:
        VLLM_BASE_URL = vllm_base_url.rstrip("/")
    if vllm_model:
        MODEL_NAME = vllm_model
    if gemini_api_key:
        GEMINI_API_KEY = gemini_api_key
    if gemini_model:
        GEMINI_MODEL_NAME = gemini_model

    if VLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise ValueError(
            "Gemini provider selected, but no API key was provided. "
            "Pass --gemini-api-key or set SVDC_GEMINI_API_KEY/GEMINI_API_KEY."
        )


def describe_vlm_backend() -> str:
    """Return a log-safe summary of the active vision model backend."""
    if VLM_PROVIDER == "gemini":
        return f"gemini api model={GEMINI_MODEL_NAME}"
    return f"vllm base_url={VLLM_BASE_URL} model={MODEL_NAME}"


class DrivingDecisionPublisher(Node):
    """ROS 2 publisher node for streaming structured driving context."""

    def __init__(self, topic_name: str, qos_depth: int):
        super().__init__(ROS_NODE_NAME)
        self.context_publisher = self.create_publisher(String, topic_name, qos_depth)
        self.scene_context_publisher = self.create_publisher(
            String,
            SCENE_CONTEXT_TOPIC,
            qos_depth,
        )
        self.traffic_cone_present_publisher = self.create_publisher(
            Bool,
            TRAFFIC_CONE_PRESENT_TOPIC,
            qos_depth,
        )
        self.signal_present_publisher = self.create_publisher(
            Bool,
            TRAFFIC_SIGNAL_PRESENT_TOPIC,
            qos_depth,
        )
        self.signal_red_publisher = self.create_publisher(
            Bool,
            TRAFFIC_SIGNAL_RED_TOPIC,
            qos_depth,
        )
        self.signal_green_publisher = self.create_publisher(
            Bool,
            TRAFFIC_SIGNAL_GREEN_TOPIC,
            qos_depth,
        )

    def publish_analysis(self, analysis_result: dict):
        """Publish the normalized JSON result and traffic-signal flags."""
        context_message = String()
        context_message.data = json.dumps(analysis_result, ensure_ascii=False)
        self.context_publisher.publish(context_message)

        scene_context_message = String()
        scene_context_message.data = analysis_result["scene_context"]
        self.scene_context_publisher.publish(scene_context_message)

        cone_present_message = Bool()
        cone_present_message.data = analysis_result["traffic_cone_present"]
        self.traffic_cone_present_publisher.publish(cone_present_message)

        signal_present_message = Bool()
        signal_present_message.data = analysis_result["traffic_signal_present"]
        self.signal_present_publisher.publish(signal_present_message)

        signal_red_message = Bool()
        signal_red_message.data = analysis_result["traffic_signal_red"]
        self.signal_red_publisher.publish(signal_red_message)

        signal_green_message = Bool()
        signal_green_message.data = analysis_result["traffic_signal_green"]
        self.signal_green_publisher.publish(signal_green_message)


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
    """Send request to the configured OpenAI-compatible VLM server."""
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
        headers={"Content-Type": "application/json"},
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


def query_gemini_api(system_prompt: str, user_text: str, image_base64: str) -> dict:
    """Send request to the native Gemini API and return response with timing."""
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is not configured.")

    prompt_text = f"{system_prompt.strip()}\n\n{user_text.strip()}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_base64,
                        },
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2048,
        },
    }

    start_time = time.time()
    response = requests.post(
        f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL_NAME}:generateContent",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        timeout=120,
    )
    elapsed_time = time.time() - start_time

    response.raise_for_status()
    result = response.json()
    parts = result["candidates"][0]["content"]["parts"]
    model_response = "".join(part.get("text", "") for part in parts).strip()
    if not model_response:
        raise ValueError("Gemini response did not contain text content.")

    return {
        "elapsed_time": elapsed_time,
        "response": model_response,
    }


def query_vision_model(system_prompt: str, user_text: str, image_base64: str) -> dict:
    """Route a VLM request to the configured backend."""
    if VLM_PROVIDER == "gemini":
        return query_gemini_api(system_prompt, user_text, image_base64)
    return query_vllm_server(system_prompt, user_text, image_base64)


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
    traffic_signal_present: bool,
    traffic_signal_red: bool,
    traffic_signal_green: bool,
):
    """Derive the final vehicle action from the first scene judgments."""
    if traffic_signal_present and traffic_signal_red:
        return "decelerate", "Red traffic signal is visible ahead."

    if hazard_present and hazard_type in {"obstacle", "pedestrian_intrusion_risk", "unknown"}:
        return "decelerate", "Hazard detected ahead, so slowing down is safest."

    if road_surface == "wet":
        return "decelerate", "Road appears wet, so reducing speed is safer."

    if traffic_signal_present and traffic_signal_green:
        if road_type == "highway":
            return "accelerate", "Green signal is visible and the roadway appears clear."
        return "maintain_speed", "Green signal is visible, so continuing smoothly is appropriate."

    if road_type == "highway":
        return "accelerate", "Highway scene appears clear and dry."

    if road_type == "city":
        return "maintain_speed", "City-road scene is clearer but still calls for steady speed."

    return "maintain_speed", "Scene certainty is limited, so maintaining speed is safer."


def parse_analysis_response(response_text: str) -> dict:
    """Parse model JSON and normalize it into a stable key-value structure."""
    parsed = json.loads(extract_json_object(response_text))

    scene_context = normalize_enum(parsed.get("scene_context"), SCENE_CONTEXTS, default="simple")
    traffic_cone_present = normalize_bool(parsed.get("traffic_cone_present"))
    road_type = normalize_enum(parsed.get("road_type"), ROAD_TYPES)
    road_surface = normalize_enum(parsed.get("road_surface"), ROAD_SURFACES)
    hazard_present = normalize_bool(parsed.get("hazard_present"))
    hazard_type = normalize_enum(parsed.get("hazard_type"), HAZARD_TYPES)
    traffic_signal_present = normalize_bool(parsed.get("traffic_signal_present"))
    traffic_signal_red = normalize_bool(parsed.get("traffic_signal_red"))
    traffic_signal_green = normalize_bool(parsed.get("traffic_signal_green"))

    if traffic_cone_present:
        scene_context = "complex"
        traffic_cone_reason = sanitize_reason(
            parsed.get("traffic_cone_reason"),
            "Traffic-cone-shaped road marker is visible.",
        )
    else:
        if scene_context == "complex":
            traffic_cone_present = True
            traffic_cone_reason = sanitize_reason(
                parsed.get("traffic_cone_reason"),
                "Traffic-cone-shaped road marker is visible.",
            )
        else:
            traffic_cone_reason = "No traffic cone shape detected."

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

    if not traffic_signal_present:
        traffic_signal_red = False
        traffic_signal_green = False

    driving_action, decision_reason = determine_driving_action(
        road_type=road_type,
        road_surface=road_surface,
        hazard_present=hazard_present,
        hazard_type=hazard_type,
        traffic_signal_present=traffic_signal_present,
        traffic_signal_red=traffic_signal_red,
        traffic_signal_green=traffic_signal_green,
    )

    return {
        "scene_context": scene_context,
        "traffic_cone_present": traffic_cone_present,
        "traffic_cone_reason": traffic_cone_reason,
        "road_type": road_type,
        "road_surface": road_surface,
        "hazard_present": hazard_present,
        "hazard_type": hazard_type,
        "hazard_reason": hazard_reason,
        "traffic_signal_present": traffic_signal_present,
        "traffic_signal_red": traffic_signal_red,
        "traffic_signal_green": traffic_signal_green,
        "driving_action": driving_action,
        "decision_reason": decision_reason,
    }


def analyze_frame_with_vlm(frame_bgr) -> tuple[dict, dict]:
    """Run the full VLM pipeline for a BGR frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    image_base64 = encode_image_to_base64(frame_pil)
    raw_result = query_vision_model(SYSTEM_PROMPT, USER_PROMPT, image_base64)
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
    cone_label = "present" if analysis_result["traffic_cone_present"] else "none"
    hazard_label = analysis_result["hazard_type"] if analysis_result["hazard_present"] else "none"
    signal_label = "none"
    if analysis_result["traffic_signal_present"]:
        if analysis_result["traffic_signal_red"]:
            signal_label = "red"
        elif analysis_result["traffic_signal_green"]:
            signal_label = "green"
        else:
            signal_label = "present"
    lines = [
        f"Frame: {frame_count}",
        f"Context: {analysis_result['scene_context']}",
        f"Cone: {cone_label}",
        f"Road: {analysis_result['road_type']}",
        f"Surface: {analysis_result['road_surface']}",
        f"Hazard: {hazard_label}",
        f"Signal: {signal_label}",
        f"Action: {analysis_result['driving_action']}",
    ]
    lines.extend(wrap_text(f"Why: {analysis_result['decision_reason']}", 55)[:2])
    return lines


def initialize_ros_publisher():
    """Create a ROS 2 publisher if rclpy is available in the runtime."""
    if rclpy is None or String is None or Bool is None:
        print("ROS 2 publisher disabled: rclpy/std_msgs are not available in this environment.\n")
        return None

    if not rclpy.ok():
        rclpy.init(args=None)

    publisher_node = DrivingDecisionPublisher(ROS_TOPIC_NAME, ROS_QOS_DEPTH)
    print(f"ROS 2 publisher ready on topic: {ROS_TOPIC_NAME}")
    print(f"Scene context topic: {SCENE_CONTEXT_TOPIC}")
    print(f"Traffic cone topic: {TRAFFIC_CONE_PRESENT_TOPIC}")
    print(
        "Traffic signal topics: "
        f"{TRAFFIC_SIGNAL_PRESENT_TOPIC}, {TRAFFIC_SIGNAL_RED_TOPIC}, {TRAFFIC_SIGNAL_GREEN_TOPIC}\n"
    )
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
