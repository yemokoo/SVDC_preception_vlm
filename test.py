"""Simple vLLM Image Query Test Script"""

import base64
import json
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


# Configuration
MODEL_NAME = "qwen3-vl:8b-instruct"
VLLM_BASE_URL = "http://192.168.64.1:11434"
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "roadimg.jpg"

# System prompt for autonomous driving decision-making
SYSTEM_PROMPT = """You are an autonomous driving decision-making system. Your task is to analyze road images and provide critical driving information in a structured format.

# Your Role
You must analyze the current road situation and determine:
1. Whether it is safe to accelerate
2. The current road surface condition
3. What potentially dangerous objects are present (moving threats only)
4. Whether the current location is a highway

# Output Format
You MUST respond with ONLY a valid JSON object. Do not include any explanations, markdown formatting, or additional text outside the JSON structure.

Your response must follow this EXACT format:
{
  "acceleration_safe": true or false,
  "road_surface": "dry" or "wet" or "snow" or "ice",
  "dangerous_objects": "description of objects" or "no",
  "is_highway": true or false
}

# Field Definitions
- "acceleration_safe": (boolean) true if it is safe to accelerate given current conditions, false otherwise
- "road_surface": (string) Current road surface condition - must be one of: "dry", "wet", "snow", "ice"
- "dangerous_objects": (string) If dangerous objects exist, describe what they are (e.g., "moving vehicle in adjacent lane", "pedestrian crossing", "motorcycle approaching"). If no dangerous objects, output "no"
- "is_highway": (boolean) true if the current location appears to be a highway/expressway, false otherwise

# Dangerous Objects Definition
IMPORTANT: Dangerous objects are defined as:
- MOVING vehicles or people that could pose a threat within approximately 10 seconds
- This includes: vehicles changing lanes, vehicles approaching from behind/front, motorcycles, bicycles, pedestrians crossing or near the road
- This does NOT include: stationary/parked vehicles, static road infrastructure, distant vehicles moving in the same direction at similar speed

Examples of dangerous objects:
- "vehicle merging from right lane"
- "car braking ahead"
- "motorcycle overtaking on left"
- "pedestrian near roadside"

If no such moving threats exist, output "no"

# Analysis Guidelines
- Consider weather conditions, visibility, and road markings
- Focus on MOVING objects that could intersect your path within 10 seconds
- Evaluate vehicle spacing and relative speeds
- Check for pedestrians, cyclists, or unexpected obstacles in motion
- Identify highway characteristics: multiple lanes, barriers, signs, road width

# Critical Rules
1. Output ONLY the JSON object - no additional text before or after
2. Do NOT use markdown code blocks (no ```json```)
3. Use lowercase "true"/"false" for booleans (not "True"/"False")
4. For "dangerous_objects": provide brief description if present, or "no" if absent
5. Ensure all four fields are present in every response
6. Be conservative in safety assessments - when in doubt, choose the safer option

# Example Outputs
{"acceleration_safe": false, "road_surface": "wet", "dangerous_objects": "vehicle in front braking suddenly", "is_highway": true}
{"acceleration_safe": true, "road_surface": "dry", "dangerous_objects": "no", "is_highway": true}
{"acceleration_safe": false, "road_surface": "wet", "dangerous_objects": "motorcycle overtaking from left lane", "is_highway": false}
"""


def smart_resize(image: Image.Image, factor: int = 28) -> Image.Image:
    """Resize image to dimensions divisible by factor (from original code)"""
    w, h = image.size
    target_w = round(w / factor) * factor
    target_h = round(h / factor) * factor
    target_w = max(target_w, factor)
    target_h = max(target_h, factor)
    
    if target_w != w or target_h != h:
        image = image.resize((target_w, target_h), resample=Image.BICUBIC)
        
    return image


def encode_image_to_base64(image_path: Path) -> str:
    """Load image, resize, and encode to base64"""
    # Load image
    image = Image.open(image_path)
    print(f"Original image size: {image.size}")
    
    # Apply smart resize
    processed_img = smart_resize(image)
    print(f"Resized image size: {processed_img.size}")
    
    # Convert to base64
    buffered = BytesIO()
    processed_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_base64


def query_vllm_server(system_prompt: str, user_text: str, image_base64: str) -> dict:
    """Send request to vLLM server and return response with timing"""
    
    # Build messages
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ]
    
    # Prepare payload
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    
    # Send request and measure time
    print("\nSending request to vLLM server...")
    start_time = time.time()
    
    response = requests.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=120
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Parse response
    response.raise_for_status()
    result = response.json()
    model_response = result['choices'][0]['message']['content']
    
    return {
        "elapsed_time": elapsed_time,
        "response": model_response,
        "full_result": result
    }


def main():
    print("="*80)
    print("vLLM Image Query Test")
    print("="*80)
    
    # Check if system prompt is set
    if not SYSTEM_PROMPT:
        print("\nWarning: SYSTEM_PROMPT is empty. Using default message.")
        system_prompt = "You are a helpful AI assistant."
    else:
        system_prompt = SYSTEM_PROMPT
    
    # User query
    user_query = "Analyze the current road situation and provide your assessment."
    
    try:
        # Encode image
        print(f"\nLoading image from: {IMAGE_PATH}")
        image_base64 = encode_image_to_base64(IMAGE_PATH)
        print(f"Image encoded successfully (base64 length: {len(image_base64)})")
        
        # Query server
        result = query_vllm_server(system_prompt, user_query, image_base64)
        
        # Display results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"\n1. Response Time: {result['elapsed_time']:.2f} seconds")
        print(f"\n2. Model Response (Raw):")
        print("-"*80)
        print(result['response'])
        print("-"*80)
        
        # Try to parse JSON
        print(f"\n3. Parsed JSON:")
        print("-"*80)
        try:
            # Clean response (remove potential markdown formatting)
            cleaned_response = result['response'].strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "").strip()
            
            parsed_json = json.loads(cleaned_response)
            print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
            
            # Validate required fields
            required_fields = ["acceleration_safe", "road_surface", "dangerous_objects", "is_highway"]
            missing_fields = [f for f in required_fields if f not in parsed_json]
            
            if missing_fields:
                print(f"\n⚠️  Warning: Missing fields: {missing_fields}")
            else:
                print("\n✅ All required fields present")
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            print("The model did not return valid JSON format")
        print("-"*80)
        
    except FileNotFoundError:
        print(f"\nError: Image file not found at {IMAGE_PATH}")
    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to vLLM server: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
