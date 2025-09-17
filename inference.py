# main.py

import json
from lmdeploy import pipeline
from lmdeploy.vl import load_image

# --- 1. Set model path ---
# Path to your fine-tuned and merged model directory.
# Based on your screenshot, 'merge_test' is the correct directory
# as it contains the fully merged model weights and configuration.
model_path = './output/merge_test'

# --- 2. Load model and create pipeline ---
# The pipeline function automatically loads the model, tokenizer,
# and sets up the high-performance TurboMind inference engine.
print(f"Loading model from: {model_path}")
pipe = pipeline(model_path, session_len=8192, cache_max_entry_count=0.5)

# --- 3. Prepare inference data ---
# Provide a path to your local image or a URL.
# Using a sample image URL here for demonstration.
image_path = '/home/avsaw1/dengyuan/Qwen2-VL-Finetune/Default_USA_US101-27_1_T-1_10.png'
image = load_image(image_path)     
mode = "Default"                
with open("/home/avsaw1/dengyuan/Qwen2-VL-Finetune/USA_US101-27_1_T-1_10.json", 'r') as f:
    vehicle_state_data = json.load(f)
vehicle_state_json_string = json.dumps(vehicle_state_data, indent=4)
image_path = file_paths['image']
relative_image_path = os.path.relpath(image_path, base_path)

# Define your text prompt.
prompt_text = f'''Act as the autonomous driving system.\n\nBased on the provided BEV image and structured vehicle state data, analyze the scene and plan a {mode} future trajectory for the next 3 seconds.\n\nYour output MUST be a single JSON object containing two top-level keys: `driving_rationale` and `trajectory`.\n\n- The `driving_rationale` must detail your scene analysis, key agent predictions, and the ego-vehicle's strategy, ensuring the strategy logically explains the trajectory.\n- The `trajectory` must contain the lists of `position`, `velocity`, `acceleration`, and `heading_radian` for a 3-second horizon ($t = {{{{0.0, 0.5, ..., 3.0}}}}).\n\n**Structured Vehicle State Data:**\n```json\n{vehicle_state_json_string}\n```'''

# The input format for multimodal models is a list of tuples.
# Each tuple represents a message, containing text and/or an image.
prompts = [(prompt_text, image)]

# --- 4. Run inference ---
print("Running inference...")
response = pipe(prompts)
print("Inference finished.")

# --- 5. Print the result ---
# The response is a list of completion objects.
# We access the generated text from the first item.
print("\n--- Model Response ---")
print(response[0].text)