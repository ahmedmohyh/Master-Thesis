# app.py
from flask import Flask, request, jsonify
from openai import OpenAI
from logic.LLM.ChatAI.config import API_KEY, BASE_URL, MODEL
from PIL import Image
import pytesseract
import requests
import io
import json
import re
from difflib import SequenceMatcher


app = Flask(__name__)

# Initialize the SAIA OpenAI-compatible client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---------- helper: extract text and bounding boxes from image ----------
def extract_ocr_data(image_url):
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))

    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    text_blocks = []

    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i].strip()
        if text:
            x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
            text_blocks.append({"text": text, "bbox": [x, y, w, h]})

    full_text = " ".join([b["text"] for b in text_blocks])
    return full_text, text_blocks, img.size

# ---------- helper: ask SAIA model ----------

def ask_model_for_properties(text):
    prompt = f"""
    Extract all technical properties (name, value, and unit) from the following text.
    Return them as a JSON list, with keys: "prop-name", "prop-value", "prop-unit".
    Example:
    [
      {{"prop-name": "Battery", "prop-value": "5000", "prop-unit": "mAh"}},
      {{"prop-name": "Screen size", "prop-value": "6.2", "prop-unit": "inch"}}
    ]

    Text:
    {text}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a precise extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    raw_output = response.choices[0].message.content

    # Remove any leading text before JSON
    match = re.search(r'(\[.*\]|\{.*\})', raw_output, flags=re.DOTALL)
    if match:
        json_text = match.group(1)
        try:
            return json.loads(json_text)
        except Exception as e:
            print("⚠️ Could not parse JSON after cleaning:", e)
            print("Cleaned text:", json_text)
            return []
    else:
        print("⚠️ No JSON found in model output")
        print("Raw output:", raw_output)
        return []


def fuzzy_match(a, b, threshold=0.8):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def clamp(v):
    return max(0, min(100, v))


# ---------- ML backend endpoints ----------
@app.route("/predict", methods=["POST"])
def predict():
    req = request.json

    # Write the JSON to a file for debugging or logging
    with open("request_payload.json", "w", encoding="utf-8") as f:
        json.dump(req, f, indent=2, ensure_ascii=False)


    # Extract all pages from all tasks
    pages = []
    if "tasks" in req:
        for task in req["tasks"]:
            task_pages = task.get("data", {}).get("pages", [])
            pages.extend(task_pages)

    print("Pages extracted:", pages)

    if not pages:
        return jsonify({"results": [], "error": "No pages found in the request."})

   #page16 = pages[16]
   #pages.clear()
   #pages.append(page16)

    results = []

    for page_url in pages:
        full_text, text_blocks, (img_w, img_h) = extract_ocr_data(page_url)
        props = ask_model_for_properties(full_text)

        with open("response_props.json", "w", encoding="utf-8") as f:
            json.dump({"props": props}, f, indent=2, ensure_ascii=False)

        for prop in props:
            for key, value in prop.items():
                for block in text_blocks:
                    if fuzzy_match(block["text"], str(value)):
                        x, y, w, h = block["bbox"]
                        results.append({
                            "from_name": "rectangles",
                            "to_name": "pdf",
                            "type": "rectanglelabels",
                            "value": {
                                "x": clamp((x / img_w) * 100),
                                "y": clamp((y / img_h) * 100),
                                "width": clamp((w / img_w) * 100),
                                "height": clamp((h / img_h) * 100),
                                "rotation": 0,
                                "rectanglelabels": [key]
                            }
                        })
                        break

    # Save prediction results to file
    with open("response.json", "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    # Return response to Label Studio
    return jsonify({"results": results})


@app.route("/setup", methods=["POST"])
def setup():
    """Required by Label Studio to recognize the ML backend."""
    data = request.json
    print("✅ Received setup from Label Studio:", json.dumps(data, indent=2))
    return jsonify({"status": "ok"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
