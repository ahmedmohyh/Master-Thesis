from flask import Flask, request, jsonify
from openai import OpenAI
from logic.LLM.ChatAI.config import API_KEY, BASE_URL, MODEL
from PIL import Image
import pytesseract
import requests
import io
import json

app = Flask(__name__)

# initialize the SAIA client (OpenAI-compatible)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---------- helper: extract text and boxes from an image ----------
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

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return []

# ---------- main prediction endpoint ----------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    results = []

    pages = data["data"].get("pages", [])
    for page_url in pages:
        full_text, text_blocks, (img_w, img_h) = extract_ocr_data(page_url)
        props = ask_model_for_properties(full_text)

        for prop in props:
            for key, value in prop.items():
                # find matching OCR block
                for block in text_blocks:
                    if block["text"].lower() == str(value).lower():
                        x, y, w, h = block["bbox"]
                        results.append({
                            "from_name": "rectangles",
                            "to_name": "pdf",
                            "type": "rectanglelabels",
                            "value": {
                                "x": (x / img_w) * 100,
                                "y": (y / img_h) * 100,
                                "width": (w / img_w) * 100,
                                "height": (h / img_h) * 100,
                                "rotation": 0,
                                "rectanglelabels": [key]
                            }
                        })
                        break

    return jsonify({"results": results})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
