import json
import os
import re
import difflib
import requests
from io import BytesIO
from PIL import Image, ImageOps
import pytesseract
from openai import OpenAI
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse


class NewModel(LabelStudioMLBase):
    """Custom ML backend that uses OCR + ChatAI to extract technical properties."""

    def setup(self):
        """Initialize model and API client"""
        self.set("model_version", "1.1.0")

        # Load API configuration
        self.api_key = os.getenv("CHAT_API_KEY")
        self.base_url = os.getenv("CHAT_BASE_URL", "https://chat-ai.academiccloud.de/v1")
        self.model_name = os.getenv("CHAT_MODEL", "meta-llama-3.1-8b-instruct")

        if not self.api_key:
            raise ValueError("CHAT_API_KEY not set in environment")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=1800)
        print(f"✅ Model initialized: {self.model_name} at {self.base_url}")

    def _ocr_image(self, image_url):
        """Perform OCR on a given image URL."""
        print(f"🔍 Running OCR for image: {image_url}")
        response = requests.get(image_url)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image = ImageOps.exif_transpose(image)

        text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        blocks = []
        for i, txt in enumerate(text_data["text"]):
            txt = txt.strip()
            if not txt:
                continue
            blocks.append({
                "text": txt,
                "bbox": (
                    text_data["left"][i],
                    text_data["top"][i],
                    text_data["width"][i],
                    text_data["height"][i]
                )
            })

        full_text = "\n".join(b["text"] for b in blocks)
        print(f"🧾 OCR extracted {len(blocks)} text blocks.")
        return full_text, blocks, image.size

    def _ask_model_for_properties(self, text):
        """Ask the ChatAI model to extract property triples from text."""
        prompt = f"""
        Extract all technical properties (name, value, and unit) from the following text.
        Return them as a *pure JSON list*, with keys: "prop-name", "prop-value", "prop-unit".
        Do not include explanations or markdown fences.
        Example:
        [
          {{"prop-name": "Battery", "prop-value": "5000", "prop-unit": "mAh"}},
          {{"prop-name": "Screen size", "prop-value": "6.2", "prop-unit": "inch"}}
        ]

        Text:
        {text}
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a precise information extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            timeout=1800,
        )

        raw_output = response.choices[0].message.content.strip()

        # --- Attempt to extract JSON safely ---
        try:
            match = re.search(r'\[.*\]', raw_output, re.DOTALL)
            json_str = match.group(0) if match else raw_output
            data = json.loads(json_str)

            # Normalize: ensure all values are strings
            for p in data:
                for k, v in p.items():
                    p[k] = str(v).strip() if v is not None else ""

            print(f"✅ Parsed {len(data)} properties from model output.")
            return data

        except Exception as e:
            print(f"⚠️ Could not parse model output: {e}")
            print("Model output:\n", raw_output)
            return []

    def predict(self, tasks, context=None, timout=1200,  **kwargs):
        """Run OCR + LLM-based property extraction on each image page."""
        predictions = []

        for task in tasks:
            pages = task.get("data", {}).get("pages", [])
            results = []

            for page_index, page_url in enumerate(pages):
                print(f"📄 Processing page {page_index}: {page_url}")

                # --- OCR ---
                full_text, text_blocks, (img_w, img_h) = self._ocr_image(page_url)

                # --- LLM Extraction ---
                props = self._ask_model_for_properties(full_text)

                # --- Match properties to OCR text (multi-match enabled) ---
                for prop in props:
                    for key, value in prop.items():
                        if not value:
                            continue

                        value_str = str(value).strip().lower()
                        if not value_str:
                            continue

                        for block in text_blocks:
                            block_text = block.get("text", "").strip().lower()
                            if not block_text:
                                continue

                            # fuzzy similarity
                            score = difflib.SequenceMatcher(None, value_str, block_text).ratio()
                            if score > 0.8:
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
                                    },
                                    "item_index": page_index  # ✅ tell LS which page this belongs to
                                })
                                print(
                                    f"✅ Matched '{value}' as '{key}' (page {page_index}, score={score:.2f}) at ({x},{y})")

            predictions.append({
                "model_version": self.get("model_version"),
                "result": results
            })

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        print(f"🧠 Received training event: {event}")
