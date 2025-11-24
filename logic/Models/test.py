import json
import requests
from io import BytesIO
from PIL import Image
import pytesseract
from pytesseract import Output

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


##############################################
# LOAD TRAINED MODEL
##############################################

model_path = "final_ner_model"

print("🔄 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"🚀 Using device: {device}")


##############################################
# HELPERS
##############################################

def download_image(url):
    print(f"⬇️ Downloading {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"⚠️ Failed to download {url}: {e}")
        return None


##############################################
# PREDICT NER FOR ONE IMAGE (WITH CHUNKING)
##############################################

def predict_single_image(url):
    img = download_image(url)
    if img is None:
        return []

    # OCR extraction
    ocr = pytesseract.image_to_data(img, output_type=Output.DICT)

    tokens = []
    positions = []

    for i, word in enumerate(ocr["text"]):
        if not word.strip():
            continue

        tokens.append(word)
        positions.append((
            ocr["left"][i],
            ocr["top"][i],
            ocr["width"][i],
            ocr["height"][i]
        ))

    if not tokens:
        return []

    MAX_LEN = 512
    results = []

    # Process in chunks
    for start in range(0, len(tokens), MAX_LEN):
        end = start + MAX_LEN

        chunk_tokens = tokens[start:end]
        chunk_positions = positions[start:end]

        encoded = tokenizer(
            chunk_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        # MUST EXTRACT WORD IDS BEFORE MOVING TO GPU
        word_ids = encoded.word_ids()

        # Move only tensors to GPU
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()

        for i, wid in enumerate(word_ids):
            if wid is None:
                continue

            label_id = predictions[i]
            label_name = model.config.id2label[label_id]
            word = chunk_tokens[wid]

            results.append({
                "word": word,
                "label": label_name,
                "position": chunk_positions[wid]
            })

    return results




##############################################
# PREDICT FOR JSON INPUT (MULTIPLE PAGES)
##############################################

def predict_from_json(json_path):
    print(f"\n📄 Loading test JSON: {json_path}\n")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both formats:
    # FORMAT A: [{...}, {...}]
    # FORMAT B: {...}
    if isinstance(data, list):
        tasks = data
    elif isinstance(data, dict):
        tasks = [data]
    else:
        raise ValueError("❌ Unsupported JSON format.")

    for task in tasks:
        if "data" not in task or "pages" not in task["data"]:
            print("❌ Invalid JSON: missing 'data' or 'pages'")
            continue

        pages = task["data"]["pages"]

        for i, url in enumerate(pages):
            print("\n" + "═" * 60)
            print(f"📄 PAGE {i}: {url}")
            print("═" * 60)

            preds = predict_single_image(url)

            found = False
            for item in preds:
                if item["label"] != "O":
                    found = True
                    print(f"{item['label']}: {item['word']}  → box={item['position']}")

            if not found:
                print("⚠️ No labeled entities found on this page.")


##############################################
# RUN
##############################################

if __name__ == "__main__":
    predict_from_json("test_images.json")
