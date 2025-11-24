import json
import os
import requests
from io import BytesIO
from PIL import Image
import pytesseract
from pytesseract import Output

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

##############################################
# CONFIG
##############################################

LABEL_MAP = {
    "prop-name": "PROP_NAME",
    "prop-value": "PROP_VALUE",
    "prop-unit": "PROP_UNIT"
}

NER_LABELS = [
    "O",
    "B-PROP_NAME",
    "B-PROP_VALUE",
    "B-PROP_UNIT"
]

label2id = {label: i for i, label in enumerate(NER_LABELS)}
id2label = {i: label for label, i in label2id.items()}


##############################################
# SAFE IMAGE DOWNLOADER
##############################################

def download_image(url):
    """Download an image safely. Skip if unreachable."""
    print(f"⬇️ Downloading: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img

    except Exception as e:
        print(f"⚠️ WARNING: Could not download {url}. Error: {e}")
        return None


##############################################
# HELPERS
##############################################

def convert_ls_bbox_to_pixels(rect, W, H):
    x0 = rect["x"] * W / 100
    y0 = rect["y"] * H / 100
    x1 = x0 + rect["width"] * W / 100
    y1 = y0 + rect["height"] * H / 100
    return [x0, y0, x1, y1]


def inside(token_box, ann_box):
    tx0, ty0, tx1, ty1 = token_box
    ax0, ay0, ax1, ay1 = ann_box
    return tx0 >= ax0 and ty0 >= ay0 and tx1 <= ax1 and ty1 <= ay1


##############################################
# PARSE LABEL STUDIO JSON → TOKENS + LABELS
##############################################

def parse_labelstudio(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_tokens = []
    all_labels = []
    preview_lines = []  # Save training inspection file

    for task in data:
        pages = task["data"]["pages"]
        annotations = task["annotations"][0]["result"]

        # Sort rectangles by page
        page_to_rects = {}
        for ann in annotations:
            page_idx = ann["item_index"]
            rect = ann["value"]
            label = rect["rectanglelabels"][0]

            page_to_rects.setdefault(page_idx, []).append(rect)

        # Process each page
        for page_idx, img_url in enumerate(pages):

            print(f"\n📄 Processing page {page_idx}: {img_url}")
            img = download_image(img_url)
            if img is None:
                print("➡️ Skipping page due to download failure.\n")
                continue

            W, H = img.size

            # OCR
            ocr = pytesseract.image_to_data(img, output_type=Output.DICT)

            tokens = []
            labels = []

            rects = [
                (rect, LABEL_MAP[rect["rectanglelabels"][0]])
                for rect in page_to_rects.get(page_idx, [])
            ]

            for i, word in enumerate(ocr["text"]):
                if not word.strip():
                    continue

                x = ocr["left"][i]
                y = ocr["top"][i]
                w = ocr["width"][i]
                h = ocr["height"][i]

                token_box = [x, y, x + w, y + h]

                assigned = "O"

                # Check if inside annotation box
                for rect, class_name in rects:
                    ann_box = convert_ls_bbox_to_pixels(rect, W, H)
                    if inside(token_box, ann_box):
                        assigned = f"B-{class_name}"
                        break

                tokens.append(word)
                labels.append(assigned)

                if assigned != "O":
                    preview_lines.append(f"{word}\t{assigned}")

            preview_lines.append("")  # spacing

            if tokens:
                all_tokens.append(tokens)
                all_labels.append(labels)

    # Save preview file
    with open("training_preview.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(preview_lines))

    print("💾 Saved extracted tokens + labels → training_preview.txt")

    return all_tokens, all_labels


##############################################
# TRAINING FUNCTION
##############################################

def train_ner(train_json_path):

    tokens, labels = parse_labelstudio(train_json_path)

    print(f"\n📊 Total training samples: {len(tokens)}")

    # Build dataset
    dataset = Dataset.from_dict({
        "tokens": tokens,
        "ner_tags": labels
    })

    # 80% train / 20% test split
    dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = dataset_split["train"]
    test_ds = dataset_split["test"]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    ##############################################
    # TOKENIZE + ALIGN LABELS
    ##############################################

    def tokenize(batch):
        encoded = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True
        )

        new_labels = []
        for i, seq_labels in enumerate(batch["ner_tags"]):
            word_ids = encoded.word_ids(i)
            aligned = []

            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)
                else:
                    aligned.append(label2id[seq_labels[wid]])

            new_labels.append(aligned)

        encoded["labels"] = new_labels
        return encoded

    encoded_train = train_ds.map(tokenize, batched=True)
    encoded_test = test_ds.map(tokenize, batched=True)

    ##############################################
    # MODEL
    ##############################################

    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(NER_LABELS),
        id2label=id2label,
        label2id=label2id
    )

    ##############################################
    # TRAINING ARGS
    ##############################################

    args = TrainingArguments(
        output_dir="ner_model",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=16,      # great for RTX 4060
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=True,                           # FULL GPU speed
        dataloader_num_workers=8,
        save_total_limit=1,
        logging_steps=20,
        optim="adamw_torch",                  # stable optimizer
    )

    ##############################################
    # TRAINER
    ##############################################

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_train,
        eval_dataset=encoded_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    print("\n🚀 Training started...\n")
    trainer.train()

    # Save final model
    trainer.save_model("final_ner_model")
    tokenizer.save_pretrained("final_ner_model")

    print("\n🎉 Training complete! Model saved → final_ner_model/")


##############################################
# RUN
##############################################

if __name__ == "__main__":
    train_ner("training_data.json")
