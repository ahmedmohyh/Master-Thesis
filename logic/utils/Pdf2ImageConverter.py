from pdf2image import convert_from_path
import os
import json
import shutil

pdf_folder = "C:/Master thesis/files/pdf"
output_root = "C:/Master thesis/files/images"
host_root = "http://host.docker.internal:9900/images"

os.makedirs(output_root, exist_ok=True)

for file in os.listdir(pdf_folder):
    name, ext = os.path.splitext(file)
    if ext.lower() == ".pdf":
        pdf_path = os.path.join(pdf_folder, file)

        # Define per-PDF output and base host paths
        output_base = os.path.join(output_root, name)
        base_host_path = f"{host_root}/{name}"
        os.makedirs(output_base, exist_ok=True)

        # Convert PDF to images
        images = convert_from_path(pdf_path)
        img_list = []
        for i, image in enumerate(images):
            img_name = f"{name}_{i+1}.jpg"
            img_path = os.path.join(output_base, img_name)
            image.save(img_path, "JPEG")
            img_list.append(f"{base_host_path}/{img_name}")

        # Create JSON data
        data_json = {"data": {"pdf_name": name, "pages": img_list}}

        # Save JSON inside the same folder
        json_path = os.path.join(output_base, "data_json.json")
        with open(json_path, "w") as f:
            json.dump(data_json, f, indent=2)

        # Move original PDF into the same folder
        shutil.move(pdf_path, os.path.join(output_base, file))

        print(f"✅ Processed '{file}' → {output_base}")
