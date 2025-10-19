from pdf2image import convert_from_path
import os, json

pdf_folder = "C:/Master thesis/files/pdf"
output_folder = "C:/Master thesis/files/images/slim_7i/"
os.makedirs(output_folder, exist_ok=True)

data_json = []
base_host_path = "http://localhost:9900/images/slim_7i"  # local HTTP server

for file in os.listdir(pdf_folder):
    name, ext = os.path.splitext(file)
    if ext.lower() == ".pdf":
        images = convert_from_path(os.path.join(pdf_folder, file))
        img_list = []
        for i, image in enumerate(images):
            file_name = f'{name}_{i}.jpg'
            image_path = os.path.join(output_folder, file_name)
            image.save(image_path, "JPEG")

            img_list.append(f"{base_host_path}/{file_name}")  # HTTP URL

        data_json.append({"data": {"pages": img_list}})

with open("../../files/images/slim_7i/data_json.json", "w") as f:
    json.dump(data_json, f, indent=2)
